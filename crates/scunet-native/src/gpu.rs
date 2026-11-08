//! The network of `network.rs` on a Vulkan GPU, in half floats, with the
//! normalisations, softmax and GELU computed in floats.
//!
//! Maps stay on the GPU, NHWC: `x` holds a level's map, `y` its blocks' split
//! channels (the convolution half first), `t` the convolution half's middle
//! map, and one buffer per level down its skip. Each block is one
//! submission; the attention and the MLP run on chunks of rows through
//! smaller buffers. Every 1x1 convolution and linear layer is one `gemm`
//! kernel, which also normalises its input rows when a layer norm comes
//! first, and whose index modes gather the 2x2 stride-2 convolution's input,
//! scatter the transposed one's output, and take attention tokens from and
//! put them back into their rolled windows.

use std::cell::Cell;
use std::ops::ControlFlow;
use std::path::Path;

use native_engine::half::{f32_to_half, floats_to_halves, halves_to_f32};
use native_engine::weight_file::WeightFile;
use native_vulkan::{Buffer, Commands, Gpu, Kernel, Words};

use crate::Error;
use crate::math::{HEAD_DIM, WINDOW_TOKENS};
use crate::network::{BLOCKS, DIM, padded};
use crate::swin::{WINDOW, bias_tables};

const ADD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const ATTENTION: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const CONV3: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv3.spv"));
const CONV_SHARED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_shared.spv"));
const CONV_SHARED_32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_shared_32.spv"));
const GEMM_SHARED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_shared.spv"));
const GEMM_SHARED_32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_shared_32.spv"));
/// What the `_shared` kernels need of a workgroup.
const SHARED_BYTES: u32 = 32_256;
const SHARED_INVOCATIONS: u32 = 256;
const AMD: u32 = 0x1002;
const MLP: [&[u8]; 2] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/mlp_32.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/mlp_64.spv")),
];
const GEMM: [&[u8]; 3] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_2.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_4.spv")),
];

/// Rows of the attention and MLP chunks: 1024 windows.
const CHUNK: usize = 1 << 16;

/// Parameters in half floats, each array starting at an even offset.
#[derive(Default)]
struct Params(Words);

impl Params {
    fn put(&mut self, values: &[f32]) -> usize {
        let n = values.len();
        let halves = values.iter().map(|&v| f32_to_half(v)).collect();
        self.0.push(halves, n, n.next_multiple_of(2))
    }

    /// A `[k][n]` matrix with its rows padded to a multiple of 64.
    fn matrix(&mut self, values: Vec<u16>, n: usize) -> usize {
        self.0.push(values, n, n.next_multiple_of(64))
    }
}

struct Linear {
    k: usize,
    n: usize,
    weights: usize,
    bias: Option<usize>,
}

impl Linear {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        k: usize,
        n: usize,
        bias: bool,
    ) -> Result<Self, Error> {
        let w = file
            .take_weights(&format!("{name}.weight"), &[k, n])?
            .into_f16();
        let weights = params.matrix(w, n);
        let bias = if bias {
            Some(params.put(&file.take(&format!("{name}.bias"), &[n])?))
        } else {
            None
        };
        Ok(Self {
            k,
            n,
            weights,
            bias,
        })
    }
}

struct Conv {
    cin: usize,
    cout: usize,
    weights: usize,
}

impl Conv {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        cin: usize,
        cout: usize,
    ) -> Result<Self, Error> {
        let w = file
            .take_weights(&format!("{name}.weight"), &[9 * cin, cout])?
            .into_f16();
        Ok(Self {
            cin,
            cout,
            weights: params.matrix(w, cout),
        })
    }
}

struct Norm {
    gamma: usize,
    beta: usize,
}

impl Norm {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        c: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            gamma: params.put(&file.take(&format!("{name}.weight"), &[c])?),
            beta: params.put(&file.take(&format!("{name}.bias"), &[c])?),
        })
    }
}

struct Block {
    half: usize,
    shifted: bool,
    split: Linear,
    conv1: Conv,
    conv2: Conv,
    norm1: Norm,
    qkv: Linear,
    bias: usize,
    proj: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,
    join: Linear,
}

impl Block {
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        c: usize,
        shifted: bool,
    ) -> Result<Self, Error> {
        let half = c / 2;
        let t = format!("{name}.trans_block");
        let heads = half / HEAD_DIM;
        let n = WINDOW_TOKENS;
        let position = file.take(&format!("{t}.msa.position_bias"), &[heads, n, n])?;
        Ok(Self {
            half,
            shifted,
            split: Linear::load(file, params, &format!("{name}.conv1_1"), c, c, true)?,
            conv1: Conv::load(file, params, &format!("{name}.conv_block.0"), half, half)?,
            conv2: Conv::load(file, params, &format!("{name}.conv_block.2"), half, half)?,
            norm1: Norm::load(file, params, &format!("{t}.ln1"), half)?,
            qkv: Linear::load(
                file,
                params,
                &format!("{t}.msa.embedding_layer"),
                half,
                3 * half,
                true,
            )?,
            bias: params.put(&bias_tables(position, shifted)),
            proj: Linear::load(file, params, &format!("{t}.msa.linear"), half, half, true)?,
            norm2: Norm::load(file, params, &format!("{t}.ln2"), half)?,
            fc1: Linear::load(file, params, &format!("{t}.mlp.0"), half, 4 * half, true)?,
            fc2: Linear::load(file, params, &format!("{t}.mlp.2"), 4 * half, half, true)?,
            join: Linear::load(file, params, &format!("{name}.conv1_2"), c, c, true)?,
        })
    }
}

fn level(
    file: &mut WeightFile,
    params: &mut Params,
    name: &str,
    first: usize,
    c: usize,
) -> Result<Vec<Block>, Error> {
    (0..BLOCKS)
        .map(|b| {
            Block::load(
                file,
                params,
                &format!("{name}.{}", first + b),
                c,
                b % 2 == 1,
            )
        })
        .collect()
}

/// Where a `gemm` reads its rows and writes its results; see `gemm.comp`.
#[derive(Clone, Copy, Default)]
struct Rows {
    src_stride: usize,
    src_off: usize,
    dst_stride: usize,
    dst_off: usize,
    gelu: bool,
    accumulate: bool,
    in_mode: usize,
    out_mode: usize,
    width: usize,
    height: usize,
    shift: usize,
    base: usize,
    /// Layer normalisation of the input rows, as `(gamma, beta)` offsets.
    norm: Option<(usize, usize)>,
}

fn constants(values: &[usize]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| (v as i32).to_le_bytes())
        .collect()
}

/// `-1` for no offset, once `constants` makes it an `i32`.
fn optional(v: Option<usize>) -> usize {
    v.unwrap_or(u32::MAX as usize)
}

fn groups(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{n} workgroups"))
}

/// The convolution and matrix product kernels: AMD GPUs read uniform
/// weights through their scalar cache, others stage them in shared memory,
/// each the faster on its kind.
enum Products {
    Scalar {
        /// 32 output channels per workgroup.
        conv3: Kernel,
        /// `gemm` with 1, 2 and 4 waves, each a block of 64 columns.
        gemm: [Kernel; 3],
    },
    /// `conv_shared` and `gemm_shared` over blocks of 64 output channels,
    /// and of 32 for layers no wider.
    Shared {
        conv: Kernel,
        conv_32: Kernel,
        gemm: Kernel,
        gemm_32: Kernel,
    },
}

struct Kernels {
    add: Kernel,
    attention: Kernel,
    products: Products,
    /// `mlp.comp` for 32 and 64 channels, on AMD GPUs.
    mlp: Option<[Kernel; 2]>,
}

impl Kernels {
    /// The fused MLP for rows of `channels`, where there is one.
    fn mlp(&self, channels: usize) -> Option<&Kernel> {
        let [narrow, wide] = self.mlp.as_ref()?;
        match channels {
            32 => Some(narrow),
            64 => Some(wide),
            _ => None,
        }
    }
}

/// Recording helpers over one submission.
struct Recorder<'a, 'b> {
    c: &'a mut Commands<'b>,
    k: &'a Kernels,
    params: &'a Buffer,
}

impl Recorder<'_, '_> {
    fn gemm(
        &mut self,
        l: &Linear,
        rows: usize,
        src: &Buffer,
        dst: &Buffer,
        r: Rows,
    ) -> Result<(), String> {
        let push = constants(&[
            l.k,
            l.n,
            r.src_stride,
            r.src_off,
            r.dst_stride,
            r.dst_off,
            l.weights,
            optional(l.bias),
            usize::from(r.gelu),
            usize::from(r.accumulate),
            r.in_mode,
            r.out_mode,
            r.width,
            r.height,
            r.shift,
            r.base,
            usize::from(r.norm.is_some()),
            r.norm.map_or(0, |n| n.0),
            r.norm.map_or(0, |n| n.1),
            rows,
        ]);
        assert!(l.k.is_multiple_of(16), "gemm needs k a multiple of 16");
        assert!(
            r.src_stride.is_multiple_of(4) && r.src_off.is_multiple_of(4),
            "the products read their input four values at a time"
        );
        let blocks = l.n.div_ceil(64);
        let (kernel, g) = match &self.k.products {
            Products::Scalar { gemm, .. } => {
                // One workgroup's waves share its input; up to four, as many
                // as there are blocks of columns.
                let (kernel, waves) = match blocks {
                    1 => (&gemm[0], 1),
                    2 => (&gemm[1], 2),
                    _ => (&gemm[2], 4),
                };
                (
                    kernel,
                    [groups(rows / 64)?, groups(blocks.div_ceil(waves))?, 1],
                )
            }
            Products::Shared { gemm_32, .. } if l.n <= 32 => {
                (gemm_32, [groups(rows.div_ceil(256))?, 1, 1])
            }
            Products::Shared { gemm, .. } => {
                (gemm, [groups(rows.div_ceil(128))?, groups(blocks)?, 1])
            }
        };
        self.c
            .dispatch(kernel, &[src, dst, self.params], &push, g)?;
        self.c.barrier();
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a dispatch's buffers and their layouts"
    )]
    fn conv(
        &mut self,
        conv: &Conv,
        (h, w): (usize, usize),
        src: &Buffer,
        src_stride: usize,
        dst: &Buffer,
        (dst_stride, dst_off): (usize, usize),
        relu: bool,
        accumulate: bool,
    ) -> Result<(), String> {
        let push = constants(&[
            w,
            h,
            conv.cin,
            src_stride,
            conv.cout,
            dst_stride,
            dst_off,
            conv.weights,
            usize::from(relu),
            usize::from(accumulate),
        ]);
        let (kernel, g) = match &self.k.products {
            Products::Scalar { conv3, .. } => (
                conv3,
                [
                    groups(w.div_ceil(8))?,
                    groups(h.div_ceil(8))?,
                    groups(conv.cout.div_ceil(32))?,
                ],
            ),
            Products::Shared { conv_32, .. } if conv.cout <= 32 => (
                conv_32,
                [groups(w.div_ceil(8))?, groups(h.div_ceil(32))?, 1],
            ),
            Products::Shared { conv: shared, .. } => (
                shared,
                [
                    groups(w.div_ceil(4))?,
                    groups(h.div_ceil(32))?,
                    groups(conv.cout.div_ceil(64))?,
                ],
            ),
        };
        self.c
            .dispatch(kernel, &[src, dst, self.params], &push, g)?;
        self.c.barrier();
        Ok(())
    }
}

/// Buffers for maps of up to `pixels` padded pixels.
struct Maps {
    pixels: usize,
    image: Buffer,
    x: Buffer,
    y: Buffer,
    t: Buffer,
    skips: Vec<Buffer>,
    qkv: Buffer,
    heads: Buffer,
    hidden: Buffer,
    out: Buffer,
    readback: Buffer,
}

impl Maps {
    fn new(gpu: &Gpu, pixels: usize) -> Result<Self, String> {
        let half = |n: usize| n * 2;
        // The chunk buffers' largest use over the levels: pixels shrink by 4
        // and channels double from one level to the next.
        let (mut chunk, mut p, mut c) = (0, pixels, DIM / 2);
        for _ in 0..4 {
            chunk = chunk.max(p.min(CHUNK) * c);
            (p, c) = (p / 4, 2 * c);
        }
        Ok(Self {
            pixels,
            image: gpu.host_buffer(half(pixels * 3), false)?,
            x: gpu.buffer(half(pixels * DIM))?,
            y: gpu.buffer(half(pixels * DIM))?,
            t: gpu.buffer(half(pixels * DIM / 2))?,
            skips: (1..=3)
                .map(|i| gpu.buffer(half(pixels * DIM / (1 << i))))
                .collect::<Result<_, _>>()?,
            qkv: gpu.buffer(half(3 * chunk))?,
            heads: gpu.buffer(half(chunk))?,
            hidden: gpu.buffer(half(4 * chunk))?,
            out: gpu.buffer(half(pixels * 3))?,
            readback: gpu.host_buffer(half(pixels * 3), true)?,
        })
    }
}

pub struct Network {
    gpu: Gpu,
    kernels: Kernels,
    params: Buffer,
    head: Conv,
    down: Vec<(Vec<Block>, Linear)>,
    body: Vec<Block>,
    up: Vec<(Linear, Vec<Block>)>,
    tail: Conv,
    maps: Option<Maps>,
}

impl Network {
    /// Open a GPU and load the weights in `path` onto it.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let mut file = WeightFile::read(path, b"SCUNET\0\0")?;
        let mut params = Params::default();
        let p = &mut params;
        let head = Conv::load(&mut file, p, "m_head.0", 3, DIM)?;
        let mut down = Vec::new();
        for (i, c) in [(1, DIM), (2, 2 * DIM), (3, 4 * DIM)] {
            let name = format!("m_down{i}");
            let blocks = level(&mut file, p, &name, 0, c)?;
            down.push((
                blocks,
                Linear::load(
                    &mut file,
                    p,
                    &format!("{name}.{BLOCKS}"),
                    4 * c,
                    2 * c,
                    false,
                )?,
            ));
        }
        let body = level(&mut file, p, "m_body", 0, 8 * DIM)?;
        let mut up = Vec::new();
        for (i, c) in [(3, 4 * DIM), (2, 2 * DIM), (1, DIM)] {
            let name = format!("m_up{i}");
            let double = Linear::load(&mut file, p, &format!("{name}.0"), 2 * c, 4 * c, false)?;
            up.push((double, level(&mut file, p, &name, 1, c)?));
        }
        let tail = Conv::load(&mut file, p, "m_tail.0", DIM, 3)?;
        file.finish()?;

        let mut gpu = Gpu::open().map_err(Error::Gpu)?;
        let kernel = |spirv, bindings, push| gpu.kernel(spirv, bindings, push).map_err(Error::Gpu);
        let shared = gpu.vendor() != AMD
            && gpu.max_shared_memory() >= SHARED_BYTES
            && gpu.max_invocations() >= SHARED_INVOCATIONS;
        let kernels = Kernels {
            add: kernel(ADD, 2, 4)?,
            attention: kernel(ATTENTION, 3, 24)?,
            mlp: if shared {
                None
            } else {
                Some([kernel(MLP[0], 2, 32)?, kernel(MLP[1], 2, 32)?])
            },
            products: if shared {
                Products::Shared {
                    conv: kernel(CONV_SHARED, 3, 40)?,
                    conv_32: kernel(CONV_SHARED_32, 3, 40)?,
                    gemm: kernel(GEMM_SHARED, 3, 80)?,
                    gemm_32: kernel(GEMM_SHARED_32, 3, 80)?,
                }
            } else {
                Products::Scalar {
                    conv3: kernel(CONV3, 3, 40)?,
                    gemm: [
                        kernel(GEMM[0], 3, 80)?,
                        kernel(GEMM[1], 3, 80)?,
                        kernel(GEMM[2], 3, 80)?,
                    ],
                }
            },
        };
        let params = gpu.upload(&params.0).map_err(Error::Gpu)?;
        Ok(Self {
            gpu,
            kernels,
            params,
            head,
            down,
            body,
            up,
            tail,
            maps: None,
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        self.gpu.name()
    }

    /// As `network::Network::run`, on the GPU.
    pub fn run(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<Vec<f32>, Error> {
        let (mut h, mut w) = padded(width, height);
        let pixels = h * w;
        if self.maps.as_ref().is_none_or(|m| m.pixels < pixels) {
            // Free the smaller maps before making the larger ones.
            self.maps = None;
            self.maps = Some(Maps::new(&self.gpu, pixels).map_err(Error::Gpu)?);
        }
        let Self {
            gpu,
            kernels,
            params,
            head,
            down,
            body,
            up,
            tail,
            maps,
        } = self;
        let m = maps.as_mut().expect("made above");
        // As network::pad, straight into the upload buffer in half floats.
        let halves = &mut m.image.bytes_mut().as_chunks_mut::<2>().0[..pixels * 3];
        for (y, row) in halves.chunks_exact_mut(w * 3).enumerate() {
            let (inside, edge) = row.split_at_mut(width * 3);
            floats_to_halves(&image[y.min(height - 1) * width * 3..][..width * 3], inside);
            let last: [[u8; 2]; 3] = inside[(width - 1) * 3..].try_into().expect("one pixel");
            for pixel in edge.as_chunks_mut::<3>().0 {
                *pixel = last;
            }
        }
        let m = &*m;
        // Each submission is queued before the previous is waited for, so
        // the GPU does not idle between them. A block's progress is reported
        // once the submission that ends it has finished.
        let mut steps = gpu.steps();
        let ends_block = Cell::new(false);
        let mut report = |finished: bool| {
            if finished && ends_block.replace(false) && step().is_break() {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        };
        let mut submit = |f: &mut dyn FnMut(&mut Recorder) -> Result<(), String>| {
            let finished = steps
                .submit(|c| {
                    let mut r = Recorder {
                        c,
                        k: kernels,
                        params,
                    };
                    f(&mut r)
                })
                .map_err(Error::Gpu)?;
            report(finished)
        };
        let mut block_done = || {
            ends_block.set(true);
            Ok(())
        };
        // The head's output is also the skip into the tail; it is computed
        // again at the end, as on the CPU.
        submit(&mut |r| r.conv(head, (h, w), &m.image, 3, &m.x, (DIM, 0), false, false))?;
        let mut c = DIM;
        for (i, (blocks, halve)) in down.iter().enumerate() {
            run_level(&mut submit, &mut block_done, m, blocks, (h, w))?;
            let skip = &m.skips[i];
            submit(&mut |r| {
                let rows = Rows {
                    src_stride: c,
                    dst_stride: 2 * c,
                    in_mode: 1,
                    width: w / 2,
                    ..Rows::default()
                };
                r.gemm(halve, h * w / 4, &m.x, skip, rows)?;
                r.c.copy(skip, &m.x, h * w / 4 * 2 * c * 2)
            })?;
            (h, w, c) = (h / 2, w / 2, 2 * c);
        }
        run_level(&mut submit, &mut block_done, m, body, (h, w))?;
        for (i, (double, blocks)) in up.iter().enumerate() {
            let skip = &m.skips[2 - i];
            submit(&mut |r| {
                let n = h * w * c / 2;
                let push = constants(&[n]);
                r.c.dispatch(
                    &r.k.add,
                    &[skip, &m.x],
                    &push,
                    [groups(n.div_ceil(256))?, 1, 1],
                )?;
                r.c.barrier();
                let rows = Rows {
                    src_stride: c,
                    dst_stride: c / 2,
                    out_mode: 1,
                    width: w,
                    ..Rows::default()
                };
                r.gemm(double, h * w, &m.x, &m.y, rows)?;
                r.c.copy(&m.y, &m.x, h * w * 4 * (c / 2) * 2)
            })?;
            (h, w, c) = (2 * h, 2 * w, c / 2);
            run_level(&mut submit, &mut block_done, m, blocks, (h, w))?;
        }
        submit(&mut |r| {
            r.conv(head, (h, w), &m.image, 3, &m.x, (DIM, 0), false, true)?;
            r.conv(tail, (h, w), &m.x, DIM, &m.out, (3, 0), false, false)?;
            r.c.copy(&m.out, &m.readback, pixels * 3 * 2)
        })?;
        report(steps.finish().map_err(Error::Gpu)?)?;
        let out = m.readback.bytes()[..pixels * 3 * 2].as_chunks::<2>().0;
        let mut cropped = vec![0.0; width * height * 3];
        for (row, dst) in out
            .chunks_exact(w * 3)
            .zip(cropped.chunks_exact_mut(width * 3))
        {
            halves_to_f32(&row[..width * 3], dst);
        }
        Ok(cropped)
    }
}

type Submit<'s> =
    dyn FnMut(&mut dyn FnMut(&mut Recorder) -> Result<(), String>) -> Result<(), Error> + 's;

/// The blocks of one level on the `h x w` map in `m.x`, as `network::run_level`.
fn run_level(
    submit: &mut Submit,
    block_done: &mut dyn FnMut() -> Result<(), Error>,
    m: &Maps,
    blocks: &[Block],
    (h, w): (usize, usize),
) -> Result<(), Error> {
    let half = blocks[0].half;
    let c = 2 * half;
    let pixels = h * w;
    let plain = Rows {
        src_stride: c,
        dst_stride: c,
        ..Rows::default()
    };
    submit(&mut |r| r.gemm(&blocks[0].split, pixels, &m.x, &m.y, plain))?;
    for (b, block) in blocks.iter().enumerate() {
        let next = blocks.get(b + 1).map(|n| &n.split);
        let shift = if block.shifted { WINDOW / 2 } else { 0 };
        submit(&mut |r| {
            r.conv(&block.conv1, (h, w), &m.y, c, &m.t, (half, 0), true, false)?;
            r.conv(&block.conv2, (h, w), &m.t, half, &m.y, (c, 0), false, true)?;
            for base in (0..pixels).step_by(CHUNK) {
                let rows = CHUNK.min(pixels - base);
                let gather = Rows {
                    src_stride: c,
                    src_off: half,
                    dst_stride: 3 * half,
                    in_mode: 2,
                    width: w,
                    height: h,
                    shift,
                    base,
                    norm: Some((block.norm1.gamma, block.norm1.beta)),
                    ..Rows::default()
                };
                r.gemm(&block.qkv, rows, &m.y, &m.qkv, gather)?;
                let push = constants(&[half, block.bias, usize::from(block.shifted), w, h, base]);
                let g = [groups(rows / WINDOW_TOKENS)?, groups(half / HEAD_DIM)?, 1];
                r.c.dispatch(&r.k.attention, &[&m.qkv, &m.heads, r.params], &push, g)?;
                r.c.barrier();
                let scatter = Rows {
                    src_stride: half,
                    dst_stride: c,
                    dst_off: half,
                    accumulate: true,
                    out_mode: 2,
                    width: w,
                    height: h,
                    shift,
                    base,
                    ..Rows::default()
                };
                r.gemm(&block.proj, rows, &m.heads, &m.y, scatter)?;
            }
            if let Some(mlp) = r.k.mlp(half) {
                let push = constants(&[
                    c,
                    half,
                    block.fc1.weights,
                    optional(block.fc1.bias),
                    block.fc2.weights,
                    optional(block.fc2.bias),
                    block.norm2.gamma,
                    block.norm2.beta,
                ]);
                r.c.dispatch(mlp, &[&m.y, r.params], &push, [groups(pixels / 64)?, 1, 1])?;
                r.c.barrier();
            } else {
                for base in (0..pixels).step_by(CHUNK) {
                    let rows = CHUNK.min(pixels - base);
                    let hidden = Rows {
                        src_stride: c,
                        src_off: base * c + half,
                        dst_stride: 4 * half,
                        gelu: true,
                        norm: Some((block.norm2.gamma, block.norm2.beta)),
                        ..Rows::default()
                    };
                    r.gemm(&block.fc1, rows, &m.y, &m.hidden, hidden)?;
                    let back = Rows {
                        src_stride: 4 * half,
                        dst_stride: c,
                        dst_off: base * c + half,
                        accumulate: true,
                        ..Rows::default()
                    };
                    r.gemm(&block.fc2, rows, &m.hidden, &m.y, back)?;
                }
            }
            r.gemm(
                &block.join,
                pixels,
                &m.y,
                &m.x,
                Rows {
                    accumulate: true,
                    ..plain
                },
            )?;
            if let Some(split) = next {
                r.gemm(split, pixels, &m.x, &m.y, plain)?;
            }
            Ok(())
        })?;
        block_done()?;
    }
    Ok(())
}
