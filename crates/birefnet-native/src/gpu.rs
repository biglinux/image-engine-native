//! The model on a Vulkan GPU, in half floats with the matrix products' sums
//! over blocks of 16 in floats.
//!
//! Nearly every layer is one `gemm`: the linear layers, the patch merges and
//! every convolution, whose taps the kernel reads from the map itself. The
//! Swin blocks gather their windows inside the product that reads them and
//! scatter the projected attention back the same way; `attention.comp` runs
//! a window's head in one workgroup. The deformable ASPP computes its offsets
//! and samples over chunks of pixels, so the 49 taps' samples of the 7x7
//! branch exist for one chunk at a time; its pooling branch becomes a bias
//! the products write before the 1x1 convolution adds onto it.
//!
//! The products read weights at an index the same in every lane, which AMD
//! GPUs serve from their scalar cache; others stage them in shared memory
//! (`gemm_shared.comp`). Each Swin block is one submission, each decoder
//! block a few: its ASPP takes one per chunk.

use std::path::Path;

use native_engine::half::halves_to_f32;
use native_vulkan::{Buffer, Commands, Gpu, Kernel};

mod weights;

use crate::Error;
use crate::decoder::{BRANCH, INTER, LEVELS};
use crate::math::HEAD_DIM;
use crate::swin::{SHIFT, STAGES, WINDOW};
use weights::{DecoderBlock, Layers, Linear, Norm};

macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

const AMD: u32 = 0x1002;
/// What `gemm_shared` needs of a workgroup.
const SHARED_BYTES: u32 = 13_824;
/// Bytes of `gemm`'s push constants.
const GEMM_PUSH: u32 = 96;
/// Fewer workgroups than this split a product along k.
const SPLIT_WORKGROUPS: usize = 192;
/// Floats of the split products' partial sums.
const PARTIALS: usize = 1 << 22;
/// Pixels the deformable ASPP takes at a time.
const CHUNK: usize = 8192;

const RELU: usize = 1;
const GELU: usize = 2;
const SIGMOID: usize = 3;

/// Where a product's rows come from and go, and what it does on the way;
/// see `gemm_layout.glsl`.
#[derive(Clone, Copy, Default)]
struct Shape {
    rows: usize,
    src_stride: usize,
    src_off: usize,
    dst_stride: usize,
    dst_off: usize,
    act: usize,
    accumulate: bool,
    in_mode: usize,
    out_mode: usize,
    width: usize,
    padded: usize,
    shift: usize,
    base: usize,
    norm: Option<Norm>,
    kernel: usize,
    height: usize,
    /// Write floats instead of half floats.
    out_float: bool,
}

impl Shape {
    /// `rows` rows of `k` values in, `n` out.
    fn plain(rows: usize, k: usize, n: usize) -> Self {
        Self {
            rows,
            src_stride: k,
            dst_stride: n,
            ..Self::default()
        }
    }

    /// The stride-1 `size x size` convolution of a `side x side` map whose
    /// pixels are `src_stride` apart, into rows `dst_stride` apart.
    fn conv(side: usize, size: usize, src_stride: usize, dst_stride: usize) -> Self {
        Self {
            rows: side * side,
            src_stride,
            dst_stride,
            in_mode: 3,
            width: side,
            height: side,
            kernel: size,
            ..Self::default()
        }
    }
}

enum Products {
    /// `gemm` with 2 and 4 waves, each a block of 32 columns.
    Scalar([Kernel; 2]),
    Shared(Kernel),
}

struct Kernels {
    gemm: Products,
    attention: Kernel,
    norm: Kernel,
    resize: Kernel,
    gather: Kernel,
    mean: Kernel,
    sample: Kernel,
    scale: Kernel,
    full: Kernel,
    reduce: Kernel,
    stats: Kernel,
}

fn groups(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{n} workgroups"))
}

/// Workgroups for `n` invocations of 256, past 65535 along y.
fn spread(n: usize) -> Result<[u32; 3], String> {
    let g = n.div_ceil(256);
    Ok([groups(g.min(65_535))?, groups(g.div_ceil(65_535))?, 1])
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

/// Recording helpers over one submission.
struct Recorder<'a, 'b> {
    c: &'a mut Commands<'b>,
    k: &'a Kernels,
    params: &'a Buffer,
    /// Floats for the shares of a product split along k.
    partials: &'a Buffer,
}

impl Recorder<'_, '_> {
    fn gemm(&mut self, l: &Linear, src: &Buffer, dst: &Buffer, s: Shape) -> Result<(), String> {
        assert!(l.k.is_multiple_of(16), "gemm needs k a multiple of 16");
        assert!(
            s.src_stride.is_multiple_of(4) && s.src_off.is_multiple_of(4),
            "the products read their input four values at a time"
        );
        let (kernel, g) = match &self.k.gemm {
            Products::Scalar(gemm) => {
                // One workgroup's waves share its input: four, or two when
                // four would leave more waves without columns. Eight would
                // share it more, but leave fewer workgroups to hide the
                // loads.
                let blocks = l.n.div_ceil(32);
                let idle = |waves: usize| blocks.next_multiple_of(waves) - blocks;
                let (kernel, columns) = if idle(2) < idle(4) {
                    (&gemm[0], 64)
                } else {
                    (&gemm[1], 128)
                };
                (kernel, [s.rows.div_ceil(64), l.n.div_ceil(columns)])
            }
            Products::Shared(gemm) => (gemm, [s.rows.div_ceil(128), l.n.div_ceil(64)]),
        };
        // Too few workgroups leave most of the GPU idle, as for the 3x3
        // convolutions over thousands of channels on the smallest maps:
        // those split k, each share of it summed into `partials`.
        let workgroups = g[0] * g[1];
        let splits = if workgroups < SPLIT_WORKGROUPS
            && s.norm.is_none()
            && s.out_mode == 0
            && !s.out_float
            && s.act <= RELU
        {
            let want = (SPLIT_WORKGROUPS / workgroups)
                .min(l.k / 512)
                .min(PARTIALS / (s.rows * l.n))
                .max(1);
            // As many shares as k, rounded to 16 per share, fills.
            l.k.div_ceil(l.k.div_ceil(want).next_multiple_of(16))
        } else {
            1
        };
        let split = splits > 1;
        let push = constants(&[
            s.rows,
            l.k,
            l.n,
            s.src_stride,
            s.src_off,
            if split { l.n } else { s.dst_stride },
            if split { 0 } else { s.dst_off },
            l.weights,
            if split {
                optional(None)
            } else {
                optional(l.bias)
            },
            if split { 0 } else { s.act },
            usize::from(s.accumulate && !split),
            s.in_mode,
            s.out_mode,
            s.width,
            s.padded,
            s.shift,
            s.base,
            usize::from(s.norm.is_some()),
            s.norm.map_or(0, |n| n.gamma),
            s.norm.map_or(0, |n| n.beta),
            s.kernel,
            s.height,
            usize::from(s.out_float || split),
            splits,
        ]);
        if s.norm.is_some() {
            // The rows' statistics once, for all the column blocks' workgroups;
            // products with a normalisation never split k, so `partials` is free.
            assert!(2 * s.rows <= PARTIALS, "statistics of {} rows", s.rows);
            let g = [groups(s.rows.div_ceil(64))?, 1, 1];
            self.c
                .dispatch(&self.k.stats, &[src, self.partials, self.params], &push, g)?;
            self.c.barrier();
        }
        let output = if split { self.partials } else { dst };
        let groups3 = [groups(g[0])?, groups(g[1])?, groups(splits)?];
        let buffers = [src, output, self.params, self.partials];
        self.c.dispatch(kernel, &buffers, &push, groups3)?;
        self.c.barrier();
        if split {
            let push = constants(&[
                s.rows,
                l.n,
                splits,
                s.dst_stride,
                s.dst_off,
                optional(l.bias),
                s.act,
                usize::from(s.accumulate),
            ]);
            let g = spread(s.rows * l.n)?;
            self.c
                .dispatch(&self.k.reduce, &[self.partials, dst, self.params], &push, g)?;
            self.c.barrier();
        }
        Ok(())
    }

    /// Layer normalisation of `rows` rows of `c` channels.
    #[expect(
        clippy::too_many_arguments,
        reason = "a dispatch's buffers and their layouts"
    )]
    fn norm(
        &mut self,
        n: Norm,
        rows: usize,
        c: usize,
        src: &Buffer,
        (src_stride, src_off): (usize, usize),
        dst: &Buffer,
        (dst_stride, dst_off): (usize, usize),
    ) -> Result<(), String> {
        let push = constants(&[
            rows, c, src_stride, src_off, dst_stride, dst_off, n.gamma, n.beta,
        ]);
        let g = [groups(rows.min(65_535))?, groups(rows.div_ceil(65_535))?, 1];
        self.c
            .dispatch(&self.k.norm, &[src, dst, self.params], &push, g)?;
        self.c.barrier();
        Ok(())
    }

    /// `resize` of a `from x from x c` map into channels `dst_off..` of the
    /// `to x to` pixels of `dst`, `dst_stride` apart.
    #[expect(
        clippy::too_many_arguments,
        reason = "a dispatch's buffers and their layouts"
    )]
    fn resize(
        &mut self,
        src: &Buffer,
        from: usize,
        c: usize,
        dst: &Buffer,
        to: usize,
        (dst_stride, dst_off): (usize, usize),
        add: bool,
    ) -> Result<(), String> {
        let push = constants(&[from, from, to, to, c, dst_stride, dst_off, usize::from(add)]);
        self.c
            .dispatch(&self.k.resize, &[src, dst], &push, spread(to * to * c)?)?;
        self.c.barrier();
        Ok(())
    }

    /// `gather` mode `mode` of the `side x side` image.
    fn gather(
        &mut self,
        mode: usize,
        image: &Buffer,
        side: usize,
        size: usize,
        dst: &Buffer,
    ) -> Result<(), String> {
        let push = constants(&[mode, side, size]);
        self.c.dispatch(
            &self.k.gather,
            &[image, dst],
            &push,
            spread(side * side * 3)?,
        )?;
        self.c.barrier();
        Ok(())
    }
}

/// Buffers for images of one side.
struct Maps {
    side: usize,
    image: Buffer,
    half_image: Buffer,
    /// The backbone's tokens, and the patch merges' outputs.
    x: Buffer,
    merged: Buffer,
    /// The attention's `qkv` rows and the MLP's hidden ones.
    wide: Buffer,
    heads: Buffer,
    /// A half-size stage output before its upsampling.
    feature: Buffer,
    /// Levels 1 to 3: each full-size stage output, then the half-size one
    /// upsampled to the same grid.
    levels: [Buffer; 3],
    /// The squeeze block's input.
    squeeze: Buffer,
    /// A decoder block's input and output.
    input: Buffer,
    output: Buffer,
    /// A decoder block's 64-channel maps.
    t: Buffer,
    aspp: Buffer,
    means: Buffer,
    pooled: Buffer,
    offsets: Buffer,
    samples: Buffer,
    cat: Buffer,
    gate_hidden: Buffer,
    gates: Buffer,
    stacked: Buffer,
    mid: Buffer,
    low: Buffer,
    logits: Buffer,
    readback: Buffer,
}

impl Maps {
    fn new(gpu: &Gpu, side: usize) -> Result<Self, String> {
        let buffer = |halves: usize| gpu.buffer(halves * 2);
        let g = side / 4;
        let tokens = g * g;
        let padded = g.div_ceil(WINDOW) * WINDOW;
        let window_rows = padded * padded;
        let chunk = CHUNK.min(tokens);
        Ok(Self {
            side,
            image: gpu.host_buffer(side * side * 3 * 2, false)?,
            half_image: buffer(side * side * 3 / 4)?,
            x: buffer(tokens * 96)?,
            merged: buffer(tokens * 48)?,
            wide: buffer((window_rows * 3 * 96).max(tokens * 4 * 96))?,
            heads: buffer(window_rows * 96)?,
            feature: buffer(tokens * 24)?,
            levels: [
                buffer(tokens * 192)?,
                buffer(tokens / 4 * 384)?,
                buffer(tokens / 16 * 768)?,
            ],
            squeeze: buffer(tokens / 64 * 2880)?,
            input: buffer(tokens * LEVELS[3].0)?,
            output: buffer(tokens * LEVELS[3].1)?,
            t: buffer(tokens * INTER)?,
            aspp: buffer(tokens * INTER)?,
            means: buffer(INTER)?,
            pooled: buffer(BRANCH)?,
            offsets: gpu.buffer(chunk * 3 * 49 * 4)?,
            samples: buffer(chunk * 49 * INTER)?,
            cat: buffer(chunk * 4 * BRANCH)?,
            gate_hidden: buffer(tokens / 4 * 16)?,
            gates: buffer(tokens / 4)?,
            stacked: buffer(side * side * 3)?,
            mid: buffer(tokens * INTER)?,
            low: buffer(tokens)?,
            logits: buffer(side * side)?,
            readback: gpu.host_buffer(side * side * 2, true)?,
        })
    }
}

/// Runs what a function records as one submission.
type Submit<'a> =
    dyn FnMut(&mut dyn FnMut(&mut Recorder) -> Result<(), String>) -> Result<(), Error> + 'a;

pub struct Network {
    gpu: Gpu,
    kernels: Kernels,
    params: Buffer,
    partials: Buffer,
    layers: Layers,
    maps: Option<Maps>,
}

impl Network {
    /// Open a GPU and load the weights in `path` onto it. The GPU comes
    /// first: without one, the weights are read only for the CPU.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let mut gpu = Gpu::open().map_err(Error::Gpu)?;
        let (layers, params) = Layers::load(path)?;
        let kernel = |spirv, bindings, push| gpu.kernel(spirv, bindings, push).map_err(Error::Gpu);
        let gemm = if gpu.vendor() != AMD && gpu.max_shared_memory() >= SHARED_BYTES {
            Products::Shared(kernel(spirv!("gemm_shared"), 4, GEMM_PUSH)?)
        } else {
            Products::Scalar([
                kernel(spirv!("gemm_2"), 4, GEMM_PUSH)?,
                kernel(spirv!("gemm_4"), 4, GEMM_PUSH)?,
            ])
        };
        let kernels = Kernels {
            gemm,
            attention: kernel(spirv!("attention"), 3, 20)?,
            norm: kernel(spirv!("norm"), 3, 32)?,
            resize: kernel(spirv!("resize"), 2, 32)?,
            gather: kernel(spirv!("gather"), 2, 12)?,
            mean: kernel(spirv!("mean"), 2, 4)?,
            sample: kernel(spirv!("sample"), 3, 20)?,
            scale: kernel(spirv!("scale"), 2, 8)?,
            full: kernel(spirv!("full"), 3, 12)?,
            reduce: kernel(spirv!("reduce"), 3, 32)?,
            stats: kernel(spirv!("stats"), 3, GEMM_PUSH)?,
        };
        let partials = gpu.buffer(PARTIALS * 4).map_err(Error::Gpu)?;
        let buffer = gpu.upload(&params.0).map_err(Error::Gpu)?;
        Ok(Self {
            gpu,
            kernels,
            params: buffer,
            partials,
            layers,
            maps: None,
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        self.gpu.name()
    }

    /// As `Model::run_normalized`, on the GPU, for the image `fill` writes
    /// as little-endian half floats.
    pub fn run(&mut self, side: usize, fill: &dyn Fn(&mut [[u8; 2]])) -> Result<Vec<f32>, Error> {
        if self.maps.as_ref().is_none_or(|m| m.side != side) {
            // Free the smaller maps before making the larger ones.
            self.maps = None;
            self.maps = Some(Maps::new(&self.gpu, side).map_err(Error::Gpu)?);
        }
        let m = self.maps.as_mut().expect("made above");
        fill(&mut m.image.bytes_mut().as_chunks_mut::<2>().0[..side * side * 3]);
        let Self {
            gpu,
            kernels,
            params,
            partials,
            layers,
            maps,
        } = self;
        let m = maps.as_ref().expect("made above");
        // Each submission is queued before the previous is waited for, so
        // the GPU does not idle between them.
        let mut steps = gpu.steps();
        let mut submit = |f: &mut dyn FnMut(&mut Recorder) -> Result<(), String>| {
            steps
                .submit(|c| {
                    f(&mut Recorder {
                        c,
                        k: kernels,
                        params,
                        partials,
                    })
                })
                .map(|_| ())
                .map_err(Error::Gpu)
        };

        // Both backbone passes, the half-size image's first.
        let half = side / 2;
        submit(&mut |r| r.resize(&m.image, side, 3, &m.half_image, half, (3, 0), false))?;
        layers.backbone(&mut submit, m, &m.half_image, half, true)?;
        layers.backbone(&mut submit, m, &m.image, side, false)?;
        layers.decoder(&mut submit, m, side)?;

        let pixels = side * side;
        submit(&mut |r| {
            r.c.copy(&m.logits, &m.readback, pixels * 2)?;
            Ok(())
        })?;
        steps.finish().map_err(Error::Gpu)?;
        let mut logits = vec![0.0; pixels];
        halves_to_f32(
            m.readback.bytes()[..pixels * 2].as_chunks::<2>().0,
            &mut logits,
        );
        Ok(logits)
    }
}

impl Layers {
    /// The backbone on the `side x side` image, its stage outputs into the
    /// levels: a full-size run's into each level's first half, a half-size
    /// run's upsampled into the second.
    fn backbone(
        &self,
        submit: &mut Submit,
        m: &Maps,
        image: &Buffer,
        side: usize,
        half: bool,
    ) -> Result<(), Error> {
        let mut grid = side / 4;
        submit(&mut |r| {
            let tokens = grid * grid;
            r.gather(0, image, side, 0, &m.wide)?;
            r.gemm(&self.embed, &m.wide, &m.x, Shape::plain(tokens, 48, 96))?;
            r.norm(self.embed_norm, tokens, 96, &m.x, (96, 0), &m.x, (96, 0))
        })?;
        let (mut x, mut merged) = (&m.x, &m.merged);
        // Where level 4 starts in the squeeze block's input.
        let mut squeeze_off = 0;
        for (s, stage) in self.stages.iter().enumerate() {
            let c = stage.channels;
            let tokens = grid * grid;
            let padded = grid.div_ceil(WINDOW) * WINDOW;
            let windows = padded / WINDOW;
            let rows = padded * padded;
            for (b, block) in stage.blocks.iter().enumerate() {
                let shift = if b % 2 == 1 { SHIFT } else { 0 };
                let window = Shape {
                    width: grid,
                    padded,
                    shift,
                    ..Shape::plain(rows, c, 3 * c)
                };
                submit(&mut |r| {
                    let gather = Shape {
                        in_mode: 2,
                        norm: Some(block.norm1),
                        ..window
                    };
                    r.gemm(&block.qkv, x, &m.wide, gather)?;
                    let push =
                        constants(&[c, c / HEAD_DIM, windows, usize::from(shift > 0), block.bias]);
                    let g = [groups(windows * windows)?, groups(c / HEAD_DIM)?, 1];
                    r.c.dispatch(&r.k.attention, &[&m.wide, &m.heads, r.params], &push, g)?;
                    r.c.barrier();
                    let scatter = Shape {
                        src_stride: c,
                        dst_stride: c,
                        out_mode: 2,
                        accumulate: true,
                        ..window
                    };
                    r.gemm(&block.proj, &m.heads, x, scatter)?;
                    let fc1 = Shape {
                        norm: Some(block.norm2),
                        act: GELU,
                        ..Shape::plain(tokens, c, 4 * c)
                    };
                    r.gemm(&block.fc1, x, &m.wide, fc1)?;
                    let fc2 = Shape {
                        accumulate: true,
                        ..Shape::plain(tokens, 4 * c, c)
                    };
                    r.gemm(&block.fc2, &m.wide, x, fc2)
                })?;
            }
            submit(&mut |r| {
                // The stage output, into its level or the squeeze block's input.
                let (level, stride, off) = match s {
                    3 => (&m.squeeze, 2880, squeeze_off),
                    _ => (&m.levels[s], 2 * c, 0),
                };
                if half {
                    r.norm(stage.norm, tokens, c, x, (c, 0), &m.feature, (c, 0))?;
                    r.resize(
                        &m.feature,
                        grid,
                        c,
                        level,
                        2 * grid,
                        (stride, off + c),
                        false,
                    )?;
                } else {
                    r.norm(stage.norm, tokens, c, x, (c, 0), level, (stride, off))?;
                }
                if let Some((norm, reduction)) = &stage.merge {
                    let merge = Shape {
                        rows: tokens / 4,
                        src_stride: c,
                        dst_stride: 2 * c,
                        in_mode: 1,
                        width: grid / 2,
                        norm: Some(*norm),
                        ..Shape::default()
                    };
                    r.gemm(reduction, x, merged, merge)?;
                }
                Ok(())
            })?;
            squeeze_off += 2 * c;
            if stage.merge.is_some() {
                std::mem::swap(&mut x, &mut merged);
                grid /= 2;
            }
        }
        Ok(())
    }

    fn decoder(&self, submit: &mut Submit, m: &Maps, side: usize) -> Result<(), Error> {
        // The squeeze block's input: levels 1 to 3 downsampled to level 4's
        // grid, then level 4, which the backbone wrote.
        let g4 = side >> 5;
        submit(&mut |r| {
            let mut off = 0;
            for (s, level) in m.levels.iter().enumerate() {
                let c = 2 * STAGES[s].0;
                r.resize(
                    level,
                    side >> (s + 2),
                    c,
                    &m.squeeze,
                    g4,
                    (2880, off),
                    false,
                )?;
                off += c;
            }
            Ok(())
        })?;
        let first = LEVELS[0].0;
        self.block(
            submit,
            m,
            &self.squeeze,
            (&m.squeeze, 2880),
            g4,
            (&m.input, first),
        )?;

        // Each level's block: its input is the squeezed map or the previous
        // block's output upsampled onto a lateral convolution of the level's
        // features, then the patch branch.
        let mut previous_grid = g4;
        for (i, block) in self.blocks.iter().enumerate() {
            let level = 3 - i;
            let (cin, cout) = LEVELS[i];
            let grid = side >> (level + 2);
            let main_c = if i == 0 { 1536 } else { LEVELS[i - 1].1 };
            submit(&mut |r| {
                if i > 0 {
                    let c = 2 * STAGES[level].0;
                    let lateral = Shape {
                        dst_stride: cin,
                        ..Shape::plain(grid * grid, c, main_c)
                    };
                    r.gemm(&self.laterals[i - 1], &m.levels[level], &m.input, lateral)?;
                    r.resize(
                        &m.output,
                        previous_grid,
                        main_c,
                        &m.input,
                        grid,
                        (cin, 0),
                        true,
                    )?;
                }
                let patches = &self.patches[i];
                let channels = 3 * (side / grid).pow(2);
                r.gather(1, &m.image, side, grid, &m.stacked)?;
                r.gemm(
                    &patches.conv1,
                    &m.stacked,
                    &m.mid,
                    Shape::conv(grid, 3, channels, INTER),
                )?;
                let out = Shape {
                    dst_off: main_c,
                    ..Shape::conv(grid, 3, INTER, cin)
                };
                r.gemm(&patches.conv_out, &m.mid, &m.input, out)
            })?;
            self.block(submit, m, block, (&m.input, cin), grid, (&m.output, cout))?;
            previous_grid = grid;
        }

        // The head on the last block's output, upsampled, plus the
        // full-resolution patch branch.
        let g = side / 4;
        submit(&mut |r| {
            r.gemm(&self.head, &m.output, &m.low, Shape::plain(g * g, 96, 1))?;
            r.resize(&m.low, g, 1, &m.logits, side, (1, 0), false)?;
            let buffers = [&m.image, &m.logits, r.params];
            let push = constants(&[side, self.full, 0]);
            r.c.dispatch(&r.k.full, &buffers, &push, spread(side * side)?)?;
            let push = constants(&[side, self.full, 1]);
            r.c.dispatch(&r.k.full, &buffers, &push, spread((4 * side - 4) * 64)?)?;
            r.c.barrier();
            Ok(())
        })
    }

    /// Decoder block `b` on a `grid x grid` map, its pixels `stride` apart in
    /// `input`, into the first channels of `out`'s pixels, `out_stride`
    /// apart.
    fn block(
        &self,
        submit: &mut Submit,
        m: &Maps,
        b: &DecoderBlock,
        (input, stride): (&Buffer, usize),
        grid: usize,
        (out, out_stride): (&Buffer, usize),
    ) -> Result<(), Error> {
        let pixels = grid * grid;
        submit(&mut |r| {
            let conv_in = Shape {
                act: RELU,
                ..Shape::conv(grid, 3, stride, INTER)
            };
            r.gemm(&b.conv_in, input, &m.t, conv_in)?;
            // The pooling branch, as a bias of every pixel of the ASPP's output.
            let push = constants(&[pixels]);
            r.c.dispatch(&r.k.mean, &[&m.t, &m.means], &push, [1, 1, 1])?;
            r.c.barrier();
            let pool = Shape {
                act: RELU,
                ..Shape::plain(1, INTER, BRANCH)
            };
            r.gemm(&b.aspp.pool, &m.means, &m.pooled, pool)?;
            let broadcast = Shape {
                src_stride: 0,
                ..Shape::plain(pixels, BRANCH, INTER)
            };
            r.gemm(&b.aspp.conv1_pool, &m.pooled, &m.aspp, broadcast)
        })?;
        for first in (0..pixels).step_by(CHUNK) {
            let rows = CHUNK.min(pixels - first);
            submit(&mut |r| {
                for (i, branch) in b.aspp.branches.iter().enumerate() {
                    let taps = branch.size * branch.size;
                    let offsets = Shape {
                        rows,
                        base: first,
                        out_float: true,
                        ..Shape::conv(grid, branch.size, INTER, 3 * taps)
                    };
                    r.gemm(&branch.offsets, &m.t, &m.offsets, offsets)?;
                    let push = constants(&[grid, grid, first, rows, branch.size]);
                    let g = spread(rows * taps * INTER / 8)?;
                    r.c.dispatch(&r.k.sample, &[&m.t, &m.offsets, &m.samples], &push, g)?;
                    r.c.barrier();
                    let weight = Shape {
                        act: RELU,
                        dst_stride: 4 * BRANCH,
                        dst_off: i * BRANCH,
                        ..Shape::plain(rows, taps * INTER, BRANCH)
                    };
                    r.gemm(&branch.weight, &m.samples, &m.cat, weight)?;
                }
                let conv1 = Shape {
                    act: RELU,
                    accumulate: true,
                    dst_off: first * INTER,
                    ..Shape::plain(rows, 4 * BRANCH, INTER)
                };
                r.gemm(&b.aspp.conv1, &m.cat, &m.aspp, conv1)
            })?;
        }
        submit(&mut |r| {
            r.gemm(
                &b.conv_out,
                &m.aspp,
                out,
                Shape::conv(grid, 3, INTER, out_stride),
            )?;
            if let Some((conv, logit)) = &b.gate {
                let cout = b.conv_out.n;
                let hidden = Shape {
                    act: RELU,
                    ..Shape::conv(grid, 3, cout, 16)
                };
                r.gemm(conv, out, &m.gate_hidden, hidden)?;
                let gate = Shape {
                    act: SIGMOID,
                    ..Shape::plain(pixels, 16, 1)
                };
                r.gemm(logit, &m.gate_hidden, &m.gates, gate)?;
                let push = constants(&[pixels, cout]);
                let g = spread(pixels * cout / 2)?;
                r.c.dispatch(&r.k.scale, &[&m.gates, out], &push, g)?;
                r.c.barrier();
            }
            Ok(())
        })
    }
}
