//! The network of `network.rs` on a Vulkan GPU, in half floats, with the
//! Fourier transforms and the convolutions' sums over blocks of channels in
//! floats.
//!
//! Each of the run's 26 steps is one submission. The convolutions pad by
//! reflection inside their kernel; the transposed convolutions run as their
//! four phases, each a matrix product that gathers the input pixels its taps
//! read. The Fourier units transform rows and columns with twiddles the host
//! computes in double precision: by a fast transform in shared memory
//! (`fft.comp`) where the line's length allows, by direct sums (`dft.comp`)
//! otherwise.
//!
//! The kernels read weights at an index the same in every lane, which AMD
//! GPUs serve from their scalar cache; on others the 3x3 convolutions and
//! the matrix products stage their weights in shared memory
//! (`conv_shared.comp`, `gemm_shared.comp`).

use std::ops::ControlFlow;
use std::path::Path;

use native_engine::half::{f32_to_half, floats_to_halves, halves_to_f32};
use native_engine::weight_file::WeightFile;
use native_vulkan::{Buffer, Commands, Gpu, Kernel, Words};

use crate::Error;
use crate::network::{BLOCKS, CHANNELS, GLOBAL, LOCAL, SPECTRAL, masked_rows, padded};

const ADD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const CONV3: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv3.spv"));
const CONV3_STRIDE2: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv3_s2.spv"));
const CONV7: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv7.spv"));
const CONV7_LAST: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv7_last.spv"));
const CONV7_PX4: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv7_px4.spv"));
const CONV7_PX2: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv7_px2.spv"));
const CONV_SHARED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_shared.spv"));
const CONV_SHARED_S2: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_shared_s2.spv"));
/// What the `_shared` kernels need of a workgroup.
const SHARED_BYTES: u32 = 39_552;
const AMD: u32 = 0x1002;
const INTEL: u32 = 0x8086;
const DFT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dft.spv"));
const FFT_64: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fft_64.spv"));
const FFT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fft.spv"));
const FFT_256: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fft_256.spv"));
const FFT_512: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fft_512.spv"));
/// What the `fft` kernels need of a workgroup: lines of 64 points for 16
/// channels, 128 for 8, 256 for 4 or 512 for 2.
const FFT_BYTES: u32 = 20_480;
const GEMM_SHARED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_shared.spv"));
const GEMM: [&[u8]; 3] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_2.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_4.spv")),
];

/// Parameters in half floats, each array starting at a multiple of 8, so
/// that a kernel can read 16 bytes at a time.
#[derive(Default)]
struct Params(Words);

impl Params {
    fn put(&mut self, values: &[f32]) -> usize {
        let n = values.len();
        let halves = values.iter().map(|&v| f32_to_half(v)).collect();
        self.0.push(halves, n, n.next_multiple_of(8))
    }
}

/// A product's weights `[k][n]` and bias, as offsets in half floats.
struct Linear {
    k: usize,
    n: usize,
    weights: usize,
    bias: Option<usize>,
}

impl Linear {
    /// The weights as a `[k][n]` matrix with its rows padded to a multiple
    /// of 64.
    fn load(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        k: usize,
        n: usize,
        bias: bool,
    ) -> Result<Self, Error> {
        Self::load_rows(file, params, name, (k, n), n.next_multiple_of(64), bias)
    }

    /// The weights as a `[k][n]` matrix with rows of `stride` halves.
    fn load_rows(
        file: &mut WeightFile,
        params: &mut Params,
        name: &str,
        (k, n): (usize, usize),
        stride: usize,
        bias: bool,
    ) -> Result<Self, Error> {
        let w = file
            .take_weights(&format!("{name}.weight"), &[k, n])?
            .into_f16();
        let weights = params.0.push(w, n, stride);
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

struct Ffc {
    local: Linear,
    cross: Linear,
    spectral_in: Linear,
    fourier: Linear,
    spectral_out: Linear,
}

impl Ffc {
    fn load(file: &mut WeightFile, params: &mut Params, name: &str) -> Result<Self, Error> {
        let mut linear = |part: &str, k, n, bias| {
            Linear::load(file, params, &format!("{name}.{part}"), k, n, bias)
        };
        Ok(Self {
            local: linear("local", 9 * CHANNELS, LOCAL, true)?,
            cross: linear("cross", 9 * LOCAL, GLOBAL, true)?,
            spectral_in: linear("spectral_in", GLOBAL, SPECTRAL, true)?,
            fourier: linear("fourier", 2 * SPECTRAL, 2 * SPECTRAL, true)?,
            spectral_out: linear("spectral_out", SPECTRAL, GLOBAL, false)?,
        })
    }
}

/// The four phases of a transposed convolution and its output channels.
struct Up {
    phases: Vec<Linear>,
    cout: usize,
}

/// Whether `fft.comp` beats direct sums on a line of `n` points: when its
/// only prime factors are 2, 3, 5 and 7. A larger one takes a stage of direct
/// sums in shared memory, slower than `dft.comp` on a line of 83 points.
fn fast(n: usize) -> bool {
    let mut left = n;
    for p in [2, 3, 5, 7] {
        while left.is_multiple_of(p) {
            left /= p;
        }
    }
    left == 1
}

fn groups(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{n} workgroups"))
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

struct Kernels {
    add: Kernel,
    /// The 3x3 convolutions of stride 1 and 2 on AMD: 16 pixels by 128
    /// output channels per workgroup, lanes over channels
    /// (`conv_lanes.comp`).
    conv3: Kernel,
    conv3_stride2: Kernel,
    /// On the GPUs that get `conv_shared`, which read each weight from
    /// memory, 32 output channels and four pixels per lane, so that each
    /// weight read serves four products; two on Intel's, whose compiler
    /// spills the four pixels' sums out of registers.
    conv7: Kernel,
    /// Output columns a workgroup of `conv7` covers on those GPUs.
    conv7_columns: usize,
    /// The last layer, 4 output channels or fewer (`conv7_last.comp`).
    conv7_last: Kernel,
    /// The 3x3 convolutions of stride 1 and 2 with their weights in shared
    /// memory, for GPUs other than AMD's, whose scalar cache the others rely
    /// on.
    conv_shared: Option<(Kernel, Kernel)>,
    dft: Kernel,
    /// `fft.comp` for lines of up to 64, 128, 256 and 512 points, where the
    /// GPU has the shared memory: 16, 8, 4 and 2 channels per workgroup, so
    /// that a radix-4 stage gives each of the 256 invocations a butterfly.
    fft: Option<[Kernel; 4]>,
    /// `gemm` with 1, 2 and 4 waves, each a block of 64 columns.
    gemm: [Kernel; 3],
    /// The matrix product with its weights in shared memory, on the GPUs
    /// that get `conv_shared`.
    gemm_shared: Option<Kernel>,
}

/// Where a convolution reads and writes: the input's size, channels and
/// pixel stride, the output's pixel stride and first channel.
#[derive(Clone, Copy)]
struct ConvLayout {
    size: (usize, usize),
    cin: usize,
    src_stride: usize,
    dst_stride: usize,
    dst_off: usize,
}

/// What a product's activation and output do.
#[derive(Clone, Copy, Default)]
struct Out {
    relu: bool,
    accumulate: bool,
}

/// Recording helpers over one submission.
struct Recorder<'a, 'b> {
    c: &'a mut Commands<'b>,
    k: &'a Kernels,
    params: &'a Buffer,
}

impl Recorder<'_, '_> {
    /// A `size x size` stride-`stride` convolution with activation `act`
    /// (0 none, 1 ReLU, 2 sigmoid).
    fn conv(
        &mut self,
        (size, stride): (usize, usize),
        l: &Linear,
        m: ConvLayout,
        src: &Buffer,
        dst: &Buffer,
        act: usize,
    ) -> Result<(), String> {
        assert!(
            m.cin.is_multiple_of(4) && m.src_stride.is_multiple_of(4),
            "the convolutions read their input four channels at a time"
        );
        let (h, w) = m.size;
        let (oh, ow) = (h / stride, w / stride);
        // Output channels, rows and columns of pixels per workgroup.
        let (kernel, block, rows, columns) = match (size, stride, &self.k.conv_shared) {
            (3, 1, Some((shared, _))) => (shared, 64, 32, 4),
            (3, 2, Some((_, shared))) => (shared, 64, 16, 4),
            (3, 1, None) => {
                assert!(
                    m.cin.is_multiple_of(8),
                    "conv_lanes reads 8 channels at a time"
                );
                (&self.k.conv3, 128, 1, 16)
            }
            (3, 2, _) => {
                assert!(
                    m.cin.is_multiple_of(8),
                    "conv_lanes reads 8 channels at a time"
                );
                (&self.k.conv3_stride2, 128, 1, 16)
            }
            _ if l.n <= 4 => {
                assert!(
                    m.cin.is_multiple_of(8),
                    "conv7_last reads 8 channels at a time"
                );
                (&self.k.conv7_last, 4, 8, 16)
            }
            (_, _, Some(_)) => (&self.k.conv7, 32, 8, self.k.conv7_columns),
            _ => (&self.k.conv7, 64, 8, 8),
        };
        let push = constants(&[
            w,
            h,
            ow,
            oh,
            m.cin,
            m.src_stride,
            l.n,
            m.dst_stride,
            m.dst_off,
            l.weights,
            optional(l.bias),
            act,
            0,
        ]);
        let g = [
            groups(ow.div_ceil(columns))?,
            groups(oh.div_ceil(rows))?,
            groups(l.n.div_ceil(block))?,
        ];
        self.c
            .dispatch(kernel, &[src, dst, self.params], &push, g)?;
        self.c.barrier();
        Ok(())
    }

    /// The product of `rows` rows; `layout` holds the strides and offsets,
    /// `width` the map's width, and `phase` a transposed convolution's
    /// phase with the map's height.
    #[expect(
        clippy::too_many_arguments,
        reason = "a dispatch's buffers and their layouts"
    )]
    fn gemm(
        &mut self,
        l: &Linear,
        rows: usize,
        src: &Buffer,
        dst: &Buffer,
        (src_stride, src_off, dst_stride, dst_off): (usize, usize, usize, usize),
        width: usize,
        phase: Option<(usize, usize, usize)>,
        out: Out,
    ) -> Result<(), String> {
        let (mode, (height, py, px)) = match phase {
            Some(p) => (1, p),
            None => (0, (0, 0, 0)),
        };
        let push = constants(&[
            rows,
            l.k,
            l.n,
            src_stride,
            src_off,
            dst_stride,
            dst_off,
            l.weights,
            optional(l.bias),
            usize::from(out.relu),
            usize::from(out.accumulate),
            mode,
            width,
            height,
            py,
            px,
        ]);
        assert!(l.k.is_multiple_of(16), "gemm needs k a multiple of 16");
        assert!(
            src_stride.is_multiple_of(4) && src_off.is_multiple_of(4),
            "the products read their input four values at a time"
        );
        let blocks = l.n.div_ceil(64);
        let (kernel, g) = match &self.k.gemm_shared {
            Some(shared) => (shared, [groups(rows.div_ceil(256))?, groups(blocks)?, 1]),
            None => {
                // One workgroup's waves share its input; up to four, as many
                // as there are blocks of columns.
                let (kernel, waves) = match blocks {
                    1 => (&self.k.gemm[0], 1),
                    2 => (&self.k.gemm[1], 2),
                    _ => (&self.k.gemm[2], 4),
                };
                let g = [
                    groups(rows.div_ceil(64))?,
                    groups(blocks.div_ceil(waves))?,
                    1,
                ];
                (kernel, g)
            }
        };
        self.c
            .dispatch(kernel, &[src, dst, self.params], &push, g)?;
        self.c.barrier();
        Ok(())
    }

    /// Pass `pass` of `dft.comp` on a `h x w` map of `c` channels.
    fn dft(
        &mut self,
        pass: usize,
        (h, w): (usize, usize),
        c: usize,
        buffers: [&Buffer; 6],
    ) -> Result<(), String> {
        // Lines summed along, their length, and outputs along each.
        let half = w / 2 + 1;
        let (lines, n, outputs) = match pass {
            0 => (h, w, half),
            1 | 2 => (half, h, h),
            _ => (h, w, w),
        };
        let push = constants(&[pass, h, w, c]);
        // A fast transform takes a whole line in a workgroup's shared
        // memory; longer lines take direct sums, 16 outputs per workgroup.
        let (kernel, g) = match &self.k.fft {
            Some([fft, ..]) if n <= 64 && fast(n) => {
                (fft, [groups(c.div_ceil(16))?, groups(lines)?, 1])
            }
            Some([_, fft, ..]) if n <= 128 && fast(n) => {
                (fft, [groups(c.div_ceil(8))?, groups(lines)?, 1])
            }
            Some([.., fft, _]) if n <= 256 && fast(n) => {
                (fft, [groups(c.div_ceil(4))?, groups(lines)?, 1])
            }
            Some([.., fft]) if n <= 512 && fast(n) => {
                (fft, [groups(c.div_ceil(2))?, groups(lines)?, 1])
            }
            _ => (
                &self.k.dft,
                [
                    groups(c.div_ceil(64))?,
                    groups(lines)?,
                    groups(outputs.div_ceil(16))?,
                ],
            ),
        };
        self.c.dispatch(kernel, &buffers, &push, g)?;
        self.c.barrier();
        Ok(())
    }
}

/// Buffers for images of up to `pixels` padded pixels whose eighth-scale
/// spectrum has up to `spectrum_pixels` pixels and whose eighth-scale sides
/// total up to `eighth_sides`: a narrower image of as many pixels needs more
/// of the last two.
struct Maps {
    pixels: usize,
    spectrum_pixels: usize,
    eighth_sides: usize,
    image: Buffer,
    a: Buffer,
    b: Buffer,
    mid: Buffer,
    out: Buffer,
    spectral: Buffer,
    spectrum: Buffer,
    filtered: Buffer,
    wide: Buffer,
    back: Buffer,
    rgb: Buffer,
    readback: Buffer,
    /// `dft.comp`'s twiddles for the eighth scale's width, then its height.
    twiddles: Buffer,
}

impl Maps {
    fn new(gpu: &Gpu, pixels: usize, spectrum: usize, eighth_sides: usize) -> Result<Self, String> {
        let half_bytes = |n: usize| 2 * n;
        let eighth = pixels / 64;
        Ok(Self {
            pixels,
            spectrum_pixels: spectrum,
            eighth_sides,
            image: gpu.host_buffer(half_bytes(pixels * 4), false)?,
            a: gpu.buffer(half_bytes(pixels * 64))?,
            b: gpu.buffer(half_bytes(pixels * 64))?,
            mid: gpu.buffer(half_bytes(eighth * CHANNELS))?,
            out: gpu.buffer(half_bytes(eighth * CHANNELS))?,
            spectral: gpu.buffer(half_bytes(eighth * SPECTRAL))?,
            spectrum: gpu.buffer(half_bytes(spectrum * 2 * SPECTRAL))?,
            filtered: gpu.buffer(half_bytes(spectrum * 2 * SPECTRAL))?,
            wide: gpu.buffer(4 * spectrum * 2 * SPECTRAL)?,
            back: gpu.buffer(half_bytes(eighth * SPECTRAL))?,
            rgb: gpu.buffer(half_bytes(pixels * 3))?,
            readback: gpu.host_buffer(half_bytes(pixels * 3), true)?,
            twiddles: gpu.host_buffer(8 * eighth_sides, false)?,
        })
    }
}

pub struct Network {
    gpu: Gpu,
    kernels: Kernels,
    params: Buffer,
    first: Linear,
    down: Vec<Linear>,
    blocks: Vec<(Ffc, Ffc)>,
    up: Vec<Up>,
    last: Linear,
    maps: Option<Maps>,
}

impl Network {
    /// Open a GPU and load the weights in `path` onto it. The GPU comes
    /// first: without one, the weights are read only for the CPU.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let mut gpu = Gpu::open().map_err(Error::Gpu)?;
        let mut file = WeightFile::read(path, b"LAMA\0\0\0\0")?;
        let mut params = Params::default();
        let p = &mut params;
        let first = Linear::load(&mut file, p, "first", 49 * 4, 64, true)?;
        let down = [(64, 128), (128, 256), (256, CHANNELS)]
            .iter()
            .enumerate()
            .map(|(i, &(cin, cout))| {
                Linear::load(&mut file, p, &format!("down{i}"), 9 * cin, cout, true)
            })
            .collect::<Result<_, _>>()?;
        let blocks = (0..BLOCKS)
            .map(|b| {
                Ok((
                    Ffc::load(&mut file, p, &format!("block{b}.ffc0"))?,
                    Ffc::load(&mut file, p, &format!("block{b}.ffc1"))?,
                ))
            })
            .collect::<Result<_, Error>>()?;
        let mut up = Vec::new();
        for (i, (cin, cout)) in [(CHANNELS, 256), (256, 128), (128, 64)]
            .into_iter()
            .enumerate()
        {
            let phases = (0..4)
                .map(|ph| {
                    // Phase (py, px) reads (py + 1) (px + 1) input pixels.
                    let taps = (ph / 2 + 1) * (ph % 2 + 1);
                    Linear::load(
                        &mut file,
                        p,
                        &format!("up{i}.phase{ph}"),
                        taps * cin,
                        cout,
                        true,
                    )
                })
                .collect::<Result<_, _>>()?;
            up.push(Up { phases, cout });
        }
        // conv7_last.comp reads rows of 4 output channels.
        let last = Linear::load_rows(&mut file, p, "last", (49 * 64, 3), 4, true)?;
        file.finish()?;

        let kernel = |spirv, bindings, push| gpu.kernel(spirv, bindings, push).map_err(Error::Gpu);
        let shared = gpu.vendor() != AMD && gpu.max_shared_memory() >= SHARED_BYTES;
        let (conv7, conv7_columns) = match (shared, gpu.vendor()) {
            (false, _) => (CONV7, 8),
            (true, INTEL) => (CONV7_PX2, 16),
            (true, _) => (CONV7_PX4, 32),
        };
        let kernels = Kernels {
            add: kernel(ADD, 2, 4)?,
            conv3: kernel(CONV3, 3, 52)?,
            conv3_stride2: kernel(CONV3_STRIDE2, 3, 52)?,
            conv7: kernel(conv7, 3, 52)?,
            conv7_columns,
            conv7_last: kernel(CONV7_LAST, 3, 52)?,
            conv_shared: if shared {
                Some((kernel(CONV_SHARED, 3, 52)?, kernel(CONV_SHARED_S2, 3, 52)?))
            } else {
                None
            },
            dft: kernel(DFT, 6, 16)?,
            fft: if gpu.max_shared_memory() >= FFT_BYTES {
                Some([
                    kernel(FFT_64, 6, 16)?,
                    kernel(FFT, 6, 16)?,
                    kernel(FFT_256, 6, 16)?,
                    kernel(FFT_512, 6, 16)?,
                ])
            } else {
                None
            },
            gemm: [
                kernel(GEMM[0], 3, 64)?,
                kernel(GEMM[1], 3, 64)?,
                kernel(GEMM[2], 3, 64)?,
            ],
            gemm_shared: if shared {
                Some(kernel(GEMM_SHARED, 3, 64)?)
            } else {
                None
            },
        };
        let params = gpu.upload(&params.0).map_err(Error::Gpu)?;
        Ok(Self {
            gpu,
            kernels,
            params,
            first,
            down,
            blocks,
            up,
            last,
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
        mask: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<Vec<f32>, Error> {
        let (mut h, mut w) = padded(width, height);
        let pixels = h * w;
        let (h8, w8) = (h / 8, w / 8);
        let spectrum_pixels = h8 * (w8 / 2 + 1);
        if self.maps.as_ref().is_none_or(|m| {
            m.pixels < pixels || m.spectrum_pixels < spectrum_pixels || m.eighth_sides < h8 + w8
        }) {
            // Free the smaller maps before making the larger ones.
            self.maps = None;
            let maps = Maps::new(&self.gpu, pixels, spectrum_pixels, h8 + w8);
            self.maps = Some(maps.map_err(Error::Gpu)?);
        }
        let Self {
            gpu,
            kernels,
            params,
            first,
            down,
            blocks,
            up,
            last,
            maps,
        } = self;
        let m = maps.as_mut().expect("made above");
        let halves = m.image.bytes_mut().as_chunks_mut::<2>().0;
        masked_rows(image, mask, width, height, |y, row| {
            floats_to_halves(row, &mut halves[y * row.len()..][..row.len()]);
        });
        let angles = (0..w8).map(|i| (i, w8)).chain((0..h8).map(|i| (i, h8)));
        let twiddles = m.twiddles.bytes_mut().as_chunks_mut::<8>().0;
        for (d, (i, n)) in twiddles.iter_mut().zip(angles) {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            d[..4].copy_from_slice(&(a.cos() as f32).to_le_bytes());
            d[4..].copy_from_slice(&(a.sin() as f32).to_le_bytes());
        }
        let m = &*m;
        // Each step is queued before the previous is waited for, so the
        // progress report after each costs the GPU no idle time.
        let mut steps = gpu.steps();
        let mut report = |finished: bool| {
            if finished && step().is_break() {
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
        let plain = |cin, size| ConvLayout {
            size,
            cin,
            src_stride: cin,
            dst_stride: 0,
            dst_off: 0,
        };
        submit(&mut |r| {
            let maps = ConvLayout {
                dst_stride: 64,
                ..plain(4, (h, w))
            };
            r.conv((7, 1), first, maps, &m.image, &m.a, 1)
        })?;
        // The maps take turns: `a` holds the current one.
        let (mut a, mut b) = (&m.a, &m.b);
        let mut c = 64;
        for l in down.iter() {
            submit(&mut |r| {
                let maps = ConvLayout {
                    dst_stride: l.n,
                    ..plain(c, (h, w))
                };
                r.conv((3, 2), l, maps, a, b, 1)
            })?;
            (a, b) = (b, a);
            (h, w, c) = (h / 2, w / 2, l.n);
        }
        let size = (h, w);
        for (ffc0, ffc1) in blocks.iter() {
            submit(&mut |r| {
                ffc(r, ffc0, m, a, &m.mid, size)?;
                ffc(r, ffc1, m, &m.mid, &m.out, size)?;
                let n = h * w * CHANNELS / 2;
                let push = constants(&[n]);
                r.c.dispatch(
                    &r.k.add,
                    &[&m.out, a],
                    &push,
                    [groups(n.div_ceil(256))?, 1, 1],
                )?;
                r.c.barrier();
                Ok(())
            })?;
        }
        for u in up.iter() {
            submit(&mut |r| {
                for (ph, l) in u.phases.iter().enumerate() {
                    let layout = (c, 0, u.cout, 0);
                    let relu = Out {
                        relu: true,
                        accumulate: false,
                    };
                    r.gemm(l, h * w, a, b, layout, w, Some((h, ph / 2, ph % 2)), relu)?;
                }
                Ok(())
            })?;
            (a, b) = (b, a);
            (h, w, c) = (2 * h, 2 * w, u.cout);
        }
        submit(&mut |r| {
            let maps = ConvLayout {
                dst_stride: 3,
                ..plain(64, (h, w))
            };
            r.conv((7, 1), last, maps, a, &m.rgb, 2)?;
            r.c.copy(&m.rgb, &m.readback, pixels * 3 * 2)
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

/// `y = ffc(x)` on two maps of `CHANNELS` channels the size `size`, as
/// `network::Ffc::run`.
fn ffc(
    r: &mut Recorder,
    f: &Ffc,
    m: &Maps,
    x: &Buffer,
    y: &Buffer,
    (h, w): (usize, usize),
) -> Result<(), String> {
    let pixels = h * w;
    let spectrum = h * (w / 2 + 1);
    let maps = ConvLayout {
        size: (h, w),
        cin: CHANNELS,
        src_stride: CHANNELS,
        dst_stride: CHANNELS,
        dst_off: 0,
    };
    r.conv((3, 1), &f.local, maps, x, y, 1)?;
    let cross = ConvLayout {
        cin: LOCAL,
        dst_off: LOCAL,
        ..maps
    };
    r.conv((3, 1), &f.cross, cross, x, y, 0)?;
    let relu = Out {
        relu: true,
        accumulate: false,
    };
    r.gemm(
        &f.spectral_in,
        pixels,
        x,
        &m.spectral,
        (CHANNELS, LOCAL, SPECTRAL, 0),
        w,
        None,
        relu,
    )?;
    let s = (h, w);
    r.dft(
        0,
        s,
        SPECTRAL,
        [
            &m.spectral,
            &m.wide,
            &m.spectrum,
            &m.wide,
            &m.back,
            &m.twiddles,
        ],
    )?;
    r.dft(
        1,
        s,
        SPECTRAL,
        [
            &m.spectral,
            &m.wide,
            &m.spectrum,
            &m.back,
            &m.back,
            &m.twiddles,
        ],
    )?;
    let spectrum_layout = (2 * SPECTRAL, 0, 2 * SPECTRAL, 0);
    r.gemm(
        &f.fourier,
        spectrum,
        &m.spectrum,
        &m.filtered,
        spectrum_layout,
        w / 2 + 1,
        None,
        relu,
    )?;
    r.dft(
        2,
        s,
        SPECTRAL,
        [
            &m.filtered,
            &m.wide,
            &m.spectrum,
            &m.wide,
            &m.back,
            &m.twiddles,
        ],
    )?;
    r.dft(
        3,
        s,
        SPECTRAL,
        [
            &m.filtered,
            &m.wide,
            &m.back,
            &m.spectrum,
            &m.spectral,
            &m.twiddles,
        ],
    )?;
    let onto = Out {
        relu: true,
        accumulate: true,
    };
    r.gemm(
        &f.spectral_out,
        pixels,
        &m.back,
        y,
        (SPECTRAL, 0, CHANNELS, LOCAL),
        w,
        None,
        onto,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use native_engine::half::half_to_f32;

    #[test]
    #[ignore = "needs a Vulkan GPU"]
    fn fft_matches_direct_sums() {
        let mut gpu = Gpu::open().unwrap();
        let dft = gpu.kernel(DFT, 6, 16).unwrap();
        let fft_64 = gpu.kernel(FFT_64, 6, 16).unwrap();
        let fft = gpu.kernel(FFT, 6, 16).unwrap();
        let fft_256 = gpu.kernel(FFT_256, 6, 16).unwrap();
        let fft_512 = gpu.kernel(FFT_512, 6, 16).unwrap();
        // Columns of 56 points take radices 4, 2 and 7, rows of 120 points 4,
        // 2, 3 and 5; 20 channels leave a partial block.
        let (h, w, c) = (56, 120, 20);
        let half = w / 2 + 1;
        let mut seed = 1u32;
        let mut values = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect()
        };
        let halves = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| f32_to_half(x).to_le_bytes())
                .collect()
        };
        let floats = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let map = halves(&values(h * w * c));
        let wide = floats(&values(h * half * 2 * c));
        let spectrum = halves(&values(half * h * 2 * c));
        let residual = halves(&values(h * w * c));
        let twiddles: Vec<u8> = (0..w)
            .map(|i| (i, w))
            .chain((0..h).map(|i| (i, h)))
            .flat_map(|(i, n)| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [
                    (a.cos() as f32).to_le_bytes(),
                    (a.sin() as f32).to_le_bytes(),
                ]
                .concat()
            })
            .collect();
        let upload = |bytes: &[u8]| {
            let mut b = gpu.host_buffer(bytes.len(), true).unwrap();
            b.bytes_mut().copy_from_slice(bytes);
            b
        };
        let ins = [&map, &wide, &spectrum, &wide, &residual, &twiddles].map(|b| upload(b));
        for pass in 0..4 {
            let (lines, outputs) = match pass {
                0 => (h, half),
                1 | 2 => (half, h),
                _ => (h, w),
            };
            let half_in = if pass == 2 { &ins[2] } else { &ins[0] };
            let mut outs = Vec::new();
            // The 64-point kernel takes the columns, the rows being longer.
            let short = (pass == 1 || pass == 2).then_some((&fft_64, [c.div_ceil(16), lines, 1]));
            for (kernel, g) in [
                (&dft, [c.div_ceil(64), lines, outputs.div_ceil(16)]),
                (&fft, [c.div_ceil(8), lines, 1]),
                (&fft_256, [c.div_ceil(4), lines, 1]),
                (&fft_512, [c.div_ceil(2), lines, 1]),
            ]
            .into_iter()
            .chain(short)
            {
                let half_out = gpu.host_buffer(h * w * c * 4, true).unwrap();
                let wide_out = gpu.host_buffer(h * half * 2 * c * 4, true).unwrap();
                let push = constants(&[pass, h, w, c]);
                let g = g.map(|n| u32::try_from(n).unwrap());
                gpu.run(|cmd| {
                    cmd.dispatch(
                        kernel,
                        &[half_in, &ins[1], &half_out, &wide_out, &ins[4], &ins[5]],
                        &push,
                        g,
                    )
                })
                .unwrap();
                outs.push(if pass % 2 == 0 {
                    wide_out.bytes()[..h * half * 2 * c * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect::<Vec<_>>()
                } else {
                    let n = if pass == 1 {
                        half * h * 2 * c
                    } else {
                        h * w * c
                    };
                    half_out.bytes()[..n * 2]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| half_to_f32(u16::from_le_bytes(b)))
                        .collect()
                });
            }
            assert!(
                outs.iter().flatten().all(|v| v.is_finite()),
                "pass {pass}: values not finite"
            );
            let scale = outs[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // Floats for passes 0 and 2, half floats for 1 and 3.
            let tolerance = if pass % 2 == 0 { 1e-5 } else { 2e-3 };
            for (variant, out) in outs[1..].iter().enumerate() {
                let worst = outs[0]
                    .iter()
                    .zip(out)
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    worst <= tolerance * scale,
                    "pass {pass}, fft {variant}: {worst} apart, values up to {scale}"
                );
            }
        }
    }
}
