//! Big LaMa's generator (FFCResNetGenerator): a 7x7 convolution from the
//! masked image and its mask to 64 channels, three stride-2 convolutions to
//! 512 channels at an eighth of the size, 18 residual blocks of fast Fourier
//! convolutions, three stride-2 transposed convolutions back to 64 channels,
//! and a 7x7 convolution to RGB through a sigmoid. Every batch norm is folded
//! into the convolution before it, and every convolution pads by reflection
//! except the transposed ones.
//!
//! At the eighth scale a pixel's 512 channels are a local part (128) and a
//! global part (384). A fast Fourier convolution computes, each followed by
//! a ReLU:
//!
//! - local: one 3x3 convolution of all 512 channels, which is the reference's
//!   local-to-local plus global-to-local convolutions;
//! - global: a 3x3 convolution of the local part, plus the spectral transform
//!   of the global part: a 1x1 convolution to 192 channels, its 2-D Fourier
//!   transform through a 1x1 convolution and back (`fourier.rs`), added to its
//!   input and brought to 384 channels by a last 1x1 convolution.
//!
//! The image is padded to a multiple of 8 by mirroring its last rows and
//! columns, as the reference does, and the output cropped back.

use std::ops::ControlFlow;

use crate::Error;
use crate::fourier::Fourier;
use crate::gemm::{Act, Input, Matrix, Pad, Taps, multiply};
use crate::pool::{Pool, Shared};
use crate::weight_file::WeightFile;
use native_engine::winograd;

pub const LOCAL: usize = 128;
pub const GLOBAL: usize = 384;
pub const CHANNELS: usize = LOCAL + GLOBAL;
pub const SPECTRAL: usize = GLOBAL / 2;
pub const BLOCKS: usize = 18;
/// Sides are padded to a multiple of this: three halvings.
const MULTIPLE: usize = 8;
/// The shortest side the network takes: the 3x3 convolutions at the eighth
/// scale pad by reflection, which needs two pixels there, 16 after padding.
pub const MIN_SIDE: usize = 9;
/// Steps of a run: the first convolution, three down, the blocks, three up
/// and the last convolution.
pub const STEPS: usize = 1 + 3 + BLOCKS + 3 + 1;

fn conv(size: usize, step: usize) -> Taps {
    Taps::Conv {
        size,
        step,
        pad: Pad::Reflect,
    }
}

const POINT: Taps = Taps::Conv {
    size: 1,
    step: 1,
    pad: Pad::Zero,
};

fn input(
    data: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    stride: usize,
    taps: Taps,
) -> Input<'_> {
    Input::Image {
        data,
        height,
        width,
        channels,
        stride,
        taps,
    }
}

/// `out[p] = act(input row p * m)` (or added to `out[p]`) for every row of
/// the input's output grid; output pixels are `stride` floats apart.
fn product(
    pool: &mut Pool,
    input: Input,
    m: &Matrix,
    out: &mut [f32],
    stride: usize,
    act: Act,
    accumulate: bool,
) {
    let (h, w) = input.out_size();
    let dst = Shared::new(out);
    pool.split(h * w, 96, |range, scratch| {
        // SAFETY: tasks own disjoint pixel ranges.
        let out = unsafe { dst.rows(range.clone(), stride, m.n) };
        multiply(
            input,
            range,
            m,
            out,
            stride,
            act,
            accumulate,
            &mut scratch.gemm,
        );
    });
}

fn matrix(
    file: &mut WeightFile,
    name: &str,
    k: usize,
    n: usize,
    bias: bool,
) -> Result<Matrix, Error> {
    let w = file.take_weights(&format!("{name}.weight"), &[k, n])?;
    let b = if bias {
        Some(file.take(&format!("{name}.bias"), &[n])?)
    } else {
        None
    };
    Ok(Matrix::new(&w, k, n, b.as_deref()))
}

struct Ffc {
    local: Matrix,
    cross: Matrix,
    spectral_in: Matrix,
    fourier: Matrix,
    spectral_out: Matrix,
}

/// Working buffers of a fast Fourier convolution, reused by all of them.
#[derive(Default)]
struct Buffers {
    spectral: Vec<f32>,
    spectrum: Vec<f32>,
    filtered: Vec<f32>,
    work: Vec<f32>,
}

/// The first `len` floats of `buffer`, whatever an earlier use left there. A
/// buffer grows as new zeroed memory, whose pages the pool's threads then
/// fault in as they write, instead of the calling thread zeroing it alone.
fn grown(buffer: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buffer.len() < len {
        *buffer = vec![0.0; len];
    }
    &mut buffer[..len]
}

impl Ffc {
    fn load(file: &mut WeightFile, name: &str) -> Result<Self, Error> {
        Ok(Self {
            local: matrix(file, &format!("{name}.local"), 9 * CHANNELS, LOCAL, true)?,
            cross: matrix(file, &format!("{name}.cross"), 9 * LOCAL, GLOBAL, true)?,
            spectral_in: matrix(file, &format!("{name}.spectral_in"), GLOBAL, SPECTRAL, true)?,
            fourier: matrix(
                file,
                &format!("{name}.fourier"),
                2 * SPECTRAL,
                2 * SPECTRAL,
                true,
            )?,
            spectral_out: matrix(
                file,
                &format!("{name}.spectral_out"),
                SPECTRAL,
                GLOBAL,
                false,
            )?,
        })
    }

    /// `out = ffc(x)`, both maps of 512 channels the size `fourier` works on.
    fn run(
        &self,
        pool: &mut Pool,
        x: &[f32],
        out: &mut [f32],
        fourier: &Fourier,
        buffers: &mut Buffers,
    ) {
        let (h, w) = fourier.size();
        let pixels = h * w;
        let spectrum_pixels = h * fourier.half();
        let conv3 = conv(3, 1);
        winograd::conv3(
            pool,
            input(x, h, w, CHANNELS, CHANNELS, conv3),
            &self.local,
            out,
            CHANNELS,
            false,
            relu,
        );
        let global_out = &mut out[LOCAL..];
        winograd::conv3(
            pool,
            input(x, h, w, LOCAL, CHANNELS, conv3),
            &self.cross,
            global_out,
            CHANNELS,
            false,
            |_| {},
        );
        let spectral = grown(&mut buffers.spectral, pixels * SPECTRAL);
        let global_in = input(&x[LOCAL..], h, w, GLOBAL, CHANNELS, POINT);
        product(
            pool,
            global_in,
            &self.spectral_in,
            spectral,
            SPECTRAL,
            Act::Relu,
            false,
        );
        let spectrum = grown(&mut buffers.spectrum, spectrum_pixels * 2 * SPECTRAL);
        let work = grown(&mut buffers.work, spectrum_pixels * 2 * SPECTRAL);
        fourier.forward(pool, spectral, SPECTRAL, work, spectrum);
        let filtered = grown(&mut buffers.filtered, spectrum_pixels * 2 * SPECTRAL);
        let rows = input(
            spectrum,
            1,
            spectrum_pixels,
            2 * SPECTRAL,
            2 * SPECTRAL,
            POINT,
        );
        product(
            pool,
            rows,
            &self.fourier,
            filtered,
            2 * SPECTRAL,
            Act::Relu,
            false,
        );
        fourier.inverse(pool, filtered, SPECTRAL, work, spectral);
        let rows = input(spectral, 1, pixels, SPECTRAL, SPECTRAL, POINT);
        product(
            pool,
            rows,
            &self.spectral_out,
            global_out,
            CHANNELS,
            Act::Relu,
            true,
        );
    }
}

fn relu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
}

/// `x += y`.
fn add(pool: &mut Pool, x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    let dst = Shared::new(x);
    pool.split_by(y.len(), 1 << 14, 1 << 16, |range, _| {
        // SAFETY: tasks own disjoint ranges.
        let dst = unsafe { dst.part(range.start, range.len()) };
        for (d, s) in dst.iter_mut().zip(&y[range]) {
            *d += s;
        }
    });
}

/// The offsets and kernel taps one output phase of a 3x3 stride-2 transposed
/// convolution reads along an axis: output `2m` reads input `m` through tap
/// 1, output `2m + 1` reads input `m` through tap 2 and `m + 1` through tap 0.
pub const PHASE_TAPS: [&[(usize, usize)]; 2] = [&[(0, 1)], &[(0, 2), (1, 0)]];

struct Up {
    /// One product per output phase `2 py + px`, of `cin` times its taps.
    phases: Vec<(Taps, Matrix)>,
    cout: usize,
}

impl Up {
    fn load(file: &mut WeightFile, name: &str, cin: usize, cout: usize) -> Result<Self, Error> {
        let mut phases = Vec::with_capacity(4);
        for p in 0..4 {
            let (ys, xs) = (PHASE_TAPS[p / 2], PHASE_TAPS[p % 2]);
            let mut offsets = [(0, 0); 4];
            let mut count = 0;
            for &(dy, _) in ys {
                for &(dx, _) in xs {
                    offsets[count] = (dy, dx);
                    count += 1;
                }
            }
            let m = matrix(file, &format!("{name}.phase{p}"), count * cin, cout, true)?;
            phases.push((Taps::Offsets { offsets, count }, m));
        }
        Ok(Self { phases, cout })
    }

    /// The doubled map of a `height x width` map `x`, through a ReLU.
    fn run(&self, pool: &mut Pool, x: &[f32], height: usize, width: usize) -> Vec<f32> {
        // Phase 0 reads one tap: its matrix has a row per input channel.
        let cin = self.phases[0].1.k;
        let cout = self.cout;
        let row = 2 * width * cout;
        let mut out = vec![0.0; 2 * height * row];
        let dst = Shared::new(&mut out);
        for (p, (taps, m)) in self.phases.iter().enumerate() {
            let (py, px) = (p / 2, p % 2);
            let input = input(x, height, width, cin, cin, *taps);
            pool.split_by(height, 1, 8, |rows, scratch| {
                for y in rows {
                    // SAFETY: tasks own disjoint rows of the phase, which are
                    // disjoint output rows.
                    let out = unsafe {
                        dst.part(
                            (2 * y + py) * row + px * cout,
                            (width - 1) * 2 * cout + cout,
                        )
                    };
                    multiply(
                        input,
                        y * width..(y + 1) * width,
                        m,
                        out,
                        2 * cout,
                        Act::Relu,
                        false,
                        &mut scratch.gemm,
                    );
                }
            });
        }
        out
    }
}

/// The sides a `width x height` image is padded to: multiples of
/// [`MULTIPLE`].
pub fn padded(width: usize, height: usize) -> (usize, usize) {
    (
        height.next_multiple_of(MULTIPLE),
        width.next_multiple_of(MULTIPLE),
    )
}

/// The masked image and the mask, 4 channels, padded to the sides of
/// [`padded`] by mirroring past the bottom and right edges ("symmetric": the
/// edge pixel repeats), handed to `row` a row at a time with its index.
pub fn masked_rows(
    image: &[f32],
    mask: &[f32],
    width: usize,
    height: usize,
    mut row: impl FnMut(usize, &[f32]),
) {
    let (h, w) = padded(width, height);
    let mut values = vec![0.0; w * 4];
    for y in 0..h {
        let sy = if y < height { y } else { 2 * height - 1 - y };
        for (xx, pixel) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let sx = if xx < width { xx } else { 2 * width - 1 - xx };
            let p = sy * width + sx;
            let m = mask[p];
            for c in 0..3 {
                pixel[c] = image[p * 3 + c] * (1.0 - m);
            }
            pixel[3] = m;
        }
        row(y, &values);
    }
}

pub struct Network {
    first: Matrix,
    down: Vec<Matrix>,
    blocks: Vec<(Ffc, Ffc)>,
    up: Vec<Up>,
    last: Matrix,
}

impl Network {
    pub fn load(file: &mut WeightFile) -> Result<Self, Error> {
        let first = matrix(file, "first", 49 * 4, 64, true)?;
        let down = [(64, 128), (128, 256), (256, CHANNELS)]
            .iter()
            .enumerate()
            .map(|(i, &(cin, cout))| matrix(file, &format!("down{i}"), 9 * cin, cout, true))
            .collect::<Result<_, _>>()?;
        let blocks = (0..BLOCKS)
            .map(|b| {
                Ok((
                    Ffc::load(file, &format!("block{b}.ffc0"))?,
                    Ffc::load(file, &format!("block{b}.ffc1"))?,
                ))
            })
            .collect::<Result<_, Error>>()?;
        let up = [(CHANNELS, 256), (256, 128), (128, 64)]
            .iter()
            .enumerate()
            .map(|(i, &(cin, cout))| Up::load(file, &format!("up{i}"), cin, cout))
            .collect::<Result<_, _>>()?;
        let last = matrix(file, "last", 49 * 64, 3, true)?;
        Ok(Self {
            first,
            down,
            blocks,
            up,
            last,
        })
    }

    /// The network's prediction for a `width x height` RGB image with values
    /// in 0..=1 and a mask, 1 where the image is to be filled and 0 elsewhere,
    /// NHWC. `step` is called after each of the [`STEPS`] steps, and the run
    /// stops with [`Error::Cancelled`] when it breaks.
    pub fn run(
        &self,
        pool: &mut Pool,
        image: &[f32],
        mask: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<Vec<f32>, Error> {
        let mut step_done = || match step() {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(()) => Err(Error::Cancelled),
        };
        let (mut h, mut w) = padded(width, height);
        let mut x = Vec::with_capacity(h * w * 4);
        masked_rows(image, mask, width, height, |_, row| {
            x.extend_from_slice(row)
        });
        let mut a = vec![0.0; h * w * 64];
        product(
            pool,
            input(&x, h, w, 4, 4, conv(7, 1)),
            &self.first,
            &mut a,
            64,
            Act::Relu,
            false,
        );
        drop(x);
        step_done()?;
        let mut channels = 64;
        for m in &self.down {
            let taps = conv(3, 2);
            let next_input = input(&a, h, w, channels, channels, taps);
            let (nh, nw) = next_input.out_size();
            let mut next = vec![0.0; nh * nw * m.n];
            product(pool, next_input, m, &mut next, m.n, Act::Relu, false);
            (a, h, w, channels) = (next, nh, nw, m.n);
            step_done()?;
        }
        let fourier = Fourier::new(h, w);
        let mut buffers = Buffers::default();
        let mut mid = vec![0.0; h * w * CHANNELS];
        let mut out = vec![0.0; h * w * CHANNELS];
        for (first, second) in &self.blocks {
            first.run(pool, &a, &mut mid, &fourier, &mut buffers);
            second.run(pool, &mid, &mut out, &fourier, &mut buffers);
            add(pool, &mut a, &out);
            step_done()?;
        }
        drop((mid, out, buffers));
        for up in &self.up {
            a = up.run(pool, &a, h, w);
            (h, w) = (2 * h, 2 * w);
            step_done()?;
        }
        let mut rgb = vec![0.0; h * w * 3];
        product(
            pool,
            input(&a, h, w, 64, 64, conv(7, 1)),
            &self.last,
            &mut rgb,
            3,
            Act::None,
            false,
        );
        drop(a);
        crate::math::sigmoid(&mut rgb);
        step_done()?;
        let mut cropped = Vec::with_capacity(width * height * 3);
        for row in rgb.chunks_exact(w * 3).take(height) {
            cropped.extend_from_slice(&row[..width * 3]);
        }
        Ok(cropped)
    }
}
