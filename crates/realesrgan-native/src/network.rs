//! Real-ESRGAN's compact network (SRVGGNetCompact, as in
//! "realesr-general-x4v3"): a 3x3 convolution from RGB to 64 channels and 32
//! more from 64 to 64, each followed by a PReLU, then a 3x3 convolution to 48
//! channels that a pixel shuffle turns into an RGB image four times larger,
//! added to the input enlarged by repeating each pixel.
//!
//! Everything before the shuffle runs at the input's size, NHWC, on two
//! 64-channel maps that take turns as a layer's input and output. The
//! convolutions run as Winograd's F(4x4, 3x3) (the engine's `winograd`), and
//! the task that computes a layer's blocks applies the PReLU to them while
//! they are in cache.

use std::ops::ControlFlow;

use crate::Error;
use crate::gemm::{Input, Matrix};
use crate::pool::{Pool, Shared};
use crate::weight_file::WeightFile;
use native_engine::winograd;

/// How much larger the output is, on each side.
pub const SCALE: usize = 4;
pub const FEATURES: usize = 64;
const BODY: usize = 32;
/// Layers of a run.
pub const STEPS: usize = BODY + 2;
/// How far, in input pixels, an output pixel depends on the input: one per
/// 3x3 convolution.
pub const REACH: usize = STEPS;

dispatch! {
    /// PReLU in place on rows of `slope.len()` channels: negative values
    /// times their channel's slope.
    prelu(x: &mut [f32], slope: &[f32]) {
        for row in x.chunks_exact_mut(slope.len()) {
            for (v, a) in row.iter_mut().zip(slope) {
                *v = if *v < 0.0 { *v * a } else { *v };
            }
        }
    }
}

struct Layer {
    conv: Matrix,
    /// The PReLU after it; the last layer has none.
    slope: Option<Vec<f32>>,
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn load(file: &mut WeightFile) -> Result<Self, Error> {
        let mut layers = Vec::with_capacity(STEPS);
        for i in 0..STEPS {
            let cin = if i == 0 { 3 } else { FEATURES };
            let last = i == STEPS - 1;
            let cout = if last { 3 * SCALE * SCALE } else { FEATURES };
            let name = format!("body.{}", 2 * i);
            let w = file.take_weights(&format!("{name}.weight"), &[9 * cin, cout])?;
            let b = file.take(&format!("{name}.bias"), &[cout])?;
            let slope = if last {
                None
            } else {
                Some(file.take(&format!("body.{}.weight", 2 * i + 1), &[FEATURES])?)
            };
            layers.push(Layer {
                conv: Matrix::new(&w, 9 * cin, cout, Some(&b)),
                slope,
            });
        }
        Ok(Self { layers })
    }

    /// The network's output for a `width x height` RGB image with values in
    /// 0..=1, NHWC, [`SCALE`] times larger on each side. `step` is called
    /// after each of the [`STEPS`] layers, and the run stops with
    /// [`Error::Cancelled`] when it breaks.
    pub fn run(
        &self,
        pool: &mut Pool,
        image: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<Vec<f32>, Error> {
        let pixels = width * height;
        let mut layer_done = || match step() {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(()) => Err(Error::Cancelled),
        };
        let mut a = vec![0.0; pixels * FEATURES];
        let mut b = vec![0.0; pixels * FEATURES];
        let mut channels = 3;
        for (i, layer) in self.layers.iter().enumerate() {
            let input = if i == 0 { image } else { &a[..] };
            let input = Input::conv(input, height, width, channels, channels, 3);
            let n = layer.conv.n;
            let slope = layer.slope.as_deref();
            winograd::conv3(pool, input, &layer.conv, &mut b, n, false, |pixel| {
                if let Some(slope) = slope {
                    prelu(pixel, slope);
                }
            });
            std::mem::swap(&mut a, &mut b);
            channels = n;
            layer_done()?;
        }
        drop(b);
        Ok(shuffled(pool, &a, image, width, height))
    }
}

/// Pixel shuffle of the last layer's 48 channels plus the input repeated:
/// output pixel `(4y + i, 4x + j)` has channel `c` from input pixel `(y, x)`'s
/// channel `16c + 4i + j`, plus that pixel's own channel `c`.
fn shuffled(pool: &mut Pool, last: &[f32], image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let row = width * SCALE * 3;
    let mut out = vec![0.0; height * SCALE * row];
    let dst = Shared::new(&mut out);
    pool.split_by(height, 1, 16, |rows, _| {
        // SAFETY: tasks own disjoint input rows, which own disjoint output rows.
        let out = unsafe { dst.part(rows.start * SCALE * row, rows.len() * SCALE * row) };
        for y in rows.clone() {
            for x in 0..width {
                let p = y * width + x;
                for i in 0..SCALE {
                    for j in 0..SCALE {
                        let o = ((y - rows.start) * SCALE + i) * row + (x * SCALE + j) * 3;
                        for c in 0..3 {
                            let v = last[p * 3 * SCALE * SCALE + c * SCALE * SCALE + i * SCALE + j];
                            out[o + c] = v + image[p * 3 + c];
                        }
                    }
                }
            }
        }
    });
    out
}
