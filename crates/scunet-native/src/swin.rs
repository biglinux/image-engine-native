//! The transformer half of an SCUNet block: window attention over 8 x 8
//! windows, shifted by 4 pixels in every second block, then an MLP, each
//! with a pre-normalisation and a residual.
//!
//! The block works on channels `offset..offset + c` of a feature map whose
//! pixels are `stride` floats apart, in place. A task gathers a few windows'
//! tokens (rolled by half a window when shifted), normalises them, and adds
//! the projected attention back onto the same pixels, which no other window
//! touches. The MLP runs on a chunk of pixels for its caller, which applies
//! the 1x1 convolutions around it to the same chunk while it is in cache.

use crate::Error;
use crate::gemm::{Act, Input, Matrix, multiply};
use crate::math::{self, HEAD_DIM, WINDOW_TOKENS};
use crate::pool::{Pool, Scratch, Shared};
use crate::weight_file::WeightFile;

pub const WINDOW: usize = 8;
const SHIFT: usize = WINDOW / 2;
const MAX_WINDOWS_PER_TASK: usize = 4;

/// The position bias of a block, followed for a shifted block by its sums
/// with the three shift masks. In a window last along an axis, the first four
/// rows (or columns) and the last four came from opposite ends of the map,
/// and the reference keeps them from attending to each other.
pub fn bias_tables(position: Vec<f32>, shifted: bool) -> Vec<f32> {
    if !shifted {
        return position;
    }
    let n = WINDOW_TOKENS;
    let region = |v: usize, last: bool| usize::from(last && v >= WINDOW - SHIFT);
    let mut tables = Vec::with_capacity(4 * position.len());
    for class in 0..4 {
        let (last_col, last_row) = (class & 1 == 1, class & 2 == 2);
        let id = |i: usize| region(i / WINDOW, last_row) * 2 + region(i % WINDOW, last_col);
        for head in position.chunks_exact(n * n) {
            for i in 0..n {
                for j in 0..n {
                    let mask = if id(i) == id(j) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    };
                    tables.push(head[i * n + j] + mask);
                }
            }
        }
    }
    tables
}

pub struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl Norm {
    fn load(file: &mut WeightFile, name: &str, c: usize) -> Result<Self, Error> {
        Ok(Self {
            gamma: file.take(&format!("{name}.weight"), &[c])?,
            beta: file.take(&format!("{name}.bias"), &[c])?,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        math::layer_norm(x, &self.gamma, &self.beta);
    }
}

pub fn linear(
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

pub struct Transformer {
    c: usize,
    shifted: bool,
    norm1: Norm,
    qkv: Matrix,
    /// Relative position bias, `[class][heads][64][64]`. A shifted block has
    /// four classes, by whether the window is last in its row (1), its column
    /// (2) or both, which sets the mask of pixels the roll brought together.
    bias: Vec<f32>,
    proj: Matrix,
    norm2: Norm,
    fc1: Matrix,
    fc2: Matrix,
}

impl Transformer {
    pub fn load(file: &mut WeightFile, name: &str, c: usize, shifted: bool) -> Result<Self, Error> {
        let heads = c / HEAD_DIM;
        let n = WINDOW_TOKENS;
        Ok(Self {
            c,
            shifted,
            norm1: Norm::load(file, &format!("{name}.ln1"), c)?,
            qkv: linear(file, &format!("{name}.msa.embedding_layer"), c, 3 * c, true)?,
            bias: bias_tables(
                file.take(&format!("{name}.msa.position_bias"), &[heads, n, n])?,
                shifted,
            ),
            proj: linear(file, &format!("{name}.msa.linear"), c, c, true)?,
            norm2: Norm::load(file, &format!("{name}.ln2"), c)?,
            fc1: linear(file, &format!("{name}.mlp.0"), c, 4 * c, true)?,
            fc2: linear(file, &format!("{name}.mlp.2"), 4 * c, c, true)?,
        })
    }

    /// The attention with its residual, on the `height x width` map whose
    /// pixel `p` starts at `map[p * stride]`; both sides are multiples of
    /// [`WINDOW`].
    pub fn attend(
        &self,
        pool: &mut Pool,
        map: &mut [f32],
        stride: usize,
        height: usize,
        width: usize,
    ) {
        let c = self.c;
        let (rows_of_windows, cols_of_windows) = (height / WINDOW, width / WINDOW);
        let heads = c / HEAD_DIM;
        let n = WINDOW_TOKENS;
        let shift = if self.shifted { SHIFT } else { 0 };
        let map = Shared::new(map);
        // Up to four windows per task, so that the projections run over more
        // rows per pass over their weights.
        pool.split_by(
            rows_of_windows * cols_of_windows,
            1,
            MAX_WINDOWS_PER_TASK,
            |batch, scratch| {
                let rows = batch.len() * n;
                let sizes = [rows * c, rows * 3 * c, rows * c, math::ATTENTION_WORK];
                let (buf, gemm) = scratch.floats(sizes.iter().sum());
                let (tokens, rest) = buf.split_at_mut(sizes[0]);
                let (qkv, rest) = rest.split_at_mut(sizes[1]);
                let (heads_out, rest) = rest.split_at_mut(sizes[2]);
                let work = &mut rest[..sizes[3]];
                let mut source = [0; MAX_WINDOWS_PER_TASK * WINDOW_TOKENS];
                for (k, w) in batch.clone().enumerate() {
                    let (wy, wx) = (w / cols_of_windows, w % cols_of_windows);
                    for i in 0..n {
                        let y = (wy * WINDOW + i / WINDOW + shift) % height;
                        let x = (wx * WINDOW + i % WINDOW + shift) % width;
                        let pixel = y * width + x;
                        source[k * n + i] = pixel;
                        let row = &mut tokens[(k * n + i) * c..][..c];
                        // SAFETY: windows partition the pixels; this task alone
                        // reads and writes the pixels of its windows.
                        row.copy_from_slice(unsafe { map.part(pixel * stride, c) });
                        self.norm1.apply(row);
                    }
                }
                multiply(
                    Input::Rows {
                        data: tokens,
                        stride: c,
                    },
                    0..rows,
                    &self.qkv,
                    qkv,
                    3 * c,
                    Act::None,
                    false,
                    gemm,
                );
                for (k, w) in batch.enumerate() {
                    let class = if self.shifted {
                        usize::from(w % cols_of_windows == cols_of_windows - 1)
                            + 2 * usize::from(w / cols_of_windows == rows_of_windows - 1)
                    } else {
                        0
                    };
                    for h in 0..heads {
                        let bias = &self.bias[(class * heads + h) * n * n..][..n * n];
                        let out = &mut heads_out[k * n * c..][..n * c];
                        math::attention(&qkv[k * n * 3 * c..][..n * 3 * c], c, h, bias, work, out);
                    }
                }
                // The projection writes over the token rows; attention is done with them.
                multiply(
                    Input::Rows {
                        data: heads_out,
                        stride: c,
                    },
                    0..rows,
                    &self.proj,
                    tokens,
                    c,
                    Act::None,
                    false,
                    gemm,
                );
                for (i, &pixel) in source[..rows].iter().enumerate() {
                    // SAFETY: as above.
                    let dst = unsafe { map.part(pixel * stride, c) };
                    for (d, v) in dst.iter_mut().zip(&tokens[i * c..][..c]) {
                        *d += v;
                    }
                }
            },
        );
    }

    /// The MLP with its residual, on the `rows` pixels whose channels start
    /// at `dst[r * stride]`, with the calling task's scratch.
    pub fn mlp(&self, dst: &mut [f32], stride: usize, rows: usize, scratch: &mut Scratch) {
        let c = self.c;
        let (buf, gemm) = scratch.floats(rows * 5 * c);
        let (normed, hidden) = buf.split_at_mut(rows * c);
        for (r, row) in normed.chunks_exact_mut(c).enumerate() {
            row.copy_from_slice(&dst[r * stride..][..c]);
        }
        self.norm2.apply(normed);
        multiply(
            Input::Rows {
                data: normed,
                stride: c,
            },
            0..rows,
            &self.fc1,
            hidden,
            4 * c,
            Act::Gelu,
            false,
            gemm,
        );
        multiply(
            Input::Rows {
                data: hidden,
                stride: 4 * c,
            },
            0..rows,
            &self.fc2,
            dst,
            stride,
            Act::None,
            true,
            gemm,
        );
    }
}
