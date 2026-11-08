//! The Swin-T backbone: a 4x4 patch embedding, then four stages of shifted
//! window attention blocks (2, 2, 6 and 2 of them, 96 to 768 channels), each
//! stage but the last ending in a 2x2 patch merge.
//!
//! Tokens are rows of a `side x side x channels` NHWC map. A block's attention
//! runs a few windows per task: the task gathers each window's 49 tokens
//! (rolled by half a window in odd blocks, zero past the map's edge, as the
//! reference pads after the first normalisation), normalises them, and adds
//! the projected attention back onto the same tokens, which no other window
//! touches. The MLP then runs over chunks of tokens and adds onto them in
//! place.

use crate::Error;
use crate::gemm::{Act, Input, Matrix, multiply};
use crate::math::{self, HEAD_DIM, WINDOW_TOKENS};
use crate::pool::{Pool, Shared};
use crate::weight_file::WeightFile;

pub(crate) const WINDOW: usize = 7;
pub(crate) const SHIFT: usize = 3;
/// What the reference adds to scores between tokens of different regions of
/// a shifted window, which makes their softmax weight vanish.
const MASKED: f32 = -100.0;
pub(crate) const STAGES: [(usize, usize); 4] = [(96, 2), (192, 2), (384, 6), (768, 2)];
const MAX_WINDOWS_PER_TASK: usize = 4;

/// The position bias of a block, followed for a shifted block by its sums with
/// the three shift masks. In a window last along an axis, the first four rows
/// (or columns) and the last three came from opposite ends of the map; the
/// reference adds -100 to scores between tokens from different regions.
pub(crate) fn bias_tables(position: Vec<f32>, shifted: bool) -> Vec<f32> {
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
                    let mask = if id(i) == id(j) { 0.0 } else { MASKED };
                    tables.push(head[i * n + j] + mask);
                }
            }
        }
    }
    tables
}

struct Norm {
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

fn linear(
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

struct Block {
    norm1: Norm,
    qkv: Matrix,
    /// Relative position bias, `[class][heads][49][49]`. A shifted block has
    /// four classes, by whether the window is last in its row (1), its column
    /// (2) or both, which sets the mask of tokens the roll brought together.
    bias: Vec<f32>,
    proj: Matrix,
    norm2: Norm,
    fc1: Matrix,
    fc2: Matrix,
}

struct Stage {
    channels: usize,
    blocks: Vec<Block>,
    /// The normalisation of the stage's output feature map.
    norm: Norm,
    merge: Option<(Norm, Matrix)>,
}

pub struct Backbone {
    embed: Matrix,
    embed_norm: Norm,
    stages: Vec<Stage>,
}

impl Backbone {
    pub fn load(file: &mut WeightFile) -> Result<Self, Error> {
        let embed = linear(file, "patch_embed", 48, 96, true)?;
        let embed_norm = Norm::load(file, "patch_embed.norm", 96)?;
        let mut stages = Vec::new();
        for (s, &(c, depth)) in STAGES.iter().enumerate() {
            let heads = c / HEAD_DIM;
            let mut blocks = Vec::new();
            for b in 0..depth {
                let p = format!("stage{s}.block{b}");
                blocks.push(Block {
                    norm1: Norm::load(file, &format!("{p}.norm1"), c)?,
                    qkv: linear(file, &format!("{p}.qkv"), c, 3 * c, true)?,
                    bias: bias_tables(
                        file.take(&format!("{p}.position_bias"), &[heads, 49, 49])?,
                        b % 2 == 1,
                    ),
                    proj: linear(file, &format!("{p}.proj"), c, c, true)?,
                    norm2: Norm::load(file, &format!("{p}.norm2"), c)?,
                    fc1: linear(file, &format!("{p}.fc1"), c, 4 * c, true)?,
                    fc2: linear(file, &format!("{p}.fc2"), 4 * c, c, true)?,
                });
            }
            let merge = if s < 3 {
                Some((
                    Norm::load(file, &format!("stage{s}.merge.norm"), 4 * c)?,
                    linear(file, &format!("stage{s}.merge"), 4 * c, 2 * c, false)?,
                ))
            } else {
                None
            };
            let norm = Norm::load(file, &format!("stage{s}.norm"), c)?;
            stages.push(Stage {
                channels: c,
                blocks,
                norm,
                merge,
            });
        }
        Ok(Self {
            embed,
            embed_norm,
            stages,
        })
    }

    /// The four normalised stage outputs for a `side x side` normalised RGB
    /// image (NHWC), at 1/4, 1/8, 1/16 and 1/32 of its side.
    pub fn run(&self, pool: &mut Pool, image: &[f32], side: usize) -> Vec<Vec<f32>> {
        let mut grid = side / 4;
        let mut x = self.embed(pool, image, side);
        let mut features = Vec::with_capacity(4);
        for stage in &self.stages {
            let c = stage.channels;
            for (b, block) in stage.blocks.iter().enumerate() {
                let shift = if b % 2 == 1 { SHIFT } else { 0 };
                block.attend(pool, &mut x, grid, c, shift);
                block.mlp(pool, &mut x, c);
            }
            features.push(normalized(pool, &x, &stage.norm));
            if let Some((norm, reduction)) = &stage.merge {
                x = merge(pool, &x, grid, c, norm, reduction);
                grid /= 2;
            }
        }
        features
    }

    fn embed(&self, pool: &mut Pool, image: &[f32], side: usize) -> Vec<f32> {
        let grid = side / 4;
        let tokens = grid * grid;
        let mut x = vec![0.0; tokens * 96];
        let out = Shared::new(&mut x);
        pool.split(tokens, 96, |range, scratch| {
            let rows = range.len();
            let (patches, gemm) = scratch.floats(rows * 48);
            for (r, row) in range.clone().zip(patches.as_chunks_mut::<48>().0) {
                let (ty, tx) = (r / grid, r % grid);
                for ky in 0..4 {
                    let src = &image[((ty * 4 + ky) * side + tx * 4) * 3..][..12];
                    row[ky * 12..][..12].copy_from_slice(src);
                }
            }
            // SAFETY: tasks own disjoint token ranges.
            let dst = unsafe { out.rows(range, 96, 96) };
            multiply(
                Input::Rows {
                    data: patches,
                    stride: 48,
                },
                0..rows,
                &self.embed,
                dst,
                96,
                Act::None,
                false,
                gemm,
            );
            self.embed_norm.apply(dst);
        });
        x
    }
}

impl Block {
    fn attend(&self, pool: &mut Pool, x: &mut [f32], grid: usize, c: usize, shift: usize) {
        let padded = grid.div_ceil(WINDOW) * WINDOW;
        let windows = padded / WINDOW;
        let total = windows * windows;
        let heads = c / HEAD_DIM;
        let n = WINDOW_TOKENS;
        let map = Shared::new(x);
        // Up to four windows per task, so that the projections run over more
        // rows per pass over their weights.
        pool.split_by(total, 1, MAX_WINDOWS_PER_TASK, |batch, scratch| {
            let (w0, w1) = (batch.start, batch.end);
            let rows = (w1 - w0) * n;
            let sizes = [rows * c, rows * 3 * c, rows * c, math::ATTENTION_WORK];
            let (buf, gemm) = scratch.floats(sizes.iter().sum());
            let (tokens, rest) = buf.split_at_mut(sizes[0]);
            let (qkv, rest) = rest.split_at_mut(sizes[1]);
            let (heads_out, rest) = rest.split_at_mut(sizes[2]);
            let work = &mut rest[..sizes[3]];
            let mut source = [usize::MAX; MAX_WINDOWS_PER_TASK * WINDOW_TOKENS];
            for (k, w) in (w0..w1).enumerate() {
                let (wy, wx) = (w / windows, w % windows);
                for i in 0..n {
                    let (py, px) = (wy * WINDOW + i / WINDOW, wx * WINDOW + i % WINDOW);
                    let (y, xx) = ((py + shift) % padded, (px + shift) % padded);
                    let row = &mut tokens[(k * n + i) * c..][..c];
                    if y < grid && xx < grid {
                        let token = y * grid + xx;
                        source[k * n + i] = token;
                        // SAFETY: windows partition the tokens; this task alone
                        // reads and writes the tokens of its windows.
                        row.copy_from_slice(unsafe { map.part(token * c, c) });
                        self.norm1.apply(row);
                    } else {
                        row.fill(0.0);
                    }
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
            for (k, w) in (w0..w1).enumerate() {
                let class = if shift > 0 {
                    usize::from(w % windows == windows - 1)
                        + 2 * usize::from(w / windows == windows - 1)
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
            for (i, &token) in source[..rows].iter().enumerate() {
                if token != usize::MAX {
                    // SAFETY: as above.
                    let dst = unsafe { map.part(token * c, c) };
                    for (d, v) in dst.iter_mut().zip(&tokens[i * c..][..c]) {
                        *d += v;
                    }
                }
            }
        });
    }

    fn mlp(&self, pool: &mut Pool, x: &mut [f32], c: usize) {
        let tokens = x.len() / c;
        let x = Shared::new(x);
        pool.split(tokens, 96, |range, scratch| {
            let rows = range.len();
            let (buf, gemm) = scratch.floats(rows * 5 * c);
            let (normed, hidden) = buf.split_at_mut(rows * c);
            // SAFETY: tasks own disjoint token ranges.
            let dst = unsafe { x.rows(range, c, c) };
            normed.copy_from_slice(dst);
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
                c,
                Act::None,
                true,
                gemm,
            );
        });
    }
}

/// A normalised copy of `x`.
fn normalized(pool: &mut Pool, x: &[f32], norm: &Norm) -> Vec<f32> {
    let c = norm.gamma.len();
    let mut out = vec![0.0; x.len()];
    let dst = Shared::new(&mut out);
    pool.split(x.len() / c, 96, |range, _| {
        let src = &x[range.start * c..range.end * c];
        // SAFETY: tasks own disjoint token ranges.
        let dst = unsafe { dst.rows(range, c, c) };
        dst.copy_from_slice(src);
        norm.apply(dst);
    });
    out
}

/// Patch merging: each 2x2 group of tokens, in the order (0,0), (1,0), (0,1),
/// (1,1) by (row, column), concatenated, normalised and reduced to `2c`.
fn merge(
    pool: &mut Pool,
    x: &[f32],
    grid: usize,
    c: usize,
    norm: &Norm,
    reduction: &Matrix,
) -> Vec<f32> {
    let half = grid / 2;
    let tokens = half * half;
    let mut out = vec![0.0; tokens * 2 * c];
    let dst = Shared::new(&mut out);
    pool.split(tokens, 96, |range, scratch| {
        let rows = range.len();
        let (buf, gemm) = scratch.floats(rows * 4 * c);
        for (r, row) in range.clone().zip(buf.chunks_exact_mut(4 * c)) {
            let (y, xx) = (2 * (r / half), 2 * (r % half));
            for (q, (dy, dx)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                row[q * c..][..c].copy_from_slice(&x[((y + dy) * grid + xx + dx) * c..][..c]);
            }
        }
        norm.apply(buf);
        // SAFETY: tasks own disjoint token ranges.
        let out = unsafe { dst.rows(range, 2 * c, 2 * c) };
        multiply(
            Input::Rows {
                data: buf,
                stride: 4 * c,
            },
            0..rows,
            reduction,
            out,
            2 * c,
            Act::None,
            false,
            gemm,
        );
    });
    out
}
