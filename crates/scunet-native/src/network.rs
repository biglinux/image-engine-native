//! The SCUNet U-Net: a 3x3 head convolution to 64 channels, three levels
//! down (64, 128 and 256 channels, each halving the map with a 2x2 stride-2
//! convolution), a 512-channel body, three levels back up (each doubling the
//! map with a 2x2 stride-2 transposed convolution after adding the skip of its
//! size), and a 3x3 tail convolution back to RGB. Every level has four
//! blocks.
//!
//! A block splits its channels with a 1x1 convolution: one half goes through
//! two 3x3 convolutions with a residual, the other through a transformer
//! block (`swin.rs`); a second 1x1 convolution joins them and adds onto the
//! block's input. Both halves stay in one buffer, the convolution half in the
//! first channels of each pixel and the transformer half after it, and are
//! updated in place.
//!
//! The image is padded to a multiple of 64 by repeating its last row and
//! column, as the reference does, and the output cropped back.

use std::ops::ControlFlow;

use crate::Error;
use crate::gemm::{Act, Input, Matrix, multiply};
use crate::pool::{Pool, Shared};
use crate::swin::{Transformer, linear};
use crate::weight_file::WeightFile;
use native_engine::winograd;

pub const DIM: usize = 64;
pub const BLOCKS: usize = 4;
/// Blocks of a run: four per level, seven levels, each about as costly as
/// the others (a level down has a quarter of the pixels and four times the
/// channel products).
pub const STEPS: usize = 7 * BLOCKS;
/// Sides are padded to a multiple of this: three halvings, then whole 8 x 8
/// windows.
const MULTIPLE: usize = 64;

fn conv3_matrix(
    file: &mut WeightFile,
    name: &str,
    cin: usize,
    cout: usize,
) -> Result<Matrix, Error> {
    let w = file.take_weights(&format!("{name}.weight"), &[9 * cin, cout])?;
    Ok(Matrix::new(&w, 9 * cin, cout, None))
}

fn conv3(data: &[f32], height: usize, width: usize, channels: usize, stride: usize) -> Input<'_> {
    Input::conv(data, height, width, channels, stride, 3)
}

/// `out[p] = act(input[p] * m)` (or added to `out[p]`) for every pixel `p`
/// of `0..pixels`; output pixels are `stride` floats apart.
#[expect(
    clippy::too_many_arguments,
    reason = "a product's input, weights and output layout"
)]
fn product(
    pool: &mut Pool,
    input: Input,
    pixels: usize,
    m: &Matrix,
    out: &mut [f32],
    stride: usize,
    act: Act,
    accumulate: bool,
) {
    let dst = Shared::new(out);
    pool.split(pixels, 96, |range, scratch| {
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

/// What a run leaves for the next tile of the same image, which needs the
/// same sizes at once: its top-level map and block buffers, kept instead of
/// freed and faulted in again.
#[derive(Default)]
pub struct Spare {
    x: Vec<f32>,
    buffers: Buffers,
}

/// Working buffers of a block, reused by every block of a run.
#[derive(Default)]
struct Buffers {
    split: Vec<f32>,
    conv: Vec<f32>,
}

/// The first `len` floats of `buffer`, whatever an earlier use left there. A
/// buffer short of capacity is replaced by new zeroed memory, whose pages the
/// pool's threads then fault in as they write, instead of the calling thread
/// zeroing it alone.
fn grown(buffer: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buffer.capacity() < len {
        *buffer = vec![0.0; len];
    } else if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    &mut buffer[..len]
}

struct Block {
    half: usize,
    split: Matrix,
    conv1: Matrix,
    conv2: Matrix,
    transformer: Transformer,
    join: Matrix,
}

impl Block {
    fn load(file: &mut WeightFile, name: &str, c: usize, shifted: bool) -> Result<Self, Error> {
        let half = c / 2;
        Ok(Self {
            half,
            split: linear(file, &format!("{name}.conv1_1"), c, c, true)?,
            conv1: conv3_matrix(file, &format!("{name}.conv_block.0"), half, half)?,
            conv2: conv3_matrix(file, &format!("{name}.conv_block.2"), half, half)?,
            transformer: Transformer::load(file, &format!("{name}.trans_block"), half, shifted)?,
            join: linear(file, &format!("{name}.conv1_2"), c, c, true)?,
        })
    }

    /// The convolution half, in place on the first `half` channels of `y`.
    fn convolve(
        &self,
        pool: &mut Pool,
        y: &mut [f32],
        height: usize,
        width: usize,
        t: &mut Vec<f32>,
    ) {
        let half = self.half;
        let pixels = height * width;
        let t = grown(t, pixels * half);
        let input = conv3(y, height, width, half, 2 * half);
        winograd::conv3(pool, input, &self.conv1, t, half, false, |pixel| {
            pixel.iter_mut().for_each(|v| *v = v.max(0.0));
        });
        let input = conv3(t, height, width, half, half);
        winograd::conv3(pool, input, &self.conv2, y, 2 * half, true, |_| {});
    }
}

/// The blocks of one level on the `height x width` map `x`. Each block's
/// MLP, its joining 1x1 convolution onto `x` and the next block's splitting
/// 1x1 convolution run on a chunk of pixels in one task: at the top levels,
/// where a pixel has few channels, those convolutions would otherwise wait on
/// memory streaming the whole map in and out. `block_done` is called after
/// each block.
fn run_level(
    pool: &mut Pool,
    blocks: &[Block],
    x: &mut [f32],
    height: usize,
    width: usize,
    buffers: &mut Buffers,
    block_done: &mut impl FnMut() -> Result<(), Error>,
) -> Result<(), Error> {
    let half = blocks[0].half;
    let c = 2 * half;
    let pixels = height * width;
    let y = grown(&mut buffers.split, pixels * c);
    let input = Input::Rows { data: x, stride: c };
    product(
        pool,
        input,
        pixels,
        &blocks[0].split,
        y,
        c,
        Act::None,
        false,
    );
    for (b, block) in blocks.iter().enumerate() {
        block.convolve(pool, y, height, width, &mut buffers.conv);
        block
            .transformer
            .attend(pool, &mut y[half..], c, height, width);
        let next = blocks.get(b + 1).map(|n| &n.split);
        let (xs, ys) = (Shared::new(x), Shared::new(y));
        pool.split(pixels, 96, |range, scratch| {
            let rows = range.len();
            // SAFETY: tasks own disjoint pixel ranges.
            let (xr, yr) = unsafe { (xs.rows(range.clone(), c, c), ys.rows(range, c, c)) };
            block.transformer.mlp(&mut yr[half..], c, rows, scratch);
            let gemm = &mut scratch.gemm;
            let input = Input::Rows {
                data: yr,
                stride: c,
            };
            multiply(input, 0..rows, &block.join, xr, c, Act::None, true, gemm);
            if let Some(split) = next {
                let input = Input::Rows {
                    data: xr,
                    stride: c,
                };
                multiply(input, 0..rows, split, yr, c, Act::None, false, gemm);
            }
        });
        block_done()?;
    }
    Ok(())
}

fn level(file: &mut WeightFile, name: &str, first: usize, c: usize) -> Result<Vec<Block>, Error> {
    (0..BLOCKS)
        .map(|b| Block::load(file, &format!("{name}.{}", first + b), c, b % 2 == 1))
        .collect()
}

pub struct Network {
    head: Matrix,
    /// Blocks and halving convolution of each level down.
    down: Vec<(Vec<Block>, Matrix)>,
    body: Vec<Block>,
    /// Doubling convolution and blocks of each level up.
    up: Vec<(Matrix, Vec<Block>)>,
    tail: Matrix,
}

impl Network {
    pub fn load(file: &mut WeightFile) -> Result<Self, Error> {
        let head = conv3_matrix(file, "m_head.0", 3, DIM)?;
        let mut down = Vec::new();
        for (i, c) in [(1, DIM), (2, 2 * DIM), (3, 4 * DIM)] {
            let name = format!("m_down{i}");
            let blocks = level(file, &name, 0, c)?;
            down.push((
                blocks,
                linear(file, &format!("{name}.{BLOCKS}"), 4 * c, 2 * c, false)?,
            ));
        }
        let body = level(file, "m_body", 0, 8 * DIM)?;
        let mut up = Vec::new();
        for (i, c) in [(3, 4 * DIM), (2, 2 * DIM), (1, DIM)] {
            let name = format!("m_up{i}");
            let double = linear(file, &format!("{name}.0"), 2 * c, 4 * c, false)?;
            up.push((double, level(file, &name, 1, c)?));
        }
        let tail = conv3_matrix(file, "m_tail.0", DIM, 3)?;
        Ok(Self {
            head,
            down,
            body,
            up,
            tail,
        })
    }

    /// The network's output for a `width x height` RGB image with values in
    /// 0..=1, NHWC. `step` is called after each of the [`STEPS`] blocks, and
    /// the run stops with [`Error::Cancelled`] when it breaks.
    pub fn run(
        &self,
        pool: &mut Pool,
        image: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
        spare: &mut Spare,
    ) -> Result<Vec<f32>, Error> {
        let mut block_done = || match step() {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(()) => Err(Error::Cancelled),
        };
        let (padded, mut h, mut w) = pad(image, width, height);
        // The head's output is also the skip into the tail; that is computed
        // again at the end from the image, which costs less than keeping a
        // full-size 64-channel map through the whole network.
        let (ph, pw) = (h, w);
        // The head overwrites every value of `x`.
        let mut x = std::mem::take(&mut spare.x);
        if x.capacity() < h * w * DIM {
            x = vec![0.0; h * w * DIM];
        }
        x.resize(h * w * DIM, 0.0);
        let head = |pool: &mut Pool, x: &mut [f32], accumulate| {
            let input = conv3(&padded, ph, pw, 3, 3);
            product(
                pool,
                input,
                ph * pw,
                &self.head,
                x,
                DIM,
                Act::None,
                accumulate,
            );
        };
        head(pool, &mut x, false);

        let mut buffers = std::mem::take(&mut spare.buffers);
        let mut skips = Vec::new();
        for (blocks, halve) in &self.down {
            run_level(pool, blocks, &mut x, h, w, &mut buffers, &mut block_done)?;
            // The next levels need a quarter of these buffers or less, and the
            // top level up grows them back when the halved maps are gone.
            buffers = Buffers::default();
            let next = halved(pool, &x, h, w, halve);
            // Free this level's map before the copy, not after.
            drop(std::mem::take(&mut x));
            x = copied(pool, &next);
            skips.push(next);
            (h, w) = (h / 2, w / 2);
        }
        run_level(
            pool,
            &self.body,
            &mut x,
            h,
            w,
            &mut buffers,
            &mut block_done,
        )?;
        for (double, blocks) in &self.up {
            add(pool, &mut x, &skips.pop().expect("one skip per level"));
            x = doubled(pool, &x, h, w, double);
            (h, w) = (2 * h, 2 * w);
            run_level(pool, blocks, &mut x, h, w, &mut buffers, &mut block_done)?;
        }
        spare.buffers = buffers;
        head(pool, &mut x, true);
        drop(padded);
        let mut out = vec![0.0; h * w * 3];
        product(
            pool,
            conv3(&x, h, w, DIM, DIM),
            h * w,
            &self.tail,
            &mut out,
            3,
            Act::None,
            false,
        );
        spare.x = x;
        let mut cropped = Vec::with_capacity(width * height * 3);
        for row in out.chunks_exact(w * 3).take(height) {
            cropped.extend_from_slice(&row[..width * 3]);
        }
        Ok(cropped)
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

/// The image padded to the sides of [`padded`] by repeating its last row and
/// column, and those sides.
pub fn pad(image: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let (h, w) = padded(width, height);
    let mut padded = vec![0.0; h * w * 3];
    for (y, row) in padded.chunks_exact_mut(w * 3).enumerate() {
        let src = &image[y.min(height - 1) * width * 3..][..width * 3];
        row[..width * 3].copy_from_slice(src);
        let last: [f32; 3] = src[(width - 1) * 3..].try_into().expect("one pixel");
        for pixel in row[width * 3..].as_chunks_mut::<3>().0 {
            *pixel = last;
        }
    }
    (padded, h, w)
}

/// `x += y`.
fn add(pool: &mut Pool, x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    let dst = Shared::new(x);
    pool.split_by(y.len(), 1 << 14, 1 << 16, |range, _| {
        let src = &y[range.clone()];
        // SAFETY: tasks own disjoint ranges.
        let dst = unsafe { dst.part(range.start, range.len()) };
        for (d, s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    });
}

/// `x` copied by the pool's threads, which also fault the copy's pages in,
/// instead of the calling thread alone.
fn copied(pool: &mut Pool, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let dst = Shared::new(&mut out);
    pool.split_by(x.len(), 1 << 14, 1 << 16, |range, _| {
        // SAFETY: tasks own disjoint ranges.
        let dst = unsafe { dst.part(range.start, range.len()) };
        dst.copy_from_slice(&x[range]);
    });
    out
}

/// The 2x2 stride-2 convolution of a `height x width` map: each output pixel
/// reads its 2x2 input pixels in row-major order, channels within each.
fn halved(pool: &mut Pool, x: &[f32], height: usize, width: usize, m: &Matrix) -> Vec<f32> {
    let c = m.k / 4;
    let (h, w) = (height / 2, width / 2);
    let mut out = vec![0.0; h * w * m.n];
    let dst = Shared::new(&mut out);
    pool.split(h * w, 96, |range, scratch| {
        let rows = range.len();
        let (buf, gemm) = scratch.floats(rows * 4 * c);
        for (r, row) in range.clone().zip(buf.chunks_exact_mut(4 * c)) {
            let (y, xx) = (2 * (r / w), 2 * (r % w));
            for (q, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                row[q * c..][..c].copy_from_slice(&x[((y + dy) * width + xx + dx) * c..][..c]);
            }
        }
        // SAFETY: tasks own disjoint pixel ranges.
        let out = unsafe { dst.rows(range, m.n, m.n) };
        multiply(
            Input::Rows {
                data: buf,
                stride: 4 * c,
            },
            0..rows,
            m,
            out,
            m.n,
            Act::None,
            false,
            gemm,
        );
    });
    out
}

/// The 2x2 stride-2 transposed convolution of a `height x width` map: input
/// pixel `(y, x)` gives output pixels `(2y + dy, 2x + dx)`, whose channels
/// are columns `(2 dy + dx) * cout..` of its product with `m`.
fn doubled(pool: &mut Pool, x: &[f32], height: usize, width: usize, m: &Matrix) -> Vec<f32> {
    let cout = m.n / 4;
    let w2 = 2 * width;
    let mut out = vec![0.0; 4 * height * width * cout];
    let dst = Shared::new(&mut out);
    pool.split(height * width, 96, |range, scratch| {
        let (buf, gemm) = scratch.floats(range.len() * m.n);
        multiply(
            Input::Rows {
                data: x,
                stride: m.k,
            },
            range.clone(),
            m,
            buf,
            m.n,
            Act::None,
            false,
            gemm,
        );
        for (i, r) in range.enumerate() {
            let (y, xx) = (r / width, r % width);
            for q in 0..4 {
                let pixel = (2 * y + q / 2) * w2 + 2 * xx + q % 2;
                // SAFETY: input pixels map to disjoint output pixels, and tasks
                // own disjoint input pixel ranges.
                let dst = unsafe { dst.part(pixel * cout, cout) };
                dst.copy_from_slice(&buf[i * m.n + q * cout..][..cout]);
            }
        }
    });
    out
}
