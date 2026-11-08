//! BiRefNet's decoder: from the backbone's feature maps (each the full-size
//! run concatenated with the half-size run) back to one logit per pixel.
//!
//! Every decoder block is a 3x3 convolution to 64 channels, a deformable
//! ASPP, and a 3x3 convolution out. The ASPP runs four deformable
//! convolutions (1x1, 1x1, 3x3, 7x7) and a global pooling branch, and
//! concatenates them into a 1x1 convolution. Here the pooling branch, constant
//! over the image, becomes a bias of that convolution, and each task takes a
//! chunk of pixels through all four branches and the 1x1 convolution, so the
//! four branches' 1024 channels never exist for more than a chunk.
//!
//! Each level also receives the input image cut into a grid of patches
//! stacked as channels (`image2patches` in the reference) through two 3x3
//! convolutions.

use crate::Error;
use std::ops::Range;

use crate::gemm::{Act, Input, Matrix, multiply};
use crate::math;
use crate::pool::{Pool, Shared};
use crate::weight_file::WeightFile;
use native_engine::winograd;

pub(crate) const INTER: usize = 64;
pub(crate) const BRANCH: usize = 256;
/// Kernel sides of the ASPP's deformable branches.
pub(crate) const SIZES: [usize; 4] = [1, 1, 3, 7];
/// Kernel side of every other convolution of the decoder.
const KERNEL: usize = 3;

fn conv_matrix(
    file: &mut WeightFile,
    name: &str,
    cin: usize,
    cout: usize,
) -> Result<Matrix, Error> {
    let k = KERNEL * KERNEL * cin;
    let w = file.take_weights(&format!("{name}.weight"), &[k, cout])?;
    let b = file.take(&format!("{name}.bias"), &[cout])?;
    Ok(Matrix::new(&w, k, cout, Some(&b)))
}

struct Branch {
    size: usize,
    /// Offsets (`2 * taps`, y then x per tap) and modulator logits (`taps`).
    offsets: Matrix,
    /// The deformable convolution with its batch norm folded in.
    weight: Matrix,
}

struct Aspp {
    branches: Vec<Branch>,
    pool_weight: Vec<f32>,
    pool_bias: Vec<f32>,
    /// The 1x1 convolution's rows for the four deformable branches.
    conv1: Matrix,
    /// Its rows for the pooling branch, `[256][64]`, and its bias.
    conv1_pool: Vec<f32>,
    conv1_bias: Vec<f32>,
}

struct Block {
    conv_in: Matrix,
    aspp: Aspp,
    conv_out: Matrix,
    /// The gradient attention of blocks 4 to 2: a 3x3 convolution to 16
    /// channels and a 1x1 one to a sigmoid gate on the block's output.
    gate: Option<(Matrix, Vec<f32>, f32)>,
}

struct Patches {
    conv1: Matrix,
    conv_out: Matrix,
}

pub struct Decoder {
    squeeze: Block,
    /// Blocks 4, 3, 2, 1.
    blocks: Vec<Block>,
    /// Lateral 1x1 convolutions of levels 3, 2, 1.
    laterals: Vec<Matrix>,
    /// Patch branches of levels 4 (the squeezed map) to 1.
    patches: Vec<Patches>,
    head: Matrix,
    head_bias: f32,
    full_resolution: FullResolution,
}

/// The full-resolution patch branch, `conv_out(conv1(image))` with the head
/// folded into `conv_out`: one output channel from two 3x3 convolutions and no
/// activation, which compose into one 5x5 convolution wherever the
/// intermediate map's zero padding is not reached.
pub(crate) struct FullResolution {
    /// `[(ky, kx, c)][64]` and its bias.
    pub conv1: Vec<f32>,
    pub bias1: Vec<f32>,
    /// `[(ky, kx, k)]`, one output.
    pub conv_out: Vec<f32>,
    pub bias_out: f32,
    /// `[(ky, kx, c)]` of the composed 5x5 convolution and its bias.
    pub composed: Vec<f32>,
    pub composed_bias: f32,
}

impl Aspp {
    fn load(file: &mut WeightFile, p: &str) -> Result<Self, Error> {
        let mut branches = Vec::new();
        for (i, &size) in SIZES.iter().enumerate() {
            let taps = size * size;
            let k = taps * INTER;
            let w = file.take_weights(&format!("{p}.branch{i}.offset.weight"), &[k, 3 * taps])?;
            let b = file.take(&format!("{p}.branch{i}.offset.bias"), &[3 * taps])?;
            let offsets = Matrix::new(&w, k, 3 * taps, Some(&b));
            let w = file.take_weights(&format!("{p}.branch{i}.weight"), &[k, BRANCH])?;
            let b = file.take(&format!("{p}.branch{i}.bias"), &[BRANCH])?;
            branches.push(Branch {
                size,
                offsets,
                weight: Matrix::new(&w, k, BRANCH, Some(&b)),
            });
        }
        let pool_weight = file.take(&format!("{p}.pool.weight"), &[INTER, BRANCH])?;
        let pool_bias = file.take(&format!("{p}.pool.bias"), &[BRANCH])?;
        let conv1 = file.take_weights(&format!("{p}.conv1.weight"), &[5 * BRANCH, INTER])?;
        let conv1_bias = file.take(&format!("{p}.conv1.bias"), &[INTER])?;
        Ok(Self {
            branches,
            pool_weight,
            pool_bias,
            conv1: Matrix::new(&conv1.rows(0..4 * BRANCH, INTER), 4 * BRANCH, INTER, None),
            conv1_pool: conv1.rows(4 * BRANCH..5 * BRANCH, INTER).to_f32(),
            conv1_bias,
        })
    }

    /// The deformable ASPP of a `height x width x 64` map.
    fn run(&self, pool: &mut Pool, t: &[f32], height: usize, width: usize) -> Vec<f32> {
        let pixels = height * width;
        // The pooling branch: mean, 1x1 convolution, ReLU; then its share of
        // conv1, added to every pixel like a bias.
        let mut mean = vec![0.0f64; INTER];
        for row in t.as_chunks::<INTER>().0 {
            for (m, &v) in mean.iter_mut().zip(row) {
                *m += f64::from(v);
            }
        }
        let mut pooled = self.pool_bias.clone();
        for (c, m) in mean.iter().enumerate() {
            let m = (m / pixels as f64) as f32;
            for (p, w) in pooled
                .iter_mut()
                .zip(&self.pool_weight[c * BRANCH..][..BRANCH])
            {
                *p += m * w;
            }
        }
        let mut bias = self.conv1_bias.clone();
        for (j, p) in pooled.iter().enumerate() {
            let p = p.max(0.0);
            for (b, w) in bias.iter_mut().zip(&self.conv1_pool[j * INTER..][..INTER]) {
                *b += p * w;
            }
        }

        let mut out = vec![0.0; pixels * INTER];
        let dst = Shared::new(&mut out);
        pool.split(pixels, 48, |range, scratch| {
            let (r0, rows) = (range.start, range.len());
            let max_taps = SIZES[3] * SIZES[3];
            let sizes = [
                rows * 3 * max_taps,
                rows * max_taps * INTER,
                rows * 4 * BRANCH,
            ];
            let (buf, gemm) = scratch.floats(sizes.iter().sum());
            let (offsets, rest) = buf.split_at_mut(sizes[0]);
            let (samples, cat) = rest.split_at_mut(sizes[1]);
            for (i, branch) in self.branches.iter().enumerate() {
                let taps = branch.size * branch.size;
                let input = Input::conv(t, height, width, INTER, INTER, branch.size);
                multiply(
                    input,
                    r0..r0 + rows,
                    &branch.offsets,
                    offsets,
                    3 * taps,
                    Act::None,
                    false,
                    gemm,
                );
                sample_rows(
                    t,
                    width,
                    r0,
                    branch.size,
                    &mut offsets[..rows * 3 * taps],
                    &mut samples[..rows * taps * INTER],
                );
                multiply(
                    Input::Rows {
                        data: samples,
                        stride: taps * INTER,
                    },
                    0..rows,
                    &branch.weight,
                    &mut cat[i * BRANCH..],
                    4 * BRANCH,
                    Act::Relu,
                    false,
                    gemm,
                );
            }
            // SAFETY: tasks own disjoint pixel ranges.
            let out = unsafe { dst.rows(range, INTER, INTER) };
            multiply(
                Input::Rows {
                    data: cat,
                    stride: 4 * BRANCH,
                },
                0..rows,
                &self.conv1,
                out,
                INTER,
                Act::None,
                false,
                gemm,
            );
            for row in out.as_chunks_mut::<INTER>().0 {
                for (v, b) in row.iter_mut().zip(&bias) {
                    *v = (*v + b).max(0.0);
                }
            }
        });
        out
    }
}

dispatch! {
    /// [`sample`] for the pixels from `first` on of a map `width` wide, one per
    /// `3 * taps` values of `offsets`, whose modulator logits go through the
    /// sigmoid first.
    sample_rows(t: &[f32], width: usize, first: usize, size: usize, offsets: &mut [f32], out: &mut [f32]) {
        let taps = size * size;
        let height = t.len() / (width * INTER);
        let rows = offsets.chunks_exact_mut(3 * taps).zip(out.chunks_exact_mut(taps * INTER));
        for (r, (o, row)) in rows.enumerate() {
            for m in &mut o[2 * taps..] {
                *m = crate::math::sigmoid_one(*m);
            }
            let p = first + r;
            sample(t, (height, width), (p / width, p % width), size, o, row);
        }
    }
}

/// The deformable convolution's input for pixel `(y, x)` of a
/// `height x width x 64` map `t`: for each tap of a `size x size` kernel, the
/// 64 channels bilinearly sampled at the tap moved by its offset, times twice
/// its modulator. `offsets` holds the offsets (y, x per tap) and the
/// modulators already through the sigmoid. As torchvision's `deform_conv2d`:
/// a point more than a pixel outside the map reads zero, and corners outside
/// it count as zero.
#[inline(always)]
fn sample(
    t: &[f32],
    (height, width): (usize, usize),
    (y, x): (usize, usize),
    size: usize,
    offsets: &[f32],
    out: &mut [f32],
) {
    let taps = size * size;
    let half = (size / 2) as f32;
    let (h, w) = (height as f32, width as f32);
    for tap in 0..taps {
        let dst = &mut out[tap * INTER..][..INTER];
        let py = (y as f32 - half + (tap / size) as f32) + offsets[2 * tap];
        let px = (x as f32 - half + (tap % size) as f32) + offsets[2 * tap + 1];
        let m = 2.0 * offsets[2 * taps + tap];
        if py <= -1.0 || py >= h || px <= -1.0 || px >= w {
            dst.fill(0.0);
            continue;
        }
        let (y0, x0) = (py.floor(), px.floor());
        let (ly, lx) = (py - y0, px - x0);
        let (hy, hx) = (1.0 - ly, 1.0 - lx);
        let (y0, x0) = (y0 as isize, x0 as isize);
        let corner = |yy: isize, xx: isize, weight: f32| -> Option<(&[f32], f32)> {
            (yy >= 0 && xx >= 0 && (yy as usize) < height && (xx as usize) < width).then(|| {
                (
                    &t[(yy as usize * width + xx as usize) * INTER..][..INTER],
                    weight,
                )
            })
        };
        let corners = [
            corner(y0, x0, hy * hx),
            corner(y0, x0 + 1, hy * lx),
            corner(y0 + 1, x0, ly * hx),
            corner(y0 + 1, x0 + 1, ly * lx),
        ];
        dst.fill(0.0);
        for (v, weight) in corners.into_iter().flatten() {
            for (d, s) in dst.iter_mut().zip(v) {
                *d += weight * s;
            }
        }
        for d in dst.iter_mut() {
            *d *= m;
        }
    }
}

impl Block {
    fn load(
        file: &mut WeightFile,
        p: &str,
        cin: usize,
        cout: usize,
        gate: bool,
    ) -> Result<Self, Error> {
        let conv_in = conv_matrix(file, &format!("{p}.conv_in"), cin, INTER)?;
        let conv_out = conv_matrix(file, &format!("{p}.conv_out"), INTER, cout)?;
        let gate = if gate {
            let conv = conv_matrix(file, &format!("{p}.gdt"), cout, 16)?;
            let w = file.take(&format!("{p}.gdt_attn.weight"), &[16, 1])?;
            let b = file.take(&format!("{p}.gdt_attn.bias"), &[1])?;
            Some((conv, w, b[0]))
        } else {
            None
        };
        let aspp = Aspp::load(file, &format!("{p}.aspp"))?;
        Ok(Self {
            conv_in,
            aspp,
            conv_out,
            gate,
        })
    }

    /// Run the block on `x`, which is freed as soon as it is read.
    fn run(
        &self,
        pool: &mut Pool,
        x: Vec<f32>,
        height: usize,
        width: usize,
        cin: usize,
    ) -> Vec<f32> {
        let pixels = height * width;
        let mut t = vec![0.0; pixels * INTER];
        conv(
            pool,
            &x,
            height,
            width,
            cin,
            &self.conv_in,
            &mut t,
            INTER,
            Act::Relu,
        );
        drop(x);
        let t = self.aspp.run(pool, &t, height, width);
        let cout = self.conv_out.n;
        let mut out = vec![0.0; pixels * cout];
        conv(
            pool,
            &t,
            height,
            width,
            INTER,
            &self.conv_out,
            &mut out,
            cout,
            Act::None,
        );
        if let Some((gdt, w, b)) = &self.gate {
            let mut gates = vec![0.0; pixels];
            let dst = Shared::new(&mut gates);
            pool.split(pixels, 96, |range, scratch| {
                let (g, gemm) = scratch.floats(range.len() * 16);
                let input = Input::conv(&out, height, width, cout, cout, 3);
                multiply(input, range.clone(), gdt, g, 16, Act::Relu, false, gemm);
                // SAFETY: tasks own disjoint pixel ranges.
                let gates = unsafe { dst.rows(range, 1, 1) };
                for (gate, row) in gates.iter_mut().zip(g.as_chunks::<16>().0) {
                    *gate = row.iter().zip(w).fold(*b, |s, (v, w)| s + v * w);
                }
                math::sigmoid(gates);
            });
            let dst = Shared::new(&mut out);
            pool.split(pixels, 96, |range, _| {
                let gates = &gates[range.clone()];
                // SAFETY: tasks own disjoint pixel ranges.
                let rows = unsafe { dst.rows(range, cout, cout) };
                for (row, g) in rows.chunks_exact_mut(cout).zip(gates) {
                    for v in row {
                        *v *= g;
                    }
                }
            });
        }
        out
    }
}

/// A 3x3 "same" convolution of a `height x width x cin` map into `out`, row `r`
/// at `out[r * stride..]`.
#[expect(
    clippy::too_many_arguments,
    reason = "a convolution's map, weights and output layout"
)]
fn conv(
    pool: &mut Pool,
    x: &[f32],
    height: usize,
    width: usize,
    cin: usize,
    m: &Matrix,
    out: &mut [f32],
    stride: usize,
    act: Act,
) {
    let input = Input::conv(x, height, width, cin, cin, KERNEL);
    match act {
        Act::None => winograd::conv3(pool, input, m, out, stride, false, |_| {}),
        Act::Relu => winograd::conv3(pool, input, m, out, stride, false, |pixel| {
            pixel.iter_mut().for_each(|v| *v = v.max(0.0));
        }),
        Act::Gelu => unreachable!("no 3x3 convolution of the decoder ends in a GELU"),
    }
}

/// Bilinear resize with aligned corners, as ONNX Runtime computes it, of a
/// `sh x sw x c` map into columns `0..c` of the `dh x dw` rows of `dst`,
/// `stride` apart; added to what is there when `add`.
#[expect(
    clippy::too_many_arguments,
    reason = "both maps' shapes and the output layout"
)]
pub fn resize(
    pool: &mut Pool,
    src: &[f32],
    (sh, sw): (usize, usize),
    c: usize,
    dst: &mut [f32],
    (dh, dw): (usize, usize),
    stride: usize,
    add: bool,
) {
    let grid = Grid {
        sh,
        sw,
        dh,
        dw,
        c,
        stride,
    };
    let out = Shared::new(dst);
    pool.split(dh, dh, |range, _| {
        let len = (range.len() - 1) * dw * stride + (dw - 1) * stride + c;
        // SAFETY: tasks own disjoint output rows.
        let rows = unsafe { out.part(range.start * dw * stride, len) };
        resize_rows(src, grid, range, rows, add);
    });
}

/// The shape of a [`resize`].
#[derive(Clone, Copy)]
pub struct Grid {
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
    c: usize,
    stride: usize,
}

/// Source pixels and weights of output coordinate `out` along one axis, as
/// ONNX Runtime computes them for aligned corners.
#[inline(always)]
fn axis(out: usize, len_in: usize, len_out: usize) -> (usize, usize, f32, f32) {
    let pos = if len_out == 1 {
        0.0
    } else {
        out as f32 * (len_in - 1) as f32 / (len_out - 1) as f32
    };
    let pos = pos.clamp(0.0, (len_in - 1) as f32);
    let i0 = pos as usize;
    let i1 = (i0 + 1).min(len_in - 1);
    if i0 == i1 {
        (i0, i1, 0.5, 0.5)
    } else {
        (i0, i1, (pos - i0 as f32).abs(), (pos - i1 as f32).abs())
    }
}

dispatch! {
    /// Output rows `rows` of a [`resize`], into `dst` starting at the first.
    resize_rows(src: &[f32], grid: Grid, rows: Range<usize>, dst: &mut [f32], add: bool) {
        let Grid { sh, sw, dh, dw, c, stride } = grid;
        let first = rows.start;
        for y in rows {
            let (y1, y2, dy1, dy2) = axis(y, sh, dh);
            let row = &mut dst[(y - first) * dw * stride..];
            for x in 0..dw {
                let (x1, x2, dx1, dx2) = axis(x, sw, dw);
                let p11 = &src[(y1 * sw + x1) * c..][..c];
                let p21 = &src[(y1 * sw + x2) * c..][..c];
                let p12 = &src[(y2 * sw + x1) * c..][..c];
                let p22 = &src[(y2 * sw + x2) * c..][..c];
                let d = &mut row[x * stride..][..c];
                for j in 0..c {
                    let v = dx2 * dy2 * p11[j] + dx1 * dy2 * p21[j] + dx2 * dy1 * p12[j] + dx1 * dy1 * p22[j];
                    d[j] = if add { d[j] + v } else { v };
                }
            }
        }
    }
}

/// Copy a `pixels x c` map into columns `offset..offset + c` of rows `stride`
/// apart.
fn place(pool: &mut Pool, src: &[f32], c: usize, dst: &mut [f32], stride: usize, offset: usize) {
    let dst = Shared::new(&mut dst[offset..]);
    pool.split(src.len() / c, 96, |range, _| {
        let src = &src[range.start * c..range.end * c];
        // SAFETY: tasks own disjoint pixel ranges.
        let rows = unsafe { dst.rows(range, stride, c) };
        for (s, d) in src.chunks_exact(c).zip(rows.chunks_mut(stride)) {
            d[..c].copy_from_slice(s);
        }
    });
}

impl Patches {
    fn load(file: &mut WeightFile, p: &str, cin: usize, cout: usize) -> Result<Self, Error> {
        Ok(Self {
            conv1: conv_matrix(file, &format!("{p}.conv1"), cin, INTER)?,
            conv_out: conv_matrix(file, &format!("{p}.conv_out"), INTER, cout)?,
        })
    }

    /// The image cut into `grid x grid` patches of `size x size` pixels, patch
    /// `(gy, gx)` of colour `c` becoming channel `(c * grid + gy) * grid + gx`,
    /// through both convolutions into `out`, `stride` apart.
    fn run(
        &self,
        pool: &mut Pool,
        image: &[f32],
        side: usize,
        size: usize,
        out: &mut [f32],
        stride: usize,
    ) {
        let grid = side / size;
        let channels = 3 * grid * grid;
        let mut stacked = vec![0.0; size * size * channels];
        let dst = Shared::new(&mut stacked);
        pool.split(size * size, 96, |range, _| {
            // SAFETY: tasks own disjoint pixel ranges.
            let rows = unsafe { dst.rows(range.clone(), channels, channels) };
            for (p, row) in range.zip(rows.chunks_exact_mut(channels)) {
                let (y, x) = (p / size, p % size);
                let mut values = row.iter_mut();
                for c in 0..3 {
                    for gy in 0..grid {
                        for gx in 0..grid {
                            let v = values.next().expect("row holds every channel");
                            *v = image[((gy * size + y) * side + gx * size + x) * 3 + c];
                        }
                    }
                }
            }
        });
        let mut mid = vec![0.0; size * size * INTER];
        conv(
            pool,
            &stacked,
            size,
            size,
            channels,
            &self.conv1,
            &mut mid,
            INTER,
            Act::None,
        );
        conv(
            pool,
            &mid,
            size,
            size,
            INTER,
            &self.conv_out,
            out,
            stride,
            Act::None,
        );
    }
}

impl FullResolution {
    pub(crate) fn load(file: &mut WeightFile) -> Result<Self, Error> {
        let conv1 = file.take("patches1.conv1.weight", &[27, INTER])?;
        let bias1 = file.take("patches1.conv1.bias", &[INTER])?;
        let conv_out = file.take("patches1.conv_out.weight", &[9 * INTER, 1])?;
        let bias_out = file.take("patches1.conv_out.bias", &[1])?[0];
        let mut composed = vec![0.0f64; 25 * 3];
        let mut composed_bias = f64::from(bias_out);
        for d in 0..9 {
            for k in 0..INTER {
                let w2 = f64::from(conv_out[d * INTER + k]);
                composed_bias += w2 * f64::from(bias1[k]);
                for e in 0..9 {
                    let (ey, ex) = (d / 3 + e / 3, d % 3 + e % 3);
                    for c in 0..3 {
                        composed[(ey * 5 + ex) * 3 + c] +=
                            w2 * f64::from(conv1[(e * 3 + c) * INTER + k]);
                    }
                }
            }
        }
        Ok(Self {
            conv1,
            bias1,
            conv_out,
            bias_out,
            composed: composed.into_iter().map(|v| v as f32).collect(),
            composed_bias: composed_bias as f32,
        })
    }

    /// Add the branch's output for every pixel of the `side x side` image to
    /// `out`.
    fn add(&self, pool: &mut Pool, image: &[f32], side: usize, out: &mut [f32]) {
        // The image as three planes with two zero pixels around, so that each
        // tap of the composed convolution reads a contiguous run of a row.
        let width = side + 4;
        let mut planes = vec![0.0; 3 * width * width];
        let dst = Shared::new(&mut planes);
        pool.split(side, side, |range, _| {
            for y in range {
                for c in 0..3 {
                    // SAFETY: tasks own disjoint rows of each plane.
                    let row = unsafe { dst.part((c * width + y + 2) * width + 2, side) };
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = image[(y * side + x) * 3 + c];
                    }
                }
            }
        });
        let pixel = |y: isize, x: isize| -> Option<&[f32]> {
            (y >= 0 && x >= 0 && (y as usize) < side && (x as usize) < side)
                .then(|| &image[(y as usize * side + x as usize) * 3..][..3])
        };
        let dst = Shared::new(out);
        pool.split(side, side, |range, _| {
            for y in range {
                // SAFETY: tasks own disjoint output rows.
                let row = unsafe { dst.part(y * side, side) };
                let yi = y as isize;
                if y == 0 || y + 1 == side {
                    for (x, o) in row.iter_mut().enumerate() {
                        *o += self.exact(yi, x as isize, &pixel);
                    }
                    continue;
                }
                row[0] += self.exact(yi, 0, &pixel);
                row[side - 1] += self.exact(yi, side as isize - 1, &pixel);
                composed_row(
                    &planes,
                    width,
                    y,
                    &self.composed,
                    self.composed_bias,
                    &mut row[1..side - 1],
                );
            }
        });
    }

    /// The two convolutions one after the other, for edge pixels, where the
    /// intermediate map's zero padding is reached.
    fn exact<'a>(
        &self,
        y: isize,
        x: isize,
        pixel: &impl Fn(isize, isize) -> Option<&'a [f32]>,
    ) -> f32 {
        let mut s = self.bias_out;
        for d in 0..9 {
            let (qy, qx) = (y + (d / 3) as isize - 1, x + (d % 3) as isize - 1);
            if pixel(qy, qx).is_none() {
                continue;
            }
            let mut mid = self.bias1.clone();
            for e in 0..9 {
                if let Some(p) = pixel(qy + (e / 3) as isize - 1, qx + (e % 3) as isize - 1) {
                    for (c, &v) in p.iter().enumerate() {
                        let w = &self.conv1[(e * 3 + c) * INTER..][..INTER];
                        for (m, w) in mid.iter_mut().zip(w) {
                            *m += v * w;
                        }
                    }
                }
            }
            s += mid
                .iter()
                .zip(&self.conv_out[d * INTER..][..INTER])
                .map(|(m, w)| m * w)
                .sum::<f32>();
        }
        s
    }
}

dispatch! {
    /// Add the composed 5x5 convolution to the interior pixels `1..` of row `y`
    /// (`out`), reading the padded planes `width` wide: per pixel the bias,
    /// then each tap's three products in order.
    composed_row(planes: &[f32], width: usize, y: usize, weights: &[f32], bias: f32, out: &mut [f32]) {
        let plane = width * width;
        let mut sums = [0.0f32; 256];
        let mut x0 = 0;
        while x0 < out.len() {
            let n = (out.len() - x0).min(sums.len());
            let s = &mut sums[..n];
            s.fill(bias);
            for e in 0..25 {
                let at = (y + e / 5) * width + 1 + e % 5 + x0;
                let (w0, w1, w2) = (weights[e * 3], weights[e * 3 + 1], weights[e * 3 + 2]);
                let p0 = &planes[at..][..n];
                let p1 = &planes[plane + at..][..n];
                let p2 = &planes[2 * plane + at..][..n];
                for k in 0..n {
                    s[k] += w0 * p0[k] + w1 * p1[k] + w2 * p2[k];
                }
            }
            for (o, v) in out[x0..x0 + n].iter_mut().zip(s.iter()) {
                *o += v;
            }
            x0 += n;
        }
    }
}

/// Input and output channels of decoder blocks 4 to 1.
pub(crate) const LEVELS: [(usize, usize); 4] = [(1728, 768), (960, 384), (480, 192), (240, 96)];

impl Decoder {
    pub fn load(file: &mut WeightFile) -> Result<Self, Error> {
        let squeeze = Block::load(file, "squeeze", 2880, 1536, false)?;
        let mut blocks = Vec::new();
        for (i, &(cin, cout)) in LEVELS.iter().enumerate() {
            blocks.push(Block::load(
                file,
                &format!("block{}", 4 - i),
                cin,
                cout,
                i < 3,
            )?);
        }
        let mut laterals = Vec::new();
        for (n, c) in [(4, 768), (3, 384), (2, 192)] {
            let w = file.take_weights(&format!("lateral{n}.weight"), &[c, c])?;
            let b = file.take(&format!("lateral{n}.bias"), &[c])?;
            laterals.push(Matrix::new(&w, c, c, Some(&b)));
        }
        let mut patches = Vec::new();
        for (n, cin, cout) in [(5, 3072, 192), (4, 768, 192), (3, 192, 96), (2, 48, 48)] {
            patches.push(Patches::load(file, &format!("patches{n}"), cin, cout)?);
        }
        let head_weight = file.take_weights("head.weight", &[96, 1])?;
        let head_bias = file.take("head.bias", &[1])?[0];
        Ok(Self {
            squeeze,
            blocks,
            laterals,
            patches,
            head: Matrix::new(&head_weight, 96, 1, None),
            head_bias,
            full_resolution: FullResolution::load(file)?,
        })
    }

    /// Logits for the `side x side` normalised image, given the backbone's
    /// outputs for it (`full`) and for its half-size copy (`half`).
    pub fn run(
        &self,
        pool: &mut Pool,
        image: &[f32],
        side: usize,
        full: Vec<Vec<f32>>,
        half: Vec<Vec<f32>>,
    ) -> Vec<f32> {
        // x1..x4: each level's full-size features, then the half-size ones
        // upsampled to the same grid.
        let mut levels = Vec::with_capacity(4);
        for (i, (f, h)) in full.into_iter().zip(half).enumerate() {
            let grid = side >> (i + 2);
            let c = f.len() / (grid * grid);
            let mut x = vec![0.0; grid * grid * 2 * c];
            place(pool, &f, c, &mut x, 2 * c, 0);
            resize(
                pool,
                &h,
                (grid / 2, grid / 2),
                c,
                &mut x[c..],
                (grid, grid),
                2 * c,
                false,
            );
            levels.push(Some((x, grid, 2 * c)));
        }
        // The squeeze block's input: x1, x2, x3 downsampled to x4's grid, then x4.
        let g4 = side >> 5;
        let mut x4 = vec![0.0; g4 * g4 * 2880];
        let mut offset = 0;
        for (x, grid, c) in levels[..3].iter().flatten() {
            resize(
                pool,
                x,
                (*grid, *grid),
                *c,
                &mut x4[offset..],
                (g4, g4),
                2880,
                false,
            );
            offset += c;
        }
        if let Some((x, _, c)) = levels[3].take() {
            place(pool, &x, c, &mut x4, 2880, offset);
        }
        let squeezed = self.squeeze.run(pool, x4, g4, g4, 2880);

        // Each level: the previous output (or the squeezed map) plus, from level
        // 3 on, a lateral convolution of that level's features, with the patch
        // branch appended, through the level's block.
        let mut previous = squeezed;
        let mut previous_grid = g4;
        for (i, block) in self.blocks.iter().enumerate() {
            let level = 3 - i;
            let cin = LEVELS[i].0;
            let grid = side >> (level + 2);
            let mut input = vec![0.0; grid * grid * cin];
            // The squeezed map, or the previous block's output.
            let main_c = if i == 0 { 1536 } else { LEVELS[i - 1].1 };
            if i == 0 {
                place(pool, &previous, main_c, &mut input, cin, 0);
            } else {
                let (x, _, c) = levels[level].take().expect("each level is read once");
                let lateral = &self.laterals[i - 1];
                let dst = Shared::new(&mut input);
                pool.split(grid * grid, 96, |range, scratch| {
                    // SAFETY: tasks own disjoint pixel ranges.
                    let out = unsafe { dst.rows(range.clone(), cin, main_c) };
                    multiply(
                        Input::Rows {
                            data: &x,
                            stride: c,
                        },
                        range,
                        lateral,
                        out,
                        cin,
                        Act::None,
                        false,
                        &mut scratch.gemm,
                    );
                });
                resize(
                    pool,
                    &previous,
                    (previous_grid, previous_grid),
                    main_c,
                    &mut input,
                    (grid, grid),
                    cin,
                    true,
                );
            }
            self.patches[i].run(pool, image, side, grid, &mut input[main_c..], cin);
            previous = block.run(pool, input, grid, grid, cin);
            previous_grid = grid;
        }

        // The head: its decoder channels before the final upsampling, then
        // the full-resolution patch branch.
        let pixels = previous_grid * previous_grid;
        let mut low = vec![0.0; pixels];
        let dst = Shared::new(&mut low);
        pool.split(pixels, 96, |range, scratch| {
            // SAFETY: tasks own disjoint pixel ranges.
            let out = unsafe { dst.rows(range.clone(), 1, 1) };
            multiply(
                Input::Rows {
                    data: &previous,
                    stride: 96,
                },
                range,
                &self.head,
                out,
                1,
                Act::None,
                false,
                &mut scratch.gemm,
            );
        });
        let mut logits = vec![self.head_bias; side * side];
        resize(
            pool,
            &low,
            (previous_grid, previous_grid),
            1,
            &mut logits,
            (side, side),
            1,
            true,
        );
        self.full_resolution.add(pool, image, side, &mut logits);
        logits
    }
}
