//! 3x3 stride-1 convolutions by Winograd's minimal filtering (Lavin and
//! Gray, "Fast Algorithms for Convolutional Neural Networks", 2016), in its
//! F(2x2, 3x3) and F(4x4, 3x3) forms.
//!
//! An `n x n` block of output pixels reads the `(n + 2) x (n + 2)` input
//! pixels around it. With `d` that block in one input channel and `g` the
//! 3x3 weights from that channel to one output channel, the block's outputs
//! are `A^T [(G g G^T) * (B^T d B)] A`, `*` taken element by element. Summed
//! over input channels, each element is a matrix product, blocks x cin
//! transformed inputs by cin x cout transformed weights, which [`multiply`]
//! computes. Per 16 output pixels the convolution takes 144 products,
//! F(2x2, 3x3) 64 and F(4x4, 3x3) 36:
//!
//! ```text
//! F(2x2, 3x3)                                          F(4x4, 3x3)
//! B^T = | 1  0 -1  0 |  G = | 1    0    0   |          B^T = | 4  0 -5  0  1  0 |
//!       | 0  1  1  0 |      | 1/2  1/2  1/2 |                | 0 -4 -4  1  1  0 |
//!       | 0 -1  1  0 |      | 1/2 -1/2  1/2 |                | 0  4 -4 -1  1  0 |
//!       | 0  1  0 -1 |      | 0    0    1   |                | 0 -2 -1  2  1  0 |
//! A^T = | 1  1  1  0 |                                       | 0  2 -1 -2  1  0 |
//!       | 0  1 -1 -1 |                                       | 0  4  0 -5  0  1 |
//!
//! G = |  1/4    0     0   |   A^T = | 1  1  1  1  1  0 |
//!     | -1/6  -1/6  -1/6  |         | 0  1 -1  2 -2  0 |
//!     | -1/6   1/6  -1/6  |         | 0  1  1  4  4  0 |
//!     |  1/24  1/12  1/6  |         | 0  1 -1  8 -8  1 |
//!     |  1/24 -1/12  1/6  |
//!     |  0     0     1    |
//! ```
//!
//! The larger form transforms more per block, and its blocks' inputs and
//! products outgrow the cache sooner: [`conv3`] takes it while a block's
//! channels in and out total at most [`F4_CHANNELS`]. In F(2x2, 3x3) each
//! output's rounding depends on its own 3x3 neighbourhood alone; in
//! F(4x4, 3x3) the transforms mix in values up to two pixels further, which
//! cancel exactly but not in rounding, so a map cut into parts gives its
//! pixels to within rounding rather than bit for bit.
//!
//! The weights are transformed at each call from the layer's packed matrix,
//! which costs one pass over them and keeps no second copy in memory; from
//! half-float weights the transformed ones are half floats too, which halves
//! the bytes the products stream at one more rounding per weight. The
//! blocks are split between threads; each task transforms its blocks' inputs
//! and keeps their products in its scratch, small enough to stay in cache.
//! The transforms add and multiply in the same order on every tier, without
//! fused multiply-adds, so tiers 1 and 0 give the same bits.

use crate::gemm::{Act, Input, Matrix, NR, Pad, Taps, multiply, reflect};
use crate::pool::{Pool, Shared};

/// Input channels below which the transforms cost more than the products
/// they save.
pub const MIN_CHANNELS: usize = 16;
/// Input plus output channels up to which F(4x4, 3x3) is the faster form:
/// measured on Zen 3 from 64 + 96 to 3072 + 64 channels.
pub const F4_CHANNELS: usize = 576;
/// Floats of a task's transformed inputs and products: half of a 512 KiB L2.
const TASK_FLOATS: usize = 1 << 16;

/// One form: its block side, its transformed elements, and its transforms of
/// the weights (one input channel, 16 output channels), of one block's
/// input pixels and of its products. [`conv3`] runs on either, compiled for
/// each.
trait Form {
    const BLOCK: usize;
    const POINTS: usize;
    fn weight(g: &[[f32; NR]; 9], v: &mut [[f32; NR]]);
    fn input(d: &[&[f32]], u: &mut [f32], cin: usize);
    fn output(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize);
}

struct F2;
struct F4;

impl Form for F2 {
    const BLOCK: usize = f2::BLOCK;
    const POINTS: usize = f2::POINTS;
    fn weight(g: &[[f32; NR]; 9], v: &mut [[f32; NR]]) {
        f2::transform_weight(g, v.try_into().expect("16 points"));
    }
    fn input(d: &[&[f32]], u: &mut [f32], cin: usize) {
        f2::transform_input(d.try_into().expect("16 pixels"), u, cin);
    }
    fn output(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize) {
        f2::transform_output(m, bias, y, cout);
    }
}

impl Form for F4 {
    const BLOCK: usize = f4::BLOCK;
    const POINTS: usize = f4::POINTS;
    fn weight(g: &[[f32; NR]; 9], v: &mut [[f32; NR]]) {
        f4::transform_weight(g, v.try_into().expect("36 points"));
    }
    fn input(d: &[&[f32]], u: &mut [f32], cin: usize) {
        f4::transform_input(d.try_into().expect("36 pixels"), u, cin);
    }
    fn output(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize) {
        f4::transform_output(m, bias, y, cout);
    }
}

/// The transformed weight matrices, `G g G^T` for every channel pair,
/// without the bias, into `points`; the pool's threads take a share of the
/// panels each. Half-float weights give half-float matrices, which the
/// half-float kernel multiplies with half the bytes.
fn transform_weights<F: Form>(pool: &mut Pool, w: &Matrix, cin: usize, points: &mut [Matrix]) {
    let half = w.is_half();
    for point in points.iter_mut() {
        if half {
            point.resize_half(cin, w.n);
        } else {
            point.resize(cin, w.n);
        }
    }
    let (floats, halves): (Vec<Shared<'_>>, Vec<Shared<'_, u16>>) = if half {
        let halves = points.iter_mut().map(|m| Shared::new(m.halves_mut()));
        (Vec::new(), halves.collect())
    } else {
        let floats = points.iter_mut().map(|m| Shared::new(m.floats_mut()));
        (floats.collect(), Vec::new())
    };
    let panels = w.n.div_ceil(NR);
    pool.split_by(panels, 1, panels, |range, _| {
        let mut v = vec![[0.0; NR]; F::POINTS];
        for p in range {
            for c in 0..cin {
                let g: [[f32; NR]; 9] = std::array::from_fn(|tap| w.panel_floats(p, tap * cin + c));
                F::weight(&g, &mut v);
                let at = (p * cin + c) * NR;
                if half {
                    for (part, values) in halves.iter().zip(&v) {
                        // SAFETY: row c of panel p, and each task has its
                        // own panels.
                        narrow(values, unsafe { part.part(at, NR) });
                    }
                } else {
                    for (part, values) in floats.iter().zip(&v) {
                        // SAFETY: as above.
                        unsafe { part.part(at, NR) }.copy_from_slice(values);
                    }
                }
            }
        }
    });
}

/// `values` as the nearest half floats.
fn narrow(values: &[f32; NR], out: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd_tier() >= crate::TIER_AVX2 {
        // SAFETY: tier 2's CPUs have F16C.
        unsafe { narrow_f16c(values, out) };
        return;
    }
    for (o, &v) in out.iter_mut().zip(values) {
        *o = crate::half::f32_to_half(v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
fn narrow_f16c(values: &[f32; NR], out: &mut [u16]) {
    use std::arch::x86_64::{
        __m128i, _MM_FROUND_TO_NEAREST_INT, _mm_storeu_si128, _mm256_cvtps_ph, _mm256_loadu_ps,
    };
    assert!(out.len() >= NR);
    for h in 0..NR / 8 {
        // SAFETY: eight floats of `values` and eight halves of `out` (asserted).
        unsafe {
            let x = _mm256_loadu_ps(values.as_ptr().add(8 * h));
            let y = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x);
            _mm_storeu_si128(out.as_mut_ptr().add(8 * h).cast::<__m128i>(), y);
        }
    }
}

/// F(2x2, 3x3).
mod f2 {
    use super::NR;

    pub const BLOCK: usize = 2;
    pub const POINTS: usize = 16;

    /// `G g G^T` for one input channel and 16 output channels: `g` holds the
    /// 3x3 taps row by row.
    pub(super) fn transform_weight(g: &[[f32; NR]; 9], v: &mut [[f32; NR]; POINTS]) {
        // t = G g: four rows of three taps.
        let mut t = [[[0.0; NR]; 3]; 4];
        for x in 0..3 {
            for o in 0..NR {
                let (g0, g1, g2) = (g[x][o], g[3 + x][o], g[6 + x][o]);
                t[0][x][o] = g0;
                t[1][x][o] = 0.5 * (g0 + g1 + g2);
                t[2][x][o] = 0.5 * (g0 - g1 + g2);
                t[3][x][o] = g2;
            }
        }
        // v = t G^T.
        for (y, row) in t.iter().enumerate() {
            for o in 0..NR {
                let (t0, t1, t2) = (row[0][o], row[1][o], row[2][o]);
                v[4 * y][o] = t0;
                v[4 * y + 1][o] = 0.5 * (t0 + t1 + t2);
                v[4 * y + 2][o] = 0.5 * (t0 - t1 + t2);
                v[4 * y + 3][o] = t2;
            }
        }
    }

    /// `B^T d B` of one block: `d` holds its 16 input pixels row by row, and
    /// `u` receives its 16 transformed elements, `cin` floats each, one after
    /// the other. Computed as `B^T (d B)`, each row of `d B` from one row of
    /// pixels, in the same order on every tier.
    pub(super) fn transform_input(d: &[&[f32]; 16], u: &mut [f32], cin: usize) {
        assert!(d.iter().all(|p| p.len() >= cin) && u.len() >= POINTS * cin);
        #[allow(unused_mut, reason = "only x86-64 has a vector path")]
        let mut start = 0;
        #[cfg(target_arch = "x86_64")]
        if crate::simd_tier() >= crate::TIER_AVX {
            // SAFETY: the CPU has AVX; every pixel holds `cin` floats and `u`
            // 16 runs of them (asserted above).
            start = unsafe { input_avx(d, u, cin) };
        }
        for c in start..cin {
            let row = |a: usize| {
                let v = |x: usize| d[4 * a + x][c];
                [v(0) - v(2), v(1) + v(2), v(2) - v(1), v(1) - v(3)]
            };
            let (q0, q1, q2, q3) = (row(0), row(1), row(2), row(3));
            for x in 0..4 {
                u[x * cin + c] = q0[x] - q2[x];
                u[(4 + x) * cin + c] = q1[x] + q2[x];
                u[(8 + x) * cin + c] = q2[x] - q1[x];
                u[(12 + x) * cin + c] = q1[x] - q3[x];
            }
        }
    }

    /// [`transform_input`] eight channels at a time, as far as whole groups of
    /// eight go; returns the channels done.
    ///
    /// # Safety
    /// The CPU must have AVX, every pixel of `d` hold `cin` floats and `u` 16
    /// `cin`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn input_avx(d: &[&[f32]; 16], u: &mut [f32], cin: usize) -> usize {
        use std::arch::x86_64::{
            __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps,
        };
        let pixels = d.map(<[f32]>::as_ptr);
        let out = u.as_mut_ptr();
        let mut c = 0;
        while c + 8 <= cin {
            // SAFETY: c + 8 <= cin, within every pixel and every run of `u`.
            unsafe {
                let row = |a: usize| -> [__m256; 4] {
                    let v = |x: usize| _mm256_loadu_ps(pixels[4 * a + x].add(c));
                    let (v0, v1, v2, v3) = (v(0), v(1), v(2), v(3));
                    [
                        _mm256_sub_ps(v0, v2),
                        _mm256_add_ps(v1, v2),
                        _mm256_sub_ps(v2, v1),
                        _mm256_sub_ps(v1, v3),
                    ]
                };
                let store = |point: usize, v: __m256| _mm256_storeu_ps(out.add(point * cin + c), v);
                let (q1, q2) = (row(1), row(2));
                let q0 = row(0);
                for x in 0..4 {
                    store(x, _mm256_sub_ps(q0[x], q2[x]));
                    store(4 + x, _mm256_add_ps(q1[x], q2[x]));
                    store(8 + x, _mm256_sub_ps(q2[x], q1[x]));
                }
                let q3 = row(3);
                for x in 0..4 {
                    store(12 + x, _mm256_sub_ps(q1[x], q3[x]));
                }
            }
            c += 8;
        }
        c
    }

    /// `A^T m A` plus `bias` of one block: `m` holds its 16 products, `cout`
    /// floats each, one after the other, and `y` receives its four output
    /// pixels row by row. Computed as `A^T (m A)`, in the same order on every
    /// tier.
    pub(super) fn transform_output(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize) {
        assert!(m.len() >= POINTS * cout && bias.len() >= cout && y.len() >= 4 * cout);
        #[allow(unused_mut, reason = "only x86-64 has a vector path")]
        let mut start = 0;
        #[cfg(target_arch = "x86_64")]
        if crate::simd_tier() >= crate::TIER_AVX {
            // SAFETY: the CPU has AVX, and the lengths are asserted above.
            start = unsafe { output_avx(m, bias, y, cout) };
        }
        for c in start..cout {
            let row = |i: usize| {
                let v = |x: usize| m[(4 * i + x) * cout + c];
                [v(0) + v(1) + v(2), v(1) - v(2) - v(3)]
            };
            let (s0, s1, s2, s3) = (row(0), row(1), row(2), row(3));
            for x in 0..2 {
                y[x * cout + c] = s0[x] + s1[x] + s2[x] + bias[c];
                y[(2 + x) * cout + c] = s1[x] - s2[x] - s3[x] + bias[c];
            }
        }
    }

    /// [`transform_output`] eight channels at a time, as far as whole groups of
    /// eight go; returns the channels done.
    ///
    /// # Safety
    /// The CPU must have AVX, `m` hold 16 `cout` floats, `bias` `cout` and `y`
    /// 4 `cout`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn output_avx(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize) -> usize {
        use std::arch::x86_64::{
            __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps,
        };
        let (m, bias, out) = (m.as_ptr(), bias.as_ptr(), y.as_mut_ptr());
        let mut c = 0;
        while c + 8 <= cout {
            // SAFETY: c + 8 <= cout, within every run of `m` and `y` and within
            // `bias`.
            unsafe {
                let row = |i: usize| -> [__m256; 2] {
                    let v = |x: usize| _mm256_loadu_ps(m.add((4 * i + x) * cout + c));
                    let (v0, v1, v2, v3) = (v(0), v(1), v(2), v(3));
                    [
                        _mm256_add_ps(_mm256_add_ps(v0, v1), v2),
                        _mm256_sub_ps(_mm256_sub_ps(v1, v2), v3),
                    ]
                };
                let (s0, s1, s2, s3) = (row(0), row(1), row(2), row(3));
                let b = _mm256_loadu_ps(bias.add(c));
                for x in 0..2 {
                    let top = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s0[x], s1[x]), s2[x]), b);
                    let bottom =
                        _mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(s1[x], s2[x]), s3[x]), b);
                    _mm256_storeu_ps(out.add(x * cout + c), top);
                    _mm256_storeu_ps(out.add((2 + x) * cout + c), bottom);
                }
            }
            c += 8;
        }
        c
    }
}

/// F(4x4, 3x3).
mod f4 {
    use super::NR;

    pub const BLOCK: usize = 4;
    const SIDE: usize = BLOCK + 2;
    pub const POINTS: usize = SIDE * SIDE;

    /// `G (a, b, c)`: three taps to six.
    fn g_column(a: f32, b: f32, c: f32) -> [f32; SIDE] {
        [
            a / 4.0,
            -(a + b + c) / 6.0,
            -(a - b + c) / 6.0,
            a / 24.0 + b / 12.0 + c / 6.0,
            a / 24.0 - b / 12.0 + c / 6.0,
            c,
        ]
    }

    /// `G g G^T` for one input channel and 16 output channels: `g` holds the
    /// 3x3 taps row by row.
    pub(super) fn transform_weight(g: &[[f32; NR]; 9], v: &mut [[f32; NR]; POINTS]) {
        for o in 0..NR {
            // t = G g: six rows of three taps.
            let mut t = [[0.0; 3]; SIDE];
            for x in 0..3 {
                let column = g_column(g[x][o], g[3 + x][o], g[6 + x][o]);
                for (row, value) in t.iter_mut().zip(column) {
                    row[x] = value;
                }
            }
            // v = t G^T.
            for (y, row) in t.iter().enumerate() {
                for (x, value) in g_column(row[0], row[1], row[2]).into_iter().enumerate() {
                    v[SIDE * y + x][o] = value;
                }
            }
        }
    }

    /// `B^T v` for six values: one column (or row) of a block.
    fn b_t(v: [f32; SIDE]) -> [f32; SIDE] {
        let a = v[4] - 4.0 * v[2];
        let b = v[3] - 4.0 * v[1];
        let c = v[4] - v[2];
        let e = 2.0 * (v[3] - v[1]);
        [
            4.0 * v[0] - 5.0 * v[2] + v[4],
            a + b,
            a - b,
            c + e,
            c - e,
            4.0 * v[1] - 5.0 * v[3] + v[5],
        ]
    }

    /// `A^T v` plus `bias` for six values.
    fn a_t(v: [f32; SIDE], bias: f32) -> [f32; BLOCK] {
        let s = v[1] + v[2];
        let d = v[1] - v[2];
        let s2 = v[3] + v[4];
        let d2 = v[3] - v[4];
        [
            v[0] + s + s2 + bias,
            d + 2.0 * d2 + bias,
            s + 4.0 * s2 + bias,
            d + 8.0 * d2 + v[5] + bias,
        ]
    }

    /// `B^T d B` of one block: `d` holds its 36 input pixels row by row, and
    /// `u` receives its 36 transformed elements, `cin` floats each, one after
    /// the other. Computed as `B^T (d B)`, in the same order on every tier.
    pub(super) fn transform_input(d: &[&[f32]; POINTS], u: &mut [f32], cin: usize) {
        assert!(d.iter().all(|p| p.len() >= cin) && u.len() >= POINTS * cin);
        #[allow(unused_mut, reason = "only x86-64 has a vector path")]
        let mut start = 0;
        #[cfg(target_arch = "x86_64")]
        if crate::simd_tier() >= crate::TIER_AVX {
            // SAFETY: the CPU has AVX; every pixel holds `cin` floats and `u`
            // 36 runs of them (asserted above).
            start = unsafe { input_avx(d, u, cin) };
        }
        for c in start..cin {
            let rows: [[f32; SIDE]; SIDE] =
                std::array::from_fn(|y| b_t(std::array::from_fn(|x| d[SIDE * y + x][c])));
            for x in 0..SIDE {
                let column = b_t(std::array::from_fn(|y| rows[y][x]));
                for (y, value) in column.into_iter().enumerate() {
                    u[(SIDE * y + x) * cin + c] = value;
                }
            }
        }
    }

    /// [`transform_input`] eight channels at a time, as far as whole groups of
    /// eight go; returns the channels done.
    ///
    /// # Safety
    /// The CPU must have AVX, every pixel of `d` hold `cin` floats and `u` 36
    /// `cin`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn input_avx(d: &[&[f32]; POINTS], u: &mut [f32], cin: usize) -> usize {
        use std::arch::x86_64::{
            __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
            _mm256_storeu_ps, _mm256_sub_ps,
        };
        let pixels = d.map(<[f32]>::as_ptr);
        let out = u.as_mut_ptr();
        let b_t = |v: [__m256; SIDE]| -> [__m256; SIDE] {
            let (k2, k4, k5) = (
                _mm256_set1_ps(2.0),
                _mm256_set1_ps(4.0),
                _mm256_set1_ps(5.0),
            );
            let a = _mm256_sub_ps(v[4], _mm256_mul_ps(k4, v[2]));
            let b = _mm256_sub_ps(v[3], _mm256_mul_ps(k4, v[1]));
            let c = _mm256_sub_ps(v[4], v[2]);
            let e = _mm256_mul_ps(k2, _mm256_sub_ps(v[3], v[1]));
            [
                _mm256_add_ps(
                    _mm256_sub_ps(_mm256_mul_ps(k4, v[0]), _mm256_mul_ps(k5, v[2])),
                    v[4],
                ),
                _mm256_add_ps(a, b),
                _mm256_sub_ps(a, b),
                _mm256_add_ps(c, e),
                _mm256_sub_ps(c, e),
                _mm256_add_ps(
                    _mm256_sub_ps(_mm256_mul_ps(k4, v[1]), _mm256_mul_ps(k5, v[3])),
                    v[5],
                ),
            ]
        };
        let end = cin - cin % 8;
        // Two passes through `u`, which stays in L1: 36 vectors and 36 pixel
        // pointers at once would not fit in the registers.
        for y in 0..SIDE {
            let row = &pixels[SIDE * y..][..SIDE];
            for c in (0..end).step_by(8) {
                // SAFETY: c + 8 <= cin, within every pixel and every run of `u`.
                unsafe {
                    let t = b_t(std::array::from_fn(|x| _mm256_loadu_ps(row[x].add(c))));
                    for (x, value) in t.into_iter().enumerate() {
                        _mm256_storeu_ps(out.add((SIDE * y + x) * cin + c), value);
                    }
                }
            }
        }
        for x in 0..SIDE {
            for c in (0..end).step_by(8) {
                // SAFETY: as above.
                unsafe {
                    let at = |y: usize| out.add((SIDE * y + x) * cin + c);
                    let column = b_t(std::array::from_fn(|y| _mm256_loadu_ps(at(y))));
                    for (y, value) in column.into_iter().enumerate() {
                        _mm256_storeu_ps(at(y), value);
                    }
                }
            }
        }
        end
    }

    /// `A^T m A` plus `bias` of one block: `m` holds its 36 products, `cout`
    /// floats each, one after the other, and `y` receives its 16 output pixels
    /// row by row. Computed as `A^T (m A)`, in the same order on every tier.
    pub(super) fn transform_output(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize) {
        assert!(m.len() >= POINTS * cout && bias.len() >= cout && y.len() >= BLOCK * BLOCK * cout);
        #[allow(unused_mut, reason = "only x86-64 has a vector path")]
        let mut start = 0;
        #[cfg(target_arch = "x86_64")]
        if crate::simd_tier() >= crate::TIER_AVX {
            // SAFETY: the CPU has AVX, and the lengths are asserted above.
            start = unsafe { output_avx(m, bias, y, cout) };
        }
        for c in start..cout {
            let rows: [[f32; BLOCK]; SIDE] = std::array::from_fn(|i| {
                a_t(std::array::from_fn(|x| m[(SIDE * i + x) * cout + c]), 0.0)
            });
            for x in 0..BLOCK {
                let column = a_t(std::array::from_fn(|i| rows[i][x]), bias[c]);
                for (i, value) in column.into_iter().enumerate() {
                    y[(BLOCK * i + x) * cout + c] = value;
                }
            }
        }
    }

    /// [`transform_output`] eight channels at a time, as far as whole groups of
    /// eight go; returns the channels done.
    ///
    /// # Safety
    /// The CPU must have AVX, `m` hold 36 `cout` floats, `bias` `cout` and `y`
    /// 16 `cout`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn output_avx(m: &[f32], bias: &[f32], y: &mut [f32], cout: usize) -> usize {
        use std::arch::x86_64::{
            __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
            _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
        };
        let (m, bias, out) = (m.as_ptr(), bias.as_ptr(), y.as_mut_ptr());
        let a_t = |v: [__m256; SIDE], b: __m256| -> [__m256; BLOCK] {
            let (k2, k4, k8) = (
                _mm256_set1_ps(2.0),
                _mm256_set1_ps(4.0),
                _mm256_set1_ps(8.0),
            );
            let s = _mm256_add_ps(v[1], v[2]);
            let d = _mm256_sub_ps(v[1], v[2]);
            let s2 = _mm256_add_ps(v[3], v[4]);
            let d2 = _mm256_sub_ps(v[3], v[4]);
            [
                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v[0], s), s2), b),
                _mm256_add_ps(_mm256_add_ps(d, _mm256_mul_ps(k2, d2)), b),
                _mm256_add_ps(_mm256_add_ps(s, _mm256_mul_ps(k4, s2)), b),
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_add_ps(d, _mm256_mul_ps(k8, d2)), v[5]),
                    b,
                ),
            ]
        };
        let mut c = 0;
        while c + 8 <= cout {
            // SAFETY: c + 8 <= cout, within every run of `m` and `y` and within
            // `bias`.
            unsafe {
                let zero = _mm256_setzero_ps();
                let rows: [[__m256; BLOCK]; SIDE] = std::array::from_fn(|i| {
                    a_t(
                        std::array::from_fn(|x| _mm256_loadu_ps(m.add((SIDE * i + x) * cout + c))),
                        zero,
                    )
                });
                let b = _mm256_loadu_ps(bias.add(c));
                #[expect(clippy::needless_range_loop, reason = "x picks a column of every row")]
                for x in 0..BLOCK {
                    let column = a_t(std::array::from_fn(|i| rows[i][x]), b);
                    for (i, value) in column.into_iter().enumerate() {
                        _mm256_storeu_ps(out.add((BLOCK * i + x) * cout + c), value);
                    }
                }
            }
            c += 8;
        }
        c
    }
}

/// The 3x3 stride-1 convolution of an [`Input::Image`] with `w` (`9 cin`
/// rows, taps row by row, channels within each, as for [`multiply`]), plus
/// its bias, stored to the pixels of `out`, `stride` floats apart, or added
/// to them with `accumulate`. `finish` then runs on each output pixel's
/// `w.n` channels. Inputs of fewer than [`MIN_CHANNELS`] channels take the
/// direct product.
pub fn conv3(
    pool: &mut Pool,
    input: Input,
    w: &Matrix,
    out: &mut [f32],
    stride: usize,
    accumulate: bool,
    finish: impl Fn(&mut [f32]) + Sync,
) {
    let Input::Image {
        data,
        height,
        width,
        channels: cin,
        stride: in_stride,
        taps: Taps::Conv {
            size: 3,
            step: 1,
            pad,
        },
    } = input
    else {
        panic!("Winograd's convolution needs a 3x3 stride-1 convolution");
    };
    let cout = w.n;
    assert!(
        w.k == 9 * cin
            && in_stride >= cin
            && data.len() >= (height * width - 1) * in_stride + cin
            && stride >= cout
            && out.len() >= (height * width - 1) * stride + cout,
        "convolution sizes do not match"
    );
    assert!(pad == Pad::Zero || (height > 1 && width > 1));
    let dst = Shared::new(out);
    if cin < MIN_CHANNELS {
        pool.split(height * width, 96, |range, scratch| {
            // SAFETY: tasks own disjoint pixel ranges.
            let rows = unsafe { dst.rows(range.clone(), stride, cout) };
            multiply(
                input,
                range,
                w,
                rows,
                stride,
                Act::None,
                accumulate,
                &mut scratch.gemm,
            );
            for pixel in rows.chunks_mut(stride) {
                finish(&mut pixel[..cout]);
            }
        });
        return;
    }
    if cin + cout <= F4_CHANNELS {
        winograd::<F4>(pool, input, w, dst, stride, accumulate, &finish);
    } else {
        winograd::<F2>(pool, input, w, dst, stride, accumulate, &finish);
    }
}

/// [`conv3`] by the form `F`, on an [`Input::Image`] of at least
/// [`MIN_CHANNELS`] channels.
fn winograd<F: Form>(
    pool: &mut Pool,
    input: Input,
    w: &Matrix,
    dst: Shared<'_>,
    stride: usize,
    accumulate: bool,
    finish: &(impl Fn(&mut [f32]) + Sync),
) {
    let Input::Image {
        data,
        height,
        width,
        channels: cin,
        stride: in_stride,
        taps: Taps::Conv { pad, .. },
    } = input
    else {
        unreachable!("conv3 checked the input");
    };
    let cout = w.n;
    let (size, side, points_n) = (F::BLOCK, F::BLOCK + 2, F::POINTS);
    let mut points = std::mem::take(&mut pool.winograd);
    if points.len() < F::POINTS {
        points.resize_with(F::POINTS, || Matrix::zeros(0, 0));
    }
    transform_weights::<F>(pool, w, cin, &mut points[..F::POINTS]);
    let bias = w.bias();
    let (blocks_y, blocks_x) = (height.div_ceil(size), width.div_ceil(size));
    let per_block = points_n * (cin + cout);
    let max = (TASK_FLOATS / per_block).max(crate::gemm::MR);
    pool.split(blocks_y * blocks_x, max, |range, scratch| {
        let n = range.len();
        let outputs = size * size;
        let (buf, gemm) = scratch.floats(n * per_block + outputs * cout + cin);
        let (u, rest) = buf.split_at_mut(n * points_n * cin);
        let (m, rest) = rest.split_at_mut(n * points_n * cout);
        let (y, zeros) = rest.split_at_mut(outputs * cout);
        let zeros = &mut zeros[..cin];
        zeros.fill(0.0);
        let zeros = &*zeros;
        for (i, block) in range.clone().enumerate() {
            let (by, bx) = (block / blocks_x, block % blocks_x);
            // Pixels more than one past the edge feed only outputs outside
            // the image.
            let d: [&[f32]; f4::POINTS] = std::array::from_fn(|k| {
                let py = (size * by + k / side) as isize - 1;
                let px = (size * bx + k % side) as isize - 1;
                let near = |i: isize, n: usize| (-1..=n as isize).contains(&i);
                let inside = |i: isize, n: usize| (0..n as isize).contains(&i);
                let (y, x) = match pad {
                    Pad::Reflect if near(py, height) && near(px, width) => {
                        (reflect(py, height), reflect(px, width))
                    }
                    _ if inside(py, height) && inside(px, width) => (py as usize, px as usize),
                    _ => return zeros,
                };
                &data[(y * width + x) * in_stride..][..cin]
            });
            F::input(&d[..points_n], &mut u[i * points_n * cin..], cin);
        }
        for (point, matrix) in points[..points_n].iter().enumerate() {
            multiply(
                Input::Rows {
                    data: &u[point * cin..],
                    stride: points_n * cin,
                },
                0..n,
                matrix,
                &mut m[point * cout..],
                points_n * cout,
                Act::None,
                false,
                gemm,
            );
        }
        for (i, block) in range.enumerate() {
            let (by, bx) = (block / blocks_x, block % blocks_x);
            F::output(&m[i * points_n * cout..], bias, y, cout);
            for (corner, values) in y[..outputs * cout].chunks_exact(cout).enumerate() {
                let (py, px) = (size * by + corner / size, size * bx + corner % size);
                if py >= height || px >= width {
                    continue;
                }
                // SAFETY: blocks partition the output pixels, and this task
                // alone owns the pixels of its blocks.
                let pixel = unsafe { dst.part((py * width + px) * stride, cout) };
                if accumulate {
                    pixel.iter_mut().zip(values).for_each(|(p, v)| *p += v);
                } else {
                    pixel.copy_from_slice(values);
                }
                finish(pixel);
            }
        }
    });
    pool.winograd = points;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{GemmScratch, Weights};

    /// The direct convolution of the same layer, as the oracle.
    fn direct(input: Input, w: &Matrix, pixels: usize) -> Vec<f32> {
        let mut out = vec![0.0; pixels * w.n];
        multiply(
            input,
            0..pixels,
            w,
            &mut out,
            w.n,
            Act::None,
            false,
            &mut GemmScratch::default(),
        );
        out
    }

    #[test]
    fn matches_direct_convolution() {
        let mut pool = Pool::new(3, "test").unwrap();
        let mut seed = 1u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        for (height, width, cin, cout, pad) in [
            (7, 9, 8, 24, Pad::Zero),
            (6, 5, 16, 16, Pad::Reflect),
            (1, 4, 3, 5, Pad::Zero),
            (5, 2, 20, 33, Pad::Reflect),
            // Past F4_CHANNELS: F(2x2, 3x3).
            (6, 7, 400, 200, Pad::Reflect),
            (3, 5, 300, 300, Pad::Zero),
        ] {
            // Half-float weights, at the AVX2 tier, give half-float transformed
            // weights, one more rounding each.
            for (half, tolerance) in [(false, 1e-4), (true, 0.05)] {
                let stride = cin + 2;
                let data: Vec<f32> = (0..height * width * stride).map(|_| next()).collect();
                let weights: Vec<f32> = (0..9 * cin * cout).map(|_| next()).collect();
                let bias: Vec<f32> = (0..cout).map(|_| next()).collect();
                let weights = if half {
                    Weights::F16(
                        weights
                            .iter()
                            .map(|&v| crate::half::f32_to_half(v))
                            .collect(),
                    )
                } else {
                    Weights::F32(weights)
                };
                let w = Matrix::new(&weights, 9 * cin, cout, Some(&bias));
                let input = Input::Image {
                    data: &data,
                    height,
                    width,
                    channels: cin,
                    stride,
                    taps: Taps::Conv {
                        size: 3,
                        step: 1,
                        pad,
                    },
                };
                let want = direct(input, &w, height * width);
                let out_stride = cout + 1;
                let mut got = vec![1.0; height * width * out_stride];
                conv3(&mut pool, input, &w, &mut got, out_stride, true, |p| {
                    p[0] *= 2.0
                });
                for (p, want) in want.chunks_exact(cout).enumerate() {
                    let got = &got[p * out_stride..][..cout];
                    for (c, (&g, &x)) in got.iter().zip(want).enumerate() {
                        let x = if c == 0 { 2.0 * (x + 1.0) } else { x + 1.0 };
                        assert!(
                            (g - x).abs() < tolerance,
                            "{height}x{width} half {half} pixel {p} channel {c}: {g} vs {x}"
                        );
                    }
                }
            }
        }
    }
}
