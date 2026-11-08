//! Attention over one window of a Swin block, in 8-lane registers when the
//! CPU has AVX.

use crate::math::{softmax_body, softmax_from};

/// Channels of one attention head.
pub const HEAD_DIM: usize = 32;

/// Floats of workspace [`attention`] needs for windows of `tokens` tokens.
pub const fn work_len(tokens: usize) -> usize {
    let padded = tokens.next_multiple_of(8);
    HEAD_DIM * padded + 2 * padded
}

/// Attention of head `head` over one window of `N` tokens. `qkv` holds `N`
/// rows of `3 * c` values, queries then keys then values; `bias` is the
/// head's `N x N` relative position bias, already including any shift mask
/// (`-inf` where two tokens may not attend). Writes the head's 32 columns of
/// the `N` rows of `out`, `c` apart.
///
/// The keys are first transposed into `work`, 32 rows of `N` padded to whole
/// 8-float vectors, so that a query row's scores are a few vectors of
/// multiply-adds; the weighted sum of values then runs over two query rows
/// at a time.
pub fn attention<const N: usize>(
    qkv: &[f32],
    c: usize,
    head: usize,
    bias: &[f32],
    work: &mut [f32],
    out: &mut [f32],
) {
    const { assert!(N > 0 && N <= 8 * MAX_VECTORS) };
    const D: usize = HEAD_DIM;
    let padded = N.next_multiple_of(8);
    assert!(head * D + D <= c && qkv.len() >= N * 3 * c && bias.len() >= N * N);
    assert!(work.len() >= work_len(N) && out.len() >= (N - 1) * c + head * D + D);
    let (keys, scores) = work.split_at_mut(D * padded);
    let k0 = c + head * D;
    for d in 0..D {
        let row = &mut keys[d * padded..][..padded];
        for (j, k) in row[..N].iter_mut().enumerate() {
            *k = qkv[j * 3 * c + k0 + d];
        }
        row[N..].fill(0.0);
    }
    #[cfg(target_arch = "x86_64")]
    match crate::simd_tier() {
        crate::TIER_AVX2 => {
            // SAFETY: the CPU has AVX2 and FMA, and the asserts above cover
            // every index the kernel reads or writes.
            return unsafe { attention_avx2::<N>(qkv, c, head, bias, keys, scores, out) };
        }
        crate::TIER_AVX => {
            // SAFETY: as above, with AVX.
            return unsafe { attention_avx::<N>(qkv, c, head, bias, keys, scores, out) };
        }
        _ => {}
    }
    attention_portable::<N>(qkv, c, head, bias, keys, scores, out);
}

/// Score vectors a query row may need: windows of up to 64 tokens.
const MAX_VECTORS: usize = 8;

fn attention_portable<const N: usize>(
    qkv: &[f32],
    c: usize,
    head: usize,
    bias: &[f32],
    keys: &[f32],
    scores: &mut [f32],
    out: &mut [f32],
) {
    const D: usize = HEAD_DIM;
    let padded = N.next_multiple_of(8);
    let scale = 1.0 / (D as f32).sqrt();
    let (q0, v0) = (head * D, 2 * c + head * D);
    let mut i = 0;
    while i < N {
        let pair = (N - i).min(2);
        for r in 0..pair {
            let row = i + r;
            let s = &mut scores[r * padded..][..padded];
            s.fill(0.0);
            for d in 0..D {
                let q = qkv[row * 3 * c + q0 + d] * scale;
                for (acc, k) in s.iter_mut().zip(&keys[d * padded..][..padded]) {
                    *acc += q * k;
                }
            }
            for (v, b) in s[..N].iter_mut().zip(&bias[row * N..][..N]) {
                *v += b;
            }
            softmax_body(&mut s[..N]);
        }
        let mut sums = [[0.0f32; D]; 2];
        for j in 0..N {
            let p = [scores[j], scores[(pair - 1) * padded + j]];
            let v = &qkv[j * 3 * c + v0..][..D];
            for d in 0..D {
                sums[0][d] += p[0] * v[d];
                sums[1][d] += p[1] * v[d];
            }
        }
        for (r, sum) in sums.iter().enumerate().take(pair) {
            out[(i + r) * c + head * D..][..D].copy_from_slice(sum);
        }
        i += pair;
    }
}

/// [`attention_portable`] in 8-lane registers; `$fma` is `fused` or `split`.
#[cfg(target_arch = "x86_64")]
macro_rules! vector_attention {
    ($(#[$doc:meta])* $name:ident, $features:literal, $fma:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must have the features enabled here, and the slices must be
        /// as large as [`attention`] asserts.
        #[target_feature(enable = $features)]
        unsafe fn $name<const N: usize>(
            qkv: &[f32],
            c: usize,
            head: usize,
            bias: &[f32],
            keys: &[f32],
            scores: &mut [f32],
            out: &mut [f32],
        ) {
            use std::arch::x86_64::*;
            const D: usize = HEAD_DIM;
            let padded = N.next_multiple_of(8);
            let vectors = padded / 8;
            let scale = 1.0 / (D as f32).sqrt();
            let (q0, v0) = (head * D, 2 * c + head * D);
            // SAFETY: every pointer below stays inside the slices, as
            // `attention` asserted: rows below N of `qkv`, `keys` of 32 rows
            // and `scores` of 2 rows, each `padded` long.
            unsafe {
                let mut i = 0;
                while i < N {
                    let pair = (N - i).min(2);
                    for r in 0..pair {
                        let row = i + r;
                        let q = qkv.as_ptr().add(row * 3 * c + q0);
                        let mut acc = [_mm256_setzero_ps(); MAX_VECTORS];
                        for d in 0..D {
                            let a = _mm256_set1_ps(*q.add(d) * scale);
                            let k = keys.as_ptr().add(d * padded);
                            for (v, sum) in acc[..vectors].iter_mut().enumerate() {
                                *sum = madd8!($fma, a, _mm256_loadu_ps(k.add(v * 8)), *sum);
                            }
                        }
                        let s = &mut scores[r * padded..][..padded];
                        for (v, sum) in acc[..vectors].iter().enumerate() {
                            _mm256_storeu_ps(s.as_mut_ptr().add(v * 8), *sum);
                        }
                        for (v, b) in s[..N].iter_mut().zip(&bias[row * N..][..N]) {
                            *v += b;
                        }
                        // The largest score from whole vectors, the padding
                        // past token N set below every score first.
                        s[N..].fill(f32::NEG_INFINITY);
                        let mut max = _mm256_loadu_ps(s.as_ptr());
                        for v in 1..vectors {
                            max = _mm256_max_ps(max, _mm256_loadu_ps(s.as_ptr().add(v * 8)));
                        }
                        max = _mm256_max_ps(max, _mm256_permute2f128_ps::<1>(max, max));
                        max = _mm256_max_ps(max, _mm256_permute_ps::<0b0100_1110>(max));
                        max = _mm256_max_ps(max, _mm256_permute_ps::<0b1011_0001>(max));
                        softmax_from(&mut s[..N], _mm256_cvtss_f32(max));
                    }
                    let mut sums = [[_mm256_setzero_ps(); D / 8]; 2];
                    for j in 0..N {
                        let p0 = _mm256_set1_ps(scores[j]);
                        let p1 = _mm256_set1_ps(scores[(pair - 1) * padded + j]);
                        let v = qkv.as_ptr().add(j * 3 * c + v0);
                        let [first, second] = &mut sums;
                        for (t, (s0, s1)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
                            let value = _mm256_loadu_ps(v.add(t * 8));
                            *s0 = madd8!($fma, p0, value, *s0);
                            *s1 = madd8!($fma, p1, value, *s1);
                        }
                    }
                    for (r, sum) in sums.iter().enumerate().take(pair) {
                        let o = out.as_mut_ptr().add((i + r) * c + head * D);
                        for (t, v) in sum.iter().enumerate() {
                            _mm256_storeu_ps(o.add(t * 8), *v);
                        }
                    }
                    i += pair;
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
vector_attention!(
    /// The AVX2 attention, one fused multiply-add per product.
    attention_avx2, "avx2,fma", fused
);
#[cfg(target_arch = "x86_64")]
vector_attention!(
    /// The AVX attention, a multiply and an add per product.
    attention_avx, "avx", split
);

#[cfg(test)]
mod tests {
    use super::*;

    fn matches_definition<const N: usize>() {
        let (d, c, head) = (HEAD_DIM, 96, 1);
        let qkv: Vec<f32> = (0..N * 3 * c).map(|i| ((i as f32) * 0.618).sin()).collect();
        let bias: Vec<f32> = (0..N * N)
            .map(|i| ((i as f32) * 0.31).cos() - 100.0 * f32::from(i % 17 == 0))
            .collect();
        let mut work = vec![0.0; work_len(N)];
        let mut out = vec![0.0; N * c];
        attention::<N>(&qkv, c, head, &bias, &mut work, &mut out);
        let at = |row: usize, col: usize| f64::from(qkv[row * 3 * c + col]);
        for i in 0..N {
            let scores: Vec<f64> = (0..N)
                .map(|j| {
                    let dot: f64 = (0..d)
                        .map(|x| at(i, head * d + x) * at(j, c + head * d + x))
                        .sum();
                    dot / (d as f64).sqrt() + f64::from(bias[i * N + j])
                })
                .collect();
            let max = scores.iter().copied().fold(f64::MIN, f64::max);
            let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
            for x in 0..d {
                let want: f64 = (0..N)
                    .map(|j| (scores[j] - max).exp() / total * at(j, 2 * c + head * d + x))
                    .sum();
                let got = f64::from(out[i * c + head * d + x]);
                assert!(
                    (got - want).abs() < 1e-5,
                    "N {N} row {i} col {x}: {got} vs {want}"
                );
            }
        }
    }

    /// Swin's 7 x 7 windows, padded, and 8 x 8 ones, which need no padding.
    #[test]
    fn attention_matches_definition() {
        matches_definition::<49>();
        matches_definition::<64>();
    }
}
