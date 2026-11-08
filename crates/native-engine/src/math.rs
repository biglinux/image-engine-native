//! Elementwise functions and row normalisations the networks share.
//!
//! Each body is written once as plain arithmetic and compiled three times:
//! inside an AVX2 and an AVX function, where LLVM vectorises it 8 wide, and as
//! is. No copy fuses multiplies and adds the source keeps apart, so all three
//! give the same bits. The exponential follows XNNPACK's f32 "p5" kernels (see `NOTICE`):
//! 2^n times a degree-5 polynomial, with `n` rounded by a magic bias.

const EXP_MAGIC_BIAS: f32 = f32::from_bits(0x4b40_007f);
const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b);
const MINUS_LN2_HI: f32 = f32::from_bits(0xbf31_7200);
const MINUS_LN2_LO: f32 = f32::from_bits(0xb5bf_be8e);
const EXP_C5: f32 = f32::from_bits(0x3c07_cfce);
const EXP_C4: f32 = f32::from_bits(0x3d2b_9d0d);
const EXP_C3: f32 = f32::from_bits(0x3e2a_ad40);
const EXP_C2: f32 = f32::from_bits(0x3eff_fee3);
const EXP_C1: f32 = f32::from_bits(0x3f7f_fffb);
/// Below this, exp(z) is denormal and taken as 0.
const EXP_CUTOFF: f32 = f32::from_bits(0xc2ae_ac4f);

/// `exp(z)` for `z <= 0`.
#[inline(always)]
fn exp_neg(z: f32) -> f32 {
    let n = z * LOG2E + EXP_MAGIC_BIAS;
    let s = f32::from_bits(n.to_bits() << 23);
    let n = n - EXP_MAGIC_BIAS;
    let t = n * MINUS_LN2_LO + (n * MINUS_LN2_HI + z);
    let p = (((EXP_C5 * t + EXP_C4) * t + EXP_C3) * t + EXP_C2) * t + EXP_C1;
    let e = (t * s) * p + s;
    if z < EXP_CUTOFF { 0.0 } else { e }
}

/// `1 / (1 + exp(-x))`, from `exp(-|x|)` so neither tail cancels.
#[inline(always)]
pub fn sigmoid_one(x: f32) -> f32 {
    let e = exp_neg(-x.abs());
    let s = e / (e + 1.0);
    if x > 0.0 { 1.0 - s } else { s }
}

// erf(x) = x * P(x^2) / Q(x^2) on [-4, 4], beyond which it is +-1 in f32:
// the coefficients of Eigen's generic_fast_erf_float (see `NOTICE`), under
// 5e-7 from erf.
const ERF_A: [f32; 7] = [
    -2.726_142_3e-10,
    2.770_681_4e-8,
    -2.101_024e-6,
    -5.692_506_4e-5,
    -7.349_906_3e-4,
    -2.954_6e-3,
    -1.609_603_3e-2,
];
const ERF_B: [f32; 5] = [
    -1.456_607_2e-5,
    -2.133_740_6e-4,
    -1.682_827e-3,
    -7.373_329e-3,
    -1.426_474e-2,
];

/// `x * (1 + erf(x / sqrt 2)) / 2`, the exact GELU.
#[inline(always)]
fn gelu_one(x: f32) -> f32 {
    let z = (x * std::f32::consts::FRAC_1_SQRT_2).clamp(-4.0, 4.0);
    let z2 = z * z;
    let mut p = ERF_A[0];
    for c in &ERF_A[1..] {
        p = p * z2 + c;
    }
    let mut q = ERF_B[0];
    for c in &ERF_B[1..] {
        q = q * z2 + c;
    }
    0.5 * x * (1.0 + z * p / q)
}

/// Sum in eight interleaved lanes, which LLVM may keep in one vector.
#[inline(always)]
fn sum(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (chunks, rest) = values.as_chunks::<8>();
    for chunk in chunks {
        for (l, v) in lanes.iter_mut().zip(chunk) {
            *l += v;
        }
    }
    let mut total = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    for v in rest {
        total += v;
    }
    total
}

#[inline(always)]
fn layer_norm_body(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len() as f32;
    let mean = sum(x) / n;
    let mut squares = [0.0f32; 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for chunk in chunks {
        for (s, v) in squares.iter_mut().zip(chunk) {
            let d = v - mean;
            *s += d * d;
        }
    }
    let mut var = sum(&squares);
    for v in rest {
        let d = v - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / n + 1e-5).sqrt();
    for ((v, g), b) in x.iter_mut().zip(gamma).zip(beta) {
        *v = (*v - mean) * inv * g + b;
    }
}

#[inline(always)]
pub fn softmax_body(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    softmax_from(x, max);
}

/// [`softmax_body`] given the largest value of `x`.
#[inline(always)]
pub fn softmax_from(x: &mut [f32], max: f32) {
    for v in x.iter_mut() {
        *v = exp_neg(*v - max);
    }
    let inv = 1.0 / sum(x);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

dispatch! {
    /// GELU in place.
    pub gelu(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = gelu_one(*v);
        }
    }
}

dispatch! {
    /// The logistic function in place.
    pub sigmoid(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = sigmoid_one(*v);
        }
    }
}

dispatch! {
    /// Layer normalisation of each `gamma.len()`-long row of `x`, epsilon 1e-5.
    pub layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
        assert!(
            !gamma.is_empty()
                && x.len().is_multiple_of(gamma.len())
                && beta.len() == gamma.len(),
            "rows of {} values with {} betas cannot hold {}",
            gamma.len(),
            beta.len(),
            x.len()
        );
        for row in x.chunks_exact_mut(gamma.len()) {
            layer_norm_body(row, gamma, beta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// erf from its Taylor series in f64: at |x| <= 4 the largest term is
    /// about 4e6, so cancellation still leaves ten correct digits.
    fn erf_series(x: f64) -> f64 {
        let (mut term, mut total) = (x, x);
        for n in 1..200 {
            term *= -x * x / f64::from(n);
            total += term / f64::from(2 * n + 1);
        }
        total * 2.0 / std::f64::consts::PI.sqrt()
    }

    #[test]
    fn gelu_matches_erf_definition() {
        let mut x: Vec<f32> = (-800..=800).map(|i| i as f32 / 100.0).collect();
        let want: Vec<f64> = x
            .iter()
            .map(|&v| {
                let z = (f64::from(v) / std::f64::consts::SQRT_2).clamp(-4.0, 4.0);
                0.5 * f64::from(v)
                    * (1.0
                        + if z.abs() >= 4.0 {
                            z.signum()
                        } else {
                            erf_series(z)
                        })
            })
            .collect();
        gelu(&mut x);
        for (g, w) in x.iter().zip(&want) {
            assert!(
                (f64::from(*g) - w).abs() < 2e-6 * (1.0 + w.abs()),
                "{g} vs {w}"
            );
        }
    }

    #[test]
    fn sigmoid_matches_libm_and_saturates() {
        let mut x: Vec<f32> = (-2000..=2000).map(|i| i as f32 / 20.0).collect();
        let want: Vec<f32> = x
            .iter()
            .map(|&v| 1.0 / (1.0 + (-f64::from(v)).exp()) as f32)
            .collect();
        sigmoid(&mut x);
        for (g, w) in x.iter().zip(&want) {
            assert!((g - w).abs() <= 2.0 * f32::EPSILON, "{g} vs {w}");
        }
        let mut ends = [-1e30, 1e30, 0.0];
        sigmoid(&mut ends);
        assert_eq!(ends, [0.0, 1.0, 0.5]);
    }

    #[test]
    fn softmax_matches_definition() {
        let mut x: Vec<f32> = (0..98)
            .map(|i| (i as f32 * 0.7).sin() * 30.0 - 100.0 * (i % 13 == 0) as u8 as f32)
            .collect();
        let want: Vec<f64> = x
            .chunks(49)
            .flat_map(|row| {
                let max = row.iter().fold(f32::MIN, |m, &v| m.max(v));
                let e: Vec<f64> = row.iter().map(|&v| f64::from(v - max).exp()).collect();
                let total: f64 = e.iter().sum();
                e.into_iter().map(move |v| v / total)
            })
            .collect();
        for row in x.as_chunks_mut::<49>().0 {
            softmax_body(row);
        }
        for (g, w) in x.iter().zip(&want) {
            assert!((f64::from(*g) - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    /// Attention from its definition in f64, for one head of one window.
    #[test]
    fn layer_norm_matches_definition() {
        let c = 37;
        let mut x: Vec<f32> = (0..3 * c)
            .map(|i| (i as f32 * 1.3).cos() * 4.0 + 2.0)
            .collect();
        let gamma: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 / 50.0).collect();
        let beta: Vec<f32> = (0..c).map(|i| i as f32 / 10.0).collect();
        let want: Vec<f64> = x
            .chunks(c)
            .flat_map(|row| {
                let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
                let var = row
                    .iter()
                    .map(|&v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / c as f64;
                let row = row.to_vec();
                (0..c).map(move |j| (f64::from(row[j]) - mean) / (var + 1e-5).sqrt())
            })
            .collect();
        layer_norm(&mut x, &gamma, &beta);
        for (i, (g, w)) in x.iter().zip(&want).enumerate() {
            let w = w * f64::from(gamma[i % c]) + f64::from(beta[i % c]);
            assert!((f64::from(*g) - w).abs() < 1e-5, "{g} vs {w}");
        }
    }
}
