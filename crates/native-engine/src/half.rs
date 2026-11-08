//! IEEE 754 half floats as the models store weights and the GPUs keep
//! activations, converted to and from floats exactly, ties to even.

use std::sync::LazyLock;

/// The IEEE 754 half float nearest `v`, ties to even; NaN stays NaN.
pub fn f32_to_half(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mantissa = x & 0x7f_ffff;
    if (x >> 23) & 0xff == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    // The exponent rebiased for half floats.
    let exponent = ((x >> 23) & 0xff) as i32 - 112;
    // `kept` with the `shift` bits below it rounded off, ties to even.
    let round = |kept: u32, low: u32, shift: u32| {
        let half = 1 << (shift - 1);
        kept + u32::from(low > half || (low == half && kept & 1 == 1))
    };
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        // Subnormal: the significand times 2^-24.
        if exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let low = significand & ((1 << shift) - 1);
        return sign | round(significand >> shift, low, shift) as u16;
    }
    // A carry out of the mantissa raises the exponent, up to infinity.
    sign | round(
        ((exponent as u32) << 10) | (mantissa >> 13),
        mantissa & 0x1fff,
        13,
    ) as u16
}

/// A little-endian half float as a float, through a table of all of them.
pub fn half_le_to_f32(h: [u8; 2]) -> f32 {
    static TABLE: LazyLock<Vec<f32>> = LazyLock::new(|| (0..=u16::MAX).map(half_to_f32).collect());
    TABLE[usize::from(u16::from_le_bytes(h))]
}

/// The little-endian half floats of `src` as floats, into `dst`, exactly;
/// eight at a time with F16C at the AVX2 tier.
pub fn halves_to_f32(src: &[[u8; 2]], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "as many floats as halves");
    #[allow(unused_mut, reason = "only x86-64 has a vector path")]
    let mut start = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::simd_tier() >= crate::TIER_AVX2 {
        // SAFETY: tier 2's CPUs have F16C; the lengths match (asserted).
        start = unsafe { widen_f16c(src, dst) };
    }
    for (d, &h) in dst[start..].iter_mut().zip(&src[start..]) {
        *d = half_le_to_f32(h);
    }
}

/// The floats of `src` as little-endian half floats, nearest with ties to
/// even as [`f32_to_half`], into `dst`; eight at a time with F16C at the
/// AVX2 tier.
pub fn floats_to_halves(src: &[f32], dst: &mut [[u8; 2]]) {
    assert_eq!(src.len(), dst.len(), "as many halves as floats");
    #[allow(unused_mut, reason = "only x86-64 has a vector path")]
    let mut start = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::simd_tier() >= crate::TIER_AVX2 {
        // SAFETY: tier 2's CPUs have F16C; the lengths match (asserted).
        start = unsafe { narrow_f16c(src, dst) };
    }
    for (d, &v) in dst[start..].iter_mut().zip(&src[start..]) {
        *d = f32_to_half(v).to_le_bytes();
    }
}

/// [`floats_to_halves`] eight at a time, as far as whole groups go; returns
/// the values done.
///
/// # Safety
/// The CPU must have F16C, and `dst` be as long as `src`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn narrow_f16c(src: &[f32], dst: &mut [[u8; 2]]) -> usize {
    use std::arch::x86_64::{
        __m128i, _MM_FROUND_TO_NEAREST_INT, _mm_storeu_si128, _mm256_cvtps_ph, _mm256_loadu_ps,
    };
    let done = src.len() / 8 * 8;
    for i in (0..done).step_by(8) {
        // SAFETY: values i..i + 8 of both, within `done`.
        unsafe {
            let h =
                _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(src.as_ptr().add(i)));
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), h);
        }
    }
    done
}

/// [`halves_to_f32`] eight at a time, as far as whole groups go; returns
/// the values done.
///
/// # Safety
/// The CPU must have F16C, and `dst` be as long as `src`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn widen_f16c(src: &[[u8; 2]], dst: &mut [f32]) -> usize {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let done = src.len() / 8 * 8;
    for i in (0..done).step_by(8) {
        // SAFETY: values i..i + 8 of both, within `done`.
        unsafe {
            let h = _mm_loadu_si128(src.as_ptr().add(i).cast::<__m128i>());
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        }
    }
    done
}

/// An IEEE 754 half float, exactly, as a float.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from((h >> 10) & 0x1f);
    let mantissa = u32::from(h & 0x3ff);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        // Subnormal: mantissa * 2^-24, exact in f32.
        (0, m) => {
            let magnitude = m as f32 * f32::from_bits(0x3380_0000);
            return if sign == 0 { magnitude } else { -magnitude };
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_narrowing_matches_scalar() {
        // Bit patterns across every exponent, ties between halves, values
        // past the largest half and below the smallest, and a partial group.
        let mut seed = 7u32;
        let mut src: Vec<f32> = (0..100_003)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(seed)
            })
            .collect();
        src.extend([
            1.0 + 2f32.powi(-11),
            1.0 + 3.0 * 2f32.powi(-11),
            65520.0,
            2f32.powi(-25),
        ]);
        let mut got = vec![[0; 2]; src.len()];
        floats_to_halves(&src, &mut got);
        for (&v, &g) in src.iter().zip(&got) {
            let (g, want) = (u16::from_le_bytes(g), f32_to_half(v));
            let nan = |h: u16| h & 0x7c00 == 0x7c00 && h & 0x3ff != 0;
            assert!(
                g == want || (nan(g) && nan(want)),
                "{v:e}: {g:#x} vs {want:#x}"
            );
        }
    }

    #[test]
    fn batch_widening_matches_scalar() {
        // Every half float, and one more so the last group is partial.
        let src: Vec<[u8; 2]> = (0..=u16::MAX)
            .chain([0x3c00])
            .map(u16::to_le_bytes)
            .collect();
        let mut got = vec![0.0; src.len()];
        halves_to_f32(&src, &mut got);
        for (&h, &g) in src.iter().zip(&got) {
            let want = half_to_f32(u16::from_le_bytes(h));
            assert!(
                g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                "{h:?}"
            );
        }
    }

    #[test]
    fn half_rounding_is_ieee() {
        // Every half float survives the round trip, bit for bit.
        for h in 0..=u16::MAX {
            let back = f32_to_half(half_to_f32(h));
            if half_to_f32(h).is_nan() {
                assert!(half_to_f32(back).is_nan());
            } else {
                assert_eq!(back, h, "{h:#06x}");
            }
        }
        // Halfway between neighbours goes to the even one.
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_half(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_half(2f32.powi(-25)), 0);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 2);
        assert_eq!(f32_to_half(-2f32.powi(-26)), 0x8000);
        // The largest half, the overflow threshold and beyond.
        assert_eq!(f32_to_half(65504.0), 0x7bff);
        assert_eq!(f32_to_half(65519.0), 0x7bff);
        assert_eq!(f32_to_half(65520.0), 0x7c00);
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xfc00);
    }

    #[test]
    fn half_floats_widen_exactly() {
        let cases: [(u16, f32); 8] = [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x7bff, 65504.0),
            (0x3555, 0.333_251_95),
            (0x0001, 2f32.powi(-24)),
            (0x83ff, -1023.0 * 2f32.powi(-24)),
            (0x8000, -0.0),
            (0x7c00, f32::INFINITY),
        ];
        for (h, want) in cases {
            assert_eq!(half_to_f32(h).to_bits(), want.to_bits(), "{h:#06x}");
        }
        assert!(half_to_f32(0x7e00).is_nan());
    }
}
