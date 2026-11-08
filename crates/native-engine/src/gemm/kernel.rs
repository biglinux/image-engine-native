//! The micro-kernels of [`multiply`](super::multiply): one 6 x 16 tile of
//! the product per row group, in plain Rust, SSE2, AVX and AVX2.

use super::{MR, NR};

/// Micro-tiles: for each row group, `c[6 x 16] = (load ? c : 0) + bias + sum
/// over taps and channels of a * b`, then `max(0, .)` when `relu`.
pub(super) struct KernelCall {
    pub(super) taps: usize,
    pub(super) channels: usize,
    /// `taps x 6` row pointers, tap-major.
    pub(super) a: *const *const f32,
    /// `taps * channels` rows of 16 weights, f32 or f16 as the kernel reads.
    pub(super) b: *const u8,
    /// 16 floats, or null.
    pub(super) bias: *const f32,
    pub(super) load: bool,
    pub(super) relu: bool,
    /// Consecutive row groups, each a tile: the pointers of group g start at
    /// `a + g * taps * 6`, its rows at `c + g * 6 * ldc`. One call for many
    /// saves the call and setup per tile, which counts in short products.
    pub(super) groups: usize,
}

pub(super) type Kernel = unsafe fn(&KernelCall, *mut f32, usize);

/// The kernel for 6 x 16 tiles, and one that computes only the first 8
/// columns of one tile, for a last panel no wider than that.
pub(super) struct Kernels {
    pub(super) full: Kernel,
    pub(super) narrow: Kernel,
}

#[cfg(target_arch = "x86_64")]
pub(super) fn kernels(half: bool) -> Kernels {
    let (full, narrow): (Kernel, Kernel) = match (crate::simd_tier(), half) {
        (crate::TIER_AVX2, true) => (kernel_avx2_half, kernel_avx2_half_narrow),
        (crate::TIER_AVX2, false) => (kernel_avx2, kernel_avx2_narrow),
        (_, true) => panic!("half-float panels need the AVX2 kernel"),
        (crate::TIER_AVX, false) => (kernel_avx, kernel_avx_narrow),
        (_, false) => (kernel_sse::<2>, kernel_sse::<1>),
    };
    Kernels { full, narrow }
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) fn kernels(half: bool) -> Kernels {
    assert!(!half, "half-float panels need the AVX2 kernel");
    Kernels {
        full: kernel_portable,
        narrow: kernel_portable,
    }
}

/// The kernel in plain Rust, for other architectures and as the oracle of the
/// vector kernels.
///
/// # Safety
/// `call` must describe readable memory as [`KernelCall`] says, and `c` must
/// be writable for 6 rows of 16 floats `ldc` apart.
#[cfg_attr(
    target_arch = "x86_64",
    allow(
        dead_code,
        reason = "x86-64 runs the SSE kernel instead; the tests compare against this one"
    )
)]
pub(super) unsafe fn kernel_portable(call: &KernelCall, c: *mut f32, ldc: usize) {
    // SAFETY: all reads and writes stay within the ranges the caller promises.
    unsafe {
        for group in 0..call.groups {
            let c = c.add(group * MR * ldc);
            let ptrs = call.a.add(group * call.taps * MR);
            let mut acc = [[0.0f32; NR]; MR];
            if call.load {
                for (i, row) in acc.iter_mut().enumerate() {
                    row.copy_from_slice(std::slice::from_raw_parts(c.add(i * ldc), NR));
                }
            }
            if !call.bias.is_null() {
                let bias = std::slice::from_raw_parts(call.bias, NR);
                for row in &mut acc {
                    for (v, b) in row.iter_mut().zip(bias) {
                        *v += b;
                    }
                }
            }
            let mut b = call.b.cast::<f32>();
            for t in 0..call.taps {
                let rows = std::slice::from_raw_parts(ptrs.add(t * MR), MR);
                for x in 0..call.channels {
                    let w = std::slice::from_raw_parts(b, NR);
                    for (row, &p) in acc.iter_mut().zip(rows) {
                        let a = *p.add(x);
                        for (v, wv) in row.iter_mut().zip(w) {
                            *v += a * wv;
                        }
                    }
                    b = b.add(NR);
                }
            }
            for (i, row) in acc.iter().enumerate() {
                let dst = std::slice::from_raw_parts_mut(c.add(i * ldc), NR);
                for (d, &v) in dst.iter_mut().zip(row) {
                    *d = if call.relu { v.max(0.0) } else { v };
                }
            }
        }
    }
}

/// [`kernel_portable`] in 8-lane registers: twelve accumulators, two weight
/// vectors and one broadcast per row and step; and `$narrow`, the same for
/// the first 8 columns only. `$half` weights are half floats, widened as they
/// are loaded; `$fma` is `fused` or `split`.
///
/// The rows are written out instead of looped over: Sandy Bridge runs the
/// schedule LLVM gives the loop slower.
#[cfg(target_arch = "x86_64")]
macro_rules! vector_kernel {
    ($(#[$doc:meta])* $name:ident, $narrow:ident, $features:literal, $half:tt, $fma:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// As [`kernel_portable`], and the CPU must have the features enabled
        /// here.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn $name(call: &KernelCall, c: *mut f32, ldc: usize) {
            use std::arch::x86_64::*;
            // SAFETY: all reads and writes stay within the ranges the caller
            // promises.
            unsafe {
                for group in 0..call.groups {
                    let c = c.add(group * MR * ldc);
                    let ptrs = call.a.add(group * call.taps * MR);
                    let mut acc = [[_mm256_setzero_ps(); 2]; MR];
                    if call.load {
                        for (i, row) in acc.iter_mut().enumerate() {
                            row[0] = _mm256_loadu_ps(c.add(i * ldc));
                            row[1] = _mm256_loadu_ps(c.add(i * ldc + 8));
                        }
                    }
                    if !call.bias.is_null() {
                        let b0 = _mm256_loadu_ps(call.bias);
                        let b1 = _mm256_loadu_ps(call.bias.add(8));
                        for row in &mut acc {
                            row[0] = _mm256_add_ps(row[0], b0);
                            row[1] = _mm256_add_ps(row[1], b1);
                        }
                    }
                    let mut b = call.b;
                    let width = if $half { 2 } else { 4 };
                    for t in 0..call.taps {
                        let p = ptrs.add(t * MR);
                        let (p0, p1, p2) = (*p, *p.add(1), *p.add(2));
                        let (p3, p4, p5) = (*p.add(3), *p.add(4), *p.add(5));
                        for x in 0..call.channels {
                            let w0 = vector_kernel!(@weights $half, b, 0);
                            let w1 = vector_kernel!(@weights $half, b, 1);
                            let a = _mm256_broadcast_ss(&*p0.add(x));
                            acc[0][0] = madd8!($fma, a, w0, acc[0][0]);
                            acc[0][1] = madd8!($fma, a, w1, acc[0][1]);
                            let a = _mm256_broadcast_ss(&*p1.add(x));
                            acc[1][0] = madd8!($fma, a, w0, acc[1][0]);
                            acc[1][1] = madd8!($fma, a, w1, acc[1][1]);
                            let a = _mm256_broadcast_ss(&*p2.add(x));
                            acc[2][0] = madd8!($fma, a, w0, acc[2][0]);
                            acc[2][1] = madd8!($fma, a, w1, acc[2][1]);
                            let a = _mm256_broadcast_ss(&*p3.add(x));
                            acc[3][0] = madd8!($fma, a, w0, acc[3][0]);
                            acc[3][1] = madd8!($fma, a, w1, acc[3][1]);
                            let a = _mm256_broadcast_ss(&*p4.add(x));
                            acc[4][0] = madd8!($fma, a, w0, acc[4][0]);
                            acc[4][1] = madd8!($fma, a, w1, acc[4][1]);
                            let a = _mm256_broadcast_ss(&*p5.add(x));
                            acc[5][0] = madd8!($fma, a, w0, acc[5][0]);
                            acc[5][1] = madd8!($fma, a, w1, acc[5][1]);
                            b = b.add(NR * width);
                        }
                    }
                    let zero = _mm256_setzero_ps();
                    for (i, row) in acc.iter().enumerate() {
                        let (v0, v1) = if call.relu {
                            (_mm256_max_ps(row[0], zero), _mm256_max_ps(row[1], zero))
                        } else {
                            (row[0], row[1])
                        };
                        _mm256_storeu_ps(c.add(i * ldc), v0);
                        _mm256_storeu_ps(c.add(i * ldc + 8), v1);
                    }
                }
            }
        }

        /// [`$name`] for the first 8 columns of one tile.
        ///
        /// # Safety
        /// As [`$name`].
        #[target_feature(enable = $features)]
        pub(super) unsafe fn $narrow(call: &KernelCall, c: *mut f32, ldc: usize) {
            /// `N` rows from group pointers `a`: one group, or two, whose
            /// twelve sums keep the multiply-adds' latency covered.
            #[target_feature(enable = $features)]
            unsafe fn rows<const N: usize>(
                call: &KernelCall,
                a: *const *const f32,
                c: *mut f32,
                ldc: usize,
            ) {
                use std::arch::x86_64::*;
                // SAFETY: as in the full kernel, over half the columns.
                unsafe {
                    let mut acc = [_mm256_setzero_ps(); N];
                    if call.load {
                        for (i, sum) in acc.iter_mut().enumerate() {
                            *sum = _mm256_loadu_ps(c.add(i * ldc));
                        }
                    }
                    if !call.bias.is_null() {
                        let b0 = _mm256_loadu_ps(call.bias);
                        for sum in &mut acc {
                            *sum = _mm256_add_ps(*sum, b0);
                        }
                    }
                    let mut b = call.b;
                    let width = if $half { 2 } else { 4 };
                    for t in 0..call.taps {
                        let p: [*const f32; N] = std::array::from_fn(|i| {
                            *a.add(i / MR * call.taps * MR + t * MR + i % MR)
                        });
                        for x in 0..call.channels {
                            let w0 = vector_kernel!(@weights $half, b, 0);
                            for (sum, row) in acc.iter_mut().zip(&p) {
                                *sum = madd8!($fma, _mm256_broadcast_ss(&*row.add(x)), w0, *sum);
                            }
                            b = b.add(NR * width);
                        }
                    }
                    let zero = _mm256_setzero_ps();
                    for (i, &sum) in acc.iter().enumerate() {
                        let value = if call.relu { _mm256_max_ps(sum, zero) } else { sum };
                        _mm256_storeu_ps(c.add(i * ldc), value);
                    }
                }
            }
            // SAFETY: as in the full kernel, over half the columns.
            unsafe {
                let mut group = 0;
                while group < call.groups {
                    let a = call.a.add(group * call.taps * MR);
                    let c = c.add(group * MR * ldc);
                    if call.groups - group >= 2 {
                        rows::<{ 2 * MR }>(call, a, c, ldc);
                        group += 2;
                    } else {
                        rows::<MR>(call, a, c, ldc);
                        group += 1;
                    }
                }
            }
        }
    };
    (@weights true, $b:ident, $v:literal) => {
        _mm256_cvtph_ps(_mm_loadu_si128($b.add($v * 16).cast()))
    };
    (@weights false, $b:ident, $v:literal) => {
        _mm256_loadu_ps($b.add($v * 32).cast())
    };
}

/// [`kernel_portable`] in SSE registers, which every x86-64 CPU has: the
/// columns as `V` halves of 6 x 8, each with twelve 4-wide accumulators, two
/// weight vectors and one broadcast per row and step.
///
/// # Safety
/// As [`kernel_portable`].
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn kernel_sse<const V: usize>(call: &KernelCall, c: *mut f32, ldc: usize) {
    use std::arch::x86_64::*;
    // SAFETY: all reads and writes stay within the ranges the caller promises;
    // SSE2 is part of x86-64.
    unsafe {
        let weights = call.b.cast::<f32>();
        for group in 0..call.groups {
            let c = c.add(group * MR * ldc);
            let ptrs = call.a.add(group * call.taps * MR);
            for half in (0..V).map(|h| h * 8) {
                let mut acc = [[_mm_setzero_ps(); 2]; MR];
                if call.load {
                    for (i, row) in acc.iter_mut().enumerate() {
                        row[0] = _mm_loadu_ps(c.add(i * ldc + half));
                        row[1] = _mm_loadu_ps(c.add(i * ldc + half + 4));
                    }
                }
                if !call.bias.is_null() {
                    let b0 = _mm_loadu_ps(call.bias.add(half));
                    let b1 = _mm_loadu_ps(call.bias.add(half + 4));
                    for row in &mut acc {
                        row[0] = _mm_add_ps(row[0], b0);
                        row[1] = _mm_add_ps(row[1], b1);
                    }
                }
                let mut b = weights.add(half);
                for t in 0..call.taps {
                    let p = ptrs.add(t * MR);
                    let rows = [*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5)];
                    for x in 0..call.channels {
                        let w0 = _mm_loadu_ps(b);
                        let w1 = _mm_loadu_ps(b.add(4));
                        for (row, &ptr) in acc.iter_mut().zip(&rows) {
                            let a = _mm_set1_ps(*ptr.add(x));
                            row[0] = _mm_add_ps(_mm_mul_ps(a, w0), row[0]);
                            row[1] = _mm_add_ps(_mm_mul_ps(a, w1), row[1]);
                        }
                        b = b.add(NR);
                    }
                }
                let zero = _mm_setzero_ps();
                for (i, row) in acc.iter().enumerate() {
                    let (v0, v1) = if call.relu {
                        (_mm_max_ps(row[0], zero), _mm_max_ps(row[1], zero))
                    } else {
                        (row[0], row[1])
                    };
                    _mm_storeu_ps(c.add(i * ldc + half), v0);
                    _mm_storeu_ps(c.add(i * ldc + half + 4), v1);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
vector_kernel!(
    /// The AVX2 kernel, one fused multiply-add per product.
    kernel_avx2, kernel_avx2_narrow, "avx2,fma", false, fused
);
#[cfg(target_arch = "x86_64")]
vector_kernel!(
    /// [`kernel_avx2`] over half-float weights.
    kernel_avx2_half, kernel_avx2_half_narrow, "avx2,fma,f16c", true, fused
);
#[cfg(target_arch = "x86_64")]
vector_kernel!(
    /// The AVX kernel: a multiply and an add per product, which Sandy and Ivy
    /// Bridge run on separate ports.
    kernel_avx, kernel_avx_narrow, "avx", false, split
);
