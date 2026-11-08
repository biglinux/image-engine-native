//! The CPU inference core the image-engine-native models share: one matrix product
//! kernel for every linear layer and convolution (`gemm`), 3x3 convolutions
//! by Winograd's minimal filtering on top of it (`winograd`), a pool of worker
//! threads (`pool`), the weights files the models' export tools write
//! (`weight_file`), the elementwise functions several networks use (`math`),
//! attention over a Swin window (`attention`), and the choice of SIMD
//! instructions ([`simd_tier`]).
//!
//! Each model's crate writes its network out layer by layer on top of these.

/// `a * b + c` on 8 lanes: one fused instruction with FMA, a multiply and an
/// add without.
#[cfg(target_arch = "x86_64")]
#[macro_export]
macro_rules! madd8 {
    (fused, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c)
    };
    (split, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_add_ps(std::arch::x86_64::_mm256_mul_ps($a, $b), $c)
    };
}

/// Define `$name`, running `$body` compiled for AVX2 or AVX when the CPU has
/// them and as plain code otherwise. The body fuses nothing, so all three
/// give the same bits.
#[macro_export]
macro_rules! dispatch {
    ($(#[$doc:meta])* $vis:vis $name:ident($($arg:ident: $ty:ty),*) $body:block) => {
        $(#[$doc])*
        $vis fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                fn avx2($($arg: $ty),*) $body
                #[target_feature(enable = "avx")]
                fn avx($($arg: $ty),*) $body
                match $crate::simd_tier() {
                    // SAFETY: the CPU supports AVX2 and FMA.
                    $crate::TIER_AVX2 => return unsafe { avx2($($arg),*) },
                    // SAFETY: the CPU supports AVX.
                    $crate::TIER_AVX => return unsafe { avx($($arg),*) },
                    _ => {}
                }
            }
            $body
        }
    };
}

pub mod attention;
pub mod gemm;
pub mod half;
pub mod math;
pub mod pool;
pub mod weight_file;
pub mod winograd;

use std::sync::OnceLock;

/// Why a weights file could not be loaded or an image not processed.
#[derive(Debug)]
pub enum Error {
    Io(String, std::io::Error),
    Weights(String),
    Input(String),
    /// The progress callback asked the run to stop.
    Cancelled,
    /// The GPU could not be used or failed during a run.
    Gpu(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(path, e) => write!(f, "{path}: {e}"),
            Error::Weights(msg) => write!(f, "weights: {msg}"),
            Error::Input(msg) => write!(f, "input: {msg}"),
            Error::Cancelled => write!(f, "cancelled"),
            Error::Gpu(msg) => write!(f, "GPU: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// [`simd_tier`] of CPUs with AVX, as Sandy Bridge and later.
pub const TIER_AVX: u8 = 1;
/// [`simd_tier`] of CPUs with AVX2, FMA and F16C, as Haswell, Zen and later.
pub const TIER_AVX2: u8 = 2;

/// Instruction set the kernels use: 2 = AVX2 with FMA and F16C, 1 = AVX,
/// 0 = SSE2 on x86-64 and plain Rust elsewhere. Results are the same on every
/// run within a tier; tiers 1 and 0 also give the same bits on x86-64, tier 2
/// does not, since it fuses each multiply-add.
///
/// `IMAGE_ENGINE_NATIVE_TIER` caps it, to run an older CPU's code on a newer one.
pub fn simd_tier() -> u8 {
    static TIER: OnceLock<u8> = OnceLock::new();
    *TIER.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let detected = if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            TIER_AVX2
        } else if std::is_x86_feature_detected!("avx") {
            TIER_AVX
        } else {
            0
        };
        #[cfg(not(target_arch = "x86_64"))]
        let detected = 0;
        std::env::var("IMAGE_ENGINE_NATIVE_TIER")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .map_or(detected, |cap| cap.min(detected))
    })
}

/// The number of threads to run on: `threads`, or one per CPU the process
/// may use when it is 0.
pub fn threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    }
}
