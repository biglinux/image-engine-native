//! LaMa image inpainting on the CPU or a Vulkan GPU, in plain Rust.
//!
//! ```no_run
//! use std::path::Path;
//! use lama_native::Model;
//!
//! let mut model = Model::load(Path::new("weights/big-lama-f16.weights"), 0)?;
//! let (width, height) = (512, 512);
//! let image = vec![0u8; width * height * 3];
//! let mut mask = vec![0u8; width * height];
//! mask[100 * width + 100] = 255; // fill this pixel
//! let filled: Vec<u8> = model.run(&image, &mask, width, height)?;
//! # Ok::<(), lama_native::Error>(())
//! ```
//!
//! The network is written out here layer by layer rather than interpreted
//! from a graph: `network.rs` holds its convolutions and `fourier.rs` the
//! Fourier transforms of its spectral branches, on the matrix products,
//! threads and sigmoid of `native-engine`, and `gpu.rs` the same layers as
//! compute shaders, through `native-vulkan`. `tools/lama/export_weights.py`
//! turns the PyTorch checkpoint into the weights file [`Model::load`] reads.

#[macro_use]
extern crate native_engine;

mod fourier;
mod gpu;
mod network;

use std::ops::ControlFlow;
use std::path::Path;

pub use native_engine::{Error, simd_tier};
use native_engine::{gemm, math, pool, weight_file};

/// A loaded model, on a GPU or on the CPU with its threads.
pub struct Model {
    engine: Engine,
}

#[expect(clippy::large_enum_variant, reason = "one per model, made once")]
enum Engine {
    Cpu {
        network: network::Network,
        pool: pool::Pool,
    },
    Gpu(gpu::Network),
}

impl Engine {
    fn run(
        &mut self,
        image: &[f32],
        mask: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<Vec<f32>, Error> {
        match self {
            Engine::Cpu { network, pool } => network.run(pool, image, mask, width, height, step),
            Engine::Gpu(network) => network.run(image, mask, width, height, step),
        }
    }
}

impl Model {
    /// Load a weights file onto a GPU that has Vulkan with half floats, or
    /// else onto the CPU with `threads` threads, one per CPU the process may
    /// use when `threads` is 0. The GPU computes in half floats (see the
    /// README for how close that stays to the CPU's floats).
    ///
    /// `IMAGE_ENGINE_NATIVE_GPU=0` keeps the model on the CPU, and
    /// `IMAGE_ENGINE_NATIVE_GPU=1` makes a missing or unsuitable GPU an error,
    /// [`Error::Gpu`] saying why, instead of a reason to use the CPU.
    pub fn load(path: &Path, threads: usize) -> Result<Self, Error> {
        let gpu = match std::env::var("IMAGE_ENGINE_NATIVE_GPU").as_deref() {
            Ok("0") => return Self::load_cpu(path, threads),
            Ok("1") => {
                return Ok(Self {
                    engine: Engine::Gpu(gpu::Network::load(path)?),
                });
            }
            _ => gpu::Network::load(path),
        };
        match gpu {
            Ok(network) => Ok(Self {
                engine: Engine::Gpu(network),
            }),
            Err(Error::Gpu(_)) => Self::load_cpu(path, threads),
            Err(e) => Err(e),
        }
    }

    /// [`Model::load`] on the CPU only.
    pub fn load_cpu(path: &Path, threads: usize) -> Result<Self, Error> {
        let mut file = weight_file::WeightFile::read(path, b"LAMA\0\0\0\0")?;
        let network = network::Network::load(&mut file)?;
        file.finish()?;
        let threads = native_engine::threads(threads);
        let pool =
            pool::Pool::new(threads, "lama").map_err(|e| Error::Io("worker threads".into(), e))?;
        Ok(Self {
            engine: Engine::Cpu { network, pool },
        })
    }

    /// The GPU the model runs on, or `None` on the CPU.
    pub fn gpu(&self) -> Option<&str> {
        match &self.engine {
            Engine::Cpu { .. } => None,
            Engine::Gpu(network) => Some(network.name()),
        }
    }

    /// A `width x height` 8-bit RGB image, row by row, with the pixels where
    /// `mask` (one byte per pixel) is not zero filled in from the rest; the
    /// other pixels keep their values. Both sides must be at least 9 pixels.
    pub fn run(
        &mut self,
        rgb: &[u8],
        mask: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, Error> {
        self.run_with_progress(rgb, mask, width, height, |_| ControlFlow::Continue(()))
    }

    /// [`Model::run`], calling `progress` with the share of the work done,
    /// from 0 to 1, after each of the network's 26 steps. Breaking from
    /// `progress` stops the run with [`Error::Cancelled`].
    pub fn run_with_progress(
        &mut self,
        rgb: &[u8],
        mask: &[u8],
        width: usize,
        height: usize,
        mut progress: impl FnMut(f32) -> ControlFlow<()>,
    ) -> Result<Vec<u8>, Error> {
        check_size(rgb.len(), mask.len(), width, height)?;
        let image: Vec<f32> = rgb.iter().map(|&v| f32::from(v) / 255.0).collect();
        let holes: Vec<f32> = mask.iter().map(|&m| f32::from(u8::from(m != 0))).collect();
        let mut done = 0;
        let predicted = self.engine.run(&image, &holes, width, height, &mut || {
            done += 1;
            progress(done as f32 / network::STEPS as f32)
        })?;
        let mut out = rgb.to_vec();
        for (p, &m) in mask.iter().enumerate() {
            if m != 0 {
                for c in 0..3 {
                    out[p * 3 + c] = (predicted[p * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        Ok(out)
    }

    /// The network's prediction for every pixel, as is: an RGB image in
    /// 0..=1 the size of `image` (values in 0..=1), given `mask` (1 to fill,
    /// 0 to keep, one value per pixel). [`Model::run`] keeps it only where
    /// the mask is set.
    pub fn run_float(
        &mut self,
        image: &[f32],
        mask: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, Error> {
        check_size(image.len(), mask.len(), width, height)?;
        self.engine.run(
            image,
            mask,
            width,
            height,
            &mut || ControlFlow::Continue(()),
        )
    }
}

fn check_size(image: usize, mask: usize, width: usize, height: usize) -> Result<(), Error> {
    if width < network::MIN_SIDE || height < network::MIN_SIDE {
        return Err(Error::Input(format!(
            "{width} x {height} image: sides of at least {} pixels",
            network::MIN_SIDE
        )));
    }
    let pixels = width.checked_mul(height);
    if pixels.and_then(|p| p.checked_mul(3)) != Some(image) || pixels != Some(mask) {
        return Err(Error::Input(format!(
            "{image} values and a mask of {mask} for a {width} x {height} RGB image"
        )));
    }
    Ok(())
}
