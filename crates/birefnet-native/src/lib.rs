//! BiRefNet-lite background removal on the CPU or a Vulkan GPU, in plain Rust.
//!
//! ```no_run
//! use std::path::Path;
//! use birefnet_native::Model;
//!
//! let mut model = Model::load(Path::new("weights/birefnet-lite.weights"), 0)?;
//! let rgb = vec![0u8; 1024 * 1024 * 3]; // the image, resized to 1024 x 1024
//! let logits = model.run(&rgb, 1024)?;
//! let alpha: Vec<u8> = logits.iter().map(|&l| (255.0 / (1.0 + (-l).exp())).round() as u8).collect();
//! # Ok::<(), birefnet_native::Error>(())
//! ```
//!
//! The network is written out here layer by layer rather than interpreted
//! from a graph: `swin.rs` holds the backbone, `decoder.rs` the decoder and
//! `math.rs` the window attention, on the matrix products, threads and
//! elementwise functions of `native-engine`, and `gpu.rs` the same layers as
//! compute shaders, through `native-vulkan`.
//! `tools/birefnet/export_weights.py` turns the ONNX model into the weights
//! file [`Model::load`] reads.

#[macro_use]
extern crate native_engine;

mod decoder;
mod gpu;
mod math;
mod swin;

use std::path::Path;

pub use native_engine::{Error, simd_tier};
use native_engine::{gemm, half, pool, weight_file};

/// ImageNet statistics the model was trained with.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A loaded model, on a GPU or on the CPU with its threads.
pub struct Model {
    engine: Engine,
}

#[expect(clippy::large_enum_variant, reason = "one per model, made once")]
enum Engine {
    Cpu {
        backbone: swin::Backbone,
        decoder: decoder::Decoder,
        pool: pool::Pool,
    },
    Gpu(gpu::Network),
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
        let mut file = weight_file::WeightFile::read(path, b"BIREFNET")?;
        let backbone = swin::Backbone::load(&mut file)?;
        let decoder = decoder::Decoder::load(&mut file)?;
        file.finish()?;
        let threads = native_engine::threads(threads);
        Ok(Self {
            engine: Engine::Cpu {
                backbone,
                decoder,
                pool: pool::Pool::new(threads, "birefnet")
                    .map_err(|e| Error::Io("worker threads".into(), e))?,
            },
        })
    }

    /// The GPU the model runs on, or `None` on the CPU.
    pub fn gpu(&self) -> Option<&str> {
        match &self.engine {
            Engine::Cpu { .. } => None,
            Engine::Gpu(network) => Some(network.name()),
        }
    }

    /// The foreground logit of every pixel of a `side x side` 8-bit RGB image,
    /// row by row; the sigmoid of a logit is the pixel's opacity. The model
    /// was trained at 1024; `side` must be a multiple of 64, so that the
    /// half-size pass the network also makes is a multiple of 32.
    pub fn run(&mut self, rgb: &[u8], side: usize) -> Result<Vec<f32>, Error> {
        check_size(rgb.len(), side)?;
        // Each colour's 256 values, normalised once.
        let table: Vec<[f32; 3]> = (0..=255u8)
            .map(|v| std::array::from_fn(|c| (f32::from(v) / 255.0 - MEAN[c]) / STD[c]))
            .collect();
        if let Engine::Gpu(network) = &mut self.engine {
            let halves: Vec<[[u8; 2]; 3]> = table
                .iter()
                .map(|t| t.map(|v| half::f32_to_half(v).to_le_bytes()))
                .collect();
            return network.run(side, &|dst| {
                for (d, p) in dst
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(rgb.as_chunks::<3>().0)
                {
                    *d = std::array::from_fn(|c| halves[usize::from(p[c])][c]);
                }
            });
        }
        let image: Vec<f32> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| std::array::from_fn::<f32, 3, _>(|c| table[usize::from(p[c])][c]))
            .collect();
        self.run_normalized(&image, side)
    }

    /// [`Model::run`] for an image already normalised with the ImageNet mean
    /// and deviation, NHWC.
    pub fn run_normalized(&mut self, image: &[f32], side: usize) -> Result<Vec<f32>, Error> {
        check_size(image.len(), side)?;
        let (backbone, decoder, pool) = match &mut self.engine {
            Engine::Gpu(network) => {
                return network.run(side, &|dst| half::floats_to_halves(image, dst));
            }
            Engine::Cpu {
                backbone,
                decoder,
                pool,
            } => (backbone, decoder, pool),
        };
        let half_side = side / 2;
        let mut half = vec![0.0; half_side * half_side * 3];
        decoder::resize(
            pool,
            image,
            (side, side),
            3,
            &mut half,
            (half_side, half_side),
            3,
            false,
        );
        let full_features = backbone.run(pool, image, side);
        let half_features = backbone.run(pool, &half, half_side);
        drop(half);
        Ok(decoder.run(pool, image, side, full_features, half_features))
    }
}

fn check_size(len: usize, side: usize) -> Result<(), Error> {
    if side == 0 || !side.is_multiple_of(64) {
        return Err(Error::Input(format!(
            "side {side} is not a positive multiple of 64"
        )));
    }
    if side.checked_mul(side).and_then(|p| p.checked_mul(3)) != Some(len) {
        return Err(Error::Input(format!(
            "{len} values for a {side} x {side} RGB image"
        )));
    }
    Ok(())
}
