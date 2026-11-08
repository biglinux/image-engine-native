//! SCUNet image denoising on the CPU or a Vulkan GPU, in plain Rust.
//!
//! ```no_run
//! use std::path::Path;
//! use scunet_native::Model;
//!
//! let mut model = Model::load(Path::new("weights/scunet-color-real-psnr.weights"), 0)?;
//! let (width, height) = (640, 480);
//! let noisy = vec![0u8; width * height * 3];
//! let clean: Vec<u8> = model.run(&noisy, width, height)?;
//! # Ok::<(), scunet_native::Error>(())
//! ```
//!
//! The network is written out here layer by layer rather than interpreted
//! from a graph: `network.rs` holds the U-Net and its convolutions, `swin.rs`
//! the window attention blocks and `math.rs` the attention itself, on the
//! matrix products, threads and elementwise functions of `native-engine`,
//! and `gpu.rs` the same layers as compute shaders, through `native-vulkan`.
//! `tools/scunet/export_weights.py` turns the PyTorch checkpoint into the
//! weights file [`Model::load`] reads.

mod gpu;
mod math;
mod network;
mod swin;

use std::ops::ControlFlow;
use std::path::Path;

pub use native_engine::{Error, simd_tier};
use native_engine::{gemm, pool, weight_file};

/// Largest tile side [`Model::run`] uses: tiles with their margins fit
/// 1024 x 1024, about 0.8 GB at the peak.
pub const TILE: usize = 896;
/// Context around each tile. 64 pixels keep tiled output within a few 8-bit
/// steps of the whole image's, at the same PSNR; 32 leave visible seams.
pub const MARGIN: usize = 64;

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
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
        spare: &mut network::Spare,
    ) -> Result<Vec<f32>, Error> {
        match self {
            Engine::Cpu { network, pool } => network.run(pool, image, width, height, step, spare),
            Engine::Gpu(network) => network.run(image, width, height, step),
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

    /// [`Model::load`] on the CPU only. Its distance from the original model,
    /// with float and with half-float weights, is in the README.
    pub fn load_cpu(path: &Path, threads: usize) -> Result<Self, Error> {
        let mut file = weight_file::WeightFile::read(path, b"SCUNET\0\0")?;
        let network = network::Network::load(&mut file)?;
        file.finish()?;
        let threads = native_engine::threads(threads);
        let pool = pool::Pool::new(threads, "scunet")
            .map_err(|e| Error::Io("worker threads".into(), e))?;
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

    /// The denoised copy of a `width x height` 8-bit RGB image, row by row,
    /// in tiles of up to [`TILE`] pixels with [`MARGIN`] pixels of context
    /// (see [`Model::run_float_tiled`]).
    pub fn run(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Error> {
        self.run_with_progress(rgb, width, height, |_| ControlFlow::Continue(()))
    }

    /// [`Model::run`], calling `progress` with the share of the work done,
    /// from 0 to 1, after each block of the network: a few dozen times per
    /// tile, every second or so on a slow CPU. Breaking from `progress` stops
    /// the run with [`Error::Cancelled`].
    pub fn run_with_progress(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        progress: impl FnMut(f32) -> ControlFlow<()>,
    ) -> Result<Vec<u8>, Error> {
        self.tiled(
            rgb,
            width,
            height,
            (TILE, MARGIN),
            to_float,
            to_byte,
            progress,
        )
    }

    /// [`Model::run`] on values in 0..=1, returning the network's output as
    /// is: neither clamped nor rounded.
    pub fn run_float(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, Error> {
        check_size(image.len(), width, height)?;
        self.engine.run(
            image,
            width,
            height,
            &mut || ControlFlow::Continue(()),
            &mut network::Spare::default(),
        )
    }

    /// [`Model::run_float`] a tile at a time: the image is cut into a grid of
    /// equal tiles at most `tile` pixels on a side, each run with up to
    /// `margin` pixels of its surroundings on every side, and only the tile
    /// kept. Memory then grows with the tile instead of the image, and time
    /// with the margins run twice. An image that fits one tile runs whole.
    /// `progress` is as for [`Model::run_with_progress`].
    pub fn run_float_tiled(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
        tile: usize,
        margin: usize,
        progress: impl FnMut(f32) -> ControlFlow<()>,
    ) -> Result<Vec<f32>, Error> {
        self.tiled(image, width, height, (tile, margin), |v| v, |v| v, progress)
    }

    /// [`Model::run_float_tiled`] on pixel values of any type: `load` turns
    /// one into the network's input and `store` an output into one, a tile
    /// at a time, so that no float copy of the whole image is ever made.
    #[expect(
        clippy::too_many_arguments,
        reason = "the tiling and the pixel conversions"
    )]
    fn tiled<T: Copy + Default>(
        &mut self,
        image: &[T],
        width: usize,
        height: usize,
        (tile, margin): (usize, usize),
        load: impl Fn(T) -> f32,
        store: impl Fn(f32) -> T,
        mut progress: impl FnMut(f32) -> ControlFlow<()>,
    ) -> Result<Vec<T>, Error> {
        check_size(image.len(), width, height)?;
        if tile == 0 {
            return Err(Error::Input("tile size 0".into()));
        }
        let mut out = vec![T::default(); image.len()];
        let mut region = Vec::new();
        let mut spare = network::Spare::default();
        let (cols, rows) = (width.div_ceil(tile), height.div_ceil(tile));
        let steps = (cols * rows * network::STEPS) as f32;
        let mut done = 0;
        for j in 0..rows {
            let (y0, y1) = (j * height / rows, (j + 1) * height / rows);
            for i in 0..cols {
                let (x0, x1) = (i * width / cols, (i + 1) * width / cols);
                let (rx0, ry0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
                let (rx1, ry1) = ((x1 + margin).min(width), (y1 + margin).min(height));
                let rw = rx1 - rx0;
                region.clear();
                for y in ry0..ry1 {
                    let row = &image[(y * width + rx0) * 3..][..rw * 3];
                    region.extend(row.iter().map(|&v| load(v)));
                }
                let result = self.engine.run(
                    &region,
                    rw,
                    ry1 - ry0,
                    &mut || {
                        done += 1;
                        progress(done as f32 / steps)
                    },
                    &mut spare,
                )?;
                for y in y0..y1 {
                    let src = &result[((y - ry0) * rw + x0 - rx0) * 3..][..(x1 - x0) * 3];
                    let dst = &mut out[(y * width + x0) * 3..][..src.len()];
                    for (d, &v) in dst.iter_mut().zip(src) {
                        *d = store(v);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// An 8-bit value as the network's input.
fn to_float(v: u8) -> f32 {
    f32::from(v) / 255.0
}

/// An output of the network as an 8-bit value.
fn to_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn check_size(len: usize, width: usize, height: usize) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error::Input(format!("empty {width} x {height} image")));
    }
    if width.checked_mul(height).and_then(|p| p.checked_mul(3)) != Some(len) {
        return Err(Error::Input(format!(
            "{len} values for a {width} x {height} RGB image"
        )));
    }
    Ok(())
}
