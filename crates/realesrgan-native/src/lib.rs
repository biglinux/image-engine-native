//! Real-ESRGAN image upscaling on the CPU or a Vulkan GPU, in plain Rust.
//!
//! ```no_run
//! use std::path::Path;
//! use realesrgan_native::Model;
//!
//! let mut model = Model::load(Path::new("weights/realesr-general-x4v3-dn50.weights"), 0)?;
//! let (width, height) = (320, 240);
//! let small = vec![0u8; width * height * 3];
//! let large: Vec<u8> = model.run(&small, width, height)?; // 1280 x 960
//! # Ok::<(), realesrgan_native::Error>(())
//! ```
//!
//! The network is written out here layer by layer rather than interpreted
//! from a graph: `network.rs` holds its convolutions, on the matrix products
//! and threads of `native-engine`, and `gpu.rs` the same layers as compute
//! shaders, through `native-vulkan`. `tools/realesrgan/export_weights.py`
//! turns the PyTorch checkpoint into the weights file [`Model::load`] reads.

#[macro_use]
extern crate native_engine;

mod gpu;
mod network;

use std::ops::ControlFlow;
use std::path::Path;

pub use network::SCALE;

pub use native_engine::{Error, simd_tier};
use native_engine::{gemm, half, pool, weight_file};

/// Largest tile side, in input pixels, [`Model::run`] uses on the CPU.
pub const TILE: usize = 512;
/// Largest tile side on a GPU, where the overlap of smaller tiles costs more
/// than their memory saves: 0.32 GB for a whole tile.
pub const GPU_TILE: usize = 768;
/// Context around each tile: the 34 3x3 convolutions see 34 pixels away, so
/// tiles with this margin give the whole image's result, to within rounding.
pub const MARGIN: usize = network::REACH;

/// A loaded model, on a GPU or on the CPU with its threads.
pub struct Model {
    engine: Engine,
}

enum Engine {
    Cpu {
        network: network::Network,
        pool: pool::Pool,
    },
    Gpu(Box<gpu::Network>),
}

impl Engine {
    fn run(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<Output<'_>, Error> {
        Ok(match self {
            Engine::Cpu { network, pool } => {
                Output::Floats(network.run(pool, image, width, height, step)?)
            }
            Engine::Gpu(network) => Output::Halves(network.run(image, width, height, step)?),
        })
    }

    /// Side of the tiles [`Model::run`] uses.
    fn tile(&self) -> usize {
        match self {
            Engine::Cpu { .. } => TILE,
            Engine::Gpu(_) => GPU_TILE,
        }
    }
}

/// A run's output: the CPU's floats, or the GPU's half floats where the GPU
/// left them.
enum Output<'a> {
    Floats(Vec<f32>),
    Halves(&'a [[u8; 2]]),
}

impl Output<'_> {
    /// Values `start..start + dst.len()`, each passed through `store`.
    fn store<T>(&self, start: usize, dst: &mut [T], store: &impl Fn(f32) -> T) {
        match self {
            Output::Floats(v) => {
                for (d, &v) in dst.iter_mut().zip(&v[start..]) {
                    *d = store(v);
                }
            }
            Output::Halves(h) => {
                let mut floats = [0.0; 512];
                for (dst, src) in dst.chunks_mut(512).zip(h[start..].chunks(512)) {
                    let floats = &mut floats[..dst.len()];
                    half::halves_to_f32(&src[..dst.len()], floats);
                    for (d, &v) in dst.iter_mut().zip(floats.iter()) {
                        *d = store(v);
                    }
                }
            }
        }
    }
}

impl Model {
    /// Load a weights file onto a GPU that has Vulkan with half floats, or
    /// else onto the CPU with `threads` threads, one per CPU the process may
    /// use when `threads` is 0. The GPU computes in half floats, twice as
    /// fast as in single ones and with the same quality (see the README).
    ///
    /// `IMAGE_ENGINE_NATIVE_GPU=0` keeps the model on the CPU, and
    /// `IMAGE_ENGINE_NATIVE_GPU=1` makes a missing or unsuitable GPU an error,
    /// [`Error::Gpu`] saying why, instead of a reason to use the CPU.
    pub fn load(path: &Path, threads: usize) -> Result<Self, Error> {
        let gpu = match std::env::var("IMAGE_ENGINE_NATIVE_GPU").as_deref() {
            Ok("0") => return Self::load_cpu(path, threads),
            Ok("1") => {
                return Ok(Self {
                    engine: Engine::Gpu(Box::new(gpu::Network::load(path)?)),
                });
            }
            _ => gpu::Network::load(path),
        };
        match gpu {
            Ok(network) => Ok(Self {
                engine: Engine::Gpu(Box::new(network)),
            }),
            Err(Error::Gpu(_)) => Self::load_cpu(path, threads),
            Err(e) => Err(e),
        }
    }

    /// [`Model::load`] on the CPU only, whose results match the original
    /// model's to 2e-4.
    pub fn load_cpu(path: &Path, threads: usize) -> Result<Self, Error> {
        let mut file = weight_file::WeightFile::read(path, b"REALESRG")?;
        let network = network::Network::load(&mut file)?;
        file.finish()?;
        let threads = native_engine::threads(threads);
        let pool = pool::Pool::new(threads, "realesrgan")
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

    /// The `SCALE` times larger copy of a `width x height` 8-bit RGB image,
    /// row by row, in tiles of up to [`TILE`] pixels, or [`GPU_TILE`] on a
    /// GPU (see [`Model::run_float_tiled`]).
    pub fn run(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Error> {
        self.run_with_progress(rgb, width, height, |_| ControlFlow::Continue(()))
    }

    /// [`Model::run`], calling `progress` with the share of the work done,
    /// from 0 to 1, after each layer of the network: 34 times per tile.
    /// Breaking from `progress` stops the run with [`Error::Cancelled`].
    pub fn run_with_progress(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        progress: impl FnMut(f32) -> ControlFlow<()>,
    ) -> Result<Vec<u8>, Error> {
        let tile = self.engine.tile();
        self.tiled(
            rgb,
            width,
            height,
            (tile, MARGIN),
            to_float,
            to_byte,
            progress,
        )
    }

    /// [`Model::run`] on values in 0..=1 and the whole image at once,
    /// returning the network's output as is: neither clamped nor rounded.
    /// On a GPU, whose memory may be smaller, it runs in tiles of
    /// [`GPU_TILE`] with [`MARGIN`], which give the same result.
    pub fn run_float(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, Error> {
        check_size(image.len(), width, height)?;
        let mut go_on = || ControlFlow::Continue(());
        match &mut self.engine {
            Engine::Cpu { network, pool } => network.run(pool, image, width, height, &mut go_on),
            Engine::Gpu(_) => self.tiled(
                image,
                width,
                height,
                (GPU_TILE, MARGIN),
                |v| v,
                |v| v,
                |_| ControlFlow::Continue(()),
            ),
        }
    }

    /// [`Model::run_float`] a tile at a time: the image is cut into a grid of
    /// equal tiles at most `tile` input pixels on a side, each run with up to
    /// `margin` pixels of its surroundings on every side, and only the tile's
    /// own output kept. With a margin of at least [`MARGIN`] the result is the
    /// whole image's to within rounding, and bit for bit on a GPU. `progress`
    /// is as for [`Model::run_with_progress`].
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
        let (ow, oh) = (width * SCALE, height * SCALE);
        let mut out = vec![T::default(); ow * oh * 3];
        let mut region = Vec::new();
        let (cols, rows) = (width.div_ceil(tile), height.div_ceil(tile));
        let steps = (cols * rows * network::STEPS) as f32;
        let mut done = 0;
        for j in 0..rows {
            let (y0, y1) = (j * height / rows, (j + 1) * height / rows);
            for i in 0..cols {
                let (x0, x1) = (i * width / cols, (i + 1) * width / cols);
                let (rx0, ry0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
                let (rx1, ry1) = (
                    x1.saturating_add(margin).min(width),
                    y1.saturating_add(margin).min(height),
                );
                let rw = rx1 - rx0;
                region.clear();
                for y in ry0..ry1 {
                    let row = &image[(y * width + rx0) * 3..][..rw * 3];
                    region.extend(row.iter().map(|&v| load(v)));
                }
                let result = self.engine.run(&region, rw, ry1 - ry0, &mut || {
                    done += 1;
                    progress(done as f32 / steps)
                })?;
                let (rw, cw) = (rw * SCALE, (x1 - x0) * SCALE);
                for y in y0 * SCALE..y1 * SCALE {
                    let start = ((y - ry0 * SCALE) * rw + (x0 - rx0) * SCALE) * 3;
                    let dst = &mut out[(y * ow + x0 * SCALE) * 3..][..cw * 3];
                    result.store(start, dst, &store);
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
    // The output, SCALE x SCALE times larger, must have a size too.
    let output = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(3 * SCALE * SCALE));
    if output.is_none() || width * height * 3 != len {
        return Err(Error::Input(format!(
            "{len} values for a {width} x {height} RGB image"
        )));
    }
    Ok(())
}
