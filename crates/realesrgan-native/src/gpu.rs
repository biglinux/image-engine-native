//! The network of `network.rs` on a Vulkan GPU, in half floats.
//!
//! Each layer is one dispatch of a 3x3 convolution kernel and its own
//! submission, which keeps every submission short and lets a run stop
//! between layers; the last one adds the pixel shuffle. Activations stay on
//! the GPU, NHWC with 64 channels (the last layer's 48 padded), in two maps
//! that take turns as input and output. The input image is read from memory
//! the CPU maps; the output, in half floats, is written on the GPU and copied
//! back in one transfer, since a discrete GPU's scattered writes across the
//! bus to the CPU's memory took longer than the whole network.
//!
//! The convolution kernel depends on the GPU: AMD's read the weights through
//! the scalar cache (`conv_scalar`), others stage them in shared memory
//! (`conv_shared`), each the faster on its kind by two to three times.

use std::ops::ControlFlow;
use std::path::Path;

use native_engine::half::{f32_to_half, floats_to_halves};
use native_engine::weight_file::WeightFile;
use native_vulkan::{Buffer, Gpu, Kernel, Words};

use crate::Error;
use crate::network::{FEATURES, SCALE, STEPS};

const CONV_SCALAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_scalar.spv"));
const CONV_SHARED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_shared.spv"));
const CONV_SHARED_PX4: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_shared_px4.spv"));
const SHUFFLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shuffle.spv"));
/// What `conv_shared` needs of a workgroup.
const SHARED_BYTES: u32 = 31_488;
const AMD: u32 = 0x1002;
const INTEL: u32 = 0x8086;
/// Bytes of the push constants: width, height, input channels and the
/// layer's weight, bias and slope offsets in half floats (-1: no slope).
const PUSH: u32 = 24;
/// The last layer's 3 x SCALE x SCALE channels.
const LAST: usize = 3 * SCALE * SCALE;

/// Offsets of a layer's parameters, in half floats.
struct Layer {
    cin: usize,
    weights: usize,
    bias: usize,
    slope: Option<usize>,
}

/// Buffers for images of up to `pixels` pixels.
struct Maps {
    pixels: usize,
    image: Buffer,
    a: Buffer,
    b: Buffer,
    out: Buffer,
    readback: Buffer,
}

impl Maps {
    fn new(gpu: &Gpu, pixels: usize) -> Result<Self, String> {
        let half = |n: usize| pixels * n * 2;
        Ok(Self {
            pixels,
            image: gpu.host_buffer(half(3), false)?,
            a: gpu.buffer(half(FEATURES))?,
            b: gpu.buffer(half(FEATURES))?,
            out: gpu.buffer(pixels * LAST * 2)?,
            readback: gpu.host_buffer(pixels * LAST * 2, true)?,
        })
    }
}

pub struct Network {
    gpu: Gpu,
    conv: Kernel,
    /// Pixels a workgroup of `conv` covers, across and down.
    block: (usize, usize),
    shuffle: Kernel,
    params: Buffer,
    layers: Vec<Layer>,
    maps: Option<Maps>,
}

impl Network {
    /// Open a GPU and load the weights in `path` onto it. The GPU comes
    /// first: without one, the weights are read only for the CPU.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let mut gpu = Gpu::open().map_err(Error::Gpu)?;
        let mut file = WeightFile::read(path, b"REALESRG")?;
        let (words, layers) = params(&mut file)?;
        file.finish()?;
        let shared = gpu.vendor() != AMD && gpu.max_shared_memory() >= SHARED_BYTES;
        // Intel's compiler spills the sums of eight pixels out of registers.
        let (spirv, block) = if shared && gpu.vendor() == INTEL {
            (CONV_SHARED_PX4, (8, 16))
        } else if shared {
            (CONV_SHARED, (8, 32))
        } else {
            (CONV_SCALAR, (8, 8))
        };
        let conv = gpu.kernel(spirv, 3, PUSH).map_err(Error::Gpu)?;
        let shuffle = gpu.kernel(SHUFFLE, 3, PUSH).map_err(Error::Gpu)?;
        let params = gpu.upload(&words).map_err(Error::Gpu)?;
        Ok(Self {
            gpu,
            conv,
            block,
            shuffle,
            params,
            layers,
            maps: None,
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        self.gpu.name()
    }

    /// As `network::Network::run`, on the GPU, but the output stays in half
    /// floats (little-endian) in memory the model keeps:
    /// `half::half_le_to_f32` turns one into a float.
    pub fn run(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
        step: &mut dyn FnMut() -> ControlFlow<()>,
    ) -> Result<&[[u8; 2]], Error> {
        let pixels = width * height;
        if self.maps.as_ref().is_none_or(|m| m.pixels < pixels) {
            // Free the smaller maps before making the larger ones.
            self.maps = None;
            self.maps = Some(Maps::new(&self.gpu, pixels).map_err(Error::Gpu)?);
        }
        let Self {
            gpu,
            conv,
            block,
            shuffle,
            params,
            layers,
            maps,
        } = self;
        let maps = maps.as_mut().expect("made above");
        let halves = &mut maps.image.bytes_mut().as_chunks_mut::<2>().0[..image.len()];
        floats_to_halves(image, halves);
        let size = |n: usize| {
            u32::try_from(n).map_err(|_| Error::Input(format!("{width} x {height} image")))
        };
        let groups = [
            size(width.div_ceil(block.0))?,
            size(height.div_ceil(block.1))?,
            1,
        ];
        let shuffle_groups = [size((pixels * SCALE * SCALE).div_ceil(256))?, 1, 1];
        // One submission per layer, each queued before the previous is
        // waited for, so that the progress reports after every layer cost
        // the GPU no idle time.
        let mut steps = gpu.steps();
        for (i, layer) in layers.iter().enumerate() {
            // Layer i writes `a` when i is even, `b` when odd: the last, odd,
            // leaves its channels in `b`.
            let (src, dst) = match i {
                0 => (&maps.image, &maps.a),
                _ if i % 2 == 1 => (&maps.a, &maps.b),
                _ => (&maps.b, &maps.a),
            };
            let push = constants([
                width,
                height,
                layer.cin,
                layer.weights,
                layer.bias,
                layer.slope.unwrap_or(usize::MAX),
            ]);
            let finished = steps
                .submit(|c| {
                    c.dispatch(conv, &[src, dst, params], &push, groups)?;
                    if i == STEPS - 1 {
                        c.barrier();
                        c.dispatch(
                            shuffle,
                            &[&maps.b, &maps.image, &maps.out],
                            &push,
                            shuffle_groups,
                        )?;
                        c.barrier();
                        c.copy(&maps.out, &maps.readback, pixels * LAST * 2)?;
                    }
                    Ok(())
                })
                .map_err(Error::Gpu)?;
            if finished && step().is_break() {
                return Err(Error::Cancelled);
            }
        }
        if steps.finish().map_err(Error::Gpu)? && step().is_break() {
            return Err(Error::Cancelled);
        }
        Ok(maps.readback.bytes()[..pixels * LAST * 2]
            .as_chunks::<2>()
            .0)
    }
}

/// Push constants from the values the shaders read as `int`s; `usize::MAX`
/// stands for -1.
fn constants(values: [usize; 6]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| if v == usize::MAX { -1 } else { v as i32 }.to_le_bytes())
        .collect()
}

/// Every layer's parameters as half floats, in the layout the kernels read:
/// weights `[tap][cin][64]`, then the bias and the PReLU slopes, 64 each.
fn params(file: &mut WeightFile) -> Result<(Words, Vec<Layer>), Error> {
    let mut all = Words::default();
    let mut layers = Vec::with_capacity(STEPS);
    let put = |all: &mut Words, values: &[f32], width: usize| {
        let halves = values.iter().map(|&v| f32_to_half(v)).collect();
        all.push(halves, width, FEATURES)
    };
    for i in 0..STEPS {
        let cin = if i == 0 { 3 } else { FEATURES };
        let last = i == STEPS - 1;
        let cout = if last { LAST } else { FEATURES };
        let name = format!("body.{}", 2 * i);
        let w = file.take(&format!("{name}.weight"), &[9 * cin, cout])?;
        let b = file.take(&format!("{name}.bias"), &[cout])?;
        let weights = put(&mut all, &w, cout);
        let bias = put(&mut all, &b, cout);
        let slope = if last {
            None
        } else {
            let s = file.take(&format!("body.{}.weight", 2 * i + 1), &[FEATURES])?;
            Some(put(&mut all, &s, FEATURES))
        };
        layers.push(Layer {
            cin,
            weights,
            bias,
            slope,
        });
    }
    Ok((all, layers))
}
