# lama-native

LaMa image inpainting on the CPU or a Vulkan GPU, in plain Rust: an image and
a mask go in, the image comes out with the masked area filled from its
surroundings, with no PyTorch, ONNX Runtime or other inference library
underneath.

- **No dependencies.** The library and its bench tool use only `std`; the
  GPU is reached through the system's `libvulkan.so.1`, opened at run time
  when it is there.
- **The original model's results.** Checked against PyTorch running LaMa's
  own generator code and weights: within 7e-5 on 20 CC0 photos, the same
  8-bit image but for rounding ties.
- **Faster and lighter than ONNX Runtime.** A 512 x 512 image takes 0.8 s and
  0.25 GB on a six-core laptop, where ONNX Runtime takes 1.8 s and 0.9 GB.
  See [Performance](#performance).

The model is Big LaMa, the authors' released generator: 51 million weights,
102 MB as half floats.

## Build

You need Rust 1.95 or newer and `glslc`, from shaderc, which compiles the
GPU shaders. From the repository's root:

```bash
cargo build --release --locked -p lama-native
```

The weights are not in the repository. They keep LaMa's Apache-2.0 licence
and are converted from the authors' checkpoint into a weights file under the
ignored `weights/` directory. The conversion needs Python with `numpy`, but not
PyTorch: the checkpoint is read with an unpickler that builds arrays and plain
containers only, and turns every other class the file names into an inert
placeholder.

```bash
curl -L -o big-lama.zip \
  https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip
mkdir -p weights
tools/lama/export_weights.py --half big-lama.zip weights/big-lama-f16.weights
```

Without `--half`, the export keeps float32 weights, which the tests use.

The archive's SHA-256 is
`f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`.

## Use

```rust,no_run
use std::path::Path;
use lama_native::Model;

fn main() -> Result<(), lama_native::Error> {
    // Load once; 0 threads means one per CPU the process may use.
    let mut model = Model::load(Path::new("weights/big-lama-f16.weights"), 0)?;
    let (width, height) = (512, 512);
    let image = vec![0u8; width * height * 3];
    let mut mask = vec![0u8; width * height];
    mask[200 * width..300 * width].fill(255); // fill these rows
    let filled = model.run(&image, &mask, width, height)?;
    assert_eq!(filled.len(), image.len());
    Ok(())
}
```

`Model::load` puts the model on a GPU that has Vulkan 1.2 with half floats (a
discrete one first), and on the CPU when there is none or it cannot be used;
`Model::gpu` names the GPU in use. `IMAGE_ENGINE_NATIVE_GPU=0` keeps the model
on the CPU, and `IMAGE_ENGINE_NATIVE_GPU=1` turns a missing GPU into an error
that says why. `Model::load_cpu` always uses the CPU.

The image is 8-bit RGB, row by row, and the mask one byte per pixel, not zero
where the image is to be filled; the other pixels come back unchanged. Both
sides must be at least 9 pixels. The whole image runs at once: the network's Fourier units see all of it, so
tiles would change the result, and memory grows with the image, 0.52 GB at
1024 x 1024. For a large photo, run a crop around the area to fill, scaled
down if it is large itself. `Model::run_with_progress` reports the share of
the work done after each of the network's 26 steps and stops with
`Error::Cancelled` when the callback returns `ControlFlow::Break`.
`Model::run_float` returns the network's prediction for every pixel.

`lama-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG]
IMAGE.rgb...` times the model on raw RGB files, each with its IMAGE.mask, and,
with `--tag`, writes the predictions.

## How it works

The network is written out layer by layer (`src/network.rs`): a 7x7
convolution from the masked image and its mask to 64 channels, three stride-2
convolutions to 512 channels at an eighth of the size, 18 residual blocks of
fast Fourier convolutions, three stride-2 transposed convolutions back to 64
channels, and a 7x7 convolution to RGB through a sigmoid.

- Every convolution runs on one matrix product kernel (the engine's
  `gemm.rs`): six rows by sixteen columns of vector accumulators, weights
  packed once into 16-column panels, and the input read through row pointers,
  which lead past the edge to the mirrored pixel (LaMa pads by reflection) or
  to zeros, so no convolution copies its input into columns.
- The 3x3 convolutions at the eighth scale use Winograd's minimal filtering
  (the engine's `winograd.rs`): F(4x4, 3x3), 36 products per 4x4 block of
  output pixels instead of 144, from the 128 local channels, and F(2x2, 3x3),
  16 per 2x2 block instead of 36, from all 512.
- The export folds every batch norm into the convolution before it.
- At the eighth scale a pixel holds 128 local and 384 global channels. Each
  fast Fourier convolution's local output is one 3x3 convolution of all 512
  channels, the reference's two local and global convolutions joined; its
  global output adds a 3x3 convolution of the local channels to a spectral
  transform of the global ones.
- The spectral transform's 2-D Fourier transforms (`src/fourier.rs`) are
  mixed-radix fast Fourier transforms over whole pixels, each step on all
  channels of a pixel at once; two real rows share one complex transform. A
  task copies a column of the spectrum in, so that its transform runs in
  cache.
- Each transposed convolution runs as four products, one per output phase
  (even or odd row and column), each reading only the input pixels and kernel
  taps that phase meets.

Work is split between threads by pixels, rows or columns; no sum is ever
split, so the result does not depend on the number of threads. The kernels
come in three tiers, chosen when the process starts: AVX2 with FMA and F16C
(tier 2: Haswell, Zen and later), AVX (tier 1: Sandy and Ivy Bridge,
Bulldozer), and SSE2 for any other x86-64 CPU (tier 0), with plain Rust on
other architectures. `IMAGE_ENGINE_NATIVE_TIER=N` caps the tier.

On a GPU (`src/gpu.rs`, `shaders/`) the same network runs in half floats, each
of its 26 steps one submission. The convolutions pad by reflection inside
their kernel and sum the products of each block of 16 input channels (8 in
AMD's 3x3 kernel and in the last layer) and one kernel row (all three rows in
the shared-memory kernel below) in half floats and the blocks in floats; the
transposed convolutions run as their four phases, each a matrix product that
gathers the input pixels its taps read. The Fourier units transform rows and
columns in floats, with the twiddles in a table the host computes: a line of
up to 512 points whose only prime factors are 2, 3, 5 and 7 goes through a
mixed-radix fast transform in one workgroup's shared memory, any other by
direct sums, 16 outputs per invocation, whose cost grows faster than the
convolutions' with the image. The kernels read weights and twiddles at an
index the same in every lane, which AMD GPUs serve from their scalar cache;
there the 3x3 convolutions spread output channels over the lanes, two per lane
and 128 per wave, each lane keeping its weights in registers for 16 pixels
whose input values every lane shares; on other GPUs the 3x3 convolutions of
stride 1 and 2, most of the work, and the matrix products stage their weights
in shared memory instead, the convolutions with each input pixel's channels
side by side, eight to a read, and the first 7x7 convolution computes four
pixels per lane so that each weight read serves four products. The last layer,
with 3 output channels, keeps each input pixel's 8 channels together in shared
memory, so that one read and one weight load feed 16 products.

## Performance

On a Ryzen 5 5600H (Zen 3, 6 cores, 12 threads, AVX2), 6 threads, half-float
weights, median over the 20 photos, peak resident memory of the whole
process:

| Image | Engine | Time | Peak memory |
|---|---|---:|---:|
| 500 x 333 | lama-native | 0.44 s | 0.20 GB |
| 512 x 512 | lama-native | 0.66 s | 0.25 GB |
| 512 x 512 | ONNX Runtime 1.31, Carve/LaMa-ONNX | 1.84 s | 0.92 GB |
| 1024 x 1024 | lama-native | 2.65 s | 0.54 GB |

A 512 x 512 run costs 0.45 TFLOP as direct convolutions, 0.26 TFLOP with the
Winograd ones. All 12 threads take 2 % less. On a Core i3-2375M (Sandy Bridge,
2 cores and 4 threads at 1.5 GHz, AVX without AVX2), 512 x 512 takes 10.0 s,
and 10.3 s with 2 threads.

On GPUs, in half floats, median over the same photos:

| Image | Radeon Vega 7 (the 5600H's, RADV) | GeForce GTX 1660 Ti (NVIDIA 615) |
|---|---:|---:|
| 500 x 333 | 0.21 s | 0.07 s |
| 512 x 512 | 0.30 s | 0.08 s |
| 1024 x 1024 | 1.17 s | 0.32 s |

against 0.44 s, 0.66 s and 2.65 s on the 5600H's CPU. Where a side's eighth
has a prime factor above 7, the Fourier sums along it grow with the cube of
the side, faster than the convolutions, and the GPU's lead narrows on larger
images. A 1920 x 1080 image, whose eighth
is 135 x 240, takes 2.8 s on the Vega 7 and 0.88 s on the GTX 1660 Ti, against
5.8 s on the CPU.

## Precision

Against PyTorch's float32 predictions on the 20 photos with holes, at
512 x 512 and at 500 x 333 (which the network pads to 504 x 336, an odd
number of Fourier columns):

| Weights | Largest difference | 8-bit values that differ | PSNR inside the holes |
|---|---:|---:|---:|
| float32 | 6.5e-5 | 0.005 % (rounding ties) | identical |
| float16 (`--half`) | 0.10 | 2.7 % | 21.455 dB, against 21.452 |

Half floats change some invented detail, by up to 25 8-bit steps in a few
pixels, without changing how well the holes are filled: no photo moved by more
than 0.04 dB. They halve the file and, at tier 2, the weights in memory, so
they are the ones to ship.

The GPU computes in half floats throughout. LaMa invents what fills the
holes, and small differences grow into other plausible fills, so against the
CPU engine the holes differ more (8-bit PSNR 35 to 55 dB on the 20 photos at
512 x 512); against the original photos the PSNR inside the holes is 21.453
dB, against 21.455 on the CPU, no photo moving by more than 0.15 dB. At
1024 x 1024 one photo moves by 0.42 dB on the Vega 7 and 0.26 dB on the
GTX 1660 Ti, the mean by 0.03 and 0.01 dB.

## Checking results

`tools/lama/evaluate.py prepare` crops and resizes photos and draws holes in
them (strokes and a box);
`tools/lama/torch_reference.py` writes PyTorch's predictions with LaMa's own
generator code; `tools/lama/evaluate.py compare` prints the PSNR inside the
holes and the differences between two sets of predictions. With a directory of
`NAME.rgb` inputs, their `NAME.mask` and `NAME.torch.out`, the ignored tests
check float32 weights against them and the progress reports:

```bash
cargo test --release
LAMA_NATIVE_WEIGHTS=$PWD/weights/big-lama.weights \
  LAMA_NATIVE_REFERENCE=DIR LAMA_NATIVE_SIZE=512x512 \
  cargo test --release --test reference -- --ignored
```

The ignored `gpu` test needs a GPU and checks it against the CPU engine on
the same photos by the PSNR inside the holes against each photo: within 0.5
dB on each, and 0.05 dB on their mean. The ignored `fft_matches_direct_sums`
unit test checks the fast transforms against the direct sums on the GPU, on
lines of 56 and 120 points (radices 4, 2, 3, 5 and 7):

```bash
cargo test --release -p lama-native --lib -- --ignored
```

## Credits and license

LaMa is the work of Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin,
Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith
Goka, Kiwoong Park and Victor Lempitsky, published under Apache-2.0.
[NOTICE](NOTICE) lists the paper and the code this builds on and carries
LaMa's licence, which applies to the weights.

The Rust implementation is by Bruno Gonçalves
([BigLinux](https://www.biglinux.com.br)) and is licensed MIT
([LICENSE](../../LICENSE)).