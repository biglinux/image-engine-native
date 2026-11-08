# scunet-native

SCUNet image denoising on the CPU or a Vulkan GPU, in plain Rust: a noisy
photo goes in, a clean one comes out, with no PyTorch, ONNX Runtime or other
inference library underneath.

- **No dependencies.** The library and its bench tool use only `std`; the
  GPU is reached through the system's `libvulkan.so.1`, opened at run time
  when it is there.
- **The original model's results.** Checked against PyTorch running SCUNet's
  own code and weights: within 2e-6 on 20 CC0 photos, the same 8-bit image
  but for rounding ties.
- **Faster and lighter than ONNX Runtime.** A 768 x 512 photo takes 2.2 s and
  0.36 GB on a six-core laptop, where ONNX Runtime needs 6.1 s and 5.8 GiB for
  a smaller one. A 12-megapixel photo still takes a minute and a half:
  SCUNet costs 2.3 TFLOP per megapixel. See [Performance](#performance).

The weights are SCUNet's "real PSNR" color model, trained on synthetic
degradations close to what cameras produce: sensor noise, blur, JPEG
compression. It cleans noise and compression artifacts alike.

## Build

You need Rust 1.95 or newer and `glslc`, from shaderc, which compiles the
GPU shaders. From the repository's root:

```bash
cargo build --release --locked -p scunet-native
```

The weights are not in the repository. They keep SCUNet's Apache-2.0 licence
and are converted from its PyTorch checkpoint into a weights file under the
ignored `weights/` directory. The conversion needs Python with `numpy`, but not
PyTorch: the checkpoint is read with an unpickler that builds nothing but
arrays.

```bash
curl -L -o scunet_color_real_psnr.pth \
  https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth
mkdir -p weights
tools/scunet/export_weights.py scunet_color_real_psnr.pth weights/scunet-color-real-psnr.weights
tools/scunet/export_weights.py --half scunet_color_real_psnr.pth weights/scunet-color-real-psnr-f16.weights
```

The checkpoint's SHA-256 is
`fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e`.

## Use

```rust,no_run
use std::path::Path;
use scunet_native::Model;

fn main() -> Result<(), scunet_native::Error> {
    // Load once; 0 threads means one per CPU the process may use.
    let mut model = Model::load(Path::new("weights/scunet-color-real-psnr-f16.weights"), 0)?;
    let (width, height) = (640, 480);
    let noisy = vec![0u8; width * height * 3];
    let clean = model.run(&noisy, width, height)?;
    assert_eq!(clean.len(), noisy.len());
    Ok(())
}
```

`Model::load` puts the model on a GPU that has Vulkan 1.2 with half floats (a
discrete one first), and on the CPU when there is none or it cannot be used;
`Model::gpu` names the GPU in use. `IMAGE_ENGINE_NATIVE_GPU=0` keeps the model
on the CPU, and `IMAGE_ENGINE_NATIVE_GPU=1` turns a missing GPU into an error
that says why. `Model::load_cpu` always uses the CPU.

The input and output are 8-bit RGB, row by row, of any size. `Model::run`
works in tiles of up to 896 x 896 pixels, each with 64 pixels of context on
every side, so memory stays near 0.7 GB whatever the photo's size.
`Model::run_float` runs the whole image at once, as PyTorch does, on values in
0..=1, and returns the network's output unclamped; `Model::run_float_tiled`
takes another tile size.

A large photo takes minutes on a slow CPU, so `Model::run_with_progress`
reports the share of the work done after each block of the network, a few
dozen times per tile, and stops with `Error::Cancelled` when the callback
returns `ControlFlow::Break`. A cancel takes effect within one block: about
0.3 s on the laptop below, 3 s on the Core i3.

`scunet-bench WEIGHTS WIDTH HEIGHT [--threads N] [--repeat N] [--tag TAG]
[--tile N --margin N] IMAGE.rgb...` times the model on raw RGB files and, with
`--tag`, writes its output.

## How it works

The network is written out layer by layer (`src/network.rs`, `src/swin.rs`)
instead of interpreted from a graph:

- Every convolution and linear layer runs on one matrix product kernel (the
  engine's `gemm.rs`): six rows by sixteen columns of vector accumulators,
  weights packed once into 16-column panels, and the input read through row
  pointers, so a 3x3 convolution never copies its input into columns.
- The two 3x3 convolutions of a block's convolution half use Winograd's F(4x4,
  3x3) (the engine's `winograd.rs`): 36 products per 4x4 block of output
  pixels instead of 144.
- A block's two halves, convolutions and transformer, live in one buffer and
  are updated in place: the convolution half reads and writes the first
  channels of each pixel, the transformer half the rest.
- A task takes up to four 8 x 8 attention windows: gather (rolled by four
  pixels in every second block), normalise, project, attend, project and add
  back, all in cache. The MLP runs over chunks of pixels in place.
- A block's MLP, the 1x1 convolution that joins its halves onto the level's
  map and the next block's splitting 1x1 convolution run on a chunk of pixels
  in one task. At the top level, 64 channels a pixel, those convolutions run
  at half speed when each streams the whole map through memory on its own.
- The head convolution, whose output is also the skip into the last layer, is
  computed again at the end from the three-channel image instead of keeping a
  64-channel map of the whole image through the network, and the blocks'
  working buffers are freed before each level halves its map.

Work is split between threads by pixels or windows; no sum is ever split, so
the result does not depend on the number of threads. The kernels come in three
tiers, chosen when the process starts: AVX2 with FMA and F16C (tier 2:
Haswell, Zen and later), AVX (tier 1: Sandy and Ivy Bridge, Bulldozer), and
SSE2 for any other x86-64 CPU (tier 0), with plain Rust on other
architectures. Results are identical from run to run within a tier; tiers 1 and
0 give the same bits, while tier 2 fuses each multiply-add and differs from
them in the last bits. `IMAGE_ENGINE_NATIVE_TIER=N` caps the tier.

On a GPU (`src/gpu.rs`, `shaders/`) the same network runs in half floats,
with layer norms, softmax and GELU in floats. Each block is one submission.
Every 1x1 convolution and linear layer is one matrix product kernel, which
normalises its input rows itself when a layer norm comes first, gathers the
2x2 convolution's input and the attention's rolled windows, and scatters the
transposed convolution's output and the attention back into place, so no
tokens are copied. The attention runs one workgroup per window and head, two
invocations per token, each with half the head's channels. The 3x3
convolutions sum each block of 16 input channels in half floats and add the
blocks in floats: summed whole in half floats, the body's 2304-term sums cost
0.05 dB of output PSNR. As in realesrgan-native, the convolution and
matrix kernels read weights through AMD's scalar cache, or stage them in
shared memory on other GPUs. On AMD GPUs the MLPs of the first two levels
(32 and 64 channels) run as one kernel that keeps each row's 128 or 256
hidden values in registers, instead of writing them out and reading them
back.

## Performance

On a Ryzen 5 5600H (Zen 3, 6 cores, 12 threads, AVX2), 6 threads, half-float
weights, median over the 20 photos (one photo at 4000 x 3000), peak resident
memory of the whole process for one photo:

| Image | Engine | Time | Peak memory |
|---|---|---:|---:|
| 512 x 512 | ONNX Runtime 1.31 | 6.1 s | 5.8 GiB |
| 768 x 512 | scunet-native, whole image | 1.89 s | 0.36 GB |
| 1024 x 1024 | ONNX Runtime 1.31 | 25.9 s | 21.7 GiB |
| 1536 x 1024 | scunet-native, whole image | 7.57 s | 1.08 GB |
| 1536 x 1024 | scunet-native, tiles | 9.23 s | 0.49 GB |
| 4000 x 3000 | scunet-native, tiles | 77 s | 0.96 GB |

ONNX Runtime ran an export of the same checkpoint from Python. The bench keeps
float copies of the image and of the output, 0.29 GB at 4000 x 3000;
`Model::run` keeps only the 8-bit result and a tile's floats, and peaks at
0.71 GB there. A run costs 2.3 TFLOP per megapixel; 68 % of the time is the
matrix kernel, which sustains about three quarters of the CPU's multiply-add
peak on the deeper levels, so the network is compute bound: half-float
weights save memory, and 2 % of the time. All 12 threads take 1.78 s for
768 x 512.

On a Core i3-2375M (Sandy Bridge, 2 cores and 4 threads at 1.5 GHz, AVX
without AVX2), a 768 x 512 photo takes 26.1 s at 0.37 GB, with 2 threads as
with 4.

On GPUs, in half floats, time of the bench's whole run (the GPUs' own
memory is not in the peak above):

| Image | Radeon Vega 7 (the 5600H's, RADV) | GeForce GTX 1660 Ti (NVIDIA 615) |
|---|---:|---:|
| 768 x 512 | 0.74 s | 0.24 s |
| 1536 x 1024, whole image | 3.02 s | 0.97 s |
| 1536 x 1024, tiles | 3.65 s | 1.22 s |

against 1.89 s, 7.57 s and 9.23 s on the 5600H's CPU. On the Vega 7, which
shares the CPU's memory, the top level's 1x1 convolutions are bound by
memory bandwidth.

Tiles cost time because their margins are computed twice. Their result
against the whole image's, on the 20 photos at 1536 x 1024:

| Tiles | Largest difference | 8-bit values that differ | Output PSNR |
|---|---:|---:|---:|
| whole image | | | 35.042 dB |
| up to 896, margin 64 (`Model::run`) | 5 steps | 1.3 % | 35.040 dB |
| 960, margin 32 | 79 steps | 31 % | 34.508 dB |
| 448, margin 32 | 88 steps | 43 % | 34.687 dB |

A 32-pixel margin leaves seams; 64 does not.

## Precision

A weights file stores its weight matrices as 32-bit floats, or with `--half`
as 16-bit floats, which halve the file and, at tier 2, the weights in memory.
The arithmetic stays 32-bit. Against PyTorch's float32 output on the 20 photos
with Gaussian noise of deviation 25, at 768 x 512 and at 500 x 333 (which the
network pads to 512 x 384):

| Weights | Largest difference | 8-bit values that differ | Output PSNR |
|---|---:|---:|---|
| float32 | 2.0e-6 | 0.002 % (rounding ties) | identical |
| float16 (`--half`) | 5.5e-4 | 1.3 %, by one step | identical to 0.01 dB |

The GPU computes in half floats. Against the CPU engine on the 20 photos at
768 x 512, it moves 8-bit values by up to 3 steps (8 to 18 % of them on the
GTX 1660 Ti; 8-bit PSNR 55 to 59 dB); against the clean originals its PSNR is
within 0.02 dB of the CPU's, above or below, on each photo, on the Vega 7 as
on the GTX 1660 Ti.

## Checking results

`tools/scunet/evaluate.py prepare` crops and resizes photos and adds noise or
JPEG compression; `tools/scunet/torch_reference.py` writes PyTorch's outputs
for them with SCUNet's own code; `tools/scunet/evaluate.py compare` prints
PSNR and the differences between two sets of outputs. With a directory of
`NAME.rgb` inputs and their `NAME.torch.out`, the ignored test checks float32
weights against them:

```bash
cargo test --release
SCUNET_NATIVE_WEIGHTS=$PWD/weights/scunet-color-real-psnr.weights \
  SCUNET_NATIVE_REFERENCE=DIR SCUNET_NATIVE_SIZE=768x512 \
  cargo test --release --test reference -- --ignored
```

The ignored `gpu` test needs a GPU and checks it against the CPU engine on
the same photos: an 8-bit PSNR above 52 dB.

## Credits and license

SCUNet is the work of Kai Zhang, Yawei Li, Jingyun Liang, Jiezhang Cao, Yulun
Zhang, Hao Tang, Deng-Ping Fan, Radu Timofte and Luc Van Gool, published under
Apache-2.0. [NOTICE](NOTICE) lists the papers and the code this builds on and
carries SCUNet's licence, which applies to the weights.

The Rust implementation is by Bruno Gonçalves
([BigLinux](https://www.biglinux.com.br)) and is licensed MIT
([LICENSE](../../LICENSE)).