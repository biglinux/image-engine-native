#!/usr/bin/env python3
"""Export SCUNet's PyTorch checkpoint into a scunet-native weights file.

    export_weights.py [--half] MODEL.pth OUT.weights

MODEL.pth is one of the color checkpoints of cszn/KAIR's v1.0 release, as
scunet_color_real_psnr.pth. It is read without PyTorch by
tools/checkpoint.py, which refuses any class but tensors and their dict. The
file holds the tensors in the layouts crates/scunet-native/src/network.rs and
swin.rs read:

* a convolution weight becomes a matrix of (tap row, tap column, input
  channel) rows by output channel columns, the order the engine's NHWC
  convolutions read their input in;
* a transposed convolution weight becomes input channel rows by (tap row,
  tap column, output channel) columns;
* a linear weight is transposed to input by output;
* a window attention's relative position parameters become its whole
  `[heads, 64, 64]` bias, named `position_bias`.

With --half, the weight matrices are stored as IEEE half floats (biases,
normalisations and position biases stay float32): half the size, and on CPUs
with AVX2 they stay half in memory too.

File format, little-endian: the bytes b"SCUNET\\0\\0", u32 version 1, u32
tensor count, then per tensor a u16 name length, the UTF-8 name, a u8 type (0
for float32, 1 for float16), a u8 rank, u32 dimensions, and the values.
"""

import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from checkpoint import load as load_checkpoint

WINDOW = 8


def load(path):
    state = load_checkpoint(path)
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: not a state dict")
    return dict(state)


def conv(w):
    """[out, in, kh, kw] to [(kh, kw, in), out]."""
    o, i, kh, kw = w.shape
    return w.transpose(2, 3, 1, 0).reshape(kh * kw * i, o)


def transposed_conv(w):
    """[in, out, kh, kw] to [in, (kh, kw, out)]."""
    i, o, kh, kw = w.shape
    return w.transpose(0, 2, 3, 1).reshape(i, kh * kw * o)


def position_bias(params):
    """[heads, 15, 15] to [heads, 64, 64]: the bias between the tokens at
    (y1, x1) and (y2, x2) of a window is params[:, y1 - y2 + 7, x1 - x2 + 7]."""
    cells = [(y, x) for y in range(WINDOW) for x in range(WINDOW)]
    dy = np.array([[a[0] - b[0] for b in cells] for a in cells]) + WINDOW - 1
    dx = np.array([[a[1] - b[1] for b in cells] for a in cells]) + WINDOW - 1
    return params[:, dy, dx]


def export(state):
    out = {}
    for name, value in state.items():
        value = np.asarray(value, dtype=np.float32)
        if name.endswith(".relative_position_params"):
            out[name.removesuffix("relative_position_params") + "position_bias"] = (
                position_bias(value)
            )
        elif name.startswith("m_up") and name.split(".")[1:] == ["0", "weight"]:
            out[name] = transposed_conv(value)
        elif value.ndim == 4:
            out[name] = conv(value)
        elif value.ndim == 2:
            out[name] = value.T
        else:
            out[name] = value
    return out


def write(path, tensors, half):
    with open(path, "wb") as f:
        f.write(b"SCUNET\0\0")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, array in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)
            as_half = half and array.ndim == 2 and name.endswith(".weight")
            f.write(struct.pack("<BB", int(as_half), array.ndim))
            f.write(struct.pack(f"<{array.ndim}I", *array.shape))
            f.write(
                np.ascontiguousarray(array)
                .astype("<f2" if as_half else "<f4")
                .tobytes()
            )


def main():
    args = sys.argv[1:]
    half = "--half" in args
    args = [a for a in args if a != "--half"]
    if len(args) != 2:
        raise SystemExit(__doc__.split("\n\n")[1])
    tensors = export(load(args[0]))
    write(args[1], tensors, half)
    count = sum(a.size for a in tensors.values())
    print(f"{len(tensors)} tensors, {count} floats")


if __name__ == "__main__":
    main()
