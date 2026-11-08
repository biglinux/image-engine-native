#!/usr/bin/env python3
"""Export Real-ESRGAN's compact model into a realesrgan-native weights file.

    export_weights.py [--half] [--weak WDN.pth --strength S] MODEL.pth OUT.weights

MODEL.pth is realesr-general-x4v3.pth from Real-ESRGAN's v0.2.5.0 release.
With --weak, its weights are blended with realesr-general-wdn-x4v3.pth's as
Real-ESRGAN's own command line does for --denoise_strength S: S times the
first plus 1 - S times the second, every tensor. S = 1 is MODEL.pth alone,
the strongest denoising; Real-ESRGAN's default is 0.5.

The checkpoints are read without PyTorch by tools/checkpoint.py, which
refuses any class but tensors and the dicts that hold them. Every convolution
weight becomes a
matrix of (tap row, tap column, input channel) rows by output channel
columns, the order the engine's NHWC convolutions read their input in.

With --half, the convolution matrices are stored as IEEE half floats (biases
and PReLU slopes stay float32): half the size, and on CPUs with AVX2 they
stay half in memory too.

File format, little-endian: the bytes b"REALESRG", u32 version 1, u32 tensor
count, then per tensor a u16 name length, the UTF-8 name, a u8 type (0 for
float32, 1 for float16), a u8 rank, u32 dimensions, and the values.
"""

import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from checkpoint import load as load_checkpoint


def load(path):
    """The model's tensors, preferring the averaged "params_ema" set as
    Real-ESRGAN does."""
    checkpoint = load_checkpoint(path)
    for key in ("params_ema", "params"):
        if key in checkpoint:
            return {k: np.asarray(v, np.float32) for k, v in checkpoint[key].items()}
    raise SystemExit(f"{path}: no params or params_ema")


def export(state):
    out = {}
    for name, value in state.items():
        if value.ndim == 4:
            o, i, kh, kw = value.shape
            value = value.transpose(2, 3, 1, 0).reshape(kh * kw * i, o)
        out[name] = value
    return out


def write(path, tensors, half):
    with open(path, "wb") as f:
        f.write(b"REALESRG")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, array in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)
            as_half = half and array.ndim == 2
            f.write(struct.pack("<BB", int(as_half), array.ndim))
            f.write(struct.pack(f"<{array.ndim}I", *array.shape))
            data = np.ascontiguousarray(array).astype("<f2" if as_half else "<f4")
            f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--weak", help="realesr-general-wdn-x4v3.pth")
    parser.add_argument("--strength", type=float)
    parser.add_argument("model")
    parser.add_argument("out")
    args = parser.parse_args()
    if (args.weak is None) != (args.strength is None):
        parser.error("--weak and --strength go together")
    state = load(args.model)
    if args.weak:
        if not 0 <= args.strength <= 1:
            parser.error("--strength is between 0 and 1")
        weak = load(args.weak)
        if weak.keys() != state.keys() or any(
            weak[k].shape != v.shape for k, v in state.items()
        ):
            raise SystemExit("the two checkpoints hold different tensors")
        s = np.float32(args.strength)
        state = {k: s * v + (np.float32(1) - s) * weak[k] for k, v in state.items()}
    tensors = export(state)
    write(args.out, tensors, args.half)
    count = sum(a.size for a in tensors.values())
    print(f"{len(tensors)} tensors, {count} floats")


if __name__ == "__main__":
    main()
