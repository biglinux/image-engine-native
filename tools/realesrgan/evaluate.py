#!/usr/bin/env python3
"""Prepare small copies of photos to upscale, and compare outputs.

    evaluate.py prepare IMG_DIR OUT_DIR WIDTH HEIGHT
    evaluate.py compare DIR WIDTH HEIGHT TAG [OTHER_TAG]

prepare crops every photo of IMG_DIR to the WIDTH:HEIGHT shape around its
centre, resizes it to 4 WIDTH x 4 HEIGHT as NAME.clean, the reference, and
reduces that to WIDTH x HEIGHT as NAME.rgb, the input (8-bit RGB rows; both
resizes Lanczos).

compare reads NAME.TAG.out (float32 RGB rows four times larger, as
realesrgan-bench --tag and torch_reference.py write them) for every NAME.rgb
in DIR and prints the PSNR against NAME.clean of a Lanczos upscale of the
input and of the output; with OTHER_TAG, also how far the two outputs are
apart: the largest and mean absolute difference, and the 8-bit values that
differ after rounding.
"""

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

SCALE = 4


def prepare(args):
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    big = (args.width * SCALE, args.height * SCALE)
    for path in sorted(pathlib.Path(args.img_dir).iterdir()):
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            print(f"skipped {path.name}: not an image", file=sys.stderr)
            continue
        w, h = image.size
        scale = min(w / big[0], h / big[1])
        cw, ch = big[0] * scale, big[1] * scale
        x0, y0 = max(0.0, (w - cw) / 2), max(0.0, (h - ch) / 2)
        box = (x0, y0, min(w, x0 + cw), min(h, y0 + ch))
        clean = image.resize(big, Image.LANCZOS, box=box)
        small = clean.resize((args.width, args.height), Image.LANCZOS)
        np.asarray(clean).tofile(out / f"{path.stem}.clean")
        np.asarray(small).tofile(out / f"{path.stem}.rgb")
        print(path.stem)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255**2 / mse)


def to_8bit(out):
    return (out.clip(0, 1) * 255).round()


def compare(args):
    small_shape = (args.height, args.width, 3)
    big = (args.width * SCALE, args.height * SCALE)
    shape = (big[1], big[0], 3)
    worst, mean, differ, total = 0.0, [], 0, 0
    for path in sorted(pathlib.Path(args.dir).glob("*.rgb")):
        clean = np.fromfile(path.with_suffix(".clean"), np.uint8).reshape(shape)
        small = np.fromfile(path, np.uint8).reshape(small_shape)
        lanczos = np.asarray(Image.fromarray(small).resize(big, Image.LANCZOS))
        out = np.fromfile(path.with_suffix(f".{args.tag}.out"), "<f4").reshape(shape)
        line = (
            f"{path.stem}: lanczos {psnr(lanczos, clean):.2f} dB, "
            f"{args.tag} {psnr(to_8bit(out), clean):.2f} dB"
        )
        if args.other:
            other = np.fromfile(path.with_suffix(f".{args.other}.out"), "<f4")
            other = other.reshape(shape)
            diff = np.abs(out.astype(np.float64) - other)
            worst = max(worst, diff.max())
            mean.append(diff.mean())
            differ += int((to_8bit(out) != to_8bit(other)).sum())
            total += out.size
            line += f", max diff {diff.max():.2e}"
        print(line)
    if args.other:
        print(
            f"largest difference {worst:.2e}, mean {np.mean(mean):.2e}, "
            f"8-bit values that differ {differ} of {total} ({differ / total:.2e})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("img_dir")
    p.add_argument("out_dir")
    p.add_argument("width", type=int)
    p.add_argument("height", type=int)
    p.set_defaults(run=prepare)
    c = commands.add_parser("compare")
    c.add_argument("dir")
    c.add_argument("width", type=int)
    c.add_argument("height", type=int)
    c.add_argument("tag")
    c.add_argument("other", nargs="?")
    c.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
