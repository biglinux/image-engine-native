#!/usr/bin/env python3
"""Prepare degraded test images and compare outputs.

    evaluate.py prepare IMG_DIR OUT_DIR WIDTH HEIGHT [--sigma S] [--jpeg Q]
    evaluate.py compare DIR WIDTH HEIGHT TAG [OTHER_TAG]

prepare crops every photo of IMG_DIR to the WIDTH:HEIGHT shape around its
centre, resizes it to WIDTH x HEIGHT, and writes the result as NAME.clean
and a degraded copy as NAME.rgb (8-bit RGB rows): Gaussian noise of
deviation S on the 0-255 scale, seeded by the name, and/or a JPEG round trip
at quality Q.

compare reads NAME.TAG.out (float32 RGB rows, as scunet-bench --tag and
torch_reference.py write them) for every NAME.rgb in DIR and prints the PSNR
of the degraded image and of the output against the clean one; with
OTHER_TAG, also how far the two outputs are apart: the largest and mean
absolute difference, and the 8-bit values that differ after rounding.
"""

import argparse
import io
import pathlib
import sys
import zlib

import numpy as np
from PIL import Image


def prepare(args):
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    photos = {}
    for path in sorted(pathlib.Path(args.img_dir).iterdir()):
        try:
            with Image.open(path) as image:
                image.load()
        except OSError:
            print(f"skipped {path.name}: not an image", file=sys.stderr)
            continue
        # Names that differ in case only collide on some file systems.
        other = photos.setdefault(path.stem.casefold(), path)
        if other != path:
            sys.exit(f"{path.name} and {other.name} would write the same files")
    for path in photos.values():
        image = Image.open(path).convert("RGB")
        w, h = image.size
        scale = min(w / args.width, h / args.height)
        cw, ch = args.width * scale, args.height * scale
        x0, y0 = max(0.0, (w - cw) / 2), max(0.0, (h - ch) / 2)
        box = (x0, y0, min(w, x0 + cw), min(h, y0 + ch))
        clean = image.resize((args.width, args.height), Image.LANCZOS, box=box)
        degraded = np.asarray(clean, dtype=np.float64)
        if args.sigma:
            rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
            degraded = degraded + rng.normal(0, args.sigma, degraded.shape)
        degraded = degraded.clip(0, 255).round().astype(np.uint8)
        if args.jpeg:
            buffer = io.BytesIO()
            Image.fromarray(degraded).save(buffer, "JPEG", quality=args.jpeg)
            degraded = np.asarray(Image.open(buffer).convert("RGB"))
        np.asarray(clean).tofile(out / f"{path.stem}.clean")
        degraded.tofile(out / f"{path.stem}.rgb")
        print(path.stem)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255**2 / mse)


def to_8bit(out):
    # Scaled in float32 as Model::run does, then halves rounded up as its
    # f32::round does, the sum taken in float64 so that it is exact.
    return np.floor((out.clip(0, 1) * np.float32(255)).astype(np.float64) + 0.5)


def compare(args):
    shape = (args.height, args.width, 3)
    worst, mean, differ, total = 0.0, [], 0, 0
    paths = sorted(pathlib.Path(args.dir).glob("*.rgb"))
    if not paths:
        sys.exit(f"no NAME.rgb in {args.dir}")
    for path in paths:
        clean = np.fromfile(path.with_suffix(".clean"), np.uint8).reshape(shape)
        degraded = np.fromfile(path, np.uint8).reshape(shape)
        out = np.fromfile(path.with_suffix(f".{args.tag}.out"), "<f4").reshape(shape)
        line = (
            f"{path.stem}: input {psnr(degraded, clean):.2f} dB, "
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
            line += f", {args.other} {psnr(to_8bit(other), clean):.2f} dB, max diff {diff.max():.2e}"
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
    p.add_argument("--sigma", type=float, default=0)
    p.add_argument("--jpeg", type=int, default=0)
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
