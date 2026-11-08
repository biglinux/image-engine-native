#!/usr/bin/env python3
"""Prepare photos with holes to fill, and compare predictions.

    evaluate.py prepare IMG_DIR OUT_DIR WIDTH HEIGHT
    evaluate.py compare DIR WIDTH HEIGHT TAG [OTHER_TAG]

prepare crops every photo of IMG_DIR to the WIDTH:HEIGHT shape around its
centre and resizes it to WIDTH x HEIGHT (Lanczos), writing NAME.rgb (8-bit
RGB rows) and NAME.mask (a byte per pixel, 255 where the photo is to be
filled): a few thick strokes and a box, drawn at random from a generator
seeded by the name; prepare prints the share of each photo they cover.

compare reads NAME.TAG.out (the prediction for every pixel, float32 RGB rows,
as lama-bench --tag and torch_reference.py write them) for every NAME.rgb in
DIR and prints the PSNR inside the holes of the filled photo against the
original; with OTHER_TAG, also how far the two predictions are apart: the
largest and mean absolute difference, and the 8-bit values that differ after
rounding.
"""

import argparse
import pathlib
import sys
import unicodedata
import zlib

import numpy as np
from PIL import Image, ImageDraw


def holes(name, width, height):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    side = min(width, height)
    for _ in range(3):
        points = [(rng.uniform(0, width), rng.uniform(0, height))]
        for _ in range(4):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(0.1, 0.3) * side
            x, y = points[-1]
            points.append((x + length * np.cos(angle), y + length * np.sin(angle)))
        draw.line(
            points, fill=255, width=int(rng.uniform(0.04, 0.08) * side), joint="curve"
        )
    bw, bh = rng.uniform(0.15, 0.3) * width, rng.uniform(0.15, 0.3) * height
    x0, y0 = rng.uniform(0, width - bw), rng.uniform(0, height - bh)
    draw.rectangle((x0, y0, x0 + bw, y0 + bh), fill=255)
    return np.asarray(mask)


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
        # Names that differ in case or Unicode normalization only collide
        # on some file systems.
        key = unicodedata.normalize("NFC", path.stem).casefold()
        other = photos.setdefault(key, path)
        if other != path:
            sys.exit(f"{path.name} and {other.name} would write the same files")
    if not photos:
        sys.exit(f"no photo in {args.img_dir}")
    for path in photos.values():
        try:
            image = Image.open(path).convert("RGB")
        except OSError as e:
            print(f"skipped {path.name}: {e}", file=sys.stderr)
            continue
        w, h = image.size
        scale = min(w / args.width, h / args.height)
        cw, ch = args.width * scale, args.height * scale
        x0, y0 = max(0.0, (w - cw) / 2), max(0.0, (h - ch) / 2)
        box = (x0, y0, min(w, x0 + cw), min(h, y0 + ch))
        photo = image.resize((args.width, args.height), Image.LANCZOS, box=box)
        np.asarray(photo).tofile(out / f"{path.stem}.rgb")
        mask = holes(path.stem, args.width, args.height)
        mask.tofile(out / f"{path.stem}.mask")
        print(f"{path.stem}: holes {(mask > 0).mean():.0%}")


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
        photo = np.fromfile(path, np.uint8).reshape(shape)
        hole = np.fromfile(path.with_suffix(".mask"), np.uint8).reshape(shape[:2]) > 0
        out = np.fromfile(path.with_suffix(f".{args.tag}.out"), "<f4").reshape(shape)
        line = f"{path.stem}: {args.tag} {psnr(to_8bit(out)[hole], photo[hole]):.2f} dB in the holes"
        if args.other:
            other = np.fromfile(path.with_suffix(f".{args.other}.out"), "<f4")
            other = other.reshape(shape)
            diff = np.abs(out.astype(np.float64) - other)
            worst = max(worst, diff.max())
            mean.append(diff.mean())
            differ += int((to_8bit(out) != to_8bit(other)).sum())
            total += out.size
            line += f", {args.other} {psnr(to_8bit(other)[hole], photo[hole]):.2f} dB, max diff {diff.max():.2e}"
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
