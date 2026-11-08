#!/usr/bin/env python3
"""Inputs, ONNX Runtime references and mask agreement for BiRefNet-lite.

    evaluate.py prepare IMAGES_DIR OUT_DIR [--side 1024]
        Resize every image to SIDE x SIDE RGB (Pillow Lanczos, as the caller
        of the model would) and write OUT_DIR/NAME.rgb, raw 8-bit RGB rows.
    evaluate.py ort MODEL.onnx RGB_DIR TAG [--threads N]
        Run ONNX Runtime on every NAME.rgb and write NAME.TAG.logits, the
        model's raw output as little-endian float32 rows.
    evaluate.py compare RGB_DIR REF_TAG TAG [--other-dir DIR]
        Compare the masks of two runs, per image and over the corpus. With
        --other-dir, TAG's logits are read from DIR (same image names), and
        resized to the reference's side when they come from a smaller input.

The mask is sigmoid(logits) rounded to 8 bits, as a caller would store it.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalized_nchw(rgb):
    x = rgb.astype(np.float32) / np.float32(255.0)
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def read_rgb(path):
    data = np.fromfile(path, dtype=np.uint8)
    side = round((data.size // 3) ** 0.5)
    if side * side * 3 != data.size:
        sys.exit(f"{path}: not a square RGB image")
    return data.reshape(side, side, 3)


def alpha8(logits):
    return np.round(255.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.uint8)


def upscale(logits, side):
    """Bilinear resize of square logits to side x side, as a lower input
    resolution would be brought back to the reference's."""
    from PIL import Image

    small = round(logits.size**0.5)
    image = Image.fromarray(logits.reshape(small, small).astype(np.float32), mode="F")
    return np.asarray(
        image.resize((side, side), Image.BILINEAR), dtype=np.float32
    ).ravel()


def prepare(args):
    from PIL import Image

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(pathlib.Path(args.images).iterdir()):
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            print(f"{path.name}: not an image, skipped", file=sys.stderr)
            continue
        image = image.resize((args.side, args.side), Image.LANCZOS)
        np.asarray(image, dtype=np.uint8).tofile(out / f"{path.stem}.rgb")
        print(path.stem)


def ort(args):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = onnxruntime.InferenceSession(
        args.model, options, providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    for path in sorted(pathlib.Path(args.rgb_dir).glob("*.rgb")):
        start = time.perf_counter()
        (logits,) = session.run(None, {name: normalized_nchw(read_rgb(path))})
        elapsed = time.perf_counter() - start
        logits.astype("<f4").tofile(path.with_suffix(f".{args.tag}.logits"))
        print(f"{path.stem} {elapsed:.3f}s")


def compare(args):
    rows = []
    for path in sorted(pathlib.Path(args.rgb_dir).glob("*.rgb")):
        ref = path.with_suffix(f".{args.ref}.logits")
        new = (
            pathlib.Path(args.other_dir or args.rgb_dir)
            / f"{path.stem}.{args.tag}.logits"
        )
        if not (ref.exists() and new.exists()):
            continue
        a = np.fromfile(ref, dtype="<f4")
        b = np.fromfile(new, dtype="<f4")
        if b.size != a.size:
            b = upscale(b, round(a.size**0.5))
        a, b = alpha8(a), alpha8(b)
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        fa, fb = a >= 128, b >= 128
        union = np.count_nonzero(fa | fb)
        iou = np.count_nonzero(fa & fb) / union if union else 1.0
        rows.append(
            (
                path.stem,
                diff.mean() / 255.0,
                int(diff.max()),
                np.count_nonzero(diff > 2) / diff.size,
                np.count_nonzero(diff > 25) / diff.size,
                iou,
            )
        )
    if not rows:
        sys.exit("no image has both runs")
    print(f"{'image':42} {'MAE':>9} {'max':>4} {'>2/255':>8} {'>25/255':>8} {'IoU':>8}")
    for name, mae, peak, small, large, iou in rows:
        print(
            f"{name[:42]:42} {mae:9.6f} {peak:4d} {small:8.4%} {large:8.4%} {iou:8.5f}"
        )
    data = np.array([r[1:] for r in rows], dtype=np.float64)
    print(
        f"{'mean of ' + str(len(rows)) + ' images':42} {data[:, 0].mean():9.6f} "
        f"{int(data[:, 1].max()):4d} {data[:, 2].mean():8.4%} {data[:, 3].mean():8.4%} "
        f"{data[:, 4].mean():8.5f}"
    )
    print(
        f"{'worst':42} {data[:, 0].max():9.6f} {'':4} {data[:, 2].max():8.4%} "
        f"{data[:, 3].max():8.4%} {data[:, 4].min():8.5f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("images")
    p.add_argument("out")
    p.add_argument("--side", type=int, default=1024)
    p.set_defaults(run=prepare)
    p = sub.add_parser("ort")
    p.add_argument("model")
    p.add_argument("rgb_dir")
    p.add_argument("tag")
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(run=ort)
    p = sub.add_parser("compare")
    p.add_argument("rgb_dir")
    p.add_argument("ref")
    p.add_argument("tag")
    p.add_argument("--other-dir")
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
