"""Geometric abstraction of flat illustrations: every object rebuilt from a few simple shapes.

    python tools/geometrize.py INPUT_DIR OUTPUT_DIR [--shapes 3] [--kinds all] [--colors 8] [--workers 0]

For every PNG/JPG under INPUT_DIR (subfolders included, same structure in OUTPUT_DIR) writes <name>.svg and a
preview <name>_geo.png. --shapes: at most this many shapes per object (1 = very abstract, 5-6 = close to the
original). --kinds: shapes to use, comma separated: circle, ellipse, rect, roundrect, triangle, capsule, or 'all';
e.g. --kinds circle for dots only, --kinds rect,triangle for a blocky look. --no-lines drops thin strokes.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from flatshapes import abstract as A  # noqa: E402
from shapes_report import quantize  # noqa: E402


def lab_to_hex(c):
    lab = np.array([[[c[0] * 255 / 100, c[1] + 128, c[2] + 128]]], np.uint8)
    b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def run(src, dst_base, a):
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:
        print("skip (unreadable):", src)
        return
    labels, cols, bg = quantize(bgr, a.colors)
    pal = [lab_to_hex(c) for c in cols]
    kinds = A.KINDS if a.kinds == "all" else tuple(k.strip() for k in a.kinds.split(","))
    prims, svg = A.geometrize(labels, pal, bg, max_shapes=a.shapes, kinds=kinds, lines=not a.no_lines)
    os.makedirs(os.path.dirname(dst_base) or ".", exist_ok=True)
    with open(dst_base + ".svg", "w", encoding="utf-8") as fh:
        fh.write(svg)
    cv2.imwrite(dst_base + "_geo.png", A.render(prims, labels.shape, pal[bg]))
    print(f"{os.path.basename(src)}: {len(prims)} shapes")


def _job(t):
    run(*t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--shapes", type=int, default=3)
    ap.add_argument("--kinds", default="all")
    ap.add_argument("--colors", type=int, default=8)
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="processes, one image each (0 = all cores)")
    a = ap.parse_args()
    jobs = []
    for root, _, files in os.walk(a.input):
        for n in sorted(files):
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                rel = os.path.relpath(os.path.join(root, n), a.input)
                jobs.append((os.path.join(root, n), os.path.join(a.output, os.path.splitext(rel)[0]), a))
    if (a.workers or os.cpu_count() or 1) > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers or None) as ex:
            list(ex.map(_job, jobs))
    else:
        for j in jobs:
            run(*j)


if __name__ == "__main__":
    main()
