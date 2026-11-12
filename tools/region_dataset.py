"""Training data for the region classifier: synthetic scenes from the benchmark generator, cut into regions the
way detect_shapes sees them (lines taken out, hidden masks from objects in front), each labelled with the true
shape it holds - or "none" (pieces of lines, two shapes stuck together, line crossings, text, bows, ribbons,
blobs, balloons on strings, curly scraps).

    python tools/region_dataset.py OUT_DIR [--scenes 8000] [--seed 1000] [--chunk 500] [--workers 0]

Writes OUT_DIR/regions_<first seed>.npz chunks (X: uint8 (n, 2, 64, 64), y: int16 class index) as it goes;
chunks already there are skipped, so an interrupted run continues where it stopped.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import flatshapes as fs  # noqa: E402
from flatshapes.classifier import CLASSES, region_tensor  # noqa: E402
import bench  # noqa: E402


def scene_regions(seed):
    """(X, y) for one synthetic scene."""
    rng = np.random.default_rng(seed)
    labels, truth = bench.make_scene(rng, distractors=True)
    lines = fs.find_lines(labels, background=0)
    shapes = [t for t in truth if t["kind"] != "line"]
    X, y = [], []
    for lab, _, m, hid, x0, y0 in fs.components(labels, 0, 40, True, lines=lines):
        inside = []
        for t in shapes:
            if t["label"] != lab:
                continue
            cx, cy = int(round(t["cx"])) - x0, int(round(t["cy"])) - y0
            if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]:
                inside.append(t)
            elif t["kind"] in ("ring", "crescent", "segment", "teardrop") and _near(m, cx, cy, t["size"]):
                inside.append(t)  # these have their model origin outside the shape (ring hole, moon bite)
        kind = inside[0]["kind"] if len(inside) == 1 else "none"
        # a shape with most of its area missing from the region (cut by a line strip) is not a clean example
        X.append((np.clip(region_tensor(m, hid) * 255, 0, 255)).astype(np.uint8))
        y.append(CLASSES.index(kind))
    return X, y


def _near(m, cx, cy, size):
    ys, xs = np.nonzero(m)
    return len(xs) and np.min(np.hypot(xs - cx, ys - cy)) <= 0.6 * size and \
        abs(xs.mean() - cx) < size and abs(ys.mean() - cy) < size


def chunk(args):
    first, n, out = args
    path = os.path.join(out, f"regions_{first:07d}.npz")
    if os.path.exists(path):
        return path, 0
    X, y = [], []
    for s in range(first, first + n):
        xs, ys = scene_regions(s)
        X += xs
        y += ys
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, X=np.stack(X), y=np.array(y, np.int16))
    os.replace(tmp, path)  # complete chunks only (a reset mid-write leaves no broken file)
    return path, len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--scenes", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1000, help="first scene seed (the benchmark uses seed 1)")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(s, min(a.chunk, a.seed + a.scenes - s), a.out) for s in range(a.seed, a.seed + a.scenes, a.chunk)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers or None) as ex:
        for path, n in ex.map(chunk, jobs):
            print(f"{os.path.basename(path)}: {n if n else 'already there'}", flush=True)


if __name__ == "__main__":
    main()
