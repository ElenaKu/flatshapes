"""Run flatshapes on flat illustrations and draw what was found.

For every PNG/JPG under the input folder (subfolders included, same structure in the output):
  <name>_shapes.png   the image with detected shapes outlined (circle cyan, ellipse blue, polygon green,
                      star magenta, line orange) and a short label at each shape
  <name>_shapes.json  every shape: colour label, kind, parameters, fit metrics
and summary.csv for the whole run.

The image is quantized to a few colours (k-means in CIELAB) to get the label map; flatshapes itself works on
the label map, so for real use pass the labels from your own vectorizer.

Usage: python tools/shapes_report.py INPUT_DIR OUTPUT_DIR [--colors 8] [--min-area 60]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flatshapes as fs  # noqa: E402

KIND_BGR = {"circle": (255, 255, 0), "ellipse": (255, 80, 0), "polygon": (0, 200, 0), "star": (255, 0, 255),
            "line": (0, 140, 255)}


def quantize(bgr, n_colors, seed=0, merge_de=10.0):
    """Label map from k-means in CIELAB, with a 3x3 median to drop anti-aliasing specks. Clusters closer than
    merge_de (CIELAB distance) are one colour (a dark background split in two would otherwise hide every shape on
    it). Returns (labels, lab colours, background label = most frequent)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    sample = lab[rng.choice(len(lab), min(len(lab), 60000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centres = cv2.kmeans(sample, n_colors, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = ((lab[:, None, :] - centres[None]) ** 2).sum(2)
    labels = d.argmin(1).astype(np.uint8).reshape(bgr.shape[:2])
    # OpenCV 8-bit Lab -> CIELAB
    cie = centres.astype(float) * np.array([100 / 255, 1, 1]) - np.array([0, 128, 128])
    area = np.bincount(labels.ravel(), minlength=n_colors)
    target = np.arange(n_colors)
    for i in np.argsort(-area):  # big clusters absorb near small ones
        for j in np.argsort(-area):
            if area[j] < area[i] and target[j] == j and target[i] == i and np.linalg.norm(cie[i] - cie[j]) < merge_de:
                target[j] = i
    labels = target[labels].astype(np.uint8)
    labels = cv2.medianBlur(labels, 3)
    bg = int(np.bincount(labels.ravel(), minlength=n_colors).argmax())
    return labels, cie, bg


def _poly_of(f):
    """Outline points (n, 2) for drawing."""
    p = f.params
    if f.kind == "line":
        return np.array([p["p0"], p["p1"]])
    if f.kind == "circle":
        t = np.linspace(0, 2 * np.pi, 120)
        return np.c_[p["cx"] + p["r"] * np.cos(t), p["cy"] + p["r"] * np.sin(t)]
    if f.kind == "ellipse":
        t = np.linspace(0, 2 * np.pi, 120)
        c, s = np.cos(p["theta"]), np.sin(p["theta"])
        x, y = p["a"] * np.cos(t), p["b"] * np.sin(t)
        return np.c_[p["cx"] + c * x - s * y, p["cy"] + s * x + c * y]
    if f.kind == "star":
        from flatshapes.star import _model
        return _model(p["centre"], p["tips"], p["bend"], 30)
    return np.asarray(p["vertices"])


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return np.round(v, 2).tolist()
    if isinstance(v, (np.floating, float)):
        return round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def process(path, out_base, args):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"skip (unreadable): {path}")
        return None
    labels, cols, bg = quantize(bgr, args.colors)
    found = fs.detect_shapes(labels, background=bg, min_area=args.min_area,
                             colours={i: cols[i] for i in range(len(cols))})
    lines = []
    for lab in range(len(cols)):
        if lab != bg:
            lines += [(lab, f) for f in fs.detect_lines(labels == lab, min_len=args.min_line) if fs.same_sides(f, labels)]
    vis = (bgr * 0.55).astype(np.uint8)
    rows = []
    for lab, _, f in found:
        rows.append((lab, f))
    rows += lines
    for lab, f in rows:
        P = np.round(_poly_of(f) * 4).astype(np.int32)
        closed = f.kind != "line"
        cv2.polylines(vis, [P], closed, KIND_BGR[f.kind], 2, cv2.LINE_AA, shift=2)
        c = P.mean(0) / 4
        cv2.putText(vis, f.kind[0].upper(), (int(c[0]) - 4, int(c[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    KIND_BGR[f.kind], 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    cv2.imwrite(out_base + "_shapes.png", vis)
    data = [dict(label=int(lab), kind=f.kind, params={k: _jsonable(v) for k, v in f.params.items()},
                 metrics={k: _jsonable(v) for k, v in f.metrics.items()}) for lab, f in rows]
    with open(out_base + "_shapes.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    counts = {k: sum(1 for _, f in rows if f.kind == k) for k in KIND_BGR}
    print(os.path.basename(path), counts)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--colors", type=int, default=8)
    ap.add_argument("--min-area", type=int, default=60)
    ap.add_argument("--min-line", type=float, default=30.0)
    args = ap.parse_args()
    summary = []
    for root, _, files in os.walk(args.input):
        for name in sorted(files):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            rel = os.path.relpath(os.path.join(root, name), args.input)
            out_base = os.path.join(args.output, os.path.splitext(rel)[0])
            counts = process(os.path.join(root, name), out_base, args)
            if counts is not None:
                summary.append(dict(file=rel, **counts))
    if summary:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)


if __name__ == "__main__":
    main()
