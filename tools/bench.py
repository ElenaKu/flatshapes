"""Benchmark: accuracy and speed of flatshapes on synthetic scenes with known shapes (and, optionally, on your
real images with verdicts from the web test bench).

    python tools/bench.py [--scenes 40] [--seed 1] [--save results.json] [--compare old_results.json]
                          [--export DIR] [--real IMAGE_DIR [--verdicts VERDICT_DIR]]

Synthetic scenes (fixed seed, so every run sees the same images): circles, ellipses, triangles, rectangles,
rounded rectangles, regular polygons, classic stars (as 10-vertex polygons), sparkles (concave-sided stars) and
thick lines, with wobbly edges, shapes partly covered by others, shapes lying on a big underlay and strings
running through sparkles. A detection matches a true shape of the same kind whose centre is within
max(3 px, 15 % of its size); lines match by both end points.

Prints precision / recall / F1 per kind, the confusion of kinds and the time per image; --save keeps the
numbers, --compare prints the change against an earlier run, --export writes the scenes as PNG + truth JSON
(open them in the web UI). --real scores your verdicts the same way the web UI shows them.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import flatshapes as fs  # noqa: E402
from flatshapes.star import _model as star_model  # noqa: E402
from flatshapes import parametric as par  # noqa: E402

KINDS = ("circle", "ellipse", "polygon", "star", "ring", "segment", "teardrop", "heart", "crescent", "line")
SIZE = 512
SS = 4  # supersampling for drawing


# ------------------------------------------------------------------ scene generation
def _wobble(P, amp, rng, waves=6):
    """Smooth radial-ish noise along a closed outline (amp px)."""
    if amp <= 0:
        return P
    n = len(P)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    noise = sum(rng.normal(0, 1) * np.sin(k * t + rng.uniform(0, 6.3)) for k in range(2, 2 + waves)) / np.sqrt(waves)
    c = P.mean(0)
    d = P - c
    nrm = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return P + nrm * (amp * noise)[:, None]


def _rot(P, ang):
    a = np.radians(ang)
    return P @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T


def _rounded_rect(w, h, r, n=12):
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180),
                       (w / 2 - r, -h / 2 + r, 270)):
        a = np.radians(a0 + np.linspace(0, 90, n))
        pts.append(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.vstack(pts)


def _densify(P, step=1.0):
    out = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        k = max(1, int(np.linalg.norm(b - a) / step))
        out.append(a + np.linspace(0, 1, k, endpoint=False)[:, None] * (b - a))
    return np.vstack(out)


def random_shape(rng, kind=None):
    """(kind, sub-kind, outline (n, 2) around (0, 0), size, extra)."""
    kind = kind or rng.choice(["circle", "ellipse", "triangle", "rect", "roundrect", "regular", "star5", "sparkle",
                               "ring", "segment", "teardrop", "heart", "crescent"])
    if kind == "ring":
        r = rng.uniform(20, 55)
        t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        return "ring", "ring", np.c_[r * np.cos(t), r * np.sin(t)], r, {"r_in": r * rng.uniform(0.4, 0.75)}
    if kind == "segment":
        r = rng.uniform(20, 55)
        h = float(rng.choice([0.0, rng.uniform(-0.4, 0.4) * r]))
        return "segment", "half circle" if h == 0 else "cut circle", par._segment_model(
            [0, 0, r, rng.uniform(0, 6.3), h], 120), r, {}
    if kind == "teardrop":
        r = rng.uniform(15, 40)
        return "teardrop", "teardrop", par._tear_model([0, 0, r, rng.uniform(0, 6.3), r * rng.uniform(1.5, 2.6)], 120), r, {}
    if kind == "heart":
        s = rng.uniform(20, 50)
        return "heart", "heart", par._heart_model([0, 0, s, s * rng.uniform(0.85, 1.15), rng.normal(0, 0.3)], 60), s, {}
    if kind == "crescent":
        r1 = rng.uniform(25, 55)
        Q = par._crescent_model([0, 0, r1, rng.uniform(0, 6.3), r1 * rng.uniform(0.3, 0.6), r1 * rng.uniform(0.8, 1.0)], 120)
        if Q is not None:
            return "crescent", "crescent", Q, r1, {}
        return random_shape(rng, "circle")
    if kind == "circle":
        r = rng.uniform(8, 50)
        t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        return "circle", "circle", np.c_[r * np.cos(t), r * np.sin(t)], r, {}
    if kind == "ellipse":
        a = rng.uniform(18, 60)
        b = a * rng.uniform(0.35, 0.8)
        t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        return "ellipse", "ellipse", _rot(np.c_[a * np.cos(t), b * np.sin(t)], rng.uniform(0, 180)), np.sqrt(a * b), {}
    if kind == "triangle":
        r = rng.uniform(18, 55)
        ang = np.radians(np.array([90, 210, 330]) + rng.normal(0, 8, 3))
        P = np.c_[r * np.cos(ang), r * np.sin(ang)]
        return "polygon", "triangle", _densify(_rot(P, rng.uniform(0, 360))), r * 0.6, {"k": 3}
    if kind == "rect":
        w, h = rng.uniform(20, 100), rng.uniform(20, 100)
        P = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        return "polygon", "rect", _densify(_rot(P, rng.uniform(0, 90))), np.sqrt(w * h / np.pi), {"k": 4}
    if kind == "roundrect":
        w, h = rng.uniform(30, 110), rng.uniform(30, 110)
        r = rng.uniform(0.1, 0.3) * min(w, h)
        return "polygon", "roundrect", _rot(_rounded_rect(w, h, r), rng.uniform(0, 90)), np.sqrt(w * h / np.pi), \
            {"k": 4, "corner_r": r}
    if kind == "regular":
        k = int(rng.integers(5, 9))
        r = rng.uniform(18, 50)
        a = np.radians(rng.uniform(0, 360)) + np.arange(k) * 2 * np.pi / k
        return "polygon", f"{k}-gon", _densify(np.c_[r * np.cos(a), r * np.sin(a)]), r * 0.9, {"k": k}
    if kind == "star5":
        r = rng.uniform(25, 60)
        a = np.radians(rng.uniform(0, 72)) + np.arange(10) * np.pi / 5
        rr = np.where(np.arange(10) % 2 == 0, r, r * 0.42)
        return "polygon", "star5", _densify(np.c_[rr * np.cos(a), rr * np.sin(a)]), r * 0.6, {"k": 10}
    k = int(rng.integers(4, 7))
    r = rng.uniform(25, 65)
    a = np.radians(rng.uniform(0, 360)) + np.arange(k) * 2 * np.pi / k
    T = np.c_[r * np.cos(a), r * np.sin(a)]
    P = star_model(np.zeros(2), T, np.full(k, rng.uniform(0.4, 0.7)), 60)
    return "star", "sparkle", P, r * 0.55, {"k": k}


def _area_centre(P):
    """Area centroid of a closed polygon (the same for truth and detection, whatever the vertex spacing)."""
    import cv2
    m = cv2.moments(np.asarray(P, np.float32).reshape(-1, 1, 2))
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else np.asarray(P).mean(0)


def _fill(canvas, P, label):
    cv2.fillPoly(canvas, [np.round(P * SS).astype(np.int32)], int(label))


def make_scene(rng, n_shapes=(6, 11)):
    """Label map (SIZE x SIZE, 0 = background) and the true shapes."""
    big = np.zeros((SIZE * SS, SIZE * SS), np.uint8)
    truth, placed = [], []
    # sometimes a big underlay (a bag, a card) that shapes lie on
    if rng.random() < 0.5:
        x0, y0 = rng.uniform(20, 200, 2)
        w, h = rng.uniform(180, 300, 2)
        _fill(big, np.array([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]), 7)
    wobble = float(rng.choice([0.0, 0.3, 0.6, 0.9]))
    for _ in range(int(rng.integers(*n_shapes))):
        kind, sub, P, size, extra = random_shape(rng)
        R = np.abs(P).max() + 4
        for _try in range(30):
            c = rng.uniform(R, SIZE - R, 2)
            if all(np.linalg.norm(c - q) > R + rq for q, rq in placed):
                break
        else:
            continue
        label = int(rng.integers(1, 7))
        Q = _wobble(P, wobble, rng) + c
        _fill(big, Q, label)
        if kind == "ring":  # punch the hole (background or the underlay below it)
            t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
            hole = np.round((c + extra["r_in"] * np.c_[np.cos(t), np.sin(t)]) * SS).astype(np.int32)
            hm = np.zeros_like(big)
            cv2.fillPoly(hm, [hole], 1)
            big[hm > 0] = 0
        placed.append((c, R))
        cc = _area_centre(P) + c if kind == "polygon" else c  # polygons: area centre (jittered triangles)
        truth.append(dict(kind=kind, sub=sub, cx=float(cc[0]), cy=float(cc[1]), size=float(size), label=label,
                          occluded=False, **{k: float(v) for k, v in extra.items()}))
        # partly covered by a shape in front (another colour)
        if rng.random() < 0.2:
            ang = rng.uniform(0, 2 * np.pi)
            oc = c + 0.9 * R * np.array([np.cos(ang), np.sin(ang)])
            ro = 0.45 * R
            t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
            _fill(big, oc + np.c_[ro * np.cos(t), ro * np.sin(t)], (label % 6) + 1)
            truth[-1]["occluded"] = True
            # the covering disc is a real, fully visible circle
            truth.append(dict(kind="circle", sub="circle (in front)", cx=float(oc[0]), cy=float(oc[1]), size=float(ro),
                              label=(label % 6) + 1, occluded=False))
        # a string through a sparkle (same colour)
        if sub == "sparkle" and rng.random() < 0.4:
            a = rng.uniform(0, np.pi)
            d = np.array([np.cos(a), np.sin(a)]) * (R * 2.2)
            w = rng.uniform(2.5, 4.5)
            cv2.line(big, tuple(np.round((c - d) * SS).astype(int)), tuple(np.round((c + d) * SS).astype(int)),
                     label, int(w * SS))
            truth[-1]["string"] = True
    # a dot stuck to another shape of the same colour (one region: found only from outline pieces)
    for _ in range(int(rng.integers(0, 3))):
        r = rng.uniform(10, 30)
        c = rng.uniform(60, SIZE - 60, 2)
        if any(np.linalg.norm(c - q) < 3 * r + rq for q, rq in placed):
            continue
        label = int(rng.integers(1, 7))
        t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
        _fill(big, c + np.c_[r * np.cos(t), r * np.sin(t)], label)
        ang = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(ang), np.sin(ang)])
        if rng.random() < 0.5:  # touching a second, smaller dot
            r2 = r * rng.uniform(0.5, 0.9)
            c2 = c + u * (r + r2 * 0.6)
            _fill(big, c2 + np.c_[r2 * np.cos(t), r2 * np.sin(t)], label)
            truth.append(dict(kind="circle", sub="circle stuck", cx=float(c2[0]), cy=float(c2[1]), size=float(r2),
                              label=label, occluded=False))
        else:  # a bar running out of it
            v = np.array([-u[1], u[0]])
            w = r * 0.5
            a0, a1 = c + u * r * 0.6, c + u * r * 2.6
            _fill(big, np.array([a0 - v * w / 2, a1 - v * w / 2, a1 + v * w / 2, a0 + v * w / 2]), label)
        truth.append(dict(kind="circle", sub="circle stuck", cx=float(c[0]), cy=float(c[1]), size=float(r),
                          label=label, occluded=False))
        placed.append((c, 3 * r))

    # free thick lines
    for _ in range(int(rng.integers(0, 3))):
        p0 = rng.uniform(20, SIZE - 20, 2)
        a = rng.uniform(0, np.pi)
        L = rng.uniform(60, 200)
        p1 = np.clip(p0 + L * np.array([np.cos(a), np.sin(a)]), 5, SIZE - 5)
        w = rng.uniform(3, 6)
        label = int(rng.integers(1, 7))
        cv2.line(big, tuple(np.round(p0 * SS).astype(int)), tuple(np.round(p1 * SS).astype(int)), label, int(w * SS))
        truth.append(dict(kind="line", sub="line", x0=float(p0[0]), y0=float(p0[1]), x1=float(p1[0]),
                          y1=float(p1[1]), cx=float((p0[0] + p1[0]) / 2), cy=float((p0[1] + p1[1]) / 2),
                          size=float(np.linalg.norm(p1 - p0)), label=label, occluded=False))
    labels = cv2.resize(big, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return labels, truth


# ------------------------------------------------------------------ detection and matching
def detect(labels, background=0):
    lines = fs.find_lines(labels, background=background)
    shapes = fs.detect_shapes(labels, background=background, lines=lines)
    out = []
    for lab, _, f in shapes:
        p = f.params
        if "cx" in p:  # model origin: circle centre (also of a cut circle / round end of a teardrop / outer disc)
            c = (p["cx"], p["cy"])
        elif f.kind == "star":
            c = tuple(p["centre"])
        else:
            c = tuple(_area_centre(p["vertices"]))
        out.append(dict(kind=f.kind, cx=float(c[0]), cy=float(c[1]), label=int(lab), k=p.get("k")))
    for lab, f in lines:
        p0, p1 = f.params["p0"], f.params["p1"]
        out.append(dict(kind="line", cx=float((p0[0] + p1[0]) / 2), cy=float((p0[1] + p1[1]) / 2), label=int(lab),
                        x0=float(p0[0]), y0=float(p0[1]), x1=float(p1[0]), y1=float(p1[1])))
    return out


def _line_match(t, d, tol=6.0):
    a = np.array([[t["x0"], t["y0"]], [t["x1"], t["y1"]]])
    b = np.array([[d["x0"], d["y0"]], [d["x1"], d["y1"]]])
    e = min(np.linalg.norm(a - b, axis=1).max(), np.linalg.norm(a - b[::-1], axis=1).max())
    return e <= max(tol, 0.08 * t["size"])


def match(truth, dets):
    """Greedy matching by distance. Returns (pairs (t, d), unmatched truth, unmatched detections)."""
    cand = []
    for i, t in enumerate(truth):
        tol = max(3.0, 0.15 * t["size"])
        for j, d in enumerate(dets):
            if t["kind"] == "line" or d["kind"] == "line":
                if t["kind"] == d["kind"] == "line" and _line_match(t, d):
                    cand.append((0.0, i, j))
                continue
            dist = np.hypot(t["cx"] - d["cx"], t["cy"] - d["cy"])
            if dist <= tol:
                cand.append((dist, i, j))
    cand.sort()
    ut, ud, pairs = set(range(len(truth))), set(range(len(dets))), []
    for _, i, j in cand:
        if i in ut and j in ud:
            pairs.append((i, j))
            ut.discard(i)
            ud.discard(j)
    return pairs, ut, ud


def score(all_truth, all_dets):
    """Per kind: tp, fp, fn; confusion[true kind][detected kind]; per sub-kind recall."""
    st = {k: dict(tp=0, fp=0, fn=0) for k in KINDS}
    conf = {k: {} for k in KINDS}
    sub = {}
    for truth, dets in zip(all_truth, all_dets):
        pairs, ut, ud = match(truth, dets)
        for i, j in pairs:
            tk, dk = truth[i]["kind"], dets[j]["kind"]
            conf[tk][dk] = conf[tk].get(dk, 0) + 1
            key = truth[i]["sub"] + (" occluded" if truth[i]["occluded"] else "") + (" +string" if truth[i].get("string") else "")
            s = sub.setdefault(key, [0, 0])
            s[1] += 1
            if tk == dk:
                st[tk]["tp"] += 1
                s[0] += 1
            else:
                st[tk]["fn"] += 1
                st[dk]["fp"] += 1
        for i in ut:
            t = truth[i]
            st[t["kind"]]["fn"] += 1
            conf[t["kind"]]["none"] = conf[t["kind"]].get("none", 0) + 1
            key = t["sub"] + (" occluded" if t["occluded"] else "") + (" +string" if t.get("string") else "")
            sub.setdefault(key, [0, 0])[1] += 1
        for j in ud:
            st[dets[j]["kind"]]["fp"] += 1
    for k, v in st.items():
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None
        r = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None
        v.update(precision=p, recall=r, f1=(2 * p * r / (p + r)) if p and r else (0.0 if p is not None and r is not None else None))
    return st, conf, {k: dict(found=v[0], total=v[1]) for k, v in sorted(sub.items())}


# ------------------------------------------------------------------ real images with verdicts
def real_score(image_dir, verdict_dir):
    """Share of your marks satisfied by the current code: correct kept, wrong gone, missed found."""
    from shapes_report import quantize
    ok = tot = 0
    by_kind = {}
    t_sum, n_img = 0.0, 0
    for root, _, files in os.walk(verdict_dir):
        for n in sorted(files):
            if not n.endswith(".json") or n == "tuned_settings.json":
                continue
            with open(os.path.join(root, n), encoding="utf-8") as fh:
                v = json.load(fh)
            bgr = cv2.imread(os.path.join(image_dir, v["image"]), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            g = ((v.get("settings") or {}).get("values") or {}).get("general", {})
            labels, _, bg = quantize(bgr, int(g.get("colors", 8)), merge_de=float(g.get("merge_de", 10)))
            t0 = time.perf_counter()
            dets = detect(labels, bg)
            t_sum += time.perf_counter() - t0
            n_img += 1
            near = lambda kind, x, y, tol: any(d["kind"] == kind and np.hypot(d["cx"] - x, d["cy"] - y) <= tol for d in dets)
            for m in v.get("marks", []):
                hit = near(m["kind"], m["centre"][0], m["centre"][1], 4.0)
                good = hit if m["verdict"] == "correct" else not hit
                bk = by_kind.setdefault(m["kind"], [0, 0])
                bk[0] += good
                bk[1] += 1
                ok += good
                tot += 1
            for m in v.get("missed", []):
                good = near(m["kind"], m["x"], m["y"], 12.0)
                bk = by_kind.setdefault(m["kind"], [0, 0])
                bk[0] += good
                bk[1] += 1
                ok += good
                tot += 1
    return dict(satisfied=ok, total=tot, by_kind=by_kind, images=n_img, sec_per_image=t_sum / max(1, n_img))


# ------------------------------------------------------------------ report
def _timed_detect(L):
    t0 = time.perf_counter()
    d = detect(L)
    return d, time.perf_counter() - t0


def _f(v):
    return "  -  " if v is None else f"{v:5.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--export")
    ap.add_argument("--real")
    ap.add_argument("--verdicts")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes, one scene each (0 = all cores); the time per image stays single-core")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    scenes = [make_scene(rng) for _ in range(a.scenes)]
    if a.export:
        os.makedirs(a.export, exist_ok=True)
        pal = np.array([(20, 20, 24), (60, 180, 230), (230, 160, 40), (90, 200, 90), (220, 80, 120), (180, 120, 230),
                        (240, 230, 120), (70, 70, 80)], np.uint8)
        for i, (L, truth) in enumerate(scenes):
            cv2.imwrite(os.path.join(a.export, f"scene_{i:03d}.png"), pal[L][:, :, ::-1])
            with open(os.path.join(a.export, f"scene_{i:03d}.json"), "w", encoding="utf-8") as fh:
                json.dump(truth, fh, indent=1)
        print(f"Exported {len(scenes)} scenes to {a.export}")

    all_truth = [t for _, t in scenes]
    t_wall = time.perf_counter()
    if a.workers != 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers or None) as ex:
            out = list(ex.map(_timed_detect, [L for L, _ in scenes]))
    else:
        detect(scenes[0][0])  # warm-up (imports, caches)
        out = [_timed_detect(L) for L, _ in scenes]
    wall = time.perf_counter() - t_wall
    all_dets = [d for d, _ in out]
    times = [t for _, t in out]
    st, conf, sub = score(all_truth, all_dets)
    res = dict(scenes=a.scenes, seed=a.seed, wall_sec=wall, sec_per_image=float(np.mean(times)), sec_max=float(np.max(times)),
               kinds=st, confusion=conf, subkinds=sub, version=getattr(fs, "__version__", "?"))
    if a.real:
        res["real"] = real_score(os.path.abspath(a.real), os.path.abspath(
            a.verdicts or os.path.join(a.real, os.pardir, "flatshapes_verdicts")))

    old = None
    if a.compare:
        with open(a.compare, encoding="utf-8") as fh:
            old = json.load(fh)

    def delta(new, old_v):
        if old is None or new is None or old_v is None:
            return ""
        d = new - old_v
        return f" ({'+' if d >= 0 else ''}{d:.2f})" if abs(d) >= 0.005 else ""

    print(f"\n{a.scenes} synthetic scenes ({SIZE}x{SIZE}), seed {a.seed}")
    print(f"time per image: {res['sec_per_image']:.2f} s (max {res['sec_max']:.2f})"
          + (delta(res["sec_per_image"], old.get("sec_per_image")) + " s" if old else "")
          + f"; all scenes {wall:.1f} s wall with {a.workers or os.cpu_count()} process(es)")
    print(f"\n{'kind':9} {'tp':>4} {'fp':>4} {'fn':>4}  {'prec':>5} {'recall':>6} {'F1':>5}")
    for k in KINDS:
        v = st[k]
        o = (old or {}).get("kinds", {}).get(k, {})
        print(f"{k:9} {v['tp']:4d} {v['fp']:4d} {v['fn']:4d}  {_f(v['precision'])} {_f(v['recall']):>6} {_f(v['f1'])}"
              f"{delta(v['f1'], o.get('f1'))}")
    print("\ntrue kind -> detected as")
    for k in KINDS:
        if conf[k]:
            print(f"  {k:8} " + ", ".join(f"{d} {n}" for d, n in sorted(conf[k].items(), key=lambda x: -x[1])))
    print("\nrecall by shape type")
    for k, v in sub.items():
        o = (old or {}).get("subkinds", {}).get(k)
        r = v["found"] / v["total"]
        ro = o["found"] / o["total"] if o and o["total"] else None
        print(f"  {k:24} {v['found']:3d}/{v['total']:<3d} {r:5.2f}{delta(r, ro)}")
    if "real" in res:
        r = res["real"]
        print(f"\nreal images with verdicts: {r['satisfied']}/{r['total']} marks satisfied on {r['images']} images, "
              f"{r['sec_per_image']:.2f} s per image")
        for k, (g, t) in sorted(r["by_kind"].items()):
            print(f"  {k:8} {g}/{t}")
    if a.save:
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nsaved {a.save}")


if __name__ == "__main__":
    main()
