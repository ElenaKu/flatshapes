"""Tune flatshapes limits from your verdicts (made in the web test bench).

    python tools/tune.py IMAGE_DIR [--verdicts VERDICT_DIR] [--passes 2]

Reads every verdict file (correct / wrong marks and missed shapes), re-fits only the regions that carry a mark,
and searches the detector limits (coordinate descent, one limit at a time over a grid around its current value)
for the settings that satisfy the most marks:
  correct mark -> that kind is still detected there
  wrong mark   -> that kind is no longer detected there
  missed mark  -> that kind is now detected there
Ties keep the value closest to the current one. Writes VERDICT_DIR/tuned_settings.json, which the web UI loads
with "Load tuned settings" (Settings tab), and prints the marks satisfied before and after. Lines are not tuned.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import flatshapes as fs  # noqa: E402
from flatshapes import detect as fs_detect  # noqa: E402
from shapes_report import quantize  # noqa: E402

KINDS = ("circle", "ellipse", "polygon", "star")
# defaults of the limits searched (same as the library and the web UI)
DEFAULTS = {
    "general": dict(colors=8, merge_de=10, min_area=60, rim=4, underlay=0.5, slack=0.15),
    "circle": dict(max_mean=0.45, max_p80=0.9, rel_mean=0.022, rel_p80=0.035),
    "ellipse": dict(max_mean=0.5, max_p80=1.0, rel_mean=0.022, rel_p80=0.035, max_aspect=6.0),
    "polygon": dict(max_mean=0.8, max_p80=1.3, rel_mean=0.022, rel_p80=0.035, shallow_turn=25.0),
    "star": dict(max_mean=0.7, max_p80=1.5, rel_mean=0.022, rel_p80=0.035, max_area_err=0.06),
}
SEARCH = [("general", "slack")] + [(k, p) for k in KINDS for p in DEFAULTS[k]]
FACTORS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.45, 1.8, 2.3)


def load_cases(image_dir, verdict_dir):
    """One case per verdict file: label map and the components that carry marks, with their marks."""
    cases = []
    for root, _, files in os.walk(verdict_dir):
        for n in sorted(files):
            if not n.endswith(".json") or n == "tuned_settings.json":
                continue
            with open(os.path.join(root, n), encoding="utf-8") as fh:
                v = json.load(fh)
            marks = [m for m in v.get("marks", []) if m["kind"] in KINDS]
            missed = [m for m in v.get("missed", []) if m["kind"] in KINDS]
            if not marks and not missed:
                continue
            img = os.path.join(image_dir, v["image"])
            bgr = cv2.imread(img, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"skip (image not found): {v['image']}")
                continue
            vals = (v.get("settings") or {}).get("values", {})
            g = {**DEFAULTS["general"], **vals.get("general", {})}
            labels, cols, bg = quantize(bgr, int(g["colors"]), merge_de=float(g["merge_de"]))
            lo = vals.get("line", {})
            lines = fs.find_lines(labels, background=bg, **{k: lo[k] for k in ("min_len", "max_width", "body_r",
                                                                                 "max_angle", "max_offset", "max_gap") if k in lo})
            pts = [(m["centre"][0], m["centre"][1], m["kind"], m["verdict"]) for m in marks]
            pts += [(m["x"], m["y"], m["kind"], "missed") for m in missed]
            regions = []
            for lab, i, m, hid, x0, y0 in fs.components(labels, bg, int(g["min_area"]), True, int(g["rim"]),
                                                         float(g["underlay"]), lines):
                ys, xs = np.nonzero(m)
                mine = []
                for (x, y, kind, verdict) in pts:
                    d = np.min(np.hypot(xs + x0 - x, ys + y0 - y))
                    if d <= 5:
                        mine.append((kind, verdict))
                if mine:
                    regions.append((m, hid, mine))
            cases.append(dict(image=v["image"], regions=regions, n=len(pts)))
            print(f"{v['image']}: {len(pts)} marks, {len(regions)} regions")
    return cases


_FIT_CACHE = {}


def _fit(region_id, kind, m, hid, opts):
    """One detector on one region, cached by its options (only the detector whose limit changed is re-run)."""
    key = (region_id, kind, tuple(sorted(opts.items())))
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = fs_detect.DETECTORS[kind](m, hid, **opts)
    return _FIT_CACHE[key]


def score(cases, values):
    """Number of satisfied marks. Same choice as detect_shape: among the fits within `slack` of the best mean
    error, the one with the fewest parameters."""
    slack = float(values["general"]["slack"])
    ok = 0
    for ci, c in enumerate(cases):
        for ri, (m, hid, marks) in enumerate(c["regions"]):
            fits = [f for f in (_fit((ci, ri), k, m, hid, values[k]) for k in KINDS) if f is not None]
            got = None
            if fits:
                bestm = min(f.metrics["mean"] for f in fits)
                got = min((f for f in fits if f.metrics["mean"] <= bestm + slack),
                          key=lambda f: (f.n_params, f.metrics["mean"])).kind
            for kind, verdict in marks:
                ok += (got == kind) if verdict in ("correct", "missed") else (got != kind)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images")
    ap.add_argument("--verdicts")
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    image_dir = os.path.abspath(a.images)
    verdict_dir = os.path.abspath(a.verdicts or os.path.join(image_dir, os.pardir, "flatshapes_verdicts"))
    cases = load_cases(image_dir, verdict_dir)
    total = sum(c["n"] for c in cases)
    if not total:
        print("No verdicts with circle / ellipse / polygon / star marks found in", verdict_dir)
        return
    values = json.loads(json.dumps(DEFAULTS))
    base = best = score(cases, values)
    print(f"\nDefaults: {base} of {total} marks satisfied")
    for p in range(a.passes):
        improved = False
        for grp, key in SEARCH:
            cur = values[grp][key]
            cands = sorted({round(cur * f, 4) for f in FACTORS})
            results = []
            for v in cands:
                values[grp][key] = v
                results.append((score(cases, values), -abs(np.log(v / cur)) if cur else 0, v))
            sc, _, v = max(results)
            values[grp][key] = v
            if sc > best:
                print(f"  pass {p + 1}: {grp}.{key} {cur} -> {v}: {sc} of {total}")
                best, improved = sc, True
        if not improved:
            break
    out = os.path.join(verdict_dir, "tuned_settings.json")
    os.makedirs(verdict_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        # general settings other than slack (colours, min area, ...) stay as set in the UI
        tuned = {**{k: values[k] for k in KINDS}, "general": {"slack": values["general"]["slack"]}}
        json.dump(dict(values=tuned, satisfied=best, total=total, defaults_satisfied=base), fh, indent=1)
    print(f"\nTuned: {best} of {total} marks satisfied (defaults: {base}). Saved {out}")
    changed = [(g, k, DEFAULTS[g][k], values[g][k]) for g in DEFAULTS for k in DEFAULTS[g] if values[g][k] != DEFAULTS[g][k]]
    for g, k, d, v in changed:
        print(f"  {g}.{k}: {d} -> {v}")


if __name__ == "__main__":
    main()
