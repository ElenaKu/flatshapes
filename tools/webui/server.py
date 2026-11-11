"""flatshapes test UI: a local web server for trying the detectors on your images.

    python tools/webui/server.py IMAGE_DIR [--verdicts VERDICT_DIR] [--port 8765]

Open http://localhost:8765 in a browser. IMAGE_DIR is walked recursively. Verdicts (correct / wrong shapes and
missed shapes) are saved as JSON under VERDICT_DIR with the same folder structure as IMAGE_DIR
(default: IMAGE_DIR/../flatshapes_verdicts). Standard library only, besides flatshapes' own dependencies.
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
import flatshapes as fs  # noqa: E402
from flatshapes import detect as fs_detect  # noqa: E402
from shapes_report import quantize, _poly_of, _jsonable  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
KINDS = ("circle", "ellipse", "polygon", "star")
# limits lifted to get metrics for a model that failed, so the inspector can say why
RELAX = {
    "circle": dict(max_mean=1e9, max_p80=1e9, max_gap_deg=360, min_visible=0.0),
    "ellipse": dict(max_mean=1e9, max_p80=1e9, max_gap_deg=360, min_visible=0.0, max_aspect=1e9),
    "polygon": dict(max_mean=1e9, max_p80=1e9),
    "star": dict(max_mean=1e9, max_p80=1e9, max_area_err=1e9),
}

STATE = {"root": "", "verdicts": "", "cache": {}, "lock": threading.Lock()}


def _safe_path(rel):
    p = os.path.realpath(os.path.join(STATE["root"], rel))
    if not p.startswith(os.path.realpath(STATE["root"])):
        raise ValueError("path outside image folder")
    return p


def list_images():
    out = []
    for root, dirs, files in os.walk(STATE["root"]):
        dirs.sort()
        for n in sorted(files):
            if n.lower().endswith(IMAGE_EXT):
                out.append(os.path.relpath(os.path.join(root, n), STATE["root"]).replace("\\", "/"))
    return out


def get_labels(rel, colors, merge_de):
    key = (rel, colors, merge_de)
    with STATE["lock"]:
        if key in STATE["cache"]:
            return STATE["cache"][key]
    bgr = cv2.imread(_safe_path(rel), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot read image")
    labels, cols, bg = quantize(bgr, colors, merge_de=merge_de)
    with STATE["lock"]:
        if len(STATE["cache"]) > 12:
            STATE["cache"].clear()
        STATE["cache"][key] = (labels, cols, bg)
    return labels, cols, bg


def _shape_json(f, label, sid):
    P = _poly_of(f)
    return dict(id=sid, kind=f.kind, label=int(label), closed=f.kind != "line",
                pts=np.round(P, 2).tolist(), centre=np.round(P.mean(0), 2).tolist(),
                params={k: _jsonable(v) for k, v in f.params.items()},
                metrics={k: _jsonable(v) for k, v in f.metrics.items()}, path=f.path)


def run_detect(req):
    s = req.get("settings", {})
    labels, cols, bg = get_labels(req["path"], int(s.get("colors", 8)), float(s.get("merge_de", 10)))
    opts = {k: req.get("options", {}).get(k, {}) for k in KINDS}
    kinds = tuple(k for k in KINDS if req.get("enabled", {}).get(k, True))
    lines = []
    if req.get("enabled", {}).get("line", True):
        lo = req.get("options", {}).get("line", {})
        kw = {k: lo[k] for k in ("max_angle", "max_offset", "max_gap", "min_len", "max_width", "body_r") if k in lo}
        lines = fs.find_lines(labels, background=bg, **kw)
    found = fs.detect_shapes(labels, background=bg, min_area=int(s.get("min_area", 60)), rim=int(s.get("rim", 4)),
                             underlay_share=float(s.get("underlay", 0.5)), lines=lines,
                             kinds=kinds, slack=float(s.get("slack", 0.15)), options=opts,
                             colours={i: cols[i] for i in range(len(cols))}) if kinds else []
    shapes = [_shape_json(f, lab, f"s{n}") for n, (lab, _, f) in enumerate(found)]
    shapes += [_shape_json(f, lab, f"l{n}") for n, (lab, f) in enumerate(lines)]
    motifs = []
    if req.get("enabled", {}).get("motif", True):
        mo = req.get("options", {}).get("motif", {})
        comps = list(fs.components(labels, bg, int(s.get("min_area", 60)), False, lines=lines))
        for gi, m in enumerate(fs.find_motifs(labels, bg, components=comps,
                                              min_iou=float(mo.get("min_iou", 0.85)),
                                              sig_tol=float(mo.get("sig_tol", 0.08)),
                                              same_colour=bool(mo.get("same_colour", 1)),
                                              allow_scale=bool(mo.get("allow_scale", 1)),
                                              allow_mirror=bool(mo.get("allow_mirror", 1)))):
            T = m.template["outline"][:: max(1, len(m.template["outline"]) // 80)]
            inst = []
            for it in m.instances:
                A = T.copy()
                if it["mirrored"]:
                    A[:, 1] = -A[:, 1]
                t = np.radians(it["angle"])
                R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                A = (A @ R.T) * it["scale"] + [it["cx"], it["cy"]]
                inst.append({**{k: _jsonable(v) for k, v in it.items()}, "pts": np.round(A, 1).tolist()})
            motifs.append(dict(id=f"m{gi}", count=m.count, instances=inst, path=m.path))
    h, w = labels.shape
    return dict(shapes=shapes, motifs=motifs, width=w, height=h, background=bg,
                palette=[_lab_to_hex(c) for c in cols])


def _lab_to_hex(c):
    lab = np.array([[[c[0] * 255 / 100, c[1] + 128, c[2] + 128]]], np.uint8)
    b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def run_inspect(req):
    """Every model on the connected component under (x, y): metrics with limits lifted and whether it passes the
    current limits."""
    from scipy import ndimage as ndi
    s = s_ = req.get("settings", {})
    labels, cols, bg = get_labels(req["path"], int(s.get("colors", 8)), float(s.get("merge_de", 10)))
    x, y = int(round(req["x"])), int(round(req["y"]))
    h, w = labels.shape
    if not (0 <= x < w and 0 <= y < h):
        return dict(error="outside image")
    lab = int(labels[y, x])
    if lab == bg:
        return dict(label=lab, background=True, colour=_lab_to_hex(cols[lab]))
    cc, _ = ndi.label(labels == lab)
    comp = cc == cc[y, x]
    ys, xs = np.nonzero(comp)
    y0, x0 = max(0, ys.min() - 3), max(0, xs.min() - 3)
    y1, x1 = min(h, ys.max() + 4), min(w, xs.max() + 4)
    m = comp[y0:y1, x0:x1]
    hid = fs.occluders(labels[y0:y1, x0:x1], m, lab, bg, int(s_.get("rim", 4)), float(s_.get("underlay", 0.5)))
    opts = req.get("options", {})
    models = []
    for k in KINDS:
        det = fs_detect.DETECTORS[k]
        strict = det(m, hid, **opts.get(k, {}))
        loose = strict or det(m, hid, **{**opts.get(k, {}), **RELAX[k]})
        entry = dict(kind=k, passed=strict is not None)
        if loose is not None:
            entry["metrics"] = {kk: _jsonable(v) for kk, v in loose.metrics.items()}
            entry["n_params"] = loose.n_params
            if loose.kind in ("polygon", "star"):
                entry["k"] = int(loose.params["k"])
            fs_detect._shift(loose, x0, y0, labels.shape)
            entry["pts"] = np.round(_poly_of(loose), 2).tolist()
        else:
            entry["note"] = "no model could be fitted (too few outline points, wrong number of tips/sides, ...)"
        models.append(entry)
    best = fs_detect.detect_shape(m, hid, kinds=KINDS, slack=float(s.get("slack", 0.15)), options=opts)
    so = opts.get("symmetry", {})
    sym = fs.find_symmetry(m, hid, rot_iou=float(so.get("rot_iou", 0.9)), mirror_iou=float(so.get("mirror_iou", 0.9)))
    symd = dict(kind=sym.kind, order=sym.order, rot_score=round(float(sym.rot_score), 3), axes=sym.axes,
                mirror_scores=sym.mirror_scores, centre=[sym.centre[0] + x0, sym.centre[1] + y0],
                scores_by_order={str(k): round(float(v), 3) for k, v in sym.scores_by_order.items()})
    if sym.kind not in ("none", "round"):
        import cv2 as _cv
        sm = fs.symmetrize(m, sym, hid).astype(np.uint8)
        cs, _ = _cv.findContours(sm, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
        if cs:
            symd["symmetrized"] = (max(cs, key=len).reshape(-1, 2) + [x0 + 0.5, y0 + 0.5]).tolist()
    return dict(label=lab, colour=_lab_to_hex(cols[lab]), area=int(comp.sum()), bbox=[int(x0), int(y0), int(x1), int(y1)],
                hidden_share=float(hid.mean()), chosen=best.kind if best else None, models=models, symmetry=symd,
                min_area=int(s.get("min_area", 60)))


def _verdict_file(rel):
    return os.path.join(STATE["verdicts"], os.path.splitext(rel)[0] + ".json")


def load_verdicts(rel):
    p = _verdict_file(rel)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return dict(image=rel, marks=[], missed=[], settings=None)


def save_verdicts(req):
    rel = req["path"]
    _safe_path(rel)
    p = _verdict_file(rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = dict(image=rel, marks=req.get("marks", []), missed=req.get("missed", []), settings=req.get("settings"))
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return dict(saved=os.path.relpath(p, STATE["verdicts"]).replace("\\", "/"))


def summary():
    """Totals over all verdict files."""
    tot = dict(images=0, correct=0, wrong=0, missed=0, by_kind={})
    for root, _, files in os.walk(STATE["verdicts"]):
        for n in files:
            if not n.endswith(".json") or n == "tuned_settings.json":
                continue
            with open(os.path.join(root, n), encoding="utf-8") as fh:
                d = json.load(fh)
            tot["images"] += 1
            for mk in d.get("marks", []):
                bk = tot["by_kind"].setdefault(mk["kind"], dict(correct=0, wrong=0, missed=0))
                bk[mk["verdict"]] += 1
                tot[mk["verdict"]] += 1
            for ms in d.get("missed", []):
                bk = tot["by_kind"].setdefault(ms["kind"], dict(correct=0, wrong=0, missed=0))
                bk["missed"] += 1
                tot["missed"] += 1
    return tot


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                with open(os.path.join(HERE, "index.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif u.path == "/api/list":
                self._send(200, dict(root=STATE["root"], verdicts=STATE["verdicts"], images=list_images()))
            elif u.path == "/api/image":
                p = _safe_path(q["path"])
                ext = os.path.splitext(p)[1].lower()
                ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")
                with open(p, "rb") as fh:
                    self._send(200, fh.read(), ctype)
            elif u.path == "/api/verdicts":
                self._send(200, load_verdicts(q["path"]))
            elif u.path == "/api/tuned":
                tp = os.path.join(STATE["verdicts"], "tuned_settings.json")
                if not os.path.exists(tp):
                    self._send(200, dict(error="no tuned_settings.json yet: run python tools/tune.py first"))
                else:
                    with open(tp, encoding="utf-8") as fh:
                        self._send(200, json.load(fh))
            elif u.path == "/api/summary":
                self._send(200, summary())
            else:
                self._send(404, dict(error="not found"))
        except Exception as e:  # report to the page instead of dropping the connection
            self._send(500, dict(error=str(e)))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            route = {"/api/detect": run_detect, "/api/inspect": run_inspect, "/api/verdicts": save_verdicts}
            fn = route.get(urlparse(self.path).path)
            if fn is None:
                self._send(404, dict(error="not found"))
            else:
                self._send(200, fn(req))
        except Exception as e:
            self._send(500, dict(error=f"{type(e).__name__}: {e}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="folder with test images (subfolders included)")
    ap.add_argument("--verdicts", help="folder for verdict JSON files")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    STATE["root"] = os.path.abspath(a.images)
    STATE["verdicts"] = os.path.abspath(a.verdicts or os.path.join(STATE["root"], os.pardir, "flatshapes_verdicts"))
    url = f"http://localhost:{a.port}"
    print(f"Images:   {STATE['root']}\nVerdicts: {STATE['verdicts']}\nOpen {url}  (Ctrl+C to stop)")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
