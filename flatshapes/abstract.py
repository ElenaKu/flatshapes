"""Geometric abstraction: every object (region of one colour) rebuilt from a few simple shapes - circles, ellipses,
rectangles, rounded rectangles, triangles, capsules - the way a designer would simplify it. A balloon becomes an
ellipse and a small triangle, a bag a rectangle and an arc of a capsule, a bow two triangles and a circle.

Greedy: the shape that best covers the still uncovered part of the region (without spilling much outside it) is
added, then the next one on what is left, until the shapes cover the region well enough or the shape budget is
spent. Each shape starts from the moments / bounding box of the uncovered part and is polished by a direct search
on the rasterised overlap.
"""
from dataclasses import dataclass

import numpy as np

KINDS = ("circle", "ellipse", "rect", "roundrect", "triangle", "capsule")
WORK = 96  # regions are fitted at about this size (px) and scaled back


@dataclass
class Prim:
    kind: str
    params: tuple       # in image px: circle (cx, cy, r); ellipse (cx, cy, a, b, deg); rect / roundrect /
    #                     capsule (cx, cy, w, h, deg[, r]); triangle (x0, y0, x1, y1, x2, y2)
    color: str = "#000"

    def svg(self):
        k, p, c = self.kind, self.params, self.color
        if k == "circle":
            return f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{p[2]:.1f}" fill="{c}"/>'
        if k == "ellipse":
            return (f'<ellipse cx="{p[0]:.1f}" cy="{p[1]:.1f}" rx="{p[2]:.1f}" ry="{p[3]:.1f}" fill="{c}" '
                    f'transform="rotate({p[4]:.1f} {p[0]:.1f} {p[1]:.1f})"/>')
        if k in ("rect", "roundrect", "capsule"):
            cx, cy, w, h, deg = p[:5]
            r = 0.0 if k == "rect" else (min(w, h) / 2 if k == "capsule" else p[5])
            return (f'<rect x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" '
                    f'fill="{c}" transform="rotate({deg:.1f} {cx:.1f} {cy:.1f})"/>')
        if k == "line":
            return (f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{p[2]:.1f}" y2="{p[3]:.1f}" stroke="{c}" '
                    f'stroke-width="{p[4]:.1f}" stroke-linecap="round"/>')
        pts = " ".join(f"{p[i]:.1f},{p[i + 1]:.1f}" for i in range(0, 6, 2))
        return f'<polygon points="{pts}" fill="{c}"/>'


# ------------------------------------------------------------------ raster of a shape (work coordinates)
def _rot(P, deg, c):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (P - c) @ R.T + c


def outline(kind, p, n=48):
    """Closed outline points of a shape (same coordinates as its params)."""
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[p[0] + p[2] * np.cos(t), p[1] + p[2] * np.sin(t)]
    if kind == "ellipse":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return _rot(np.c_[p[0] + p[2] * np.cos(t), p[1] + p[3] * np.sin(t)], p[4], np.array(p[:2]))
    if kind in ("rect", "roundrect", "capsule"):
        cx, cy, w, h, deg = p[:5]
        w, h = abs(w), abs(h)
        r = 0.0 if kind == "rect" else (min(w, h) / 2 if kind == "capsule" else float(np.clip(p[5], 0, min(w, h) / 2)))
        if r < 0.5:
            P = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        else:
            m = max(3, n // 4)
            pts = []
            for (sx, sy, a0) in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
                a = np.radians(a0 + np.linspace(0, 90, m))
                pts.append(np.c_[sx * (w / 2 - r) + r * np.cos(a), sy * (h / 2 - r) + r * np.sin(a)])
            P = np.vstack(pts)
        return _rot(P + [cx, cy], deg, np.array([cx, cy]))
    return np.asarray(p, float).reshape(3, 2)


def raster(kind, p, shape):
    import cv2
    m = np.zeros(shape, np.uint8)
    P = outline(kind, p)
    if not np.isfinite(P).all():
        return m.astype(bool)
    cv2.fillPoly(m, [np.round(P * 4).astype(np.int32)], 1, shift=2)
    return m > 0


# ------------------------------------------------------------------ start values from the uncovered part
def _starts(T, kind):
    import cv2
    ys, xs = np.nonzero(T)
    if len(xs) < 5:
        return []
    pts = np.c_[xs, ys].astype(np.float32) + 0.5
    c = pts.mean(0)
    (rx, ry), (rw, rh), rdeg = cv2.minAreaRect(pts)
    cov = np.cov(pts.T)
    w_, v_ = np.linalg.eigh(cov)
    a, b = 2 * np.sqrt(max(w_[1], 1e-6)), 2 * np.sqrt(max(w_[0], 1e-6))
    deg = float(np.degrees(np.arctan2(v_[1, 1], v_[0, 1])))
    if kind == "circle":
        (ex, ey), er = cv2.minEnclosingCircle(pts)
        r_area = np.sqrt(len(xs) / np.pi)
        return [(c[0], c[1], r_area), (ex, ey, er * 0.9)]
    if kind == "ellipse":
        return [(c[0], c[1], a, b, deg)]
    if kind == "rect":
        return [(rx, ry, rw, rh, rdeg), (c[0], c[1], 2 * a * 0.85, 2 * b * 0.85, deg)]
    if kind == "roundrect":
        return [(rx, ry, rw, rh, rdeg, 0.25 * min(rw, rh))]
    if kind == "capsule":
        return [(rx, ry, rw, rh, rdeg)]
    _, tri = cv2.minEnclosingTriangle(pts.reshape(-1, 1, 2))
    if tri is None:
        return []
    t = tri.reshape(3, 2)
    t = c + (t - c) * 0.9  # the enclosing triangle is a bit too big for the shape inside
    return [tuple(t.ravel())]


# ------------------------------------------------------------------ fit one shape
def _objective(kind, p, T, R, covered, w_out, w_over):
    P = raster(kind, p, T.shape)
    a = P.sum()
    if a == 0:
        return 1e9
    gain = np.count_nonzero(P & T)
    spill = np.count_nonzero(P & ~R)
    over = np.count_nonzero(P & covered & R)
    return -(gain - w_out * spill - w_over * over)


def fit_shape(T, R, covered, kinds=KINDS, w_out=1.5, w_over=0.15, iters=160):
    """Best single shape for the uncovered part T of region R (bool, work coords). Returns (kind, params, score)
    or None. Score = covered new pixels - w_out * pixels outside the region - w_over * pixels on covered parts."""
    from scipy.optimize import minimize
    best = None
    for kind in kinds:
        for x0 in _starts(T, kind):
            f = lambda x: _objective(kind, tuple(x), T, R, covered, w_out, w_over)
            x0 = np.array(x0, float)
            scale = np.maximum(np.abs(x0) * 0.08, 1.0)
            if kind in ("ellipse", "rect", "roundrect", "capsule"):
                scale[4] = 8.0  # degrees
            simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * scale[i] for i in range(len(x0))])
            sol = minimize(f, x0, method="Nelder-Mead",
                           options=dict(maxfev=iters, initial_simplex=simplex, xatol=0.3, fatol=0.5))
            v = -sol.fun
            if best is None or v > best[2]:
                best = (kind, tuple(sol.x), v)
    return best


def simplify_region(mask, max_shapes=3, kinds=KINDS, cover=0.92, min_gain=0.04, w_out=1.5, work=WORK):
    """Rebuild one region from up to max_shapes simple shapes. Returns (list of (kind, params) in mask coords,
    IoU of their union with the region)."""
    import cv2
    K = np.asarray(mask, bool)
    ys, xs = np.nonzero(K)
    if len(xs) < 4:
        return [], 0.0
    y0, x0 = ys.min(), xs.min()
    crop = K[y0:ys.max() + 1, x0:xs.max() + 1]
    s = min(1.0, work / max(crop.shape))
    pad = 3
    if s < 1.0:
        small = cv2.resize(crop.astype(np.uint8), (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                           interpolation=cv2.INTER_AREA) > 0
    else:
        small = crop
    R = np.pad(small, pad)
    covered = np.zeros_like(R)
    area = R.sum()
    out = []
    for _ in range(max_shapes):
        T = R & ~covered
        if T.sum() < min_gain * area:
            break
        # work on the biggest uncovered blob: a shape for scraps everywhere at once fits none of them
        n, lab, stats, _ = cv2.connectedComponentsWithStats(T.astype(np.uint8), connectivity=8)
        if n <= 1:
            break
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[j, cv2.CC_STAT_AREA] < min_gain * area:
            break
        Tb = lab == j
        r = fit_shape(Tb, R, covered, kinds, w_out=w_out)
        if r is None or r[2] < min_gain * area:
            break
        kind, p, _ = r
        covered |= raster(kind, p, R.shape)
        out.append((kind, p))
        if np.count_nonzero(covered & R) / max(1, np.count_nonzero(covered | R)) >= cover:
            break
    iou = np.count_nonzero(covered & R) / max(1, np.count_nonzero(covered | R))
    # back to mask coordinates
    res = []
    for kind, p in out:
        res.append((kind, _to_image(kind, p, s, x0 - pad / s, y0 - pad / s)))
    return res, float(iou)


def _to_image(kind, p, s, ox, oy):
    p = list(p)
    if kind == "triangle":
        return tuple(v / s + (ox if i % 2 == 0 else oy) for i, v in enumerate(p))
    p[0], p[1] = p[0] / s + ox, p[1] / s + oy
    if kind == "circle":
        p[2] /= s
    elif kind == "ellipse":
        p[2], p[3] = abs(p[2]) / s, abs(p[3]) / s
    else:
        p[2], p[3] = abs(p[2]) / s, abs(p[3]) / s
        if kind == "roundrect":
            p[5] = float(np.clip(p[5] / s, 0, min(p[2], p[3]) / 2))
    return tuple(float(v) for v in p)


def geometrize(labels, palette, background=None, max_shapes=3, kinds=KINDS, min_area=40, cover=0.92,
               w_out=1.5, workers=1, lines=True, min_line=30.0):
    """Whole label map -> list of Prim (bottom to top: bigger regions first, straight strokes on top) and an SVG
    document. palette: list of '#rrggbb' per label. lines: thin straight strokes (strings, frames) are kept as
    lines and left out of the regions (they would vanish otherwise: no filled shape fits a 3 px line)."""
    import cv2
    from scipy import ndimage as ndi
    L = np.asarray(labels)
    H, W = L.shape
    line_prims, strip = [], {}
    if lines:
        from .detect import find_lines
        for lab, f in find_lines(L, background=background, min_len=min_line):
            p0, p1, w = f.params["p0"], f.params["p1"], f.params["width"]
            line_prims.append(Prim("line", (float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), max(1.0, float(w))),
                                   palette[lab]))
            m = strip.setdefault(lab, np.zeros((H, W), np.uint8))
            cv2.line(m, tuple(np.round(np.asarray(p0) * 4).astype(int)), tuple(np.round(np.asarray(p1) * 4).astype(int)),
                     1, max(1, int(round(w + 4))), shift=2)
    regions = []
    for lab in np.unique(L):
        if background is not None and lab == background:
            continue
        own = L == lab
        if lab in strip:
            own &= strip[lab] == 0
        cc, n = ndi.label(own)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            m = cc[sl] == i
            if m.sum() >= min_area:
                regions.append((int(lab), sl[1].start, sl[0].start, m))
    regions.sort(key=lambda r: -int(r[3].sum()))
    args = [(r[3], max_shapes, kinds, cover, w_out) for r in regions]
    if workers != 1 and len(args) > 4:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers or None) as ex:
            fits = list(ex.map(_simplify_task, args, chunksize=4))
    else:
        fits = [_simplify_task(a) for a in args]
    prims = []
    for (lab, x0, y0, _), (shapes, _iou) in zip(regions, fits):
        for kind, p in shapes:
            if kind == "triangle":
                p = tuple(v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(p))
            else:
                p = (p[0] + x0, p[1] + y0) + tuple(p[2:])
            prims.append(Prim(kind, p, palette[lab]))
    prims += line_prims
    bg = palette[background] if background is not None else "none"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<rect width="{W}" height="{H}" fill="{bg}"/>' + "".join(q.svg() for q in prims) + "</svg>")
    return prims, svg


def _simplify_task(a):
    m, max_shapes, kinds, cover, w_out = a
    return simplify_region(m, max_shapes=max_shapes, kinds=kinds, cover=cover, w_out=w_out)


def render(prims, shape, background="#000000"):
    """BGR preview of the shapes (no SVG renderer needed)."""
    import cv2
    H, W = shape
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = _bgr(background)
    for q in prims:
        if q.kind == "line":
            p = q.params
            cv2.line(img, (int(round(p[0] * 4)), int(round(p[1] * 4))), (int(round(p[2] * 4)), int(round(p[3] * 4))),
                     _bgr(q.color), max(1, int(round(p[4]))), cv2.LINE_AA, shift=2)
            continue
        P = outline(q.kind, q.params, 96)
        cv2.fillPoly(img, [np.round(P * 4).astype(np.int32)], _bgr(q.color), lineType=cv2.LINE_AA, shift=2)
    return img


def _bgr(hexcol):
    h = hexcol.lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
