"""Circles: algebraic (Kasa) fit, RANSAC, occlusion-aware fit on a mask, exact SVG arcs for outline chains, and
equal radius for groups of same-size discs (polka dots).

Ported from flat2svg core v2 (core2/stage3.py, core2/primitives.py, core2/stars.py).
"""
import numpy as np

from .result import ShapeFit, outline_points, fmt, limit

K_ARC = 0.5522847498  # cubic Bezier control distance for a quarter circle


def kasa(P):
    """Algebraic circle fit of points P (n, 2) -> (cx, cy, r) or None."""
    P = np.asarray(P, float)
    A = np.c_[2 * P, np.ones(len(P))]
    try:
        sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    return (float(sol[0]), float(sol[1]), float(np.sqrt(r2))) if r2 > 0 else None


def ransac_circle(P, tol, iters=80, min_inliers=8, rng=None):
    """Circle through most points of P within `tol` px. Returns ((cx, cy, r), inlier mask) or None."""
    P = np.asarray(P, float)
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(P)
    if n < 3:
        return None
    best = None
    for _ in range(iters):
        f = kasa(P[rng.choice(n, 3, replace=False)])
        if f is None:
            continue
        inl = np.abs(np.hypot(P[:, 0] - f[0], P[:, 1] - f[1]) - f[2]) <= tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_inliers:
        return None
    f = kasa(P[best])
    if f is None:
        return None
    inl = np.abs(np.hypot(P[:, 0] - f[0], P[:, 1] - f[1]) - f[2]) <= tol
    return (kasa(P[inl]) or f), inl


def coverage_deg(P, c, inl=None):
    """Angular extent (deg) of the points around centre c: 360 minus the largest empty gap."""
    P = np.asarray(P, float)
    if inl is not None:
        P = P[inl]
    if len(P) < 3:
        return 0.0
    a = np.sort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    return float(360.0 - np.degrees(np.diff(np.r_[a, a[0] + 2 * np.pi]).max()))


def circle_path(cx, cy, r):
    """Closed circle as four cubic quarters (4 nodes, what Illustrator writes)."""
    q = K_ARC * r
    return (f"M{fmt((cx + r, cy))}C{fmt((cx + r, cy + q))} {fmt((cx + q, cy + r))} {fmt((cx, cy + r))}"
            f"C{fmt((cx - q, cy + r))} {fmt((cx - r, cy + q))} {fmt((cx - r, cy))}"
            f"C{fmt((cx - r, cy - q))} {fmt((cx - q, cy - r))} {fmt((cx, cy - r))}"
            f"C{fmt((cx + q, cy - r))} {fmt((cx + r, cy - q))} {fmt((cx + r, cy))}Z")


def fit_circle(mask, hidden=None, min_r=3.0, max_mean=0.45, max_p80=0.9, best_share=0.9, min_visible=0.3,
               max_gap_deg=200.0, rel_mean=0.022, rel_p80=0.035):
    """Circle fitted to the visible outline of one shape mask. Returns ShapeFit(kind="circle") or None.
    The visible outline must go round the circle (largest angular gap <= max_gap_deg): a straight edge is not a
    circle. Error limits grow with the radius: max(max_mean, rel_mean * r), max(max_p80, rel_p80 * r)."""
    import cv2
    from scipy.optimize import least_squares
    K = np.asarray(mask, bool)
    h, w = K.shape
    P, vis = outline_points(K, hidden)
    if len(P) < 20 or vis.mean() < min_visible:
        return None
    Pv = P[vis]
    f0 = kasa(Pv)
    if f0 is None or f0[2] < min_r:
        return None
    res = lambda x: np.hypot(Pv[:, 0] - x[0], Pv[:, 1] - x[1]) - x[2]
    cx, cy, r = least_squares(res, np.array(f0), loss="soft_l1", f_scale=0.5).x
    dd = np.abs(res((cx, cy, r)))
    best = np.sort(dd)[: max(1, int(len(dd) * best_share))]
    if r < min_r or best.mean() > limit(max_mean, rel_mean, r) or np.percentile(dd, 80) > limit(max_p80, rel_p80, r):
        return None
    if 360.0 - coverage_deg(Pv, (cx, cy)) > max_gap_deg:
        return None
    M = np.zeros((h, w), np.uint8)
    cv2.circle(M, (int(round(cx * 4)), int(round(cy * 4))), int(round(r * 4)), 1, -1, shift=2)
    met = dict(mean=float(dd.mean()), p80=float(np.percentile(dd, 80)), max=float(dd.max()), visible=float(vis.mean()))
    return ShapeFit("circle", dict(cx=float(cx), cy=float(cy), r=float(r)), circle_path(cx, cy, r), met, M > 0)


def circle_segments(P, c, r, closed):
    """Exact arc segments ('A', p0, p1, r, large, sweep, centre) for an outline chain P lying on circle (c, r).
    A closed chain becomes two half circles through its first point; an open chain keeps its end points and is
    split into arcs of at most 110 degrees. Returns None when P does not run monotonically around the centre."""
    c = np.asarray(c, float)
    P = np.asarray(P, float)
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    if closed:
        sgn = 1.0 if ang[-1] >= ang[0] else -1.0
        a = ang[0] + sgn * np.array([0.0, np.pi, 2 * np.pi])
        pts = [c + r * np.array([np.cos(t), np.sin(t)]) for t in a]
        pts[-1] = pts[0].copy()
        sweep = 1 if sgn > 0 else 0
        return [("A", pts[i], pts[i + 1], float(r), 0, sweep, c.copy()) for i in range(2)]
    span = float(ang[-1] - ang[0])
    if abs(span) < np.radians(6) or abs(span) > 2 * np.pi:
        return None
    if np.any(np.diff(ang) * np.sign(span) < -0.15):
        return None
    n = int(np.ceil(abs(span) / np.radians(110)))
    edges = ang[0] + span * np.arange(n + 1) / n
    pts = [P[0].copy()] + [c + r * np.array([np.cos(t), np.sin(t)]) for t in edges[1:-1]] + [P[-1].copy()]
    sweep = 1 if span > 0 else 0
    segs = []
    for i in range(n):
        p0, p1 = pts[i], pts[i + 1]
        chord = float(np.linalg.norm(p1 - p0))
        if chord < 1e-6:
            return None
        rr = max(r, chord / 2 + 1e-6)
        m = (p0 + p1) / 2
        d = np.sqrt(max(rr * rr - (chord / 2) ** 2, 0.0))
        nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / chord
        cc = min((m + d * nrm, m - d * nrm), key=lambda q: np.linalg.norm(q - c))
        segs.append(("A", p0.copy(), p1.copy(), float(rr), int(abs(span) / n > np.pi), sweep, cc))
    return segs


def equalize_radii(circles, colours=None, size_rel=0.12, size_tol=1.0, group_de=4.0, min_group=3):
    """Same-size dots get one radius. circles: list of ShapeFit(kind="circle"); colours: optional list of CIELAB
    colours (same length). A group = at least `min_group` circles with radii within `size_rel` (relative) and
    colours within `group_de` (CIEDE2000); the group gets its median radius only when no circle moves more than
    `size_tol` px. Changes the fits in place; returns the number of groups."""
    from skimage.color import deltaE_ciede2000
    idx = sorted(range(len(circles)), key=lambda i: circles[i].params["r"])
    done, groups = set(), 0

    def near_colour(i, j):
        if colours is None:
            return True
        a = np.asarray(colours[i], np.float32)[None]
        b = np.asarray(colours[j], np.float32)[None]
        return float(deltaE_ciede2000(a, b)[0]) <= group_de

    for i in idx:
        if i in done:
            continue
        ri = circles[i].params["r"]
        grp = [i] + [j for j in idx if j != i and j not in done
                     and abs(circles[j].params["r"] - ri) <= size_rel * max(ri, circles[j].params["r"])
                     and near_colour(i, j)]
        if len(grp) < min_group:
            continue
        med = float(np.median([circles[k].params["r"] for k in grp]))
        if all(abs(circles[k].params["r"] - med) <= size_tol for k in grp):
            groups += 1
            for k in grp:
                p = circles[k].params
                p["r"] = med
                circles[k].path = circle_path(p["cx"], p["cy"], med)
            done.update(grp)
    return groups
