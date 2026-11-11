"""Boundary geometry fitting (ported from flat2svg core v2, core2/geom.py).

A boundary chain between two corners / junctions (dense, already smoothed polyline, 1x pixel units) is replaced by the
fewest primitives that stay within a tolerance: a straight line, a circular arc, or cubic Beziers (Schneider fit with
G1 continuity at the split points). Fitted chains are lists of segments

    ("L", p0, p1)                      line
    ("A", p0, p1, r, large, sweep)     circular arc (SVG arc flags)
    ("C", p0, c1, c2, p1)              cubic Bezier

A chain is fitted once in forward direction; the neighbour on the other side uses `reverse_segments`, so both sides
of a shared edge are the same curve (no gaps, no overlaps).
"""
import numpy as np


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(2)


def _arclen(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def end_tangents(P, window):
    """Unit tangents at the first and last point (pointing along the curve), from the points within `window` px."""
    s = _arclen(P)
    i = int(np.searchsorted(s, min(window, s[-1] * 0.5)))
    i = max(1, min(i, len(P) - 1))
    j = int(np.searchsorted(s, s[-1] - min(window, s[-1] * 0.5)))
    j = max(0, min(j, len(P) - 2))
    return _unit(P[i] - P[0]), _unit(P[-1] - P[j])


# ---------------------------------------------------------------- line / arc

def _line_err(P):
    d = P[-1] - P[0]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return float(np.linalg.norm(P - P[0], axis=1).max())
    nrm = np.array([-d[1], d[0]]) / n
    return float(np.abs((P - P[0]) @ nrm).max())


def _try_arc(P, tol, min_turn_deg=6.0, r_max=5000.0):
    """One circular arc through the end points of P within `tol`; returns (segment, error) or None."""
    n = len(P)
    if n < 5:
        return None
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(n)]
    b = (P ** 2).sum(1)
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    c0 = sol[:2]
    r0 = float(np.sqrt(max(sol[2] + c0 @ c0, 0.0)))
    if not np.isfinite(r0) or r0 > r_max or r0 < 0.5:
        return None
    p0, p1 = P[0], P[-1]
    chord = np.linalg.norm(p1 - p0)
    if chord < 6.0:
        return None
    r = max(r0, chord / 2 + 1e-6)
    m = (p0 + p1) / 2
    h = chord / 2
    dist = np.sqrt(max(r * r - h * h, 0.0))
    nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / chord
    cands = [m + dist * nrm, m - dist * nrm]
    c = min(cands, key=lambda q: np.linalg.norm(q - c0))
    err = float(np.abs(np.linalg.norm(P - c, axis=1) - r).max())
    if err > tol:
        return None
    ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    span = float(ang[-1] - ang[0])
    if abs(np.degrees(span)) < min_turn_deg or abs(span) > 1.75 * np.pi:
        return None
    # the samples must run monotonically around the centre (no folds)
    dang = np.diff(ang)
    if not (np.all(dang > -1e-3) or np.all(dang < 1e-3)):
        return None
    sweep = 1 if span > 0 else 0
    large = 1 if abs(span) > np.pi else 0
    return ("A", p0.copy(), p1.copy(), float(r), large, sweep, c.copy()), err


# ---------------------------------------------------------------- Schneider cubic fit

def _bez(c, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return mt ** 3 * c[0] + 3 * mt ** 2 * t * c[1] + 3 * mt * t ** 2 * c[2] + t ** 3 * c[3]


def _bez_d1(c, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return 3 * mt ** 2 * (c[1] - c[0]) + 6 * mt * t * (c[2] - c[1]) + 3 * t ** 2 * (c[3] - c[2])


def _bez_d2(c, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return 6 * mt * (c[2] - 2 * c[1] + c[0]) + 6 * t * (c[3] - 2 * c[2] + c[1])


def _generate(P, u, t1, t2):
    p0, p3 = P[0], P[-1]
    chord = np.linalg.norm(p3 - p0)
    b1 = 3 * (1 - u) ** 2 * u
    b2 = 3 * (1 - u) * u ** 2
    b0 = (1 - u) ** 3
    b3 = u ** 3
    A1 = b1[:, None] * t1
    A2 = b2[:, None] * t2
    C = np.array([[(A1 * A1).sum(), (A1 * A2).sum()], [(A1 * A2).sum(), (A2 * A2).sum()]])
    tmp = P - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3
    X = np.array([(A1 * tmp).sum(), (A2 * tmp).sum()])
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    a1 = a2 = chord / 3
    if abs(det) > 1e-12:
        x1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        x2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
        if x1 > 1e-6 * chord and x2 > 1e-6 * chord and x1 < 1.0 * chord and x2 < 1.0 * chord:
            a1, a2 = x1, x2
    return np.array([p0, p0 + a1 * t1, p3 + a2 * t2, p3])


def _reparam(P, c, u):
    d = _bez(c, u) - P
    d1 = _bez_d1(c, u)
    d2 = _bez_d2(c, u)
    num = (d * d1).sum(1)
    den = (d1 * d1).sum(1) + (d * d2).sum(1)
    ok = np.abs(den) > 1e-12
    out = u.copy()
    out[ok] = u[ok] - num[ok] / den[ok]
    return np.clip(out, 0.0, 1.0)


def _max_err(P, c, u):
    d = np.linalg.norm(_bez(c, u) - P, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), i


def _fit_cubic(P, t1, t2, tol, depth=0):
    s = _arclen(P)
    u = s / max(s[-1], 1e-12)
    c = _generate(P, u, t1, t2)
    err, split = _max_err(P, c, u)
    if err <= tol:
        return [("C", c[0], c[1], c[2], c[3])], err
    for _ in range(5):
        u = _reparam(P, c, u)
        c = _generate(P, u, t1, t2)
        err, split = _max_err(P, c, u)
        if err <= tol:
            return [("C", c[0], c[1], c[2], c[3])], err
    return None, err, split


def _fit(P, t1, t2, tol, line_tol, arc_tol, depth=0):
    """Fewest primitives within tolerance: line > arc > one cubic > split at the worst point."""
    n = len(P)
    if n == 2 or _line_err(P) <= line_tol:
        return [("L", P[0].copy(), P[-1].copy())]
    arc = _try_arc(P, arc_tol)
    if arc is not None:
        return [arc[0]]
    res = _fit_cubic(P, t1, t2, tol)
    if res[0] is not None:
        return res[0]
    split = res[2]
    split = max(2, min(split, n - 3)) if n >= 6 else n // 2
    if n < 6 or depth > 40:
        c = _generate(P, _arclen(P) / max(_arclen(P)[-1], 1e-12), t1, t2)
        return [("C", c[0], c[1], c[2], c[3])]
    tc = _unit(P[split - 1] - P[split + 1])
    left = _fit(P[:split + 1], t1, tc, tol, line_tol, arc_tol, depth + 1)
    right = _fit(P[split:], -tc, t2, tol, line_tol, arc_tol, depth + 1)
    return left + right


def fit_open(P, tol=0.3, line_tol=0.2, arc_tol=0.3, window=1.5):
    """Fit an open dense polyline (>= 2 points, 1x units) -> list of segments from P[0] to P[-1]."""
    P = np.asarray(P, np.float64)
    keep = np.ones(len(P), bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9
    P = P[keep]
    if len(P) < 2:
        return []
    if len(P) == 2:
        return [("L", P[0].copy(), P[-1].copy())]
    t1, t2 = end_tangents(P, window)
    return _fit(P, t1, -t2, tol, line_tol, arc_tol)


def fit_closed(Q, tol=0.3, line_tol=0.2, arc_tol=0.3, window=1.5):
    """Fit a smooth closed dense loop Q (no repeated last point): two open halves with common tangents."""
    Q = np.asarray(Q, np.float64)
    n = len(Q)
    if n < 6:
        return [("L", Q[i].copy(), Q[(i + 1) % n].copy()) for i in range(n)]
    s = _arclen(np.vstack([Q, Q[:1]]))
    h = int(np.searchsorted(s, s[-1] / 2))
    h = max(3, min(h, n - 3))

    def tangent(i):
        a = Q[(i - 3) % n]
        b = Q[(i + 3) % n]
        return _unit(b - a)

    ta, tb = tangent(0), tangent(h)
    first = _fit(Q[:h + 1], ta, -tb, tol, line_tol, arc_tol)
    second = _fit(np.vstack([Q[h:], Q[:1]]), tb, -ta, tol, line_tol, arc_tol)
    return first + second


def reverse_segments(segs):
    out = []
    for sg in reversed(segs):
        k = sg[0]
        if k == "L":
            out.append(("L", sg[2], sg[1]))
        elif k == "A":
            out.append(("A", sg[2], sg[1], sg[3], sg[4], 1 - sg[5], sg[6]))
        else:
            out.append(("C", sg[4], sg[3], sg[2], sg[1]))
    return out


def sample_segment(sg, n=8):
    """Points along a segment (excluding the start point, including the end point)."""
    t = np.linspace(0, 1, n + 1)[1:]
    if sg[0] == "L":
        return sg[1] + t[:, None] * (sg[2] - sg[1])
    if sg[0] == "C":
        return _bez(np.array(sg[1:5]), t)
    c = sg[6]
    a0 = np.arctan2(*(sg[1] - c)[::-1])
    a1 = np.arctan2(*(sg[2] - c)[::-1])
    d = a1 - a0
    if sg[5] == 1:
        d = d % (2 * np.pi)
    else:
        d = -((-d) % (2 * np.pi))
    ang = a0 + t * d
    return c + sg[3] * np.stack([np.cos(ang), np.sin(ang)], 1)


def ring_polygon(ring, n=8):
    pts = [ring[0][1][None]]
    for sg in ring:
        pts.append(sample_segment(sg, n))
    return np.vstack(pts)


def straighten(segs):
    """The same chain as straight lines between the segment end points."""
    out = []
    for sg in segs:
        p0 = sg[1]
        p1 = sg[2] if sg[0] != "C" else sg[4]
        if np.linalg.norm(p1 - p0) > 1e-9:
            out.append(("L", p0, p1))
    return out


def self_intersects(poly):
    """True when the closed polyline crosses itself (proper crossings only)."""
    a = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    n = len(a)
    b = np.roll(a, -1, axis=0)
    for i in range(n):
        d1 = b[i] - a[i]
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j < n - 1]
        if j.size == 0:
            continue
        d2 = b[j] - a[j]
        den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
        ok = np.abs(den) > 1e-12
        w = a[j] - a[i]
        t = (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / np.where(ok, den, 1)
        u = (w[:, 0] * d1[1] - w[:, 1] * d1[0]) / np.where(ok, den, 1)
        if (ok & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)).any():
            return True
    return False


def crossing_points(poly, cap=50):
    """Points where the closed polyline crosses itself (proper crossings only); [] when it does not. Sweep over x, so
    long rings are cheap (the same test as svg_check PATH_LOOP)."""
    a = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    m = len(a)
    if m < 4:
        return []
    b = np.roll(a, -1, axis=0)
    xmin, xmax = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
    ymin, ymax = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    order = np.argsort(xmin, kind="stable")
    A, B = a[order], b[order]
    xmin, xmax, ymin, ymax = xmin[order], xmax[order], ymin[order], ymax[order]
    out = []
    for k in range(m):
        e = int(np.searchsorted(xmin, xmax[k], side="right"))
        if e <= k + 1:
            continue
        idx = np.arange(k + 1, e)
        idx = idx[(ymin[idx] <= ymax[k]) & (ymax[idx] >= ymin[k])]
        if not idx.size:
            continue
        p, q = A[idx], B[idx]
        d = B[k] - A[k]
        o1 = d[0] * (p[:, 1] - A[k][1]) - d[1] * (p[:, 0] - A[k][0])
        o2 = d[0] * (q[:, 1] - A[k][1]) - d[1] * (q[:, 0] - A[k][0])
        e2 = q - p
        o3 = e2[:, 0] * (A[k][1] - p[:, 1]) - e2[:, 1] * (A[k][0] - p[:, 0])
        o4 = e2[:, 0] * (B[k][1] - p[:, 1]) - e2[:, 1] * (B[k][0] - p[:, 0])
        hit = (o1 * o2 < 0) & (o3 * o4 < 0)
        for j in np.nonzero(hit)[0]:
            oj, ok = int(order[idx[j]]), int(order[k])
            if abs(oj - ok) <= 1 or abs(oj - ok) == m - 1:
                continue
            t = o3[j] / (o3[j] - o4[j])
            out.append(A[k] + t * d)
            if len(out) >= cap:
                return out
    return out
