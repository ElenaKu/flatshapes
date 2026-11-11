"""Polygons with straight sides (triangles, squares, rhombi, ...): Douglas-Peucker corners, then every side is
refitted as a least-squares line and the corners are the intersections of neighbouring sides. Hidden outline
(objects in front) does not take part in the side fits. New in flatshapes (core v2 had rectangles/triangles only
in the old flat2svg path)."""
import numpy as np

from .result import ShapeFit, outline_points, outline_metrics, fmt


def _line_fit(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]


def _intersect(c1, d1, c2, d2):
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def fit_polygon(mask, hidden=None, max_sides=12, eps_rel=0.02, max_mean=0.8, max_p80=1.3, min_side=4.0):
    """Polygon fitted to one shape mask. Returns ShapeFit(kind="polygon") with params vertices (k, 2), k or None."""
    import cv2
    K = np.asarray(mask, bool)
    P, vis = outline_points(K, hidden, every=1)
    if len(P) < 12:
        return None
    peri = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum())
    approx = cv2.approxPolyDP(P.astype(np.float32).reshape(-1, 1, 2), eps_rel * peri, True).reshape(-1, 2)
    k = len(approx)
    if not 3 <= k <= max_sides:
        return None
    # outline index of each approximate corner
    ci = sorted(int(np.argmin(np.linalg.norm(P - v, axis=1))) for v in approx)
    lines = []
    n = len(P)
    for i in range(k):
        a, b = ci[i], ci[(i + 1) % k]
        idx = np.arange(a, b if b > a else b + n) % n
        m = max(1, len(idx) // 8)  # leave the rounded corner ends out of the side fit
        idx = idx[m:-m] if len(idx) > 2 * m + 2 else idx
        idx = idx[vis[idx]]
        if len(idx) < 3:
            return None
        lines.append(_line_fit(P[idx]))
    V = []
    for i in range(k):
        v = _intersect(*lines[i - 1], *lines[i])
        if v is None:
            return None
        V.append(v)
    V = np.array(V)
    if np.linalg.norm(np.diff(np.vstack([V, V[:1]]), axis=0), axis=1).min() < min_side:
        return None
    dense = np.vstack([V[i] + np.linspace(0, 1, 60, endpoint=False)[:, None] * (V[(i + 1) % k] - V[i]) for i in range(k)])
    met = outline_metrics(K, hidden, dense)
    if met["mean"] > max_mean or met["p80"] > max_p80:
        return None
    M = np.zeros(K.shape, np.uint8)
    cv2.fillPoly(M, [np.round(V * 4).astype(np.int32)], 1, shift=2)
    d = "M" + "L".join(fmt(v) for v in V) + "Z"
    return ShapeFit("polygon", dict(vertices=V, k=k), d, met, M > 0)
