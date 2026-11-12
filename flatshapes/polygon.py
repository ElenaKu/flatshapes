"""Polygons with straight sides (triangles, squares, rhombi, ...), sharp or with rounded corners: Douglas-Peucker corners, then every side is
refitted as a least-squares line and the corners are the intersections of neighbouring sides. Hidden outline
(objects in front) does not take part in the side fits. New in flatshapes (core v2 had rectangles/triangles only
in the old flat2svg path)."""
import numpy as np

from .result import ShapeFit, outline_points, outline_metrics, fmt, eq_radius, limit


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


def fit_polygon(mask, hidden=None, max_sides=12, eps_rel=0.02, max_mean=0.8, max_p80=1.3, min_side=4.0,
                round_corners=True, rel_mean=0.022, rel_p80=0.035, shallow_turn=25.0):
    """Polygon fitted to one shape mask, corners optionally rounded with one common radius. Returns
    ShapeFit(kind="polygon") with params vertices (k, 2) (sharp corner points), k, corner_r, or None.
    Two neighbouring corners that each turn less than `shallow_turn` degrees mean a curve cut into chords: no
    polygon."""
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
    sides = np.linalg.norm(np.diff(np.vstack([V, V[:1]]), axis=0), axis=1)
    if sides.min() < min_side:
        return None
    # a smooth curve cut into chords (the bowl of a D or R) has neighbouring shallow corners; a real polygon not
    din = V - np.roll(V, 1, 0)
    dout = np.roll(V, -1, 0) - V
    turn = np.degrees(np.abs(np.arctan2(din[:, 0] * dout[:, 1] - din[:, 1] * dout[:, 0], (din * dout).sum(1))))
    shallow = turn < shallow_turn
    if (shallow & np.roll(shallow, 1)).any():
        return None
    # one corner radius for all corners (how designers round shapes): grid search, keep the best
    best = None
    rmax = 0.5 * sides.min() if round_corners else 0.0
    from scipy.spatial import cKDTree
    Pfull, vfull = outline_points(K, hidden)  # once: the radius search below only moves the model
    Pq = Pfull[vfull]
    if not len(Pq):
        return None

    def metrics_of(dense):
        d, _ = cKDTree(dense).query(Pq)
        return dict(mean=float(d.mean()), p80=float(np.percentile(d, 80)), max=float(d.max()),
                    visible=float(vfull.mean()))

    # coarse radius grid, then a finer one around the best
    grid = list(np.linspace(0, rmax, 9)) if rmax > 0 else [0.0]
    tried = {}
    for stage in range(2):
        for r in grid:
            r = float(r)
            if r in tried:
                continue
            dense, segs = _rounded(V, r)
            tried[r] = None if dense is None else (metrics_of(dense), dense, segs)
        ok = [(v[0]["mean"], r) for r, v in tried.items() if v is not None]
        if not ok or rmax <= 0:
            break
        r0 = min(ok)[1]
        step = rmax / 8
        grid = list(np.clip(np.linspace(r0 - step, r0 + step, 7), 0, rmax))
    for r, v in tried.items():
        if v is None:
            continue
        met, dense, segs = v
        if best is None or met["mean"] < best[1]["mean"] - 1e-6:
            best = (r, met, dense, segs)
    if best is None:
        return None
    r, met, dense, segs = best
    R = eq_radius(K)
    if met["mean"] > limit(max_mean, rel_mean, R) or met["p80"] > limit(max_p80, rel_p80, R):
        return None
    M = np.zeros(K.shape, np.uint8)
    cv2.fillPoly(M, [np.round(dense * 4).astype(np.int32)], 1, shift=2)
    met["corner_r"] = float(r)
    return ShapeFit("polygon", dict(vertices=V, k=k, corner_r=float(r)), polygon_path(V, r), met, M > 0)


def _rounded(V, r, n_arc=12):
    """Outline of polygon V with every corner rounded by radius r: dense points (n, 2) and a list of
    ('L', p0, p1) / ('A', p0, p1, r, large, sweep, centre) segments. None when r does not fit a side."""
    k = len(V)
    if r <= 1e-6:
        dense = np.vstack([V[i] + np.linspace(0, 1, 60, endpoint=False)[:, None] * (V[(i + 1) % k] - V[i])
                           for i in range(k)])
        return dense, [("L", V[i], V[(i + 1) % k]) for i in range(k)]
    corners = []
    for i in range(k):
        u_in = V[i] - V[i - 1]
        u_out = V[(i + 1) % k] - V[i]
        u_in, u_out = u_in / np.linalg.norm(u_in), u_out / np.linalg.norm(u_out)
        cos_t = float(np.clip(-u_in @ u_out, -1, 1))
        theta = np.arccos(cos_t)  # angle between the two sides at the corner
        if theta < 1e-3 or theta > np.pi - 1e-3:
            return None, None
        t = r / np.tan(theta / 2)
        A, B = V[i] - u_in * t, V[i] + u_out * t
        bis = (-u_in + u_out)
        bis /= np.linalg.norm(bis)
        C = V[i] + bis * (r / np.sin(theta / 2))
        cross = u_in[0] * u_out[1] - u_in[1] * u_out[0]
        corners.append((A, B, C, t, 1 if cross > 0 else 0))
    for i in range(k):  # the two tangent points on each side must not overlap
        if corners[i][3] + corners[(i + 1) % k][3] > np.linalg.norm(V[(i + 1) % k] - V[i]) + 1e-6:
            return None, None
    pts, segs = [], []
    for i in range(k):
        A, B, C, _, sweep = corners[i]
        a0 = np.arctan2(A[1] - C[1], A[0] - C[0])
        a1 = np.arctan2(B[1] - C[1], B[0] - C[0])
        d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        ang = a0 + d * np.linspace(0, 1, n_arc)
        pts.append(C + r * np.c_[np.cos(ang), np.sin(ang)])
        segs.append(("A", A, B, float(r), 0, sweep, C))
        nA = corners[(i + 1) % k][0]
        pts.append(B + np.linspace(0, 1, 40, endpoint=False)[1:, None] * (nA - B))
        segs.append(("L", B, nA))
    return np.vstack(pts), segs


def polygon_path(V, r=0.0):
    """SVG path data of polygon V, corners rounded by r."""
    if r <= 1e-6:
        return "M" + "L".join(fmt(v) for v in V) + "Z"
    _, segs = _rounded(np.asarray(V, float), r)
    from .svg import segments_to_d
    return segments_to_d(segs, closed=True)
