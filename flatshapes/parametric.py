"""More simple shapes, each a small parametric model fitted to the visible outline (robust least squares, the same
way as the star): ring, circular segment (half circle and other cut circles), teardrop, heart, crescent.

Every model is a function params -> dense outline points; the fit minimises the distance of the visible outline
points to that outline. Start values come from cheap moments of the mask. Each detector returns a ShapeFit with
exact SVG path data (arcs where the shape has arcs) or None.
"""
import numpy as np

from .result import ShapeFit, outline_points, fmt, eq_radius, limit

TWO_PI = 2 * np.pi


# ------------------------------------------------------------------ helpers
def _rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _visible(mask, hidden, cap=160):
    P, vis = outline_points(mask, hidden)
    Pv = P[vis]
    return P, vis, Pv[:: max(1, len(Pv) // cap)], Pv


def _d_seg(P, a, b):
    """Distance of points P (n, 2) to segment a-b."""
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.hypot(*(P - (a + t[:, None] * ab)).T)


def _d_arc(P, c, r, a0, span):
    """Distance of points P to the arc of circle (c, r) from angle a0 counter-clockwise (y down: increasing angle)
    over `span` radians."""
    v = P - c
    ang = (np.arctan2(v[:, 1], v[:, 0]) - a0) % TWO_PI
    on = ang <= span
    d_circ = np.abs(np.hypot(v[:, 0], v[:, 1]) - r)
    e0 = c + r * np.array([np.cos(a0), np.sin(a0)])
    e1 = c + r * np.array([np.cos(a0 + span), np.sin(a0 + span)])
    d_end = np.minimum(np.hypot(*(P - e0).T), np.hypot(*(P - e1).T))
    return np.where(on, d_circ, d_end)


def _local(P, cx, cy, th):
    """Points in the model's local frame (centre at 0, axis along +x)."""
    return (P - [cx, cy]) @ _rot(th)


def _dist_segment(x, P):
    cx, cy, r, th, h = x
    if r <= 0 or not -r < h < r:
        return None
    L = _local(P, cx, cy, th)
    a = np.arccos(h / r)
    return np.minimum(_d_arc(L, np.zeros(2), r, a, TWO_PI - 2 * a),
                      _d_seg(L, np.array([h, -r * np.sin(a)]), np.array([h, r * np.sin(a)])))


def _dist_tear(x, P):
    cx, cy, r, th, Lt = x
    if r <= 0 or Lt <= r * 1.02:
        return None
    L = _local(P, cx, cy, th)
    a = np.arccos(r / Lt)
    tip = np.array([Lt, 0.0])
    t0, t1 = np.array([r * np.cos(a), r * np.sin(a)]), np.array([r * np.cos(a), -r * np.sin(a)])
    return np.minimum.reduce([_d_arc(L, np.zeros(2), r, a, TWO_PI - 2 * a), _d_seg(L, t1, tip), _d_seg(L, t0, tip)])


def _dist_crescent(x, P):
    cx, cy, r1, th, dd, r2 = x
    if r1 <= 0 or r2 <= 0 or not abs(r1 - r2) < dd < r1 + r2:
        return None
    xi = (dd * dd + r1 * r1 - r2 * r2) / (2 * dd)
    yi2 = r1 * r1 - xi * xi
    if yi2 <= 0:
        return None
    yi = np.sqrt(yi2)
    L = _local(P, cx, cy, th)
    a1 = np.arctan2(yi, xi)
    a2 = np.arctan2(yi, xi - dd)
    # outer arc: angles a1 .. 2pi - a1 on circle 1; inner arc: angles a2 .. 2pi - a2 on circle 2 (through pi)
    return np.minimum(_d_arc(L, np.zeros(2), r1, a1, TWO_PI - 2 * a1),
                      _d_arc(L, np.array([dd, 0.0]), r2, a2, TWO_PI - 2 * a2))


ANALYTIC = {}


def _fit(model, x0, Pq, lo=None, hi=None, nfev=80):
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    dist = ANALYTIC.get(model)

    def resid(x):
        if dist is not None:  # exact distance to the arcs and lines: no point sampling, no KD tree
            d = dist(x, Pq)
            return np.full(len(Pq), 50.0) if d is None or not np.isfinite(d).all() else d
        Q = model(x, 24)
        if Q is None or not np.isfinite(Q).all():
            return np.full(len(Pq), 50.0)
        return cKDTree(Q).query(Pq)[0]

    bounds = (lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    try:
        sol = least_squares(resid, np.asarray(x0, float), bounds=bounds, loss="soft_l1", f_scale=1.0,
                            max_nfev=nfev)
    except Exception:
        return None
    return sol.x


def _accept(model, x, mask, Pall, R, max_mean, max_p80, rel_mean, rel_p80, kind, params, path, vis, hidden=None,
            min_iou=0.88):
    """Metrics on all visible outline points; overlap of the model with the mask (outside hidden parts) at least
    min_iou - a curve hugging one side of a thin sliver or a speck is not the shape; ShapeFit or None."""
    import cv2
    from scipy.spatial import cKDTree
    Q = model(x, 120)
    if Q is None or not np.isfinite(Q).all():
        return None
    d, _ = cKDTree(Q).query(Pall)
    mean, p80 = float(d.mean()), float(np.percentile(d, 80))
    if mean > limit(max_mean, rel_mean, R) or p80 > limit(max_p80, rel_p80, R):
        return None
    M = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(M, [np.round(Q * 4).astype(np.int32)], 1, shift=2)
    K = np.asarray(mask, bool)
    care = np.ones_like(K) if hidden is None else ~cv2.dilate(np.asarray(hidden, np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    u = np.count_nonzero((K | (M > 0)) & care)
    iou = np.count_nonzero(K & (M > 0) & care) / u if u else 0.0
    if iou < min_iou:
        return None
    met = dict(mean=mean, p80=p80, max=float(d.max()), visible=float(vis.mean()), iou=float(iou))
    return ShapeFit(kind, params, path, met, M > 0)


def _moments(mask):
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()]) + 0.5
    cov = np.cov(np.c_[xs, ys].T)
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]
    return c, float(np.arctan2(major[1], major[0])), np.sqrt(np.maximum(w, 1e-9))


# ------------------------------------------------------------------ ring
def fit_ring(mask, hidden=None, max_mean=0.5, max_p80=1.0, rel_mean=0.022, rel_p80=0.035, min_hole=0.05,
             max_offset=0.08):
    """Ring (annulus): outer and inner outline both circles with (about) one centre. params: cx, cy, r_out, r_in.
    The SVG path holds both circles; draw it with fill-rule="evenodd"."""
    import cv2
    from .circle import kasa, circle_path
    m8 = np.asarray(mask, np.uint8)
    cs, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or not len(cs):
        return None
    hier = hier[0]
    outer = max((i for i in range(len(cs)) if hier[i][3] < 0), key=lambda i: cv2.contourArea(cs[i]))
    holes = [i for i in range(len(cs)) if hier[i][3] == outer]
    hole = max(holes, key=lambda i: cv2.contourArea(cs[i])) if holes else None
    if hole is None or cv2.contourArea(cs[hole]) < min_hole * cv2.contourArea(cs[outer]):
        if hidden is None or not np.asarray(hidden, bool).any():
            return None
        # an object in front cuts the ring open (a C shape, no hole left): the one outline holds visible pieces
        # of both circles - split it into the circle found first and the rest
        return _open_ring(cs[outer].reshape(-1, 2).astype(float) + 0.5, m8, hidden, max_mean, max_p80, rel_mean,
                          rel_p80, max_offset)
    # pixel-centre contours: outer edge lies half a pixel out, hole edge half a pixel in
    Po = cs[outer].reshape(-1, 2).astype(float) + 0.5
    Ph = cs[hole].reshape(-1, 2).astype(float) + 0.5
    # circles from the visible parts only (RANSAC: an object in front cuts a bite out of the outer edge)
    from .circle import ransac_circle
    hid = np.asarray(hidden, bool) if hidden is not None else np.zeros(m8.shape, bool)
    hid = cv2.dilate(hid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def vis_pts(Q):
        q = np.clip(Q.astype(int), 0, [m8.shape[1] - 1, m8.shape[0] - 1])
        return Q[~hid[q[:, 1], q[:, 0]]]

    def circ(Q):
        Qv = vis_pts(Q)
        if len(Qv) < 8:
            return None
        rc = ransac_circle(Qv[:: max(1, len(Qv) // 300)], tol=1.0, iters=60)
        return rc[0] if rc is not None and rc[1].mean() > 0.6 else None

    fo, fh = circ(Po), circ(Ph)
    if fo is None or fh is None:
        return None
    ro, rh = fo[2] + 0.5, fh[2] - 0.5
    if rh <= 0 or rh >= ro:
        return None
    if np.hypot(fo[0] - fh[0], fo[1] - fh[1]) > max_offset * ro:
        return None
    cx, cy = (fo[0] * len(Po) + fh[0] * len(Ph)) / (len(Po) + len(Ph)), (fo[1] * len(Po) + fh[1] * len(Ph)) / (len(Po) + len(Ph))
    Pov, Phv = vis_pts(Po), vis_pts(Ph)
    if len(Pov) + len(Phv) < 0.4 * (len(Po) + len(Ph)):
        return None
    do = np.abs(np.hypot(Pov[:, 0] - cx, Pov[:, 1] - cy) - (ro - 0.5))
    dh = np.abs(np.hypot(Phv[:, 0] - cx, Phv[:, 1] - cy) - (rh + 0.5))
    d = np.r_[do, dh]
    mean, p80 = float(d.mean()), float(np.percentile(d, 80))
    if mean > limit(max_mean, rel_mean, ro) or p80 > limit(max_p80, rel_p80, ro):
        return None
    M = np.zeros(m8.shape, np.uint8)
    cv2.circle(M, (int(round(cx * 4)), int(round(cy * 4))), int(round(ro * 4)), 1, -1, shift=2)
    cv2.circle(M, (int(round(cx * 4)), int(round(cy * 4))), int(round(rh * 4)), 0, -1, shift=2)
    path = circle_path(cx, cy, ro) + circle_path(cx, cy, rh)  # draw with fill-rule="evenodd"
    return ShapeFit("ring", dict(cx=float(cx), cy=float(cy), r_out=float(ro), r_in=float(rh)), path,
                    dict(mean=mean, p80=p80, max=float(d.max()), visible=1.0), M > 0)


def _open_ring(P, m8, hidden, max_mean, max_p80, rel_mean, rel_p80, max_offset):
    import cv2
    from .circle import ransac_circle, circle_path
    hid = cv2.dilate(np.asarray(hidden, np.uint8), np.ones((5, 5), np.uint8)) > 0
    q = np.clip(P.astype(int), 0, [m8.shape[1] - 1, m8.shape[0] - 1])
    Pv = P[~hid[q[:, 1], q[:, 0]]]
    if len(Pv) < 30:
        return None
    Pv = Pv[:: max(1, len(Pv) // 400)]
    r1 = ransac_circle(Pv, tol=1.0, iters=80)
    if r1 is None:
        return None
    rest = Pv[~r1[1]]
    if len(rest) < 10:
        return None
    r2 = ransac_circle(rest, tol=1.0, iters=80)
    if r2 is None:
        return None
    (ax, ay, ar), (bx, by, br) = r1[0], r2[0]
    ro, rh = max(ar, br), min(ar, br)
    if rh < 0.2 * ro or np.hypot(ax - bx, ay - by) > max_offset * ro:
        return None
    w1, w2 = r1[1].sum(), r2[1].sum()
    cx, cy = (ax * w1 + bx * w2) / (w1 + w2), (ay * w1 + by * w2) / (w1 + w2)
    d = np.minimum(np.abs(np.hypot(Pv[:, 0] - cx, Pv[:, 1] - cy) - ro), np.abs(np.hypot(Pv[:, 0] - cx, Pv[:, 1] - cy) - rh))
    mean, p80 = float(d.mean()), float(np.percentile(d, 80))
    if mean > limit(max_mean, rel_mean, ro) or p80 > limit(max_p80, rel_p80, ro):
        return None
    M = np.zeros(m8.shape, np.uint8)
    cv2.circle(M, (int(round(cx * 4)), int(round(cy * 4))), int(round(ro * 4)), 1, -1, shift=2)
    cv2.circle(M, (int(round(cx * 4)), int(round(cy * 4))), int(round(rh * 4)), 0, -1, shift=2)
    # the ring must explain the visible shape (a C of the right thickness)
    care = ~hid
    u = np.count_nonzero(((m8 > 0) | (M > 0)) & care)
    if not u or np.count_nonzero((m8 > 0) & (M > 0) & care) / u < 0.85:
        return None
    return ShapeFit("ring", dict(cx=float(cx), cy=float(cy), r_out=float(ro), r_in=float(rh)),
                    circle_path(cx, cy, ro) + circle_path(cx, cy, rh),
                    dict(mean=mean, p80=p80, max=float(d.max()), visible=float(len(Pv) / max(1, len(P)))), M > 0)


# ------------------------------------------------------------------ circular segment (half circle etc.)
def _segment_model(x, n):
    cx, cy, r, th, h = x
    if r <= 0 or not -r < h < r:
        return None
    a = np.arccos(np.clip(h / r, -1, 1))          # half opening of the arc, measured from the axis away from the cut
    # local frame: cut line at x = h (perpendicular to the axis), the arc on the far side (x < h)
    t = np.linspace(a, TWO_PI - a, n)
    arc = np.c_[r * np.cos(t), r * np.sin(t)]
    chord = np.c_[np.full(n // 2, h), np.linspace(-r * np.sin(a), r * np.sin(a), n // 2)]
    return np.vstack([arc, chord]) @ _rot(th).T + [cx, cy]


def fit_segment(mask, hidden=None, max_mean=0.5, max_p80=1.0, rel_mean=0.022, rel_p80=0.035):
    """Circle cut by a straight line: a half circle (h = 0), a bigger or smaller cap. params: cx, cy, r, theta
    (direction from the centre to the cut), h (signed distance of the cut from the centre, -r..r)."""
    from .circle import kasa
    K = np.asarray(mask, bool)
    P, vis, Pq, Pall = _visible(K, hidden)
    if len(Pq) < 20:
        return None
    # start: RANSAC circle through the curved part; the points off it are the straight cut - a line through them
    # gives the cut's direction and distance from the centre
    from scipy.spatial import cKDTree
    from .circle import ransac_circle
    rc = ransac_circle(Pq, tol=max(1.0, 0.02 * eq_radius(K)), iters=60)
    starts = []
    if rc is not None:
        (cx, cy, r), inl = rc
        off = Pq[~inl]
        if len(off) >= 5:
            mu = off.mean(0)
            _, _, vt = np.linalg.svd(off - mu, full_matrices=False)
            nrm = np.array([-vt[0][1], vt[0][0]])
            h = float((mu - [cx, cy]) @ nrm)
            if h < 0:
                nrm, h = -nrm, -h
            starts.append([cx, cy, r, float(np.arctan2(nrm[1], nrm[0])), float(np.clip(h, -0.9 * r, 0.9 * r))])
    f = kasa(Pq)
    if f is not None:
        c_m, th_m, _ = _moments(K)
        v = c_m - [f[0], f[1]]
        th0 = float(np.arctan2(v[1], v[0])) if np.hypot(*v) > 1e-6 else th_m
        starts += [[f[0], f[1], f[2], th0 + np.pi, 0.0], [f[0], f[1], f[2], th0, 0.0]]
    best = None
    for x0 in starts:
        x = _fit(_segment_model, x0, Pq, lo=[-np.inf, -np.inf, 1, -np.inf, -np.inf], hi=[np.inf] * 5)
        if x is None or _segment_model(x, 8) is None:
            continue
        e = float(cKDTree(_segment_model(x, 60)).query(Pq)[0].mean())
        if best is None or e < best[0]:
            best = (e, x)
        if e < 0.6:
            break
    if best is None:
        return None
    cx, cy, r, th, h = best[1]
    if abs(h) > 0.92 * r:  # a tiny cut is just a circle, a thin sliver is not this shape
        return None
    a_ = np.arccos(h / r)
    on_cut = _d_seg(_local(Pall, cx, cy, th), np.array([h, -r * np.sin(a_)]), np.array([h, r * np.sin(a_)])) < 1.0
    if on_cut.mean() < 0.12:  # the straight cut must really be there, not a dent in a wobbly dot
        return None
    p = dict(cx=float(cx), cy=float(cy), r=float(r), theta=float(th), h=float(h))
    return _accept(_segment_model, best[1], K, Pall, r, max_mean, max_p80, rel_mean, rel_p80, "segment", p,
                   segment_path(p), vis, hidden)


def segment_path(p):
    cx, cy, r, th, h = p["cx"], p["cy"], p["r"], p["theta"], p["h"]
    a = np.arccos(np.clip(h / r, -1, 1))
    R = _rot(th)
    p0 = R @ [r * np.cos(a), r * np.sin(a)] + [cx, cy]
    p1 = R @ [r * np.cos(a), -r * np.sin(a)] + [cx, cy]
    large = 1 if a < np.pi / 2 else 0  # the arc spans 2pi - 2a
    return f"M{fmt(p0)}A{r:.2f} {r:.2f} 0 {large} 1 {fmt(p1)}Z"


# ------------------------------------------------------------------ teardrop
def _tear_model(x, n):
    cx, cy, r, th, L = x
    if r <= 0 or L <= r * 1.02:
        return None
    a = np.arccos(r / L)                    # tangent points: angle from the tip direction
    t = np.linspace(a, TWO_PI - a, n)
    arc = np.c_[r * np.cos(t), r * np.sin(t)]
    tip = np.array([L, 0.0])
    t1 = np.array([r * np.cos(a), -r * np.sin(a)])
    t0 = np.array([r * np.cos(a), r * np.sin(a)])
    s = np.linspace(0, 1, n // 2, endpoint=False)[:, None]
    line1 = t1 + s * (tip - t1)
    line0 = tip + s * (t0 - tip)
    return np.vstack([arc, line1, line0]) @ _rot(th).T + [cx, cy]


def fit_teardrop(mask, hidden=None, max_mean=0.5, max_p80=1.0, rel_mean=0.022, rel_p80=0.035):
    """Teardrop: a circle with straight sides running out to one tip. params: cx, cy, r (round end), theta
    (direction to the tip), L (centre-to-tip distance)."""
    K = np.asarray(mask, bool)
    P, vis, Pq, Pall = _visible(K, hidden)
    if len(Pq) < 20:
        return None
    c_m, _, _ = _moments(K)
    d = np.hypot(Pq[:, 0] - c_m[0], Pq[:, 1] - c_m[1])
    tip = Pq[int(np.argmax(d))]
    th0 = float(np.arctan2(tip[1] - c_m[1], tip[0] - c_m[0]))
    r0 = float(np.percentile(d, 30))
    cen0 = c_m - 0.25 * r0 * np.array([np.cos(th0), np.sin(th0)])
    L0 = max(float(np.hypot(*(tip - cen0))), 1.3 * r0)
    x = _fit(_tear_model, [cen0[0], cen0[1], r0, th0, L0], Pq, lo=[-np.inf, -np.inf, 1, -np.inf, 1],
             hi=[np.inf] * 5)
    if x is None or _tear_model(x, 8) is None:
        return None
    cx, cy, r, th, L = x
    if L > 4 * r:
        return None
    p = dict(cx=float(cx), cy=float(cy), r=float(r), theta=float(th), L=float(L))
    return _accept(_tear_model, x, K, Pall, r, max_mean, max_p80, rel_mean, rel_p80, "teardrop", p,
                   teardrop_path(p), vis, hidden)


def teardrop_path(p):
    cx, cy, r, th, L = p["cx"], p["cy"], p["r"], p["theta"], p["L"]
    R = _rot(th)
    a = np.arccos(r / L)
    tip_p = R @ [L, 0] + [cx, cy]
    t0 = R @ [r * np.cos(a), r * np.sin(a)] + [cx, cy]
    t1 = R @ [r * np.cos(a), -r * np.sin(a)] + [cx, cy]
    return f"M{fmt(tip_p)}L{fmt(t0)}A{r:.2f} {r:.2f} 0 1 1 {fmt(t1)}Z"


# ------------------------------------------------------------------ heart
def _heart_unit(n):
    t = np.linspace(0, TWO_PI, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    P = np.c_[x, y] / 16.0
    return P - [0, P[:, 1].mean()]


def _heart_model(x, n):
    cx, cy, sx, sy, th = x
    if sx <= 0 or sy <= 0:
        return None
    return (_heart_unit(n * 4) * [sx, sy]) @ _rot(th).T + [cx, cy]


def fit_heart(mask, hidden=None, max_mean=0.6, max_p80=1.2, rel_mean=0.025, rel_p80=0.04):
    """Heart (the classic curve, stretched by sx / sy). params: cx, cy, sx, sy, theta (0 = tip pointing down)."""
    K = np.asarray(mask, bool)
    P, vis, Pq, Pall = _visible(K, hidden)
    if len(Pq) < 30:
        return None
    from scipy.spatial import cKDTree
    c_m, _, sd = _moments(K)
    s0 = float(np.sqrt(K.sum() / 2.9))  # area of the unit heart is about 2.9
    if s0 < 6:  # a speck has no readable notch and tip
        return None
    # the tip is the outline point farthest from the centroid; the unit heart's tip points down (+y)
    d = np.hypot(Pq[:, 0] - c_m[0], Pq[:, 1] - c_m[1])
    tip = Pq[int(np.argmax(d))]
    th_tip = float(np.arctan2(tip[1] - c_m[1], tip[0] - c_m[0])) - np.pi / 2
    best = None
    # four orientations scored without fitting; only the two best starts are optimised
    starts = []
    for th0 in (th_tip, th_tip + np.pi, th_tip + np.pi / 2, th_tip - np.pi / 2):
        x0 = [c_m[0], c_m[1], s0, s0, th0]
        starts.append((float(cKDTree(_heart_model(x0, 20)).query(Pq)[0].mean()), x0))
    starts.sort(key=lambda t: t[0])
    if starts[0][0] > 0.25 * s0:  # nothing near a heart
        return None
    for _, x0 in starts[:2]:
        x = _fit(_heart_model, x0, Pq, lo=[-np.inf, -np.inf, 1, 1, -np.inf], hi=[np.inf] * 5, nfev=60)
        if x is None or _heart_model(x, 8) is None:
            continue
        e = float(cKDTree(_heart_model(x, 30)).query(Pq)[0].mean())
        if best is None or e < best[0]:
            best = (e, x)
        if e < 0.8:
            break
    if best is None:
        return None
    x = best[1]
    cx, cy, sx, sy, th = x
    if not 0.6 < sx / sy < 1.7:
        return None
    p = dict(cx=float(cx), cy=float(cy), sx=float(sx), sy=float(sy), theta=float(th))
    return _accept(_heart_model, x, K, Pall, np.sqrt(sx * sy), max_mean, max_p80, rel_mean, rel_p80, "heart", p,
                   heart_path(p), vis, hidden)


def heart_path(p):
    """The two halves between the notch (t = 0) and the tip (t = pi) as line / arc / cubic runs."""
    from .geom import fit_open
    from .svg import segments_to_d
    Q = _heart_model([p["cx"], p["cy"], p["sx"], p["sy"], p["theta"]], 100)
    n = len(Q)
    half1, half2 = Q[: n // 2 + 1], np.vstack([Q[n // 2:], Q[:1]])
    return segments_to_d(fit_open(half1, tol=0.25) + fit_open(half2, tol=0.25), closed=True)


# ------------------------------------------------------------------ crescent
def _crescent_model(x, n):
    cx, cy, r1, th, dd, r2 = x
    if r1 <= 0 or r2 <= 0:
        return None
    c2 = np.array([dd, 0.0])
    # intersection of circle (0, r1) and circle (c2, r2)
    if not abs(r1 - r2) < dd < r1 + r2:
        return None
    xi = (dd * dd + r1 * r1 - r2 * r2) / (2 * dd)
    yi2 = r1 * r1 - xi * xi
    if yi2 <= 0:
        return None
    yi = np.sqrt(yi2)
    a1 = np.arctan2(yi, xi)                       # outer arc: the part of circle 1 outside circle 2
    t1 = np.linspace(a1, TWO_PI - a1, n)
    outer = np.c_[r1 * np.cos(t1), r1 * np.sin(t1)]
    a2 = np.arctan2(yi, xi - dd)                  # inner arc: the part of circle 2 inside circle 1, back again
    t2 = np.linspace(TWO_PI - a2, a2, n) if a2 > 0 else np.linspace(-a2, a2, n)
    inner = c2 + np.c_[r2 * np.cos(t2), r2 * np.sin(t2)]
    return np.vstack([outer, inner]) @ _rot(th).T + [cx, cy]


def fit_crescent(mask, hidden=None, max_mean=0.5, max_p80=1.0, rel_mean=0.022, rel_p80=0.035):
    """Crescent (moon): a disc with a second disc cut out of it. params: cx, cy, r1 (outer disc), theta (direction
    to the cut-out disc), d (distance of its centre), r2 (its radius)."""
    import cv2
    from .circle import kasa
    K = np.asarray(mask, bool)
    P, vis, Pq, Pall = _visible(K, hidden)
    if len(Pq) < 30:
        return None
    hull = cv2.convexHull(np.argwhere(K)[:, ::-1].astype(np.float32))
    hm = cv2.moments(hull)
    if not hm["m00"]:
        return None
    hc = np.array([hm["m10"] / hm["m00"], hm["m01"] / hm["m00"]]) + 0.5
    c_m, _, _ = _moments(K)
    v = hc - c_m
    th0 = float(np.arctan2(v[1], v[0]))
    f = kasa(hull.reshape(-1, 2) + 0.5)
    if f is None:
        return None
    r1 = f[2]
    best = None
    from scipy.spatial import cKDTree
    for k2 in (0.8, 1.0):
        for kd in (0.35, 0.6):
            x0 = [f[0], f[1], r1, th0, kd * r1, k2 * r1]
            x = _fit(_crescent_model, x0, Pq, lo=[-np.inf, -np.inf, 1, -np.inf, 0.5, 1], hi=[np.inf] * 6, nfev=60)
            if x is None or _crescent_model(x, 8) is None:
                continue
            e = float(cKDTree(_crescent_model(x, 40)).query(Pq)[0].mean())
            if best is None or e < best[0]:
                best = (e, x)
    if best is None:
        return None
    x = best[1]
    cx, cy, r1, th, dd, r2 = x
    # a moon has a body: widest part at least 15 % of the outer radius, outer disc not much bigger than the shape
    if r1 - (r2 - dd) < max(3.0, 0.12 * r1) or r1 > 4.0 * eq_radius(K):
        return None
    p = dict(cx=float(cx), cy=float(cy), r1=float(r1), theta=float(th), d=float(dd), r2=float(r2))
    return _accept(_crescent_model, x, K, Pall, r1, max_mean, max_p80, rel_mean, rel_p80, "crescent", p,
                   crescent_path(p), vis, hidden)


def crescent_path(p):
    """Outer arc of disc 1 from cusp A to cusp B (sweeping away from disc 2), inner arc of disc 2 back to A."""
    cx, cy, r1, th, dd, r2 = p["cx"], p["cy"], p["r1"], p["theta"], p["d"], p["r2"]
    R = _rot(th)
    xi = (dd * dd + r1 * r1 - r2 * r2) / (2 * dd)
    yi = np.sqrt(max(r1 * r1 - xi * xi, 0.0))
    pa = R @ [xi, yi] + [cx, cy]
    pb = R @ [xi, -yi] + [cx, cy]
    large1 = 1 if xi > 0 else 0   # outer arc spans 2pi - 2 atan2(yi, xi)
    large2 = 1 if xi > dd else 0  # inner arc spans 2pi - 2 atan2(yi, xi - d), drawn the other way
    return f"M{fmt(pa)}A{r1:.2f} {r1:.2f} 0 {large1} 1 {fmt(pb)}A{r2:.2f} {r2:.2f} 0 {large2} 0 {fmt(pa)}Z"


def ring_path(p):
    from .circle import circle_path
    return circle_path(p["cx"], p["cy"], p["r_out"]) + circle_path(p["cx"], p["cy"], p["r_in"])


ANALYTIC.update({_segment_model: _dist_segment, _tear_model: _dist_tear, _crescent_model: _dist_crescent})

PATHS = dict(ring=ring_path, segment=segment_path, teardrop=teardrop_path, heart=heart_path, crescent=crescent_path)
