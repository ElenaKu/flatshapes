"""Star detection: a k-point star (k = 4..8 by default) with straight or concave sides.

Model: centre C, tips T_0..T_k-1 sorted by angle; side i from T_i to T_i+1 is a cubic Bezier with control points
T_i + b_i (C - T_i) and T_i+1 + b_i (C - T_i+1). b = 0 is a straight side, b -> 1 bends it towards the centre.

The model is fitted to the visible outline only: outline points under the `hidden` mask (objects in front, lines)
or on the image border are left out. A tip hidden that way is put back from the even angular spacing of the others,
with the radius of the opposite tip (a star is point-symmetric). A tip much longer than the opposite one (a line
glued to it) is cut to the opposite length.

Ported from flat2svg core v2 (core2/stars.py).
"""
import numpy as np

from .result import ShapeFit, outline_metrics, eq_radius, limit

def _bez(p0, p1, p2, p3, t):
    t = t[:, None]
    s = 1 - t
    return s ** 3 * p0 + 3 * s * s * t * p1 + 3 * s * t * t * p2 + t ** 3 * p3


def _model(C, T, b, ns=40):
    """Outline points of the star model, ns per side (vectorised over the sides)."""
    T = np.asarray(T, float)
    C = np.asarray(C, float)
    b = np.asarray(b, float)[:, None]
    A, Z = T, np.roll(T, -1, axis=0)
    P1, P2 = A + b * (C - A), Z + b * (C - Z)
    t = np.linspace(0, 1, ns, endpoint=False)[None, :, None]
    s = 1 - t
    pts = s ** 3 * A[:, None] + 3 * s * s * t * P1[:, None] + 3 * s * t * t * P2[:, None] + t ** 3 * Z[:, None]
    return pts.reshape(-1, 2)


def _path(C, T, b):
    k = len(T)
    f = lambda p: f"{p[0]:.2f} {p[1]:.2f}"
    d = "M" + f(T[0])
    for i in range(k):
        a, z = T[i], T[(i + 1) % k]
        d += "C" + f(a + b[i] * (C - a)) + " " + f(z + b[i] * (C - z)) + " " + f(z)
    return d + "Z"


def _fit_star(K, occ, off, size, min_ratio, max_mean, max_p80, best_share, k_range):
    """Fit the star model to mask K (local crop); occ: hidden outline (line strips). Returns (C, T, b, model mask)
    in image coordinates (mask in crop coordinates) or None."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    hh, ww = K.shape
    x0, y0 = off
    W, H = size
    dt = cv2.distanceTransform(K.astype(np.uint8), cv2.DIST_L2, 5)
    r_in = float(dt.max())
    if r_in < 1.5:
        return None
    cy, cx = np.unravel_index(dt.argmax(), dt.shape)
    C0 = np.array([cx + 0.5 + x0, cy + 0.5 + y0])
    # outline on the pixel edges: contour of the 4x nearest-upsampled mask
    up = np.repeat(np.repeat(K.astype(np.uint8), 4, 0), 4, 1)
    cs, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    P = (max(cs, key=len).reshape(-1, 2).astype(float) + 0.5) / 4.0
    P = P[::2]
    N = len(P)
    if N < 30:
        return None
    pi = np.clip(P.astype(int), 0, [ww - 1, hh - 1])
    # outline under a strip or a few px from one is hidden (line remnants glued next to the strips); the margin
    # grows with the star (up to 4 px), a small star on a line would otherwise have little outline left
    mr = int(np.clip(round(r_in / 5), 1, 4))
    occ_h = cv2.dilate(occ.astype(np.uint8), np.ones((2 * mr + 1, 2 * mr + 1), np.uint8)) > 0
    hidden = occ_h[pi[:, 1], pi[:, 0]].copy()
    P = P + [x0, y0]
    hidden |= (P[:, 0] < 1) | (P[:, 1] < 1) | (P[:, 0] > W - 1) | (P[:, 1] > H - 1)
    vis = ~hidden
    if vis.mean() < 0.45:
        return None
    d = P - C0
    rr = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    win = max(2, N // 60)
    rs = np.convolve(np.r_[rr[-win:], rr, rr[:win]], np.ones(2 * win + 1) / (2 * win + 1), "valid")
    span = max(3, N // 16)
    peaks = []
    for i in range(N):
        if not vis[i] or rr[i] < min_ratio * r_in:
            continue
        idx = np.arange(i - span, i + span + 1) % N
        if rs[i] >= rs[idx].max() - 1e-9:
            peaks.append(i)
    # one tip per lobe: merge peaks closer than 25 degrees
    peaks.sort(key=lambda i: -rr[i])
    tips = []
    for i in peaks:
        if all(abs((ang[i] - ang[j] + np.pi) % (2 * np.pi) - np.pi) > np.radians(25) for j in tips):
            tips.append(i)
    if len(tips) < 2:
        return None
    ta = sorted((float(ang[i]), P[i]) for i in tips)
    angs = np.array([a for a, _ in ta])
    T = [p for _, p in ta]
    gaps = np.diff(np.r_[angs, angs[0] + 2 * np.pi])
    med = float(np.median(gaps))
    if len(T) == 2:
        # two opposite tips seen, the other two under lines (strings through the top and bottom tips)
        if abs(abs((angs[1] - angs[0] + np.pi) % (2 * np.pi) - np.pi) - np.pi) > np.radians(25):
            return None
        med = np.pi / 2  # two opposite tips seen, the other two under a line: a 4-point star is tried
    rmed = float(np.median([np.linalg.norm(p - C0) for p in T]))
    # a tip hidden under lines: put back at even spacing
    ins = []
    for i, g in enumerate(gaps):
        m = int(round(g / med)) - 1
        if g > 1.6 * med and m >= 1:
            for j in range(1, m + 1):
                a = angs[i] + g * j / (m + 1)
                # radius of the opposite tip if there is one (a star is point-symmetric), else the median
                opp = [np.linalg.norm(p - C0) for aa, p in zip(angs, T)
                       if abs((aa - a - np.pi + np.pi) % (2 * np.pi) - np.pi) < np.radians(25)]
                ri = float(opp[0]) if opp else rmed
                ins.append((a, C0 + ri * np.array([np.cos(a), np.sin(a)])))
    hid = [False] * len(T)
    if ins:
        allt = sorted([(a % (2 * np.pi), p, False) for a, p in zip(angs, T)] + [(a % (2 * np.pi), p, True) for a, p in ins], key=lambda t: t[0])
        T = [p for _, p, _ in allt]
        hid = [hh_ for _, _, hh_ in allt]
        angs = np.array([a for a, _, _ in allt])
        gaps = np.diff(np.r_[angs, angs[0] + 2 * np.pi])
    k = len(T)
    if not k_range[0] <= k <= k_range[1] or gaps.max() > 2.0 * np.median(gaps):
        return None
    # a tip that runs on along a line (the line was not fully covered by its strip) is much longer than the
    # opposite tip: cut to the opposite length
    if k % 2 == 0:
        T = [np.array(p, float) for p in T]
        for i in range(k):
            j = (i + k // 2) % k
            # centre for this pair: midpoint of the most even opposite pair among the others (4 tips: the cross pair)
            others = [(abs(np.log(max(1e-6, np.linalg.norm(T[a] - C0)) / max(1e-6, np.linalg.norm(T[(a + k // 2) % k] - C0)))), a)
                      for a in range(k // 2) if a % (k // 2) != i % (k // 2)]
            Cp = C0
            if others:
                a = min(others)[1]
                Cp = (T[a] + T[(a + k // 2) % k]) / 2
            rad = {i: float(np.linalg.norm(T[i] - Cp)), j: float(np.linalg.norm(T[j] - Cp))}
            if rad[i] > 1.4 * rad[j]:
                if True:  # opposite tips of a star are about equal; a much longer one has a line glued on
                    T[i] = Cp + (T[i] - Cp) * (rad[j] / rad[i])
                    hid[i] = True
    T0 = np.array(T)
    Pv_all = P[vis]
    # the fit runs on at most 400 outline points (the acceptance check below uses all of them)
    Pv = Pv_all[:: max(1, len(Pv_all) // 400)]
    wt = np.where(hid, 5.0, 0.05)  # a tip put back under a line has nothing to hold it: keep it where symmetry says

    def unpack(x):
        return x[:2], x[2:2 + 2 * k].reshape(k, 2), x[2 + 2 * k:]

    def resid(x):
        C, T_, b = unpack(x)
        dd, _ = cKDTree(_model(C, T_, b)).query(Pv)
        prior_t = (wt[:, None] * (T_ - T0)).ravel()
        prior_b = 0.5 * (b - b.mean())
        return np.r_[dd, prior_t, prior_b]

    # quick reject before the optimiser: with the start tips and the best common bend the outline must already be
    # roughly right (letters and blobs with a few spikes stop here)
    pre = []
    for bb in (0.15, 0.35, 0.55, 0.75, 0.9):
        dd0, _ = cKDTree(_model(C0, T0, np.full(k, bb), 20)).query(Pv)
        pre.append((float(np.sort(dd0)[: max(1, int(len(dd0) * best_share))].mean()), bb))
    pre_err, b0 = min(pre)
    if pre_err > 6 * max_mean + 1.0:
        return None
    x0v = np.r_[C0, T0.ravel(), np.full(k, 0.5)]  # (starting from b0 lands in worse local minima)
    lo = np.r_[np.full(2 + 2 * k, -np.inf), np.zeros(k)]
    hi = np.r_[np.full(2 + 2 * k, np.inf), np.full(k, 0.98)]
    try:
        sol = least_squares(resid, x0v, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=120)
    except Exception:
        return None
    C, T_, b = unpack(sol.x)
    dd, _ = cKDTree(_model(C, T_, b, 80)).query(Pv_all)
    # a few outline points may belong to a line stub glued to the star: judged on the best 90%
    ds = np.sort(dd)[: max(1, int(len(dd) * best_share))]
    if ds.mean() > max_mean or np.percentile(dd, 80) > max_p80:
        return None
    poly = _model(C, T_, b, 30) - [x0, y0]
    M = np.zeros((hh, ww), np.uint8)
    cv2.fillPoly(M, [np.round(poly * 4).astype(np.int32)], 1, shift=2)
    return C, T_, b, M


def fit_star(mask, hidden=None, min_ratio=1.5, max_mean=0.7, max_p80=1.5, best_share=0.85, k_range=(4, 8),
             max_area_err=0.06, rel_mean=0.022, rel_p80=0.035):
    """Fit a star to one binary shape mask (H, W). `hidden`: optional bool mask of outline that must be ignored.
    Returns ShapeFit(kind="star") or None.

    min_ratio:   a tip lies at least min_ratio * (inscribed radius) from the centre
    max_mean:    mean outline distance (px) over the best `best_share` of the visible points
    max_p80:     80th percentile of the outline distance (px)
    rel_mean, rel_p80: the two distance limits grow with the size (equivalent radius of the mask)
    max_area_err: share of pixels that differ by more than 1 px from the model edge (outside hidden areas)
    """
    import cv2
    K = np.asarray(mask, bool)
    h, w = K.shape
    occ = np.zeros((h, w), bool) if hidden is None else np.asarray(hidden, bool)
    R = eq_radius(K)
    res = _fit_star(K, occ, (0, 0), (w, h), min_ratio, limit(max_mean, rel_mean, R), limit(max_p80, rel_p80, R),
                    best_share, k_range)
    if res is None:
        return None
    C, T, b, M = res
    ker = np.ones((3, 3), np.uint8)
    o_near = cv2.dilate(occ.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    Md, Me = cv2.dilate(M, ker) > 0, cv2.erode(M, ker) > 0
    bad = ((K & ~Md) | (Me & ~K)) & ~o_near
    area_err = float(bad.sum()) / max(1, int(M.sum()))
    if area_err > max_area_err:
        return None
    met = outline_metrics(K, occ, _model(C, T, b, 80))
    met["area_err"] = area_err
    return ShapeFit("star", dict(centre=C, tips=T, bend=b, k=len(T)), _path(C, T, b), met, M.astype(bool))
