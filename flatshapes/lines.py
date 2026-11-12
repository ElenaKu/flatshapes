"""Straight lines: thin straight strokes in a mask (strings, ribbons, rays) found with a probabilistic Hough
transform on the skeleton of the thin part, and pieces of one line joined across gaps (a line passing behind a
star shows on both sides of it).

Ported from flat2svg core v2 (core2/hybrid.split_lines); the merging is rewritten as a standalone function.
"""
import numpy as np

from .result import ShapeFit, fmt


def _seg_dir(s):
    d = s[1] - s[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def _try_merge(a, b, max_angle, max_offset, max_gap, noisy_len):
    """Merged segment of pieces a and b when they lie on one line, else None."""
    da, db = a[1] - a[0], b[1] - b[0]
    la, lb = float(np.hypot(*da)), float(np.hypot(*db))
    if la < 1e-9 or lb < 1e-9:
        return None
    da, db = da / la, db / lb
    lim = max(max_angle, np.degrees(np.arctan2(noisy_len, min(la, lb))))
    if np.degrees(np.arccos(min(1.0, abs(float(da @ db))))) > lim:
        return None
    u, base = (da, a[0]) if la >= lb else (db, b[0])
    nrm = np.array([-u[1], u[0]])
    if max(abs(float((p - base) @ nrm)) for p in (a[0], a[1], b[0], b[1])) > max_offset:
        return None
    ta = sorted(float((p - base) @ u) for p in a)
    tb = sorted(float((p - base) @ u) for p in b)
    if max(ta[0], tb[0]) - min(ta[1], tb[1]) > max_gap:
        return None
    t = ta + tb
    return (base + min(t) * u, base + max(t) * u)


def merge_collinear(segments, max_angle=4.0, max_offset=2.0, max_gap=90.0, noisy_len=3.0):
    """Join pieces of one straight line. segments: list of (p0, p1) arrays (2,) each.
    Two pieces join when their directions differ by <= max_angle degrees (wider for short pieces, whose direction
    is noisy: atan(noisy_len / length)), both lie within max_offset px of the other's line, and the gap between
    them along the line is <= max_gap px. Returns the merged list of (p0, p1)."""
    segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in segments]
    changed = True
    while changed:  # passes over all pairs; a merged piece keeps absorbing in the same pass
        changed = False
        alive = [True] * len(segs)
        for i in range(len(segs)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(segs)):
                if not alive[j]:
                    continue
                m = _try_merge(segs[i], segs[j], max_angle, max_offset, max_gap, noisy_len)
                if m is not None:
                    segs[i] = m
                    alive[j] = False
                    changed = True
        segs = [sg for sg, ok in zip(segs, alive) if ok]
    return segs


def detect_lines(mask, body_r=4.0, min_len=20.0, max_width=9.0, hough_threshold=15, max_line_gap=3, merge=True,
                 side_gap=2.0, min_free=0.7, **merge_kw):
    """Thin straight strokes in a binary mask. body = what survives an opening with a disc of radius body_r (thick
    parts, e.g. a star the strings run into); lines are searched on the skeleton of the rest. A line is kept when it
    is at least min_len long, at most max_width wide (twice the median distance to the edge along it) and has
    empty space on both sides (at least min_free of the samples side_gap px beyond its edges).
    Returns a list of ShapeFit(kind="line") with params p0, p1, width."""
    import cv2
    from skimage.morphology import skeletonize
    K = np.asarray(mask, np.uint8)
    k = int(np.ceil(body_r))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    body = cv2.morphologyEx(K, cv2.MORPH_OPEN, disk)
    thin = (K > 0) & (body == 0)
    if not thin.any():
        return []
    skel = skeletonize(thin).astype(np.uint8) * 255
    hl = cv2.HoughLinesP(skel, 1, np.pi / 360, hough_threshold, minLineLength=int(min_len * 0.5),
                         maxLineGap=max_line_gap)
    if hl is None:
        return []
    segs = [(np.array(l[:2], float), np.array(l[2:], float)) for l in hl.reshape(-1, 4)]
    if merge:
        segs = merge_collinear(segs, **merge_kw)
    dt = cv2.distanceTransform(K, cv2.DIST_L2, 5)
    h, w = K.shape
    # where a stroke runs through a thick body (a string through a star) neither its width nor the empty space
    # beside it can be judged: those samples are left out
    near_body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 5, 2 * k + 5))) > 0
    out = []
    for p0, p1 in segs:
        L = float(np.linalg.norm(p1 - p0))
        if L < min_len:
            continue
        t = np.linspace(0, 1, max(4, int(L)))
        q = np.clip(np.round(p0 + t[:, None] * (p1 - p0)).astype(int), 0, [w - 1, h - 1])
        dv = dt[q[:, 1], q[:, 0]]
        on = dv > 0
        if on.mean() < 0.5:
            continue
        free_part = ~near_body[q[:, 1], q[:, 0]]
        if free_part.sum() < max(4, 0.25 * len(q)):
            continue
        width = 2.0 * float(np.median(dv[on & free_part])) if (on & free_part).any() else 2.0 * float(np.median(dv[on]))
        if width > max_width:
            continue
        # a real stroke has empty space on both sides; a straight run along the edge of a big shape (a letter)
        # has the shape on one side
        u = (p1 - p0) / L
        nrm = np.array([-u[1], u[0]])
        free = []
        tf = t[free_part]
        for sgn in (1, -1):
            qs = np.round(p0 + tf[:, None] * (p1 - p0) + sgn * (width / 2 + side_gap) * nrm).astype(int)
            inside = (qs[:, 0] >= 0) & (qs[:, 1] >= 0) & (qs[:, 0] < w) & (qs[:, 1] < h)
            qs = qs[inside]
            free.append(float((K[qs[:, 1], qs[:, 0]] == 0).mean()) if len(qs) else 1.0)
        if min(free) < min_free:
            continue
        out.append(ShapeFit("line", dict(p0=p0, p1=p1, width=width), f"M{fmt(p0)}L{fmt(p1)}",
                            dict(length=L, coverage=float(on.mean()))))
    return out


def same_sides(line, labels, offset=None, min_share=0.6):
    """True when the line has the same colour on both sides along most of its length (a stroke over its
    background, which may change along the way: from the background onto a card and back). An anti-aliasing rim
    along the edge of a shape has the shape on one side and something else on the other all along, so it fails.
    Samples are `offset` px beyond the stroke edges (default width / 2 + 2)."""
    L = np.asarray(labels)
    h, w = L.shape
    p0, p1 = np.asarray(line.params["p0"], float), np.asarray(line.params["p1"], float)
    off = line.params["width"] / 2 + 2.0 if offset is None else offset
    n = max(4, int(np.linalg.norm(p1 - p0)))
    u = (p1 - p0) / max(1e-9, np.linalg.norm(p1 - p0))
    nrm = np.array([-u[1], u[0]])
    t = np.linspace(0.05, 0.95, n)[:, None]
    q = [np.round(p0 + t * (p1 - p0) + sgn * off * nrm).astype(int) for sgn in (1, -1)]
    inside = np.all([(qq[:, 0] >= 0) & (qq[:, 1] >= 0) & (qq[:, 0] < w) & (qq[:, 1] < h) for qq in q], axis=0)
    if inside.sum() < 4:
        return False
    a = L[q[0][inside, 1], q[0][inside, 0]]
    b = L[q[1][inside, 1], q[1][inside, 0]]
    return float((a == b).mean()) >= min_share
