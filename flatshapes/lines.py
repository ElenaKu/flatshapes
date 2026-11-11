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


def merge_collinear(segments, max_angle=4.0, max_offset=2.0, max_gap=90.0, noisy_len=3.0):
    """Join pieces of one straight line. segments: list of (p0, p1) arrays (2,) each.
    Two pieces join when their directions differ by <= max_angle degrees (wider for short pieces, whose direction
    is noisy: atan(noisy_len / length)), both lie within max_offset px of the other's line, and the gap between
    them along the line is <= max_gap px. Returns the merged list of (p0, p1)."""
    segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in segments]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                la, lb = np.linalg.norm(a[1] - a[0]), np.linalg.norm(b[1] - b[0])
                da, db = _seg_dir(a), _seg_dir(b)
                lim = max(max_angle, np.degrees(np.arctan2(noisy_len, max(1e-6, min(la, lb)))))
                cos = abs(float(da @ db))
                if np.degrees(np.arccos(min(1.0, cos))) > lim:
                    continue
                u = da if la >= lb else db
                if float(da @ db) < 0 and la < lb:
                    u = db
                nrm = np.array([-u[1], u[0]])
                base = a[0] if la >= lb else b[0]
                off = [abs(float((p - base) @ nrm)) for p in (*a, *b)]
                if max(off) > max_offset:
                    continue
                ta = sorted(float((p - base) @ u) for p in a)
                tb = sorted(float((p - base) @ u) for p in b)
                gap = max(ta[0], tb[0]) - min(ta[1], tb[1])
                if gap > max_gap:
                    continue
                t = ta + tb
                lo, hi = min(t), max(t)
                segs[i] = (base + lo * u, base + hi * u)
                segs.pop(j)
                changed = True
                break
            if changed:
                break
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
        width = 2.0 * float(np.median(dv[on]))
        if width > max_width:
            continue
        # a real stroke has empty space on both sides; a straight run along the edge of a big shape (a letter)
        # has the shape on one side
        u = (p1 - p0) / L
        nrm = np.array([-u[1], u[0]])
        free = []
        for sgn in (1, -1):
            qs = np.round(p0 + t[:, None] * (p1 - p0) + sgn * (width / 2 + side_gap) * nrm).astype(int)
            inside = (qs[:, 0] >= 0) & (qs[:, 1] >= 0) & (qs[:, 0] < w) & (qs[:, 1] < h)
            qs = qs[inside]
            free.append(float((K[qs[:, 1], qs[:, 0]] == 0).mean()) if len(qs) else 1.0)
        if min(free) < min_free:
            continue
        out.append(ShapeFit("line", dict(p0=p0, p1=p1, width=width), f"M{fmt(p0)}L{fmt(p1)}",
                            dict(length=L, coverage=float(on.mean()))))
    return out


def same_sides(line, labels, offset=None, min_share=0.7):
    """True when a line has the same label on both sides (a stroke over one background), judged on samples
    `offset` px beyond its edges (default width / 2 + 2). An anti-aliasing rim along the edge of a shape has the
    shape on one side and the background on the other, so it fails."""
    L = np.asarray(labels)
    h, w = L.shape
    p0, p1 = line.params["p0"], line.params["p1"]
    off = line.params["width"] / 2 + 2.0 if offset is None else offset
    n = max(4, int(np.linalg.norm(p1 - p0)))
    u = (p1 - p0) / max(1e-9, np.linalg.norm(p1 - p0))
    nrm = np.array([-u[1], u[0]])
    t = np.linspace(0.1, 0.9, n)[:, None]
    sides = []
    for sgn in (1, -1):
        q = np.clip(np.round(p0 + t * (p1 - p0) + sgn * off * nrm).astype(int), 0, [w - 1, h - 1])
        v = L[q[:, 1], q[:, 0]]
        vals, cnt = np.unique(v, return_counts=True)
        sides.append((vals[cnt.argmax()], cnt.max() / len(v)))
    return sides[0][0] == sides[1][0] and min(sides[0][1], sides[1][1]) >= min_share
