"""Cheap shape features (well under a millisecond per shape) used to decide which detectors are worth running:
the expensive fits (stars above all) only run on shapes that could be that kind."""
import numpy as np


def shape_features(mask):
    """area, circularity (4 pi A / P^2, 1 = disc), solidity (area / convex hull area), vertices (Douglas-Peucker
    corners at 2 % of the perimeter) and tips (outward peaks of the distance from the centroid)."""
    import cv2
    m8 = np.asarray(mask, np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 4:
        return None
    peri = float(cv2.arcLength(c, True))
    hull = float(cv2.contourArea(cv2.convexHull(c)))
    verts = len(cv2.approxPolyDP(c, 0.02 * peri, True))
    P = c.reshape(-1, 2).astype(float)
    mo = cv2.moments(c)
    cen = np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]]) if mo["m00"] else P.mean(0)
    # radial profile by angle (max distance per bin), circular smoothing, count clear peaks
    ang = (np.arctan2(P[:, 1] - cen[1], P[:, 0] - cen[0]) % (2 * np.pi)) / (2 * np.pi) * 90
    rad = np.hypot(P[:, 0] - cen[0], P[:, 1] - cen[1])
    prof = np.zeros(90)
    np.maximum.at(prof, np.minimum(ang.astype(int), 89), rad)
    if (prof == 0).any():
        idx = np.arange(90)
        ok = prof > 0
        prof[~ok] = np.interp(idx[~ok], idx[ok], prof[ok], period=90) if ok.sum() >= 2 else prof.max()
    prof = np.convolve(np.r_[prof[-2:], prof, prof[:2]], np.ones(5) / 5, "valid")
    lo, hi = prof.min(), prof.max()
    thr = lo + 0.5 * (hi - lo)
    tips = int(sum(1 for i in range(90) if prof[i] > thr and prof[i] >= prof[i - 1] and prof[i] > prof[(i + 1) % 90]))
    return dict(area=area, circularity=4 * np.pi * area / max(peri * peri, 1e-9), solidity=area / max(hull, 1e-9),
                vertices=verts, tips=tips, depth=float((hi - lo) / max(hi, 1e-9)))


def candidate_kinds(feat, hidden_share=0.0, kinds=("circle", "ellipse", "polygon", "star")):
    """Detectors worth running for a shape with these features. Shapes with outline hidden by objects in front
    keep more candidates (their visible outline says less)."""
    if feat is None:
        return tuple(kinds)
    occl = hidden_share > 0.02
    out = []
    for k in kinds:
        if k in ("circle", "ellipse"):
            ok = feat["solidity"] > (0.75 if occl else 0.9)
        elif k == "star":
            # spiky: clearly concave and 3..9 outward peaks (a hidden tip may be missing, a string adds one)
            ok = feat["solidity"] < (0.92 if occl else 0.88) and feat["depth"] > 0.25 and 3 <= feat["tips"] <= 10
        elif k == "polygon":
            ok = feat["circularity"] < 0.97 or occl
        else:
            ok = True
        if ok:
            out.append(k)
    return tuple(out)
