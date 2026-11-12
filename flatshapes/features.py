"""Cheap shape features (well under a millisecond per shape) used to decide which detectors are worth running:
the expensive fits (stars above all) only run on shapes that could be that kind."""
import numpy as np


def shape_features(mask):
    """area, circularity (4 pi A / P^2, 1 = disc), solidity (area / convex hull area), vertices (Douglas-Peucker
    corners at 2 % of the perimeter) and tips (outward peaks of the distance from the centroid)."""
    import cv2
    m8 = np.asarray(mask, np.uint8)
    cs, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    outer = [i for i in range(len(cs)) if hier[0][i][3] < 0]
    oi = max(outer, key=lambda i: cv2.contourArea(cs[i]))
    c = cs[oi]
    holes = sum(1 for i in range(len(cs)) if hier[0][i][3] == oi and cv2.contourArea(cs[i]) > 0.05 * cv2.contourArea(c))
    area = float(cv2.contourArea(c))
    if area < 4:
        return None
    # perimeter of a lightly simplified outline: the pixel staircase would make every disc look jagged
    peri = float(cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True))
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
    # share of the outline on one circle (RANSAC on ~120 points): 1 for discs, about half for a half circle or
    # a teardrop, low for polygons
    from .circle import ransac_circle
    Ps = P[:: max(1, len(P) // 120)]
    req = np.sqrt(area / np.pi)
    rc = ransac_circle(Ps, tol=max(1.0, 0.03 * req), iters=40, min_inliers=3) if len(Ps) >= 8 else None
    arc_share = float(rc[1].mean()) if rc is not None and 0.3 * req < rc[0][2] < 3 * req else 0.0
    return dict(area=area, circularity=4 * np.pi * area / max(peri * peri, 1e-9), solidity=area / max(hull, 1e-9),
                vertices=verts, tips=tips, depth=float((hi - lo) / max(hi, 1e-9)), holes=holes, arc_share=arc_share)


def candidate_kinds(feat, hidden_share=0.0, kinds=("circle", "ellipse", "polygon", "star", "ring", "segment",
                                                    "teardrop", "heart", "crescent")):
    """Detectors worth running for a shape with these features. Shapes with outline hidden by objects in front
    keep more candidates (their visible outline says less)."""
    if feat is None:
        return tuple(kinds)
    occl = hidden_share > 0.02
    if feat.get("holes"):  # a shape with a hole: of the simple shapes only a ring has one
        return tuple(k for k in kinds if k == "ring")
    sol, circ = feat["solidity"], feat["circularity"]
    out = []
    for k in kinds:
        if k in ("circle", "ellipse"):
            ok = feat["solidity"] > (0.75 if occl else 0.9)
        elif k == "star":
            # spiky: clearly concave and 3..9 outward peaks (a hidden tip may be missing, a string adds one)
            ok = feat["solidity"] < (0.92 if occl else 0.88) and feat["depth"] > 0.25 and 3 <= feat["tips"] <= 10
        elif k == "polygon":
            ok = feat["circularity"] < 0.97 or occl
        elif k == "ring":
            ok = False
        elif k == "segment":  # a cut circle: convex, part of the outline on one circle, part straight
            ok = sol > (0.85 if occl else 0.93) and 0.45 < circ < 0.95 and 0.3 < feat["arc_share"] < 0.93
        elif k == "teardrop":  # one round end (on a circle) and one tip
            ok = sol > (0.8 if occl else 0.88) and 0.5 < circ < 0.95 and 0.3 < feat["arc_share"] < 0.93 \
                and feat["tips"] <= 2
        elif k == "heart":  # a notch makes it a little concave; two lobes
            ok = (0.78 if occl else 0.82) < sol < 0.98 and circ > 0.5 and feat["arc_share"] < 0.88
        elif k == "crescent":  # two horns around a bite
            ok = sol < (0.85 if occl else 0.8) and circ > 0.15 and feat["tips"] <= 4
        else:
            ok = True
        if ok:
            out.append(k)
    return tuple(out)
