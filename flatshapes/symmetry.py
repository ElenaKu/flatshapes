"""Symmetry of a shape of any form: rotational order (snowflakes, flowers, suns) and mirror axes (bows, leaves,
fir trees, hearts), measured by overlap of the mask with its rotated / reflected copy. Outline under `hidden`
(objects in front) takes no part in the comparison.

symmetrize() turns a slightly uneven shape (wobbly AI edges, a lopsided bow) into an exactly symmetric one:
the average of all its symmetric copies.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Symmetry:
    centre: tuple                   # (x, y) in mask coordinates: the centroid
    order: int = 1                  # rotational order n (1 = none); a 5-point star has 5
    rot_score: float = 0.0          # overlap (IoU) of the mask with itself rotated by 360 / order
    axes: list = field(default_factory=list)   # mirror axes, degrees in [0, 180) (0 = horizontal, y down)
    mirror_scores: list = field(default_factory=list)
    scores_by_order: dict = field(default_factory=dict)

    round: bool = False             # every rotation fits: a circle or a disc-like blob

    @property
    def kind(self):
        if self.round:
            return "round"
        if self.order > 1 and self.axes:
            return f"D{self.order}"      # rotation + mirrors (a regular star)
        if self.order > 1:
            return f"C{self.order}"      # rotation only (a pinwheel)
        if self.axes:
            return "mirror"
        return "none"


def _iou(a, b, care):
    a, b = a & care, b & care
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def _rotate(mask, c, deg):
    import cv2
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), deg, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, mask.shape[::-1], flags=cv2.INTER_NEAREST) > 0


def _reflect(mask, c, axis_deg):
    """Mirror about the line through c at axis_deg (image coordinates, y down)."""
    import cv2
    t = np.radians(axis_deg)
    d = np.array([np.cos(t), np.sin(t)])
    R = 2 * np.outer(d, d) - np.eye(2)
    off = np.asarray(c, float) - R @ np.asarray(c, float)
    M = np.c_[R, off].astype(np.float64)
    return cv2.warpAffine(mask.astype(np.uint8), M, mask.shape[::-1], flags=cv2.INTER_NEAREST) > 0


def _pad(mask, hidden):
    """Pad so rotated copies stay inside; returns padded mask, hidden, care mask (not hidden in either)."""
    h, w = mask.shape
    p = int(np.ceil(np.hypot(h, w) / 2)) + 2
    M = np.pad(np.asarray(mask, bool), p)
    H = np.pad(np.asarray(hidden, bool), p) if hidden is not None else np.zeros_like(M)
    return M, H, p


def find_symmetry(mask, hidden=None, max_order=12, rot_iou=0.9, mirror_iou=0.9, axis_step=1.0):
    """Rotational order and mirror axes of one shape mask. Returns Symmetry.
    rot_iou / mirror_iou: overlap a copy needs to count as symmetric (0.9 tolerates wobbly edges of generated art).
    The order is the largest n whose rotation by 360/n and all its multiples overlap; a circle reports max_order."""
    import cv2
    M, H, p = _pad(mask, hidden)
    ys, xs = np.nonzero(M)
    if len(xs) < 10:
        return Symmetry(centre=(0.0, 0.0))
    c = np.array([xs.mean(), ys.mean()])
    sym = Symmetry(centre=(float(c[0] - p), float(c[1] - p)))
    scores = {}
    for n in range(2, max_order + 1):
        worst = 1.0
        for j in range(1, n // 2 + 1):  # rotations by j * 360/n (the rest mirror these)
            ang = 360.0 * j / n
            Rm, Rh = _rotate(M, c, ang), _rotate(H, c, ang)
            worst = min(worst, _iou(M, Rm, ~(H | Rh)))
            if worst < rot_iou:
                break
        scores[n] = worst
    sym.scores_by_order = scores
    ok = [n for n, s in scores.items() if s >= rot_iou]
    if ok:
        sym.order = max(ok)
        sym.rot_score = scores[sym.order]
    if sym.order == max_order:  # symmetric under every tested rotation: round, axes are meaningless
        sym.round = True
        return sym
    # mirror axes: coarse search over [0, 180), refine around peaks
    angs = np.arange(0.0, 180.0, 3.0)
    sc = np.array([_iou(M, _reflect(M, c, a), ~(H | _reflect(H, c, a))) for a in angs])
    cand = [angs[i] for i in range(len(angs)) if sc[i] >= mirror_iou - 0.06 and sc[i] >= sc[i - 1] and sc[i] >= sc[(i + 1) % len(angs)]]
    for a0 in cand:
        fine = np.arange(a0 - 3, a0 + 3 + 1e-9, axis_step)
        fs_ = [_iou(M, _reflect(M, c, a), ~(H | _reflect(H, c, a))) for a in fine]
        k = int(np.argmax(fs_))
        if fs_[k] >= mirror_iou:
            a = float(fine[k] % 180.0)
            if all(min(abs(a - b), 180 - abs(a - b)) > 2 * axis_step for b in sym.axes):
                sym.axes.append(round(a, 2))
                sym.mirror_scores.append(round(float(fs_[k]), 4))
    return sym


def symmetrize(mask, sym, hidden=None):
    """Exactly symmetric version of `mask` (same shape): majority vote of the mask and all its copies under the
    found rotations and reflections. Hidden pixels of a copy do not vote. Returns a bool mask."""
    M, H, p = _pad(mask, hidden)
    c = np.array(sym.centre) + p
    votes = M.astype(np.float32) * (~H)
    weight = (~H).astype(np.float32)
    copies = [(_rotate, 360.0 * j / sym.order) for j in range(1, sym.order)] if sym.order > 1 else []
    for a in sym.axes:
        copies.append((_reflect, a))
        for j in range(1, sym.order):
            copies.append((_reflect, (a + 180.0 * j / sym.order) % 180.0))
    for fn, a in copies:
        vis = ~fn(H, c, a)
        votes += fn(M, c, a) * vis
        weight += vis
    out = votes > 0.5 * np.maximum(weight, 1e-6)
    h, w = mask.shape
    return out[p:p + h, p:p + w]
