"""Common result type and helpers shared by all detectors."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShapeFit:
    """One detected shape.

    kind:    "circle", "ellipse", "star", "polygon", "line"
    params:  model parameters (see each detector)
    path:    SVG path data in image pixel coordinates
    metrics: fit quality: mean / p80 / max outline distance in px, visible share, area_err when known
    mask:    model mask in image coordinates (None for lines)
    """
    kind: str
    params: dict
    path: str
    metrics: dict = field(default_factory=dict)
    mask: np.ndarray = None

    @property
    def n_params(self):
        """Model complexity, used to prefer the simpler model when errors are close."""
        return {"circle": 3, "ellipse": 5, "line": 4}.get(self.kind, 2 + 3 * int(self.params.get("k", 3)))


def outline_points(mask, hidden=None, sub=4, every=2):
    """Outer outline of a binary mask on pixel edges (contour of the `sub`x nearest-upsampled mask), in image px.
    Returns (P (n, 2), visible (n,) bool). Points under `hidden` (dilated by 1 px) or on the image border are not
    visible."""
    import cv2
    K = np.asarray(mask, np.uint8)
    h, w = K.shape
    up = np.repeat(np.repeat(K, sub, 0), sub, 1)
    cs, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2)), np.zeros(0, bool)
    P = (max(cs, key=len).reshape(-1, 2).astype(float) + 0.5) / sub
    P = P[::every]
    vis = (P[:, 0] >= 1) & (P[:, 1] >= 1) & (P[:, 0] <= w - 1) & (P[:, 1] <= h - 1)
    if hidden is not None:
        occ = cv2.dilate(np.asarray(hidden, np.uint8), np.ones((3, 3), np.uint8)) > 0
        pi = np.clip(P.astype(int), 0, [w - 1, h - 1])
        vis &= ~occ[pi[:, 1], pi[:, 0]]
    return P, vis


def outline_metrics(mask, hidden, model_pts):
    """Distances of the visible outline points of `mask` to a densely sampled model outline."""
    from scipy.spatial import cKDTree
    P, vis = outline_points(mask, hidden)
    if not vis.any():
        return dict(mean=np.inf, p80=np.inf, max=np.inf, visible=0.0)
    d, _ = cKDTree(model_pts).query(P[vis])
    return dict(mean=float(d.mean()), p80=float(np.percentile(d, 80)), max=float(d.max()), visible=float(vis.mean()))


def eq_radius(mask):
    """Equivalent radius sqrt(area / pi) of a mask: the size used for size-relative tolerances."""
    return float(np.sqrt(np.count_nonzero(mask) / np.pi))


def limit(abs_px, rel, size, cap=1.5):
    """Tolerance in px: the absolute limit, or `rel` times the shape size when that is larger, but at most `cap`
    times the absolute limit. Edges of generated and hand-drawn art wobble by a share of the shape size, so a
    40 px dot gets more slack than a 6 px one; the cap keeps big shapes (letters) from passing as polygons."""
    return max(float(abs_px), min(float(rel) * float(size), cap * float(abs_px)))


def fmt(p):
    return f"{p[0]:.2f} {p[1]:.2f}"
