"""Ellipses: occlusion-aware fit on a mask, written as four exact cubic quarters.

Ported from flat2svg core v2 (core2/primitives.py); the visible-outline part is new: only visible outline points
are fitted, so an ellipse partly hidden by an object in front is still found.
"""
import numpy as np

from .result import ShapeFit, outline_points, fmt

K_ARC = 0.5522847498


def ellipse_quarters(c, a, b, theta, start=0.0):
    """Four cubic quarters (('C', p0, c1, c2, p1) tuples) of the ellipse, starting at parameter `start`."""
    c = np.asarray(c, float)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pt = lambda t: c + R @ np.array([a * np.cos(t), b * np.sin(t)])
    tan = lambda t: R @ np.array([-a * np.sin(t), b * np.cos(t)])
    segs = []
    for i in range(4):
        ta, tb = start + i * np.pi / 2, start + (i + 1) * np.pi / 2
        p0, p3 = pt(ta), pt(tb)
        segs.append(("C", p0, p0 + K_ARC * tan(ta), p3 - K_ARC * tan(tb), p3))
    segs[-1] = ("C", segs[-1][1], segs[-1][2], segs[-1][3], segs[0][1].copy())
    return segs


def ellipse_path(c, a, b, theta):
    segs = ellipse_quarters(c, a, b, theta)
    return "M" + fmt(segs[0][1]) + "".join(f"C{fmt(s[2])} {fmt(s[3])} {fmt(s[4])}" for s in segs) + "Z"


def _dist_to_ellipse(P, c, a, b, theta, n=720):
    from scipy.spatial import cKDTree
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Q = np.asarray(c) + (R @ np.vstack([a * np.cos(t), b * np.sin(t)])).T
    return cKDTree(Q).query(P)[0]


def fit_ellipse(mask, hidden=None, min_axis=2.0, max_mean=0.5, max_p80=1.0, min_visible=0.4, max_gap_deg=200.0,
                max_aspect=6.0):
    """Ellipse fitted to the visible outline of one shape mask. Returns ShapeFit(kind="ellipse") or None.
    params: cx, cy, a, b (semi-axes), theta (radians). A very long thin shape (a line, a stroke; axis ratio above
    max_aspect) is not an ellipse."""
    import cv2
    K = np.asarray(mask, bool)
    h, w = K.shape
    P, vis = outline_points(K, hidden)
    if len(P) < 12 or vis.mean() < min_visible:
        return None
    Pv = P[vis]
    if len(Pv) < 6:
        return None
    (cx, cy), (W, H), ang = cv2.fitEllipse(Pv.astype(np.float32).reshape(-1, 1, 2))
    a, b, th = W / 2, H / 2, np.radians(ang)
    if min(a, b) < min_axis or not np.isfinite([cx, cy, a, b]).all() or max(a, b) > max_aspect * min(a, b):
        return None
    a_ = np.sort(np.arctan2(Pv[:, 1] - cy, Pv[:, 0] - cx))
    if np.degrees(np.diff(np.r_[a_, a_[0] + 2 * np.pi]).max()) > max_gap_deg:
        return None
    dd = _dist_to_ellipse(Pv, (cx, cy), a, b, th)
    if dd.mean() > max_mean or np.percentile(dd, 80) > max_p80:
        return None
    M = np.zeros((h, w), np.uint8)
    cv2.ellipse(M, ((cx, cy), (W, H), ang), 1, -1)
    met = dict(mean=float(dd.mean()), p80=float(np.percentile(dd, 80)), max=float(dd.max()), visible=float(vis.mean()))
    return ShapeFit("ellipse", dict(cx=float(cx), cy=float(cy), a=float(a), b=float(b), theta=float(th)),
                    ellipse_path((cx, cy), a, b, th), met, M > 0)
