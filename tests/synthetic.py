"""Synthetic test shapes: anti-aliased-free masks drawn at 4x and downsampled, with optional occluders."""
import numpy as np
import cv2


def _draw(fn, size=200):
    big = np.zeros((size * 4, size * 4), np.uint8)
    fn(big)
    return cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA) > 127


def circle(cx=100, cy=100, r=40, size=200):
    return _draw(lambda m: cv2.circle(m, (cx * 4, cy * 4), r * 4, 255, -1), size)


def ellipse(cx=100, cy=100, a=60, b=30, angle=25, size=200):
    return _draw(lambda m: cv2.ellipse(m, (cx * 4, cy * 4), (a * 4, b * 4), angle, 0, 360, 255, -1), size)


def star(k=5, cx=100, cy=100, r_out=70, r_in=30, size=200, rot=-90):
    a = np.radians(rot) + np.arange(2 * k) * np.pi / k
    rr = np.where(np.arange(2 * k) % 2 == 0, r_out, r_in)
    pts = np.c_[cx + rr * np.cos(a), cy + rr * np.sin(a)] * 4
    return _draw(lambda m: cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255), size)


def polygon(pts, size=200):
    P = np.asarray(pts, float) * 4
    return _draw(lambda m: cv2.fillPoly(m, [np.round(P).astype(np.int32)], 255), size)


def bar(p0, p1, width, size=200):
    return _draw(lambda m: cv2.line(m, tuple(int(v * 4) for v in p0), tuple(int(v * 4) for v in p1), 255,
                                    int(width * 4)), size)


def sparkle(k=4, cx=100, cy=100, r_out=70, bend=0.6, size=200, rot=-90):
    """Star with concave Bezier sides (the flatshapes star model), e.g. a 4-point sparkle."""
    from flatshapes.star import _model
    a = np.radians(rot) + np.arange(k) * 2 * np.pi / k
    T = np.c_[cx + r_out * np.cos(a), cy + r_out * np.sin(a)]
    P = _model(np.array([cx, cy], float), T, np.full(k, bend), 200) * 4
    return _draw(lambda m: cv2.fillPoly(m, [np.round(P).astype(np.int32)], 255), size)
