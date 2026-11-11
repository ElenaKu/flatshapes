"""Shape classification: every detector is tried on a mask and the model with the fewest parameters whose error is
close to the best wins (a circle beats an ellipse that is only slightly better; a 4-point star, 14 parameters,
beats the 8-vertex polygon through its tips and notches, 26 parameters)."""
import numpy as np

from .circle import fit_circle, equalize_radii
from .ellipse import fit_ellipse
from .polygon import fit_polygon
from .star import fit_star

DETECTORS = {"circle": fit_circle, "ellipse": fit_ellipse, "polygon": fit_polygon, "star": fit_star}
ORDER = ("circle", "ellipse", "polygon", "star")  # simple -> complex


def detect_shape(mask, hidden=None, kinds=ORDER, slack=0.15, options=None):
    """Best geometric model for one shape mask, or None.
    kinds:   detectors to try; options: {kind: {kwarg: value}} passed to that detector
    slack:   a simpler model wins when its mean error is at most `slack` px worse than the best one"""
    options = options or {}
    fits = []
    for k in kinds:
        f = DETECTORS[k](mask, hidden, **options.get(k, {}))
        if f is not None:
            fits.append(f)
    if not fits:
        return None
    best = min(f.metrics["mean"] for f in fits)
    ok = [f for f in fits if f.metrics["mean"] <= best + slack]
    return min(ok, key=lambda f: (f.n_params, f.metrics["mean"]))


def detect_shapes(labels, background=None, min_area=40, hidden_by_others=True, equal_radius=True, colours=None, rim=4,
                  underlay_share=0.5, **kw):
    """Run detect_shape on every connected component of every label of a label map (H, W).
    hidden_by_others: outline that touches another non-background label counts as hidden (the neighbour may be in
    front); parts of another label thinner than `rim` px (anti-aliasing fringes) do not count, nor does a label
    that takes up at least `underlay_share` of the ring around the shape (the shape lies on it, e.g. a star on a bag). colours: optional {label: CIELAB} for grouping same-size dots. Returns a list of
    (label, component index, ShapeFit)."""
    import cv2
    from scipy import ndimage as ndi
    L = np.asarray(labels)
    out = []
    for lab in np.unique(L):
        if background is not None and lab == background:
            continue
        cc, n = ndi.label(L == lab)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            y0, x0 = max(0, sl[0].start - 3), max(0, sl[1].start - 3)
            y1, x1 = min(L.shape[0], sl[0].stop + 3), min(L.shape[1], sl[1].stop + 3)
            m = cc[y0:y1, x0:x1] == i
            if m.sum() < min_area:
                continue
            hid = occluders(L[y0:y1, x0:x1], m, lab, background, rim, underlay_share) if hidden_by_others else None
            f = detect_shape(m, hid, **kw)
            if f is None:
                continue
            _shift(f, x0, y0, L.shape)
            out.append((int(lab), i, f))
    if equal_radius:
        circ = [f for _, _, f in out if f.kind == "circle"]
        cols = [colours[l] for l, _, f in out if f.kind == "circle"] if colours else None
        equalize_radii(circ, cols)
    return out


def _shift(f, dx, dy, shape):
    """Move a fit from crop to image coordinates (params, path and mask)."""
    from . import circle, ellipse, star
    p = f.params
    if f.kind in ("circle", "ellipse"):
        p["cx"] += dx
        p["cy"] += dy
        f.path = circle.circle_path(p["cx"], p["cy"], p["r"]) if f.kind == "circle" else \
            ellipse.ellipse_path((p["cx"], p["cy"]), p["a"], p["b"], p["theta"])
    elif f.kind == "star":
        p["centre"] = p["centre"] + [dx, dy]
        p["tips"] = p["tips"] + [dx, dy]
        f.path = star._path(p["centre"], p["tips"], p["bend"])
    elif f.kind == "polygon":
        p["vertices"] = p["vertices"] + [dx, dy]
        from .result import fmt
        f.path = "M" + "L".join(fmt(v) for v in p["vertices"]) + "Z"
    if f.mask is not None:
        full = np.zeros(shape, bool)
        h, w = f.mask.shape
        full[dy:dy + h, dx:dx + w] = f.mask
        f.mask = full


def occluders(L, m, lab, background=None, rim=4, underlay_share=0.5):
    """Mask of outline that may be covered by objects in front, for component `m` (bool, same crop as label map L)
    of label `lab`. Other labels count, except:
      - the background;
      - a label that fills at least `underlay_share` of the 2 px ring around the shape: the shape lies on it
        (a star on a bag, a dot on a ribbon), it is not in front;
      - parts thinner than `rim` px (anti-aliasing fringes)."""
    import cv2
    m8 = m.astype(np.uint8)
    ring = (cv2.dilate(m8, np.ones((5, 5), np.uint8)) > 0) & ~m
    other = L != lab
    if background is not None:
        other &= L != background
    if ring.any():
        vals, cnt = np.unique(L[ring], return_counts=True)
        for v, c in zip(vals, cnt):
            if v != lab and c >= underlay_share * ring.sum():
                other &= L != v
    body = cv2.morphologyEx(other.astype(np.uint8), cv2.MORPH_OPEN, np.ones((rim + 1, rim + 1), np.uint8))
    return cv2.dilate(body, np.ones((3, 3), np.uint8)) > 0
