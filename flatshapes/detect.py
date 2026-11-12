"""Shape classification: every detector is tried on a mask and the model with the fewest parameters whose error is
close to the best wins (a circle beats an ellipse that is only slightly better; a 4-point star, 14 parameters,
beats the 8-vertex polygon through its tips and notches, 26 parameters)."""
import numpy as np

from .circle import fit_circle, equalize_radii
from .ellipse import fit_ellipse
from .polygon import fit_polygon
from .star import fit_star

from .parametric import fit_ring, fit_segment, fit_teardrop, fit_heart, fit_crescent

DETECTORS = {"circle": fit_circle, "ellipse": fit_ellipse, "polygon": fit_polygon, "star": fit_star,
             "ring": fit_ring, "segment": fit_segment, "teardrop": fit_teardrop, "heart": fit_heart,
             "crescent": fit_crescent}
ORDER = ("circle", "ring", "ellipse", "segment", "teardrop", "heart", "crescent", "polygon", "star")  # simple -> complex


def detect_shape(mask, hidden=None, kinds=ORDER, slack=0.15, options=None, prefilter=True, slack_rel=0.5):
    """Best geometric model for one shape mask, or None.
    kinds:   detectors to try; options: {kind: {kwarg: value}} passed to that detector
    slack, slack_rel: a simpler model wins when its mean error is at most best * (1 + slack_rel) + slack px
             (an 8-vertex polygon with rounded corners hugs a sparkle a little closer than the 4-tip star)
    prefilter: cheap shape features (features.py) skip detectors that cannot fit (no star fit on a round blob)"""
    options = options or {}
    fits = []
    if prefilter:
        from .features import shape_features, candidate_kinds
        hs = 0.0
        if hidden is not None:
            import cv2
            ring = (cv2.dilate(np.asarray(mask, np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~np.asarray(mask, bool)
            hs = float(np.asarray(hidden, bool)[ring].mean()) if ring.any() else 0.0
        kinds = candidate_kinds(shape_features(mask), hs, kinds)
    # simple models first; once a fit is so close that no later (more complex) model could beat it under the
    # slack rule - that model would need an error below the ~0.2 px floor of pixel outlines - stop
    stop = 0.2 * (1 + slack_rel) + slack
    for k in sorted(kinds, key=lambda k: ORDER.index(k) if k in ORDER else 99):
        f = DETECTORS[k](mask, hidden, **options.get(k, {}))
        if f is not None:
            fits.append(f)
            if f.metrics["mean"] <= stop:
                break
    if not fits:
        return None
    best = min(f.metrics["mean"] for f in fits)
    ok = [f for f in fits if f.metrics["mean"] <= best * (1 + slack_rel) + slack]
    return min(ok, key=lambda f: (f.n_params, f.metrics["mean"]))


def detect_shapes(labels, background=None, min_area=40, hidden_by_others=True, equal_radius=True, colours=None, rim=4,
                  underlay_share=0.5, lines=None, line_pad=3.0, workers=1, pieces=True, **kw):
    """Run detect_shape on every connected component of every label of a label map (H, W).
    hidden_by_others: outline that touches another non-background label counts as hidden (the neighbour may be in
    front); parts of another label thinner than `rim` px (anti-aliasing fringes) do not count, nor does a label
    that takes up at least `underlay_share` of the ring around the shape (the shape lies on it, e.g. a star on a bag). lines: optional list of (label, ShapeFit line) from find_lines: line pixels are taken out of
    the shapes (strip widened by line_pad px on each side: drawn strokes are not perfectly straight) and
    count as objects in front, so a star with strings glued to it is fitted as a star. colours:
    optional {label: CIELAB} for grouping same-size dots. Returns a list of
    (label, component index, ShapeFit).
    pieces: regions that fit no shape are searched for circles in their outline pieces (circles_in_region).
    workers: processes for the per-region fits (1 = none, None or 0 = all cores); the pool is started once and
    reused. On Windows the calling script needs the usual `if __name__ == "__main__":` guard."""
    out = []
    comps = list(components(labels, background, min_area, hidden_by_others, rim, underlay_share, lines, line_pad))
    pool = get_pool(workers) if workers != 1 and len(comps) >= 6 else None
    if pool is not None:
        # biggest regions first so no worker is left with one large star fit at the end
        order = sorted(range(len(comps)), key=lambda j: -int(comps[j][2].sum()))
        futs = {j: pool.submit(_detect_task, comps[j][2], comps[j][3], kw) for j in order}
        fits = {j: futs[j].result() for j in order}
    else:
        fits = {j: detect_shape(c[2], c[3], **kw) for j, c in enumerate(comps)}
    for j, (lab, i, m, hid, x0, y0) in enumerate(comps):
        f = fits[j]
        # a region that fits no simple shape - or only a many-cornered / rounded polygon, which can wrap two
        # stuck dots - is searched for circles in its outline pieces; circles that cover at least half of it win
        if pieces and (f is None or (f.kind == "polygon" and (f.params.get("corner_r", 0) > 0 or f.params["k"] >= 5))):
            from .circle import circles_in_region
            cs = circles_in_region(m, hid)
            if cs:
                cover = np.zeros(m.shape, bool)
                for cf in cs:
                    cover |= cf.mask
                share = (cover & m).sum() / max(1, m.sum())
                # one circle that is (nearly) the whole region is just the region: the shape fit above decides
                # (a small octagon is close to a circle); stuck shapes leave part of the region outside the circles
                single = len(cs) == 1 and share >= 0.88
                if f is None or (share >= 0.5 and not single):
                    for cf in cs:
                        _shift(cf, x0, y0, np.asarray(labels).shape)
                        out.append((lab, i, cf))
                    continue
        if f is None:
            continue
        _shift(f, x0, y0, np.asarray(labels).shape)
        out.append((lab, i, f))
    if equal_radius:
        circ = [f for _, _, f in out if f.kind == "circle"]
        cols = [colours[l] for l, _, f in out if f.kind == "circle"] if colours else None
        equalize_radii(circ, cols)
    return out


_POOL = {"pool": None, "n": 0}


def get_pool(workers=None):
    """Shared process pool (started on first use, reused by later calls). workers None/0 = all cores."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    n = workers or os.cpu_count() or 1
    if n <= 1:
        return None
    if _POOL["pool"] is None or _POOL["n"] != n:
        if _POOL["pool"] is not None:
            _POOL["pool"].shutdown(cancel_futures=True)
        _POOL["pool"], _POOL["n"] = ProcessPoolExecutor(max_workers=n), n
    return _POOL["pool"]


def _detect_task(m, hid, kw):
    return detect_shape(m, hid, **kw)


def components(labels, background=None, min_area=40, hidden_by_others=True, rim=4, underlay_share=0.5, lines=None,
               line_pad=3.0, body_r=5):
    """Connected components of every label, as detect_shapes sees them: yields (label, index, mask crop, hidden
    crop or None, x0, y0). Line strips (see detect_shapes) are removed from the masks and added to hidden."""
    import cv2
    from scipy import ndimage as ndi
    L = np.asarray(labels)
    line_mask = {}
    for lab, f in (lines or []):
        mm = line_mask.setdefault(int(lab), np.zeros(L.shape, np.uint8))
        p0 = np.round(np.asarray(f.params["p0"]) * 4).astype(int)
        p1 = np.round(np.asarray(f.params["p1"]) * 4).astype(int)
        cv2.line(mm, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1,
                 max(1, int(round(f.params["width"] + 2 * line_pad))), shift=2)
    any_line = np.zeros(L.shape, bool)
    for mm in line_mask.values():
        any_line |= mm > 0
    for lab in np.unique(L):
        if background is not None and lab == background:
            continue
        own = L == lab
        if int(lab) in line_mask:
            # strokes are cut away only outside the thick bodies they run into: a string through the middle of
            # a star must not cut the star in two (the strip still counts as hidden there)
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * body_r + 1, 2 * body_r + 1))
            body = cv2.morphologyEx(own.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
            own &= (line_mask[int(lab)] == 0) | body
        cc, _ = ndi.label(own)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            y0, x0 = max(0, sl[0].start - 3), max(0, sl[1].start - 3)
            y1, x1 = min(L.shape[0], sl[0].stop + 3), min(L.shape[1], sl[1].stop + 3)
            m = cc[y0:y1, x0:x1] == i
            if m.sum() < min_area:
                continue
            hid = occluders(L[y0:y1, x0:x1], m, lab, background, rim, underlay_share) if hidden_by_others else None
            if lines:
                ln = cv2.dilate(any_line[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                hid = ln if hid is None else (hid | ln)
            yield int(lab), i, m, hid, x0, y0


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
    elif f.kind in ("ring", "segment", "teardrop", "heart", "crescent"):
        from .parametric import PATHS
        p["cx"] += dx
        p["cy"] += dy
        f.path = PATHS[f.kind](p)
    elif f.kind == "polygon":
        p["vertices"] = p["vertices"] + [dx, dy]
        from .polygon import polygon_path
        f.path = polygon_path(p["vertices"], p.get("corner_r", 0.0))
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


def find_lines(labels, background=None, **kw):
    """Thin straight strokes of every label (detect_lines + same_sides), as (label, ShapeFit) pairs; pass them
    to detect_shapes(lines=...) to fit shapes with the strokes taken out."""
    from .lines import detect_lines, same_sides
    L = np.asarray(labels)
    out = []
    for lab in np.unique(L):
        if background is not None and lab == background:
            continue
        for f in detect_lines(L == lab, **kw):
            if same_sides(f, L):
                out.append((int(lab), f))
    return out
