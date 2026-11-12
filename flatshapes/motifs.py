"""Repeated motifs: the same shape drawn several times (stars, leaves, dots, snowflakes), each copy moved,
rotated, scaled and possibly mirrored. Every group gets one template and a transform per copy, so the SVG can
hold the motif once (<symbol>) and place it with <use> - smaller files, and editing the template edits all.

Matching: a rotation-invariant radial signature (outline distance from the centroid by angle, normalised by
size) gates the candidates; the rotation comes from circular cross-correlation of the signatures; every match
is confirmed by overlaying the transformed template on the copy (IoU).
"""
from dataclasses import dataclass, field

import numpy as np

from .result import outline_points, fmt

N_SIG = 128


@dataclass
class Motif:
    template: dict                   # label, index, cx, cy, r (equivalent radius), outline (n, 2) around centroid
    instances: list = field(default_factory=list)   # dicts: label, index, cx, cy, angle (deg), scale, mirrored, iou
    path: str = ""                   # template outline as SVG path data, centred on (0, 0)

    @property
    def count(self):
        return len(self.instances)


def _signature(P, c, R):
    """Max outline distance from the centroid in N_SIG angle bins, divided by the size."""
    d = P - c
    ang = (np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)) / (2 * np.pi) * N_SIG
    rad = np.hypot(d[:, 0], d[:, 1]) / R
    s = np.full(N_SIG, np.nan)
    b = np.minimum(ang.astype(int), N_SIG - 1)
    for i in range(N_SIG):
        v = rad[b == i]
        if len(v):
            s[i] = v.max()
    ok = ~np.isnan(s)
    if ok.sum() < 3:
        return None
    idx = np.arange(N_SIG)
    s[~ok] = np.interp(idx[~ok], idx[ok], s[ok], period=N_SIG)
    return s


def _describe(m, x0, y0):
    P, _ = outline_points(m, None, every=1)
    if len(P) < 12:
        return None
    ys, xs = np.nonzero(m)
    c = np.array([xs.mean(), ys.mean()]) + 0.5
    R = float(np.sqrt(len(xs) / np.pi))
    s = _signature(P, c, R)
    if s is None:
        return None
    return dict(c=c + [x0, y0], R=R, sig=s, desc=np.abs(np.fft.rfft(s))[:16] / N_SIG,
                P=P - c, area=len(xs))


def _best_rotation(sa, sb, n=1):
    """Shift k (bins) and error so that sb[i] ~ sa[i - k]; with n > 1 the n best shifts (local correlation
    peaks), best first, as a list."""
    F = np.fft.irfft(np.conj(np.fft.rfft(sa)) * np.fft.rfft(sb), N_SIG)
    if n == 1:
        k = int(np.argmax(F))
        return k, float(np.sqrt(np.mean((np.roll(sa, k) - sb) ** 2)))
    peaks = [i for i in range(N_SIG) if F[i] >= F[i - 1] and F[i] >= F[(i + 1) % N_SIG]]
    peaks = sorted(peaks, key=lambda i: -F[i])[:n]
    return [(k, float(np.sqrt(np.mean((np.roll(sa, k) - sb) ** 2)))) for k in peaks]


def _overlay_iou(da, db, angle, scale, mirrored, shape_pad=4):
    """IoU of template outline polygon da (transformed) and copy db, drawn in a common local frame."""
    import cv2
    A = da["P"].copy()
    if mirrored:
        A[:, 1] = -A[:, 1]
    t = np.radians(angle)
    Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = (A @ Rm.T) * scale
    B = db["P"]
    lo = np.minimum(A.min(0), B.min(0)) - shape_pad
    hi = np.maximum(A.max(0), B.max(0)) + shape_pad
    size = np.ceil(hi - lo).astype(int) + 1
    ma = np.zeros((size[1], size[0]), np.uint8)
    mb = np.zeros_like(ma)
    cv2.fillPoly(ma, [np.round((A - lo) * 4).astype(np.int32)], 1, shift=2)
    cv2.fillPoly(mb, [np.round((B - lo) * 4).astype(np.int32)], 1, shift=2)
    u = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / u if u else 0.0


def find_motifs(labels, background=None, min_area=60, same_colour=True, allow_scale=True, allow_mirror=True,
                desc_tol=0.06, sig_tol=0.08, min_iou=0.85, scale_range=(0.25, 4.0), min_count=2, components=None,
                template_tol=0.5, split=True, split_kw=None):
    """Groups of repeated shapes in a label map. Returns a list of Motif (largest groups first).
    same_colour: copies must have the same label; allow_scale / allow_mirror: accept resized / mirrored copies;
    desc_tol / sig_tol: signature gates (normalised radius units); min_iou: overlay check of every copy.
    components: optional output of flatshapes.components(...) to reuse (e.g. with lines removed).
    split: regions that matched no motif are tried as several copies stuck together (two balloons that touch, a
    balloon with its string); found copies join the motif with split=True (see split_merged)."""
    from .detect import components as comps_fn
    comps = list(components if components is not None else comps_fn(labels, background, min_area, False))
    items = []
    for lab, i, m, _hid, x0, y0 in comps:
        d = _describe(m, x0, y0)
        if d is not None:
            d.update(label=lab, index=i)
            items.append(d)
    items.sort(key=lambda d: -d["area"])
    used = [False] * len(items)
    motifs = []
    for a, da in enumerate(items):
        if used[a]:
            continue
        group = [dict(label=da["label"], index=da["index"], cx=float(da["c"][0]), cy=float(da["c"][1]),
                      angle=0.0, scale=1.0, mirrored=False, iou=1.0)]
        members = [a]
        for b in range(a + 1, len(items)):
            db = items[b]
            if used[b] or (same_colour and db["label"] != da["label"]):
                continue
            scale = db["R"] / da["R"]
            if not allow_scale and abs(scale - 1) > 0.08:
                continue
            if not scale_range[0] <= scale <= scale_range[1]:
                continue
            if np.abs(da["desc"] - db["desc"]).max() > desc_tol:
                continue
            # a few rotation candidates (near-symmetric shapes - a balloon with a small knot - have a second peak
            # half a turn away); the overlay decides. The reversed signature is exactly the mirror (x axis).
            cands = [(k, e, False) for k, e in _best_rotation(da["sig"], db["sig"], 3)]
            if allow_mirror:
                cands += [(k, e, True) for k, e in _best_rotation(da["sig"][::-1].copy(), db["sig"], 3)]
            cands = [c for c in cands if c[1] <= sig_tol]
            if not cands:
                continue
            # ties (a mirror-symmetric shape) go to the copy without mirroring
            _, _, iou, angle, mir = max((round(o, 2), not m, o, k * 360.0 / N_SIG, m) for o, k, m in
                                        ((_overlay_iou(da, db, k * 360.0 / N_SIG, scale, m), k, m) for k, _, m in cands))
            if iou < min_iou:
                continue
            group.append(dict(label=db["label"], index=db["index"], cx=float(db["c"][0]), cy=float(db["c"][1]),
                              angle=round(angle % 360, 2), scale=round(scale, 4), mirrored=mir, iou=round(float(iou), 4)))
            members.append(b)
        if len(group) >= min_count:
            for b in members:
                used[b] = True
            motifs.append(Motif(template=dict(label=da["label"], index=da["index"], cx=float(da["c"][0]),
                                              cy=float(da["c"][1]), r=da["R"], outline=da["P"]),
                                instances=group, path=_template_path(da["P"], template_tol)))
    if split and motifs:
        left = [(c[0], c[1], c[2], c[4], c[5], c[3]) for c in comps]
        done = {(it["label"], it["index"]) for mo in motifs for it in mo.instances}
        left = [c for c in left if (c[0], c[1]) not in done]
        split_merged(motifs, left, image_shape=np.asarray(labels).shape, **(split_kw or {}))
    motifs.sort(key=lambda m: -m.count)
    return motifs


def _shrink(img, d):
    """Area-average downscale by an integer factor d (1 = unchanged)."""
    import cv2
    if d <= 1:
        return img
    h, w = img.shape
    return cv2.resize(img, (max(1, w // d), max(1, h // d)), interpolation=cv2.INTER_AREA)


def _place_scores(region, todo, band, vis, tm, edge, min_inside, min_edge):
    """Score of every placement of template tm (filled) / edge (its outline) over a region: on_edge^4 *
    sqrt(new share), -1 where the copy sticks out (inside < min_inside), adds little (< 25 % new) or does not
    follow the outline (on_edge < min_edge). Returns (score map, inside map) or (None, None) if it does not fit."""
    import cv2
    th, tw = tm.shape
    if th > region.shape[0] or tw > region.shape[1]:
        return None, None
    ta = float(tm.sum())
    inside = cv2.matchTemplate(region, tm, cv2.TM_CCORR) / ta
    new = cv2.matchTemplate(todo, tm, cv2.TM_CCORR)
    # a true copy runs along the region outline (except where another copy overlaps it); a copy placed across
    # two stuck copies sits inside them. Judged on the copy's outline that is not hidden.
    e_vis = cv2.matchTemplate(vis, edge, cv2.TM_CCORR)
    on_edge = cv2.matchTemplate(band, edge, cv2.TM_CCORR) / np.maximum(e_vis, 1e-6)
    on_edge[e_vis < 0.5 * float(edge.sum())] = 0  # mostly hidden: no evidence
    # a copy may reach under objects in front, but needlessly much of it there means it is too big or tilted
    # (the hidden area always "fits"): prefer copies that stay on what is seen
    under = np.clip(1.0 - cv2.matchTemplate(vis, tm, cv2.TM_CCORR) / ta, 0, 1)
    score = on_edge ** 4 * np.sqrt(np.clip(new / ta, 0, 1)) * (1 - 0.8 * under)
    score[(inside < min_inside) | (new < 0.25 * ta) | (on_edge < min_edge)] = -1
    return score, inside


def _template_raster(P, angle, scale, mirrored):
    """Filled template (outline P around its centroid) transformed; returns mask and centroid position in it."""
    import cv2
    A = P.copy()
    if mirrored:
        A[:, 1] = -A[:, 1]
    t = np.radians(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A = (A @ R.T) * scale
    lo = np.floor(A.min(0)) - 1
    size = (np.ceil(A.max(0) - lo) + 2).astype(int)
    m = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(m, [np.round((A - lo) * 4).astype(np.int32)], 1, shift=2)
    return m, -lo


def split_merged(motifs, regions, min_inside=0.92, min_cover=0.85, coarse_step=15.0, fine_step=5.0,
                 coarse_size=14.0, n_refine=3, scale_steps=(0.9, 1.0, 1.1),
                 max_copies=6, area_range=(0.6, 6.0), min_edge=0.65, image_shape=None):
    """Explain a region as copies of a known motif stuck together. regions: (label, index, mask crop, x0, y0) of
    the components that matched no motif, optionally with a 6th item: hidden crop (outline covered by objects in
    front; a copy may extend under it). For each region of a motif's colour, copies are placed greedily: the
    placement (angle, scale, mirror) that lies inside the region (>= min_inside of the copy), follows the region
    outline best and covers still unexplained pixels, until the copies cover min_cover of the region. Scales tried: those of the motif's
    copies times scale_steps. A region explained by >= 1 copy (one copy = the motif with something attached,
    e.g. a string) adds its copies to the motif (split=True). Returns the number of copies added.
    A copy must run along the region outline for at least min_edge of its own outline. Motifs that are plain
    triangles / quadrilaterals are not used: they tile any bar or band (a ribbon would be cut into pieces).
    Search: coarse_step-degree angles on images shrunk so the motif is ~coarse_size px in radius, the best
    n_refine placements refined at full size with fine_step-degree angles.
    image_shape (H, W): where a region touches the image border, copies may extend past it (cut-off copies)."""
    import cv2
    added = 0
    from .polygon import fit_polygon
    for mo in motifs:
        T = mo.template
        tm0, _ = _template_raster(T["outline"], 0.0, 1.0, False)
        poly = fit_polygon(np.pad(tm0, 3) > 0)
        if poly is not None and poly.params["k"] <= 4:
            continue
        tarea = np.pi * T["r"] ** 2
        scales = sorted({round(it["scale"] * f, 3) for it in mo.instances for f in scale_steps})
        mirrors = (False, True) if any(it["mirrored"] for it in mo.instances) else (False,)
        # the template's own symmetry cuts the search: a dot needs one angle, a 5-point star 72 degrees,
        # a shape with a mirror axis no mirrored copies
        from .symmetry import find_symmetry
        sy = find_symmetry(np.pad(tm0, 3) > 0, max_order=8)
        turn = 360.0 if sy.round else 360.0 / max(1, sy.order)
        if sy.axes:
            mirrors = (False,)
        # coarse to fine: all placements are scored on images shrunk by `d` with angles every coarse_step degrees,
        # the best few are refined at full size near their spot with angles every fine_step degrees
        d = max(1, int(round(T["r"] * min(scales) / coarse_size)))
        angles = [0.0] if sy.round else list(np.arange(0, turn, coarse_step))
        full_cache = {}

        def full_raster(a, sc, mi):
            key = (round(a % 360, 3), sc, mi)
            if key not in full_cache:
                tm, off = _template_raster(T["outline"], a, sc, mi)
                edge = (tm - cv2.erode(tm, np.ones((3, 3), np.uint8))).astype(np.float32)
                full_cache[key] = (tm, off, edge)
            return full_cache[key]

        coarse = []
        base_scales = sorted({round(it["scale"] * 20) / 20 for it in mo.instances})  # refine tries scale_steps
        for sc in base_scales:
            for mi in mirrors:
                for a in angles:
                    tm, _, _ = full_raster(a, sc, mi)
                    tl = _shrink(tm.astype(np.float32), d)
                    tb = (tl > 0.5).astype(np.uint8)
                    el = (tb - cv2.erode(tb, np.ones((3, 3), np.uint8))).astype(np.float32)
                    if tl.sum() > 0 and el.sum() > 0:
                        coarse.append((a, sc, mi, tl, el))
        for reg in regions:
            lab, idx, m, x0, y0 = reg[:5]
            hid = reg[5] if len(reg) > 5 and reg[5] is not None else np.zeros_like(m)
            if lab != T["label"]:
                continue
            if image_shape is not None:  # room past the image border, counted as hidden
                pad = int(np.ceil(2 * T["r"] * max(scales))) + 2
                H, W = image_shape
                h0, w0 = m.shape
                sides = (pad if y0 == 0 else 0, pad if y0 + h0 >= H else 0, pad if x0 == 0 else 0,
                         pad if x0 + w0 >= W else 0)
                if any(sides):
                    m = np.pad(m, ((sides[0], sides[1]), (sides[2], sides[3])))
                    hid = np.pad(hid, ((sides[0], sides[1]), (sides[2], sides[3])), constant_values=True)
                    x0, y0 = x0 - sides[2], y0 - sides[0]
            area = int(m.sum())
            if not area_range[0] * tarea * min(scales) ** 2 <= area <= area_range[1] * tarea * max(scales) ** 2:
                continue
            # a copy may extend under objects in front; its outline there counts as matching
            region = (m | hid).astype(np.float32)
            m8 = m.astype(np.uint8)
            band = (cv2.dilate(m8 - cv2.erode(m8, np.ones((3, 3), np.uint8)), np.ones((5, 5), np.uint8)) > 0) & ~hid
            vis = (~hid).astype(np.float32)
            region_l, vis_l = _shrink(region, d), _shrink(vis, d)
            todo = m.copy()
            copies = []
            while len(copies) < max_copies and todo.sum() > (1 - min_cover) * area:
                todo_f = todo.astype(np.float32)
                # outline of a copy behind copies already placed is not visible: it counts as matching there
                band_now = (band | (~todo & m)).astype(np.float32)
                todo_l, band_l = _shrink(todo_f, d), np.minimum(1.0, _shrink(band_now, d) * 2)
                cands = []
                for a, sc, mi, tl, el in coarse:
                    sc_map, _ = _place_scores(region_l, todo_l, band_l, vis_l, tl, el, min_inside - 0.1,
                                              min_edge - 0.2)
                    if sc_map is None:
                        continue
                    k = int(np.argmax(sc_map))
                    yy, xx = divmod(k, sc_map.shape[1])
                    if sc_map[yy, xx] > 0:
                        cands.append((float(sc_map[yy, xx]), a, sc, mi, xx * d, yy * d))
                cands.sort(key=lambda c: -c[0])
                best = None
                tries = []
                for _, a0, sc0, mi, X, Y in cands[:n_refine]:
                    for sc in (round(sc0 * f, 3) for f in scale_steps):
                        for a in ([a0] if sy.round else a0 + np.arange(-coarse_step / 2, coarse_step / 2 + 1e-6, fine_step)):
                            tries.append((a, sc, mi, X, Y))
                for a, sc, mi, X, Y in tries:
                    tm, off, edge = full_raster(a, sc, mi)
                    th, tw = tm.shape
                    r0, c0 = max(0, Y - 2 * d - 2), max(0, X - 2 * d - 2)
                    r1, c1 = min(m.shape[0], Y + 2 * d + 3 + th), min(m.shape[1], X + 2 * d + 3 + tw)
                    if r1 - r0 < th or c1 - c0 < tw:
                        continue
                    win = (slice(r0, r1), slice(c0, c1))
                    sc_map, inside = _place_scores(region[win], todo_f[win], band_now[win], vis[win],
                                                   tm.astype(np.float32), edge, min_inside, min_edge)
                    if sc_map is None:
                        continue
                    k = int(np.argmax(sc_map))
                    yy, xx = divmod(k, sc_map.shape[1])
                    if sc_map[yy, xx] > 0 and (best is None or sc_map[yy, xx] > best[0]):
                        best = (float(sc_map[yy, xx]), float(a % 360), sc, mi, xx + c0, yy + r0, tm, off,
                                float(inside[yy, xx]))
                if best is None:
                    break
                _, a, sc, mi, xx, yy, tm, off, ins = best
                th, tw = tm.shape
                todo[yy:yy + th, xx:xx + tw] &= ~tm.astype(bool)
                copies.append(dict(label=lab, index=idx, cx=float(x0 + xx + off[0]), cy=float(y0 + yy + off[1]),
                                   angle=round(a, 2), scale=float(sc), mirrored=bool(mi), iou=round(ins, 4), split=True))
            if copies and todo.sum() <= (1 - min_cover) * area:
                mo.instances.extend(copies)
                added += len(copies)
    return added


def _template_path(P, tol):
    """Template outline (centred) as SVG path data via the line / arc / Bezier fitter."""
    from .geom import fit_closed
    from .svg import segments_to_d
    Q = P[:: max(1, len(P) // 400)]
    try:
        return segments_to_d(fit_closed(Q, tol=tol, line_tol=tol * 0.7, arc_tol=tol), closed=True)
    except Exception:  # fall back to the polygon of the outline
        return "M" + "L".join(fmt(p) for p in Q) + "Z"


def motifs_svg(motifs, palette=None, width=None, height=None):
    """SVG document with every motif defined once and placed with <use>. palette: {label: '#rrggbb'}."""
    defs, uses = [], []
    for n, mo in enumerate(motifs):
        defs.append(f'<symbol id="motif{n}" overflow="visible"><path d="{mo.path}"/></symbol>')
        for it in mo.instances:
            sx = it["scale"]
            sy = -it["scale"] if it["mirrored"] else it["scale"]
            fill = (palette or {}).get(it["label"], "#888")
            uses.append(f'<use href="#motif{n}" fill="{fill}" transform="translate({it["cx"]:.2f} {it["cy"]:.2f}) '
                        f'rotate({it["angle"]:.2f}) scale({sx:.4f} {sy:.4f})"/>')
    size = f' width="{width}" height="{height}" viewBox="0 0 {width} {height}"' if width and height else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg"{size}><defs>' + "".join(defs) + "</defs>"
            + "".join(uses) + "</svg>")
