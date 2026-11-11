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


def _best_rotation(sa, sb):
    """Shift k (bins) and error so that sb[i] ~ sa[i - k]."""
    F = np.fft.irfft(np.conj(np.fft.rfft(sa)) * np.fft.rfft(sb), N_SIG)
    k = int(np.argmax(F))
    err = float(np.sqrt(np.mean((np.roll(sa, k) - sb) ** 2)))
    return k, err


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
                template_tol=0.5):
    """Groups of repeated shapes in a label map. Returns a list of Motif (largest groups first).
    same_colour: copies must have the same label; allow_scale / allow_mirror: accept resized / mirrored copies;
    desc_tol / sig_tol: signature gates (normalised radius units); min_iou: overlay check of every copy.
    components: optional output of flatshapes.components(...) to reuse (e.g. with lines removed)."""
    from .detect import components as comps_fn
    comps = components if components is not None else comps_fn(labels, background, min_area, False)
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
            k, err = _best_rotation(da["sig"], db["sig"])
            mir = False
            if allow_mirror:
                sm = da["sig"][::-1].copy()
                km, errm = _best_rotation(sm, db["sig"])
                if errm < err:
                    k, err, mir = km, errm, True
            if err > sig_tol:
                continue
            angle = k * 360.0 / N_SIG  # the reversed signature is exactly the mirror (about the x axis)
            iou = _overlay_iou(da, db, angle, scale, mir)
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
    motifs.sort(key=lambda m: -m.count)
    return motifs


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
