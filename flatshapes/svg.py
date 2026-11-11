"""SVG path data for segment lists from geom.fit_open / fit_closed / circle.circle_segments."""
from .result import fmt


def segments_to_d(segs, closed=False):
    """('L', p0, p1) / ('A', p0, p1, r, large, sweep, centre) / ('C', p0, c1, c2, p1) -> SVG path data."""
    if not segs:
        return ""
    d = "M" + fmt(segs[0][1])
    for s in segs:
        if s[0] == "L":
            d += "L" + fmt(s[2])
        elif s[0] == "A":
            d += f"A{s[3]:.2f} {s[3]:.2f} 0 {s[4]} {s[5]} " + fmt(s[2])
        else:
            d += f"C{fmt(s[2])} {fmt(s[3])} {fmt(s[4])}"
    return d + ("Z" if closed else "")
