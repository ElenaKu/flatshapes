import numpy as np

import flatshapes as fs
import synthetic as syn


def test_circle():
    f = fs.fit_circle(syn.circle(r=40))
    assert f and abs(f.params["r"] - 40) < 0.6 and abs(f.params["cx"] - 100) < 0.6


def test_circle_hidden_part():
    m = syn.circle(r=40)
    hid = np.zeros_like(m)
    hid[:, 120:] = True  # an object in front covers the right side
    f = fs.fit_circle(m & ~hid, hid)
    assert f and abs(f.params["r"] - 40) < 1.0


def test_ellipse():
    f = fs.fit_ellipse(syn.ellipse(a=60, b=30))
    assert f and abs(max(f.params["a"], f.params["b"]) - 60) < 1.0


def test_detect_prefers_circle():
    assert fs.detect_shape(syn.circle()).kind == "circle"
    assert fs.detect_shape(syn.ellipse()).kind == "ellipse"


def test_star():
    for k in (4, 5, 6, 8):
        f = fs.fit_star(syn.sparkle(k=k))
        assert f and f.params["k"] == k, k


def test_star_hidden_tip():
    m = syn.sparkle(k=4)
    hid = np.zeros_like(m)
    hid[:45, 90:110] = True  # a string covers the top tip
    f = fs.fit_star(m & ~hid, hid)
    assert f and f.params["k"] == 4


def test_sharp_star_is_polygon():
    f = fs.detect_shape(syn.star(k=5))
    assert f.kind == "polygon" and f.params["k"] == 10


def test_polygon():
    f = fs.fit_polygon(syn.polygon([(30, 30), (170, 40), (100, 170)]))
    assert f and f.params["k"] == 3
    assert fs.detect_shape(syn.polygon([(40, 40), (160, 40), (160, 160), (40, 160)])).kind == "polygon"


def test_lines_merge_behind_star():
    m = syn.bar((5, 100), (195, 100), 4) & ~syn.circle(r=30)  # a string with a gap where a disc covers it
    lines = fs.detect_lines(m, min_len=20)
    assert len(lines) == 1 and lines[0].metrics["length"] > 170


def test_equal_radii():
    cs = [fs.fit_circle(syn.circle(r=r)) for r in (20, 20, 21)]
    assert fs.equalize_radii(cs) == 1 and len({round(c.params["r"], 6) for c in cs}) == 1


def test_open_chain_fit():
    t = np.linspace(0, np.pi, 200)
    P = np.c_[50 * np.cos(t), 50 * np.sin(t)]
    segs = fs.fit_open(P)
    assert [s[0] for s in segs] == ["A"]


def test_symmetry():
    s = fs.find_symmetry(syn.star(k=5))
    assert s.kind == "D5" and len(s.axes) == 5
    assert fs.find_symmetry(syn.sparkle(k=4)).order == 4
    assert fs.find_symmetry(syn.circle()).kind == "round"


def test_motifs_rotated_scaled_mirrored():
    import cv2
    base = np.array([(0, 0), (60, 0), (60, 15), (25, 15), (25, 45), (0, 45)], float) - [25, 20]
    L = np.zeros((400, 400), np.uint8)
    for cx, cy, a, sc, mir in [(80, 80, 0, 1, False), (250, 90, 40, 1, False), (100, 280, 120, 0.7, False),
                               (280, 280, -30, 1.3, True)]:
        P = base.copy()
        if mir:
            P[:, 1] *= -1
        t = np.radians(a)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        cv2.fillPoly(L, [np.round(((P @ R.T) * sc + [cx, cy]) * 4).astype(np.int32)], 1, shift=2)
    mos = fs.find_motifs(L, background=0)
    assert len(mos) == 1 and mos[0].count == 4
    assert sum(i["mirrored"] for i in mos[0].instances) == 3  # template is the mirrored copy


def test_split_stuck_copies():
    import cv2

    def balloon(L, cx, cy, a, sc):
        t = np.linspace(0, 2 * np.pi, 100)
        th = np.radians(a)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for Q in (np.c_[30 * np.cos(t), 40 * np.sin(t)], np.array([[-6, 48], [6, 48], [0, 38]])):
            cv2.fillPoly(L, [np.round(((Q @ R.T) * sc + [cx, cy]) * 4).astype(np.int32)], 1, shift=2)

    L = np.zeros((500, 600), np.uint8)
    balloon(L, 80, 80, 0, 1)
    balloon(L, 250, 90, 15, 1.1)
    balloon(L, 420, 90, -10, 0.9)
    balloon(L, 120, 320, -10, 1)   # two balloons stuck together
    balloon(L, 165, 330, 12, 1.05)
    mos = fs.find_motifs(L, background=0)
    split = [i for i in mos[0].instances if i.get("split")]
    assert len(split) == 2
    assert sorted(round(i["cx"] / 10) for i in split) == [12, 17]
