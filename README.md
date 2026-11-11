# flatshapes

Geometric shape detection for vectorizing flat illustrations (clipart, icons, flat backgrounds).
Finds **straight lines, circular arcs, circles, ellipses, polygons and stars** in shape masks and returns exact
SVG path data, so a dot becomes a true circle and a sparkle becomes a clean 4-point star instead of a cloud of
Bezier nodes.

What makes it different from generic detectors (OpenCV Hough, scikit-image RANSAC):

- **Occlusion-aware.** Every detector takes an optional `hidden` mask. Outline under an object in front (a string
  crossing a star, a balloon over a disc) is left out of the fit, and hidden star tips are restored from symmetry.
- **Star model with concave sides.** k tips plus one bend per side (cubic Beziers towards the centre), fitted with
  robust least squares.
- **Simplest model wins.** `detect_shape` tries all models and keeps the simplest one whose error is close to the
  best (circle > ellipse > polygon > star).
- **Group regularity.** Same-size dots of one colour get one radius; pieces of one straight line on both sides of a
  shape are joined into one line.
- **Vectorizer-ready.** Outline chains can be fitted as line / arc / cubic segments (`fit_open`, `fit_closed`);
  a shared edge fitted once and reversed for the neighbour leaves no gaps.

Extracted from the `core-v2` branch of flat2svg (ElenaKu/scripts).

## Symmetry and repeated motifs

```python
sym = fs.find_symmetry(mask, hidden)   # kind: "D5" (rotation x5 + mirrors), "C3", "mirror", "round", "none"
sym.order, sym.axes                    # rotational order, mirror axes in degrees
clean = fs.symmetrize(mask, sym)       # exactly symmetric version of a slightly uneven shape

motifs = fs.find_motifs(labels, background=0)    # same shape drawn several times
for mo in motifs:
    mo.instances                       # cx, cy, angle, scale, mirrored, iou per copy
svg = fs.motifs_svg(motifs, palette)   # each motif once as <symbol>, copies as <use transform=...>
```

Copies may be moved, rotated, scaled and mirrored; every match is confirmed by overlaying the transformed
template (IoU >= 0.85).

## Install

```
pip install git+https://github.com/ElenaKu/flatshapes
```

Requires numpy, scipy, opencv-python, scikit-image.

## Usage

```python
import numpy as np
import flatshapes as fs

mask = ...                      # bool (H, W): one shape
hidden = ...                    # optional bool (H, W): outline covered by objects in front

fit = fs.detect_shape(mask, hidden)
if fit:
    print(fit.kind, fit.params, fit.metrics)
    svg = f'<path d="{fit.path}"/>'

# single detectors
fs.fit_circle(mask, hidden)     # params: cx, cy, r
fs.fit_ellipse(mask, hidden)    # params: cx, cy, a, b, theta
fs.fit_polygon(mask, hidden)    # params: vertices, k
fs.fit_star(mask, hidden)       # params: centre, tips, bend, k
fs.detect_lines(mask)           # thin straight strokes -> [ShapeFit(kind="line")]

# whole label map (palette index per pixel)
for label, comp, fit in fs.detect_shapes(labels, background=0):
    ...

# outline chains (dense polylines in px)
segs = fs.fit_open(points)      # [('L', ...), ('A', ...), ('C', ...)]
d = fs.segments_to_d(segs)
```

Every `ShapeFit` has `kind`, `params`, `path` (SVG path data), `metrics` (mean / p80 / max outline distance in px,
visible share) and `mask` (model mask).

## Defaults

Thresholds are the values tuned in flat2svg on real illustration packs; all are keyword arguments.
Distance limits grow with the shape size (`rel_mean` = 0.022, `rel_p80` = 0.035 of the equivalent radius, at most
1.5x the absolute limit): edges of generated art wobble by a share of the shape size.

| detector | accepted when |
|---|---|
| circle | mean dist <= 0.45 px (best 90 % of points), p80 <= 0.9 px, visible outline covers the circle (gap <= 200 deg) |
| ellipse | mean <= 0.5 px, p80 <= 1.0 px, gap <= 200 deg, axis ratio <= 6 |
| polygon | 3..12 sides, one common corner radius (0 = sharp), mean <= 0.8 px, p80 <= 1.3 px, no two neighbouring corners under 25 deg (a curve cut into chords) |
| star | 4..8 tips, tip >= 1.5 x inscribed radius, mean <= 0.7 px (best 85 %), p80 <= 1.5 px, area error <= 6 % |
| line | length >= 20 px, width <= 9 px, empty space on both sides; pieces join at <= 4 deg, <= 2 px offset, <= 90 px gap |

A star with sharp inner corners (a classic 5-point star) is a 10-vertex polygon; the star model is for stars with
curved sides (sparkles, twinkles).

## Trying it on images

```
python tools/shapes_report.py INPUT_DIR OUTPUT_DIR --colors 8
```

Walks INPUT_DIR recursively (same folder structure in OUTPUT_DIR), quantizes each image with k-means, runs
`detect_shapes` and `detect_lines`, and writes `<name>_shapes.png` (shapes outlined by kind), `<name>_shapes.json`
and `summary.csv`.

## Test bench (web UI)

```
python tools/webui/server.py [IMAGE_DIR] [--verdicts VERDICT_DIR] [--port 8765]
```

On Windows just double-click `start_ui.bat`. Opens a local page (http://localhost:8765, standard library server,
nothing extra to install):

- **📁 Folder**: folder browser (drives, home, recent folders, type or paste a path) to choose the image folder;
  the last folder is reopened on the next start

- file list of IMAGE_DIR (subfolders included); detected shapes drawn over the image, kinds toggled with tabs;
  wheel to zoom, drag to pan, `F` to fit
- **Settings**: sliders for every detector limit (tabs per detector), detection re-runs on each change
- **Inspector**: click any spot to see every model fitted to that region, its errors and pass / fail, and draw it
- **Verdicts**: mark shapes correct (`C`) / wrong (`X`), mark missed shapes by kind; saved as JSON under
  VERDICT_DIR with the same folder structure (default IMAGE_DIR/../flatshapes_verdicts), totals and precision per kind

## Tuning limits from verdicts

```
python tools/tune.py IMAGE_DIR [--verdicts VERDICT_DIR]
```

Re-fits only the regions you marked in the test bench and searches the detector limits for the settings that
satisfy the most marks (correct kept, wrong removed, missed found). Writes `VERDICT_DIR/tuned_settings.json`;
load it in the web UI with **Load tuned settings** (Settings tab).

## Tests

```
pip install -e .[test]
pytest tests
```

`tests/synthetic.py` draws test shapes (circles, ellipses, polygons, stars, sparkles, bars) at 4x and downsamples.

## License

MIT
