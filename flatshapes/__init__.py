"""flatshapes: geometric shape detection for vectorizing flat illustrations.

Lines, circular arcs, circles, ellipses, polygons and stars, fitted to shape masks with outline hidden by
objects in front left out of the fit. Results carry exact SVG path data.
"""
from .result import ShapeFit, outline_points
from .circle import fit_circle, ransac_circle, kasa, circle_segments, equalize_radii, circle_path
from .ellipse import fit_ellipse, ellipse_path
from .polygon import fit_polygon
from .star import fit_star
from .lines import detect_lines, merge_collinear, same_sides
from .geom import fit_open, fit_closed, reverse_segments, sample_segment
from .svg import segments_to_d
from .detect import detect_shape, detect_shapes

__version__ = "0.1.0"
