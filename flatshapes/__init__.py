"""flatshapes: geometric shape detection for vectorizing flat illustrations.

Lines, circular arcs, circles, ellipses, polygons and stars, fitted to shape masks with outline hidden by
objects in front left out of the fit. Results carry exact SVG path data.
"""
from .result import ShapeFit, outline_points
from .circle import fit_circle, ransac_circle, kasa, circle_segments, equalize_radii, circle_path, circles_in_region
from .ellipse import fit_ellipse, ellipse_path
from .polygon import fit_polygon
from .star import fit_star
from .lines import detect_lines, merge_collinear, same_sides
from .geom import fit_open, fit_closed, reverse_segments, sample_segment
from .svg import segments_to_d
from .detect import detect_shape, detect_shapes, occluders, find_lines, components, get_pool

from .parametric import fit_ring, fit_segment, fit_teardrop, fit_heart, fit_crescent
from .classifier import region_tensor, classify, kinds_from_probs
from .features import shape_features, candidate_kinds
from .symmetry import Symmetry, find_symmetry, symmetrize
from .motifs import Motif, find_motifs, motifs_svg, split_merged

__version__ = "0.1.0"
