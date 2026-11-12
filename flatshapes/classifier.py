"""Region classifier: a tiny CNN that tells from a region's mask which shape it is, so detect_shapes fits only
that model instead of trying several (the fits are where the time goes).

Input per region: 2 x 64 x 64 floats - the region mask and the hidden mask (outline covered by objects in front),
cropped to the region's box, padded to a square and shrunk. Output: probabilities over CLASSES.

The model is an ONNX file (models/region_classifier.onnx, trained with notebooks/train_region_classifier.ipynb)
run with onnxruntime on the CPU, all regions of an image in one batch. Without onnxruntime or without the file
everything works as before (the hand-written feature rules pick the detectors).
"""
import json
import os

import numpy as np

SIZE = 64
CLASSES = ["none", "circle", "ellipse", "polygon", "star", "ring", "segment", "teardrop", "heart", "crescent"]
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "region_classifier.onnx")

_SESSION = {"path": None, "sess": None, "classes": CLASSES, "failed": False}


def region_tensor(mask, hidden=None, size=SIZE):
    """2 x size x size float32: region mask and hidden mask, box crop padded to a square (2 px margin)."""
    import cv2
    K = np.asarray(mask, bool)
    ys, xs = np.nonzero(K)
    out = np.zeros((2, size, size), np.float32)
    if not len(xs):
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0) + 4
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    oy, ox = int(round(cy - side / 2)), int(round(cx - side / 2))
    H = np.asarray(hidden, bool) if hidden is not None else np.zeros_like(K)
    for ch, src in enumerate((K, H)):
        sq = np.zeros((side, side), np.float32)
        sy0, sx0 = max(0, oy), max(0, ox)
        sy1, sx1 = min(K.shape[0], oy + side), min(K.shape[1], ox + side)
        if sy1 > sy0 and sx1 > sx0:
            sq[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = src[sy0:sy1, sx0:sx1]
        out[ch] = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    return out


def load(path=None):
    """onnxruntime session for the model (cached), or None when onnxruntime or the file is missing."""
    path = path or DEFAULT_MODEL
    if _SESSION["path"] == path and (_SESSION["sess"] is not None or _SESSION["failed"]):
        return _SESSION["sess"]
    _SESSION.update(path=path, sess=None, failed=True)
    if not os.path.exists(path):
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # detect_shapes already runs regions in several processes
    _SESSION["sess"] = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    meta = os.path.splitext(path)[0] + ".json"
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as fh:
            _SESSION["classes"] = json.load(fh).get("classes", CLASSES)
    _SESSION["failed"] = False
    return _SESSION["sess"]


def classify(regions, path=None, batch=256):
    """regions: list of (mask, hidden). Returns (probabilities (n, n_classes), class names) or (None, None)."""
    sess = load(path)
    if sess is None or not regions:
        return None, None
    X = np.stack([region_tensor(m, h) for m, h in regions])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(X), batch):
        logits = sess.run(None, {name: X[i:i + batch]})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        out.append(e / e.sum(1, keepdims=True))
    return np.vstack(out), _SESSION["classes"]


def kinds_from_probs(p, classes, confident=0.6, cover=0.97, max_kinds=2, skip_none=0.85):
    """Detectors to run for one region from its class probabilities:
    - sure it is no simple shape (none >= skip_none): () - no fit at all
    - confident (top >= confident; training uses label smoothing, so probabilities stay below ~0.95): the best classes until they cover `cover` of the probability (at most
      max_kinds), 'none' left out
    - otherwise None: let the feature rules decide."""
    order = np.argsort(-p)
    top = classes[order[0]]
    if top == "none":
        return () if p[order[0]] >= skip_none else None
    if p[order[0]] < confident:
        return None
    kinds, acc = [], 0.0
    for i in order:
        if classes[i] == "none":
            continue
        kinds.append(classes[i])
        acc += p[i]
        if acc >= cover or len(kinds) >= max_kinds:
            break
    return tuple(kinds)
