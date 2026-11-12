Put `region_classifier.onnx` and `region_classifier.json` here (made by
`notebooks/train_region_classifier.ipynb`). With them and `pip install onnxruntime`, `detect_shapes` lets the
classifier pick the detectors per region; without them it uses the feature rules.
