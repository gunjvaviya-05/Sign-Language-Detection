from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np

from ml_models.cnn_image.model import NearestCentroidClassifier
from shared.feature_engineering import image_path_to_feature, is_image_file


def predict_on_paths(model_path: str, paths: List[str], img_size: int, color: bool) -> List[Tuple[str, str, float]]:
    model = NearestCentroidClassifier.load(model_path)
    target_size = (img_size, img_size)

    features: List[np.ndarray] = []
    valid_paths: List[str] = []
    for p in paths:
        if not os.path.isfile(p) or not is_image_file(p):
            continue
        try:
            feat = image_path_to_feature(p, target_size=target_size, grayscale=not color)
            features.append(feat.astype(np.float32))
            valid_paths.append(p)
        except Exception as exc:
            print(f"Skipping {p}: {exc}")

    if not features:
        return []

    X = np.stack(features, axis=0)
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    results: List[Tuple[str, str, float]] = []
    for i, path in enumerate(valid_paths):
        cls_idx = int(preds[i])
        cls_name = model.class_names[cls_idx]
        conf = float(np.max(probs[i]))
        results.append((path, cls_name, conf))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on image(s) using a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    parser.add_argument("--image", required=True, help="Path to an image file or directory of images")
    parser.add_argument("--img-size", type=int, default=64, help="Square size used during training")
    parser.add_argument("--color", action="store_true", help="Use RGB instead of grayscale")
    args = parser.parse_args()

    input_path = args.image
    paths: List[str] = []
    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            fpath = os.path.join(input_path, fname)
            if os.path.isfile(fpath) and is_image_file(fpath):
                paths.append(fpath)
    else:
        paths = [input_path]

    results = predict_on_paths(args.model, paths, args.img_size, args.color)
    if not results:
        print("No valid images to run inference on.")
        return

    for path, label, conf in results:
        print(f"{path}: {label} ({conf:.3f})")


if __name__ == "__main__":
    main()
