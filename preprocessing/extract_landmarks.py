from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ml_models.cnn_image.utils import list_images_with_labels
from shared.feature_engineering import image_path_to_feature


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract simple flattened features and save to NPZ")
    parser.add_argument("--dataset", required=True, help="Path to dataset root. Subfolders are classes")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    parser.add_argument("--img-size", type=int, default=64, help="Square size to resize images to")
    parser.add_argument("--color", action="store_true", help="Use RGB instead of grayscale")
    args = parser.parse_args()

    items = list_images_with_labels(args.dataset)
    target_size = (args.img_size, args.img_size)

    features: List[np.ndarray] = []
    labels: List[str] = []
    paths: List[str] = []

    for path, label in tqdm(items, desc="Extracting features"):
        feat = image_path_to_feature(path, target_size=target_size, grayscale=not args.color)
        features.append(feat.astype(np.float32))
        labels.append(label)
        paths.append(path)

    X = np.stack(features, axis=0)
    labels = np.array(labels)
    paths = np.array(paths)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, X=X, labels=labels, paths=paths)
    print(f"Saved features to: {args.output}")


if __name__ == "__main__":
    main()
