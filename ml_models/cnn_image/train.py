from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ml_models.cnn_image.model import NearestCentroidClassifier
from ml_models.cnn_image.utils import list_images_with_labels, split_dataset
from shared.feature_engineering import image_path_to_feature


def build_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    return label_to_idx, classes


def prepare_features(
    items: List[Tuple[str, str]],
    target_size: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    paths, labels = zip(*items)
    label_to_idx, classes = build_label_mappings(list(labels))

    features: List[np.ndarray] = []
    y_indices: List[int] = []

    for path, label in tqdm(items, desc="Extracting features"):
        feat = image_path_to_feature(path, target_size=target_size, grayscale=grayscale)
        features.append(feat.astype(np.float32))
        y_indices.append(label_to_idx[label])

    X = np.stack(features, axis=0)
    y = np.array(y_indices, dtype=np.int64)
    return X, y, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple image classifier (nearest centroid)")
    parser.add_argument("--dataset", required=True, help="Path to dataset root. Subfolders are classes")
    parser.add_argument("--output", default="models/nearest_centroid.pkl", help="Where to save the trained model")
    parser.add_argument("--img-size", type=int, default=64, help="Square size to resize images to")
    parser.add_argument("--color", action="store_true", help="Use RGB instead of grayscale")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of data for test set")
    args = parser.parse_args()

    items = list_images_with_labels(args.dataset)
    train_items, test_items = split_dataset(items, test_fraction=args.test_fraction, seed=42)

    target_size = (args.img_size, args.img_size)
    X_train, y_train, classes = prepare_features(train_items, target_size=target_size, grayscale=not args.color)
    X_test, y_test, _ = prepare_features(test_items, target_size=target_size, grayscale=not args.color)

    model = NearestCentroidClassifier()
    model.fit(X_train, y_train, classes)

    # Simple evaluation
    preds = model.predict(X_test)
    accuracy = float((preds == y_test).mean()) if len(y_test) else 0.0
    print(f"Test accuracy: {accuracy:.4f} on {len(y_test)} samples")

    # Save model
    model.save(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
