from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from shared.feature_engineering import IMAGE_EXTENSIONS, is_image_file


def list_images_with_labels(root_dir: str) -> List[Tuple[str, str]]:
    """
    List (image_path, label) pairs from a dataset directory.
    Expects subdirectories per class. If none, images in root get label "default".
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {root_dir}")

    # Detect class subdirectories that contain at least one image
    class_dirs: List[str] = []
    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if os.path.isdir(full):
            # Check if this directory contains images
            has_img = False
            for fname in os.listdir(full):
                if is_image_file(os.path.join(full, fname)):
                    has_img = True
                    break
            if has_img:
                class_dirs.append(entry)

    pairs: List[Tuple[str, str]] = []
    if class_dirs:
        for cls in class_dirs:
            folder = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if is_image_file(fpath):
                    pairs.append((fpath, cls))
    else:
        # Fall back to images directly under root
        for fname in sorted(os.listdir(root_dir)):
            fpath = os.path.join(root_dir, fname)
            if os.path.isfile(fpath) and is_image_file(fpath):
                pairs.append((fpath, "default"))

    if not pairs:
        raise RuntimeError(
            f"No images found in dataset directory: {root_dir}. "
            f"Supported extensions: {', '.join(IMAGE_EXTENSIONS)}"
        )

    return pairs


def split_dataset(
    items: Sequence[Tuple[str, str]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Randomly split a sequence of (path, label) into train/test lists."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    test_size = int(len(shuffled) * test_fraction)
    test_items = shuffled[:test_size]
    train_items = shuffled[test_size:]
    return train_items, test_items
