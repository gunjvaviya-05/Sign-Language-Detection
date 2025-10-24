from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import os

import cv2
import numpy as np

# Common image extensions for simple dataset discovery
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
)


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in IMAGE_EXTENSIONS


def load_image(
    path: str,
    target_size: Tuple[int, int] | None = (64, 64),
    grayscale: bool = True,
) -> np.ndarray:
    """
    Load an image from disk and optionally resize.

    Returns a numpy array in grayscale (H, W) when grayscale=True, else RGB (H, W, 3).
    Raises FileNotFoundError if the image cannot be read.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    if not grayscale:
        # Convert BGR (OpenCV default) to RGB for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 and normalize to [0, 1]."""
    return image.astype(np.float32) / 255.0


def extract_flat_features(image: np.ndarray) -> np.ndarray:
    """
    Produce a simple flattened feature vector from the image.

    This keeps the dependency surface minimal and avoids heavyweight ML frameworks.
    """
    normalized = normalize_image(image)
    return normalized.reshape(-1)


def image_path_to_feature(
    path: str,
    target_size: Tuple[int, int] | None = (64, 64),
    grayscale: bool = True,
) -> np.ndarray:
    image = load_image(path, target_size=target_size, grayscale=grayscale)
    return extract_flat_features(image)
