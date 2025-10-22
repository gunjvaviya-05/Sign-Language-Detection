from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import functools
import os

import cv2
import numpy as np

# Optional mediapipe import; code must work without it
try:
    import mediapipe as mp  # type: ignore
    HAS_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    HAS_MEDIAPIPE = False

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


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 and normalize to [0, 1]."""
    return image.astype(np.float32) / 255.0


def _load_image_bgr(
    path: str,
    target_size: Tuple[int, int] | None,
) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def extract_flat_features(image_any: np.ndarray) -> np.ndarray:
    """
    Produce a simple flattened feature vector from the image array.
    Accepts grayscale (H, W) or color (H, W, 3) arrays.
    """
    normalized = normalize_image(image_any)
    return normalized.reshape(-1)


# ----- MediaPipe landmark extraction (optional) -----

_HOLISTIC = None  # cached holistic detector


def _get_holistic():
    global _HOLISTIC
    if not HAS_MEDIAPIPE:
        return None
    if _HOLISTIC is None:
        _HOLISTIC = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )
    return _HOLISTIC


def _flatten_landmarks(landmarks, dims: int, max_points: int) -> np.ndarray:
    """
    Flatten a list of normalized landmarks into a fixed-length vector of size max_points * dims.
    Pads with zeros when landmarks are missing or count is smaller than max_points.
    """
    if landmarks is None or landmarks.landmark is None:
        return np.zeros((max_points * dims,), dtype=np.float32)
    pts = landmarks.landmark
    arr = np.zeros((max_points, dims), dtype=np.float32)
    for i in range(min(len(pts), max_points)):
        lm = pts[i]
        if dims == 4:
            arr[i] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
        elif dims == 3:
            arr[i] = [lm.x, lm.y, lm.z]
        else:
            arr[i] = [lm.x, lm.y, 0.0][:dims]
    return arr.reshape(-1)


def extract_mediapipe_features_rgb(image_rgb: np.ndarray) -> np.ndarray | None:
    """
    Extract a fixed-length vector of holistic landmarks from an RGB image using MediaPipe.
    Returns None when MediaPipe is unavailable.
    Output vector layout: [face(468*3), pose(33*4), left(21*3), right(21*3)] length 1662.
    """
    if not HAS_MEDIAPIPE:
        return None
    holistic = _get_holistic()
    results = holistic.process(image_rgb)
    face_vec = _flatten_landmarks(results.face_landmarks, dims=3, max_points=468)
    pose_vec = _flatten_landmarks(results.pose_landmarks, dims=4, max_points=33)
    left_vec = _flatten_landmarks(results.left_hand_landmarks, dims=3, max_points=21)
    right_vec = _flatten_landmarks(results.right_hand_landmarks, dims=3, max_points=21)
    return np.concatenate([face_vec, pose_vec, left_vec, right_vec], axis=0)


# ----- Public helpers -----

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
    image_bgr = _load_image_bgr(path, target_size)
    if grayscale:
        return _to_gray(image_bgr)
    return _to_rgb(image_bgr)


def array_to_feature(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] | None = (64, 64),
    grayscale: bool = True,
    use_mediapipe: bool | None = None,
) -> np.ndarray:
    """
    Convert a BGR image array into a feature vector.
    - When use_mediapipe is True and available, extract holistic landmarks from RGB image (ignores grayscale).
    - Otherwise, resize and flatten (grayscale or RGB).
    """
    if use_mediapipe is None:
        use_mediapipe = HAS_MEDIAPIPE

    # For MediaPipe, we need RGB
    if use_mediapipe and HAS_MEDIAPIPE:
        if target_size is not None:
            image_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
        image_rgb = _to_rgb(image_bgr)
        mp_feat = extract_mediapipe_features_rgb(image_rgb)
        if mp_feat is not None:
            return mp_feat.astype(np.float32)
        # Fallback in rare case of failure

    # Fallback: grayscale or color flatten
    if target_size is not None:
        image_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    if grayscale:
        arr = _to_gray(image_bgr)
    else:
        arr = _to_rgb(image_bgr)
    return extract_flat_features(arr).astype(np.float32)


def image_path_to_feature(
    path: str,
    target_size: Tuple[int, int] | None = (64, 64),
    grayscale: bool = True,
    use_mediapipe: bool | None = None,
) -> np.ndarray:
    image_bgr = _load_image_bgr(path, target_size)
    return array_to_feature(
        image_bgr=image_bgr,
        target_size=None,  # resizing already applied above
        grayscale=grayscale,
        use_mediapipe=use_mediapipe,
    )
