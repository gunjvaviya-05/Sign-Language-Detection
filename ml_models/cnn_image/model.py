from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
import pickle
import numpy as np


@dataclass
class NearestCentroidClassifier:
    """
    Lightweight classifier that represents each class by the centroid (mean) of its
    training embeddings and predicts by nearest centroid (Euclidean distance).
    """

    class_names: List[str] = field(default_factory=list)
    centroids: Dict[int, np.ndarray] = field(default_factory=dict)
    feature_length: int | None = None

    def fit(self, X: np.ndarray, y_indices: np.ndarray, class_names: List[str]) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array [num_samples, num_features]")
        if X.shape[0] != y_indices.shape[0]:
            raise ValueError("X and y_indices must have the same number of rows")

        self.class_names = list(class_names)
        self.feature_length = X.shape[1]
        self.centroids.clear()

        for idx in range(len(self.class_names)):
            mask = y_indices == idx
            if not np.any(mask):
                # No samples seen for this class; skip
                continue
            self.centroids[idx] = X[mask].mean(axis=0)

    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return distances [num_samples, num_classes] to each class centroid."""
        if self.feature_length is None:
            raise RuntimeError("Model is not fitted yet")
        if X.ndim != 2 or X.shape[1] != self.feature_length:
            raise ValueError("X has wrong shape for this model")

        num_samples = X.shape[0]
        num_classes = len(self.class_names)
        distances = np.empty((num_samples, num_classes), dtype=np.float32)
        for cls_idx in range(num_classes):
            centroid = self.centroids.get(cls_idx)
            if centroid is None:
                distances[:, cls_idx] = np.inf
            else:
                diff = X - centroid[np.newaxis, :]
                distances[:, cls_idx] = np.sqrt(np.sum(diff * diff, axis=1))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self._distance_matrix(X)
        preds = np.argmin(distances, axis=1)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = self._distance_matrix(X)
        # Convert distances to similarities and then softmax
        # Add small epsilon to avoid overflow on large distances
        similarities = -distances
        # Softmax along classes
        max_sim = np.max(similarities, axis=1, keepdims=True)
        exp_sim = np.exp(similarities - max_sim)
        probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        return probs

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "class_names": self.class_names,
            "centroids": self.centroids,
            "feature_length": self.feature_length,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str) -> "NearestCentroidClassifier":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model = NearestCentroidClassifier()
        model.class_names = payload["class_names"]
        model.centroids = payload["centroids"]
        model.feature_length = payload["feature_length"]
        return model
