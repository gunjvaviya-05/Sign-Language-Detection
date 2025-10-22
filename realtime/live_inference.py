from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ml_models.cnn_image.model import NearestCentroidClassifier
from realtime.overlay import draw_fps, draw_label
from shared.feature_engineering import extract_flat_features


def preprocess_frame(frame, img_size: int, color: bool) -> np.ndarray:
    if not color:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if color:
        feat = resized[:, :, ::-1]  # BGR to RGB
    else:
        feat = resized
    feat = feat.astype(np.float32) / 255.0
    return feat.reshape(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam inference with a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    parser.add_argument("--img-size", type=int, default=64, help="Square size used during training")
    parser.add_argument("--color", action="store_true", help="Use RGB instead of grayscale")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the camera device")
    args = parser.parse_args()

    model = NearestCentroidClassifier.load(args.model)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        X = preprocess_frame(frame, args.img_size, args.color)
        probs = model.predict_proba(X)
        cls_idx = int(np.argmax(probs[0]))
        label = model.class_names[cls_idx]
        conf = float(np.max(probs[0]))

        draw_label(frame, f"{label} ({conf:.2f})")

        now = time.time()
        dt = now - last
        if dt > 0:
            fps = 1.0 / dt
        draw_fps(frame, fps)
        last = now

        cv2.imshow("Live Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
