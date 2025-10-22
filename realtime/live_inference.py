from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ml_models.cnn_image.model import NearestCentroidClassifier
from realtime.overlay import draw_fps, draw_label
from shared.feature_engineering import array_to_feature


def preprocess_frame(frame, img_size: int, color: bool, use_mediapipe: bool | None) -> np.ndarray:
    feat = array_to_feature(
        image_bgr=frame,
        target_size=(img_size, img_size),
        grayscale=not color,
        use_mediapipe=use_mediapipe,
    )
    return feat.reshape(1, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam inference with a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    parser.add_argument("--img-size", type=int, default=64, help="Square size used during training")
    parser.add_argument("--color", action="store_true", help="Use RGB instead of grayscale")
    parser.add_argument("--mediapipe", action="store_true", help="Use MediaPipe holistic landmarks if available")
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

        X = preprocess_frame(frame, args.img_size, args.color, args.mediapipe)
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
