from __future__ import annotations

import argparse
import os
import time

import cv2

from realtime.overlay import draw_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture images from webcam")
    parser.add_argument("--output-dir", default="captures", help="Directory to save snapshots")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the camera device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Press 's' to save a snapshot, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_label(frame, "Press 's' to save, 'q' to quit")
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            ts = int(time.time() * 1000)
            path = os.path.join(args.output_dir, f"snapshot_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
