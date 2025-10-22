from __future__ import annotations

from typing import Tuple

import cv2


def draw_label(
    frame,
    text: str,
    org: Tuple[int, int] = (10, 30),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a text label with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - h - 4), (x + w + 4, y + baseline + 4), bg_color, thickness=-1)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    draw_label(frame, f"FPS: {fps:.1f}", org=(10, 60))
