"""Privacy helpers: AeroFlow tracks anonymous silhouettes, never identities.

* No face recognition anywhere in the pipeline.
* Raw video is never persisted. Optional debug previews are anonymised first.
* Stored data = zone counts and alerts only; retention is enforced by Store.purge().
"""

from __future__ import annotations

import cv2
import numpy as np


def blur_heads(frame: np.ndarray, person_boxes, head_frac: float = 0.22) -> np.ndarray:
    """Blur the top `head_frac` of every person box (works without a face detector)."""
    out = frame.copy()
    h_img, w_img = out.shape[:2]
    for x1, y1, x2, y2 in person_boxes:
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w_img), min(int(y2), h_img)
        hy = y1 + max(int((y2 - y1) * head_frac), 4)
        roi = out[y1:hy, x1:x2]
        if roi.size:
            k = max((x2 - x1) // 2 * 2 + 1, 3)
            out[y1:hy, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def blur_faces(frame: np.ndarray) -> np.ndarray:
    """Extra safety net: Haar-cascade face blur for close-up cameras."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = frame.copy()
    for x, y, w, h in cascade.detectMultiScale(gray, 1.1, 5):
        k = max(w // 2 * 2 + 1, 3)
        out[y : y + h, x : x + w] = cv2.GaussianBlur(out[y : y + h, x : x + w], (k, k), 0)
    return out
