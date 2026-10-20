"""Homography and polygon helpers."""

from __future__ import annotations

import cv2
import numpy as np


class Homography:
    """Maps image pixels <-> floor-plan metres using >=4 point correspondences."""

    def __init__(self, image_points, floor_points):
        img = np.asarray(image_points, dtype=np.float64)
        flo = np.asarray(floor_points, dtype=np.float64)
        if len(img) < 4 or len(img) != len(flo):
            raise ValueError("need >=4 matching image/floor point pairs")
        h, _ = cv2.findHomography(img, flo, 0)
        if h is None:
            raise ValueError("could not estimate homography (degenerate points?)")
        self.h = h
        self.h_inv = np.linalg.inv(h)

    @staticmethod
    def _apply(m: np.ndarray, x: float, y: float) -> np.ndarray:
        p = m @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    def to_floor(self, x: float, y: float) -> np.ndarray:
        return self._apply(self.h, x, y)

    def to_image(self, x: float, y: float) -> np.ndarray:
        return self._apply(self.h_inv, x, y)


def point_in_polygon(pt, polygon) -> bool:
    """Ray casting."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def polygon_area(polygon) -> float:
    a = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))
