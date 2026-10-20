import numpy as np
import pytest

from aeroflow.geometry import Homography, cosine, point_in_polygon, polygon_area


def test_homography_roundtrip():
    h = Homography([[0, 0], [1280, 0], [1280, 720], [0, 720]], [[0, 0], [24, 0], [24, 24], [0, 24]])
    f = h.to_floor(640, 360)
    assert np.allclose(f, [12, 12], atol=1e-6)
    assert np.allclose(h.to_image(*f), [640, 360], atol=1e-6)


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        Homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


def test_polygon_helpers():
    sq = [[0, 0], [4, 0], [4, 3], [0, 3]]
    assert polygon_area(sq) == 12
    assert point_in_polygon((2, 1), sq)
    assert not point_in_polygon((5, 1), sq)


def test_cosine():
    assert cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert cosine(None, np.array([1.0])) is None
