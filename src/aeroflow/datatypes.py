"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One object detected in one camera frame."""

    camera_id: str
    ts: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    cls: str  # "person" | "bag"
    conf: float = 1.0
    embedding: np.ndarray | None = None
    gt_id: int | None = None  # ground truth id, simulator/evaluation only


@dataclass
class Observation:
    """A confirmed local (single-camera) track projected onto the floor plan."""

    camera_id: str
    local_id: int
    cls: str
    pos: np.ndarray  # floor coordinates in metres
    embedding: np.ndarray | None
    aspect: float  # bbox width / height (fall cue)
    ts: float
    gt_id: int | None = None


@dataclass
class GlobalTrack:
    """A cross-camera identity living on the floor plan."""

    gid: int
    cls: str
    pos: np.ndarray
    embedding: np.ndarray | None
    first_seen: float
    last_seen: float
    fallen: bool = False
    hits: int = 1
    cameras: set = field(default_factory=set)
