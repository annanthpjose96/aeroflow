"""Lightweight per-camera multi-object tracker (greedy nearest-foot association)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datatypes import Detection
from .geometry import cosine


@dataclass
class LocalTrack:
    local_id: int
    cls: str
    bbox: tuple
    embedding: np.ndarray | None
    hits: int = 1
    missed: int = 0
    gt_id: int | None = None
    vel: tuple = (0.0, 0.0)  # foot-point velocity in px/frame (EMA)

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    @property
    def aspect(self) -> float:
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        return w / max(h, 1e-6)


class CameraTracker:
    def __init__(
        self,
        camera_id: str,
        max_missed: int = 6,
        min_hits: int = 2,
        match_ratio: float = 1.0,
        min_scale: float = 30.0,
        app_weight: float = 1.0,
    ):
        self.camera_id = camera_id
        self.max_missed, self.min_hits, self.match_ratio = max_missed, min_hits, match_ratio
        self.min_scale, self.app_weight = min_scale, app_weight
        self.tracks: dict[int, LocalTrack] = {}
        self._next = 1

    def update(self, dets: list[Detection]) -> tuple[list[LocalTrack], list[int]]:
        """Returns (confirmed tracks seen this frame, ids of tracks that just died)."""
        pairs = []
        for tid, tr in self.tracks.items():
            fx, fy = tr.foot
            k = tr.missed + 1  # frames since last seen
            tx, ty = fx + tr.vel[0] * k, fy + tr.vel[1] * k  # constant-velocity prediction
            scale = max(tr.bbox[3] - tr.bbox[1], tr.bbox[2] - tr.bbox[0], self.min_scale)
            gate = self.match_ratio * (1.0 + 0.5 * tr.missed)  # widen gate after misses
            for j, d in enumerate(dets):
                if d.cls != tr.cls:
                    continue
                dx, dy = (d.bbox[0] + d.bbox[2]) / 2 - tx, d.bbox[3] - ty
                cost = float(np.hypot(dx, dy)) / scale
                if cost < gate:
                    sim = cosine(tr.embedding, d.embedding)
                    app = 0.0 if sim is None else (1.0 - sim)
                    pairs.append((cost + self.app_weight * app, tid, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            tr, d = self.tracks[tid], dets[j]
            ox, oy = tr.foot
            k = tr.missed + 1
            nx, ny = (d.bbox[0] + d.bbox[2]) / 2, d.bbox[3]
            tr.vel = (0.5 * tr.vel[0] + 0.5 * (nx - ox) / k, 0.5 * tr.vel[1] + 0.5 * (ny - oy) / k)
            tr.bbox, tr.hits, tr.missed, tr.gt_id = d.bbox, tr.hits + 1, 0, d.gt_id
            if d.embedding is not None:
                tr.embedding = d.embedding if tr.embedding is None else 0.8 * tr.embedding + 0.2 * d.embedding
        dead = []
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
                    dead.append(tid)
        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks[self._next] = LocalTrack(self._next, d.cls, d.bbox, d.embedding, gt_id=d.gt_id)
                self._next += 1
        seen = [t for t in self.tracks.values() if t.missed == 0 and t.hits >= self.min_hits]
        return seen, dead
