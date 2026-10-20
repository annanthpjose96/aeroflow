"""Cross-camera identity fusion (multi-camera Re-ID on the floor plane).

Local tracks from every camera are projected to floor coordinates and associated to global
identities using floor distance (gated by how long the identity has been unseen) plus
appearance-embedding similarity. A global identity can be observed by several cameras at
once but by at most one local track per camera.
"""

from __future__ import annotations

import numpy as np

from .datatypes import GlobalTrack, Observation
from .geometry import cosine


class GlobalFusion:
    def __init__(
        self, max_dist_m=1.5, appearance_weight=0.6, max_cost=1.6, ttl_s=8.0, active_s=2.0, walk_speed_mps=1.0
    ):
        self.max_dist, self.w, self.max_cost = max_dist_m, appearance_weight, max_cost
        self.ttl, self.active_s, self.speed = ttl_s, active_s, walk_speed_mps
        self.tracks: dict[int, GlobalTrack] = {}
        self.map: dict[tuple[str, int], int] = {}
        self._next = 1

    def _cost(self, g: GlobalTrack, o: Observation, ts: float) -> float | None:
        gap = max(0.0, min(ts - g.last_seen, self.ttl))
        allowed = self.max_dist + gap * self.speed
        d = float(np.linalg.norm(g.pos - o.pos))
        if d > allowed:
            return None
        sim = cosine(g.embedding, o.embedding)
        app = 0.5 if sim is None else (1.0 - sim)
        cost = d / allowed + self.w * app * 2
        return cost if cost <= self.max_cost else None

    def forget(self, camera_id: str, local_ids: list[int]) -> None:
        for lid in local_ids:
            self.map.pop((camera_id, lid), None)

    def update(self, ts: float, obs: list[Observation]) -> list[tuple[Observation, int]]:
        assigned: dict[int, int] = {}
        unmapped = []
        for i, o in enumerate(obs):
            gid = self.map.get((o.camera_id, o.local_id))
            if gid is not None and gid in self.tracks:
                assigned[i] = gid
            else:
                unmapped.append(i)
        taken = {(g, obs[i].camera_id) for i, g in assigned.items()}

        # stage 1: match against identities that existed before this frame
        pairs = []
        for i in unmapped:
            for g in self.tracks.values():
                if g.cls == obs[i].cls:
                    c = self._cost(g, obs[i], ts)
                    if c is not None:
                        pairs.append((c, i, g.gid))
        pairs.sort()
        done = set()
        for _, i, gid in pairs:
            if i in done or (gid, obs[i].camera_id) in taken:
                continue
            assigned[i] = gid
            taken.add((gid, obs[i].camera_id))
            done.add(i)

        # stage 2: match to identities created this frame by other cameras, else create
        created: list[int] = []
        for i in unmapped:
            if i in done:
                continue
            o = obs[i]
            best = None
            for gid in created:
                g = self.tracks[gid]
                if (gid, o.camera_id) in taken or g.cls != o.cls:
                    continue
                c = self._cost(g, o, ts)
                if c is not None and (best is None or c < best[0]):
                    best = (c, gid)
            if best:
                gid = best[1]
            else:
                gid = self._next
                self._next += 1
                self.tracks[gid] = GlobalTrack(gid, o.cls, o.pos.copy(), o.embedding, ts, ts)
                created.append(gid)
            taken.add((gid, o.camera_id))
            assigned[i] = gid

        # update identities from all their observations this frame
        groups: dict[int, list[Observation]] = {}
        for i, gid in assigned.items():
            groups.setdefault(gid, []).append(obs[i])
            self.map[(obs[i].camera_id, obs[i].local_id)] = gid
        for gid, os_ in groups.items():
            g = self.tracks[gid]
            g.pos = np.mean([o.pos for o in os_], axis=0)
            embs = [o.embedding for o in os_ if o.embedding is not None]
            if embs:
                e = np.mean(embs, axis=0)
                g.embedding = e if g.embedding is None else 0.9 * g.embedding + 0.1 * e
            g.last_seen, g.hits = ts, g.hits + 1
            g.fallen = any(o.aspect > 1.1 and o.cls == "person" for o in os_)
            g.cameras = {o.camera_id for o in os_}

        for gid in [k for k, g in self.tracks.items() if ts - g.last_seen > self.ttl]:
            del self.tracks[gid]
        self.map = {k: v for k, v in self.map.items() if v in self.tracks}
        return [(obs[i], gid) for i, gid in assigned.items()]

    def active(self, ts: float) -> list[GlobalTrack]:
        return [g for g in self.tracks.values() if ts - g.last_seen <= self.active_s and g.hits >= 2]
