"""Synthetic multi-camera airport simulator.

Generates passengers who walk check-in -> security queue -> gate -> exit, with bags,
and scripted incidents (abandoned bag, intruder, fall). Ground-truth floor positions are
projected into each camera through the configured homographies, with pixel noise, missed
detections and noisy appearance embeddings. This lets the whole pipeline (tracking, Re-ID,
analytics, API) run and be tested with no GPU, video or dataset.
"""

from __future__ import annotations

import time

import numpy as np

from .config import Config
from .datatypes import Detection
from .geometry import Homography

EMB_DIM = 16


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


class _Agent:
    def __init__(self, aid, pos, app, speed):
        self.id, self.pos, self.app, self.speed = aid, np.array(pos, float), app, speed
        self.state = "to_check_in"
        self.target = None
        self.timer = 0.0
        self.bag = None
        self.fallen_until = 0.0
        self.drop_bag = False


class _Bag:
    def __init__(self, bid, pos, app, owner):
        self.id, self.pos, self.app, self.owner = bid, np.array(pos, float), app, owner


class Simulator:
    def __init__(self, cfg: Config):
        s = cfg["sim"]
        self.s = s
        self.rng = np.random.default_rng(s["seed"])
        self.dt = float(s["dt"])
        self.t0 = self.t = float(s.get("start_ts") or time.time())
        self.zones = {z.id: z for z in cfg.zones}
        self.cams = {c.id: (c, Homography(c.image_points, c.floor_points)) for c in cfg.cameras}
        self.cam_bias = {cid: self.rng.normal(0, 0.05, EMB_DIM) for cid in self.cams}
        self.agents: list[_Agent] = []
        self.bags: list[_Bag] = []
        self.queue: list[_Agent] = []
        self.next_id = 1
        self.next_service = self.t
        self.done: set[str] = set()

    # ---- helpers
    def _nid(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def _rand_in(self, zone_id: str) -> np.ndarray:
        xs = [p[0] for p in self.zones[self.s[zone_id]].polygon]
        ys = [p[1] for p in self.zones[self.s[zone_id]].polygon]
        return np.array(
            [self.rng.uniform(min(xs) + 0.5, max(xs) - 0.5), self.rng.uniform(min(ys) + 0.5, max(ys) - 0.5)]
        )

    def _slot(self, i: int) -> np.ndarray:
        poly = self.zones[self.s["queue_zone"]].polygon
        x1, x2 = min(p[0] for p in poly), max(p[0] for p in poly)
        y1 = min(p[1] for p in poly)
        per_row = max(1, int((x2 - x1) // 0.8))
        row, col = divmod(i, per_row)
        return np.array([x2 - 0.5 - 0.8 * col, y1 + 0.6 + 0.8 * row])

    def _spawn(self, drop_bag=False, intruder=False):
        ent = np.array(self.s["entrance"]) + self.rng.normal(0, 0.4, 2)
        a = _Agent(self._nid(), ent, _unit(self.rng.normal(size=EMB_DIM)), self.rng.uniform(1.0, 1.4))
        if intruder:
            a.state, a.target = "to_restricted", self._rand_in("restricted_zone")
        else:
            a.target = self._rand_in("check_in_zone")
        if drop_bag or self.rng.random() < self.s["bag_probability"]:
            b = _Bag(self._nid(), a.pos, _unit(self.rng.normal(size=EMB_DIM)), a)
            a.bag, a.drop_bag = b, drop_bag
            self.bags.append(b)
        self.agents.append(a)

    def _move(self, a: _Agent, target) -> bool:
        d = np.asarray(target) - a.pos
        dist = float(np.linalg.norm(d))
        step = a.speed * self.dt
        if dist <= max(step, 0.2):
            a.pos = np.asarray(target, float).copy()
            return True
        a.pos = a.pos + d / dist * step
        return False

    # ---- dynamics
    def _advance(self):
        t, el, s = self.t, self.t - self.t0, self.s
        for _ in range(self.rng.poisson(s["arrival_rate_per_min"] / 60 * self.dt)):
            self._spawn()
        if "bag" not in self.done and el >= s["abandoned_bag_at_s"]:
            self._spawn(drop_bag=True)
            self.done.add("bag")
        if "intruder" not in self.done and el >= s["intruder_at_s"]:
            self._spawn(intruder=True)
            self.done.add("intruder")
        if "fall" not in self.done and el >= s["fall_at_s"]:
            for a in self.agents:
                if a.state == "gate":
                    a.fallen_until = t + 40
                    self.done.add("fall")
                    break
        if self.queue and self.queue[0].state == "queued" and t >= self.next_service:
            head = self.queue.pop(0)
            head.state, head.target = "to_gate", self._rand_in("gate_zone")
            self.next_service = t + s["service_time_s"]

        for a in list(self.agents):
            if a.fallen_until > t:
                continue
            if a.state == "to_check_in":
                if self._move(a, a.target):
                    a.state, a.timer = "check_in", t + self.rng.uniform(15, 40)
                    if a.drop_bag and a.bag:
                        a.bag.owner, a.bag = None, None
            elif a.state == "check_in" and t >= a.timer:
                a.state = "to_queue"
                self.queue.append(a)
            elif a.state in ("to_queue", "queued"):
                if self._move(a, self._slot(self.queue.index(a))) and a.state == "to_queue":
                    a.state = "queued"
            elif a.state == "to_gate":
                if self._move(a, a.target):
                    a.state, a.timer = "gate", t + self.rng.uniform(60, 120)
            elif a.state == "gate" and t >= a.timer:
                a.state, a.target = "to_exit", np.array(s["exit"], float)
            elif a.state == "to_restricted":
                if self._move(a, a.target):
                    a.state, a.timer = "intruding", t + 25
            elif a.state == "intruding" and t >= a.timer:
                a.state, a.target = "to_exit", np.array(s["exit"], float)
            elif a.state == "to_exit" and self._move(a, a.target):
                a.state = "gone"
        self.agents = [a for a in self.agents if a.state != "gone"]
        self.bags = [b for b in self.bags if b.owner is None or b.owner.state != "gone"]
        for b in self.bags:
            if b.owner is not None:
                b.pos = b.owner.pos + np.array([0.3, 0.2])

    # ---- rendering into cameras
    def step(self) -> tuple[float, list[Detection]]:
        self.t += self.dt
        self._advance()
        objs = [(a.id, "person", a.pos, a.app, a.fallen_until > self.t) for a in self.agents]
        objs += [(b.id, "bag", b.pos, b.app, False) for b in self.bags]
        dets: list[Detection] = []
        noise = self.s["pixel_noise"]
        for cid, (cam, hom) in self.cams.items():
            w, h = cam.resolution
            for gid, cls, pos, app, fallen in objs:
                u, v = hom.to_image(*pos)
                if not (0 <= u < w and 0 <= v < h) or self.rng.random() < self.s["miss_prob"]:
                    continue
                u += self.rng.normal(0, noise)
                v += self.rng.normal(0, noise)
                scale = float(np.linalg.norm(hom.to_image(pos[0], pos[1] + 1) - hom.to_image(*pos)))
                bw, bh = (0.5, 1.7) if cls == "person" else (0.4, 0.4)
                bw, bh = bw * scale, bh * scale
                if fallen:
                    bw, bh = bh, bw
                emb = _unit(app + self.cam_bias[cid] + self.rng.normal(0, 0.1, EMB_DIM))
                dets.append(Detection(cid, self.t, (u - bw / 2, v - bh, u + bw / 2, v), cls, 0.9, emb, gt_id=gid))
        return self.t, dets
