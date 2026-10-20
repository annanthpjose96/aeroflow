"""Frame sources.

SimSource  - synthetic cameras (default, no hardware)
RtspSource - real RTSP/IP cameras + YOLO detector + colour-histogram appearance embedding

For production-scale multi-stream GPU inference use the DeepStream pipeline in /deepstream and
feed its metadata into the same Engine (see docs/architecture.md). RtspSource is the
straightforward single-process path for development and small deployments.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import Config
from .datatypes import Detection
from .simulation import Simulator

# COCO ids: person=0, backpack=24, handbag=26, suitcase=28
COCO_MAP = {0: "person", 24: "bag", 26: "bag", 28: "bag"}


class SimSource:
    def __init__(self, cfg: Config):
        self.sim = Simulator(cfg)
        self.dt = self.sim.dt

    def read(self) -> tuple[float, list[Detection]]:
        return self.sim.step()

    def close(self):
        pass


def hist_embedding(frame: np.ndarray, box) -> np.ndarray | None:
    """Appearance descriptor: HSV histograms of upper and lower body (clothing colour)."""
    import cv2

    x1, y1, x2, y2 = (max(int(v), 0) for v in box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    half = hsv.shape[0] // 2
    parts = []
    for region in (hsv[:half], hsv[half:]):
        h = cv2.calcHist([region], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        parts.append(h / (h.sum() + 1e-9))
    v = np.concatenate(parts)
    return v / (np.linalg.norm(v) + 1e-9)


class YoloDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # pip install aeroflow[vision]

        self.model = YOLO(cfg["weights"])  # .pt, .onnx or TensorRT .engine all work
        self.conf, self.imgsz, self.half = cfg["conf"], cfg["imgsz"], cfg["half"]
        self.device = None if cfg["device"] == "auto" else cfg["device"]

    def __call__(self, frame):
        res = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            half=self.half,
            device=self.device,
            classes=list(COCO_MAP),
            verbose=False,
        )[0]
        out = []
        for box, c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append((tuple(box.tolist()), COCO_MAP[int(c)], float(p)))
        return out


class RtspSource:
    def __init__(self, cfg: Config):
        import cv2

        self.cfg = cfg
        self.detector = YoloDetector(cfg["detector"])
        self.caps = {}
        for c in cfg.cameras:
            if not c.source:
                raise ValueError(f"camera {c.id} has no `source` URL")
            self.caps[c.id] = cv2.VideoCapture(c.source)
        self.dt = 0.0
        self.preview_dir = cfg["privacy"].get("save_preview_dir")
        self._n = 0

    def read(self) -> tuple[float, list[Detection]]:
        dets: list[Detection] = []
        ts = time.time()
        for cid, cap in self.caps.items():
            ok, frame = cap.read()
            if not ok:
                continue
            found = self.detector(frame)
            for box, cls, conf in found:
                dets.append(Detection(cid, ts, box, cls, conf, hist_embedding(frame, box)))
            if self.preview_dir and self._n % 100 == 0:
                import cv2

                from .privacy import blur_heads

                Path(self.preview_dir).mkdir(parents=True, exist_ok=True)
                safe = blur_heads(frame, [b for b, c, _ in found if c == "person"])
                cv2.imwrite(f"{self.preview_dir}/{cid}_{self._n}.jpg", safe)
        self._n += 1
        return ts, dets

    def close(self):
        for cap in self.caps.values():
            cap.release()
