"""YAML configuration loading with defaults."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "floor": {"width": 40.0, "height": 24.0},
    "tracking": {"max_missed": 6, "min_hits": 2, "match_ratio": 1.0, "min_scale": 50.0, "app_weight": 1.0},
    "fusion": {
        "max_dist_m": 1.5,
        "appearance_weight": 0.6,
        "max_cost": 1.6,
        "ttl_s": 8.0,
        "active_s": 2.0,
        "walk_speed_mps": 1.0,
    },
    "analytics": {
        "exit_grace_s": 2.0,
        "snapshot_interval_s": 5.0,
        "unattended_after_s": 30.0,
        "owner_radius_m": 3.0,
        "carry_radius_m": 0.8,
        "carry_min_s": 6.0,
        "bag_dropout_s": 30.0,
        "soft_alert_cooldown_s": 120.0,
        "stationary_radius_m": 0.8,
        "fall_after_s": 5.0,
        "fall_aspect": 1.1,
        "heatmap_cell_m": 1.0,
        "forecast_horizon_s": 600.0,
        "forecast_sample_s": 5.0,
        "throughput_window_s": 300.0,
        "min_exits_for_wait": 4,
    },
    "sim": {
        "seed": 7,
        "dt": 0.5,
        "speed": 4.0,
        "arrival_rate_per_min": 12.0,
        "service_time_s": 6.0,
        "bag_probability": 0.3,
        "abandoned_bag_at_s": 240.0,
        "fall_at_s": 420.0,
        "intruder_at_s": 300.0,
        "entrance": [1.0, 12.0],
        "exit": [39.0, 12.0],
        "check_in_zone": "check_in",
        "queue_zone": "security_queue",
        "gate_zone": "gate_5",
        "restricted_zone": "restricted",
        "pixel_noise": 1.5,
        "miss_prob": 0.03,
    },
    "storage": {"db_path": "aeroflow.db", "retention_days": 7},
    "events": {
        "backend": "jsonl",
        "jsonl_path": "events.jsonl",
        "kafka_bootstrap": "localhost:9092",
        "mqtt_host": "localhost",
        "mqtt_port": 1883,
        "topic_prefix": "aeroflow",
    },
    "detector": {"weights": "yolo11n.pt", "device": "auto", "half": True, "conf": 0.35, "imgsz": 640},
    "privacy": {"blur_heads": True, "save_preview_dir": None},
    "rag": {"enabled": True, "sop_dir": "sops", "backend": "bm25", "top_k": 3},
}


@dataclass
class CameraCfg:
    id: str
    resolution: tuple[int, int]
    image_points: list
    floor_points: list
    source: str | None = None


@dataclass
class ZoneCfg:
    id: str
    name: str
    type: str  # area | queue | restricted
    polygon: list
    capacity: int | None = None
    max_wait_s: float | None = None
    max_dwell_s: float | None = None
    max_density: float | None = None


@dataclass
class Config:
    raw: dict
    cameras: list[CameraCfg] = field(default_factory=list)
    zones: list[ZoneCfg] = field(default_factory=list)

    def __getitem__(self, key):
        return self.raw[key]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    data: dict = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = _merge(DEFAULTS, data)
    raw = _merge(raw, overrides or {})
    raw["_config_path"] = str(path) if path else None
    cams = [
        CameraCfg(
            id=c["id"],
            resolution=tuple(c.get("resolution", (1280, 720))),
            image_points=c["image_points"],
            floor_points=c["floor_points"],
            source=c.get("source"),
        )
        for c in raw.get("cameras", [])
    ]
    zones = [
        ZoneCfg(
            id=z["id"],
            name=z.get("name", z["id"]),
            type=z.get("type", "area"),
            polygon=z["polygon"],
            capacity=z.get("capacity"),
            max_wait_s=z.get("max_wait_s"),
            max_dwell_s=z.get("max_dwell_s"),
            max_density=z.get("max_density"),
        )
        for z in raw.get("zones", [])
    ]
    if not cams:
        raise ValueError("config must define at least one camera")
    return Config(raw=raw, cameras=cams, zones=zones)
