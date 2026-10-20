"""Prometheus metrics."""

from prometheus_client import Counter, Gauge, Histogram

ZONE_COUNT = Gauge("aeroflow_zone_count", "People currently in zone", ["zone"])
ZONE_WAIT = Gauge("aeroflow_zone_wait_seconds", "Estimated queue wait", ["zone"])
ZONE_PRED_WAIT = Gauge("aeroflow_zone_forecast_wait_seconds", "Forecast queue wait", ["zone"])
ALERTS = Counter("aeroflow_alerts_total", "Alerts raised", ["type"])
TRACKS = Gauge("aeroflow_active_tracks", "Active global tracks", ["cls"])
STEP_LATENCY = Histogram(
    "aeroflow_step_latency_seconds",
    "Pipeline step latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
FPS = Gauge("aeroflow_pipeline_fps", "Pipeline steps per second")
