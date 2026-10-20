"""Benchmark a detector across PyTorch / ONNX Runtime / TensorRT (FP16, optional INT8).

Needs the `vision` extra and, for TensorRT, an NVIDIA GPU with TensorRT installed:
    pip install -e ".[vision]"
    python benchmarks/benchmark_inference.py --weights yolo11n.pt --runs 200 --imgsz 640

Reports latency (mean / p95) and throughput. Accuracy (mAP) is NOT measured here: run
`yolo val model=<exported file> data=coco8.yaml` for each exported artefact and record the
drop next to the speed numbers in docs/benchmarks.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def bench(model, img, runs, warmup=20, **kw):
    for _ in range(warmup):
        model.predict(img, verbose=False, **kw)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(img, verbose=False, **kw)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"mean_ms": round(statistics.mean(times), 2), "p95_ms": round(times[int(0.95 * len(times)) - 1], 2),
            "fps": round(1000 / statistics.mean(times), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--int8", action="store_true", help="also build an INT8 TensorRT engine (needs calibration data)")
    a = ap.parse_args()

    from ultralytics import YOLO

    try:
        import torch
        cuda = torch.cuda.is_available()
        device = 0 if cuda else "cpu"
    except ImportError:
        cuda, device = False, "cpu"
    img = np.random.randint(0, 255, (a.imgsz, a.imgsz, 3), dtype=np.uint8)
    results = {"device": "cuda" if cuda else "cpu", "imgsz": a.imgsz}

    pt = YOLO(a.weights)
    results["pytorch_fp32"] = bench(pt, img, a.runs, device=device, imgsz=a.imgsz)
    if cuda:
        results["pytorch_fp16"] = bench(pt, img, a.runs, device=device, imgsz=a.imgsz, half=True)

    onnx_path = pt.export(format="onnx", imgsz=a.imgsz)
    results["onnxruntime"] = bench(YOLO(onnx_path), img, a.runs, device=device, imgsz=a.imgsz)

    if cuda:
        eng16 = pt.export(format="engine", half=True, imgsz=a.imgsz)
        results["tensorrt_fp16"] = bench(YOLO(eng16), img, a.runs, device=device, imgsz=a.imgsz)
        if a.int8:
            eng8 = pt.export(format="engine", int8=True, data="coco8.yaml", imgsz=a.imgsz)
            results["tensorrt_int8"] = bench(YOLO(eng8), img, a.runs, device=device, imgsz=a.imgsz)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
