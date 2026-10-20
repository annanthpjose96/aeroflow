# Evaluation (simulator)

Reproduce: `aeroflow eval --seconds 900` (override the seed with a custom config or
`load_config(..., {"sim": {"seed": N}})`).

The simulator knows ground truth, so we can measure how well the tracker + cross-camera fusion keep
identities. 900 simulated seconds, 2 cameras with a 8 m overlap, 12 arrivals/min, 3 scripted incidents:

| Seed | Real objects | Global IDs | ID consistency¹ | ID purity² | Scripted incidents detected | False incident alerts |
|---|---|---|---|---|---|---|
| 7  | 224 | 234 | 0.941 | 0.954 | 3 / 3 | 0 |
| 21 | 220 | 223 | 0.951 | 0.952 | 3 / 3 | 0 |
| 99 | 237 | 254 | 0.877 | 0.896 | 3 / 3 | 0 |
| 5  | 251 | 267 | 0.898 | 0.909 | 3 / 3 | 0 |

¹ fraction of an object's observations carrying its most common global ID (higher = fewer identity splits).
² fraction of a global ID's observations belonging to its most common real object (higher = fewer merges).

Incidents are an abandoned bag (t=240 s), an intruder in the staff-only zone (t=300 s) and a person
falling at the gate (t=420 s). Each is detected exactly once per seed; alert latency is roughly one
minute after the bag is dropped (the owner must walk away, then the 30 s rule runs), ~25 s for the
intruder (mostly walking time) and ~5 s for the fall (sustained-lying rule).

Throughput of the tracking + fusion + analytics core on a shared cloud CPU: roughly 80-230 pipeline
steps/s for 2 cameras and ~50 objects (the detector is **not** included).

## RAG retrieval quality

Reproduce: `pytest tests/test_rag.py` (questions and expected documents are in that file). Metric: does the
expected SOP document appear first (top-1) or in the top 3 retrieved sections?

| Question set | Questions | Top-1 | Top-3 |
|---|---|---|---|
| Tuned on (used to build the synonym map) | 15 | 14 | 15 |
| Held-out (written after tuning) | 8 | 8 | 8 |

Caveats: tiny corpus (6 documents, 17 sections), questions written by the author who knew the corpus, and a
hand-built synonym map. Expect lower scores on real operator phrasing and larger corpora. Known miss:
"passenger fell on the floor and is not moving" ranks the bag SOP first because "not"/"move" appear in it.
Alert-to-SOP mapping (alert type metadata) is deterministic and tested for every alert type.

## Honest limits

* **This is a simulator**, with synthetic appearance vectors and a 2 Hz frame rate that is harsher on
  motion than real video but kinder on appearance than real clothing. Numbers say the *logic* works;
  they are **not** real-world accuracy.
* Real-world validation still to do: run on WILDTRACK / MMPTRACK (multi-camera with calibration) and
  report IDF1/MOTA; evaluate YOLO mAP and FP16/INT8 drop with `benchmarks/benchmark_inference.py`.

## Real-hardware benchmark table (fill in)

| Backend | Latency mean (ms) | p95 (ms) | FPS | mAP50-95 |
|---|---|---|---|---|
| PyTorch FP32 | | | | |
| PyTorch FP16 | | | | |
| ONNX Runtime | | | | |
| TensorRT FP16 | | | | |
| TensorRT INT8 | | | | |

*Hardware: ________ · Driver/CUDA/TensorRT: ________ · Input: 640×640*
