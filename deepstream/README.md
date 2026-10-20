# DeepStream integration (GPU path)

> **Status: templates, not tested in this repo's CI.** They require an NVIDIA GPU, DeepStream 7.x
> and a TensorRT engine. Check keys against the DeepStream docs for your installed version.

The Python `RtspSource` is fine for development and a handful of cameras. For many high-resolution
streams use DeepStream so decode, batching, inference and tracking stay on the GPU:

```
rtsp cams -> nvurisrcbin x N -> nvstreammux (batch) -> nvinfer (YOLO, TensorRT FP16/INT8)
          -> nvtracker -> probe (pyds) -> AeroFlow Engine -> analytics / API / event bus
```

1. Export the detector to ONNX, then build an engine (FP16 or INT8) - see `benchmarks/`.
2. Use a YOLO output parser compatible with your model (e.g. the community
   [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) library) and point
   `custom-lib-path` / `parse-bbox-func-name` in `config_infer_primary.txt` at it.
3. Run `deepstream-app -c deepstream_app_multicam.txt` to validate decode + inference first.
4. In a pad probe on the tracker's src pad, read `NvDsObjectMeta` (bbox, class, tracker id) and
   build `aeroflow.datatypes.Detection` objects per `source_id`, then pass them to
   `Engine.process(ts, detections)` (a one-line refactor of `Engine.step`, which currently
   pulls detections from a `source`). The rest of the pipeline is unchanged.
5. For multi-camera 3D tracking NVIDIA also offers the Multi-View 3D Tracking (MV3DT) reference;
   AeroFlow's floor-plane fusion is a lightweight alternative that works from 2D detections.

`nvmsgbroker` can publish straight to Kafka/MQTT/AMQP if you prefer to bypass Python for metadata.
