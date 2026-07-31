Hybrid YOLO model complexity (measured with Ultralytics `model.info`)

Script: `python scripts/run_complexity.py`
Both checkpoints share the same architecture; GFLOPs depend on imgsz only.

--------------------------------------------------------------------------------
MEASURED MATRIX
--------------------------------------------------------------------------------
  Checkpoint                    imgsz     Params      GFLOPs
  ----------------------------  -----  -----------  --------
  hybrid_fdd_long_best.pt         640    3,371,158      8.88
  hybrid_fdd_long_best.pt         768    3,371,158     12.78
  hybrid_rare_ft_epoch20.pt       640    3,371,158      8.88
  hybrid_rare_ft_epoch20.pt       768    3,371,158     12.78

  Layers: 258 (both)
  Scale: GFLOPs_768 / GFLOPs_640 = 1.440 = (768/640)^2

Thesis reporting tip:
  - When comparing to long-stage / YOLOv11n at 640 → report **8.9 GFLOPs**
  - When reporting rare-FT epoch20 operating point (train/val imgsz=768) → also
    report **12.8 GFLOPs @768** so compute matches the evaluation resolution
  - Params stay **3.37M** at both sizes (weights unchanged by imgsz)

--------------------------------------------------------------------------------
PRIOR LOG NOTES (consistent with @640)
--------------------------------------------------------------------------------
Training-time summary (imgsz=640 protocol):
  hybrid_YOLO summary: 258 layers, 3,371,158 parameters, 3,371,142 gradients, 8.9 GFLOPs

Some fused val loads: ~207 layers / 3,366,342 params / 8.8 GFLOPs — prefer the
unfused training summary / script numbers above.

Inference speed (epoch20.pt val on Tesla T4, imgsz=768):
  Speed: 0.8ms preprocess, 4.5ms inference, 0.0ms loss, 0.8ms postprocess per image

Hardware:
  Kaggle Tesla T4 x2 (train) / single T4 (val)
  Ultralytics 8.4.103; local measure used ultralytics 8.4.x
