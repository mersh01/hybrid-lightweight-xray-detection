Hybrid YOLO model complexity evidence (from Kaggle Ultralytics train logs)

Architecture summary printed at train start (hybrid_yolo.yaml, scale n):
  hybrid_YOLO summary: 258 layers, 3,371,158 parameters, 3,371,142 gradients, 8.9 GFLOPs
  (Some val loads report 207 layers / 3,366,342 params / 8.8 GFLOPs after fuse/strip —
   use training-time summary 3.37M params, 8.9 GFLOPs as primary.)

Transferred weights: 520/520 items from pretrained start checkpoint when fine-tuning.

Checkpoint file sizes (approx, from local copies):
  epoch20.pt / epoch*.pt (full train ckpt): ~13.3 MB
  last.pt after optimizer strip: ~7.1 MB

Inference speed (epoch20.pt val on Tesla T4, imgsz=768):
  Speed: 0.8ms preprocess, 4.5ms inference, 0.0ms loss, 0.8ms postprocess per image

Hardware used:
  Kaggle Tesla T4 x2 (training DDP) / single T4 for val
  CUDA enabled; Ultralytics 8.4.103; torch 2.10.0+cu128 (example session)

Baseline comparison note:
  YOLOv11n baseline overall mAP50 ≈ 0.803 (prior experiment record in this project)
