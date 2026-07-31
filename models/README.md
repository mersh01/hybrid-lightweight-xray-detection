# Final released model

**File:** [`hybrid_rare_ft_epoch20.pt`](hybrid_rare_ft_epoch20.pt)

| Field | Value |
|-------|--------|
| Stage | Rare-class fine-tune (round 1) from FDD-protocol `best.pt` |
| Selected epoch | **20** (early-stopped at 35; best rare + overall peak) |
| Architecture | Hybrid YOLO (`configs/hybrid_yolo.yaml` + `modules/hybrid_modules.py`) |
| Train imgsz | 768 |
| Class weights | Cos=1.5, Lighter=3.0, others=1.0 |
| Oversample | Cos×2, Lighter×5 |
| Val (official re-val) | ALL mAP50 **0.809**, Cosmetic **0.665**, Nonmetallic_Lighter **0.086** |
| Size | ~26 MB |

## Load on PC (recommended)

```bash
# 1) Edit configs/dataset.yaml → set path: to your HiXray_YOLO2 root
# 2) Validate
python scripts/run_val.py

# 3) Predict
python scripts/run_predict.py --source path/to/image.jpg
```

These scripts call `modules/register_hybrid.py` so custom blocks + `RareClassTrainer` load correctly.

## Load manually

```python
from modules.register_hybrid import register
register()

from ultralytics import YOLO
model = YOLO("models/hybrid_rare_ft_epoch20.pt")
metrics = model.val(data="configs/dataset.yaml", imgsz=768, conf=0.001)
```

## Kaggle

Copy [`kaggle/cell_01_eval.py`](../kaggle/cell_01_eval.py) into one notebook cell (see [`kaggle/README.md`](../kaggle/README.md)).
