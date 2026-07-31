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

## Load (needs custom modules)

```python
# On Kaggle / local: put hybrid_modules.py and custom_rare_trainer.py on PYTHONPATH
# (see scripts/eval_epoch20_kaggle.txt for the full working cell)

from ultralytics import YOLO
model = YOLO("models/hybrid_rare_ft_epoch20.pt")
metrics = model.val(data="dataset.yaml", imgsz=768, conf=0.001)
```

Because the checkpoint was trained with `RareClassTrainer`, you must have `custom_rare_trainer.py` importable before `YOLO(...)`.
