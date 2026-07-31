# Released checkpoints

## Selected fine-tuned model (primary)

**File:** [`hybrid_rare_ft_epoch20.pt`](hybrid_rare_ft_epoch20.pt)

| Field | Value |
|-------|--------|
| Stage | Rare-class fine-tune (round 1) from FDD-protocol `best.pt` |
| Selected epoch | **20** (early-stopped at 35; best rare + overall peak) |
| Train imgsz | 768 |
| Class weights | Cos=1.5, Lighter=3.0, others=1.0 |
| Oversample | Cos×2, Lighter×5 |
| Val @768 (re-val) | ALL mAP50 **0.809**, Cosmetic **0.665**, Lighter **0.086** |
| Size | ~26 MB |

## Long-stage baseline (for fair comparison)

**File:** [`hybrid_fdd_long_best.pt`](hybrid_fdd_long_best.pt)

| Field | Value |
|-------|--------|
| Stage | FDD-protocol 300-ep SGD (`results166` best ≈ ep 235) |
| Train imgsz | 640 |
| Val @640 (re-val) | ALL mAP50 **0.800**, Lighter **0.017** |
| Size | ~13 MB |

## Fair resolution rule

Before attributing gains to fine-tuning alone, validate **both** checkpoints at **640 and 768**:

```bash
# Edit configs/dataset.yaml path: first
python scripts/run_fair_res_eval.py
```

Or on Kaggle: paste [`kaggle/cell_03_fair_res_eval.py`](../kaggle/cell_03_fair_res_eval.py).

## Load on PC

```bash
python scripts/run_val.py --weights models/hybrid_rare_ft_epoch20.pt --imgsz 768
python scripts/run_val.py --weights models/hybrid_fdd_long_best.pt --imgsz 640
python scripts/run_predict.py --source path/to/image.jpg
```

```python
from modules.register_hybrid import register
register()
from ultralytics import YOLO
model = YOLO("models/hybrid_rare_ft_epoch20.pt")
```
