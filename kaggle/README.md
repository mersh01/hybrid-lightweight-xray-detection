# Kaggle copy-paste cells

Open a new Kaggle Notebook (GPU T4 recommended). Add Inputs:

1. **HiXray_YOLO2** dataset (`images/` + `labels/` with `train` and `val`)
2. This GitHub repo **or** upload `models/hybrid_rare_ft_epoch20.pt`

Then:

| File | Action |
|------|--------|
| [`cell_01_eval.py`](cell_01_eval.py) | Copy **entire file** into one notebook cell → Run (official val metrics) |
| [`cell_02_predict.py`](cell_02_predict.py) | Copy into next cell → Run (saves prediction overlays) |
| [`cell_03_fair_res_eval.py`](cell_03_fair_res_eval.py) | **Fair matrix:** long vs epoch20 at imgsz **640 and 768** (required for FT attribution) |
| [`cell_04_complexity.py`](cell_04_complexity.py) | Params + **GFLOPs @640 and @768** for both checkpoints |

## Full training on Kaggle

For Stage A / Stage B training (long runs), copy cells from:

- `scripts/train_fdd_protocol_kaggle.py` — Stage A
- `scripts/resume_fdd_protocol_kaggle.py` — resume after 12h timeout
- `scripts/train_rare_finetune_kaggle.py` — Stage B (produced `epoch20.pt`)

Each `CELL_*` string in those files is one notebook cell — paste in order.

See [`docs/TRAINING_RECIPE.md`](../docs/TRAINING_RECIPE.md).
