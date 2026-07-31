# Hybrid Lightweight X-Ray Prohibited-Item Detection (HiXray)

MSc research project: a **lightweight hybrid YOLO detector** for prohibited items in security X-ray images, focused on **small / rare classes** (especially `Nonmetallic_Lighter`) on the **HiXray** benchmark (8 classes).

**Author:** Maraol Feye  
**Domain:** Computer Vision · Object Detection · Aviation / Baggage Security Screening  

---

## Quick start — run on your PC

```bash
git clone https://github.com/mersh01/hybrid-lightweight-xray-detection.git
cd hybrid-lightweight-xray-detection
pip install -r requirements.txt
```

### 1) Point to your dataset (one edit)

Open [`configs/dataset.yaml`](configs/dataset.yaml) and set **only** the `path:` line to your HiXray YOLO root:

```yaml
path: /path/to/HiXray_YOLO2   # ← edit this
```

Expected folder layout:

```text
HiXray_YOLO2/
  images/train  images/val
  labels/train  labels/val
```

If you have the official HiXray release (not YOLO format yet):

```bash
python scripts/convert_hixray_local.py --source /path/to/HiXray --target /path/to/HiXray_YOLO2
```

Then set `path:` to that `--target` folder.

### 2) Validate the final model

```bash
python scripts/run_val.py
```

### 2b) Fair resolution check (long vs epoch20 @ 640 and 768)

Do this before claiming fine-tune gains (isolates imgsz effects):

```bash
python scripts/run_fair_res_eval.py
```

Or on Kaggle: paste [`kaggle/cell_03_fair_res_eval.py`](kaggle/cell_03_fair_res_eval.py).

### 2c) Complexity (GFLOPs @ 640 and 768)

```bash
python scripts/run_complexity.py
```

Measured: **8.9 GFLOPs @640**, **12.8 GFLOPs @768** (3.37M params, both checkpoints).

### 3) Predict on images

```bash
python scripts/run_predict.py --source /path/to/image_or_folder
```

Results save under `runs/predict/`.

### 4) (Optional) Train locally

```bash
# Stage A — long FDD-style (needs strong GPU + many hours)
python scripts/run_train.py --stage a --data configs/dataset.yaml

# Stage B — rare fine-tune from Stage-A best.pt
python scripts/run_train.py --stage b --data configs/dataset.yaml --weights path/to/best.pt
```

For the full thesis recipe (T4×2, oversampling, multi-session resume), use the Kaggle cells below.

---

## Quick start — run on Kaggle (copy-paste)

1. New Kaggle Notebook → **GPU T4**
2. **Add Input:** HiXray_YOLO2 dataset
3. **Add Input:** this GitHub repo *or* upload `models/hybrid_rare_ft_epoch20.pt`
4. Open [`kaggle/cell_01_eval.py`](kaggle/cell_01_eval.py) → **copy the whole file** into one cell → Run  
5. (Optional) Open [`kaggle/cell_02_predict.py`](kaggle/cell_02_predict.py) → paste → Run  

Full training cells (Stage A / resume / rare FT): see [`kaggle/README.md`](kaggle/README.md).

---

## Highlights

| Item | Result |
|------|--------|
| Architecture | Hybrid YOLOv11n-scale model with FDDN, DualConv, SSCAM, DAPA-FPN, SobelConv |
| Long training (FDD-style protocol) | Overall mAP50 ≈ **0.800** |
| Rare-class fine-tune (`epoch20.pt`) | Overall mAP50 ≈ **0.809** (above YOLOv11n ≈ 0.803) |
| Nonmetallic_Lighter | **0.017 → 0.086** after rare-class fine-tune (~5×) |
| Model size | 3.37M params · **8.9 GFLOPs @640** · **12.8 GFLOPs @768** |

---

## Final model

**Released weights:** [`models/hybrid_rare_ft_epoch20.pt`](models/hybrid_rare_ft_epoch20.pt)  
(~26 MB · rare-class fine-tune epoch 20 · overall mAP50 ≈ **0.809**)

See [`models/README.md`](models/README.md).

## Final code

| Path | Role |
|------|------|
| [`modules/hybrid_modules.py`](modules/hybrid_modules.py) | Architecture blocks |
| [`modules/custom_rare_trainer.py`](modules/custom_rare_trainer.py) | Rare-class loss + trainer |
| [`modules/register_hybrid.py`](modules/register_hybrid.py) | Register custom modules for Ultralytics |
| [`configs/hybrid_yolo.yaml`](configs/hybrid_yolo.yaml) | Model YAML |
| [`configs/dataset.yaml`](configs/dataset.yaml) | **Edit `path:` here** |
| [`scripts/run_val.py`](scripts/run_val.py) | Local validation |
| [`scripts/run_predict.py`](scripts/run_predict.py) | Local inference |
| [`scripts/run_train.py`](scripts/run_train.py) | Local train starter |
| [`kaggle/cell_01_eval.py`](kaggle/cell_01_eval.py) | Kaggle eval (copy-paste) |

## Repository layout

```text
models/hybrid_rare_ft_epoch20.pt   # FINAL checkpoint
configs/dataset.yaml               # EDIT path: then run
modules/                           # architecture + trainer + register
scripts/run_val.py / run_predict.py / run_train.py
kaggle/cell_01_eval.py             # copy-paste into Kaggle
kaggle/cell_02_predict.py
docs/                              # results, recipe, citations
```

---

## Selected per-class results (rare-FT `epoch20.pt`)

Eval: HiXray val, `imgsz=768`, `conf=0.001`

| Class | mAP50 |
|-------|------:|
| Cosmetic | 0.665 |
| Laptop | 0.980 |
| Mobile_Phone | 0.979 |
| **Nonmetallic_Lighter** | **0.086** |
| Portable_Charger_1 | 0.951 |
| Portable_Charger_2 | 0.937 |
| Tablet | 0.955 |
| Water | 0.919 |
| **ALL** | **0.809** |

Full tables: [`docs/RESULTS.md`](docs/RESULTS.md)

---

## Module sources (citations)

See [`docs/MODULE_SOURCES.md`](docs/MODULE_SOURCES.md).

---

## LinkedIn blurb (copy-paste)

> Built a hybrid lightweight YOLO detector for prohibited-item detection in X-ray baggage images (HiXray, 8 classes). Combined frequency-domain, attention, and distraction-aware FPN modules into one Ultralytics architecture, trained with a long FDD-style protocol, then rare-class fine-tuned with weighted loss + oversampling. Improved overall mAP50 to ~0.809 (above YOLOv11n baseline) and raised Nonmetallic_Lighter mAP50 from 0.017 to 0.086. Code & results: https://github.com/mersh01/hybrid-lightweight-xray-detection

---

## Disclaimer

- Re-implementations are **adapted for Ultralytics YAML**, not official author code.  
- The **final** fine-tuned checkpoint is included under `models/`. Intermediate `.pt` files and the full HiXray image dataset are not.  
- This is academic / research code from an MSc thesis workflow.

## License

Code in this repository is released under the MIT License (see `LICENSE`).  
Dataset and third-party papers remain under their original licenses.
