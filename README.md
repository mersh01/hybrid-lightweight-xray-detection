# Hybrid Lightweight X-Ray Prohibited-Item Detection (HiXray)

MSc research project: a **lightweight hybrid YOLO detector** for prohibited items in security X-ray images, focused on **small / rare classes** (especially `Nonmetallic_Lighter`) on the **HiXray** benchmark (8 classes).

**Author:** Maraol Feye  
**Domain:** Computer Vision · Object Detection · Aviation / Baggage Security Screening  

---

## Highlights

| Item | Result |
|------|--------|
| Architecture | Hybrid YOLOv11n-scale model with FDDN, DualConv, SSCAM, DAPA-FPN, SobelConv |
| Long training (FDD-style protocol) | Overall mAP50 ≈ **0.800** |
| Rare-class fine-tune (`epoch20.pt`) | Overall mAP50 ≈ **0.809** (above YOLOv11n ≈ 0.803) |
| Nonmetallic_Lighter | **0.017 → 0.086** after rare-class fine-tune (~5×) |
| Model size | ~3.37M parameters · ~8.9 GFLOPs · ~13 MB checkpoint |

---

## What I built / learned

1. **Hybrid architecture design** — composed proven X-ray modules (not one copied paper):
   - `DualConv` + `FDDN` ← FDD-YOLO  
   - `SSCAM` ← DGDN  
   - `DAPA-FPN` ← iX-Det  
   - `SobelConv` edge branch ← E-MPDNet inspiration  
   - `HybridBlock` = FDDN + SSCAM (**own composition**)

2. **Fair long-schedule training** — matched FDD-YOLO-style protocol (SGD, 300 epochs, `imgsz=640`, cosine LR) for fair comparison.

3. **Rare-class fine-tuning** — class-weighted BCE (`pos_weight`), image oversampling for Cosmetic / Lighter, custom `RareClassTrainer` that saves best by rare-class fitness.

4. **Engineering for Kaggle / Ultralytics** — custom-module registration, `parse_model` channel patching, DDP-safe trainer, multi-session resume after 12h limits.

5. **Honest evaluation** — independent per-class validation; report re-val numbers (not only training-time logs).

---

## Final model (download)

**Released weights:** [`models/hybrid_rare_ft_epoch20.pt`](models/hybrid_rare_ft_epoch20.pt)  
(~26 MB · rare-class fine-tune epoch 20 · overall mAP50 ≈ **0.809**)

See [`models/README.md`](models/README.md) for load instructions.

## Final code (what to look at)

| Path | Role |
|------|------|
| [`modules/hybrid_modules.py`](modules/hybrid_modules.py) | Final architecture blocks |
| [`modules/custom_rare_trainer.py`](modules/custom_rare_trainer.py) | Final rare-class loss + trainer |
| [`configs/hybrid_yolo.yaml`](configs/hybrid_yolo.yaml) | Final model YAML |
| [`scripts/train_fdd_protocol_kaggle.py`](scripts/train_fdd_protocol_kaggle.py) | Stage A long training |
| [`scripts/train_rare_finetune_kaggle.py`](scripts/train_rare_finetune_kaggle.py) | Stage B fine-tune that produced `epoch20.pt` |
| [`scripts/eval_epoch20_kaggle.txt`](scripts/eval_epoch20_kaggle.txt) | Official per-class eval cell |

## Repository layout

```text
models/
  hybrid_rare_ft_epoch20.pt # FINAL released checkpoint
  README.md
configs/
  hybrid_yolo.yaml          # hybrid architecture
  dataset.yaml.example      # HiXray YOLO layout example
modules/
  hybrid_modules.py         # DualConv, FDDN, SobelConv, SSCAM, DAPA-*, HybridBlock
  custom_rare_trainer.py    # weighted cls loss + rare-class best.pt criterion
scripts/
  convert_hixray_local.py
  train_fdd_protocol_kaggle.py
  resume_fdd_protocol_kaggle.py
  train_rare_finetune_kaggle.py
  train_rare_finetune_round2_kaggle.py
  eval_epoch20_kaggle.txt
docs/
  TRAINING_RECIPE.md
  RESULTS.md
  MODULE_SOURCES.md
  MODEL_COMPLEXITY.md
  DATASET_STATS.md
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

Full tables and training curves: [`docs/RESULTS.md`](docs/RESULTS.md)

---

## Quick start (local / Kaggle)

### Requirements

```bash
pip install ultralytics pillow
```

See [`requirements.txt`](requirements.txt). Training in this project was primarily run on **Kaggle (Tesla T4 ×2)**.

### Data

1. Obtain the [HiXray](https://github.com/hi-xray) (or your HiXray YOLO export) dataset.  
2. Convert with `scripts/convert_hixray_local.py` if needed.  
3. Copy `configs/dataset.yaml.example` → `dataset.yaml` and set `path:`.

### Train (Kaggle cells)

The `scripts/*_kaggle.py` files contain full notebook cells used for:

1. **Stage A** — long FDD-protocol training (`train_fdd_protocol_kaggle.py`)  
2. **Resume** after timeout (`resume_fdd_protocol_kaggle.py`)  
3. **Stage B** — rare-class fine-tune (`train_rare_finetune_kaggle.py`)  

Details: [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md)

---

## Module sources (citations)

See [`docs/MODULE_SOURCES.md`](docs/MODULE_SOURCES.md). Key DOIs:

- FDD-YOLO: https://doi.org/10.4108/airo.10277  
- iX-Det: https://doi.org/10.1093/jcde/qwaf126  
- DGDN / SSCAM: https://doi.org/10.1109/ACCESS.2025.3581450  
- E-MPDNet: https://doi.org/10.1109/ACCESS.2025.3622506  

---

## LinkedIn blurb (copy-paste)

> Built a hybrid lightweight YOLO detector for prohibited-item detection in X-ray baggage images (HiXray, 8 classes). Combined frequency-domain, attention, and distraction-aware FPN modules into one Ultralytics architecture, trained with a long FDD-style protocol, then rare-class fine-tuned with weighted loss + oversampling. Improved overall mAP50 to ~0.809 (above YOLOv11n baseline) and raised Nonmetallic_Lighter mAP50 from 0.017 to 0.086. Code & results: &lt;this-repo-URL&gt;

---

## Disclaimer

- Re-implementations are **adapted for Ultralytics YAML**, not official author code.  
- The **final** fine-tuned checkpoint is included under `models/`. Intermediate `.pt` files and the full HiXray image dataset are not.  
- This is academic / research code from an MSc thesis workflow.

## License

Code in this repository is released under the MIT License (see `LICENSE`).  
Dataset and third-party papers remain under their original licenses.
