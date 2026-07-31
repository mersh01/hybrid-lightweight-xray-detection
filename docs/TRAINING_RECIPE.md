TWO-STAGE TRAINING RECIPE (final reported pipeline)

================================================================================
STAGE A — Long FDD-protocol training (hybrid_fdd_protocol_300)
================================================================================
Start: random / YAML hybrid architecture, pretrained=false
epochs: 300 (early stop ~296, patience=50)
imgsz: 640
batch: 16
device: 0,1 (T4 x2)
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.9
cos_lr: true
warmup_epochs: 3
seed: 0
deterministic: true
save_period: 25
No rare-class oversampling / no class-weighted loss in this stage

Code: hybrid_fdd_protocol_kaggle.py + hybrid_fdd_resume_kaggle.py
Args samples: copied under long_stage/

================================================================================
STAGE B — Rare-class fine-tune round 1 (SELECTED) — hybrid_fdd_rare_ft
================================================================================
Start: Stage A best.pt (results166)
epochs: 40 (early stop at 35, patience=15); BEST = epoch 20
imgsz: 768
batch: 16
optimizer: AdamW
lr0: 0.0002
lrf: 0.01
cls: 0.8
copy_paste: 0.4
mosaic: 0.8
close_mosaic: 10
Class weights (BCE pos_weight): [1.5, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]
  Cosmetic=1.5, Nonmetallic_Lighter=3.0, others=1.0
Oversample: Cosmetic images x2, Lighter images x5
RareClassTrainer saves best by (Cosmetic_mAP50 + Lighter_mAP50)/2

Code: fdd_rare_finetune_kaggle.py / FDD_Rare_Finetune_Package
Args: rare_finetune/args.yaml

================================================================================
STAGE B2 — Rare FT round 2 (ABLATION ONLY, not primary)
================================================================================
Start: epoch20.pt from Stage B
weights Cos=2.0 Lighter=5.0; oversample Cos x3 Lighter x10; imgsz=896; 25 ep; lr0=5e-5
Result: Lighter 0.100 but Cos/ALL slightly worse → keep Stage B epoch20 as primary.
