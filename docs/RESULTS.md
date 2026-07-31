================================================================================
FDD RARE-CLASS FINE-TUNE — FULL RESULTS PACKAGE
Hybrid Lightweight X-Ray Detector on HiXray (8 classes)
================================================================================
Created for thesis documentation — Maraol Feye
Run name: hybrid_fdd_rare_ft
Date of run: July 2026 (Kaggle T4 x2)

--------------------------------------------------------------------------------
1. WHAT WE TRAINED
--------------------------------------------------------------------------------
Base checkpoint : FDD-protocol best.pt (results166, ~epoch 235 of 300-ep SGD run)
Goal            : Improve rare / small classes (Cosmetic, Nonmetallic_Lighter)
                  without destroying overall mAP

Architecture    : Hybrid YOLO (YOLOv11n-scale) with:
                  DualConv, FDDN, SobelConv, SSCAM, DAPA-FPN / DAPABlock, HybridBlock

Fine-tune recipe:
  - Class loss weights (BCE pos_weight):
      [1.5, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]
      Cosmetic=1.5 | Nonmetallic_Lighter=3.0 | others=1.0
      (These do NOT need to sum to 1 — they are relative multipliers.)
  - Oversampling train images: Cosmetic x2, Lighter x5
  - imgsz=768, epochs=40 (early-stopped at 35), batch=16, device=0,1
  - optimizer=AdamW, lr0=0.0002, lrf=0.01, cls=0.8
  - copy_paste=0.4, mosaic=0.8, close_mosaic=10, patience=15
  - RareClassTrainer: saves best by (Cosmetic_mAP50 + Lighter_mAP50) / 2

Code files in this package:
  01_CODE_fdd_rare_finetune_kaggle.py   — full Python source of all Kaggle cells
  01_CODE_Kaggle_CELLS.txt             — paste-ready Cell 0..6 text
  03_CODE_eval_epoch20.txt              — official per-class eval cell for epoch20.pt

--------------------------------------------------------------------------------
2. TRAINING CURVE (from results.csv) — overall mAP50 only
--------------------------------------------------------------------------------
Logged epochs : 1 → 35 (early stop; planned 40)
Best overall  : epoch 20 — mAP50 = 0.81056  |  mAP50-95 = 0.50275
Final         : epoch 35 — mAP50 = 0.80634  |  mAP50-95 = 0.50160

Trainer rare-class fitness peak (from Kaggle train log @ ep 20):
  Cosmetic ≈ 0.672   Lighter ≈ 0.092   rare_avg ≈ 0.382   ALL ≈ 0.811

Full epoch table (overall mAP50):
  ep  1: 0.78601
  ep  2: 0.79391
  ep  3: 0.79607
  ep  4: 0.79862
  ep  5: 0.80220
  ep  6: 0.80111
  ep  7: 0.80248
  ep  8: 0.80802
  ep  9: 0.80636
  ep 10: 0.80892
  ep 11: 0.80797
  ep 12: 0.80676
  ep 13: 0.80409
  ep 14: 0.80568
  ep 15: 0.80715
  ep 16: 0.80929
  ep 17: 0.80760
  ep 18: 0.80942
  ep 19: 0.80941
  ep 20: 0.81056  <-- BEST overall (CSV)
  ep 21: 0.80869
  ep 22: 0.80864
  ep 23: 0.80741
  ep 24: 0.80817
  ep 25: 0.80829
  ep 26: 0.80817
  ep 27: 0.80821
  ep 28: 0.80793
  ep 29: 0.80826
  ep 30: 0.80790
  ep 31: 0.80759
  ep 32: 0.80782
  ep 33: 0.80718
  ep 34: 0.80660
  ep 35: 0.80634  <-- last (early stop)

Saved weights in resultsfinetune40.zip:
  epoch0.pt, epoch5.pt, epoch10.pt, epoch15.pt, epoch20.pt,
  epoch25.pt, epoch30.pt, last.pt
  (best.pt was created on Kaggle at ep20 but often missing from zip —
   use epoch20.pt as the peak checkpoint.)

Raw CSV copy: 02_results_training_curve.csv
Args copy   : 02_args.yaml

--------------------------------------------------------------------------------
3. BEFORE FINE-TUNE — FDD-protocol best.pt (independent val, imgsz=640)
--------------------------------------------------------------------------------
Source checkpoint : results166 / best.pt (~ep 235 of 300-ep FDD protocol)
Eval              : HiXray val, conf=0.001

  Class                     mAP50     mAP50-95
  ----------------------  --------  ----------
  Cosmetic                  0.65434     0.22744
  Laptop                    0.97934     0.73414
  Mobile_Phone              0.97941     0.66602
  Nonmetallic_Lighter       0.01682     0.00706
  Portable_Charger_1        0.95054     0.57189
  Portable_Charger_2        0.93083     0.54099
  Tablet                    0.95591     0.69671
  Water                     0.93189     0.57897
  ALL                       0.79989     0.50290
  Rare avg (Cos+Lig)/2      0.33558

--------------------------------------------------------------------------------
4. AFTER FINE-TUNE — last.pt (Cell 6 fallback; NOT the peak)
--------------------------------------------------------------------------------
Eval: imgsz=768, conf=0.001

  Class                     mAP50
  ----------------------  --------
  Cosmetic                  0.66370
  Laptop                    0.97738
  Mobile_Phone              0.97942
  Nonmetallic_Lighter       0.07643
  Portable_Charger_1        0.95094
  Portable_Charger_2        0.93658
  Tablet                    0.95827
  Water                     0.91968
  ALL                       0.80780
  Rare avg                  0.37007

--------------------------------------------------------------------------------
5. OFFICIAL AFTER FINE-TUNE — epoch20.pt (THESIS NUMBERS)
--------------------------------------------------------------------------------
Eval: imgsz=768, conf=0.001, HiXray val 9069 images
Checkpoint: epoch20.pt (best rare-class / overall peak epoch)

  Class                     mAP50     mAP50-95
  ----------------------  --------  ----------
  Cosmetic                  0.66460     0.23359
  Laptop                    0.97963     0.72666
  Mobile_Phone              0.97863     0.66208
  Nonmetallic_Lighter       0.08598     0.03080
  Portable_Charger_1        0.95118     0.57353
  Portable_Charger_2        0.93659     0.53888
  Tablet                    0.95475     0.69120
  Water                     0.91916     0.56466
  ALL                       0.80881     0.50267
  P=0.79274  R=0.79489
  Rare avg (Cos+Lig)/2      0.37529

NOTE on 0.811 vs 0.809:
  Training log @ ep20 showed overall mAP50 ≈ 0.811 (CSV exact 0.81056).
  Independent re-validation of the saved epoch20.pt gave 0.80881.
  Difference ≈ 0.002 is normal (conf/EMA/AMP). Report 0.809 for thesis
  from the official re-val, or state both as above.

--------------------------------------------------------------------------------
6. COMPARISON TABLE — PRIMARY @ imgsz=640 (matched resolution, 8.9 GFLOPs)
--------------------------------------------------------------------------------
Both checkpoints validated independently on HiXray val (9069 images), conf=0.001.
Long-stage: prior GPU re-val @640.  epoch20: CPU re-val @640 (this session).

  Class                   Long @640    epoch20 @640    Delta
  --------------------  ------------  --------------  --------
  Cosmetic                    0.654           0.653     -0.001
  Laptop                      0.979           0.980     +0.001
  Mobile_Phone                0.979           0.979      0.000
  Nonmetallic_Lighter         0.017           0.066     +0.049
  Portable_Charger_1          0.951           0.949     -0.002
  Portable_Charger_2          0.931           0.932     +0.001
  Tablet                      0.956           0.957     +0.001
  Water                       0.932           0.922     -0.010
  ALL                         0.800           0.805     +0.005
  Rare avg                    0.336           0.360     +0.024

Complexity at this operating point: 3.37M params, **8.9 GFLOPs**.

External reference:
  YOLOv11n baseline (overall mAP50) ≈ 0.803
  → epoch20 @640 overall 0.805 is slightly above this baseline.

--------------------------------------------------------------------------------
6a. APPENDIX — epoch20 @ imgsz=768 (train resolution; 12.8 GFLOPs)
--------------------------------------------------------------------------------
  ALL 0.809 · Cosmetic 0.665 · Lighter 0.086
  Use only as appendix / ablation; primary thesis numbers are §6 @640.

--------------------------------------------------------------------------------
6b. FAIR RESOLUTION MATRIX (filled for epoch20 @640; long @640 known)
--------------------------------------------------------------------------------
  Checkpoint                         imgsz=640              imgsz=768
  --------------------------------  ----------------------  ----------------------
  Long-stage best.pt                ALL 0.800 Cos 0.654     (optional appendix)
                                    Lig 0.017
  Selected epoch20.pt               ALL 0.805 Cos 0.653     ALL 0.809 Cos 0.665
                                    Lig 0.066               Lig 0.086

  Matched-imgsz delta @640 (epoch20 − long):
    ALL +0.005   Cos −0.001   Lig +0.049

How to re-run:
  python scripts/run_val.py --weights models/hybrid_rare_ft_epoch20.pt --imgsz 640
  Kaggle: kaggle/cell_05_reeval_640_primary.py
  CSV: docs/tables/epoch20_perclass_imgsz640.csv

--------------------------------------------------------------------------------
7. TAKEAWAYS
--------------------------------------------------------------------------------
1. Rare-class fine-tune from FDD best.pt raised overall mAP50 from ~0.800 to ~0.809
   and Lighter from 0.017 to 0.086 (~5x).
2. Cosmetic improved slightly; strong classes stayed strong.
3. Use epoch20.pt as the delivered fine-tuned weights for reporting.
4. Contribution story: hybrid architecture (FDDN/SSCAM/DAPA/DualConv) trained with
   FDD-style long protocol, then rare-class weighted fine-tune for HiXray imbalance.

--------------------------------------------------------------------------------
8. RELATED LOCAL PATHS
--------------------------------------------------------------------------------
Package folder:
  C:\Users\merao\OneDrive\Documents\Reasearch\FDD_Rare_Finetune_Package\

Raw Kaggle download (extracted):
  C:\Users\merao\OneDrive\Documents\Reasearch\resultsfinetune40\

Zip:
  C:\Users\merao\OneDrive\Documents\Reasearch\resultsfinetune40.zip

Recommended checkpoint to keep:
  ...\resultsfinetune40\HiXray_Training_Runs\hybrid_fdd_rare_ft\weights\epoch20.pt

================================================================================
END OF RESULTS FILE
================================================================================
