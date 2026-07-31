#!/usr/bin/env python3
"""Optional local training starter (Stage A or Stage B).

For the full thesis recipe (T4×2, oversampling, resume), prefer the
copy-paste Kaggle cells under kaggle/.

Examples:
  # Stage A — long FDD-style from YAML (needs GPU + time)
  python scripts/run_train.py --stage a --data configs/dataset.yaml

  # Stage B — rare fine-tune from a Stage-A best.pt
  python scripts/run_train.py --stage b --data configs/dataset.yaml --weights path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local hybrid YOLO training")
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--data", type=Path, default=REPO / "configs" / "dataset.yaml")
    parser.add_argument("--weights", type=Path, default=None, help="Required for stage b")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=REPO / "runs" / "train")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Edit configs/dataset.yaml path: to your HiXray_YOLO2 root."
        )

    from modules.register_hybrid import register

    register()
    from ultralytics import YOLO

    if args.stage == "a":
        model = YOLO(str(REPO / "configs" / "hybrid_yolo.yaml"))
        epochs = args.epochs or 300
        model.train(
            data=str(args.data),
            epochs=epochs,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.9,
            cos_lr=True,
            warmup_epochs=3,
            pretrained=False,
            project=str(args.project),
            name="hybrid_fdd_protocol",
            exist_ok=True,
            seed=0,
        )
    else:
        if args.weights is None or not args.weights.is_file():
            raise SystemExit("--weights path/to/stageA_best.pt is required for stage b")
        from modules.custom_rare_trainer import RareClassTrainer

        model = YOLO(str(args.weights))
        epochs = args.epochs or 40
        model.train(
            data=str(args.data),
            epochs=epochs,
            imgsz=768,
            batch=args.batch,
            device=args.device,
            optimizer="AdamW",
            lr0=2e-4,
            lrf=0.01,
            cls=0.8,
            copy_paste=0.4,
            mosaic=0.8,
            close_mosaic=10,
            patience=15,
            trainer=RareClassTrainer,
            project=str(args.project),
            name="hybrid_rare_ft",
            exist_ok=True,
            seed=0,
        )
    print("Training finished. Check runs/train/ for weights.")


if __name__ == "__main__":
    main()
