#!/usr/bin/env python3
"""Short imgsz=640 polish fine-tune to push overall mAP50 above 0.808.

Gap to close (from epoch20 @640 re-val): ~0.003 (0.805 → 0.808 target).

Start: models/hybrid_rare_ft_epoch20.pt
Goal:  imgsz=640, 8.9 GFLOPs, ALL mAP50 > 0.808

Usage (GPU required):
  python scripts/run_polish_640.py --data configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description="Short 640 polish from epoch20")
    parser.add_argument("--data", type=Path, default=REPO / "configs" / "dataset.yaml")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO / "models" / "hybrid_rare_ft_epoch20.pt",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=REPO / "runs" / "polish_640")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Edit {args.data} path: to your HiXray_YOLO2 root.")
    if not args.weights.is_file():
        raise SystemExit(f"Missing {args.weights}")

    from modules.register_hybrid import register

    register()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=3e-5,
        lrf=0.01,
        warmup_epochs=1,
        cos_lr=True,
        cls=0.5,
        mosaic=0.3,
        close_mosaic=3,
        copy_paste=0.0,
        patience=5,
        # Standard trainer — save best by overall mAP50 (not rare-class fitness)
        project=str(args.project),
        name="hybrid_polish_640",
        exist_ok=True,
        seed=0,
        val=True,
    )
    print("Polish done. Re-validate best.pt at imgsz=640:")
    print("  python scripts/run_val.py --weights runs/polish_640/hybrid_polish_640/weights/best.pt --imgsz 640")


if __name__ == "__main__":
    main()
