#!/usr/bin/env python3
"""Validate the released hybrid checkpoint on HiXray val.

Usage:
  1. Copy configs/dataset.yaml.example → configs/dataset.yaml
  2. Edit only: path: /your/HiXray_YOLO2
  3. python scripts/run_val.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hybrid_rare_ft_epoch20.pt")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO / "configs" / "dataset.yaml",
        help="Dataset YAML (edit path: to your HiXray_YOLO2 root)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO / "models" / "hybrid_rare_ft_epoch20.pt",
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    if not args.data.is_file():
        example = REPO / "configs" / "dataset.yaml.example"
        raise SystemExit(
            f"Missing {args.data}\n"
            f"  copy {example.name} → dataset.yaml and set path: to your dataset root"
        )
    if not args.weights.is_file():
        raise SystemExit(f"Missing weights: {args.weights}")

    from modules.register_hybrid import register

    register()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        plots=True,
        verbose=True,
    )
    print(f"\nALL mAP50={metrics.box.map50:.5f}  mAP50-95={metrics.box.map:.5f}")
    for i, name in metrics.names.items():
        print(f"  {name:22s}  mAP50={float(metrics.box.ap50[i]):.5f}")


if __name__ == "__main__":
    main()
