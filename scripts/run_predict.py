#!/usr/bin/env python3
"""Run inference with the released hybrid checkpoint.

Usage:
  python scripts/run_predict.py --source path/to/image_or_folder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with hybrid_rare_ft_epoch20.pt")
    parser.add_argument("--source", required=True, help="Image, folder, or video path")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO / "models" / "hybrid_rare_ft_epoch20.pt",
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO / "runs" / "predict",
    )
    parser.add_argument("--name", default="hybrid_epoch20")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Missing weights: {args.weights}")

    from modules.register_hybrid import register

    register()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        exist_ok=True,
    )
    out = Path(results[0].save_dir) if results else args.project / args.name
    print(f"Saved predictions → {out}")


if __name__ == "__main__":
    main()
