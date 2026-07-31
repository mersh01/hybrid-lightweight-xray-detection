#!/usr/bin/env python3
"""Fair resolution re-validation: long-stage vs epoch20 at imgsz 640 and 768.

This isolates fine-tuning gains from resolution effects.

Usage (PC, GPU recommended):
  # Edit configs/dataset.yaml path: first
  python scripts/run_fair_res_eval.py

  python scripts/run_fair_res_eval.py \\
      --long models/hybrid_fdd_long_best.pt \\
      --ft models/hybrid_rare_ft_epoch20.pt \\
      --data configs/dataset.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLASSES = [
    "Cosmetic",
    "Laptop",
    "Mobile_Phone",
    "Nonmetallic_Lighter",
    "Portable_Charger_1",
    "Portable_Charger_2",
    "Tablet",
    "Water",
]


def eval_one(weights: Path, data: Path, imgsz: int, batch: int, device: str, conf: float) -> dict:
    from modules.register_hybrid import register

    register(force=True)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        plots=False,
        verbose=False,
    )
    row = {
        "weights": weights.name,
        "imgsz": imgsz,
        "ALL_mAP50": float(metrics.box.map50),
        "ALL_mAP50_95": float(metrics.box.map),
    }
    for i, name in metrics.names.items():
        row[f"{name}_mAP50"] = float(metrics.box.ap50[i])
    row["rare_avg_mAP50"] = (
        row["Cosmetic_mAP50"] + row["Nonmetallic_Lighter_mAP50"]
    ) / 2.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair 640/768 re-val matrix")
    parser.add_argument(
        "--long",
        type=Path,
        default=REPO / "models" / "hybrid_fdd_long_best.pt",
        help="Long-stage FDD-protocol best.pt",
    )
    parser.add_argument(
        "--ft",
        type=Path,
        default=REPO / "models" / "hybrid_rare_ft_epoch20.pt",
        help="Selected rare-FT epoch20.pt",
    )
    parser.add_argument("--data", type=Path, default=REPO / "configs" / "dataset.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 768])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "runs" / "fair_res_eval" / "fair_res_matrix.csv",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Missing {args.data} — set path: in configs/dataset.yaml")
    for w in (args.long, args.ft):
        if not w.is_file():
            raise SystemExit(f"Missing checkpoint: {w}")

    rows = []
    for label, weights in (("long", args.long), ("epoch20", args.ft)):
        for imgsz in args.imgsz:
            print(f"\n=== {label} | {weights.name} | imgsz={imgsz} ===")
            row = eval_one(weights, args.data, imgsz, args.batch, args.device, args.conf)
            row["stage"] = label
            rows.append(row)
            print(
                f"ALL={row['ALL_mAP50']:.5f}  "
                f"Cos={row['Cosmetic_mAP50']:.5f}  "
                f"Lig={row['Nonmetallic_Lighter_mAP50']:.5f}  "
                f"rare={row['rare_avg_mAP50']:.5f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FAIR RESOLUTION MATRIX (mAP50)")
    print("=" * 72)
    print(f"{'stage':10s} {'imgsz':>6s} {'ALL':>8s} {'Cos':>8s} {'Lighter':>8s} {'rare':>8s}")
    for r in rows:
        print(
            f"{r['stage']:10s} {r['imgsz']:6d} "
            f"{r['ALL_mAP50']:8.5f} {r['Cosmetic_mAP50']:8.5f} "
            f"{r['Nonmetallic_Lighter_mAP50']:8.5f} {r['rare_avg_mAP50']:8.5f}"
        )

    # Same-resolution deltas (FT − long)
    print("\nDELTA at matched imgsz (epoch20 − long):")
    by_key = {(r["stage"], r["imgsz"]): r for r in rows}
    for imgsz in args.imgsz:
        a, b = by_key[("long", imgsz)], by_key[("epoch20", imgsz)]
        print(
            f"  imgsz={imgsz}:  ALL {b['ALL_mAP50']-a['ALL_mAP50']:+.5f}  "
            f"Cos {b['Cosmetic_mAP50']-a['Cosmetic_mAP50']:+.5f}  "
            f"Lig {b['Nonmetallic_Lighter_mAP50']-a['Nonmetallic_Lighter_mAP50']:+.5f}"
        )
    print(f"\nSaved: {args.out}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
