#!/usr/bin/env python3
"""Measure params + GFLOPs for the hybrid model at imgsz 640 and 768.

Architecture is identical for long-stage and epoch20 weights; FLOPs depend on
input size only. We still load both checkpoints to confirm matching params.

Usage:
  python scripts/run_complexity.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def measure(weights: Path, imgsz: int) -> dict:
    from modules.register_hybrid import register

    register(force=True)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # model.info returns (n_layers, n_params, n_gradients, n_flops) in recent Ultralytics
    info = model.info(detailed=False, verbose=True, imgsz=imgsz)
    if isinstance(info, (tuple, list)) and len(info) >= 4:
        n_l, n_p, n_g, flops = info[0], info[1], info[2], info[3]
    else:
        # Fallback: parse from model attributes after info()
        n_l = getattr(model.model, "model", model.model)
        n_p = sum(p.numel() for p in model.model.parameters())
        n_g = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        flops = None
        # Try thop via ultralytics utils
        try:
            import torch
            from ultralytics.utils.torch_utils import get_flops

            flops = get_flops(model.model, imgsz=imgsz)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not measure FLOPs: {e}") from e

    return {
        "weights": weights.name,
        "imgsz": imgsz,
        "layers": int(n_l) if n_l is not None else None,
        "params": int(n_p),
        "gradients": int(n_g) if n_g is not None else None,
        "GFLOPs": float(flops),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GFLOPs at 640 and 768")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=Path,
        default=[
            REPO / "models" / "hybrid_fdd_long_best.pt",
            REPO / "models" / "hybrid_rare_ft_epoch20.pt",
        ],
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 768])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "runs" / "complexity" / "complexity_matrix.csv",
    )
    args = parser.parse_args()

    rows = []
    for w in args.weights:
        if not w.is_file():
            print(f"SKIP missing {w}")
            continue
        for imgsz in args.imgsz:
            print(f"\n=== {w.name} | imgsz={imgsz} ===")
            row = measure(w, imgsz)
            rows.append(row)
            print(
                f"params={row['params']:,}  GFLOPs={row['GFLOPs']:.3f}  layers={row['layers']}"
            )

    if not rows:
        raise SystemExit("No checkpoints measured.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("COMPLEXITY MATRIX")
    print("=" * 72)
    print(f"{'weights':32s} {'imgsz':>6s} {'params':>12s} {'GFLOPs':>8s}")
    for r in rows:
        print(
            f"{r['weights']:32s} {r['imgsz']:6d} {r['params']:12,d} {r['GFLOPs']:8.3f}"
        )

    # Theoretical scale check: FLOPs ~ imgsz^2
    by = {(r["weights"], r["imgsz"]): r for r in rows}
    print("\nScale check (GFLOPs_768 / GFLOPs_640) vs (768/640)^2 = 1.440:")
    for w in args.weights:
        if not w.is_file():
            continue
        a, b = by[(w.name, 640)], by[(w.name, 768)]
        ratio = b["GFLOPs"] / a["GFLOPs"] if a["GFLOPs"] else float("nan")
        print(f"  {w.name}: {ratio:.3f}")

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
