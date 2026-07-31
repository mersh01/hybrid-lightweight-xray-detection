#!/usr/bin/env python3
"""Convert official HiXray annotations to YOLO layout.

Usage:
  python scripts/convert_hixray_local.py \\
      --source /path/to/HiXray \\
      --target /path/to/HiXray_YOLO2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

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


def convert_dataset(source_dir: Path, target_dir: Path) -> None:
    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    splits = [("train", "train"), ("test", "val")]

    for _, yolo_split in splits:
        (target_dir / "images" / yolo_split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / yolo_split).mkdir(parents=True, exist_ok=True)

    for original_split, yolo_split in splits:
        img_dir = source_dir / original_split / f"{original_split}_image"
        ann_dir = source_dir / original_split / f"{original_split}_annotation"
        if not img_dir.exists() or not ann_dir.exists():
            print(f"Warning: missing {img_dir} or {ann_dir} — skip")
            continue

        print(f"Processing {original_split} → {yolo_split}...")
        for ann_file in tqdm(list(ann_dir.glob("*.txt"))):
            lines = ann_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not lines:
                continue

            img_path = img_dir / f"{ann_file.stem}.jpg"
            if not img_path.exists():
                img_name = lines[0].strip().split(" ")[0]
                img_path = img_dir / img_name
                if not img_path.exists():
                    continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            yolo_lines = []
            for line in lines:
                parts = line.strip().split(" ")
                if len(parts) < 6:
                    continue
                _, cls_name, xmin, ymin, xmax, ymax = parts[:6]
                if cls_name not in class_to_id:
                    continue
                xmin, ymin, xmax, ymax = map(float, (xmin, ymin, xmax, ymax))
                x_c = ((xmin + xmax) / 2.0) / w_img
                y_c = ((ymin + ymax) / 2.0) / h_img
                w = (xmax - xmin) / w_img
                h = (ymax - ymin) / h_img
                x_c, y_c = max(0, min(1, x_c)), max(0, min(1, y_c))
                w, h = max(0, min(1, w)), max(0, min(1, h))
                yolo_lines.append(
                    f"{class_to_id[cls_name]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )

            if not yolo_lines:
                continue
            shutil.copy2(img_path, target_dir / "images" / yolo_split / img_path.name)
            (target_dir / "labels" / yolo_split / f"{ann_file.stem}.txt").write_text(
                "\n".join(yolo_lines) + "\n", encoding="utf-8"
            )

    yaml_path = target_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {target_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 8\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
        + "\n",
        encoding="utf-8",
    )
    print(f"Done. Wrote {yaml_path}")
    print("Point configs/dataset.yaml path: to this folder, or use --data with this yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HiXray → YOLO layout")
    parser.add_argument("--source", type=Path, required=True, help="Official HiXray root")
    parser.add_argument("--target", type=Path, required=True, help="Output YOLO root")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source not found: {args.source}")
    convert_dataset(args.source, args.target)


if __name__ == "__main__":
    main()
