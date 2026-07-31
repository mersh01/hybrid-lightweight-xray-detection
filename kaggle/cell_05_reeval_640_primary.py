# =============================================================================
# Kaggle FAST re-eval — epoch20 + long-stage at imgsz=640 (and 768)
# Copy ENTIRE cell. GPU T4. ~15–25 min for all 4 runs.
# Inputs: HiXray_YOLO2 + GitHub repo (or both .pt files)
# =============================================================================

!pip -q install ultralytics==8.4.103 pillow==11.0.0

import os, re, sys, importlib, shutil, csv, json
from pathlib import Path

WORK = Path("/kaggle/working/reeval_640")
WORK.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")

DATASET = next((p for p in INPUT.rglob("HiXray_YOLO2") if (p / "images" / "val").exists()), None)
if DATASET is None:
    DATASET = next((p.parent.parent for p in INPUT.rglob("images/val") if (p.parent.parent / "labels" / "val").exists()), None)
assert DATASET, "Add HiXray_YOLO2"

CLASSES = ["Cosmetic","Laptop","Mobile_Phone","Nonmetallic_Lighter","Portable_Charger_1","Portable_Charger_2","Tablet","Water"]
yaml = WORK / "dataset.yaml"
yaml.write_text(f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n" + "\n".join(f"  {i}: {n}" for i,n in enumerate(CLASSES)) + "\n", encoding="utf-8")

def find(*names):
    for n in names:
        hits = [p for p in INPUT.rglob(n) if p.stat().st_size > 5_000_000]
        if hits:
            return sorted(hits, key=lambda p: -p.stat().st_size)[0]
    return None

LONG = find("hybrid_fdd_long_best.pt") or find("best.pt")
FT = find("hybrid_rare_ft_epoch20.pt") or find("epoch20.pt")
assert LONG and FT, "Add long best.pt and epoch20.pt"

MOD = next(INPUT.rglob("hybrid_modules.py"), None)
assert MOD, "Add GitHub repo (modules/) as Input"
for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
    shutil.copy2(MOD.parent / f, WORK / f)

sys.path.insert(0, str(WORK))
import custom_rare_trainer
from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks
CUSTOM = {"DualConv":DualConv,"SobelConv":SobelConv,"FDDN":FDDN,"SSCAM":SSCAM,"DAPA_FPN":DAPA_FPN,"HybridBlock":HybridBlock,"DAPABlock":DAPABlock}
for k,v in CUSTOM.items(): setattr(yolo_tasks, k, v)
tasks_py = Path(yolo_tasks.__file__); t = tasks_py.read_text(encoding="utf-8")
if "from hybrid_modules import" not in t:
    t = f"import sys\nif r'{WORK}' not in sys.path:\n    sys.path.insert(0, r'{WORK}')\nfrom hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock\n" + t
if "elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv)" not in t:
    inj = ("        elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv):\n"
           "            c1, c2 = ch[f], args[0]\n"
           "            if c2 != nc:\n"
           "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
           "            args = [c1, c2, *args[1:]]\n"
           "        elif m is SSCAM:\n"
           "            c1 = ch[f]\n"
           "            c2 = c1\n"
           "            args = [c1]\n")
    pat = re.compile(r"(^[ \t]+)else:\n[ \t]+c2 = ch\[f\]\n", re.MULTILINE)
    m = pat.search(t); assert m; t, _ = pat.subn(inj + m.group(0), t, count=1)
    tasks_py.write_text(t, encoding="utf-8")
importlib.reload(yolo_tasks)
for k,v in CUSTOM.items(): setattr(yolo_tasks, k, v)
os.environ["PYTHONPATH"] = str(WORK) + os.pathsep + os.environ.get("PYTHONPATH", "")

from ultralytics import YOLO
rows = []
for stage, ckpt in (("long", LONG), ("epoch20", FT)):
    for imgsz in (640, 768):
        print(f"\n=== {stage} | imgsz={imgsz} ===")
        metrics = YOLO(str(ckpt)).val(data=str(yaml), imgsz=imgsz, batch=16, device=0, conf=0.001, plots=False, verbose=False)
        row = {"stage": stage, "imgsz": imgsz, "ALL_mAP50": float(metrics.box.map50), "ALL_mAP50_95": float(metrics.box.map)}
        for i, name in metrics.names.items():
            row[f"{name}_mAP50"] = float(metrics.box.ap50[i])
        row["rare_avg"] = (row["Cosmetic_mAP50"] + row["Nonmetallic_Lighter_mAP50"]) / 2
        rows.append(row)
        print(f"ALL={row['ALL_mAP50']:.5f} Cos={row['Cosmetic_mAP50']:.5f} Lig={row['Nonmetallic_Lighter_mAP50']:.5f}")

print("\nPRIMARY (imgsz=640, GFLOPs≈8.9):")
for r in rows:
    if r["imgsz"] == 640:
        print(f"  {r['stage']:8s} ALL={r['ALL_mAP50']:.5f} Cos={r['Cosmetic_mAP50']:.5f} Lig={r['Nonmetallic_Lighter_mAP50']:.5f}")

out = WORK / "fair_res_matrix.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("Saved", out)
print("Paste the PRIMARY block back into chat to update thesis tables.")
