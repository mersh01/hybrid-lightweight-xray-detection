# =============================================================================
# Kaggle FAST re-eval — epoch20 + long-stage at imgsz=640 (and 768)
# Copy ENTIRE cell. GPU T4. ~15–25 min for all 4 runs.
# Inputs: HiXray_YOLO2 only (repo weights auto-cloned) OR add GitHub repo / .pt files
# =============================================================================

!pip -q install ultralytics==8.4.103 pillow==11.0.0

import os, re, sys, importlib, shutil, csv, json, subprocess
from pathlib import Path

WORK = Path("/kaggle/working/reeval_640")
WORK.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")
REPO_URL = "https://github.com/mersh01/hybrid-lightweight-xray-detection.git"
REPO_DIR = Path("/kaggle/working/hybrid-lightweight-xray-detection")

def ensure_repo():
    if not (REPO_DIR / "models" / "hybrid_rare_ft_epoch20.pt").exists():
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR, ignore_errors=True)
        print("Cloning GitHub repo for weights + modules...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    return REPO_DIR

def list_pts():
    roots = [INPUT, ensure_repo()]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            if p.is_file() and p.stat().st_size > 5_000_000:
                out.append(p)
    return sorted(set(out), key=lambda p: -p.stat().st_size)

def find_long_and_ft():
    pts = list_pts()
    ft = long = None
    for p in pts:
        if p.name.lower() in ("hybrid_rare_ft_epoch20.pt", "epoch20.pt") or "epoch20" in p.name.lower():
            ft = p
            break
    if ft is None:
        big = [p for p in pts if p.stat().st_size > 20_000_000]
        if big:
            ft = big[0]
    for p in pts:
        if p == ft:
            continue
        if p.name.lower() == "hybrid_fdd_long_best.pt":
            long = p
            break
    if long is None:
        for p in pts:
            if p == ft or "epoch20" in p.name.lower():
                continue
            if 8_000_000 < p.stat().st_size < 20_000_000:
                long = p
                break
    if long is None:
        for p in pts:
            if p != ft and "best" in p.name.lower():
                long = p
                break
    if not (long and ft):
        found = "\n".join(f"  {p} ({p.stat().st_size/1e6:.1f} MB)" for p in pts[:20])
        raise AssertionError(
            "Missing checkpoint(s). Re-run cell (auto-clones repo) OR add Inputs:\n"
            "  models/hybrid_fdd_long_best.pt + models/hybrid_rare_ft_epoch20.pt\n"
            f"Found .pt files:\n{found or '  (none)'}"
        )
    return long, ft

DATASET = next((p for p in INPUT.rglob("HiXray_YOLO2") if (p / "images" / "val").exists()), None)
if DATASET is None:
    DATASET = next((p.parent.parent for p in INPUT.rglob("images/val") if (p.parent.parent / "labels" / "val").exists()), None)
assert DATASET, "Add HiXray_YOLO2 as Kaggle Input"

CLASSES = ["Cosmetic","Laptop","Mobile_Phone","Nonmetallic_Lighter","Portable_Charger_1","Portable_Charger_2","Tablet","Water"]
yaml = WORK / "dataset.yaml"
yaml.write_text(f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n" + "\n".join(f"  {i}: {n}" for i,n in enumerate(CLASSES)) + "\n", encoding="utf-8")

LONG, FT = find_long_and_ft()
print("LONG", LONG, f"({LONG.stat().st_size/1e6:.1f} MB)")
print("FT  ", FT, f"({FT.stat().st_size/1e6:.1f} MB)")

MOD = next(INPUT.rglob("hybrid_modules.py"), None)
if MOD is None:
    MOD = REPO_DIR / "modules" / "hybrid_modules.py"
assert MOD.exists(), f"hybrid_modules.py not found — clone failed? {REPO_DIR}"
mod_dir = MOD.parent if MOD.is_file() else MOD
for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
    shutil.copy2(mod_dir / f, WORK / f)

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
