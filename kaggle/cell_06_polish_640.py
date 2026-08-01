# =============================================================================
# Kaggle — SHORT 640 POLISH (copy ENTIRE cell)
# Goal: beat YOLOv11n paper baseline ALL mAP50 = 0.808 at imgsz=640 (8.9 GFLOPs)
# Current epoch20 @640 re-val: 0.805 → need ~+0.003
#
# Inputs: HiXray_YOLO2 only (epoch20 auto-cloned from GitHub) OR add repo / .pt
# GPU T4, ~1–2 hours for 8 epochs
# =============================================================================

!pip -q install ultralytics==8.4.103 pillow==11.0.0

import os, re, sys, importlib, shutil, subprocess
from pathlib import Path

WORK = Path("/kaggle/working/polish_640")
WORK.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")
PROJECT = Path("/kaggle/working/HiXray_Training_Runs")
REPO_URL = "https://github.com/mersh01/hybrid-lightweight-xray-detection.git"
REPO_DIR = Path("/kaggle/working/hybrid-lightweight-xray-detection")

def ensure_repo():
    if not (REPO_DIR / "models" / "hybrid_rare_ft_epoch20.pt").exists():
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR, ignore_errors=True)
        print("Cloning GitHub repo for weights + modules...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    return REPO_DIR

def find_ft():
    roots = [INPUT, ensure_repo()]
    pts = []
    for root in roots:
        for p in root.rglob("*.pt"):
            if p.is_file() and p.stat().st_size > 5_000_000:
                pts.append(p)
    pts = sorted(set(pts), key=lambda p: -p.stat().st_size)
    for p in pts:
        if "epoch20" in p.name.lower() or p.name.lower() == "hybrid_rare_ft_epoch20.pt":
            return p
    big = [p for p in pts if p.stat().st_size > 20_000_000]
    if big:
        return big[0]
    found = "\n".join(f"  {p} ({p.stat().st_size/1e6:.1f} MB)" for p in pts[:15])
    raise AssertionError(f"Missing epoch20.pt. Re-run cell or add Input.\nFound:\n{found or '  (none)'}")

DATASET = next((p for p in INPUT.rglob("HiXray_YOLO2") if (p / "images" / "val").exists()), None)
if DATASET is None:
    DATASET = next((p.parent.parent for p in INPUT.rglob("images/val") if (p.parent.parent / "labels" / "val").exists()), None)
assert DATASET, "Add HiXray_YOLO2"

FT = find_ft()
print("Start checkpoint:", FT)

MOD = next(INPUT.rglob("hybrid_modules.py"), None)
if MOD is None:
    MOD = REPO_DIR / "modules" / "hybrid_modules.py"
assert MOD.exists(), "hybrid_modules.py not found"
mod_dir = MOD.parent if MOD.is_file() else MOD
for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
    shutil.copy2(mod_dir / f, WORK / f)

CLASSES = ["Cosmetic","Laptop","Mobile_Phone","Nonmetallic_Lighter","Portable_Charger_1","Portable_Charger_2","Tablet","Water"]
yaml = WORK / "dataset.yaml"
yaml.write_text(
    f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n"
    + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES)) + "\n",
    encoding="utf-8",
)

sys.path.insert(0, str(WORK))
import custom_rare_trainer  # noqa: F401 — needed to load epoch20 checkpoint
from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks
CUSTOM = {"DualConv":DualConv,"SobelConv":SobelConv,"FDDN":FDDN,"SSCAM":SSCAM,"DAPA_FPN":DAPA_FPN,"HybridBlock":HybridBlock,"DAPABlock":DAPABlock}
for k,v in CUSTOM.items(): setattr(yolo_tasks, k, v)
tasks_py = Path(yolo_tasks.__file__); t = tasks_py.read_text(encoding="utf-8")
if "from hybrid_modules import" not in t:
    t = f"import sys\nif r'{WORK}' not in sys.path:\n    sys.path.insert(0, r'{WORK}')\nfrom hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock\n" + t
if "elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv)" not in t:
    inj = ("        elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv):\n"
           "            c1, c2 = ch[f], args[0]\n            if c2 != nc:\n                c2 = make_divisible(min(c2, max_channels) * width, 8)\n            args = [c1, c2, *args[1:]]\n"
           "        elif m is SSCAM:\n            c1 = ch[f]\n            c2 = c1\n            args = [c1]\n")
    pat = re.compile(r"(^[ \t]+)else:\n[ \t]+c2 = ch\[f\]\n", re.MULTILINE)
    m = pat.search(t); assert m; t, _ = pat.subn(inj + m.group(0), t, count=1)
    tasks_py.write_text(t, encoding="utf-8")
importlib.reload(yolo_tasks)
for k,v in CUSTOM.items(): setattr(yolo_tasks, k, v)
os.environ["PYTHONPATH"] = str(WORK) + os.pathsep + os.environ.get("PYTHONPATH", "")

from ultralytics import YOLO

EPOCHS = 8
IMGSZ = 640
print("Start:", FT)
print("Target: ALL mAP50 > 0.808 @640 (paper YOLOv11n baseline)")

model = YOLO(str(FT))
model.train(
    data=str(yaml),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=16,
    device=0,
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
    project=str(PROJECT),
    name="hybrid_polish_640",
    exist_ok=True,
    seed=0,
    val=True,
    plots=False,
)

BEST = PROJECT / "hybrid_polish_640" / "weights" / "best.pt"
print("\n=== OFFICIAL RE-VAL @640 (conf=0.001) ===")
metrics = YOLO(str(BEST)).val(data=str(yaml), imgsz=640, batch=16, device=0, conf=0.001, plots=False, verbose=False)
print(f"ALL mAP50={metrics.box.map50:.5f}  (target > 0.808)")
print(f"Lighter mAP50={float(metrics.box.ap50[3]):.5f}")
for i, name in metrics.names.items():
    print(f"  {name:22s}  mAP50={float(metrics.box.ap50[i]):.5f}")
print("Best weights:", BEST)
