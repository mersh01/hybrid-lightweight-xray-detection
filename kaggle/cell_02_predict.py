# =============================================================================
# Kaggle — COPY THIS ENTIRE CELL (after cell_01_eval OR standalone)
# Inputs: HiXray images (optional) + hybrid_rare_ft_epoch20.pt / cloned repo
# What it does: run inference on a few val images and save overlays
# =============================================================================

!pip -q install ultralytics==8.4.103 pillow==11.0.0

import os, re, sys, importlib, shutil
from pathlib import Path

WORK = Path("/kaggle/working/hybrid_predict")
WORK.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")

DATASET = None
for p in INPUT.rglob("images/val"):
    DATASET = p
    break
assert DATASET is not None, "Add HiXray_YOLO2 with images/val"

CKPT = None
for name in ("hybrid_rare_ft_epoch20.pt", "epoch20.pt"):
    hits = sorted(INPUT.rglob(name), key=lambda x: -x.stat().st_size)
    if hits:
        CKPT = hits[0]
        break
assert CKPT is not None, "Add hybrid_rare_ft_epoch20.pt"

MOD_SRC = None
for p in INPUT.rglob("hybrid_modules.py"):
    MOD_SRC = p.parent
    break
if MOD_SRC is not None:
    for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
        if (MOD_SRC / f).exists():
            shutil.copy2(MOD_SRC / f, WORK / f)

# If modules missing, re-run cell_01_eval first (it writes them), or paste modules from repo.
assert (WORK / "hybrid_modules.py").exists() or (Path("/kaggle/working/hybrid_eval") / "hybrid_modules.py").exists(), \
    "Run kaggle/cell_01_eval.py first, or add the GitHub repo as Input"

if not (WORK / "hybrid_modules.py").exists():
    for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
        shutil.copy2(Path("/kaggle/working/hybrid_eval") / f, WORK / f)

sys.path.insert(0, str(WORK))
import custom_rare_trainer  # noqa: F401
from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks
CUSTOM = {
    "DualConv": DualConv, "SobelConv": SobelConv, "FDDN": FDDN, "SSCAM": SSCAM,
    "DAPA_FPN": DAPA_FPN, "HybridBlock": HybridBlock, "DAPABlock": DAPABlock,
}
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)

tasks_py = Path(yolo_tasks.__file__)
t = tasks_py.read_text(encoding="utf-8")
if "from hybrid_modules import" not in t:
    t = (
        "import sys\n"
        f"if r'{WORK}' not in sys.path:\n"
        f"    sys.path.insert(0, r'{WORK}')\n"
        "from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock\n"
        + t
    )
MARKER = "elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv)"
if MARKER not in t:
    injection = (
        "        elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv):\n"
        "            c1, c2 = ch[f], args[0]\n"
        "            if c2 != nc:\n"
        "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
        "            args = [c1, c2, *args[1:]]\n"
        "        elif m is SSCAM:\n"
        "            c1 = ch[f]\n"
        "            c2 = c1\n"
        "            args = [c1]\n"
    )
    pat = re.compile(r"(^[ \t]+)else:\n[ \t]+c2 = ch\[f\]\n", re.MULTILINE)
    m = pat.search(t)
    assert m
    t, _ = pat.subn(injection + m.group(0), t, count=1)
tasks_py.write_text(t, encoding="utf-8")
importlib.reload(yolo_tasks)
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)
os.environ["PYTHONPATH"] = str(WORK) + os.pathsep + os.environ.get("PYTHONPATH", "")

from ultralytics import YOLO
model = YOLO(str(CKPT))
# Predict on first 8 val images
sources = sorted(DATASET.glob("*.*"))[:8]
results = model.predict(source=sources, imgsz=768, conf=0.25, device=0, project=str(WORK), name="preds", save=True, exist_ok=True)
print("Saved →", results[0].save_dir if results else WORK / "preds")
