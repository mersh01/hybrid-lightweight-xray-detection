# =============================================================================
# Kaggle — COMPLEXITY @ 640 and 768 (copy ENTIRE cell)
# Inputs: GitHub repo (or either .pt under models/)
# No dataset needed. CPU or GPU OK.
# =============================================================================

!pip -q install ultralytics==8.4.103 pillow==11.0.0

import os, re, sys, importlib, shutil, csv, json
from pathlib import Path

WORK = Path("/kaggle/working/complexity")
WORK.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")

pts = {}
for name in ("hybrid_fdd_long_best.pt", "hybrid_rare_ft_epoch20.pt", "epoch20.pt", "best.pt"):
    hits = [p for p in INPUT.rglob(name) if p.stat().st_size > 5_000_000]
    if hits and name not in pts:
        pts[name] = sorted(hits, key=lambda p: -p.stat().st_size)[0]

LONG = pts.get("hybrid_fdd_long_best.pt") or pts.get("best.pt")
FT = pts.get("hybrid_rare_ft_epoch20.pt") or pts.get("epoch20.pt")
assert LONG is not None and FT is not None, "Add long best.pt and epoch20.pt (or clone repo)"

MOD_SRC = next(INPUT.rglob("hybrid_modules.py"), None)
if MOD_SRC is not None:
    for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
        src = MOD_SRC.parent / f
        if src.exists():
            shutil.copy2(src, WORK / f)

assert (WORK / "hybrid_modules.py").exists(), "Add repo modules/ as Input"
assert (WORK / "custom_rare_trainer.py").exists()

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

rows = []
for label, ckpt in (("long", LONG), ("epoch20", FT)):
    for imgsz in (640, 768):
        model = YOLO(str(ckpt))
        info = model.info(detailed=False, verbose=True, imgsz=imgsz)
        n_l, n_p, n_g, flops = info[0], info[1], info[2], info[3]
        row = {"stage": label, "weights": ckpt.name, "imgsz": imgsz,
               "layers": int(n_l), "params": int(n_p), "GFLOPs": float(flops)}
        rows.append(row)
        print(f"{label:8s} imgsz={imgsz}  params={n_p:,}  GFLOPs={flops:.3f}")

print("\nCOMPLEXITY MATRIX")
print(f"{'stage':10s} {'imgsz':>6s} {'params':>12s} {'GFLOPs':>8s}")
for r in rows:
    print(f"{r['stage']:10s} {r['imgsz']:6d} {r['params']:12,d} {r['GFLOPs']:8.3f}")

out = WORK / "complexity_matrix.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("Saved", out)
