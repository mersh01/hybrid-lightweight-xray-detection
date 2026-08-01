# =============================================================================
# Kaggle — FAIR RESOLUTION RE-VAL (copy ENTIRE cell)
# Compares long-stage best.pt vs selected epoch20.pt at imgsz 640 AND 768
#
# Inputs: HiXray_YOLO2 only (weights auto-cloned from GitHub) OR add repo / .pt files
# Accelerator: GPU T4
# =============================================================================

!pip -q install ultralytics==8.4.103 pillow==11.0.0

import os, re, sys, importlib, shutil, csv, json, subprocess
from pathlib import Path

WORK = Path("/kaggle/working/fair_res_eval")
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
            "Missing checkpoint(s). Re-run cell (auto-clones repo) OR add .pt Inputs.\n"
            f"Found .pt files:\n{found or '  (none)'}"
        )
    return long, ft

# --- dataset ---
DATASET = None
for p in INPUT.rglob("HiXray_YOLO2"):
    if (p / "images" / "val").exists():
        DATASET = p
        break
if DATASET is None:
    for p in INPUT.rglob("images/val"):
        root = p.parent.parent
        if (root / "labels" / "val").exists():
            DATASET = root
            break
assert DATASET is not None, "Add HiXray_YOLO2 as Input"

CLASSES = [
    "Cosmetic", "Laptop", "Mobile_Phone", "Nonmetallic_Lighter",
    "Portable_Charger_1", "Portable_Charger_2", "Tablet", "Water",
]
yaml = WORK / "dataset.yaml"
yaml.write_text(
    f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n"
    + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    + "\n",
    encoding="utf-8",
)

# --- checkpoints ---
LONG, FT = find_long_and_ft()

print("DATASET", DATASET)
print("LONG   ", LONG, f"({LONG.stat().st_size/1e6:.1f} MB)")
print("FT     ", FT, f"({FT.stat().st_size/1e6:.1f} MB)")

# --- modules from repo input or cloned repo ---
MOD_SRC = None
for p in INPUT.rglob("hybrid_modules.py"):
    MOD_SRC = p.parent
    break
if MOD_SRC is None and (REPO_DIR / "modules" / "hybrid_modules.py").exists():
    MOD_SRC = REPO_DIR / "modules"
if MOD_SRC is not None:
    for f in ("hybrid_modules.py", "custom_rare_trainer.py"):
        if (MOD_SRC / f).exists():
            shutil.copy2(MOD_SRC / f, WORK / f)

if not (WORK / "hybrid_modules.py").exists():
    (WORK / "hybrid_modules.py").write_text(r'''
from __future__ import annotations
import torch
import torch.nn as nn
class DualConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.gconv = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pconv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn2(self.pconv(self.act(self.bn1(self.gconv(x))))))
class SobelConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        with torch.no_grad():
            for i in range(min(c2, c1)):
                self.conv.weight[i, i % c1] = sobel_x if i % 2 == 0 else sobel_y
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
class FDDN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.low_path = DualConv(c1, c2 // 2)
        self.high_path = SobelConv(c1, c2 // 2)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2 // 4, 1), nn.SiLU(), nn.Conv2d(c2 // 4, c2, 1), nn.Sigmoid())
    def forward(self, x):
        low = self.pool(x); high = x - low
        feat = torch.cat([self.low_path(low), self.high_path(high)], dim=1)
        return feat * self.se(feat)
class SSCAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        mid = max(c1 // reduction, 4)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c1, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(mid, c1, 1, bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        out = x * self.ca(x)
        avg = torch.mean(out, dim=1, keepdim=True); mx, _ = torch.max(out, dim=1, keepdim=True)
        return out * self.sa(torch.cat([avg, mx], dim=1))
class DAPA_FPN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fg_branch = nn.Sequential(DualConv(c1, c2), SSCAM(c2))
        self.bg_gate = nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.Sigmoid())
        self.fuse = DualConv(c2, c2)
    def forward(self, x):
        return self.fuse(self.fg_branch(x) * (1.0 + self.bg_gate(x)))
class HybridBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.block = nn.Sequential(FDDN(c1, c2), SSCAM(c2))
    def forward(self, x):
        return self.block(x)
class DAPABlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.block = DAPA_FPN(c1, c2)
    def forward(self, x):
        return self.block(x)
''', encoding="utf-8")

if not (WORK / "custom_rare_trainer.py").exists():
    (WORK / "custom_rare_trainer.py").write_text(
        "from ultralytics.models.yolo.detect import DetectionTrainer\n"
        "from ultralytics.nn.tasks import DetectionModel\n"
        "from ultralytics.utils.loss import v8DetectionLoss\n"
        "import torch\nimport torch.nn as nn\n"
        "RARE_CLS_WEIGHTS = [1.5, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]\n"
        "class WeightedDetectionLoss(v8DetectionLoss):\n"
        "    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):\n"
        "        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)\n"
        "        if class_weights is not None:\n"
        "            self.bce = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device), reduction='none')\n"
        "class WeightedDetectionModel(DetectionModel):\n"
        "    def init_criterion(self):\n"
        "        return WeightedDetectionLoss(self, class_weights=torch.tensor(RARE_CLS_WEIGHTS, dtype=torch.float32))\n"
        "class RareClassTrainer(DetectionTrainer):\n"
        "    pass\n",
        encoding="utf-8",
    )


def register_modules():
    sys.path.insert(0, str(WORK))
    for name in list(sys.modules):
        if name in ("hybrid_modules", "custom_rare_trainer") or name.startswith("hybrid_modules.") or name.startswith("custom_rare_trainer."):
            del sys.modules[name]
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
        assert m, "Pin ultralytics==8.4.103"
        t, nsub = pat.subn(injection + m.group(0), t, count=1)
        assert nsub == 1
        tasks_py.write_text(t, encoding="utf-8")
    importlib.reload(yolo_tasks)
    for k, v in CUSTOM.items():
        setattr(yolo_tasks, k, v)
    os.environ["PYTHONPATH"] = str(WORK) + os.pathsep + os.environ.get("PYTHONPATH", "")


register_modules()
from ultralytics import YOLO

rows = []
jobs = [
    ("long", LONG, 640),
    ("long", LONG, 768),
    ("epoch20", FT, 640),
    ("epoch20", FT, 768),
]
for stage, ckpt, imgsz in jobs:
    print(f"\n=== {stage} | {ckpt.name} | imgsz={imgsz} ===")
    register_modules()
    model = YOLO(str(ckpt))
    metrics = model.val(
        data=str(yaml), imgsz=imgsz, batch=16, device=0,
        conf=0.001, plots=False, verbose=False,
    )
    row = {
        "stage": stage,
        "weights": ckpt.name,
        "imgsz": imgsz,
        "ALL_mAP50": float(metrics.box.map50),
        "ALL_mAP50_95": float(metrics.box.map),
    }
    for i, name in metrics.names.items():
        row[f"{name}_mAP50"] = float(metrics.box.ap50[i])
    row["rare_avg_mAP50"] = (row["Cosmetic_mAP50"] + row["Nonmetallic_Lighter_mAP50"]) / 2.0
    rows.append(row)
    print(
        f"ALL={row['ALL_mAP50']:.5f}  Cos={row['Cosmetic_mAP50']:.5f}  "
        f"Lig={row['Nonmetallic_Lighter_mAP50']:.5f}  rare={row['rare_avg_mAP50']:.5f}"
    )

csv_path = WORK / "fair_res_matrix.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
(WORK / "fair_res_matrix.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

print("\n" + "=" * 72)
print("FAIR RESOLUTION MATRIX (mAP50) — use this for thesis attribution")
print("=" * 72)
print(f"{'stage':10s} {'imgsz':>6s} {'ALL':>8s} {'Cos':>8s} {'Lighter':>8s} {'rare':>8s}")
for r in rows:
    print(
        f"{r['stage']:10s} {r['imgsz']:6d} {r['ALL_mAP50']:8.5f} "
        f"{r['Cosmetic_mAP50']:8.5f} {r['Nonmetallic_Lighter_mAP50']:8.5f} {r['rare_avg_mAP50']:8.5f}"
    )

by = {(r["stage"], r["imgsz"]): r for r in rows}
print("\nDELTA at matched imgsz (epoch20 − long) — fine-tune only:")
for imgsz in (640, 768):
    a, b = by[("long", imgsz)], by[("epoch20", imgsz)]
    print(
        f"  imgsz={imgsz}: ALL {b['ALL_mAP50']-a['ALL_mAP50']:+.5f}  "
        f"Cos {b['Cosmetic_mAP50']-a['Cosmetic_mAP50']:+.5f}  "
        f"Lig {b['Nonmetallic_Lighter_mAP50']-a['Nonmetallic_Lighter_mAP50']:+.5f}"
    )
print("\nSaved:", csv_path)
print("Selected fine-tuned checkpoint remains: epoch20 / hybrid_rare_ft_epoch20.pt")
