# =============================================================================
# Hybrid FDD rare-class FINE-TUNE ROUND 2
# Start from: epoch20.pt (round-1 peak)
# Stronger rare push:
#   weights: Cos=2.0, Lighter=5.0, others=1.0
#   oversample: Cos×3, Lighter×10
#   imgsz=896, epochs=25, lr0=5e-5, patience=10
#
# Kaggle Inputs:
#   1) HiXray_YOLO2
#   2) epoch20.pt from hybrid_fdd_rare_ft (round 1)
# Accelerator: GPU T4 x2
# Run cells 0 → 6
# =============================================================================

CELL_0 = r'''
# Cell 0 — install
!pip -q install ultralytics==8.4.103 pillow==11.0.0
import torch, ultralytics
print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
print('ultralytics', ultralytics.__version__)
assert torch.cuda.is_available(), 'Enable GPU T4 x2'
print('GPU0:', torch.cuda.get_device_name(0))
if torch.cuda.device_count() >= 2:
    print('GPU1:', torch.cuda.get_device_name(1))
'''

CELL_1 = r'''
# Cell 1 — paths: HiXray + epoch20.pt (round-1 peak)
import os
from pathlib import Path

WORK_DIR = '/kaggle/working/hybrid_fdd_rare_ft2'
PROJECT_DIR = '/kaggle/working/HiXray_Training_Runs'
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

INPUT_ROOT = Path('/kaggle/input')

def find_hixray_root():
    for rel in (
        'datasets/maraolfeye/hixray-yolo/HiXray_YOLO2',
        'datasets/gosaye/hixray-yolo/HiXray_YOLO2',
        'hixray-yolo/HiXray_YOLO2',
        'HiXray_YOLO2',
        'HiXray_YOLO',
    ):
        root = INPUT_ROOT / rel
        if (root / 'images' / 'train').is_dir() and (root / 'labels' / 'train').is_dir():
            return str(root)
    for p in INPUT_ROOT.rglob('images/train'):
        if p.is_dir() and (p.parent.parent / 'labels' / 'train').is_dir():
            return str(p.parent.parent)
    return None

DATASET_PATH = find_hixray_root()
assert DATASET_PATH, 'Add HiXray_YOLO2 to Inputs'

pts = [p for p in INPUT_ROOT.rglob('*.pt') if p.stat().st_size > 5_000_000]
# Prefer epoch20.pt from round-1
pts.sort(key=lambda p: (
    0 if 'epoch20' in p.name.lower() else 1,
    0 if 'rare_ft' in str(p).lower() else 1,
    -p.stat().st_size,
))
assert pts, 'Add epoch20.pt from round-1 fine-tune'
START_CKPT = str(pts[0])

print('WORK_DIR', WORK_DIR)
print('DATASET_PATH', DATASET_PATH)
print('START_CKPT', START_CKPT, f'({Path(START_CKPT).stat().st_size/1e6:.1f} MB)')
print('train OK', (Path(DATASET_PATH) / 'images' / 'train').is_dir())
print('val OK', (Path(DATASET_PATH) / 'images' / 'val').is_dir())
'''

CELL_2 = r'''
# Cell 2 — stronger oversample Cos×3, Lighter×10
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm

WORK_DIR = '/kaggle/working/hybrid_fdd_rare_ft2'
CLASSES = [
    'Cosmetic', 'Laptop', 'Mobile_Phone', 'Nonmetallic_Lighter',
    'Portable_Charger_1', 'Portable_Charger_2', 'Tablet', 'Water',
]
OVERSAMPLE = {0: 3, 3: 10}  # Cos×3, Lighter×10
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

dataset_dir = Path(DATASET_PATH)
train_img_dir = dataset_dir / 'images' / 'train'
train_lbl_dir = dataset_dir / 'labels' / 'train'

img_index = {}
for img in tqdm(list(train_img_dir.iterdir()), desc='Index train images'):
    if img.suffix.lower() in IMG_EXTS:
        img_index[img.stem] = (train_img_dir / img.name).as_posix()

train_entries = []
class_counter = Counter()
focus_images = {0: 0, 3: 0}

for lbl in tqdm(sorted(train_lbl_dir.glob('*.txt')), desc='Build rare-class list'):
    abs_path = img_index.get(lbl.stem)
    if abs_path is None:
        continue
    classes_in_image = set()
    for ln in lbl.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 5:
            cid = int(parts[0])
            classes_in_image.add(cid)
            class_counter[cid] += 1
    if not classes_in_image:
        continue
    repeats = 1
    for cid, factor in OVERSAMPLE.items():
        if cid in classes_in_image:
            repeats = max(repeats, factor)
    for cid in (0, 3):
        if cid in classes_in_image:
            focus_images[cid] += 1
    train_entries.extend([abs_path] * repeats)

assert train_entries, 'No train entries'

TRAIN_TXT = Path(WORK_DIR) / 'train_rare_ft2.txt'
TRAIN_TXT.write_text('\n'.join(train_entries) + '\n', encoding='utf-8')

DATA_YAML = str(Path(WORK_DIR) / 'dataset.yaml')
names_block = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES))
Path(DATA_YAML).write_text(
    f"# FDD rare FT2 (Cosmetic×3, Lighter×10)\n"
    f"path: {dataset_dir.as_posix()}\n"
    f"train: {TRAIN_TXT.as_posix()}\n"
    f"val: images/val\n"
    f"nc: {len(CLASSES)}\n"
    f"names:\n{names_block}\n",
    encoding='utf-8',
)

print('dataset.yaml ->', DATA_YAML)
print('Original train images:', len(img_index))
print('Train entries after oversample:', len(train_entries))
print('Images with Cosmetic:', focus_images[0], '| Lighter:', focus_images[3])
for i, n in enumerate(CLASSES):
    print(f'  {i} {n}: {class_counter[i]}')
'''

CELL_3 = r'''
# Cell 3 — write hybrid_modules.py
from pathlib import Path

WORK_DIR = '/kaggle/working/hybrid_fdd_rare_ft2'
Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

MODULES = r"""
from __future__ import annotations
import torch
import torch.nn as nn

class DualConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
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
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1), nn.SiLU(),
            nn.Conv2d(c2 // 4, c2, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        low = self.pool(x)
        high = x - low
        feat = torch.cat([self.low_path(low), self.high_path(high)], dim=1)
        return feat * self.se(feat)

class SSCAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        mid = max(c1 // reduction, 4)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c1, 1, bias=False), nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        out = x * self.ca(x)
        avg = torch.mean(out, dim=1, keepdim=True)
        mx, _ = torch.max(out, dim=1, keepdim=True)
        return out * self.sa(torch.cat([avg, mx], dim=1))

class DAPA_FPN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fg_branch = nn.Sequential(DualConv(c1, c2), SSCAM(c2))
        self.bg_gate = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.Sigmoid(),
        )
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
"""

p = Path(WORK_DIR) / 'hybrid_modules.py'
p.write_text(MODULES, encoding='utf-8')
print('Wrote', p)
'''

CELL_4 = r'''
# Cell 4 — write hybrid_yolo.yaml
from pathlib import Path

WORK_DIR = '/kaggle/working/hybrid_fdd_rare_ft2'
YAML_LINES = [
    '# Hybrid Lightweight X-Ray Detector',
    'nc: 8',
    'scales:',
    '  n: [0.33, 0.25, 1024]',
    '',
    'backbone:',
    '  - [-1, 1, Conv,         [64, 3, 2]]',
    '  - [-1, 1, Conv,         [128, 3, 2]]',
    '  - [-1, 3, C2f,          [128, True]]',
    '  - [-1, 1, Conv,         [256, 3, 2]]',
    '  - [-1, 4, C2f,          [256, True]]',
    '  - [-1, 1, HybridBlock,  [256]]',
    '  - [-1, 1, Conv,         [512, 3, 2]]',
    '  - [-1, 4, C2f,          [512, True]]',
    '  - [-1, 1, HybridBlock,  [512]]',
    '  - [-1, 1, Conv,         [1024, 3, 2]]',
    '  - [-1, 3, C2f,          [1024, True]]',
    '  - [-1, 1, SPPF,         [1024, 5]]',
    '  - [-1, 1, HybridBlock,  [1024]]',
    '',
    'head:',
    "  - [-1, 1, nn.Upsample,  [None, 2, 'nearest']]",
    '  - [[-1, 8], 1, Concat,  [1]]',
    '  - [-1, 3, C2f,          [512]]',
    '  - [-1, 1, DAPABlock,    [512]]',
    "  - [-1, 1, nn.Upsample,  [None, 2, 'nearest']]",
    '  - [[-1, 5], 1, Concat,  [1]]',
    '  - [-1, 3, C2f,          [256]]',
    '  - [-1, 1, DAPABlock,    [256]]',
    '  - [-1, 1, DualConv,     [256, 3, 2]]',
    '  - [[-1, 16], 1, Concat, [1]]',
    '  - [-1, 3, C2f,          [512]]',
    '  - [-1, 1, DAPABlock,    [512]]',
    '  - [-1, 1, DualConv,     [512, 3, 2]]',
    '  - [[-1, 12], 1, Concat, [1]]',
    '  - [-1, 3, C2f,          [1024]]',
    '  - [-1, 1, SSCAM,        []]',
    '  - [[20, 24, 28], 1, Detect, [nc]]',
]
yp = Path(WORK_DIR) / 'hybrid_yolo.yaml'
yp.write_text('\n'.join(YAML_LINES) + '\n', encoding='utf-8')
print('Wrote', yp)
'''

CELL_5 = r'''
# Cell 5 — FT2 train: Cos=2.0 Lighter=5.0, imgsz=896, 25 ep
import os, re, sys, importlib, shutil, subprocess
from pathlib import Path
import torch

WORK_DIR = '/kaggle/working/hybrid_fdd_rare_ft2'
PROJECT_DIR = '/kaggle/working/HiXray_Training_Runs'
DATA_YAML = f'{WORK_DIR}/dataset.yaml'
RUN_NAME = 'hybrid_fdd_rare_ft2'

INPUT_ROOT = Path('/kaggle/input')
pts = [p for p in INPUT_ROOT.rglob('*.pt') if p.stat().st_size > 5_000_000]
pts.sort(key=lambda p: (0 if 'epoch20' in p.name.lower() else 1, -p.stat().st_size))
START_CKPT = str(pts[0])
assert Path(START_CKPT).exists(), 'epoch20.pt not found'
print('FT2 from:', START_CKPT)

sys.path.insert(0, WORK_DIR)
for name in list(sys.modules):
    if name == 'hybrid_modules' or name.startswith('hybrid_modules.'):
        del sys.modules[name]
for name in list(sys.modules):
    if name == 'custom_rare_trainer' or name.startswith('custom_rare_trainer.'):
        del sys.modules[name]

from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks

CUSTOM = {
    'DualConv': DualConv, 'SobelConv': SobelConv, 'FDDN': FDDN, 'SSCAM': SSCAM,
    'DAPA_FPN': DAPA_FPN, 'HybridBlock': HybridBlock, 'DAPABlock': DAPABlock,
}
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)

tasks_py = Path(yolo_tasks.__file__)
t = tasks_py.read_text(encoding='utf-8')
imp = (
    'import sys\n'
    f"if r'{WORK_DIR}' not in sys.path:\n"
    f"    sys.path.insert(0, r'{WORK_DIR}')\n"
    'from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock\n'
)
if 'from hybrid_modules import' not in t:
    t = imp + t
    print('Injected hybrid_modules import')

MARKER = 'elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv)'
if MARKER not in t:
    injection = (
        '        elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv):\n'
        '            c1, c2 = ch[f], args[0]\n'
        '            if c2 != nc:\n'
        '                c2 = make_divisible(min(c2, max_channels) * width, 8)\n'
        '            args = [c1, c2, *args[1:]]\n'
        '        elif m is SSCAM:\n'
        '            c1 = ch[f]\n'
        '            c2 = c1\n'
        '            args = [c1]\n'
    )
    pat = re.compile(r'(^[ \t]+)else:\n[ \t]+c2 = ch\[f\]\n', re.MULTILINE)
    m = pat.search(t)
    if not m:
        raise RuntimeError('Could not find parse_model else:c2=ch[f]')
    t, nsub = pat.subn(injection + m.group(0), t, count=1)
    assert nsub == 1
    print('Applied channel patch')
else:
    print('Channel patch already present')

tasks_py.write_text(t, encoding='utf-8')
importlib.reload(yolo_tasks)
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)

# Cos=2.0, Lighter=5.0, others=1.0
TRAINER_PY = Path(WORK_DIR) / 'custom_rare_trainer.py'
TRAINER_PY.write_text(
    'from __future__ import annotations\n'
    'import torch\n'
    'import torch.nn as nn\n'
    'from ultralytics.models.yolo.detect import DetectionTrainer\n'
    'from ultralytics.nn.tasks import DetectionModel\n'
    'from ultralytics.utils import RANK\n'
    'from ultralytics.utils.loss import v8DetectionLoss\n'
    '\n'
    'RARE_CLS_WEIGHTS = [2.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]\n'
    'COSMETIC_ID = 0\n'
    'LIGHTER_ID = 3\n'
    '\n'
    '\n'
    'class WeightedDetectionLoss(v8DetectionLoss):\n'
    '    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):\n'
    '        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)\n'
    '        if class_weights is not None:\n'
    '            self.bce = nn.BCEWithLogitsLoss(\n'
    '                pos_weight=class_weights.to(self.device),\n'
    '                reduction="none",\n'
    '            )\n'
    '\n'
    '\n'
    'class WeightedDetectionModel(DetectionModel):\n'
    '    def init_criterion(self):\n'
    '        w = torch.tensor(RARE_CLS_WEIGHTS, dtype=torch.float32)\n'
    '        return WeightedDetectionLoss(self, class_weights=w)\n'
    '\n'
    '\n'
    'class RareClassTrainer(DetectionTrainer):\n'
    '    def get_model(self, cfg=None, weights=None, verbose=True):\n'
    '        model = WeightedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})\n'
    '        if weights:\n'
    '            model.load(weights)\n'
    '        return model\n'
    '\n'
    '    def validate(self):\n'
    '        metrics, fitness = super().validate()\n'
    '        maps50 = None\n'
    '        validator = getattr(self, "validator", None)\n'
    '        if validator is not None and getattr(validator, "metrics", None) is not None:\n'
    '            box = validator.metrics.box\n'
    '            if hasattr(box, "maps50") and box.maps50 is not None:\n'
    '                maps50 = box.maps50\n'
    '            elif hasattr(box, "ap50") and box.ap50 is not None:\n'
    '                maps50 = box.ap50\n'
    '        if maps50 is not None and len(maps50) > LIGHTER_ID:\n'
    '            cos = float(maps50[COSMETIC_ID])\n'
    '            lig = float(maps50[LIGHTER_ID])\n'
    '            fitness = (cos + lig) / 2.0\n'
    '            print(f"Rare-class fitness: {fitness:.5f}  Cosmetic={cos:.5f}  Lighter={lig:.5f}")\n'
    '        elif isinstance(metrics, dict):\n'
    '            fitness = float(metrics.get("metrics/mAP50(B)", fitness))\n'
    '        return metrics, fitness\n',
    encoding='utf-8',
)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, WORK_DIR)
importlib.import_module('custom_rare_trainer')
from custom_rare_trainer import RareClassTrainer
print('Wrote', TRAINER_PY)

_prev_pp = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = WORK_DIR if not _prev_pp else f'{WORK_DIR}{os.pathsep}{_prev_pp}'
subprocess.run(
    [sys.executable, '-c', 'from custom_rare_trainer import RareClassTrainer; print("DDP import OK")'],
    env=os.environ,
    check=True,
)

ckpt_local = Path(WORK_DIR) / 'start_epoch20.pt'
shutil.copy2(START_CKPT, ckpt_local)

from ultralytics import YOLO
model = YOLO(str(ckpt_local))
print('Loaded epoch20.pt OK')

n_gpu = torch.cuda.device_count()
device = '0,1' if n_gpu >= 2 else '0'
EPOCHS = 25
IMGSZ = 896
BATCH = 10 if n_gpu >= 2 else 4  # lower if OOM at 896

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=device,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    resume=False,
    optimizer='AdamW',
    lr0=0.00005,
    lrf=0.01,
    warmup_epochs=2,
    patience=10,
    save=True,
    save_period=5,
    plots=False,
    workers=8,
    cls=1.0,
    copy_paste=0.5,
    close_mosaic=8,
    multi_scale=0.5,
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.3,
    degrees=5.0,
    translate=0.1,
    scale=0.9,
    fliplr=0.5,
    mosaic=0.5,
    trainer=RareClassTrainer,
)
print('Done FT2. EPOCHS=', EPOCHS, 'imgsz=', IMGSZ)
print('cls weights: Cos=2.0 Lighter=5.0 others=1.0')
print('Target: Lighter > 0.086 | keep ALL >= ~0.80')

WEIGHTS_DIR = Path(PROJECT_DIR) / RUN_NAME / 'weights'
print('Weights:', sorted(p.name for p in WEIGHTS_DIR.glob('*.pt')) if WEIGHTS_DIR.exists() else 'NONE')
'''

CELL_6 = r'''
# Cell 6 — per-class eval (prefer best.pt, else last / highest epoch*.pt)
import sys
from pathlib import Path

WORK_DIR = '/kaggle/working/hybrid_fdd_rare_ft2'
PROJECT_DIR = '/kaggle/working/HiXray_Training_Runs'
DATA_YAML = f'{WORK_DIR}/dataset.yaml'
RUN_NAME = 'hybrid_fdd_rare_ft2'
IMGSZ = 896
WEIGHTS_DIR = Path(PROJECT_DIR) / RUN_NAME / 'weights'
CLASSES = [
    'Cosmetic', 'Laptop', 'Mobile_Phone', 'Nonmetallic_Lighter',
    'Portable_Charger_1', 'Portable_Charger_2', 'Tablet', 'Water',
]

sys.path.insert(0, WORK_DIR)
# ensure trainer stub exists for unpickle
(Path(WORK_DIR) / 'custom_rare_trainer.py').write_text(
    'from ultralytics.models.yolo.detect import DetectionTrainer\n'
    'from ultralytics.nn.tasks import DetectionModel\n'
    'from ultralytics.utils.loss import v8DetectionLoss\n'
    'import torch, torch.nn as nn\n'
    'RARE_CLS_WEIGHTS = [2.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]\n'
    'class WeightedDetectionLoss(v8DetectionLoss):\n'
    '    def __init__(self, model, class_weights=None, tal_topk=10, tal_topk2=None):\n'
    '        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)\n'
    '        if class_weights is not None:\n'
    '            self.bce = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device), reduction="none")\n'
    'class WeightedDetectionModel(DetectionModel):\n'
    '    def init_criterion(self):\n'
    '        return WeightedDetectionLoss(self, class_weights=torch.tensor(RARE_CLS_WEIGHTS, dtype=torch.float32))\n'
    'class RareClassTrainer(DetectionTrainer):\n'
    '    pass\n',
    encoding='utf-8',
)
import custom_rare_trainer  # noqa: F401

from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks
for k, v in {
    'DualConv': DualConv, 'SobelConv': SobelConv, 'FDDN': FDDN, 'SSCAM': SSCAM,
    'DAPA_FPN': DAPA_FPN, 'HybridBlock': HybridBlock, 'DAPABlock': DAPABlock,
}.items():
    setattr(yolo_tasks, k, v)

from ultralytics import YOLO

ckpt = WEIGHTS_DIR / 'best.pt'
if not ckpt.exists():
    epochs = sorted(WEIGHTS_DIR.glob('epoch*.pt'), key=lambda p: int(p.stem.replace('epoch', '') or '0'))
    ckpt = epochs[-1] if epochs else (WEIGHTS_DIR / 'last.pt')
assert ckpt.exists(), f'No checkpoint in {WEIGHTS_DIR}'
print('Evaluating:', ckpt)
print('Available:', sorted(p.name for p in WEIGHTS_DIR.glob('*.pt')))

model = YOLO(str(ckpt))
metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, batch=8, conf=0.001, iou=0.5, device=0, plots=False)

print(f'\nALL mAP50={metrics.box.map50:.5f} mAP50-95={metrics.box.map:.5f} '
      f'P={metrics.box.mp:.5f} R={metrics.box.mr:.5f}')
maps50 = metrics.box.ap50
for i, name in enumerate(CLASSES):
    print(f'  {name:22s} mAP50={float(maps50[i]):.5f}')
rare = (float(maps50[0]) + float(maps50[3])) / 2.0
print(f'\nRare avg (Cos+Lighter)/2 = {rare:.5f}')
print('Compare to FT1 epoch20: ALL~0.809 Cos~0.665 Lighter~0.086')
'''

if __name__ == '__main__':
    from pathlib import Path
    out = Path(__file__).with_name('fdd_rare_finetune2_CELLS.txt')
    parts = []
    for i, c in enumerate([CELL_0, CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6]):
        parts.append(f"{'='*20} CELL {i} {'='*20}\n{c.strip()}\n")
    out.write_text('\n'.join(parts), encoding='utf-8')
    print('Wrote', out)
