# =============================================================================
# Hybrid architecture + FDD-YOLO training protocol (fair comparison)
# =============================================================================
# Protocol (match FDD paper rebuild):
#   epochs=300, imgsz=640, batch=16, SGD lr0=0.01 momentum=0.9, cos_lr=True
#   Standard HiXray split — NO rare-class oversampling
#   GPU: T4 x2 (NOT P100)
#
# Usage on Kaggle:
#   1) Accelerator = GPU T4 x2
#   2) Add Input: HiXray_YOLO2 only
#   3) Paste CELL_0 … CELL_5
#   4) Set SMOKE = True in Cell 4 → run (2 epochs)
#   5) Set SMOKE = False → run Cell 4 again (300 epochs, ~25–40 h)
#   6) Run Cell 5 for per-class eval
# =============================================================================

CELL_0 = r'''
# Cell 0 — install (pin Pillow to avoid plot encoder bugs)
!pip -q install ultralytics==8.4.103 pillow==11.0.0
import torch
print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'Enable GPU T4 (prefer T4 x2)'
print('GPU0:', torch.cuda.get_device_name(0))
'''

CELL_1 = r'''
# Cell 1 — find HiXray + write dataset.yaml (NO oversampling)
import os
from pathlib import Path

WORK_DIR = Path('/kaggle/working/hybrid_fdd_protocol')
PROJECT_DIR = Path('/kaggle/working/HiXray_Training_Runs')
WORK_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

INPUT = Path('/kaggle/input')
DATASET = None
for p in INPUT.rglob('HiXray_YOLO2'):
    if (p / 'images' / 'train').exists() and (p / 'images' / 'val').exists():
        DATASET = p
        break
if DATASET is None:
    for p in INPUT.rglob('images/train'):
        root = p.parent.parent
        if (root / 'images' / 'val').exists():
            DATASET = root
            break
assert DATASET is not None, 'Add HiXray_YOLO2 dataset to Inputs'
print('DATASET:', DATASET)

CLASSES = [
    'Cosmetic', 'Laptop', 'Mobile_Phone', 'Nonmetallic_Lighter',
    'Portable_Charger_1', 'Portable_Charger_2', 'Tablet', 'Water',
]
DATA_YAML = WORK_DIR / 'dataset.yaml'
DATA_YAML.write_text(
    f"# Hybrid @ FDD protocol — standard split, no oversample\n"
    f"path: {DATASET.as_posix()}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"nc: 8\n"
    f"names:\n" + '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES)) + '\n',
    encoding='utf-8',
)
print('DATA_YAML:', DATA_YAML)
print(DATA_YAML.read_text())

# share paths with later cells
Path('/kaggle/working/hybrid_fdd_paths.txt').write_text(
    f'{WORK_DIR}\n{PROJECT_DIR}\n{DATA_YAML}\n{DATASET}\n'
)
print('OK — run Cell 2')
'''

CELL_2 = r'''
# Cell 2 — write hybrid_modules.py (your architecture)
from pathlib import Path

WORK_DIR = Path('/kaggle/working/hybrid_fdd_protocol')
WORK_DIR.mkdir(parents=True, exist_ok=True)

(WORK_DIR / 'hybrid_modules.py').write_text(r'''
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
''', encoding='utf-8')
print('Wrote', WORK_DIR / 'hybrid_modules.py')
'''

CELL_3 = r'''
# Cell 3 — write hybrid_yolo.yaml (your architecture)
from pathlib import Path

WORK_DIR = Path('/kaggle/working/hybrid_fdd_protocol')
YAML_LINES = [
    '# Hybrid Lightweight X-Ray Detector (same arch as V7/V9)',
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
    '  - [-1, 1, nn.Upsample,  [None, 2, \'nearest\']]',
    '  - [[-1, 8], 1, Concat,  [1]]',
    '  - [-1, 3, C2f,          [512]]',
    '  - [-1, 1, DAPABlock,    [512]]',
    '  - [-1, 1, nn.Upsample,  [None, 2, \'nearest\']]',
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
yp = WORK_DIR / 'hybrid_yolo.yaml'
yp.write_text('\n'.join(YAML_LINES) + '\n', encoding='utf-8')
print('Wrote', yp)
'''

CELL_4 = r'''
# Cell 4 — TRAIN hybrid with FDD-YOLO protocol
# ============================================================
SMOKE = True   # <<< True = 2 epochs test | False = full 300 epochs
# ============================================================
import os, re, sys, importlib
from pathlib import Path
import torch

WORK_DIR, PROJECT_DIR, DATA_YAML, DATASET = [
    Path(x) for x in Path('/kaggle/working/hybrid_fdd_paths.txt').read_text().strip().splitlines()
]
RUN_NAME = 'hybrid_fdd_protocol_300'
YAML_PATH = WORK_DIR / 'hybrid_yolo.yaml'
EPOCHS = 2 if SMOKE else 300

print('SMOKE=', SMOKE, '| EPOCHS=', EPOCHS)
print('WORK_DIR', WORK_DIR)
print('DATA_YAML', DATA_YAML)
print('Protocol: SGD lr0=0.01 mom=0.9 cos_lr imgsz=640 batch=16 | no oversample')

sys.path.insert(0, str(WORK_DIR))
for name in list(sys.modules):
    if name == 'hybrid_modules' or name.startswith('hybrid_modules.'):
        del sys.modules[name]

from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks

CUSTOM = {
    'DualConv': DualConv, 'SobelConv': SobelConv, 'FDDN': FDDN, 'SSCAM': SSCAM,
    'DAPA_FPN': DAPA_FPN, 'HybridBlock': HybridBlock, 'DAPABlock': DAPABlock,
}
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)

# Patch tasks.py so DDP workers can import custom modules
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
        raise RuntimeError('Could not find parse_model else:c2=ch[f] — check Ultralytics version')
    t, nsub = pat.subn(injection + m.group(0), t, count=1)
    if nsub != 1:
        raise RuntimeError('Channel patch failed')
    print('Applied channel patch')
else:
    print('Channel patch already present')

tasks_py.write_text(t, encoding='utf-8')
importlib.reload(yolo_tasks)
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)
print('tasks ready')

# PYTHONPATH for DDP
_prev = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = str(WORK_DIR) if not _prev else f'{WORK_DIR}{os.pathsep}{_prev}'

from ultralytics import YOLO

model = YOLO(str(YAML_PATH))
try:
    model.info()
except Exception as e:
    print('model.info warning:', e)

n_gpu = torch.cuda.device_count()
device = '0,1' if n_gpu >= 2 else '0'
print('device=', device, 'gpus=', n_gpu)

model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=640,
    batch=16,
    device=device,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    resume=False,
    # --- FDD-YOLO paper protocol ---
    optimizer='SGD',
    momentum=0.9,
    lr0=0.01,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3,
    patience=50 if not SMOKE else 0,
    pretrained=False,   # match FDD rebuild; set True only if you want ImageNet/YOLO init boost
    # --- saves ---
    save=True,
    save_period=25 if not SMOKE else 1,
    plots=False,
    workers=8,
    seed=0,
    deterministic=True,
)

print('Done. SMOKE=', SMOKE, 'EPOCHS=', EPOCHS)
print('Weights:', PROJECT_DIR / RUN_NAME / 'weights')
wdir = PROJECT_DIR / RUN_NAME / 'weights'
if wdir.exists():
    print('Saved:', sorted(p.name for p in wdir.glob('*.pt')))
'''

CELL_5 = r'''
# Cell 5 — per-class eval after training (conf=0.001)
import sys
from pathlib import Path

WORK_DIR, PROJECT_DIR, DATA_YAML, DATASET = [
    Path(x) for x in Path('/kaggle/working/hybrid_fdd_paths.txt').read_text().strip().splitlines()
]
RUN_NAME = 'hybrid_fdd_protocol_300'
WEIGHTS = PROJECT_DIR / RUN_NAME / 'weights'
CLASSES = [
    'Cosmetic', 'Laptop', 'Mobile_Phone', 'Nonmetallic_Lighter',
    'Portable_Charger_1', 'Portable_Charger_2', 'Tablet', 'Water',
]

sys.path.insert(0, str(WORK_DIR))
from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock
import ultralytics.nn.tasks as yolo_tasks
for k, v in {
    'DualConv': DualConv, 'SobelConv': SobelConv, 'FDDN': FDDN, 'SSCAM': SSCAM,
    'DAPA_FPN': DAPA_FPN, 'HybridBlock': HybridBlock, 'DAPABlock': DAPABlock,
}.items():
    setattr(yolo_tasks, k, v)

from ultralytics import YOLO

ckpt = None
for name in ('best.pt', 'last.pt'):
    p = WEIGHTS / name
    if p.exists():
        ckpt = p
        break
if ckpt is None:
    eps = sorted(WEIGHTS.glob('epoch*.pt'), key=lambda p: int(p.stem.replace('epoch', '') or 0))
    ckpt = eps[-1] if eps else None
assert ckpt is not None, f'No weights in {WEIGHTS}'
print('Evaluating:', ckpt)

model = YOLO(str(ckpt))
m = model.val(
    data=str(DATA_YAML), imgsz=640, batch=16, conf=0.001, iou=0.5,
    device=0, plots=False,
)
print(f'\nALL mAP50={m.box.map50:.5f}  mAP50-95={m.box.map:.5f}  P={m.box.mp:.5f}  R={m.box.mr:.5f}')
maps = m.box.maps50 if hasattr(m.box, 'maps50') else m.box.ap50
for i, n in enumerate(CLASSES):
    print(f'  {n:22s} mAP50={float(maps[i]):.5f}')
print('\nCompare to paper FDD-YOLO: mAP50=0.840  mAP50-95=0.527')
print('Compare to paper YOLOv11n:  mAP50=0.808  mAP50-95=0.511')
print('Compare to your V7 finetune: mAP50~0.799 (different protocol)')
'''

if __name__ == '__main__':
    print('Hybrid @ FDD protocol — copy cells into Kaggle notebook\n')
    for i, c in enumerate([CELL_0, CELL_1, CELL_2, CELL_3, CELL_4, CELL_5]):
        print(f'\n===== CELL {i} =====\n')
        print(c)
