# =============================================================================
# Resume Hybrid @ FDD protocol after Kaggle 12h timeout
# =============================================================================
# Your run stopped ~epoch 85 (timeout 43200s), NOT a code crash.
#
# What to upload to the NEW Kaggle account:
#   1) HiXray_YOLO2 dataset
#   2) results.zip  (must contain last.pt — ideally full run folder)
#      OR upload last.pt alone as a dataset
#
# Inside results.zip, preferred layout:
#   HiXray_Training_Runs/hybrid_fdd_protocol_300/weights/last.pt
#   HiXray_Training_Runs/hybrid_fdd_protocol_300/results.csv
#   HiXray_Training_Runs/hybrid_fdd_protocol_300/args.yaml
#   (optional) hybrid_fdd_protocol/hybrid_modules.py + hybrid_yolo.yaml
#
# Plan to finish 300 epochs:
#   ~85 epochs / 12h  →  ~7 ep/hour
#   Remaining ~215 ep →  ~30h more →  about 3 more Save&Run All sessions
#   Each session: resume from latest last.pt, download new results.zip after
# =============================================================================

CELL_0 = r'''
# Cell 0 — install
!pip -q install ultralytics==8.4.103 pillow==11.0.0
import torch
print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'Use GPU T4 x2 (not P100)'
print('GPU0:', torch.cuda.get_device_name(0))
'''

CELL_1 = r'''
# Cell 1 — unzip results + find last.pt + HiXray
import zipfile, shutil
from pathlib import Path

WORK_DIR = Path('/kaggle/working/hybrid_fdd_protocol')
PROJECT_DIR = Path('/kaggle/working/HiXray_Training_Runs')
RUN_NAME = 'hybrid_fdd_protocol_300'
RUN_DIR = PROJECT_DIR / RUN_NAME
WEIGHTS_DIR = RUN_DIR / 'weights'
WORK_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

INPUT = Path('/kaggle/input')

# --- HiXray ---
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
assert DATASET is not None, 'Add HiXray_YOLO2 to Inputs'
print('DATASET:', DATASET)

# --- unzip any results.zip from Inputs into /kaggle/working ---
zips = list(INPUT.rglob('results.zip')) + list(INPUT.rglob('*.zip'))
zips = [z for z in zips if 'result' in z.name.lower() or 'hybrid' in z.name.lower() or 'weight' in z.name.lower()]
if not zips:
    zips = list(INPUT.rglob('*.zip'))
print('Zip candidates:', zips)
for z in zips:
    print('Extracting', z)
    with zipfile.ZipFile(z, 'r') as zf:
        zf.extractall('/kaggle/working/_unzipped')
    break  # first matching zip

# Also accept a bare last.pt uploaded as dataset
LAST = None
search_roots = [
    Path('/kaggle/working/_unzipped'),
    Path('/kaggle/working'),
    INPUT,
]
for root in search_roots:
    if not root.exists():
        continue
    # prefer run-folder last.pt
    hits = list(root.rglob('**/hybrid_fdd_protocol_300/weights/last.pt'))
    if not hits:
        hits = list(root.rglob('last.pt'))
    if hits:
        # pick largest (full ckpt) / most nested hybrid path
        hits.sort(key=lambda p: (0 if 'hybrid_fdd' in str(p) else 1, -p.stat().st_size))
        LAST = hits[0]
        break

assert LAST is not None and LAST.exists(), (
    'last.pt not found. Upload results.zip containing '
    'HiXray_Training_Runs/hybrid_fdd_protocol_300/weights/last.pt'
)
print('FOUND last.pt:', LAST, 'size_MB=', round(LAST.stat().st_size / 1e6, 2))

# Copy into canonical resume location Ultralytics expects
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
dst_last = WEIGHTS_DIR / 'last.pt'
if LAST.resolve() != dst_last.resolve():
    shutil.copy2(LAST, dst_last)
print('Resume checkpoint:', dst_last)

# Copy sibling files if present (args.yaml, results.csv, best.pt, epoch*.pt)
src_w = LAST.parent
for name in ('best.pt', 'args.yaml', 'results.csv'):
    # args/results live in run dir, not always weights/
    pass
for p in src_w.glob('*.pt'):
    target = WEIGHTS_DIR / p.name
    if not target.exists():
        shutil.copy2(p, target)
        print('Copied weight', p.name)
run_src = src_w.parent if src_w.name == 'weights' else src_w
for name in ('args.yaml', 'results.csv'):
    sp = run_src / name
    if sp.exists():
        shutil.copy2(sp, RUN_DIR / name)
        print('Copied', name)

# dataset.yaml
CLASSES = [
    'Cosmetic', 'Laptop', 'Mobile_Phone', 'Nonmetallic_Lighter',
    'Portable_Charger_1', 'Portable_Charger_2', 'Tablet', 'Water',
]
DATA_YAML = WORK_DIR / 'dataset.yaml'
DATA_YAML.write_text(
    f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n"
    + '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES)) + '\n',
    encoding='utf-8',
)

Path('/kaggle/working/hybrid_fdd_resume_paths.txt').write_text(
    f'{WORK_DIR}\n{PROJECT_DIR}\n{DATA_YAML}\n{DATASET}\n{dst_last}\n{RUN_NAME}\n'
)
print('OK — run Cell 2 (modules) then Cell 3 (resume train)')
'''

CELL_2 = r'''
# Cell 2 — write hybrid_modules + yaml (same architecture)
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
(WORK_DIR / 'hybrid_yolo.yaml').write_text('\n'.join(YAML_LINES) + '\n', encoding='utf-8')
print('Wrote hybrid_modules.py + hybrid_yolo.yaml')
'''

CELL_3 = r'''
# Cell 3 — RESUME training from last.pt (continues epoch 86 → 300)
import os, re, sys, importlib
from pathlib import Path
import torch

WORK_DIR, PROJECT_DIR, DATA_YAML, DATASET, LAST_PT, RUN_NAME = [
    Path(x) if i < 5 else x
    for i, x in enumerate(Path('/kaggle/working/hybrid_fdd_resume_paths.txt').read_text().strip().splitlines())
]
LAST_PT = Path(LAST_PT)
RUN_NAME = str(RUN_NAME)

print('Resuming from:', LAST_PT)
assert LAST_PT.exists(), LAST_PT

# MUST import hybrid_modules BEFORE torch.load / YOLO(last.pt)
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

# Peek epoch stored in checkpoint (after hybrid_modules is importable)
ckpt = torch.load(str(LAST_PT), map_location='cpu', weights_only=False)
ep = ckpt.get('epoch', None)
print('Checkpoint epoch field:', ep, '(Ultralytics often stores 0-based → next train epoch = ep+1)')
print('Keys:', list(ckpt.keys())[:12])

# DDP-safe patch
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
        raise RuntimeError('parse_model patch point not found')
    t, nsub = pat.subn(injection + m.group(0), t, count=1)
    assert nsub == 1
    print('Applied channel patch')
else:
    print('Channel patch already present')

tasks_py.write_text(t, encoding='utf-8')
importlib.reload(yolo_tasks)
for k, v in CUSTOM.items():
    setattr(yolo_tasks, k, v)

_prev = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = str(WORK_DIR) if not _prev else f'{WORK_DIR}{os.pathsep}{_prev}'

from ultralytics import YOLO

# IMPORTANT: load last.pt then resume=True (keeps optimizer + epoch counter)
model = YOLO(str(LAST_PT))
n_gpu = torch.cuda.device_count()
device = '0,1' if n_gpu >= 2 else '0'
print('device=', device)

model.train(
    resume=True,          # continue from checkpoint
    data=str(DATA_YAML),  # re-point data for new machine paths
    device=device,
    # keep same protocol; resume restores most args from ckpt
    exist_ok=True,
    plots=False,
    workers=8,
)

print('Resume session finished (or hit 12h again).')
print('Download NEW results / last.pt before the next session.')
wdir = Path(PROJECT_DIR) / RUN_NAME / 'weights'
if wdir.exists():
    print('Weights now:', sorted(p.name for p in wdir.glob('*.pt')))
'''

CELL_4 = r'''
# Cell 4 — quick status from results.csv (optional)
import csv
from pathlib import Path

PROJECT_DIR = Path('/kaggle/working/HiXray_Training_Runs')
csv_path = PROJECT_DIR / 'hybrid_fdd_protocol_300' / 'results.csv'
if not csv_path.exists():
    print('No results.csv yet')
else:
    rows = list(csv.DictReader(csv_path.open()))
    last = rows[-1]
    best = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
    print(f'Epochs logged: {len(rows)}')
    print(f'Last  ep {last["epoch"]}: mAP50={float(last["metrics/mAP50(B)"]):.5f}')
    print(f'Best  ep {best["epoch"]}: mAP50={float(best["metrics/mAP50(B)"]):.5f}')
    print(f'Remaining to 300: {300 - int(float(last["epoch"]))} epochs')
'''

if __name__ == '__main__':
    print('Resume notebook cells:\n')
    for i, c in enumerate([CELL_0, CELL_1, CELL_2, CELL_3, CELL_4]):
        print(f'\n===== CELL {i} =====\n')
        print(c)
