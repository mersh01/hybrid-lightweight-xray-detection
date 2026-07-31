"""Register hybrid custom modules with Ultralytics before loading .pt / YAML.

Call `register()` once at the start of any local or Kaggle script.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULES_DIR = Path(__file__).resolve().parent
_REGISTERED = False


def register(force: bool = False) -> None:
    """Make HybridBlock / RareClassTrainer importable for YOLO(.pt) load."""
    global _REGISTERED
    if _REGISTERED and not force:
        return

    for p in (str(_MODULES_DIR), str(_REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Checkpoint args reference RareClassTrainer — import must succeed first.
    import custom_rare_trainer  # noqa: F401
    from hybrid_modules import (
        DualConv,
        SobelConv,
        FDDN,
        SSCAM,
        DAPA_FPN,
        HybridBlock,
        DAPABlock,
    )

    import ultralytics.nn.tasks as yolo_tasks

    custom = {
        "DualConv": DualConv,
        "SobelConv": SobelConv,
        "FDDN": FDDN,
        "SSCAM": SSCAM,
        "DAPA_FPN": DAPA_FPN,
        "HybridBlock": HybridBlock,
        "DAPABlock": DAPABlock,
    }
    for name, cls in custom.items():
        setattr(yolo_tasks, name, cls)

    _patch_parse_model(yolo_tasks, _MODULES_DIR)
    importlib.reload(yolo_tasks)
    for name, cls in custom.items():
        setattr(yolo_tasks, name, cls)

    prev = os.environ.get("PYTHONPATH", "")
    extra = str(_MODULES_DIR)
    if extra not in prev.split(os.pathsep):
        os.environ["PYTHONPATH"] = extra if not prev else f"{extra}{os.pathsep}{prev}"

    _REGISTERED = True


def _patch_parse_model(yolo_tasks, modules_dir: Path) -> None:
    """Inject channel-handling for custom modules into ultralytics.nn.tasks."""
    tasks_py = Path(yolo_tasks.__file__)
    text = tasks_py.read_text(encoding="utf-8")

    imp = (
        "import sys\n"
        f"if r'{modules_dir}' not in sys.path:\n"
        f"    sys.path.insert(0, r'{modules_dir}')\n"
        "from hybrid_modules import DualConv, SobelConv, FDDN, SSCAM, DAPA_FPN, HybridBlock, DAPABlock\n"
    )
    if "from hybrid_modules import" not in text:
        text = imp + text

    marker = "elif m in (HybridBlock, DAPABlock, DAPA_FPN, FDDN, DualConv, SobelConv)"
    if marker not in text:
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
        match = pat.search(text)
        if match is None:
            raise RuntimeError(
                "Could not patch ultralytics parse_model. "
                "Try: pip install 'ultralytics==8.4.103'"
            )
        text, nsub = pat.subn(injection + match.group(0), text, count=1)
        if nsub != 1:
            raise RuntimeError("Failed to inject custom-module parse_model branch.")

    tasks_py.write_text(text, encoding="utf-8")
