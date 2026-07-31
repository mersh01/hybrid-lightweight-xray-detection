#!/usr/bin/env python3
"""
Generate thesis figures with Python (matplotlib) from real experiment CSVs.

No AI-generated images. Re-run anytime after updating CSVs in ./data/

Usage:
  python generate_thesis_figures.py
  python generate_thesis_figures.py --out out --dpi 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

# Academic-friendly style (not AI-default purple/cream)
COLORS = {
    "long": "#1f4e79",
    "ft": "#c45911",
    "base": "#548235",
    "accent": "#833c0c",
    "grid": "#d9d9d9",
    "box": "#f2f2f2",
    "box_edge": "#404040",
    "arrow": "#595959",
    "rare": "#c00000",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def save(fig, name: str, out: Path, dpi: int):
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    pdf = out / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png.name} / {pdf.name}")


def _box(ax, xy, w, h, text, fc=None, ec=None, fontsize=9, weight="normal"):
    fc = fc or COLORS["box"]
    ec = ec or COLORS["box_edge"]
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2),
    )


# ---------------------------------------------------------------------------
# Figure 2.1 — PRISMA (edit counts in PRISMA_COUNTS if needed)
# ---------------------------------------------------------------------------
PRISMA_COUNTS = {
    "identified": 87,
    "duplicates_removed": 12,
    "screened": 75,
    "excluded_title": 48,
    "full_text": 27,
    "excluded_full": 15,
    "included": 12,
}


def fig_2_1_prisma(out: Path, dpi: int):
    c = PRISMA_COUNTS
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("Figure 2.1  PRISMA literature screening flow", pad=12)

    _box(ax, (2.5, 12.2), 5, 1.1, f"Records identified\nn = {c['identified']}", fc="#ddebf7")
    _box(ax, (2.5, 10.5), 5, 1.0, f"Duplicates removed\nn = {c['duplicates_removed']}")
    _arrow(ax, (5, 12.2), (5, 11.55))
    _box(ax, (2.5, 8.8), 5, 1.1, f"Records screened\nn = {c['screened']}", fc="#ddebf7")
    _arrow(ax, (5, 10.5), (5, 9.95))
    _box(ax, (7.7, 8.8), 2.1, 1.1, f"Excluded\nn = {c['excluded_title']}", fc="#fce4d6")
    _arrow(ax, (7.5, 9.35), (7.7, 9.35))
    _box(ax, (2.5, 7.1), 5, 1.1, f"Full-text assessed\nn = {c['full_text']}", fc="#ddebf7")
    _arrow(ax, (5, 8.8), (5, 8.25))
    _box(ax, (7.7, 7.1), 2.1, 1.1, f"Excluded\nn = {c['excluded_full']}", fc="#fce4d6")
    _arrow(ax, (7.5, 7.65), (7.7, 7.65))
    _box(
        ax,
        (2.5, 5.2),
        5,
        1.3,
        f"Studies included\nin synthesis\nn = {c['included']}",
        fc="#e2efda",
        weight="bold",
    )
    _arrow(ax, (5, 7.1), (5, 6.55))
    ax.text(
        5,
        4.4,
        "Edit PRISMA_COUNTS in generate_thesis_figures.py\nto match your Section 2.2 numbers.",
        ha="center",
        va="top",
        fontsize=8,
        color="#666666",
        style="italic",
    )
    save(fig, "Fig_2_1_PRISMA", out, dpi)


# ---------------------------------------------------------------------------
# Figures 3.x — architecture diagrams (drawn, not AI images)
# ---------------------------------------------------------------------------
def fig_3_1_pipeline(out: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Figure 3.1  Hybrid lightweight X-ray detection framework", pad=10)

    stages = [
        (0.3, "Input\nX-ray", "#fff2cc"),
        (3.0, "Backbone\nConv/C2f +\nHybridBlock", "#ddebf7"),
        (6.5, "SPPF", "#ddebf7"),
        (8.8, "Neck\nDAPABlock\nFPN", "#fce4d6"),
        (12.3, "DualConv\ndownsample", "#fce4d6"),
        (15.5, "SSCAM", "#e2efda"),
        (18.3, "Detect\nP3/P4/P5", "#c6efce"),
    ]
    for x, text, fc in stages:
        _box(ax, (x, 2.0), 2.5, 2.2, text, fc=fc, fontsize=8)
    for i in range(len(stages) - 1):
        x0 = stages[i][0] + 2.5
        x1 = stages[i + 1][0]
        _arrow(ax, (x0, 3.1), (x1, 3.1))
    ax.text(
        11,
        0.6,
        "Modules: DualConv/FDDN (FDD-YOLO) · SSCAM (DGDN) · DAPA-FPN (iX-Det) · SobelConv (E-MPDNet-inspired)",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    save(fig, "Fig_3_1_framework_pipeline", out, dpi)


def fig_3_2_layer_map(out: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Figure 3.2  Hybrid YOLO layer / scale map", pad=10)

    backbone = [
        "0 Conv 64",
        "1 Conv 128",
        "2 C2f ×3",
        "3 Conv 256",
        "4 C2f ×4",
        "5 HybridBlock 256",
        "6 Conv 512",
        "7 C2f ×4",
        "8 HybridBlock 512",
        "9 Conv 1024",
        "10 C2f ×3",
        "11 SPPF",
        "12 HybridBlock 1024",
    ]
    head = [
        "13 Upsample",
        "14 Concat + C2f",
        "15 DAPABlock 512",
        "16 Upsample",
        "17 Concat + C2f",
        "18 DAPABlock 256",
        "19 DualConv ↓",
        "20 Concat + C2f",
        "21 DAPABlock 512",
        "22 DualConv ↓",
        "23 Concat + C2f",
        "24 SSCAM",
        "25 Detect [P3,P4,P5]",
    ]
    ax.text(2.5, 11.4, "Backbone", ha="center", fontweight="bold")
    ax.text(7.5, 11.4, "Head / Neck", ha="center", fontweight="bold")
    for i, t in enumerate(backbone):
        fc = "#ddebf7" if "Hybrid" in t or "SPPF" in t else COLORS["box"]
        if "Hybrid" in t:
            fc = "#9dc3e6"
        _box(ax, (0.5, 10.5 - i * 0.75), 4.0, 0.65, t, fc=fc, fontsize=8)
    for i, t in enumerate(head):
        fc = COLORS["box"]
        if "DAPA" in t:
            fc = "#f8cbad"
        elif "DualConv" in t:
            fc = "#ffe699"
        elif "SSCAM" in t or "Detect" in t:
            fc = "#c6efce"
        _box(ax, (5.5, 10.5 - i * 0.75), 4.0, 0.65, t, fc=fc, fontsize=8)
    save(fig, "Fig_3_2_layer_map", out, dpi)


def fig_3_3_hybridblock(out: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Figure 3.3  HybridBlock = FDDN + SSCAM", pad=10)

    _box(ax, (0.4, 2.8), 1.8, 1.4, "Input\nX", fc="#fff2cc")
    _box(ax, (3.0, 4.6), 2.4, 1.3, "AvgPool\n(low)", fc="#ddebf7")
    _box(ax, (3.0, 1.2), 2.4, 1.3, "X − low\n(high)", fc="#ddebf7")
    _box(ax, (6.2, 4.6), 2.4, 1.3, "DualConv", fc="#bdd7ee")
    _box(ax, (6.2, 1.2), 2.4, 1.3, "SobelConv", fc="#bdd7ee")
    _box(ax, (9.4, 2.8), 2.2, 1.4, "Concat\n+ SE", fc="#fce4d6")
    _box(ax, (12.4, 2.8), 2.2, 1.4, "SSCAM", fc="#c6efce", weight="bold")
    _box(ax, (14.8, 2.8), 1.0, 1.4, "Y", fc="#e2efda")
    _arrow(ax, (2.2, 3.5), (3.0, 5.25))
    _arrow(ax, (2.2, 3.5), (3.0, 1.85))
    _arrow(ax, (5.4, 5.25), (6.2, 5.25))
    _arrow(ax, (5.4, 1.85), (6.2, 1.85))
    _arrow(ax, (8.6, 5.25), (9.4, 3.9))
    _arrow(ax, (8.6, 1.85), (9.4, 3.2))
    _arrow(ax, (11.6, 3.5), (12.4, 3.5))
    _arrow(ax, (14.6, 3.5), (14.8, 3.5))
    save(fig, "Fig_3_3_HybridBlock", out, dpi)


def fig_3_4_dapa(out: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Figure 3.4  DAPABlock distraction-aware fusion", pad=10)

    _box(ax, (0.4, 2.8), 1.8, 1.4, "Input\nX", fc="#fff2cc")
    _box(ax, (3.2, 4.5), 3.2, 1.5, "Foreground\nDualConv + SSCAM", fc="#f8cbad")
    _box(ax, (3.2, 1.0), 3.2, 1.5, "Background gate g\nConv1×1 + Sigmoid", fc="#fce4d6")
    _box(ax, (8.0, 2.6), 3.4, 1.8, "Fuse:\nFG × (1 + g)\nthen DualConv", fc="#ddebf7", weight="bold")
    _box(ax, (12.5, 2.8), 2.0, 1.4, "Output", fc="#c6efce")
    _arrow(ax, (2.2, 3.5), (3.2, 5.25))
    _arrow(ax, (2.2, 3.5), (3.2, 1.75))
    _arrow(ax, (6.4, 5.25), (8.0, 3.8))
    _arrow(ax, (6.4, 1.75), (8.0, 3.2))
    _arrow(ax, (11.4, 3.5), (12.5, 3.5))
    save(fig, "Fig_3_4_DAPABlock", out, dpi)


# ---------------------------------------------------------------------------
# Figure 4.x — results from CSVs
# ---------------------------------------------------------------------------
def fig_4_1_training_curves(out: Path, dpi: int):
    long_df = pd.read_csv(DATA / "long_stage_results.csv")
    rare_df = pd.read_csv(DATA / "rare_ft_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
    ax = axes[0]
    ax.plot(
        long_df["epoch"],
        long_df["metrics/mAP50(B)"],
        color=COLORS["long"],
        lw=1.4,
        label="Long-stage hybrid (FDD protocol)",
    )
    best_ep = int(long_df.loc[long_df["metrics/mAP50(B)"].idxmax(), "epoch"])
    best_v = float(long_df["metrics/mAP50(B)"].max())
    ax.scatter([best_ep], [best_v], color=COLORS["rare"], zorder=5, s=28)
    ax.annotate(
        f"peak ep{best_ep}\n{best_v:.3f}",
        (best_ep, best_v),
        textcoords="offset points",
        xytext=(8, -18),
        fontsize=8,
        color=COLORS["rare"],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title("(a) Long-stage training")
    ax.grid(True, color=COLORS["grid"], lw=0.6)
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1]
    ax.plot(
        rare_df["epoch"],
        rare_df["metrics/mAP50(B)"],
        color=COLORS["ft"],
        lw=1.6,
        marker="o",
        markersize=3.5,
        label="Rare-class fine-tune",
    )
    # selected epoch 20
    if (rare_df["epoch"] == 20).any():
        row = rare_df.loc[rare_df["epoch"] == 20].iloc[0]
        ax.scatter([20], [row["metrics/mAP50(B)"]], color=COLORS["rare"], s=55, zorder=5, marker="*")
        ax.annotate(
            f"selected ep20\n{row['metrics/mAP50(B)']:.3f}",
            (20, row["metrics/mAP50(B)"]),
            textcoords="offset points",
            xytext=(10, -20),
            fontsize=8,
            color=COLORS["rare"],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.set_title("(b) Rare-class fine-tune")
    ax.grid(True, color=COLORS["grid"], lw=0.6)
    ax.legend(loc="lower right", frameon=True)

    fig.suptitle("Figure 4.1  Training curves (mAP50 vs epoch)", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "Fig_4_1_training_curves_mAP50", out, dpi)


def fig_4_1b_losses(out: Path, dpi: int):
    rare_df = pd.read_csv(DATA / "rare_ft_results.csv")
    long_df = pd.read_csv(DATA / "long_stage_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for col, lab, c in [
        ("train/box_loss", "box", COLORS["long"]),
        ("train/cls_loss", "cls", COLORS["ft"]),
        ("train/dfl_loss", "dfl", COLORS["base"]),
    ]:
        ax.plot(long_df["epoch"], long_df[col], label=lab, color=c, lw=1.2)
    ax.set_title("(a) Long-stage train losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, color=COLORS["grid"], lw=0.6)
    ax.legend()

    ax = axes[1]
    for col, lab, c in [
        ("train/box_loss", "box", COLORS["long"]),
        ("train/cls_loss", "cls", COLORS["ft"]),
        ("train/dfl_loss", "dfl", COLORS["base"]),
    ]:
        ax.plot(rare_df["epoch"], rare_df[col], label=lab, color=c, lw=1.4, marker="o", ms=3)
    ax.set_title("(b) Rare-FT train losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, color=COLORS["grid"], lw=0.6)
    ax.legend()
    fig.suptitle("Figure 4.1b  Training losses", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "Fig_4_1b_training_losses", out, dpi)


def fig_4_2_perclass(out: Path, dpi: int):
    long_df = pd.read_csv(DATA / "long_stage_perclass.csv")
    ft_df = pd.read_csv(DATA / "epoch20_perclass.csv")
    classes = [
        "Cosmetic",
        "Laptop",
        "Mobile_Phone",
        "Nonmetallic_Lighter",
        "Portable_Charger_1",
        "Portable_Charger_2",
        "Tablet",
        "Water",
    ]
    short = {
        "Cosmetic": "Cos",
        "Laptop": "Lap",
        "Mobile_Phone": "Phone",
        "Nonmetallic_Lighter": "Lighter",
        "Portable_Charger_1": "PC1",
        "Portable_Charger_2": "PC2",
        "Tablet": "Tab",
        "Water": "Water",
    }
    long_m = {r["class"]: r["mAP50"] for _, r in long_df.iterrows()}
    ft_m = {r["class"]: r["mAP50"] for _, r in ft_df.iterrows()}
    y1 = [long_m[c] for c in classes]
    y2 = [ft_m[c] for c in classes]
    labels = [short[c] for c in classes]

    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bars1 = ax.bar(x - w / 2, y1, w, label="Long-stage best @640", color=COLORS["long"])
    bars2 = ax.bar(x + w / 2, y2, w, label="Rare-FT epoch20 @768", color=COLORS["ft"])
    # highlight Lighter
    i_lig = classes.index("Nonmetallic_Lighter")
    bars1[i_lig].set_edgecolor(COLORS["rare"])
    bars2[i_lig].set_edgecolor(COLORS["rare"])
    bars1[i_lig].set_linewidth(2.0)
    bars2[i_lig].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Figure 4.2  Per-class mAP50: long-stage vs selected rare-FT (epoch20)\n"
        "(Note: resolutions differ 640 vs 768; run fair-res eval for matched-imgsz deltas)"
    )
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend(loc="lower right")
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        if h < 0.15:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save(fig, "Fig_4_2_perclass_mAP50", out, dpi)


def fig_4_2b_overall_delta(out: Path, dpi: int):
    """Simple overall / rare comparison bars."""
    labels = ["ALL", "Cosmetic", "Nonmetallic_Lighter", "Rare avg\n(Cos+Lig)/2"]
    long_v = [0.79989, 0.65434, 0.01682, (0.65434 + 0.01682) / 2]
    ft_v = [0.80881, 0.66460, 0.08598, (0.66460 + 0.08598) / 2]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - w / 2, long_v, w, label="Long-stage @640", color=COLORS["long"])
    ax.bar(x + w / 2, ft_v, w, label="epoch20 @768", color=COLORS["ft"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 4.2b  Overall and rare-class mAP50")
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.6)
    ax.legend()
    fig.tight_layout()
    save(fig, "Fig_4_2b_overall_rare_bars", out, dpi)


def fig_4_dataset(out: Path, dpi: int):
    df = pd.read_csv(DATA / "dataset_class_counts.csv")
    train = df[df["split"] == "train"].sort_values("class_id")
    val = df[df["split"] == "val"].sort_values("class_id")
    labels = [
        n.replace("Nonmetallic_", "").replace("Portable_Charger_", "PC")
        for n in train["class_name"]
    ]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.bar(x - w / 2, train["instances"], w, label="Train instances", color=COLORS["long"])
    ax.bar(x + w / 2, val["instances"], w, label="Val instances", color=COLORS["ft"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Instance count")
    ax.set_title("Figure 4.0  HiXray class instance distribution")
    ax.set_yscale("log")
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.6, which="both")
    ax.legend()
    fig.tight_layout()
    save(fig, "Fig_4_0_dataset_class_counts", out, dpi)


def fig_4_complexity(out: Path, dpi: int):
    df = pd.read_csv(DATA / "complexity_matrix.csv")
    # one architecture — take epoch20 rows
    sub = df[df["weights"].str.contains("epoch20")].copy()
    if sub.empty:
        sub = df.drop_duplicates(subset=["imgsz"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs = sub["imgsz"].astype(str) + " px"
    ax.bar(xs, sub["GFLOPs"], color=[COLORS["long"], COLORS["ft"]], width=0.55)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["GFLOPs"] + 0.25, f"{r['GFLOPs']:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("GFLOPs")
    ax.set_ylim(0, max(sub["GFLOPs"]) * 1.25)
    params = int(sub.iloc[0]["params"])
    ax.set_title(f"Figure 4.4  Hybrid model GFLOPs vs input size\n(params = {params:,}; same for both checkpoints)")
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.6)
    fig.tight_layout()
    save(fig, "Fig_4_4_GFLOPs_by_imgsz", out, dpi)


def write_caption_index(out: Path):
    text = """Thesis figures generated by generate_thesis_figures.py
=====================================================
All PNGs/PDFs are produced by Python (matplotlib) from CSV data or
explicit diagram code — not AI image generators.

File                                      Suggested caption
----------------------------------------  ----------------------------------
Fig_2_1_PRISMA.png                        Figure 2.1 PRISMA flow
Fig_3_1_framework_pipeline.png            Figure 3.1 Framework overview
Fig_3_2_layer_map.png                     Figure 3.2 Layer / scale map
Fig_3_3_HybridBlock.png                   Figure 3.3 HybridBlock
Fig_3_4_DAPABlock.png                     Figure 3.4 DAPABlock
Fig_4_0_dataset_class_counts.png          Figure 4.0 Dataset distribution
Fig_4_1_training_curves_mAP50.png         Figure 4.1 Training curves
Fig_4_1b_training_losses.png              Figure 4.1b Loss curves
Fig_4_2_perclass_mAP50.png                Figure 4.2 Per-class mAP50
Fig_4_2b_overall_rare_bars.png            Figure 4.2b Overall / rare bars
Fig_4_4_GFLOPs_by_imgsz.png               Figure 4.4 GFLOPs @640 vs @768

Insert into Word: Insert → Pictures → This Device → select PNG.
Or run: python insert_figures_into_docx.py
"""
    (out / "FIGURE_INDEX.txt").write_text(text, encoding="utf-8")
    print(f"  wrote FIGURE_INDEX.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    setup_style()
    print(f"Data: {DATA}")
    print(f"Out : {args.out}")
    print("Generating figures...")

    fig_2_1_prisma(args.out, args.dpi)
    fig_3_1_pipeline(args.out, args.dpi)
    fig_3_2_layer_map(args.out, args.dpi)
    fig_3_3_hybridblock(args.out, args.dpi)
    fig_3_4_dapa(args.out, args.dpi)
    fig_4_dataset(args.out, args.dpi)
    fig_4_1_training_curves(args.out, args.dpi)
    fig_4_1b_losses(args.out, args.dpi)
    fig_4_2_perclass(args.out, args.dpi)
    fig_4_2b_overall_delta(args.out, args.dpi)
    fig_4_complexity(args.out, args.dpi)
    write_caption_index(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
