# Thesis figures — Python-generated (not AI images)

## Generate

```bash
cd thesis_figures
pip install matplotlib pandas numpy python-docx
python generate_thesis_figures.py
```

PNGs + PDFs appear in `out/`. Source numbers are in `data/*.csv`.

## Insert into Word thesis

```bash
python insert_figures_into_docx.py
```

Creates `Maraol_Feye_Thesis_with_Python_Figures.docx` with figures placed above matching “Figure X.Y …” captions.

## PRISMA counts

Edit `PRISMA_COUNTS` at the top of `generate_thesis_figures.py` to match Section 2.2, then re-run.

## Qualitative detections (Figure 4.3)

Those need real model predictions (`run_predict.py` / Kaggle val with `plots=True`). This folder covers charts + architecture diagrams only.
