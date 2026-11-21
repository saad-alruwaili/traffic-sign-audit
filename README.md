# Traffic Sign Visibility Audit (GTSRB)

A reproducible, rule-based audit that answers: **which traffic-sign images should a human review first, and what evidence supports that priority?**
It uses Python, NumPy, pandas, Matplotlib and OpenCV only - **no deep learning**. The review flag is a *triage aid*, not a safety verdict, and
the cut-offs are project-specific percentiles, not official standards.

## Dataset and citation
German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum.
* Official page: https://benchmark.ini.rub.de/gtsrb_dataset.html
* Archival record: https://zenodo.org/records/13741936
* Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks, 32, 323-332.

The dataset is **not included** in this repository (check the source terms before redistributing).

## Getting the data
1. Download the **training images** archive (e.g. `GTSRB_Final_Training_Images.zip`) from the official page or the Zenodo record.
2. Extract it so the folder looks like (raw data stays unchanged):
```
data/GTSRB/Final_Training/Images/00000/00000_00000.ppm
data/GTSRB/Final_Training/Images/00000/GT-00000.csv   (annotation, ';'-separated)
...
```
The Kaggle-style copy (`Train.csv` + `Train/<class>/*.png`) is also supported, even when its `Path` column holds paths from another computer (the prefix is ignored and this is logged).
Any other location works with `--data-root <folder>`. A single `Train.csv` layout is also detected.

## Installation
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Script (recreates all tables, figures and the report from the raw annotations):
```bash
python run_audit.py --data-root data/GTSRB          # defaults: classes 0 1 13 14 17 25, 500/class, seed 42
python run_audit.py --classes 1 2 13 14 17 25 38 --max-per-class 500 --seed 42
```
Notebook: `pip install jupyter`, then `jupyter notebook notebooks/traffic_sign_audit.ipynb` and *Run All*.

Smoke test without downloading anything (synthetic images, **never use for results**):
```bash
python tools/make_demo_data.py data/GTSRB_demo
python run_audit.py --data-root data/GTSRB_demo --output-dir outputs_demo
```
Tests: `python -m unittest discover -s tests -v`

## Folder structure
```
traffic-sign-audit/
├── README.md  requirements.txt  run_audit.py
├── data/                 # dataset goes here (not committed)
├── notebooks/            # traffic_sign_audit.ipynb
├── src/                  # traffic_sign_audit.py (functions + TrafficSignAudit class), report.py
├── tests/                # unit tests
├── tools/                # demo-data generator, notebook builder
└── outputs/
    ├── figures/          # 8 labelled figures (incl. contact sheet)
    ├── processed/        # derived tables, exclusion log, review queue
    ├── report.md         # auto-generated full report (numbers come from the run)
    └── report_short.md   # condensed report text (5 pages)

report/                   # final report as PDF and Word (condensed, 5 pages)
```

## Design
* Functions: `load_annotations`, `validate_annotations`, `extract_image_features`, `compute_thresholds`, `assign_review_priority`, `run_validation_checks`.
* Class `TrafficSignAudit`: `load_annotations`, `select_sample`, `extract_features`, `assign_review_priority`, `summarize`, `plot_*`, `save_tables`, `run_all`.
* Full-dataset counts (all annotation rows) are kept separate from the sampled image analysis (<= 500 images/class, fixed seed).
* Skipped files (missing, unreadable, invalid ROI) are counted and listed in `outputs/processed/exclusion_log.csv`; nothing is dropped silently.
* Rule: five conditions (dark, overexposed, low contrast, low sharpness, small) from percentile cut-offs; sensitivity with P5/P95 and P20/P80.

## Final steps for submission
1. Run on the real GTSRB data. 2. Open `outputs/report.md`, add your name, read the numbers, refine the wording (4-6 pages) and export to PDF/Word.
3. Be ready to explain the feature formulas, the rule, and one limitation in class.
