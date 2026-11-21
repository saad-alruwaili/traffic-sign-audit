"""Builds notebooks/traffic_sign_audit.ipynb (no extra dependency). Run: python tools/build_notebook.py"""
import json
from pathlib import Path

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}

cells = [
md("# Traffic Sign Visibility Audit (GTSRB)\nRule-based audit with NumPy, pandas, Matplotlib and OpenCV. **No deep learning.** "
   "Flags are triage aids, not safety verdicts. Run all cells from top to bottom."),
code("""import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import traffic_sign_audit as tsa
from report import write_report
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40)"""),
md("## 1. Configuration (project choices)"),
code("""DATA_ROOT = ROOT / 'data' / 'GTSRB'      # change to data/GTSRB_demo only for smoke tests
SELECTED = [0, 1, 13, 14, 17, 25]         # >= 5 classes
audit = tsa.TrafficSignAudit(DATA_ROOT, ROOT / 'outputs', SELECTED, max_per_class=500, seed=42)"""),
md("## 2. Load and inspect the FULL annotations (full-dataset counts)"),
code("""ann = audit.load_annotations()
display(ann.head())
print('Rows:', audit.quality['n_rows'], '| classes:', audit.quality['n_classes'])
print('dtypes:', audit.quality['dtypes'])
pd.Series({k: v for k, v in audit.quality.items() if k.startswith('n_') or k == 'missing_values'})"""),
code("""counts = audit.class_counts()
counts.sort_values('n_full', ascending=False).head(43)"""),
md("## 3. Stratified sample, OpenCV features, tidy table"),
code("""audit.select_sample()
feats = audit.extract_features()
print('Skipped:', dict(audit.skipped))
pd.DataFrame(audit.exclusions).head(20)"""),
code("""feats[['image_path','class_id','width','height','brightness_gray','contrast_gray','sharpness_lapvar','roi_valid']].head()"""),
md("## 4. Review-priority rule (NumPy) and sensitivity check"),
code("""audit.assign_review_priority()
audit.summarize()
print(audit.thresholds)
audit.sensitivity"""),
code("""print('Flagged overall:', int(audit.features['review_flag'].sum()), 'of', len(audit.features))
audit.flag_rate_wide"""),
md("## 5. pandas summaries"),
code("""display(audit.class_summary)
display(audit.effect_sizes)
display(audit.rank_table)
display(audit.correlations)
audit.roi_comparison"""),
md("## 6. Figures"),
code("audit.plot_contact_sheet(show=True)"),
code("audit.plot_distributions(show=True)"),
code("audit.plot_boxplots(show=True)"),
code("audit.plot_flag_rates(show=True)"),
code("audit.plot_size_contrast(show=True)"),
code("audit.plot_relationships(show=True)"),
code("audit.plot_examples(show=True)"),
code("audit.plot_roi_vs_full(show=True)"),
md("## 7. Validation cell"),
code("""checks = tsa.run_validation_checks(audit)
checks"""),
md("## 8. Save tables and write the report"),
code("""audit.save_tables()
print(write_report(audit))"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "traffic_sign_audit.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
