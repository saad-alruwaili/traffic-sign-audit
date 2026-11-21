"""Main script: recreates every table, figure and the report from the raw annotations.

Usage:  python run_audit.py --data-root data/GTSRB
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display needed when running as a script

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from traffic_sign_audit import DEFAULT_CLASSES, TrafficSignAudit  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Traffic Sign Visibility Audit (GTSRB)")
    p.add_argument("--data-root", default=str(Path("data") / "GTSRB"), help="folder containing the extracted GTSRB training data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES, help="at least five class IDs (0-42)")
    p.add_argument("--max-per-class", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = parse_args()
    try:
        audit = TrafficSignAudit(a.data_root, a.output_dir, a.classes, a.max_per_class, a.seed)
        report = audit.run_all()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    print(f"Done. Report: {report}\nTables: {audit.proc_dir}\nFigures: {audit.fig_dir}")
    print(f"Skipped files: {dict(audit.skipped) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
