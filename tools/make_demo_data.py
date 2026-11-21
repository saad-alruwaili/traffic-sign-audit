"""Create a SMALL SYNTHETIC dataset with the GTSRB folder layout, only to smoke-test the pipeline.

The images are random drawings - NEVER use this data for your real results/report.
Usage: python tools/make_demo_data.py            -> data/GTSRB_demo
"""
from pathlib import Path
import sys

import cv2
import numpy as np

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / "GTSRB_demo"
rng = np.random.default_rng(0)
counts = {0: 60, 1: 220, 13: 200, 14: 120, 17: 150, 25: 180, 38: 90}   # unequal on purpose

for cid, n in counts.items():
    folder = out / "GTSRB" / "Final_Training" / "Images" / f"{cid:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = ["Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"]
    for i in range(n):
        side = int(rng.integers(20, 120))
        base = int(rng.integers(30, 230))
        img = np.full((side, side, 3), base, np.uint8)
        img = np.clip(img + rng.normal(0, rng.integers(3, 40), img.shape), 0, 255).astype(np.uint8)
        pad = max(1, side // 10)
        cv2.circle(img, (side // 2, side // 2), side // 2 - pad, (int(rng.integers(0, 255)),) * 3, -1)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (7, 7), 0)
        name = f"{cid:05d}_{i // 30:05d}_{i % 30:05d}.ppm"
        cv2.imwrite(str(folder / name), img)
        x1, y1, x2, y2 = pad, pad, side - pad - 1, side - pad - 1
        rows.append(f"{name};{side};{side};{x1};{y1};{x2};{y2};{cid}")
    if cid == 14:   # deliberate defects to exercise error handling
        (folder / "00014_99999_00000.ppm").write_bytes(b"corrupt")          # unreadable image
        rows.append("00014_99999_00000.ppm;40;40;3;3;36;36;14")
        rows.append("00014_missing_file.ppm;40;40;3;3;36;36;14")            # missing file
        (folder / "bad_roi.ppm").write_bytes((folder / "00014_00000_00000.ppm").read_bytes())
        rows.append("bad_roi.ppm;40;40;50;50;10;10;14")                      # invalid ROI
    (folder / f"GT-{cid:05d}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Synthetic demo data written to {out}")
