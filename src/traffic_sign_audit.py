"""
Traffic Sign Visibility Audit (GTSRB)
=====================================

Reusable code for a transparent, rule-based audit of image visibility.

Reusable functions
------------------
load_annotations        -> read all GTSRB annotation CSV files
validate_annotations    -> validity flags (file exists, class, size, ROI)
extract_image_features  -> OpenCV measurements for one image
compute_thresholds      -> percentile cut-offs (NumPy)
assign_review_priority  -> vectorised review flags / priority score (NumPy)
run_validation_checks   -> small self-test (read failure, columns, bool flags)

Workflow class
--------------
TrafficSignAudit        -> orchestrates load -> sample -> features -> rule
                           -> summaries -> figures -> report

IMPORTANT: no deep learning is used. The review flag is a *triage aid*,
not a safety verdict and not an official standard.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import textwrap
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

LOG = logging.getLogger("traffic_sign_audit")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Filename", "Width", "Height",
                    "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]

CLASS_NAMES = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50",
    3: "Speed limit 60", 4: "Speed limit 70", 5: "Speed limit 80",
    6: "End of speed limit 80", 7: "Speed limit 100", 8: "Speed limit 120",
    9: "No passing", 10: "No passing (>3.5 t)",
    11: "Right-of-way at next intersection", 12: "Priority road",
    13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles >3.5 t prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right",
    21: "Double curve", 22: "Bumpy road", 23: "Slippery road",
    24: "Road narrows on the right", 25: "Road work", 26: "Traffic signals",
    27: "Pedestrians", 28: "Children crossing", 29: "Bicycles crossing",
    30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed/passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing (>3.5 t)",
}

# Classes with very different sizes in the full training set (class 0 is small).
DEFAULT_CLASSES = [0, 1, 13, 14, 17, 25]

# (low percentile, high percentile) for every scenario. First one = baseline.
DEFAULT_SCENARIOS = {
    "baseline_P10_P90": (10, 90),
    "strict_P5_P95": (5, 95),
    "lenient_P20_P80": (20, 80),
}

FEATURE_COLUMNS = ["brightness_gray", "contrast_gray", "sharpness_lapvar"]

FEATURE_DEFINITIONS = [
    ("brightness_gray", "Mean of the grayscale image (cv2.cvtColor BGR2GRAY)", "intensity, 0-255"),
    ("contrast_gray", "Standard deviation of the grayscale image", "intensity, 0-255"),
    ("sharpness_lapvar", "Variance of cv2.Laplacian(gray, CV_64F) - a PROXY for sharpness, "
                         "not a human readability score; depends on image size/noise", "unit-less (intensity^2)"),
    ("brightness_hsv_v", "Mean of the Value channel of HSV", "intensity, 0-255"),
    ("width / height", "Pixel size of the loaded image", "pixels"),
    ("min_side_px", "min(width, height)", "pixels"),
    ("aspect_ratio", "width / height", "ratio"),
    ("roi_* (brightness/contrast/sharpness)", "Same three measures computed on the annotated "
                                              "sign region (ROI crop), when the ROI is valid", "as above"),
]


class ImageReadError(Exception):
    """Raised when an image file cannot be decoded by OpenCV."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _relative_posix(path: Path, root: Path) -> str:
    """Return `path` relative to `root` using forward slashes (no personal paths)."""
    path = Path(os.path.normpath(path))
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _split_parts(name: str) -> tuple:
    """Split a file name/path from an annotation CSV into components (handles / and \\, drops root)."""
    parts = PurePosixPath(str(name).replace("\\", "/")).parts
    return tuple(p for p in parts if p not in ("/", "") and not p.endswith(":"))


def _foreign_prefix_length(names, base: Path, probe: int = 20) -> int:
    """Number of leading path components to ignore so that the paths resolve under `base`.

    Some CSV copies of GTSRB store paths of the author's machine (e.g. '/code/Train/20/x.png').
    The count is found on the first rows and then applied to all rows. Returns 0 if paths already resolve.
    """
    for name in list(names)[:probe]:
        parts = _split_parts(name)
        for k in range(len(parts)):
            if base.joinpath(*parts[k:]).is_file():
                return k
    return 0


def roi_valid_mask(x1, y1, x2, y2, width, height):
    """Vectorised ROI validation.

    Inputs: array-likes (or scalars) of ROI corners and image size.
    Returns: boolean array - True when 0 <= x1 < x2 <= width and
    0 <= y1 < y2 <= height and all values are finite.
    """
    x1, y1, x2, y2, w, h = (np.asarray(v, dtype=float) for v in (x1, y1, x2, y2, width, height))
    finite = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2) \
        & np.isfinite(w) & np.isfinite(h)
    with np.errstate(invalid="ignore"):
        ok = (x1 >= 0) & (y1 >= 0) & (x1 < x2) & (y1 < y2) & (x2 <= w) & (y2 <= h)
    return finite & ok


def spearman_corr(a, b) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks (pandas + NumPy only, no scipy)."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank(method="average")
    rb = pd.Series(np.asarray(b, dtype=float)).rank(method="average")
    return float(np.corrcoef(ra.to_numpy(), rb.to_numpy())[0, 1])


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix of the columns of df (no scipy needed)."""
    ranks = df.rank(method="average")
    return ranks.corr(method="pearson")


def wilson_interval(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n (NumPy, vectorised)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def eta_squared(df: pd.DataFrame, value_col: str, group_col: str) -> float:
    """Share of total variance explained by the group (between-class variance / total)."""
    v = df[value_col].astype(float)
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    g = v.groupby(df[group_col])
    ss_between = float((g.count() * (g.mean() - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else float("nan")


def effect_label(eta2: float) -> str:
    """Rule-of-thumb label (Cohen): <0.01 negligible, <0.06 small, <0.14 medium, else large."""
    if not np.isfinite(eta2):
        return "n/a"
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


# --------------------------------------------------------------------------
# 1) Annotations
# --------------------------------------------------------------------------
def load_annotations(data_root):
    """Load every GTSRB annotation CSV found under `data_root`.

    Supports the official layout (``Final_Training/Images/<class>/GT-<class>.csv``,
    ';'-separated) and the common single-file ``Train.csv`` variant.

    Returns
    -------
    (DataFrame, list[dict])
        All annotation rows (one per image) with extra columns ``source_csv``,
        ``source_row`` and ``image_path`` (relative to data_root), and a list of
        problems (unreadable CSV files, missing columns, malformed numbers).
    Raises
    ------
    FileNotFoundError if the folder or the annotation files do not exist.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder '{data_root}' not found. Download GTSRB (training images) and "
            f"extract it there - see README.md, section 'Getting the data'.")
    csv_files = sorted(p for p in data_root.rglob("GT-*.csv") if "test" not in p.name.lower())
    if not csv_files:
        csv_files = sorted(data_root.rglob("Train.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No annotation files (GT-*.csv or Train.csv) found under '{data_root}'.")

    frames, problems = [], []
    for csv_path in csv_files:
        rel_csv = _relative_posix(csv_path, data_root)
        try:
            df = pd.read_csv(csv_path, sep=";")
            if df.shape[1] == 1:               # maybe comma-separated variant
                df = pd.read_csv(csv_path, sep=",")
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
            problems.append({"file": rel_csv, "reason": f"unreadable annotation file: {exc}"})
            continue
        if "Filename" not in df.columns and "Path" in df.columns:
            df = df.rename(columns={"Path": "Filename"})
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            problems.append({"file": rel_csv, "reason": f"missing required columns {missing}"})
            continue
        df = df[REQUIRED_COLUMNS].copy()
        for col in NUMERIC_COLUMNS:            # malformed numbers -> NaN (counted, not silently lost)
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = int((coerced.isna() & df[col].notna()).sum())
            if n_bad:
                problems.append({"file": rel_csv, "reason": f"{n_bad} malformed values in column {col}"})
            df[col] = coerced.astype(float)
        df["Filename"] = df["Filename"].astype(str)
        df["source_csv"] = rel_csv
        df["source_row"] = np.arange(len(df)) + 2          # line number in the CSV file
        k = _foreign_prefix_length(df["Filename"], csv_path.parent)
        if k:
            problems.append({"file": rel_csv, "reason": f"image paths contained a foreign prefix; the first {k} "
                                                        f"path component(s) were ignored when locating files"})
        df["image_path"] = [_relative_posix(csv_path.parent.joinpath(*_split_parts(f)[k:]), data_root)
                            for f in df["Filename"]]
        frames.append(df)

    if not frames:
        raise ValueError(f"All annotation files were unusable: {problems}")
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "annotation_id", np.arange(len(out)))
    return out, problems


def validate_annotations(df: pd.DataFrame, data_root):
    """Add validity columns and build a data-quality report.

    Adds boolean columns ``image_exists``, ``class_valid``, ``dims_valid``, ``roi_valid``.
    Rows are NEVER dropped here.
    Returns (annotated DataFrame, quality dict).
    """
    data_root = Path(data_root)
    out = df.copy()
    out["image_exists"] = [(data_root / p).is_file() for p in out["image_path"]]

    cid = out["ClassId"].to_numpy(float)
    class_valid = np.isfinite(cid) & (cid == np.floor(np.where(np.isfinite(cid), cid, 0))) \
        & np.isin(cid, list(CLASS_NAMES))
    out["class_valid"] = class_valid
    out["ClassId"] = pd.array(np.where(class_valid, cid, np.nan), dtype="Int64")

    w, h = out["Width"].to_numpy(float), out["Height"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        out["dims_valid"] = np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
    out["roi_valid"] = roi_valid_mask(out["Roi.X1"], out["Roi.Y1"], out["Roi.X2"], out["Roi.Y2"], w, h)

    quality = {
        "n_rows": int(len(out)),
        "n_csv_files": int(out["source_csv"].nunique()),
        "n_classes": int(out.loc[out["class_valid"], "ClassId"].nunique()),
        "dtypes": {c: str(t) for c, t in out[REQUIRED_COLUMNS].dtypes.items()},
        "missing_values": {c: int(out[c].isna().sum()) for c in REQUIRED_COLUMNS},
        "n_duplicate_rows": int(out.duplicated(subset=REQUIRED_COLUMNS + ["source_csv"]).sum()),
        "n_duplicate_image_paths": int(out["image_path"].duplicated().sum()),
        "n_invalid_image_paths": int((~out["image_exists"]).sum()),
        "n_invalid_class_id": int((~out["class_valid"]).sum()),
        "n_invalid_dimensions": int((~out["dims_valid"]).sum()),
        "n_invalid_or_absent_roi": int((~out["roi_valid"]).sum()),
    }
    return out, quality


# --------------------------------------------------------------------------
# 2) OpenCV measurements
# --------------------------------------------------------------------------
def read_image_bgr(path) -> np.ndarray:
    """Read an image with cv2.imread (BGR order) and verify it loaded.

    Raises FileNotFoundError (missing file) or ImageReadError (undecodable file).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path.name}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ImageReadError(f"OpenCV could not decode image: {path.name}")
    return img


def compute_gray_metrics(gray: np.ndarray, prefix: str = "") -> dict:
    """Brightness (mean), contrast (std) and sharpness (Laplacian variance) of a uint8 gray image."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        f"{prefix}brightness_gray": float(gray.mean()),
        f"{prefix}contrast_gray": float(gray.std()),
        f"{prefix}sharpness_lapvar": float(lap.var()),
    }


def extract_image_features(path, roi=None) -> dict:
    """Measure one image with OpenCV.

    Parameters
    ----------
    path : str | Path   image file
    roi  : (x1, y1, x2, y2) or None   annotated sign region (inclusive pixel coords)
    Returns
    -------
    dict with width, height, min_side_px, aspect_ratio, brightness_gray, contrast_gray,
    sharpness_lapvar, brightness_hsv_v, roi_valid, roi_* measures (NaN if ROI invalid).
    Raises FileNotFoundError / ImageReadError for unreadable images.
    """
    bgr = read_image_bgr(path)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = {
        "width": int(w), "height": int(h), "min_side_px": int(min(w, h)),
        "aspect_ratio": float(w) / float(h),
        "brightness_hsv_v": float(hsv[:, :, 2].mean()),
    }
    feats.update(compute_gray_metrics(gray))

    roi_ok, roi_cols = False, {f"roi_{k}": np.nan for k in FEATURE_COLUMNS}
    roi_cols["roi_area_px"] = np.nan
    if roi is not None and bool(roi_valid_mask(*roi, w, h)):
        x1, y1, x2, y2 = (int(v) for v in roi)
        crop = gray[y1:y2 + 1, x1:x2 + 1]       # GTSRB ROI corners are inclusive
        if crop.shape[0] >= 3 and crop.shape[1] >= 3:
            roi_ok = True
            roi_cols = compute_gray_metrics(crop, prefix="roi_")
            roi_cols["roi_area_px"] = int(crop.shape[0] * crop.shape[1])
    feats["roi_valid"] = roi_ok
    feats.update(roi_cols)
    return feats


# --------------------------------------------------------------------------
# 3) Review-priority rule (NumPy)
# --------------------------------------------------------------------------
def compute_thresholds(df: pd.DataFrame, low_pct: float = 10, high_pct: float = 90) -> dict:
    """Percentile cut-offs derived from the analysis sample (project-specific, NOT a standard).

    low_pct is used for 'too dark', 'low contrast', 'low sharpness', 'too small';
    high_pct is used for 'overexposed' (very bright).
    """
    b = df["brightness_gray"].to_numpy(float)
    c = df["contrast_gray"].to_numpy(float)
    s = df["sharpness_lapvar"].to_numpy(float)
    m = df["min_side_px"].to_numpy(float)
    return {
        "low_pct": float(low_pct), "high_pct": float(high_pct),
        "brightness_low": float(np.percentile(b, low_pct)),
        "brightness_high": float(np.percentile(b, high_pct)),
        "contrast_low": float(np.percentile(c, low_pct)),
        "sharpness_low": float(np.percentile(s, low_pct)),
        "min_side_low": float(np.percentile(m, low_pct)),
    }


def assign_review_priority(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    """Vectorised review flags.

    Five boolean conditions (dark, overexposed, low contrast, blurry, small) are
    evaluated with NumPy masks. ``priority_score`` = number of conditions met (0-5);
    ``review_flag`` = score >= 1 (a plain bool); ``priority_tier`` and the main
    ``flag_reason`` are assigned with np.select.
    Returns a copy of df with the new columns.
    """
    b = df["brightness_gray"].to_numpy(float)
    c = df["contrast_gray"].to_numpy(float)
    s = df["sharpness_lapvar"].to_numpy(float)
    m = df["min_side_px"].to_numpy(float)

    dark = b < th["brightness_low"]
    bright = b > th["brightness_high"]
    low_c = c < th["contrast_low"]
    blur = s < th["sharpness_low"]
    small = m < th["min_side_low"]

    score = dark.astype(int) + bright.astype(int) + low_c.astype(int) + blur.astype(int) + small.astype(int)
    out = df.copy()
    out["flag_dark"], out["flag_overexposed"], out["flag_low_contrast"] = dark, bright, low_c
    out["flag_blurry"], out["flag_small"] = blur, small
    out["priority_score"] = score
    out["review_flag"] = (score >= 1).astype(bool)
    out["priority_tier"] = np.select([score >= 3, score == 2, score == 1],
                                     ["high", "medium", "low"], default="none")
    out["flag_reason"] = np.select([blur, dark, low_c, small, bright],
                                   ["low_sharpness", "too_dark", "low_contrast", "too_small", "overexposed"],
                                   default="none")
    return out


# --------------------------------------------------------------------------
# 4) Self-test
# --------------------------------------------------------------------------
def run_validation_checks(audit: "TrafficSignAudit", strict: bool = True) -> dict:
    """Validate: (1) read failures are caught, (2) feature columns exist, (3) flags are only True/False."""
    results = {}
    try:                                                  # (1a) missing file
        read_image_bgr(Path("this_file_does_not_exist.png"))
        results["missing_file_error_caught"] = False
    except FileNotFoundError:
        results["missing_file_error_caught"] = True
    with tempfile.TemporaryDirectory() as tmp:            # (1b) corrupt file
        bad = Path(tmp) / "corrupt.png"
        bad.write_bytes(b"this is not an image")
        try:
            extract_image_features(bad)
            results["corrupt_image_error_caught"] = False
        except ImageReadError:
            results["corrupt_image_error_caught"] = True
    feats = audit.features
    needed = FEATURE_COLUMNS + ["brightness_hsv_v", "width", "height", "min_side_px",
                                "review_flag", "priority_score", "class_id", "image_path"]
    results["feature_columns_exist"] = all(c in feats.columns for c in needed)
    flags = feats["review_flag"]
    results["review_flags_are_bool"] = bool(flags.dtype == bool and set(flags.unique()) <= {True, False})
    results["no_missing_core_features"] = bool(feats[FEATURE_COLUMNS].notna().all().all())
    results["every_row_traceable_to_annotation"] = bool(
        feats["annotation_id"].isin(audit.annotations["annotation_id"]).all())
    if strict:
        failed = [k for k, v in results.items() if not v]
        if failed:
            raise AssertionError(f"Validation failed: {failed}")
    return results


# --------------------------------------------------------------------------
# 5) Workflow class
# --------------------------------------------------------------------------
class TrafficSignAudit:
    """Organises the whole audit: load -> validate -> sample -> measure -> flag -> summarise -> plot -> report."""

    def __init__(self, data_root, output_dir="outputs", selected_classes=None,
                 max_per_class=500, seed=42, scenarios=None):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.proc_dir = self.output_dir / "processed"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.selected_classes = list(selected_classes or DEFAULT_CLASSES)
        if len(self.selected_classes) < 5:
            raise ValueError("The project requires at least five sign classes.")
        self.max_per_class = int(max_per_class)
        self.seed = int(seed)
        self.scenarios = dict(scenarios or DEFAULT_SCENARIOS)
        self.baseline = next(iter(self.scenarios))
        self.exclusions: list[dict] = []
        self.skipped = Counter()
        self.captions: dict[str, str] = {}
        self.annotations = self.sample = self.features = None
        self.quality = {}
        self.thresholds = {}
        self.scenario_thresholds = {}

    # ---- logging of exclusions -------------------------------------------
    def _log_exclusion(self, stage, item, reason):
        self.exclusions.append({"stage": stage, "item": item, "reason": reason})

    # ---- steps -----------------------------------------------------------
    def load_annotations(self):
        """Load + validate the FULL annotation tables (counts here are full-dataset counts)."""
        df, problems = load_annotations(self.data_root)
        for p in problems:
            self._log_exclusion("annotation_file", p["file"], p["reason"])
        self.annotations, self.quality = validate_annotations(df, self.data_root)
        self.quality["annotation_file_problems"] = problems
        LOG.info("Annotations: %d rows, %d classes", self.quality["n_rows"], self.quality["n_classes"])
        return self.annotations

    def class_counts(self) -> pd.DataFrame:
        """Full-dataset image counts per class (all valid-class rows) + counts in the sample."""
        ann = self.annotations
        full = ann[ann["class_valid"]].groupby("ClassId").size()
        t = pd.DataFrame({"n_full": full})
        t.index = t.index.astype(int)
        t.index.name = "class_id"
        t["class_name"] = [CLASS_NAMES[i] for i in t.index]
        t["selected"] = t.index.isin(self.selected_classes)
        if self.features is not None:
            t["n_sampled"] = self.features.groupby("class_id").size().reindex(t.index).fillna(0).astype(int)
        return t.reset_index()

    def select_sample(self):
        """Stratified random sample: up to `max_per_class` images per selected class (fixed seed)."""
        ann = self.annotations
        parts = []
        for cid in self.selected_classes:
            rows = ann[ann["ClassId"] == cid]
            if rows.empty:
                self._log_exclusion("class", f"class {cid}", "class not present in annotations")
                continue
            ok = rows["image_exists"]
            for _, r in rows[~ok].iterrows():
                self._log_exclusion("sampling", r["image_path"], "image file missing on disk")
                self.skipped["missing_file"] += 1
            eligible = rows[ok]
            n = min(len(eligible), self.max_per_class)
            parts.append(eligible.sample(n=n, random_state=self.seed))
        if not parts:
            raise RuntimeError("No eligible images found for the selected classes.")
        self.sample = pd.concat(parts).sort_values("annotation_id").reset_index(drop=True)
        return self.sample

    def extract_features(self):
        """Run OpenCV measurements for every sampled image; failures are counted and logged."""
        records = []
        for row in self.sample.to_dict("records"):
            roi = (row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"])
            path = self.data_root / row["image_path"]
            try:
                feats = extract_image_features(path, roi=roi)
            except FileNotFoundError as exc:
                self.skipped["missing_file"] += 1
                self._log_exclusion("feature_extraction", row["image_path"], str(exc))
                continue
            except ImageReadError as exc:
                self.skipped["unreadable_image"] += 1
                self._log_exclusion("feature_extraction", row["image_path"], str(exc))
                continue
            except (cv2.error, ValueError) as exc:
                self.skipped["opencv_error"] += 1
                self._log_exclusion("feature_extraction", row["image_path"], f"OpenCV error: {exc}")
                continue
            feats["annotation_id"] = row["annotation_id"]
            if not row["roi_valid"]:
                self._log_exclusion("roi", row["image_path"],
                                    "annotated ROI invalid/absent: image kept, ROI measures set to NaN")
                self.skipped["invalid_roi_kept"] += 1
            records.append(feats)
        if not records:
            raise RuntimeError("No image could be processed - check the dataset location.")
        feat_df = pd.DataFrame(records)
        keep = ["annotation_id", "source_csv", "source_row", "image_path", "ClassId",
                "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]
        merged = self.sample[keep].merge(feat_df, on="annotation_id", how="inner", validate="one_to_one")
        merged = merged.rename(columns={"ClassId": "class_id", "Width": "ann_width", "Height": "ann_height"})
        merged["class_id"] = merged["class_id"].astype(int)
        merged.insert(5, "class_name", merged["class_id"].map(CLASS_NAMES))
        merged["size_mismatch"] = (merged["ann_width"] != merged["width"]) | (merged["ann_height"] != merged["height"])
        merged["log10_sharpness"] = np.log10(merged["sharpness_lapvar"] + 1.0)
        self.features = merged
        LOG.info("Features extracted for %d images (skipped: %s)", len(merged), dict(self.skipped))
        return merged

    def assign_review_priority(self):
        """Baseline rule + sensitivity scenarios (flags for each scenario stored as flag_<scenario>)."""
        df = self.features
        for name, (lo, hi) in self.scenarios.items():
            self.scenario_thresholds[name] = compute_thresholds(df, lo, hi)
        self.thresholds = self.scenario_thresholds[self.baseline]
        out = assign_review_priority(df, self.thresholds)
        for name, th in self.scenario_thresholds.items():
            out[f"flag_{name}"] = assign_review_priority(df, th)["review_flag"].to_numpy(bool)
        self.features = out
        return out

    # ---- summaries -------------------------------------------------------
    def summarize(self):
        """pandas group-by summaries answering the case questions."""
        df = self.features
        agg = {"n_images": ("annotation_id", "count")}
        for col in ["width", "height", "min_side_px", "brightness_gray", "contrast_gray", "sharpness_lapvar"]:
            agg[f"{col}_mean"] = (col, "mean")
            agg[f"{col}_median"] = (col, "median")
            agg[f"{col}_std"] = (col, "std")
        cs = df.groupby("class_id").agg(**agg).round(2)
        cs.insert(0, "class_name", [CLASS_NAMES[i] for i in cs.index])
        self.class_summary = cs

        rows = []
        for col in ["min_side_px", "brightness_gray", "contrast_gray", "log10_sharpness"]:
            e = eta_squared(df, col, "class_id")
            rows.append({"feature": col, "eta_squared": round(e, 3), "effect_size": effect_label(e)})
        self.effect_sizes = pd.DataFrame(rows)

        # flag rate by class and scenario (with Wilson 95% CI)
        rate_rows = []
        for name in self.scenarios:
            g = df.groupby("class_id")[f"flag_{name}"].agg(["sum", "count"])
            lo, hi = wilson_interval(g["sum"], g["count"])
            for i, cid in enumerate(g.index):
                rate_rows.append({"scenario": name, "class_id": cid, "class_name": CLASS_NAMES[cid],
                                  "n_images": int(g["count"].iloc[i]), "n_flagged": int(g["sum"].iloc[i]),
                                  "flag_rate_pct": round(100 * g["sum"].iloc[i] / g["count"].iloc[i], 2),
                                  "ci95_low_pct": round(100 * lo[i], 2), "ci95_high_pct": round(100 * hi[i], 2)})
        self.flag_rates = pd.DataFrame(rate_rows)
        wide = self.flag_rates.pivot(index="class_id", columns="scenario", values="flag_rate_pct")[list(self.scenarios)]
        self.flag_rate_wide = wide
        self.rank_table = wide.rank(ascending=False, method="min").astype(int)
        self.rank_corr = {n: spearman_corr(wide[self.baseline], wide[n])
                          for n in self.scenarios if n != self.baseline}

        # sensitivity: overall counts
        srows = []
        for name, th in self.scenario_thresholds.items():
            k = int(df[f"flag_{name}"].sum())
            srows.append({"scenario": name, **{k2: round(v, 2) for k2, v in th.items()},
                          "n_flagged": k, "pct_flagged": round(100 * k / len(df), 2)})
        self.sensitivity = pd.DataFrame(srows)

        # reasons by class (baseline)
        self.reasons_by_class = pd.crosstab(df["class_id"], df["flag_reason"])

        # correlations (Spearman: monotonic association, not causation)
        self.correlations = spearman_matrix(df[["sharpness_lapvar", "contrast_gray", "brightness_gray"]]).round(3)

        # ROI vs full image on a small sample
        roi = df[df["roi_valid"]]
        small = roi.sample(n=min(200, len(roi)), random_state=self.seed) if len(roi) else roi
        comp = []
        for f in FEATURE_COLUMNS:
            if len(small) > 2:
                comp.append({"feature": f, "mean_full": small[f].mean(), "mean_roi": small[f"roi_{f}"].mean(),
                             "mean_abs_diff": (small[f] - small[f"roi_{f}"]).abs().mean(),
                             "spearman_full_vs_roi": spearman_corr(small[f], small[f"roi_{f}"])})
        self.roi_comparison = pd.DataFrame(comp).round(3)
        self._roi_sample = small

        # error analysis (false alarms / possible misses), heuristic using ROI measures vs same cut-offs
        th = self.thresholds
        r = df[df["roi_valid"]]
        roi_bad = (r["roi_contrast_gray"] < th["contrast_low"]) | (r["roi_sharpness_lapvar"] < th["sharpness_low"]) \
            | (r["roi_brightness_gray"] < th["brightness_low"]) | (r["roi_brightness_gray"] > th["brightness_high"])
        self.error_analysis = {
            "n_flagged_with_valid_roi": int(r["review_flag"].sum()),
            "n_flagged_but_roi_looks_normal": int((r["review_flag"] & ~roi_bad).sum()),
            "n_unflagged_with_valid_roi": int((~r["review_flag"]).sum()),
            "n_unflagged_but_roi_below_cutoffs": int((~r["review_flag"] & roi_bad).sum()),
        }
        return cs

    # ---- saving ----------------------------------------------------------
    def save_tables(self):
        """Write all derived tables (never touches the raw dataset)."""
        p = self.proc_dir
        self.class_counts().to_csv(p / "class_counts_full_vs_sample.csv", index=False)
        self.features.to_csv(p / "image_features.csv", index=False)
        self.class_summary.to_csv(p / "class_summary.csv")
        self.effect_sizes.to_csv(p / "effect_sizes.csv", index=False)
        self.flag_rates.to_csv(p / "flag_rates_by_class_and_scenario.csv", index=False)
        self.sensitivity.to_csv(p / "sensitivity_scenarios.csv", index=False)
        self.correlations.to_csv(p / "correlations_spearman.csv")
        self.roi_comparison.to_csv(p / "roi_vs_full_comparison.csv", index=False)
        self.reasons_by_class.to_csv(p / "flag_reasons_by_class.csv")
        queue = self.features[self.features["review_flag"]].sort_values(
            ["priority_score", "sharpness_lapvar"], ascending=[False, True])
        queue[["image_path", "class_id", "class_name", "priority_score", "priority_tier", "flag_reason",
               "brightness_gray", "contrast_gray", "sharpness_lapvar", "min_side_px"]].to_csv(
            p / "review_queue.csv", index=False)
        pd.DataFrame(self.exclusions, columns=["stage", "item", "reason"]).to_csv(p / "exclusion_log.csv", index=False)
        with open(p / "data_quality.json", "w", encoding="utf-8") as fh:
            json.dump(self.quality, fh, indent=2, default=str)
        cfg = {"selected_classes": self.selected_classes, "max_per_class": self.max_per_class,
               "seed": self.seed, "scenarios": self.scenarios, "baseline_thresholds": self.thresholds,
               "skipped_counts": dict(self.skipped), "n_images_analysed": int(len(self.features))}
        with open(p / "run_config.json", "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    # ---- plotting --------------------------------------------------------
    def _label(self, cid):
        return f"{cid}\n" + textwrap.fill(CLASS_NAMES[cid], 12)

    def _rgb(self, image_path, size=None):
        bgr = read_image_bgr(self.data_root / image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # BGR -> RGB for Matplotlib
        if size:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        return rgb

    def _finish(self, fig, name, caption, show):
        path = self.fig_dir / f"{name}.png"
        top = 0.93 if fig._suptitle is not None else 1.0
        fig.tight_layout(rect=[0, 0, 1, top], h_pad=2.5)
        fig.savefig(path, dpi=130)
        self.captions[name] = caption
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_contact_sheet(self, show=False):
        """Fig 1: flagged (top row) vs unflagged (bottom row) example per class."""
        df, cls = self.features, self.selected_classes
        cls = [c for c in cls if c in set(df["class_id"])]
        fig, axes = plt.subplots(2, len(cls), figsize=(2.3 * len(cls), 6.6), squeeze=False)
        for j, cid in enumerate(cls):
            for i, flagged in enumerate([True, False]):
                ax = axes[i, j]
                sub = df[(df["class_id"] == cid) & (df["review_flag"] == flagged)]
                ax.set_xticks([]); ax.set_yticks([])
                if sub.empty:
                    ax.text(0.5, 0.5, "none", ha="center", va="center"); continue
                r = sub.sample(1, random_state=self.seed + j).iloc[0]
                ax.imshow(self._rgb(r["image_path"], 96))
                ax.set_title(f"{'FLAGGED' if flagged else 'unflagged'}\nclass {cid}", fontsize=8)
                ax.set_xlabel(f"B={r['brightness_gray']:.0f} C={r['contrast_gray']:.0f}\n"
                              f"S={r['sharpness_lapvar']:.0f} {int(r['width'])}x{int(r['height'])}px", fontsize=7)
        fig.suptitle("Contact sheet: flagged (top) vs unflagged (bottom) images by class", fontsize=11)
        cap = ("Random flagged/unflagged example per class (seeded). B=brightness, C=contrast, S=Laplacian-variance "
               "sharpness proxy. Images are resized only for display; flagged images are triage candidates, not verdicts.")
        return self._finish(fig, "fig01_contact_sheet", cap, show)

    def plot_distributions(self, show=False):
        """Fig 2: brightness and sharpness histograms with cut-off lines."""
        df, th = self.features, self.thresholds
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(df["brightness_gray"], bins=40, color="#4C78A8", edgecolor="white")
        axes[0].axvline(th["brightness_low"], color="red", ls="--", label=f"low cut-off (P{th['low_pct']:.0f}) = {th['brightness_low']:.1f}")
        axes[0].axvline(th["brightness_high"], color="darkorange", ls="--", label=f"high cut-off (P{th['high_pct']:.0f}) = {th['brightness_high']:.1f}")
        axes[0].set_title("Brightness distribution (grayscale mean)")
        axes[0].set_xlabel("Mean grayscale intensity (0-255)"); axes[0].set_ylabel("Number of images")
        axes[0].legend(fontsize=8)
        s = df["sharpness_lapvar"].to_numpy(float)
        bins = np.logspace(np.log10(max(s.min(), 1e-1)), np.log10(s.max() + 1), 40)
        axes[1].hist(s, bins=bins, color="#54A24B", edgecolor="white")
        axes[1].set_xscale("log")
        axes[1].axvline(th["sharpness_low"], color="red", ls="--", label=f"low cut-off (P{th['low_pct']:.0f}) = {th['sharpness_low']:.1f}")
        axes[1].set_title("Sharpness proxy distribution (Laplacian variance)")
        axes[1].set_xlabel("Variance of Laplacian (log scale)"); axes[1].set_ylabel("Number of images")
        axes[1].legend(fontsize=8)
        cap = ("Distributions over the analysed sample with the baseline project-specific cut-offs (percentiles of this sample; "
               "not safety standards). Sharpness is strongly right-skewed, hence the log axis.")
        return self._finish(fig, "fig02_distributions_with_cutoffs", cap, show)

    def _boxes(self, ax, col, ylabel, title, log=False):
        cls = sorted(self.features["class_id"].unique())
        data = [self.features.loc[self.features["class_id"] == c, col] for c in cls]
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(cls) + 1))
        ax.set_xticklabels([self._label(c) for c in cls], fontsize=7)
        ax.set_title(title); ax.set_xlabel("Sign class"); ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")

    def plot_boxplots(self, show=False):
        """Fig 3: class-wise boxplots of brightness, contrast, sharpness."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        self._boxes(axes[0], "brightness_gray", "Mean grayscale (0-255)", "Brightness by class")
        self._boxes(axes[1], "contrast_gray", "Std of grayscale", "Contrast by class")
        self._boxes(axes[2], "sharpness_lapvar", "Laplacian variance (log)", "Sharpness proxy by class", log=True)
        cap = ("Class-wise distributions. Boxes show median and IQR; points are outliers. Compare the spread within a class "
               "with the shift between classes before calling a difference meaningful (see eta-squared table).")
        return self._finish(fig, "fig03_boxplots_by_class", cap, show)

    def plot_flag_rates(self, show=False):
        """Fig 4: review-flag rate by class for every cut-off scenario (with 95% Wilson intervals)."""
        fr = self.flag_rates
        cls = sorted(fr["class_id"].unique())
        names = list(self.scenarios)
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for k, name in enumerate(names):
            sub = fr[fr["scenario"] == name].set_index("class_id").loc[cls]
            x = np.arange(len(cls)) + k * width
            err = np.vstack([sub["flag_rate_pct"] - sub["ci95_low_pct"], sub["ci95_high_pct"] - sub["flag_rate_pct"]])
            ax.bar(x, sub["flag_rate_pct"], width, yerr=err, capsize=2, label=name)
        ax.set_xticks(np.arange(len(cls)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels([self._label(c) for c in cls], fontsize=8)
        ax.set_title("Review-flag rate by class under three cut-off settings")
        ax.set_xlabel("Sign class"); ax.set_ylabel("Flagged images (% of class sample)")
        ax.legend(fontsize=8)
        cap = ("Share of sampled images flagged per class. Error bars: 95% Wilson intervals (wider when a class has fewer images). "
               "Compare bar heights with the intervals before reading differences as real.")
        return self._finish(fig, "fig04_flag_rate_by_class", cap, show)

    def plot_size_contrast(self, show=False):
        """Fig 5: image size and contrast by class."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        self._boxes(axes[0], "min_side_px", "Shorter image side (pixels)", "Image size by class")
        cls = sorted(self.features["class_id"].unique())
        g = self.features.groupby("class_id")["contrast_gray"]
        m, se = g.mean().loc[cls], (g.std() / np.sqrt(g.count())).loc[cls]
        axes[1].bar(range(len(cls)), m, yerr=1.96 * se, capsize=3, color="#B279A2")
        axes[1].set_xticks(range(len(cls)))
        axes[1].set_xticklabels([self._label(c) for c in cls], fontsize=7)
        axes[1].set_title("Mean contrast by class (95% CI of the mean)")
        axes[1].set_xlabel("Sign class"); axes[1].set_ylabel("Mean std of grayscale")
        cap = "GTSRB images vary strongly in size; small images carry fewer pixels and are harder to inspect. Contrast bars show mean +/- 1.96 SE."
        return self._finish(fig, "fig05_size_and_contrast_by_class", cap, show)

    def plot_relationships(self, show=False):
        """Fig 6: sharpness vs contrast and vs brightness (association, not causation)."""
        df = self.features
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, col, lab in [(axes[0], "contrast_gray", "Contrast (std of grayscale)"),
                             (axes[1], "brightness_gray", "Brightness (mean grayscale)")]:
            rho = self.correlations.loc["sharpness_lapvar", col]
            ax.scatter(df[col], df["sharpness_lapvar"], s=5, alpha=0.35,
                       c=np.where(df["review_flag"], "#E45756", "#4C78A8"))
            ax.set_yscale("log")
            ax.set_xlabel(lab); ax.set_ylabel("Sharpness proxy (log)")
            ax.set_title(f"Sharpness vs {col.split('_')[0]} (Spearman rho = {rho:.2f})")
        cap = ("Red = flagged, blue = unflagged. Spearman rho describes a monotonic association only; it does not show that "
               "low contrast or brightness causes low sharpness (image size and noise are confounders).")
        return self._finish(fig, "fig06_sharpness_vs_contrast_brightness", cap, show)

    def _pick_examples(self):
        df, th = self.features, self.thresholds
        rs = np.random.RandomState(self.seed)
        flagged = df[df["review_flag"]]
        a = flagged.sort_values(["priority_score", "sharpness_lapvar"], ascending=[False, True]).iloc[0] if len(flagged) else df.iloc[0]
        ok_roi = (df["roi_contrast_gray"] >= th["contrast_low"]) & (df["roi_sharpness_lapvar"] >= th["sharpness_low"]) \
            & df["roi_brightness_gray"].between(th["brightness_low"], th["brightness_high"])
        fb = df[df["review_flag"] & df["roi_valid"] & ok_roi]
        if len(fb):
            b, tag_b = fb.iloc[rs.randint(len(fb))], "Possible FALSE ALARM (flagged, sign region looks normal)"
        elif len(flagged):
            b, tag_b = flagged[flagged["priority_score"] == flagged["priority_score"].min()].iloc[0], "Flagged, single reason"
        else:
            b, tag_b = df.iloc[0], "example"
        fc = df[~df["review_flag"] & df["roi_valid"] & ~ok_roi]
        un = df[~df["review_flag"]]
        if len(fc):
            c, tag_c = fc.iloc[rs.randint(len(fc))], "Possible MISS (unflagged, sign region below cut-offs)"
        elif len(un):
            c, tag_c = un.iloc[rs.randint(len(un))], "Typical UNFLAGGED image"
        else:
            c, tag_c = df.iloc[-1], "example"
        return [(a, "Top-priority FLAGGED image"), (b, tag_b), (c, tag_c)]

    def plot_examples(self, show=False):
        """Fig 7: three images that best illustrate the rule, with measurements and class labels."""
        ex = self._pick_examples()
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))
        for ax, (r, tag) in zip(axes, ex):
            ax.imshow(self._rgb(r["image_path"]))
            if r["roi_valid"]:
                ax.add_patch(Rectangle((r["Roi.X1"], r["Roi.Y1"]), r["Roi.X2"] - r["Roi.X1"], r["Roi.Y2"] - r["Roi.Y1"],
                                       fill=False, edgecolor="lime", lw=1.5))
            ax.set_title(f"{textwrap.fill(tag, 34)}\nClass {int(r['class_id'])}: {r['class_name']}", fontsize=8)
            ax.set_xlabel(f"brightness {r['brightness_gray']:.1f} | contrast {r['contrast_gray']:.1f}\n"
                          f"sharpness {r['sharpness_lapvar']:.1f} | size {int(r['width'])}x{int(r['height'])}px\n"
                          f"score {int(r['priority_score'])} ({r['flag_reason']}); ROI sharp. {r['roi_sharpness_lapvar']:.1f}",
                          fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Three examples illustrating the review rule (green box = annotated sign region)", fontsize=11)
        cap = ("Left: highest-priority flagged image. Middle/right: heuristic false-alarm and missed-case candidates, found by "
               "comparing ROI-crop measures with the same cut-offs. These labels are illustrative, not ground truth.")
        return self._finish(fig, "fig07_three_examples", cap, show)

    def plot_roi_vs_full(self, show=False):
        """Fig 8: ROI-crop measurements vs full-image measurements for a small sample."""
        s = self._roi_sample
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, f, lab in zip(axes, FEATURE_COLUMNS, ["Brightness", "Contrast", "Sharpness proxy (log)"]):
            ax.scatter(s[f], s[f"roi_{f}"], s=8, alpha=0.5)
            lo = min(s[f].min(), s[f"roi_{f}"].min()); hi = max(s[f].max(), s[f"roi_{f}"].max())
            if f == "sharpness_lapvar":
                ax.set_xscale("log"); ax.set_yscale("log"); lo = max(lo, 1e-1)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
            ax.set_title(f"{lab}: ROI crop vs full image")
            ax.set_xlabel("Full image"); ax.set_ylabel("ROI crop"); ax.legend(fontsize=8)
        cap = f"Random sample of {len(s)} images with valid ROI. Points off the diagonal show where the background changes the whole-image measure."
        return self._finish(fig, "fig08_roi_vs_full_image", cap, show)

    def make_all_figures(self, show=False):
        """Create and save every figure; returns list of paths."""
        return [self.plot_contact_sheet(show), self.plot_distributions(show), self.plot_boxplots(show),
                self.plot_flag_rates(show), self.plot_size_contrast(show), self.plot_relationships(show),
                self.plot_examples(show), self.plot_roi_vs_full(show)]

    # ---- one-call pipeline -----------------------------------------------
    def run_all(self, show=False):
        """Recreate every table, figure and the report from the raw annotations."""
        self.load_annotations()
        self.select_sample()
        self.extract_features()
        self.assign_review_priority()
        self.summarize()
        self.make_all_figures(show)
        self.save_tables()
        run_validation_checks(self)
        from report import write_report
        return write_report(self)
