"""Generates outputs/report.md from the computed tables (all numbers come from the run)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic_sign_audit import CLASS_NAMES, FEATURE_DEFINITIONS


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency needed)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _name(cid) -> str:
    return f"{int(cid)} ({CLASS_NAMES[int(cid)]})"


def write_report(audit) -> Path:
    """Write the analysis report (case question, data prep, features, findings, memo, limitations)."""
    q, df, th = audit.quality, audit.features, audit.thresholds
    ct = audit.class_counts()
    sel = ct[ct["selected"]].sort_values("n_full", ascending=False)
    big, small = sel.iloc[0], sel.iloc[-1]
    wide, ranks = audit.flag_rate_wide, audit.rank_table
    base = audit.baseline
    top_cls = wide[base].sort_values(ascending=False)
    n_total = len(df)
    n_flag = int(df["review_flag"].sum())
    n_high = int((df["priority_tier"] == "high").sum())
    n_med = int((df["priority_tier"] == "medium").sum())
    corr = audit.correlations
    ea = audit.error_analysis
    es = audit.effect_sizes.set_index("feature")
    reasons_total = df.loc[df["review_flag"], "flag_reason"].value_counts()
    top_reason = reasons_total.index[0] if len(reasons_total) else "none"

    # rank stability
    top_by_scn = {n: int(wide[n].idxmax()) for n in audit.scenarios}
    same_top = len(set(top_by_scn.values())) == 1
    rho_txt = "; ".join(f"{n}: rho = {v:.2f}" for n, v in audit.rank_corr.items())

    s_rho = corr.loc["sharpness_lapvar", "contrast_gray"]
    s_rho_b = corr.loc["sharpness_lapvar", "brightness_gray"]

    def strength(r):
        a = abs(r)
        return "weak" if a < 0.3 else "moderate" if a < 0.6 else "strong"

    fig_names = list(audit.captions)
    figs_md = "\n\n".join(f"![{n}](figures/{n}.png)\n\n*{audit.captions[n]}*" for n in fig_names)

    sens_cols = ["scenario", "brightness_low", "brightness_high", "contrast_low", "sharpness_low",
                 "min_side_low", "n_flagged", "pct_flagged"]
    feat_defs = pd.DataFrame(FEATURE_DEFINITIONS, columns=["Feature", "Definition", "Unit"])

    excl = pd.DataFrame(audit.exclusions, columns=["stage", "item", "reason"])
    excl_summary = (excl.groupby(["stage", "reason"]).size().rename("count").reset_index().head(15)
                    if len(excl) else pd.DataFrame({"note": ["No exclusions occurred."]}))

    cs = audit.class_summary
    cs_view = cs[["class_name", "n_images", "min_side_px_median", "brightness_gray_median",
                  "contrast_gray_median", "sharpness_lapvar_median"]]

    report = f"""# Traffic Sign Visibility Audit - Analysis Report

*Author: Saad Saeed ALRuwaili | Course: Advanced AI Programming and Frameworks | Dataset: GTSRB | Seed: {audit.seed}*

## 1. Case question
Road-safety analysts cannot inspect every traffic-sign photograph manually. **Which images should be reviewed first, and what
evidence supports that priority?** The audit measures brightness, contrast and a sharpness proxy with OpenCV, builds a
transparent percentile-based rule with NumPy, and compares classes with pandas. The flag is a **triage aid only**: it is not proof
that a sign is unsafe, unreadable or misclassified. No deep-learning model is used.

## 2. Data and preparation
**Source.** German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum
(https://benchmark.ini.rub.de/gtsrb_dataset.html; archival copy: https://zenodo.org/records/13741936).
Citation: Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms
for traffic sign recognition.* Neural Networks, 32, 323-332. The archive is not redistributed in this repository.

**Full-dataset counts (all annotation rows, not sampled).**
{q['n_rows']:,} annotation rows from {q['n_csv_files']} CSV file(s) covering {q['n_classes']} classes.

Data types after loading: {', '.join(f'{k}: {v}' for k, v in q['dtypes'].items())}.

{md_table(pd.DataFrame({"Check": ["Duplicate rows", "Duplicate image paths", "Invalid image paths (file missing)", "Invalid class IDs",
                                  "Invalid dimensions", "Invalid or absent ROI", "Missing values (all columns)"],
                        "Count": [q['n_duplicate_rows'], q['n_duplicate_image_paths'], q['n_invalid_image_paths'],
                                  q['n_invalid_class_id'], q['n_invalid_dimensions'], q['n_invalid_or_absent_roi'],
                                  sum(q['missing_values'].values())]}))}

**Scope and sampling (separate from the full counts).** Selected classes: {", ".join(_name(c) for c in audit.selected_classes)}.
For each selected class, up to {audit.max_per_class} images were drawn at random (seed {audit.seed}); the image-metric analysis therefore covers
**{n_total:,} images**. Ineligible rows (missing file, invalid class) are never dropped silently: they are written to
`outputs/processed/exclusion_log.csv`. Images with an invalid/absent ROI are kept; only their ROI measures are NaN.
Skipped-file counters: {dict(audit.skipped) if audit.skipped else "none"}.

{md_table(excl_summary)}

**Class counts: full vs sample.**

{md_table(ct[ct["selected"]][["class_id", "class_name", "n_full", "n_sampled"]])}

## 3. Feature definitions
OpenCV loads colour images in BGR; images are converted to RGB before display with Matplotlib.

{md_table(feat_defs)}

## 4. Review-priority rule (project-specific, not a safety standard)
An image meets a condition if it is **too dark** (brightness < P{th['low_pct']:.0f}), **overexposed** (brightness > P{th['high_pct']:.0f}),
**low contrast** (< P{th['low_pct']:.0f}), **blurry by proxy** (sharpness < P{th['low_pct']:.0f}) or **small** (shorter side < P{th['low_pct']:.0f}).
Percentiles are computed on the analysed sample with `np.percentile`; flags are vectorised boolean masks and `np.select`.
`priority_score` = number of conditions met; `review_flag` = score >= 1; tiers: high (>=3), medium (2), low (1).

Baseline cut-offs: brightness {th['brightness_low']:.1f} / {th['brightness_high']:.1f}, contrast {th['contrast_low']:.1f},
sharpness {th['sharpness_low']:.1f}, shorter side {th['min_side_low']:.0f}px.
The baseline flags **{n_flag:,} of {n_total:,} images ({100 * n_flag / n_total:.1f}%)**: {n_high} high, {n_med} medium priority.

**Sensitivity check (three cut-off settings).**

{md_table(audit.sensitivity[sens_cols])}

## 5. Findings
**Q1 - Class sizes and imbalance.** Among the selected classes the largest in the full data is {_name(big['class_id'])}
({int(big['n_full']):,} images) and the smallest {_name(small['class_id'])} ({int(small['n_full']):,}); ratio {big['n_full'] / small['n_full']:.1f}:1.
Capping at {audit.max_per_class} per class reduces but does not remove this imbalance (sample sizes: {int(sel['n_sampled'].max())} vs {int(sel['n_sampled'].min())}).
Small classes give wide confidence intervals for flag rates (Figure 4), and global percentile cut-offs are dominated by larger classes.

**Q2 - Size, brightness, contrast, sharpness across classes.** Share of variance explained by class (eta-squared; rule-of-thumb labels):

{md_table(audit.effect_sizes)}

Class medians:

{md_table(cs_view, index=True)}

Features whose class effect is "medium" or "large" are the ones where class differences matter for review priority; "negligible/small"
differences are dominated by within-class spread (Figure 3, Figure 5).

**Q3 - Flag rate by class and ranking stability.** Baseline flag rates (%):

{md_table(audit.flag_rates[audit.flag_rates['scenario'] == base][['class_id', 'class_name', 'n_images', 'n_flagged', 'flag_rate_pct', 'ci95_low_pct', 'ci95_high_pct']])}

Highest baseline rate: {_name(top_cls.index[0])} ({top_cls.iloc[0]:.1f}%); lowest: {_name(top_cls.index[-1])} ({top_cls.iloc[-1]:.1f}%).
Class ranks under the three settings (1 = highest flag rate):

{md_table(ranks, index=True)}

Spearman rank correlation with the baseline ranking - {rho_txt}. The class with the highest flag rate is
{"the same in all settings" if same_top else "NOT the same in all settings (" + ", ".join(f"{k}: class {v}" for k, v in top_by_scn.items()) + ")"}.
Most common main reason among flagged images: **{top_reason}**.

**Q4 - Association between sharpness, contrast and brightness (Spearman).**
Sharpness vs contrast: rho = {s_rho:.2f} ({strength(s_rho)}); sharpness vs brightness: rho = {s_rho_b:.2f} ({strength(s_rho_b)}).
This describes how the measures move together in this sample; it does **not** show causation (image size, noise, compression
processing and scene content also affect all three measures).

**Q5 - Illustrative examples.** Figure 7 shows the top-priority flagged image and two heuristic cases found by checking the ROI crop against
the same cut-offs. Among images with a valid ROI, {ea['n_flagged_but_roi_looks_normal']} of {ea['n_flagged_with_valid_roi']} flagged images have a sign region
that looks normal (possible **false alarms**, e.g. dark background), and {ea['n_unflagged_but_roi_below_cutoffs']} of {ea['n_unflagged_with_valid_roi']} unflagged images
have a sign region below a cut-off (cases the whole-image rule may **miss**). These are heuristics, not ground truth.

**ROI crop vs full image (small sample).**

{md_table(audit.roi_comparison)}

## 6. Figures
{figs_md}

## 7. Decision memo (one page)
**To:** Road-safety review team | **Re:** How to prioritise manual review of GTSRB-style sign photographs

**Recommendation.** Review in this order: (1) the {n_high} *high-priority* images (three or more conditions); (2) the {n_med} *medium* images
(two conditions); (3) single-condition images, starting with `{top_reason}`, the most common reason. Use `outputs/processed/review_queue.csv`
(sorted by score, then lowest sharpness). Spot-check a small random sample of *unflagged* images each cycle to estimate what the rule misses.

**Where to look.** Highest baseline flag rates: {", ".join(f"{_name(c)} {top_cls[c]:.1f}%" for c in top_cls.index[:2])}.
{"This ordering is stable across the alternative cut-offs." if all(v > 0.8 for v in audit.rank_corr.values()) else "This ordering changes noticeably across alternative cut-offs, so class-level conclusions should be treated cautiously."}
Prefer the stricter setting ({list(audit.scenarios)[1]}) when review capacity is small and the lenient setting when missing a poor image is costly.

**Evidence that would change this recommendation.** (a) A human-labelled subset showing that flagged images are not more often hard to read than
unflagged ones; (b) unstable class rankings under more cut-off settings; (c) ROI-based measures disagreeing strongly with whole-image measures
(then switch the rule to ROI crops); (d) data from local roads in different lighting/weather.

**Limitations.** See Section 8. The cut-offs are project-specific percentiles, not safety standards, and a flag is not a safety verdict.

## 8. Limitations
1. **Sample:** at most {audit.max_per_class} images per selected class, only {len(audit.selected_classes)} of 43 classes; results do not generalise to unselected classes. GTSRB contains
   several frames of the same physical sign (tracks), so images are not fully independent.
2. **Sharpness is a proxy:** Laplacian variance depends on image size, noise and texture, and is not a human readability score; small images are partly
   flagged because of their size.
3. **Percentile cut-offs are relative:** they always flag a fixed share of the sample, even if all images were acceptable; they are not safety thresholds.
4. **Dataset scope:** GTSRB shows German roads; it does not represent every local road environment, camera, or weather condition.
5. **Whole-image measures include background:** the sign region may differ (see ROI comparison); the ROI heuristic used for examples is not ground truth.

## 9. Reproducibility
Seed {audit.seed}; relative paths only; derived tables in `outputs/processed/`, figures in `outputs/figures/`. Re-run with `python run_audit.py`.
"""
    path = audit.output_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path
