import json, sys
from pathlib import Path
import numpy as np, pandas as pd
P = Path("/mnt/user-data/outputs/traffic-sign-audit/outputs/processed")
out = Path(sys.argv[1])
df = pd.read_csv(P/"image_features.csv"); cc = pd.read_csv(P/"class_counts_full_vs_sample.csv")
cfg = json.load(open(P/"run_config.json")); q = json.load(open(P/"data_quality.json"))
fr = pd.read_csv(P/"flag_rates_by_class_and_scenario.csv"); sens = pd.read_csv(P/"sensitivity_scenarios.csv")
eff = pd.read_csv(P/"effect_sizes.csv"); corr = pd.read_csv(P/"correlations_spearman.csv", index_col=0)
roi = pd.read_csv(P/"roi_vs_full_comparison.csv"); th = cfg["baseline_thresholds"]
names = df.groupby("class_id")["class_name"].first()
sel = cc[cc.selected].set_index("class_id")
def tbl(d):
    cols=[str(c) for c in d.columns]
    return "\n".join(["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]+["| "+" | ".join(str(v) for v in r)+" |" for r in d.values.tolist()])
n=len(df); nflag=int(df.review_flag.sum()); nh=int((df.priority_tier=="high").sum()); nm=int((df.priority_tier=="medium").sum())
g = df.groupby("class_id")
cls = pd.DataFrame({"Class":[f"{i} {names[i]}" for i in sel.index],"Full":sel.n_full.values,"Sample":sel.n_sampled.values,
  "Size (px)":g.min_side_px.median().loc[sel.index].round(0).astype(int).values,
  "Bright.":g.brightness_gray.median().loc[sel.index].round(1).values,
  "Contrast":g.contrast_gray.median().loc[sel.index].round(1).values,
  "Sharp.":g.sharpness_lapvar.median().loc[sel.index].round(0).astype(int).values})
wide = fr.pivot(index="class_id",columns="scenario",values="flag_rate_pct")[list(cfg["scenarios"])]
rank = wide.rank(ascending=False,method="min").astype(int)
base = fr[fr.scenario=="baseline_P10_P90"].set_index("class_id")
rt = pd.DataFrame({"Class":[f"{i} {names[i]}" for i in wide.index],
  "Baseline % (95% CI)":[f"{base.loc[i,'flag_rate_pct']:.1f} ({base.loc[i,'ci95_low_pct']:.1f}-{base.loc[i,'ci95_high_pct']:.1f})" for i in wide.index],
  "Strict %":wide.iloc[:,1].values,"Lenient %":wide.iloc[:,2].values,
  "Rank base/strict/lenient":[f"{a}/{b}/{c}" for a,b,c in rank.values]})
def sp(a,b):
    return float(np.corrcoef(a.rank(),b.rank())[0,1])
rho = {c:sp(wide.iloc[:,0],wide[c]) for c in wide.columns[1:]}
top = base.flag_rate_pct.sort_values(ascending=False)
r = df[df.roi_valid]
bad = (r.roi_contrast_gray<th["contrast_low"])|(r.roi_sharpness_lapvar<th["sharpness_low"])|(r.roi_brightness_gray<th["brightness_low"])|(r.roi_brightness_gray>th["brightness_high"])
fa=int((r.review_flag&~bad).sum()); nf=int(r.review_flag.sum()); ms=int((~r.review_flag&bad).sum()); nu=int((~r.review_flag).sum())
reason = df[df.review_flag].flag_reason.value_counts()
st = sens.copy(); st = st[["scenario","brightness_low","brightness_high","contrast_low","sharpness_low","min_side_low","n_flagged","pct_flagged"]]
st["scenario"]=st["scenario"].map({"baseline_P10_P90":"Baseline","strict_P5_P95":"Strict","lenient_P20_P80":"Lenient"})
st.columns=["Setting","Dark <","Bright >","Low contrast <","Blurry <","Small <","Flagged","% flagged"]
sr = corr.loc["sharpness_lapvar","contrast_gray"]; sb = corr.loc["sharpness_lapvar","brightness_gray"]
e = eff.set_index("feature")
md = f"""---
title: "Traffic Sign Visibility Audit"
subtitle: "Analysis report (GTSRB) - Saad Saeed ALRuwaili - Advanced AI Programming and Frameworks"
---

## 1. Case question
Which traffic-sign images should a human review first, and what evidence supports that priority? The audit measures brightness, contrast and a sharpness proxy with OpenCV, builds a transparent percentile rule with NumPy, and compares classes with pandas and Matplotlib. The flag is a **triage aid only**; it does not prove that a sign is unsafe, unreadable or misclassified. No deep-learning model is used.

## 2. Data and preparation
**Source.** German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum (<https://benchmark.ini.rub.de/gtsrb_dataset.html>; archival copy <https://zenodo.org/records/13741936>). Cite: Stallkamp, Schlipsing, Salmen and Igel (2012), *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*, Neural Networks 32, 323-332. The archive is not redistributed.

**Full dataset (not sampled).** {q['n_rows']:,} annotation rows, {q['n_classes']} classes. Checks: duplicate rows {q['n_duplicate_rows']}, missing values 0, invalid image paths {q['n_invalid_image_paths']}, invalid class IDs {q['n_invalid_class_id']}, invalid dimensions {q['n_invalid_dimensions']}, invalid or absent ROI {q['n_invalid_or_absent_roi']}. Column types: Width, Height and ROI corners float64 (after coercion), ClassId Int64, Filename string. The Kaggle-style `Train.csv` stores paths from another computer; the foreign first path component was ignored and logged. Nothing was dropped silently (`exclusion_log.csv`); ROI coordinates are validated before cropping.

**Sample (separate from the full counts).** Six classes, up to {cfg['max_per_class']} random images each (seed {cfg['seed']}): **{n:,} images**, 0 skipped. Class 0 has only 210 images, so the sample keeps all of them; the other classes are capped, which reduces but does not remove the imbalance (full-data ratio 2,220 : 210 = 10.6 : 1).

## 3. Feature definitions
OpenCV reads images in BGR; they are converted to RGB only for Matplotlib display.

| Feature | Definition | Unit |
|---|---|---|
| Brightness | mean of the grayscale image | 0-255 |
| Contrast | standard deviation of the grayscale image | 0-255 |
| Sharpness (proxy) | variance of `cv2.Laplacian(gray, CV_64F)`; depends on size and noise, not a human readability score | unit-less |
| Size | shorter image side | pixels |
| ROI versions | same three measures on the annotated sign crop (valid ROI only) | as above |

## 4. Review-priority rule (project choice, not a safety standard)
An image meets a condition if it is too dark (brightness < P10), overexposed (> P90), low contrast (< P10), blurry by proxy (sharpness < P10) or small (shorter side < P10). Percentiles are computed on the sample with `np.percentile`; flags are vectorised NumPy masks and `np.select`. Priority score = number of conditions met; flagged = score >= 1 (high >= 3, medium 2, low 1). Baseline flags **{nflag:,} of {n:,} images ({100*nflag/n:.1f}%)**: {nh} high, {nm} medium priority.

{tbl(st)}

*Sensitivity check: strict (P5/P95) and lenient (P20/P80) settings change how many images are flagged by a factor of about four, which is why the cut-offs must be labelled as project choices.*

## 5. Findings
![Figure 1. Contact sheet: flagged (top) and unflagged (bottom) example per class. B = brightness, C = contrast, S = sharpness proxy.](figures/fig01_contact_sheet.png){{width=6.3in}}

**Q1 - Class sizes and imbalance.** Largest class: 1 Speed limit 30 (2,220 images); smallest: 0 Speed limit 20 (210). Small classes have wider confidence intervals (Figure 4), and sample-wide percentile cut-offs are dominated by the larger classes.

**Q2 - Size, brightness, contrast, sharpness by class** (medians; Figures 3 and 5).

{tbl(cls)}

Class explains a medium share of variance for every feature (eta-squared: size {e.loc['min_side_px','eta_squared']}, brightness {e.loc['brightness_gray','eta_squared']}, contrast {e.loc['contrast_gray','eta_squared']}, log-sharpness {e.loc['log10_sharpness','eta_squared']}), so class differences matter for review priority. The largest gaps: No entry has the smallest images (34 px) and Road work the darkest and least sharp ones.

![Figure 2. Brightness and sharpness distributions with the baseline cut-off lines.](figures/fig02_distributions_with_cutoffs.png){{width=6in}}

![Figure 3. Class-wise boxplots of brightness, contrast and sharpness.](figures/fig03_boxplots_by_class.png){{width=6.3in}}

**Q3 - Flag rate by class and ranking stability.**

{tbl(rt)}

![Figure 4. Review-flag rate by class under three cut-off settings (95% Wilson intervals).](figures/fig04_flag_rate_by_class.png){{width=5.4in}}

No entry is ranked first and Yield second under all three settings; the other ranks move (Spearman with the baseline ranking: strict {rho['strict_P5_P95']:.2f}, lenient {rho['lenient_P20_P80']:.2f}). Stop has the lowest baseline rate ({top.iloc[-1]:.1f}%). The most common main reason among flagged images is **{reason.index[0].replace('_',' ')}** ({int(reason.iloc[0])} images).

![Figure 5. Image size (left) and mean contrast (right) by class.](figures/fig05_size_and_contrast_by_class.png){{width=6in}}

**Q4 - Association (Spearman).** Sharpness vs contrast: rho = {sr:.2f}; sharpness vs brightness: rho = {sb:.2f}. Both are strong and positive in this sample: darker or flatter images also tend to score lower on the sharpness proxy. This is an association, not causation; size, noise and scene content affect all three measures.

**Q5 - Three illustrative images (Figure 6).** Left: the highest-priority flagged image. Middle and right: a possible false alarm and a possible miss, found by checking the ROI crop against the same cut-offs. Among images with a valid ROI, {fa} of {nf} flagged images have a sign region that looks normal (possible false alarms, e.g. dark background), and {ms} of {nu} unflagged images have a sign region below a cut-off (cases the rule may miss). ROI and full-image measures agree well (Spearman {roi.spearman_full_vs_roi.min():.2f}-{roi.spearman_full_vs_roi.max():.2f}), though ROI sharpness is higher on average. These are heuristics, not ground truth.

![Figure 6. Three examples illustrating the rule (green box = annotated sign region).](figures/fig07_three_examples.png){{width=6.1in}}

## 6. Decision memo
**To:** road-safety review team. **Recommendation.** Review in this order: (1) the {nh} high-priority images (three or more conditions); (2) the {nm} medium-priority images; (3) single-condition images, starting with low sharpness. Use `outputs/processed/review_queue.csv` (sorted by score, then lowest sharpness) and spot-check a small random sample of unflagged images each cycle to estimate what the rule misses. **Where to look:** the highest baseline flag rates are No entry ({top.iloc[0]:.1f}%) and Yield ({top.iloc[1]:.1f}%), and this ordering is stable across the three settings. Use the strict setting when review capacity is small and the lenient one when missing a poor image is costly. **What would change the recommendation:** (a) a human-labelled subset showing that flagged images are not harder to read than unflagged ones; (b) unstable class rankings under more cut-off settings; (c) ROI measures disagreeing strongly with whole-image measures, which would justify switching the rule to ROI crops; (d) data from local roads in other lighting and weather.

## 7. Limitations
1. **Sample:** at most {cfg['max_per_class']} images per class and only 6 of 43 classes; GTSRB contains several frames of each physical sign (tracks), so images are not independent.
2. **Sharpness is a proxy:** Laplacian variance depends on image size, noise and texture; small images are partly flagged because of their size.
3. **Percentile cut-offs are relative:** they flag a fixed share of any sample, even if all images were acceptable, and are not safety thresholds.
4. **Scope:** GTSRB shows German roads and does not represent every local road environment, camera or weather condition.
5. **Whole-image measures include background;** the false-alarm and miss counts are heuristics without human labels.

*Reproducibility: seed {cfg['seed']}, relative paths, derived tables in `outputs/processed/`, figures in `outputs/figures/`; the repository also contains Figure 6 (sharpness vs contrast and brightness) and an ROI-versus-full-image figure not reproduced here.*
"""
out.write_text(md, encoding="utf-8"); print("written", len(md.split()), "words")
