#!/usr/bin/env python3
"""
TLE lateralization signal-check — the actual lateralization statistic.

Region-of-interest sets (pre-specified BEFORE looking at results, per DK-68
ENIGMA order — LH indices 0-33, matching pair_00..pair_33 in
unam_node_asymmetry.csv, since node_asymmetry = avg_ctrl[L] - avg_ctrl[R]
computed per homologous region pair).

PRIMARY set — mesial temporal lobe, the core TLE pathology target
(entorhinal, parahippocampal, fusiform, temporal pole — the structures
most directly implicated in hippocampal sclerosis / MTL epilepsy):
    pair_04 (entorhinal), pair_05 (fusiform),
    pair_14 (parahippocampal), pair_31 (temporalpole)

SECONDARY/EXPLORATORY set — broader temporal neocortex (reported
separately, not used to inflate the primary claim):
    pair_00 (bankssts), pair_07 (inferiortemporal), pair_13 (middletemporal),
    pair_28 (superiortemporal), pair_32 (transversetemporal), pair_33 (insula)

PREDICTION DIRECTION (stated a priori, not tuned to the data):
    Epilepsy/NCT literature motivates reduced controllability in the
    epileptogenic hemisphere (lesion-associated network disruption).
    Prediction rule: primary_MTL_asymmetry (L-R) < 0  -> predict Left-TLE
                      primary_MTL_asymmetry (L-R) > 0  -> predict Right-TLE
    This direction is fixed before computing accuracy -- flipping it
    post-hoc to maximize apparent accuracy would be circular and is
    explicitly NOT done here.

Two independent tests:
  1. Binomial classification: does the primary MTL asymmetry score predict
     TLEside better than chance (50%), reported with exact binomial CI.
  2. Convergent validity: does the primary MTL asymmetry score correlate
     with the independent structural ground truth (hippocampal volume
     asymmetry), across the full cohort including healthy controls.

Usage:
    python tle_lateralization_test.py \
        --results   /path/to/unam_results.csv \
        --asymmetry /path/to/unam_node_asymmetry.csv \
        --out-dir   /path/to/output
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_ROIS = {
    "pair_04_LminusR": "entorhinal",
    "pair_05_LminusR": "fusiform",
    "pair_14_LminusR": "parahippocampal",
    "pair_31_LminusR": "temporalpole",
}
SECONDARY_ROIS = {
    "pair_00_LminusR": "bankssts",
    "pair_07_LminusR": "inferiortemporal",
    "pair_13_LminusR": "middletemporal",
    "pair_28_LminusR": "superiortemporal",
    "pair_32_LminusR": "transversetemporal",
    "pair_33_LminusR": "insula",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--asymmetry", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    res = pd.read_csv(args.results)
    asym = pd.read_csv(args.asymmetry)
    df = res.merge(asym, on=["subject_id", "Group_detail"], suffixes=("", "_asym"))

    # ── Compute ROI-set summary scores per subject (MEAN and MEDIAN) ───────
    # Outlier check found extreme-magnitude values (10-30x typical) in a
    # handful of subjects, only weakly/ambiguously linked to Gramian
    # conditioning (Spearman rho=0.25, p=0.049 -- not a clean explanation).
    # Report both statistics transparently rather than picking one.
    df["primary_MTL_asym"] = df[list(PRIMARY_ROIS.keys())].mean(axis=1)
    df["primary_MTL_asym_median"] = df[list(PRIMARY_ROIS.keys())].median(axis=1)
    df["secondary_temporal_asym"] = df[list(SECONDARY_ROIS.keys())].mean(axis=1)
    df["secondary_temporal_asym_median"] = df[list(SECONDARY_ROIS.keys())].median(axis=1)

    print("=" * 60)
    print("PRE-SPECIFIED ROI SETS")
    print("=" * 60)
    print(f"Primary (MTL):    {list(PRIMARY_ROIS.values())}")
    print(f"Secondary (temporal neocortex): {list(SECONDARY_ROIS.values())}")

    # ── TEST 1: Binomial classification, primary MTL score only ────────────
    print(f"\n{'='*60}")
    print("TEST 1 — Binomial classification (PRIMARY, pre-specified)")
    print(f"{'='*60}")

    lateralizable = df[df["TLEside"].isin(["Left-TLE", "Right-TLE"])].copy()
    lateralizable["predicted"] = np.where(
        lateralizable["primary_MTL_asym"] < 0, "Left-TLE", "Right-TLE")
    lateralizable["correct"] = (lateralizable["predicted"] == lateralizable["TLEside"])

    n = len(lateralizable)
    k = lateralizable["correct"].sum()
    acc = k / n
    binom = stats.binomtest(k, n, p=0.5, alternative="two-sided")
    ci = binom.proportion_ci(confidence_level=0.95)

    print(f"  N lateralizable patients: {n} "
          f"(Left={sum(lateralizable.TLEside=='Left-TLE')}, "
          f"Right={sum(lateralizable.TLEside=='Right-TLE')})")
    print(f"  [MEAN]   Correct predictions: {k}/{n} = {acc:.1%}")
    print(f"  [MEAN]   95% exact binomial CI: [{ci.low:.1%}, {ci.high:.1%}]")
    print(f"  [MEAN]   p-value (vs 50% chance): {binom.pvalue:.4f}")

    # Robustness check: same test using MEDIAN across ROI regions
    lateralizable["predicted_median"] = np.where(
        lateralizable["primary_MTL_asym_median"] < 0, "Left-TLE", "Right-TLE")
    lateralizable["correct_median"] = (
        lateralizable["predicted_median"] == lateralizable["TLEside"])
    k_med = lateralizable["correct_median"].sum()
    acc_med = k_med / n
    binom_med = stats.binomtest(k_med, n, p=0.5, alternative="two-sided")
    ci_med = binom_med.proportion_ci(confidence_level=0.95)
    print(f"\n  [MEDIAN, robustness check] Correct: {k_med}/{n} = {acc_med:.1%}")
    print(f"  [MEDIAN] 95% exact binomial CI: [{ci_med.low:.1%}, {ci_med.high:.1%}]")
    print(f"  [MEDIAN] p-value: {binom_med.pvalue:.4f}")
    print(f"\n  Mean vs median agreement: "
          f"{(lateralizable['predicted']==lateralizable['predicted_median']).mean():.1%} "
          f"of subjects classified the same way by both statistics")
    print(f"  {'SIGNIFICANT (mean)' if binom.pvalue < 0.05 else 'not significant (mean)'}, "
          f"{'SIGNIFICANT (median)' if binom_med.pvalue < 0.05 else 'not significant (median)'} "
          f"at alpha=0.05")

    # ── TEST 1b: same test, secondary/exploratory ROI set ──────────────────
    print(f"\n--- Secondary/exploratory (broader temporal neocortex) ---")
    lateralizable["predicted_sec"] = np.where(
        lateralizable["secondary_temporal_asym"] < 0, "Left-TLE", "Right-TLE")
    lateralizable["correct_sec"] = (lateralizable["predicted_sec"] == lateralizable["TLEside"])
    k_sec = lateralizable["correct_sec"].sum()
    acc_sec = k_sec / n
    binom_sec = stats.binomtest(k_sec, n, p=0.5, alternative="two-sided")
    ci_sec = binom_sec.proportion_ci(confidence_level=0.95)
    print(f"  Correct predictions: {k_sec}/{n} = {acc_sec:.1%}")
    print(f"  95% exact binomial CI: [{ci_sec.low:.1%}, {ci_sec.high:.1%}]")
    print(f"  p-value: {binom_sec.pvalue:.4f}  (EXPLORATORY -- do not use to")
    print(f"  inflate the primary claim; report alongside, not instead of, Test 1)")

    # ── TEST 2: Convergent validity vs hippocampal volume asymmetry ────────
    print(f"\n{'='*60}")
    print("TEST 2 — Convergent validity: MTL asymmetry vs hippocampal volume")
    print(f"{'='*60}")

    df["hipp_vol_asym"] = df["L_hipp_vol"] - df["R_hipp_vol"]
    valid = df.dropna(subset=["primary_MTL_asym", "hipp_vol_asym"])
    print(f"  N with valid hippocampal volumes: {len(valid)} / {len(df)}")

    rho, p = stats.spearmanr(valid["primary_MTL_asym"], valid["hipp_vol_asym"])
    print(f"  Spearman rho (all groups, N={len(valid)}): {rho:.3f}  p={p:.4f}")

    tle_only = valid[valid["TLEside"].isin(["Left-TLE", "Right-TLE"])]
    if len(tle_only) >= 5:
        rho_tle, p_tle = stats.spearmanr(
            tle_only["primary_MTL_asym"], tle_only["hipp_vol_asym"])
        print(f"  Spearman rho (TLE patients only, N={len(tle_only)}): "
              f"{rho_tle:.3f}  p={p_tle:.4f}")

    # ── Save everything ──────────────────────────────────────────────────
    df.to_csv(os.path.join(args.out_dir, "unam_lateralization_scores.csv"), index=False)
    lateralizable.to_csv(
        os.path.join(args.out_dir, "unam_lateralization_classification.csv"), index=False)

    summary = pd.DataFrame([
        {"test": "primary_MTL_classification", "n": n, "correct": k,
         "accuracy": acc, "ci_low": ci.low, "ci_high": ci.high, "p_value": binom.pvalue},
        {"test": "secondary_temporal_classification", "n": n, "correct": k_sec,
         "accuracy": acc_sec, "ci_low": ci_sec.low, "ci_high": ci_sec.high,
         "p_value": binom_sec.pvalue},
        {"test": "hippocampal_convergent_validity_all", "n": len(valid),
         "correct": np.nan, "accuracy": np.nan, "ci_low": np.nan, "ci_high": np.nan,
         "p_value": p, "rho": rho},
    ])
    summary.to_csv(os.path.join(args.out_dir, "unam_lateralization_summary.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Saved: unam_lateralization_scores.csv, unam_lateralization_classification.csv, "
          f"unam_lateralization_summary.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()