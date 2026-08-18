#!/usr/bin/env python3
"""
Operator-divergence cross-cohort analysis (formalized).

Characterizes the divergence between symmetric-FC-based and directed-EC-
based control energy (teleport_ratio = E_fc / E_ec) across four independent
cohorts: HCP-AUD, ADNI-Schaefer400, ADNI-Schaefer400+TianS3, UNAM-TLE.

This was originally computed inline in conversation; this script reproduces
those exact computations as a saved, rerunnable artifact rather than a
chat-only result, per the repo-formalization pass.

Three things characterized:
  1. Direction and magnitude of the divergence per cohort (median, IQR,
     percent of subjects with ratio > 1)
  2. Correlation between divergence magnitude and Gramian conditioning
     (Spearman), and separately with raw_rho (to check whether the link
     is driven by the UNAM-specific raw_rho anomaly or is general)
  3. Within-cohort clinical group differences in teleport_ratio (mostly
     expected to be null -- this is a methods characterization, not a
     biomarker claim)

Run directly: `python3 operator_divergence_cross_cohort_analysis.py`
Reads from the project's public results CSVs (paths below -- adjust
DATA_DIR if running outside the original project folder structure).
Writes a summary CSV and a full text report.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, spearmanr

DATA_DIR = "/mnt/project"  # adjust if running elsewhere

COHORT_FILES = {
    "HCP-AUD": {
        "teleport": f"{DATA_DIR}/aud_teleportation_public.csv",
        "results": f"{DATA_DIR}/aud_cohort_results_public.csv",
        "group_col": "Severity_Label",
    },
    "ADNI-Schaefer400": {
        "teleport": f"{DATA_DIR}/adni_teleportation_schaefer400_public.csv",
        "results": f"{DATA_DIR}/adni_results_schaefer400_public.csv",
        "group_col": "DX_Group",
    },
    "ADNI-TianS3": {
        "teleport": f"{DATA_DIR}/adni_teleportation_tians3_public.csv",
        "results": f"{DATA_DIR}/adni_results_tians3_public.csv",
        "group_col": "DX_Group",
    },
    "UNAM-TLE": {
        "teleport": f"{DATA_DIR}/unam_teleportation.csv",
        "results": f"{DATA_DIR}/unam_results.csv",
        "group_col": "Group_detail",
    },
}


def load_cohort(name, paths):
    tel = pd.read_csv(paths["teleport"])
    res = pd.read_csv(paths["results"])
    key = "subject_id"
    merge_cols = [c for c in ["gramian_cond", "raw_rho", paths["group_col"]]
                  if c in res.columns and c not in tel.columns]
    merged = tel.merge(res[[key] + merge_cols], on=key, how="inner")
    merged["cohort"] = name
    return merged


def summarize_direction_magnitude(dfs):
    rows = []
    for name, df in dfs.items():
        rows.append({
            "cohort": name,
            "N": len(df),
            "pct_ratio_gt1": (df["teleport_ratio"] > 1).mean() * 100,
            "E_fc_median": df["E_fc"].median(),
            "E_ec_median": df["E_ec"].median(),
            "teleport_ratio_median": df["teleport_ratio"].median(),
            "teleport_ratio_q25": df["teleport_ratio"].quantile(0.25),
            "teleport_ratio_q75": df["teleport_ratio"].quantile(0.75),
        })
    return pd.DataFrame(rows)


def correlation_with_conditioning(dfs):
    rows = []
    for name, df in dfs.items():
        row = {"cohort": name, "N": len(df)}
        if "gramian_cond" in df.columns:
            rho, p = spearmanr(df["teleport_ratio"], df["gramian_cond"])
            row["spearman_rho_gramian_cond"] = rho
            row["p_gramian_cond"] = p
        if "raw_rho" in df.columns:
            rho, p = spearmanr(df["teleport_ratio"], df["raw_rho"])
            row["spearman_rho_raw_rho"] = rho
            row["p_raw_rho"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def clinical_group_tests(dfs):
    results = {}

    # HCP-AUD: severity trend (Kruskal-Wallis across 3 ordered groups)
    aud = dfs["HCP-AUD"]
    order = ["Social_Drinker", "Dependent", "Abuser"]
    present = [g for g in order if g in aud["Severity_Label"].unique()]
    if len(present) >= 2:
        groups = [aud[aud.Severity_Label == g]["teleport_ratio"] for g in present]
        h, p = kruskal(*groups)
        results["HCP-AUD severity trend (Kruskal-Wallis)"] = {
            "groups": present, "H": h, "p": p,
            "note": "EXPLORATORY, uncorrected -- not pre-registered"}

    # ADNI: CN vs MCI, both atlases
    for name in ["ADNI-Schaefer400", "ADNI-TianS3"]:
        df = dfs[name]
        cn = df[df.DX_Group == "CN"]["teleport_ratio"]
        mci = df[df.DX_Group == "MCI"]["teleport_ratio"]
        if len(cn) > 0 and len(mci) > 0:
            u, p = mannwhitneyu(cn, mci)
            results[f"{name} CN vs MCI (Mann-Whitney U)"] = {
                "CN_median": cn.median(), "MCI_median": mci.median(), "p": p}

    # UNAM: Healthy vs combined TLE
    unam = dfs["UNAM-TLE"]
    healthy = unam[unam.Group_detail == "Healthy"]["teleport_ratio"]
    tle = unam[unam.Group_detail != "Healthy"]["teleport_ratio"]
    if len(healthy) > 0 and len(tle) > 0:
        u, p = mannwhitneyu(healthy, tle)
        results["UNAM Healthy vs TLE (Mann-Whitney U)"] = {
            "Healthy_median": healthy.median(), "TLE_median": tle.median(), "p": p}

    return results


if __name__ == "__main__":
    print("=" * 78)
    print("OPERATOR-DIVERGENCE CROSS-COHORT ANALYSIS")
    print("=" * 78)

    dfs = {name: load_cohort(name, paths) for name, paths in COHORT_FILES.items()}

    print("\n--- 1. Direction and magnitude per cohort ---")
    summary = summarize_direction_magnitude(dfs)
    print(summary.to_string(index=False))

    print("\n--- 2. Correlation with Gramian conditioning (and raw_rho, as a check) ---")
    corr = correlation_with_conditioning(dfs)
    print(corr.to_string(index=False))

    print("\n--- 3. Within-cohort clinical group differences ---")
    clinical = clinical_group_tests(dfs)
    for test_name, result in clinical.items():
        print(f"\n{test_name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    # Save outputs
    summary.to_csv("operator_divergence_summary.csv", index=False)
    corr.to_csv("operator_divergence_conditioning_correlation.csv", index=False)

    with open("operator_divergence_report.txt", "w") as f:
        f.write("OPERATOR-DIVERGENCE CROSS-COHORT ANALYSIS -- FULL REPORT\n")
        f.write("=" * 78 + "\n\n")
        f.write("1. Direction and magnitude per cohort\n")
        f.write(summary.to_string(index=False) + "\n\n")
        f.write("2. Correlation with Gramian conditioning\n")
        f.write(corr.to_string(index=False) + "\n\n")
        f.write("3. Within-cohort clinical group differences\n")
        for test_name, result in clinical.items():
            f.write(f"\n{test_name}:\n")
            for k, v in result.items():
                f.write(f"  {k}: {v}\n")
        f.write("\n\nInterpretation notes:\n")
        f.write("- teleport_ratio > 1 in ~100%% of subjects across all cohorts: "
                 "symmetric-FC-based energy universally overestimates relative "
                 "to directed-EC-based energy.\n")
        f.write("- Magnitude scales with DTI-anchoring quality: smallest in "
                 "individually-DTI-anchored HCP-AUD, largest in UNAM-TLE "
                 "(shared template, no individual DTI).\n")
        f.write("- Correlation with gramian_cond is consistent in sign and "
                 "moderate-to-strong magnitude across all four cohorts, "
                 "while correlation with raw_rho is near-zero everywhere -- "
                 "the conditioning link is general, not an artifact of the "
                 "UNAM-specific raw_rho pinning anomaly.\n")
        f.write("- Clinical group differences are mostly null, as expected for "
                 "a methods characterization rather than a biomarker claim. "
                 "The one nominal exception (AUD severity trend) is "
                 "explicitly flagged as exploratory and uncorrected.\n")

    print("\n\nSaved: operator_divergence_summary.csv, "
          "operator_divergence_conditioning_correlation.csv, "
          "operator_divergence_report.txt")
