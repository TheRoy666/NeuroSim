#!/usr/bin/env python3
"""
Cross-cohort N/T ratio comparison — tests whether UNAM's node-to-timepoint
ratio is a genuine outlier vs HCP/ADNI, as a candidate mechanism for the
raw_rho~1.0 pinning (independent of DTI availability or regularization,
since both lambda_graph and lambda_ridge sweeps came back negative).

Pulls actual N and T directly from each cohort's results CSV (not
approximate/remembered values) so the comparison is exact.

Usage:
    python compare_N_T_ratio.py \
        --hcp   /path/to/aud_cohort_results_public.csv \
        --adni-s400  /path/to/adni_results_schaefer400_public.csv \
        --adni-tians3 /path/to/adni_results_tians3_public.csv \
        --unam  /path/to/unam_results.csv
"""
import argparse
import pandas as pd
import numpy as np


def summarize(name, df, n_col="N", t_col="T"):
    if n_col not in df.columns or t_col not in df.columns:
        print(f"  {name}: MISSING '{n_col}' or '{t_col}' column — check file")
        return None

    N = df[n_col].iloc[0]  # N should be constant within a cohort/atlas
    N_unique = df[n_col].nunique()
    if N_unique > 1:
        print(f"  {name}: WARNING — N varies within cohort ({N_unique} "
              f"distinct values: {sorted(df[n_col].unique())})")

    T = df[t_col]
    ratio = N / T  # per-subject N/T (T can vary per subject)

    print(f"\n  {name}:")
    print(f"    N (nodes):        {N}" + (" (varies!)" if N_unique > 1 else ""))
    print(f"    T (timepoints):   median={T.median():.0f}  "
          f"range=[{T.min():.0f}, {T.max():.0f}]")
    print(f"    N/T ratio:        median={ratio.median():.3f}  "
          f"range=[{ratio.min():.3f}, {ratio.max():.3f}]")

    return {"cohort": name, "N": N, "T_median": T.median(),
            "T_min": T.min(), "T_max": T.max(),
            "NT_ratio_median": ratio.median(),
            "NT_ratio_min": ratio.min(), "NT_ratio_max": ratio.max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp", required=True)
    ap.add_argument("--adni-s400", required=True)
    ap.add_argument("--adni-tians3", required=True)
    ap.add_argument("--unam", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("CROSS-COHORT N/T RATIO COMPARISON")
    print("(Testing whether UNAM's node/timepoint ratio is an outlier,")
    print(" independent of DTI availability or regularization weight —")
    print(" both lambda_graph and lambda_ridge sweeps came back negative)")
    print("=" * 60)

    rows = []
    r = summarize("HCP AUD",         pd.read_csv(args.hcp));         rows.append(r)
    r = summarize("ADNI Schaefer400", pd.read_csv(args.adni_s400));  rows.append(r)
    r = summarize("ADNI TianS3",      pd.read_csv(args.adni_tians3)); rows.append(r)
    r = summarize("UNAM DK-68",       pd.read_csv(args.unam));       rows.append(r)

    rows = [r for r in rows if r is not None]
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv("NT_ratio_comparison.csv", index=False)

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))

    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print("If UNAM's N/T ratio is substantially higher than all three")
    print("DTI-based cohorts -> supports N/T ratio (not DTI availability)")
    print("as the candidate mechanism for raw_rho pinning at ~1.0.")
    print("If UNAM's ratio is similar to e.g. ADNI's -> ratio alone doesn't")
    print("explain it either; look for a different UNAM-specific property")
    print("(e.g. BOLD preprocessing differences, TR, spatial smoothness")
    print("from the surface-based fsaverage5 parcellation vs volumetric).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()