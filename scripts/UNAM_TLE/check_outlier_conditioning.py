#!/usr/bin/env python3
"""
Do the extreme-magnitude primary_MTL_asym outliers correlate with high
gramian_cond? If yes, they're numerical instability artifacts, not signal,
and Test 1/2 need to be rerun with a robust statistic (median instead of
mean, or explicit exclusion) before trusting the result.
"""
import argparse
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="unam_lateralization_scores.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)

    print("=== Distribution of primary_MTL_asym magnitude ===")
    abs_asym = df["primary_MTL_asym"].abs()
    print(f"  median={abs_asym.median():.3f}  "
          f"90th pctile={abs_asym.quantile(.9):.3f}  "
          f"max={abs_asym.max():.3f}")

    print("\n=== Top 8 subjects by |primary_MTL_asym|, vs gramian_cond ===")
    df["abs_MTL_asym"] = abs_asym
    top = df.nlargest(8, "abs_MTL_asym")
    print(top[["subject_id", "TLEside", "primary_MTL_asym",
               "abs_MTL_asym", "gramian_cond", "final_rho"]].to_string(index=False))

    print("\n=== Correlation: |primary_MTL_asym| vs gramian_cond ===")
    rho, p = np.corrcoef(abs_asym, df["gramian_cond"])[0, 1], None
    from scipy import stats
    rho_s, p_s = stats.spearmanr(abs_asym, df["gramian_cond"])
    print(f"  Pearson r:  {rho:.3f}")
    print(f"  Spearman rho: {rho_s:.3f}  p={p_s:.4f}")

    print(f"\n{'='*60}")
    if rho_s > 0.3:
        print("CONFIRMED: outlier magnitude correlates with Gramian conditioning.")
        print("These are numerical instability artifacts, not biological signal.")
        print("-> Rerun Test 1/2 using MEDIAN across ROI regions (robust to")
        print("   outliers) instead of MEAN, or explicitly exclude subjects")
        print("   above a gramian_cond threshold (e.g. >90th percentile) and")
        print("   report the excluded-N explicitly.")
    else:
        print("No strong link to conditioning found -- outliers may be a")
        print("genuine (if extreme) feature of these specific subjects.")
        print("Still recommend median-based robustness check before trusting")
        print("the mean-based classification result.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()