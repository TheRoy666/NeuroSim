#!/usr/bin/env python3
"""
Decisive go/no-go check: does the raw_rho pinning anomaly corrupt
the actual quantity the TLE lateralization signal-check needs (per-node average_controllability L-R
asymmetry), or is it isolated to that one pre-normalization scalar?

normalise_matrix rescales every subject to identical target_rho=0.9
regardless of raw_rho -- so the real question is whether the NORMALIZED
matrix still carries genuine subject-specific structure, or whether
subjects are suspiciously similar to each other post-normalization too.

Checks, using files already on disk (no new EC computation needed):
  1. Does ec_asymmetry (already in unam_results.csv) show real spread,
     or is it also suspiciously pinned like raw_rho was?
  2. Do node_asymmetry values (unam_node_asymmetry.csv) show genuine
     inter-subject variation per region-pair, or are they collapsed
     to near-identical values across subjects?
  3. Cross-subject correlation: are any two random subjects' full
     34-region asymmetry vectors suspiciously highly correlated
     (would indicate the template SC is dominating post-normalization
     structure too, not just raw_rho)?

Usage:
    python sanity_check_asymmetry_variation.py \
        --results /path/to/unam_results.csv \
        --asymmetry /path/to/unam_node_asymmetry.csv
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--asymmetry", required=True)
    args = ap.parse_args()

    res = pd.read_csv(args.results)
    asym = pd.read_csv(args.asymmetry)
    pair_cols = [c for c in asym.columns if c.startswith("pair_")]

    print("=" * 60)
    print("CHECK 1: ec_asymmetry spread (already computed, in unam_results.csv)")
    print("=" * 60)
    ea = res["ec_asymmetry"]
    cv_ea = ea.std() / ea.mean()
    print(f"  ec_asymmetry: mean={ea.mean():.4f}  std={ea.std():.4f}  "
          f"CV={cv_ea:.4f}")
    print(f"  (compare: raw_rho CV was ~0.0001 -- pinned)")
    print(f"  {'PASS -- real variation' if cv_ea > 0.005 else 'FAIL -- also pinned'}")

    print(f"\n{'='*60}")
    print("CHECK 2: node_asymmetry spread across subjects (per region-pair)")
    print(f"{'='*60}")
    stds = asym[pair_cols].std()
    means = asym[pair_cols].mean().abs()
    cv_per_pair = stds / (means + 1e-8)
    print(f"  Per-region-pair CV across 62 subjects:")
    print(f"    median={cv_per_pair.median():.3f}  "
          f"range=[{cv_per_pair.min():.3f}, {cv_per_pair.max():.3f}]")
    n_degenerate = (stds < 1e-6).sum()
    print(f"  Region-pairs with near-zero std (all subjects identical): "
          f"{n_degenerate}/{len(pair_cols)}")
    print(f"  {'PASS -- real inter-subject variation' if n_degenerate == 0 else 'FAIL -- degenerate'}")

    print(f"\n{'='*60}")
    print("CHECK 3: pairwise subject correlation (are subjects too similar?)")
    print(f"{'='*60}")
    M = asym[pair_cols].values  # (62, 34)
    corr = np.corrcoef(M)
    off_diag = corr[np.triu_indices_from(corr, k=1)]
    print(f"  Pairwise subject-subject correlation (34-dim asymmetry vectors):")
    print(f"    mean={off_diag.mean():.3f}  median={np.median(off_diag):.3f}  "
          f"max={off_diag.max():.3f}")
    print(f"  (Near 1.0 for most pairs would mean subjects are collapsing")
    print(f"   to the same pattern -- template SC dominating post-normalization")
    print(f"   structure, not just raw_rho)")
    high_corr_frac = (off_diag > 0.9).mean()
    print(f"  Fraction of subject pairs with corr>0.9: {high_corr_frac:.1%}")
    print(f"  {'PASS -- subjects distinguishable' if high_corr_frac < 0.5 else 'FAIL -- subjects collapsing together'}")

    print(f"\n{'='*60}")
    print("OVERALL VERDICT")
    print(f"{'='*60}")
    checks = [cv_ea > 0.005, n_degenerate == 0, high_corr_frac < 0.5]
    if all(checks):
        print("ALL CHECKS PASS. The raw_rho pinning appears isolated to that")
        print("pre-normalization scalar. Post-normalization structure (what")
        print("the signal-check actually uses) shows genuine subject-specific variation.")
        print("-> Reasonable to proceed with the lateralization asymmetry statistic,")
        print("   with the raw_rho anomaly documented as an open limitation.")
    else:
        print("AT LEAST ONE CHECK FAILED. The anomaly may not be isolated --")
        print("post-normalization structure could be compromised too.")
        print("-> Do NOT proceed with lateralization statistics on this data yet.")
        print("   Needs further investigation before trusting node_asymmetry.csv.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()