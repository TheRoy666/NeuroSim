#!/usr/bin/env python3
"""
Frassle & Stephan (2022) cross-check: they found, via real repeated
scanning (not resampling), that stronger EC connections are more
reliable (test-retest ICC increases with connection strength) --
consistent across nearly every condition in their paper, most robustly
for resting-state data.

This tests the same qualitative relationship using our own bootstrap-
derived per-connection uncertainty (moving-block resampling on a single
session, not repeated scanning -- a genuinely different method) as an
independent cross-check: does connection strength ALSO predict tighter
bootstrap confidence intervals in our data?

Note on scope, decided after checking what Frassle & Stephan actually
publish: they report distributional summaries (density plots, mean ICC
by subset), not a literal per-connection lookup table indexed by region
pair. A direct connection-by-connection join against their data isn't
well-defined across different atlases and different estimation methods
(rDCM vs. our graphnet EC). The strength-vs-reliability relationship is
their own headline, most robustly reported finding, and is method/atlas-
independent by construction -- a more defensible cross-check than
attempting literal connection matching would have been.

Requires per-subject *_EC_boot.npy files (n_boot x N x N arrays),
produced by run_ec_bootstrap_batch.py with --save-ec-boot (not saved in
any of the existing real-data runs -- this is why new data was needed).
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def analyze_subject(ec_boot_path):
    """For one subject's bootstrap EC replicates, compute per-connection
    strength (mean across replicates) and uncertainty (std across
    replicates), then test whether strength predicts tighter
    (lower-std) intervals -- the same relationship Frassle & Stephan
    report via real repeated scanning.

    Reports BOTH raw std and coefficient of variation (std/|mean|) as
    uncertainty measures. Raw std mechanically tends to scale with
    connection magnitude for many estimators (more numerical room to
    vary), which is a DIFFERENT question from relative reliability --
    Frassle & Stephan's ICC is a normalized measure, not raw variance.
    Comparing raw std against raw strength risks a scale artifact rather
    than testing the actual relationship of interest. Added after the
    first real run showed a suspiciously clean, uniformly OPPOSITE-
    direction result (rho~0.5-0.55, p~0, identical across all 5
    subjects) -- a pattern more characteristic of a mechanical scaling
    relationship than a genuine, subtle reliability finding."""
    EC_boot = np.load(ec_boot_path)  # (n_boot, N, N)
    n_boot, N, _ = EC_boot.shape

    mean_ec = EC_boot.mean(axis=0)
    std_ec = EC_boot.std(axis=0)

    mask = ~np.eye(N, dtype=bool)
    strength = np.abs(mean_ec[mask])
    uncertainty_raw = std_ec[mask]
    uncertainty_cv = std_ec[mask] / (strength + 1e-10)  # coefficient of variation

    rho_raw, p_raw = spearmanr(strength, uncertainty_raw)
    rho_cv, p_cv = spearmanr(strength, uncertainty_cv)

    return {
        "n_boot": n_boot, "N": N, "n_connections": mask.sum(),
        "spearman_rho_strength_vs_raw_std": rho_raw,
        "p_value_raw_std": p_raw,
        "direction_matches_frassle_stephan_raw_std": rho_raw < 0,
        "spearman_rho_strength_vs_coefficient_of_variation": rho_cv,
        "p_value_cv": p_cv,
        "direction_matches_frassle_stephan_cv": rho_cv < 0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec-boot-dir", required=True,
                     help="Directory containing *_EC_boot.npy files")
    ap.add_argument("--out", default="frassle_stephan_cross_check_results.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ec_boot_dir, "*_EC_boot.npy")))
    print(f"Found {len(files)} subject EC_boot files")

    rows = []
    for f in files:
        sub = os.path.basename(f).replace("_EC_boot.npy", "")
        result = analyze_subject(f)
        result["subject_id"] = sub
        rows.append(result)
        print(f"  {sub}:")
        print(f"    raw std:  rho={result['spearman_rho_strength_vs_raw_std']:.4f}, "
              f"p={result['p_value_raw_std']:.2e}, "
              f"matches F&S direction: {result['direction_matches_frassle_stephan_raw_std']}")
        print(f"    coef.var: rho={result['spearman_rho_strength_vs_coefficient_of_variation']:.4f}, "
              f"p={result['p_value_cv']:.2e}, "
              f"matches F&S direction: {result['direction_matches_frassle_stephan_cv']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_matching_raw = df["direction_matches_frassle_stephan_raw_std"].sum()
    n_matching_cv = df["direction_matches_frassle_stephan_cv"].sum()
    print(f"Raw std:        {n_matching_raw}/{len(df)} subjects match Frassle & Stephan's direction")
    print(f"Coef. variation: {n_matching_cv}/{len(df)} subjects match Frassle & Stephan's direction")
    print(f"\nSaved to {args.out}")
