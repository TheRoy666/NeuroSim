#!/usr/bin/env python3
"""
EC-estimation uncertainty compute-cost benchmark -- EC estimation and moving-block bootstrap
cost at real ADNI scale, on real server hardware, before committing to
the full batch (per the pre-registered analysis plan: "benchmark on 2-3 subjects
first").

Uses synthetic BOLD-like data (random-walk, autocorrelated) and a
synthetic symmetric SC matrix -- not scientifically meaningful, purely for
timing, since real per-subject files aren't available in this environment.
Timing depends on N, T, n_boot -- not on the specific values inside the
matrices -- so this gives an honest cost estimate.

Run directly on the server: `python3 benchmark_ec_bootstrap_cost.py`
"""
import time
import numpy as np
from neurosim import connectivity
from neurosim import physics

REAL_COHORT_DIMS = {
    "UNAM (N=68, T~197/130)": (68, 197),
    "ADNI-Schaefer400 (N=400, T=197)": (400, 197),
    "ADNI-Schaefer400+TianS3 (N=450, T=197)": (450, 197),
    "HCP-AUD (N=410, T unknown -- using 197 as placeholder)": (410, 197),
}


def make_synthetic_data(N, T, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, T)).cumsum(axis=1) * 0.01  # autocorrelated
    SC = rng.uniform(0, 1, (N, N))
    SC = (SC + SC.T) / 2
    np.fill_diagonal(SC, 0)
    return X, SC


def ctrl_func(A):
    A_norm = physics.normalise_matrix(A, target_rho=0.9)
    return physics.average_controllability(A_norm)


def benchmark_dims(N, T, label, n_boot_test=20, n_boot_target=200,
                    block_length=15):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    X, SC = make_synthetic_data(N, T)

    t0 = time.time()
    _ = connectivity.graphnet_effective_connectivity(X, SC)
    t_single = time.time() - t0
    print(f"  Single graphnet_effective_connectivity call: {t_single:.4f}s")

    t0 = time.time()
    EC_boot = connectivity.block_bootstrap_ec(
        X, SC, n_boot=n_boot_test, block_length=block_length, seed=0)
    t_boot_small = time.time() - t0
    per_boot = t_boot_small / n_boot_test
    print(f"  block_bootstrap_ec (n_boot={n_boot_test}): {t_boot_small:.4f}s "
          f"total, {per_boot:.4f}s/resample")

    t0 = time.time()
    result = connectivity.driver_node_rank_stability(EC_boot, ctrl_func, top_k=5)
    t_rank = time.time() - t0
    n_pairs = n_boot_test * (n_boot_test - 1) // 2
    print(f"  driver_node_rank_stability (n_boot={n_boot_test}, {n_pairs} pairs): "
          f"{t_rank:.4f}s")

    est_boot_full = per_boot * n_boot_target
    est_pairs_full = n_boot_target * (n_boot_target - 1) // 2
    est_rank_full = (t_rank / n_pairs) * est_pairs_full if n_pairs > 0 else 0
    est_total_per_subject = est_boot_full + est_rank_full

    print(f"\n  Extrapolated to n_boot={n_boot_target} (the analysis plan target):")
    print(f"    Bootstrap EC estimation: {est_boot_full:.1f}s ({est_boot_full/60:.1f} min)")
    print(f"    Rank-stability computation: {est_rank_full:.1f}s ({est_rank_full/60:.1f} min)")
    print(f"    TOTAL per subject (sequential): {est_total_per_subject:.1f}s "
          f"({est_total_per_subject/60:.1f} min)")

    return est_total_per_subject


if __name__ == "__main__":
    print("EC-estimation uncertainty compute-cost benchmark -- synthetic data, real dimensions")
    print("(timing depends on N/T/n_boot, not connectome identity, so real")
    print("per-subject files are not needed for this step)\n")

    estimates = {}
    for label, (N, T) in REAL_COHORT_DIMS.items():
        est = benchmark_dims(N, T, label)
        estimates[label] = est

    print(f"\n{'='*70}\nSUMMARY -- per-subject cost (sequential), n_boot=200")
    print(f"{'='*70}")
    for label, est in estimates.items():
        print(f"  {label:>50}: {est:>8.1f}s  ({est/60:>6.1f} min)")

    print("\nNOTE: given the server's confirmed 80-core capacity, this is")
    print("embarrassingly parallel across subjects (or across bootstrap")
    print("resamples within a subject) -- if N_subjects <= n_cores, expect")
    print("near-total wall-clock reduction to the per-subject estimate above,")
    print("not the sum across all subjects.")