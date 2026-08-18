#!/usr/bin/env python3
"""
Recheck the N=410 (HCP-AUD) rank-stability timing anomaly from the Path B
benchmark: driver_node_rank_stability took ~2x longer at N=410 than at
either N=400 or N=450, with no algorithmic reason (same pair count, cost
depends only on N and n_boot, both fixed). Likely server contention noise,
same signature as the earlier A1 compute_gramian_doubling spike at N=410.
Reruns the N=410 case 3x to check for repeatability.

Run directly: `python3 recheck_n410_path_b_timing.py`
"""
import benchmark_path_b_bootstrap_cost as bench

print("Rerunning HCP-AUD (N=410, T=197) 3x to check whether the")
print("driver_node_rank_stability spike repeats (real) or doesn't (noise).\n")

for run in range(3):
    print(f"\n{'#'*70}\nRUN {run+1} of 3\n{'#'*70}")
    bench.benchmark_dims(410, 197, f"HCP-AUD recheck run {run+1}",
                          n_boot_test=20, n_boot_target=200)
