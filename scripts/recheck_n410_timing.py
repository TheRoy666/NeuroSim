#!/usr/bin/env python3
"""
Recheck the N=410 (HCP-AUD) timing anomaly from the first benchmark run:
compute_gramian_doubling jumped ~3x at T=100/200 specifically at N=410,
while N=400 stayed flat across the same T range. Pure matrix-multiplication
has no reason to behave this way based on N alone -- most likely server
contention at that moment, not a real cost. This reruns N=410 three times
to check for repeatability.

Run directly: `python3 recheck_n410_timing.py`
"""
import numpy as np
import benchmark_wc_compute_cost as bench

print("Rerunning N=410 three times to check whether the T=100/200 spike")
print("in compute_gramian_doubling repeats (real) or doesn't (server noise).\n")

for run in range(3):
    print(f"\n{'#'*70}\nRUN {run+1} of 3\n{'#'*70}")
    timings, est = bench.benchmark_at_N(410, seed=0)
