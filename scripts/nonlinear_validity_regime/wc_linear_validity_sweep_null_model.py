#!/usr/bin/env python3
"""
Path A1 -- null-model comparison. Answers the question the real-coupling
sweep alone cannot: does the interior-minimum finding depend on the
SPECIFIC topology of real brain connectomes, or would it appear in any
network with the same degree sequence and the same edge-weight
distribution?

Takes each subject's real SC matrix, applies degree-preserving rewiring
(Maslov-Sneppen double-edge-swap -- see null_model_rewiring.py for the
full method, including why a standard textbook implementation had to be
redesigned to be tractable at real connectome scale) plus a random
redistribution of the real edge weights onto the new topology, then
feeds the rewired matrix through EXACTLY the same downstream pipeline as
the real-coupling sweep (build_network, check_stable_fixed_point,
process_one_combo -- imported directly from
wc_linear_validity_sweep_real_coupling.py, not reimplemented) so the
ONLY thing that differs between a real-coupling run and this run is the
network topology itself.

Run directly:
    python3 wc_linear_validity_sweep_null_model.py \\
        --sc-dir /path/to/SC_matrices --sc-suffix _SC_SIFT2_410.csv \\
        --w-ee-values 3.0,3.3402,3.6805,4.0207,4.361 \\
        --n-workers 55
"""
import os
# Same fix as every other parallel script in this family -- see
# wc_linear_validity_sweep_real_coupling.py's docstring for the full
# explanation of why this matters.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from null_model_rewiring import (
    rewire_and_redistribute_weights,
    DEFAULT_N_SWAPS_PER_EDGE,
    DEFAULT_TIME_BUDGET_S,
    DEFAULT_MAX_ATTEMPTS_MULTIPLIER,
)
# Reusing the already-verified sweep machinery directly, not
# reimplementing it -- the only thing this script changes is what
# topology goes into build_network.
from wc_linear_validity_sweep_real_coupling import (
    discover_sc_files,
    build_network,
    check_stable_fixed_point,
    process_one_combo,
    W_EE_VALUES_HCP_DEFAULT,
)


def load_and_rewire(sc_path, n_swaps_per_edge, rewire_seed, time_budget_s,
                     max_attempts_multiplier=DEFAULT_MAX_ATTEMPTS_MULTIPLIER):
    """
    Load a real SC matrix and construct its degree-preserving null-model
    counterpart. The log1p-transform-then-row-sum-scale step applied
    here is IDENTICAL to load_and_scale_real_coupling's transform in
    wc_linear_validity_sweep_real_coupling.py, applied AFTER rewiring
    instead of directly on the real matrix -- duplicated here (not
    imported) deliberately, so this file's rewiring logic doesn't
    require importing anything from the already-deployed, currently-
    running real-coupling script beyond the pure-function machinery
    that doesn't touch data loading. Kept numerically identical on
    purpose: this transform's formula must match exactly for the
    real-vs-null comparison to be controlled on everything except
    topology.
    """
    SC_raw = np.loadtxt(sc_path, delimiter=",")
    N = SC_raw.shape[0]
    np.fill_diagonal(SC_raw, 0)

    SC_null, diagnostics = rewire_and_redistribute_weights(
        SC_raw, n_swaps_per_edge=n_swaps_per_edge,
        seed=rewire_seed, time_budget_s=time_budget_s,
        max_attempts_multiplier=max_attempts_multiplier)

    SC_log = np.where(SC_null > 0, np.log1p(SC_null), 0)
    target_row_sum_mean = (0.075 / N) * 20 * (N - 1)
    scale_factor = target_row_sum_mean / SC_log.sum(axis=1).mean()
    C_null = SC_log * scale_factor

    return C_null, N, diagnostics


def _rewire_one_subject(args):
    """Top-level function (not a closure/lambda) so it's picklable for
    multiprocessing.Pool -- rewiring is parallelized across subjects
    since it costs ~110s each at real connectome scale (measured
    directly, see null_model_rewiring.py), which would add ~15 minutes
    of purely sequential wait for an 8-subject pilot otherwise."""
    (subject_id, sc_path, n_swaps_per_edge, rewire_seed, time_budget_s,
     max_attempts_multiplier) = args
    C_null, N, diagnostics = load_and_rewire(
        sc_path, n_swaps_per_edge, rewire_seed, time_budget_s,
        max_attempts_multiplier)
    diagnostics["subject_id"] = subject_id
    return subject_id, C_null, N, diagnostics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", default="_SC_SIFT2_410.csv")
    ap.add_argument("--n-subjects", type=int, default=8,
                     help="Number of real subjects, matching the "
                          "real-coupling sweep's own N=8 pilot convention "
                          "so the two are directly comparable.")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--w-ee-values", default=None,
                     help="Comma-separated w_EE grid -- use the SAME "
                          "confirmed grid as the real-coupling run being "
                          "compared against, not a fresh one, or the "
                          "comparison isn't controlled.")
    ap.add_argument("--n-swaps-per-edge", type=int,
                     default=DEFAULT_N_SWAPS_PER_EDGE,
                     help=f"Rewiring target, as a multiple of edge count. "
                          f"Default {DEFAULT_N_SWAPS_PER_EDGE} -- see "
                          f"null_model_rewiring.py's module docstring for "
                          f"why this value, not the textbook-conservative "
                          f"10x (measured directly to be impractical at "
                          f"real connectome scale).")
    ap.add_argument("--rewire-seed", type=int, default=None,
                     help="Base seed for rewiring RNG. Each subject gets "
                          "seed + (index in sorted subject list), so runs "
                          "are reproducible but subjects don't share "
                          "identical rewiring randomness.")
    ap.add_argument("--rewire-time-budget-s", type=int,
                     default=DEFAULT_TIME_BUDGET_S)
    ap.add_argument("--max-attempts-multiplier", type=int,
                     default=DEFAULT_MAX_ATTEMPTS_MULTIPLIER,
                     help=f"Safety-valve ceiling on rewiring attempts, as "
                          f"a multiple of target_swaps. Previously "
                          f"hardcoded to {DEFAULT_MAX_ATTEMPTS_MULTIPLIER} "
                          f"-- exposed after directly observing this, not "
                          f"time, is what actually stops completion for "
                          f"some subjects (confirmed on real HCP data: "
                          f"subjects hit this ceiling using only a "
                          f"fraction of the time budget available). "
                          f"Raising this trades more attempts, and "
                          f"proportionally more time, for a chance at "
                          f"higher completion -- not guaranteed, since the "
                          f"rejection rate appears to climb as a network "
                          f"approaches full randomization, not stay "
                          f"constant.")
    ap.add_argument("--out", default="wc_linear_validity_sweep_null_model_results.csv")
    ap.add_argument("--rewiring-diagnostics-out",
                     default="null_model_rewiring_diagnostics.csv",
                     help="Where the per-subject rewiring diagnostics "
                          "(swaps achieved vs. target, time taken, "
                          "whether the time budget was hit) get saved -- "
                          "always written, never silently discarded.")
    args = ap.parse_args()

    n_workers = args.n_workers or cpu_count()
    print(f"Running with {n_workers} parallel workers ({cpu_count()} cores detected)")

    files = discover_sc_files(args.sc_dir, args.sc_suffix)
    subjects = sorted(files.keys())[:args.n_subjects]
    print(f"Found {len(files)} SC files; using {len(subjects)} as null-model seeds")

    if args.w_ee_values is not None:
        w_ee_grid = [float(x) for x in args.w_ee_values.split(",")]
    else:
        w_ee_grid = W_EE_VALUES_HCP_DEFAULT
        print("*** WARNING: no --w-ee-values given, using the HCP-confirmed "
              "default. Use the SAME grid as whichever real-coupling run "
              "this is being compared against. ***\n")
    print(f"w_EE grid: {w_ee_grid}")
    print(f"Rewiring target: {args.n_swaps_per_edge}x edge count per subject, "
          f"max_attempts ceiling: {args.max_attempts_multiplier}x target_swaps\n")

    # Rewiring step, parallelized across subjects
    rewire_tasks = [
        (sub, files[sub], args.n_swaps_per_edge,
         (args.rewire_seed + i) if args.rewire_seed is not None else None,
         args.rewire_time_budget_s, args.max_attempts_multiplier)
        for i, sub in enumerate(subjects)
    ]
    print("Rewiring all subjects (parallelized)...")
    t_rewire_start = time.time()
    with Pool(min(n_workers, len(subjects))) as pool:
        rewire_results = pool.map(_rewire_one_subject, rewire_tasks)
    t_rewire_elapsed = time.time() - t_rewire_start
    print(f"Rewiring complete: {t_rewire_elapsed:.1f}s\n")

    diagnostics_rows = []
    coupling_by_subject = {}
    for subject_id, C_null, N, diagnostics in rewire_results:
        coupling_by_subject[subject_id] = (C_null, N)
        diagnostics_rows.append(diagnostics)
        flag = " *** HIT TIME BUDGET ***" if diagnostics["hit_time_budget"] else ""
        print(f"  {subject_id}: {diagnostics['n_success']}/{diagnostics['target_swaps']} "
              f"swaps ({100*diagnostics['fraction_of_target']:.1f}%), "
              f"{diagnostics['elapsed_s']:.1f}s{flag}")

    diag_df = pd.DataFrame(diagnostics_rows)
    diag_df.to_csv(args.rewiring_diagnostics_out, index=False)
    print(f"\nRewiring diagnostics saved to {args.rewiring_diagnostics_out}")
    if diag_df["hit_time_budget"].any():
        n_hit = diag_df["hit_time_budget"].sum()
        print(f"*** WARNING: {n_hit}/{len(diag_df)} subjects hit the time "
              f"budget before reaching the full rewiring target -- check "
              f"{args.rewiring_diagnostics_out} before treating this run "
              f"as a fully-controlled comparison. ***\n")

    # Main sweep, identical machinery to the real-coupling script
    tasks = []
    for sub in subjects:
        C, N = coupling_by_subject[sub]
        for w_EE in w_ee_grid:
            tasks.append((sub, N, w_EE, C))

    t_start = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(process_one_combo, tasks)
    elapsed = time.time() - t_start

    rows = [row for sublist in results for row in sublist]
    n_skipped = sum(1 for r in results if len(r) == 0)
    print(f"Sweep complete: {len(rows)} runs in {elapsed:.1f}s "
          f"({n_skipped} (subject, w_EE) combos skipped as unstable/non-convergent)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(f"\nTotal wall-clock (rewiring + sweep): {t_rewire_elapsed + elapsed:.1f}s")
