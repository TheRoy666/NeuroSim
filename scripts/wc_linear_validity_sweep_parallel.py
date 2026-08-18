#!/usr/bin/env python3
"""
Parallel WC linear-validity-regime sweep.

Same sweep as wc_linear_validity_sweep.py (target-distance, horizon,
bifurcation-proximity axes), but parallelized across (coupling_seed, w_EE)
combinations using multiprocessing -- these are independent, embarrassingly
parallel units of work (confirmed by the compute-cost benchmark). At N=410
this turns an estimated ~174 min sequential run into roughly ~22 min on an
8-core machine, since there are exactly 8 coupling seeds.

Refactored so the coupling matrix C can be swapped for a real per-subject
connectome later (see build_network) without changing the parallelization
logic -- currently still uses random-uniform coupling (mean-field scaled by
1/N, per the earlier N=100 convergence-failure fix) as a placeholder.

Run directly: `python3 wc_linear_validity_sweep_parallel.py [--n-regions N] [--n-workers W]`
Writes results to wc_linear_validity_sweep_results.csv in the same directory.
"""
import os
# Must happen before numpy/scipy are imported anywhere -- caps each worker
# process to a single BLAS thread, so parallelism comes from the process
# pool (n_workers processes), not from each process also fighting for all
# cores internally. Same missing fix found in
# wc_linear_validity_sweep_real_coupling.py -- see that script's comment
# for the full explanation. This script never had it either.
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
import simulation
import physics

DT = 2.0  # ms, per dt-calibration session (fine, decoupled from real TR)
BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                    tau_E=10.0, tau_I=20.0)

W_EE_VALUES = [3.0, 3.426, 3.852, 4.278, 4.704]  # FINAL: validated across all
                                                   # four real cohort N (68, 400,
                                                   # 410, 450) -- boundary is
                                                   # ~4.80 at every N, spread=0.000,
                                                   # confirming a single shared
                                                   # grid works for all cohorts
T_VALUES = [5, 20, 50, 100, 200]
TARGET_SCALES = [0.005, 0.01, 0.02, 0.05, 0.1]
N_COUPLING_SEEDS = 8
N_TARGET_DIRECTIONS_PER_SEED = 3


def build_network(N, w_EE, coupling_seed, real_C=None):
    """Build the WC network for one (seed, w_EE) combination.

    real_C : optional (N,N) ndarray -- pass a real per-subject connectome
        here to replace the random-uniform placeholder. Kept as a hook so
        swapping in real data doesn't require touching the parallelization
        logic below.
    """
    if real_C is not None:
        C = real_C.copy()
    else:
        rng = np.random.default_rng(coupling_seed)
        # mean-field coupling scale, N-independent (fixes the N=100
        # convergence failure found during benchmarking)
        C = rng.uniform(0, 0.15, (N, N)) / N * 20
    np.fill_diagonal(C, 0)
    params = dict(w_EE=w_EE, **BASE_PARAMS)
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)


def check_stable_fixed_point(net):
    try:
        E_star, I_star = net.find_fixed_point()
    except RuntimeError:
        return None
    J = net.jacobian_at(E_star, I_star)
    max_real_eig = np.linalg.eigvals(J).real.max()
    return E_star, I_star, max_real_eig, (max_real_eig < 0)


def process_one_combo(args):
    """The unit of parallel work: one (coupling_seed, w_EE) pair, expanded
    over the full T x target_scale x dir_seed sub-grid. Returns a list of
    row dicts (empty if the fixed point isn't stable/convergent)."""
    N, w_EE, coupling_seed, real_C = args
    net = build_network(N, w_EE, coupling_seed, real_C=real_C)
    fp_result = check_stable_fixed_point(net)
    if fp_result is None:
        return []
    E_star, I_star, max_real_eig, is_stable = fp_result
    if not is_stable:
        return []

    J = net.jacobian_at(E_star, I_star)
    G = simulation.input_jacobian_at(net, E_star, I_star)
    A, B = simulation.discretize_system(J, G, DT)
    rho_A = np.max(np.abs(np.linalg.eigvals(A)))
    x_star = np.concatenate([E_star, I_star])

    rows = []
    for T in T_VALUES:
        for target_scale in TARGET_SCALES:
            for dir_seed in range(N_TARGET_DIRECTIONS_PER_SEED):
                drng = np.random.default_rng(
                    hash((coupling_seed, int(w_EE * 10), T,
                          int(target_scale * 1000), dir_seed)) % (2**32)
                )
                direction = drng.standard_normal(N)
                direction /= np.linalg.norm(direction)
                target_perturb = target_scale * direction

                delta_x0 = np.zeros(2 * N)
                delta_xT = np.concatenate([target_perturb, np.zeros(N)])

                energy, U = physics.minimum_energy_trajectory(
                    A, B, delta_x0, delta_xT, T)
                u_func = physics.zero_order_hold(U, DT)

                result = net.simulate_controlled(
                    u_func=u_func, t_span=(0.0, T * DT),
                    n_points=max(100, T), E0=E_star, I0=I_star,
                )
                realized_final = np.concatenate(
                    [result["E"][:, -1], result["I"][:, -1]])
                xT_absolute = x_star + delta_xT
                target_movement = np.linalg.norm(delta_xT)
                err = np.linalg.norm(realized_final - xT_absolute)
                rel_err = err / target_movement

                rows.append({
                    "coupling_seed": coupling_seed,
                    "w_EE": w_EE,
                    "stability_margin": -max_real_eig,
                    "rho_A": rho_A,
                    "T": T,
                    "target_scale": target_scale,
                    "dir_seed": dir_seed,
                    "energy": energy,
                    "rel_reachability_error": rel_err,
                })
    return rows


def run_sweep_parallel(N, n_workers=None, real_C_per_seed=None):
    """
    real_C_per_seed : optional dict {coupling_seed: (N,N) ndarray} of real
        per-subject connectomes. If None, uses random-uniform placeholder
        coupling for every seed (current default, toy/benchmark mode).
    """
    if n_workers is None:
        n_workers = cpu_count()
    print(f"Running with {n_workers} parallel workers "
          f"({cpu_count()} cores detected)")

    tasks = []
    for coupling_seed in range(N_COUPLING_SEEDS):
        real_C = real_C_per_seed[coupling_seed] if real_C_per_seed else None
        for w_EE in W_EE_VALUES:
            tasks.append((N, w_EE, coupling_seed, real_C))

    t_start = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(process_one_combo, tasks)
    elapsed = time.time() - t_start

    rows = [row for sublist in results for row in sublist]
    n_skipped = sum(1 for r in results if len(r) == 0)
    print(f"Sweep complete: {len(rows)} runs in {elapsed:.1f}s "
          f"({n_skipped} (seed, w_EE) combos skipped as unstable/non-convergent)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-regions", type=int, default=10,
                         help="Number of brain regions (network size N)")
    parser.add_argument("--n-workers", type=int, default=None,
                         help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    df = run_sweep_parallel(args.n_regions, n_workers=args.n_workers)
    df.to_csv("wc_linear_validity_sweep_results.csv", index=False)
    print(f"Saved {len(df)} rows to wc_linear_validity_sweep_results.csv")
