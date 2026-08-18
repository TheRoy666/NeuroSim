#!/usr/bin/env python3
"""
Path A1 -- the real-coupling sweep. Loads real per-subject SC matrices
(using the confirmed log-transform row-sum scaling from
real_coupling_boundary_recheck.py) as the WC network's coupling matrix,
uses the newly-calibrated real-coupling W_EE_VALUES grid (confirmed via
4/4 exact-agreement fine scan, boundary ~4.45-4.50), and runs the full
validity-regime sweep (target-distance x horizon x stability-margin) with
real brain connectivity instead of the synthetic random-uniform
placeholder used everywhere up to this point.

This is the "real coupling" half of the Path A1 upgrade that's been
scoped since early in the project -- the placeholder-coupling sweep
answers "does the finding hold at real cohort scale"; this answers "does
it hold on real brains."

Run directly:
    python3 wc_linear_validity_sweep_real_coupling.py \\
        --sc-dir /path/to/SC_matrices --sc-suffix _SC_SIFT2_410.csv \\
        --n-workers 55
"""
import os
# Must happen before numpy/scipy are imported anywhere -- caps each worker
# process to a single BLAS thread, so parallelism comes from the process
# pool (n_workers processes), not from each process also fighting for all
# cores internally. This exact bug (each worker process ALSO internally
# multithreading via BLAS, causing massive core oversubscription) was
# found and fixed in Path B's bootstrap scripts, where it caused real
# per-subject runs to take ~70-140x longer than expected. This script
# imported os but never actually set these variables -- found while
# investigating why the N=8 real-coupling sweeps (40 combos, well within
# the 55-75 available workers, should parallelize in minutes) instead
# took 11-25 hours each. Applying the same fix here.
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

DT = 2.0
BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                    tau_E=10.0, tau_I=20.0)

# Default: confirmed via real_coupling_boundary_recheck.py on 4 real HCP
# subjects, exact 4/4 agreement at 0.05 resolution, boundary at
# w_EE=4.45-4.50. NOT necessarily correct for other cohorts -- real
# connectome structure (atlas, population, edge statistics) determines
# the boundary, confirmed to differ from the synthetic-coupling case
# already. Override with --w-ee-values once a cohort's own boundary is
# confirmed via real_coupling_boundary_recheck.py.
W_EE_VALUES_HCP_DEFAULT = [3.0, 3.3402, 3.6805, 4.0207, 4.361]

T_VALUES = [5, 20, 50, 100, 200]
TARGET_SCALES = [0.005, 0.01, 0.02, 0.05, 0.1]
N_TARGET_DIRECTIONS_PER_SEED = 3


def discover_sc_files(sc_dir, sc_suffix):
    return {f[:-len(sc_suffix)]: os.path.join(sc_dir, f)
            for f in os.listdir(sc_dir) if f.endswith(sc_suffix)}


def load_and_scale_real_coupling(sc_path):
    """Identical logic to real_coupling_boundary_recheck.py -- log1p
    transform before row-sum scaling, confirmed to resolve the hub-driven
    sigmoid-overflow mechanism on real HCP data."""
    SC = np.loadtxt(sc_path, delimiter=",")
    N = SC.shape[0]
    np.fill_diagonal(SC, 0)
    SC_log = np.where(SC > 0, np.log1p(SC), 0)
    target_row_sum_mean = (0.075 / N) * 20 * (N - 1)
    scale_factor = target_row_sum_mean / SC_log.sum(axis=1).mean()
    return SC_log * scale_factor, N


def build_network(N, w_EE, C):
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
    """Same per-combo logic as wc_linear_validity_sweep_parallel.py, now
    fed a real per-subject C instead of random-uniform placeholder."""
    subject_id, N, w_EE, C = args
    net = build_network(N, w_EE, C)
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
                    hash((subject_id, int(w_EE * 10), T,
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
                    "subject_id": subject_id,
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", default="_SC_SIFT2_410.csv")
    ap.add_argument("--n-subjects", type=int, default=8,
                     help="Number of real subjects to use as coupling "
                          "seeds (default 8, matching the synthetic-"
                          "coupling sweep's N_COUPLING_SEEDS)")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--w-ee-values", default=None,
                     help="Comma-separated w_EE grid, calibrated for THIS "
                          "cohort's real coupling via "
                          "real_coupling_boundary_recheck.py first. "
                          "Defaults to the HCP-confirmed grid if omitted -- "
                          "do NOT rely on that default for a different "
                          "cohort without checking its own boundary first.")
    ap.add_argument("--out", default="wc_linear_validity_sweep_real_coupling_results.csv")
    args = ap.parse_args()

    n_workers = args.n_workers or cpu_count()
    print(f"Running with {n_workers} parallel workers ({cpu_count()} cores detected)")

    files = discover_sc_files(args.sc_dir, args.sc_suffix)
    subjects = sorted(files.keys())[:args.n_subjects]
    print(f"Found {len(files)} SC files; using {len(subjects)} as coupling seeds")
    if args.w_ee_values is not None:
        w_ee_grid = [float(x) for x in args.w_ee_values.split(",")]
    else:
        w_ee_grid = W_EE_VALUES_HCP_DEFAULT
        print("*** WARNING: no --w-ee-values given, using the HCP-confirmed "
              "default. If this is a different cohort, confirm its own "
              "boundary via real_coupling_boundary_recheck.py first -- "
              "reusing HCP's grid blind is exactly the mistake already "
              "caught once with the synthetic-vs-real-coupling grids. ***\n")
    print(f"Real-coupling grid: {w_ee_grid}\n")

    tasks = []
    for sub in subjects:
        C, N = load_and_scale_real_coupling(files[sub])
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
