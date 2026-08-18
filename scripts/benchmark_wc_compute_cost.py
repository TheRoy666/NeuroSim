#!/usr/bin/env python3
"""
Benchmark WC network compute cost at real cohort N values, on real
server hardware, before committing to any A1/A2 grid size.

Uses random-uniform coupling matrices, not real connectome data -- timing
of the linear-algebra operations (expm, eig, matrix_power) depends on
matrix dimension, not on the specific values inside the matrix, so this
gives an honest cost estimate without needing any real data files present.

Tests the four real cohort N values:
  N=68   UNAM (DK, shared template)
  N=400  ADNI Schaefer400 (cortex-only)
  N=410  HCP-AUD (likely HCP-MMP + Tian-S3)
  N=450  ADNI Schaefer400+TianS3

State dimension for the WC network is 2N (E and I per region), so this
benchmarks the 2N x 2N Jacobian/discretization operations, plus the
Gramian-doubling and full A1 per-run cost, and separately the A2
nonlinear-ceiling cost (the known bottleneck from the toy-scale session).

Run directly on the server: `python3 benchmark_wc_compute_cost.py`
Prints per-operation timing and extrapolates full-grid cost at each N so
the actual sweep grid can be sized to the available wall-clock budget.
"""
import time
import numpy as np
import simulation
import physics

REAL_COHORT_N = {
    "UNAM (N=68)": 68,
    "ADNI-Schaefer400 (N=400)": 400,
    "HCP-AUD (N=410)": 410,
    "ADNI-Schaefer400+TianS3 (N=450)": 450,
}

DAMPED_PARAMS = dict(w_EE=3.0, w_IE=4.0, w_EI=3.0, w_II=2.0,
                      c_E=-2.0, c_I=-2.0, tau_E=10.0, tau_I=20.0)
DT = 2.0  # ms, matches the A1/A2 dt-calibration finding


def time_one(func, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - t0


def benchmark_at_N(N, seed=0, T_for_gramian=(5, 20, 50, 100, 200)):
    print(f"\n{'='*70}\nN = {N}  (state dimension 2N = {2*N})\n{'='*70}")
    rng = np.random.default_rng(seed)
    # Coupling scaled by 1/N (mean-field convention): without this, total
    # input per node grows with N (more neighbors, same per-edge strength),
    # which can push the system out of a stable fixed point entirely at
    # larger N even though the same raw magnitude was fine at small N --
    # caught during benchmarking at N=100 (solver failed to converge with
    # unscaled coupling that worked at N=20).
    C = rng.uniform(0, 0.15, (N, N)) / N * 20  # *20 keeps ~N=20 behavior at that scale
    np.fill_diagonal(C, 0)
    net = simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=DAMPED_PARAMS)

    timings = {}

    (E_star, I_star), t = time_one(net.find_fixed_point)
    timings["find_fixed_point"] = t
    print(f"  find_fixed_point:              {t:.4f}s")

    J, t = time_one(net.jacobian_at, E_star, I_star)
    timings["jacobian_at"] = t
    print(f"  jacobian_at:                    {t:.4f}s")

    G, t = time_one(simulation.input_jacobian_at, net, E_star, I_star)
    timings["input_jacobian_at"] = t
    print(f"  input_jacobian_at:              {t:.4f}s")

    (A, B), t = time_one(simulation.discretize_system, J, G, DT)
    timings["discretize_system"] = t
    print(f"  discretize_system (expm):      {t:.4f}s")

    rho_A = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"  (rho_A = {rho_A:.4f}, for reference)")

    for T in T_for_gramian:
        W, t = time_one(physics.compute_gramian_doubling, A, B, T)
        timings[f"gramian_T{T}"] = t
        print(f"  compute_gramian_doubling(T={T:>3}):  {t:.4f}s")

    # Full single-run cost (fixed point + jacobian + discretize + one
    # minimum_energy_trajectory + one simulate_controlled call) -- this
    # is the per-grid-point cost that determines total sweep wall-clock
    T_test = 50
    target_perturb = 0.02 * rng.standard_normal(N)
    delta_x0 = np.zeros(2 * N)
    delta_xT = np.concatenate([target_perturb, np.zeros(N)])

    t0 = time.time()
    energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT, T_test)
    t_energy = time.time() - t0

    u_func = physics.zero_order_hold(U, DT)
    t0 = time.time()
    result = net.simulate_controlled(u_func=u_func, t_span=(0.0, T_test * DT),
                                      n_points=200, E0=E_star, I0=I_star)
    t_sim = time.time() - t0

    timings["minimum_energy_trajectory"] = t_energy
    timings["simulate_controlled"] = t_sim
    print(f"  minimum_energy_trajectory (T=50): {t_energy:.4f}s")
    print(f"  simulate_controlled (T=50):       {t_sim:.4f}s")

    full_run_cost = (timings["find_fixed_point"] + timings["jacobian_at"] +
                      timings["input_jacobian_at"] + timings["discretize_system"] +
                      t_energy + t_sim)
    print(f"\n  FULL SINGLE-RUN COST (fixed-pt through simulate): {full_run_cost:.4f}s")

    # Extrapolate to the A1 grid used at toy scale: 5 stability-margin x
    # 5 T x 5 target-scale x N_seeds x 3 directions. Note: fixed-point/
    # jacobian/discretize only need recomputing per (seed, stability-margin)
    # pair, not per full grid point -- but report both bounds.
    n_seeds = 8
    n_stability = 5
    n_T = 5
    n_target = 5
    n_dirs = 3
    fixed_pt_cost_total = timings["find_fixed_point"] + timings["jacobian_at"] + \
                          timings["input_jacobian_at"] + timings["discretize_system"]
    per_point_cost = t_energy + t_sim
    total_fixed_pt_evals = n_seeds * n_stability
    total_control_evals = n_seeds * n_stability * n_T * n_target * n_dirs
    est_total = (fixed_pt_cost_total * total_fixed_pt_evals +
                 per_point_cost * total_control_evals)
    print(f"\n  Estimated FULL A1 GRID cost at this N (same grid density as")
    print(f"  toy-scale run: {n_seeds} seeds x {n_stability} stability x {n_T} T x "
          f"{n_target} target x {n_dirs} dirs):")
    print(f"    {est_total:.1f}s  ({est_total/60:.1f} min)")

    return timings, est_total


if __name__ == "__main__":
    print("WC compute-cost benchmark -- real cohort N values, random coupling")
    print("(timing depends on matrix dimension, not connectome identity, so")
    print("real connectome files are not needed for this step)")

    all_estimates = {}
    for label, N in REAL_COHORT_N.items():
        _, est = benchmark_at_N(N)
        all_estimates[label] = est

    print(f"\n{'='*70}\nSUMMARY -- estimated full A1 grid cost per cohort N")
    print(f"{'='*70}")
    for label, est in all_estimates.items():
        print(f"  {label:>35}: {est:>8.1f}s  ({est/60:>6.1f} min)")
    print("\nUse these numbers to size the actual grid (fewer seeds/T/target")
    print("values, or accept the wall-clock cost) before committing to a run.")
    print("\nNOTE: this does NOT benchmark the A2 nonlinear-optimal-control")
    print("ceiling (Powell direct shooting), which was the known bottleneck")
    print("at toy scale (~2.75 min/case at N=10). That should be benchmarked")
    print("separately once this script's results are in, since it will be")
    print("substantially more expensive at real N and may need the adjoint-")
    print("gradient method (Phase 5, Section 2b) before it's affordable at all.")
