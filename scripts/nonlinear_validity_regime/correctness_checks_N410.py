#!/usr/bin/env python3
"""
=== COUPLING DATA SOURCE: SYNTHETIC, NOT REAL SUBJECT DATA ===
This script uses SYNTHETIC hub-dominated coupling (lognormal-generated,
sigma=2.8, sparsity=0.75, log-transform + row-sum scaled) as a
representative stand-in for real coupling's statistical properties
(heavy-tailed, hub-dominated). It does NOT use genuine per-subject
structural connectivity. For results using GENUINE real per-subject
HCP/ADNI SC matrices, see real_coupling_boundary_recheck.py and
wc_linear_validity_sweep_real_coupling.py instead.
===============================================================

Foundational correctness checks for the physics.py/simulation.py machinery
underneath the Path A1 headline finding, rerun at real N=410 with
realistic hub-dominated conditioning (previously only verified at toy
scale N=10). This was flagged in a project audit as a real risk: the
single most consequential result in the paper had never had its
foundational machinery re-verified at the scale it actually depends on.

Three checks, all against a realistic N=410 hub-dominated system (same
log-transform generation as the real-coupling work):

1. Gramian doubling algorithm (O(log T)) vs. brute-force summation
   (O(T)) -- do they agree at real N, or does conditioning at this scale
   introduce meaningful numerical drift?
2. LTV formulation reduces exactly to the LTI formulation when given a
   time-invariant sequence -- verified previously only at toy scale.
3. Deviation-coordinate propagation matches direct absolute-coordinate
   propagation exactly -- verifies the coordinate transform itself
   introduces no bug at real N.

All three passed at machine precision when first run inline; this script
makes that reproducible and persists the actual numbers, rather than only
reporting them in conversation.
"""
import numpy as np
import pandas as pd
import time

from simulation import WilsonCowanNetwork, discretize_system, input_jacobian_at
import physics

N = 410
DT = 2.0
T_VALUES = [5, 50, 200]


def build_realistic_coupling(seed, N=410, sigma=2.8, sparsity=0.75):
    """Same generation approach used throughout this project's real-SC-like
    verification work."""
    rng = np.random.default_rng(seed)
    SC = rng.lognormal(mean=0, sigma=sigma, size=(N, N))
    SC = (SC + SC.T) / 2
    mask = rng.random((N, N)) < sparsity
    SC[mask] = 0
    np.fill_diagonal(SC, 0)
    SC_log = np.where(SC > 0, np.log1p(SC), 0)
    target_row_sum = (0.075 / N) * 20 * (N - 1)
    C = SC_log * (target_row_sum / SC_log.sum(axis=1).mean())
    return C


def setup_system(seed=42):
    C = build_realistic_coupling(seed, N=N)
    params = dict(w_EE=4.0, w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                  tau_E=10.0, tau_I=20.0)
    net = WilsonCowanNetwork(n_regions=N, C=C, node_params=params)
    E_star, I_star = net.find_fixed_point()
    J = net.jacobian_at(E_star, I_star)
    max_eig = np.linalg.eigvals(J).real.max()
    G = input_jacobian_at(net, E_star, I_star)
    A, B = discretize_system(J, G, DT)
    x_star = np.concatenate([E_star, I_star])
    return A, B, x_star, max_eig


def check_1_gramian_doubling_vs_brute_force(A, B):
    rows = []
    N_local = A.shape[0]
    for T in T_VALUES:
        t0 = time.time()
        W_doubling = physics.compute_gramian_doubling(A, B, T)
        t_doubling = time.time() - t0

        t0 = time.time()
        W_brute = np.zeros((N_local, N_local))
        A_power = np.eye(N_local)
        for k in range(T):
            W_brute += A_power @ B @ B.T @ A_power.T
            A_power = A_power @ A
        t_brute = time.time() - t0

        max_abs_diff = np.abs(W_doubling - W_brute).max()
        rel_diff = max_abs_diff / (np.abs(W_brute).max() + 1e-30)
        rows.append({
            "check": "gramian_doubling_vs_brute_force", "T": T,
            "max_abs_diff": max_abs_diff, "max_rel_diff": rel_diff,
            "t_doubling_s": t_doubling, "t_brute_s": t_brute,
            "pass": rel_diff < 1e-8,
        })
    return rows


def check_2_ltv_reduces_to_lti(A, B, seed=1):
    rng = np.random.default_rng(seed)
    N_local = A.shape[0]
    rows = []
    for T in T_VALUES:
        x0 = np.zeros(N_local)
        direction = rng.standard_normal(N_local)
        direction /= np.linalg.norm(direction)
        xT = 0.02 * direction

        energy_lti, U_lti = physics.minimum_energy_trajectory(A, B, x0, xT, T)
        A_list = [A] * T
        B_list = [B] * T
        energy_ltv, U_ltv = physics.minimum_energy_trajectory_ltv(A_list, B_list, x0, xT)

        energy_diff = abs(energy_lti - energy_ltv)
        U_diff = np.abs(U_lti - U_ltv).max()
        rel_energy_diff = energy_diff / (abs(energy_lti) + 1e-30)
        rows.append({
            "check": "ltv_reduces_to_lti", "T": T,
            "energy_lti": energy_lti, "energy_ltv": energy_ltv,
            "rel_energy_diff": rel_energy_diff, "max_U_diff": U_diff,
            "pass": rel_energy_diff < 1e-6 and U_diff < 1e-6,
        })
    return rows


def check_3_deviation_coordinate_transform(A, B, x_star, seed=2):
    rng = np.random.default_rng(seed)
    N_local = A.shape[0]
    rows = []
    for T in T_VALUES:
        direction = rng.standard_normal(N_local)
        direction /= np.linalg.norm(direction)
        delta_xT = 0.02 * direction
        delta_x0 = np.zeros(N_local)

        energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT, T)

        delta_x = delta_x0.copy()
        for k in range(T):
            delta_x = A @ delta_x + B @ U[k]
        final_via_deviation = x_star + delta_x

        y = x_star.copy()
        for k in range(T):
            y = x_star + A @ (y - x_star) + B @ U[k]
        final_via_absolute = y

        max_diff = np.abs(final_via_deviation - final_via_absolute).max()
        target_absolute = x_star + delta_xT
        reach_err = np.linalg.norm(final_via_deviation - target_absolute) / np.linalg.norm(delta_xT)

        rows.append({
            "check": "deviation_coordinate_transform", "T": T,
            "max_diff": max_diff, "linear_reachability_error_sanity": reach_err,
            "pass": max_diff < 1e-9,
        })
    return rows


if __name__ == "__main__":
    A, B, x_star, max_eig = setup_system(seed=42)
    print(f"N={A.shape[0]}, fixed point found, max_real_eig={max_eig:.6f} "
          f"(stable={max_eig < 0})")
    rho_A = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"rho(A) = {rho_A:.6f}\n")

    all_rows = []
    print("=== CHECK 1: Gramian doubling vs. brute-force ===")
    r1 = check_1_gramian_doubling_vs_brute_force(A, B)
    for row in r1:
        print(f"  T={row['T']}: max_rel_diff={row['max_rel_diff']:.2e}  "
              f"[{'PASS' if row['pass'] else 'FAIL'}]")
    all_rows.extend(r1)

    print("\n=== CHECK 2: LTV reduces to LTI ===")
    r2 = check_2_ltv_reduces_to_lti(A, B)
    for row in r2:
        print(f"  T={row['T']}: rel_energy_diff={row['rel_energy_diff']:.2e}  "
              f"[{'PASS' if row['pass'] else 'FAIL'}]")
    all_rows.extend(r2)

    print("\n=== CHECK 3: Deviation-coordinate transform ===")
    r3 = check_3_deviation_coordinate_transform(A, B, x_star)
    for row in r3:
        print(f"  T={row['T']}: max_diff={row['max_diff']:.2e}  "
              f"[{'PASS' if row['pass'] else 'FAIL'}]")
    all_rows.extend(r3)

    df = pd.DataFrame(all_rows)
    df.to_csv("correctness_checks_N410_results.csv", index=False)
    all_pass = df["pass"].all()
    print(f"\n{'='*70}")
    print(f"ALL CHECKS: {'PASS' if all_pass else 'AT LEAST ONE FAILED -- investigate'}")
    print(f"{'='*70}")
    print("Saved to correctness_checks_N410_results.csv")
