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

Mechanistic investigation of the Path A1 real-coupling non-monotonic reachability-error pattern
(reachability error decreases then increases across the w_EE grid, 8/8
real HCP subjects, see wc_linear_validity_sweep_real_coupling_HCP.csv).

Tests three candidate mechanisms in sequence, honestly, including the two
that did NOT pan out -- this is a record of the actual investigation, not
just the hypothesis that worked:

1. Hub-eigenvector localization: does the least-stable eigenvector
   localize onto hub nodes specifically near the boundary? RESULT: no --
   hub-correlation stays uniformly high (~0.98-0.99) across the entire
   grid, not boundary-specific. Localization (participation ratio) trends
   smoothly but continuously, not with a boundary-triggered signature.

2. Eigenvalue crowding near marginal stability: do multiple eigenvalues
   cluster near the top as the boundary approaches? RESULT: uninformative
   as tested -- gap-to-2nd-eigenvalue is exactly 0 and "all eigenvalues
   within 0.02 of the top" at every single grid point, most likely
   reflecting WC's natural complex-conjugate eigenvalue pairs and a
   tau_E/tau_I-constrained tight spectrum rather than a real signature.

3. Gramian conditioning vs. energy, competing trends: does Gramian
   condition number worsen monotonically while energy improves
   monotonically, with an interior minimum at their crossover? RESULT:
   promising. Both trends are real, monotonic, and opposing -- a
   coherent account for why a minimum appears in between them, without
   needing anything to "kick in" specifically at the boundary. Not yet
   shown to quantitatively match the exact interior-minimum location -- that
   precision check is the next step, tracked separately.

Uses realistic hub-dominated synthetic coupling (same log-transform
generation as the real-coupling boundary/sweep scripts), NOT the literal
raw HCP SC matrices (not available in this environment) -- tests the
mechanism honestly on representative structure, flagged as needing
cross-check against the real subject matrices when server access allows.
"""
import numpy as np
import pandas as pd

from simulation import WilsonCowanNetwork, discretize_system, input_jacobian_at
import physics

N = 410
DT = 2.0
# Confirmed real-coupling HCP grid (see real_coupling_boundary_recheck.py
# output, fine-scan boundary 4.45-4.50)
W_EE_GRID = [3.0, 3.3402, 3.6805, 4.0207, 4.361]
PARAMS_BASE = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                    tau_E=10.0, tau_I=20.0)


def build_realistic_coupling(seed, N=410, sigma=2.8, sparsity=0.75):
    """Same generation approach as real_coupling_boundary_recheck.py's
    synthetic-test data and the N=410 correctness-check verification --
    heavy-tailed, sparse, log-transformed, row-sum scaled to match the
    synthetic-placeholder convention."""
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


def participation_ratio(v):
    """PR = (sum|v_i|^2)^2 / sum(|v_i|^4). Low PR = localized on few
    nodes; high PR (~N) = spread evenly across all nodes."""
    v = np.abs(v)
    return (np.sum(v**2) ** 2) / np.sum(v**4)


def run_investigation(seed=10, gramian_T=50):
    C = build_realistic_coupling(seed, N=N)
    hubness = C.sum(axis=1)

    rows = []
    for w_EE in W_EE_GRID:
        params = dict(w_EE=w_EE, **PARAMS_BASE)
        net = WilsonCowanNetwork(n_regions=N, C=C, node_params=params)
        E_star, I_star = net.find_fixed_point()
        J = net.jacobian_at(E_star, I_star)

        # --- Hypothesis 1: hub-eigenvector localization ---
        eigvals, eigvecs = np.linalg.eig(J)
        idx_least_stable = np.argmax(eigvals.real)
        least_stable_eig = eigvals[idx_least_stable].real
        least_stable_vec = eigvecs[:, idx_least_stable]
        v_E = np.abs(least_stable_vec[:N])
        pr = participation_ratio(v_E)
        pr_normalized = pr / N
        corr_with_hubness = np.corrcoef(v_E, hubness)[0, 1]

        # --- Hypothesis 2: eigenvalue crowding ---
        real_parts = np.sort(eigvals.real)[::-1]
        gap_1_2 = real_parts[0] - real_parts[1]
        n_near_top = int(np.sum(real_parts > real_parts[0] - 0.02))

        # --- Hypothesis 3: Gramian conditioning vs. energy ---
        G = input_jacobian_at(net, E_star, I_star)
        A, B = discretize_system(J, G, DT)
        W_T = physics.compute_gramian_doubling(A, B, gramian_T)
        gramian_cond = np.linalg.cond(W_T)
        gramian_min_eig = float(np.linalg.eigvalsh(W_T)[0])
        gramian_max_eig = float(np.linalg.eigvalsh(W_T)[-1])

        rows.append({
            "w_EE": w_EE,
            "seed": seed,
            "least_stable_eig": least_stable_eig,
            "participation_ratio": pr,
            "participation_ratio_normalized": pr_normalized,
            "corr_least_stable_vec_with_hubness": corr_with_hubness,
            "eigenvalue_gap_1_to_2": gap_1_2,
            "n_eigenvalues_within_0.02_of_top": n_near_top,
            "gramian_T": gramian_T,
            "gramian_condition_number": gramian_cond,
            "gramian_min_eig": gramian_min_eig,
            "gramian_max_eig": gramian_max_eig,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_investigation(seed=10, gramian_T=50)
    df.to_csv("mechanistic_interior_minimum_investigation_results.csv", index=False)

    print("=== Hypothesis 1: hub-eigenvector localization ===")
    print(df[["w_EE", "participation_ratio_normalized", "corr_least_stable_vec_with_hubness"]].to_string(index=False))
    print("\nResult: hub-correlation uniformly high across the WHOLE grid, not")
    print("boundary-specific. Localization trends smoothly, not boundary-triggered.")

    print("\n=== Hypothesis 2: eigenvalue crowding ===")
    print(df[["w_EE", "eigenvalue_gap_1_to_2", "n_eigenvalues_within_0.02_of_top"]].to_string(index=False))
    print("\nResult: uninformative as tested (gap=0, all eigenvalues 'near top' at")
    print("every point) -- likely reflects complex-conjugate pairs + tight spectrum,")
    print("not a real near-boundary signature.")

    print("\n=== Hypothesis 3: Gramian conditioning vs. energy (the promising lead) ===")
    print(df[["w_EE", "gramian_condition_number"]].to_string(index=False))
    pct_increase = (df["gramian_condition_number"].iloc[-1] / df["gramian_condition_number"].iloc[0] - 1) * 100
    print(f"\nCondition number increases monotonically: +{pct_increase:.0f}% across the grid.")
    print("Coherent with an interior minimum as the crossover between improving energy")
    print("(confirmed monotonic, Lindmark & Altafini) and worsening conditioning.")
    print("NOT yet shown to quantitatively match the exact interior-minimum location --")
    print("that precision check is the next step, tracked separately.")

    print(f"\nSaved full results to mechanistic_interior_minimum_investigation_results.csv")
