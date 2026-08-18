#!/usr/bin/env python3
"""
Phase 6 robustness check: does the Path A1 validity-regime characterization
(target-distance, horizon, bifurcation-proximity) replicate in FitzHugh-
Nagumo, a structurally different nonlinear model (relaxation oscillator,
cubic nonlinearity) from Wilson-Cowan (sigmoidal saturating population)?

Toy scale first (N=10, random coupling), mirroring exactly how the WC
sweep was staged originally -- prove the machinery and get an initial
read before any real-N/real-coupling extension.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fhn_simulation as fhn
import physics

N = 10
N_SEEDS = 8
DT = 0.5  # roughly matches WC's dt/fast-timescale ratio

# Confirmed numerically: Hopf boundary at I_ext ~ 0.32-0.325 (weak coupling,
# single-node reference). Grid spans safely below it, mirroring how the WC
# grid stayed safely below its own boundary.
I_EXT_VALUES = [0.0, 0.08, 0.16, 0.24, 0.30]

T_VALUES = [5, 20, 50, 100, 200]
TARGET_SCALES = [0.005, 0.01, 0.02, 0.05, 0.1]
N_TARGET_DIRECTIONS_PER_SEED = 3


def run_one_combo(seed, I_ext):
    rng = np.random.default_rng(seed)
    C = rng.uniform(0, 0.15, (N, N)) / N * 20
    np.fill_diagonal(C, 0)

    net = fhn.FitzHughNagumoNetwork(n_regions=N, C=C, I_ext=I_ext)
    try:
        v_star, w_star = net.find_fixed_point()
    except RuntimeError:
        return []
    J = net.jacobian_at(v_star, w_star)
    max_real_eig = np.linalg.eigvals(J).real.max()
    if max_real_eig >= 0:
        return []

    # G = [[I],[0]] -- u enters additively into dv/dt only, no sigmoid
    # nonlinearity to differentiate through (structurally simpler than WC).
    G = np.vstack([np.eye(N), np.zeros((N, N))])
    from simulation import discretize_system
    A, B = discretize_system(J, G, DT)
    rho_A = np.max(np.abs(np.linalg.eigvals(A)))
    x_star = np.concatenate([v_star, w_star])

    rows = []
    for T in T_VALUES:
        for target_scale in TARGET_SCALES:
            for dir_seed in range(N_TARGET_DIRECTIONS_PER_SEED):
                drng = np.random.default_rng(
                    hash((seed, int(I_ext * 100), T, int(target_scale * 1000), dir_seed)) % (2**32)
                )
                direction = drng.standard_normal(N)
                direction /= np.linalg.norm(direction)
                target_perturb = target_scale * direction

                delta_x0 = np.zeros(2 * N)
                delta_xT = np.concatenate([target_perturb, np.zeros(N)])

                energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT, T)
                u_func = physics.zero_order_hold(U, DT)

                result = net.simulate_controlled(
                    u_func=u_func, t_span=(0.0, T * DT),
                    n_points=max(100, T), E0=v_star, I0=w_star,
                )
                realized_final = np.concatenate([result["E"][:, -1], result["I"][:, -1]])
                xT_absolute = x_star + delta_xT
                target_movement = np.linalg.norm(delta_xT)
                err = np.linalg.norm(realized_final - xT_absolute)
                rel_err = err / target_movement

                rows.append({
                    "seed": seed, "I_ext": I_ext,
                    "stability_margin": -max_real_eig, "rho_A": rho_A,
                    "T": T, "target_scale": target_scale, "dir_seed": dir_seed,
                    "energy": energy, "rel_reachability_error": rel_err,
                })
    return rows


if __name__ == "__main__":
    t_start = time.time()
    all_rows = []
    n_skipped = 0
    for seed in range(N_SEEDS):
        for I_ext in I_EXT_VALUES:
            rows = run_one_combo(seed, I_ext)
            if not rows:
                n_skipped += 1
            all_rows.extend(rows)
    elapsed = time.time() - t_start

    df = pd.DataFrame(all_rows)
    print(f"Sweep complete: {len(df)} rows in {elapsed:.1f}s ({n_skipped} combos skipped)")
    df.to_csv("fhn_validity_sweep_toy_results.csv", index=False)
    print("Saved to fhn_validity_sweep_toy_results.csv")

    print("\n=== 1. Target-distance scaling ===")
    print(df.groupby('target_scale')['rel_reachability_error'].median())

    print("\n=== 2. Horizon plateau ===")
    print(df.groupby('T')['rel_reachability_error'].median())

    print("\n=== 3. Bifurcation-proximity pattern (monotonic decrease, or sweet spot?) ===")
    print(df.groupby('I_ext')['rel_reachability_error'].median())
    rho, p = spearmanr(df['I_ext'], df['rel_reachability_error'])
    print(f"\nSpearman (I_ext vs error, pooled): rho={rho:.4f}, p={p:.2e}")

    print("\n=== Per-seed pattern (checking individual consistency, like the real-coupling HCP check) ===")
    pivot = df.groupby(['seed', 'I_ext'])['rel_reachability_error'].median().unstack()
    print(pivot.to_string())
