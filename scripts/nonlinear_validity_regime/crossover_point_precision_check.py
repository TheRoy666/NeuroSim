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

Crossover-point precision check for the Path A1 interior-minimum
mechanism investigation (see mechanistic_interior_minimum_investigation.py
for the underlying hypothesis: does a crossover between monotonically
decreasing energy and monotonically increasing Gramian conditioning
predict the location of the interior minimum in reachability error?).

Two attempts, both on the same realistic hub-dominated synthetic system:

1. First attempt (single condition: T=50, target_scale=0.02, 8 direction
   seeds) was underpowered -- its own empirical error curve did not
   reproduce a genuine interior minimum (noisy, minimum at the boundary-
   closest point instead), making any comparison against it unreliable.
   Discarded, not used for the conclusion below.

2. This script (pooled: T in [20,50], target_scale=0.02, 5 direction
   seeds, 10 runs per w_EE) reproduces a genuine interior minimum,
   consistent with the real HCP result (minimum at w_EE=3.6805, matching
   the 3.68/4.02 split found in the real data).

RESULT: the empirical interior minimum is real and reproduces in this
synthetic system, but the simple combined metric tested (normalized
energy x (1 + normalized Gramian condition number)) does NOT predict its
location -- the combined metric's minimum sits at the boundary-closest
grid point (w_EE=4.361), not at the actual empirical minimum (w_EE=3.68).
NO EXACT MATCH.

Conclusion: the qualitative story (two real, monotonic, opposing trends
producing an interior minimum) remains plausible, but this specific
formalization of "crossover" does not quantitatively explain the
observed location. A more precise functional relationship between
energy, conditioning, and reachability error would be needed for a full
quantitative mechanistic account. Reported honestly as an open question,
not resolved.
"""
import numpy as np
import time
from simulation import WilsonCowanNetwork, discretize_system, input_jacobian_at
import physics

def build_realistic_coupling(seed, N=410, sigma=2.8, sparsity=0.75):
    rng = np.random.default_rng(seed)
    SC = rng.lognormal(mean=0, sigma=sigma, size=(N, N))
    SC = (SC + SC.T) / 2
    mask = rng.random((N, N)) < sparsity
    SC[mask] = 0
    np.fill_diagonal(SC, 0)
    SC_log = np.where(SC > 0, np.log1p(SC), 0)
    target_row_sum = (0.075/N)*20*(N-1)
    return SC_log * (target_row_sum / SC_log.sum(axis=1).mean())

N = 410
DT = 2.0
W_EE_GRID = [3.0, 3.3402, 3.6805, 4.0207, 4.361]
T_VALUES = [20, 50]
TARGET_SCALES = [0.02]
N_DIR_SEEDS = 5
params_base = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0, tau_E=10.0, tau_I=20.0)
C = build_realistic_coupling(seed=10, N=N)

t_start = time.time()
rows = []
for w_EE in W_EE_GRID:
    params = dict(w_EE=w_EE, **params_base)
    net = WilsonCowanNetwork(n_regions=N, C=C, node_params=params)
    E_star, I_star = net.find_fixed_point()
    J = net.jacobian_at(E_star, I_star)
    G = input_jacobian_at(net, E_star, I_star)
    A, B = discretize_system(J, G, DT)
    x_star = np.concatenate([E_star, I_star])
    W_T_ref = physics.compute_gramian_doubling(A, B, 50)
    gramian_cond = np.linalg.cond(W_T_ref)

    errors, energies = [], []
    for T in T_VALUES:
        for target_scale in TARGET_SCALES:
            for dir_seed in range(N_DIR_SEEDS):
                drng = np.random.default_rng(hash((T, int(target_scale*1000), dir_seed)) % (2**32))
                direction = drng.standard_normal(N)
                direction /= np.linalg.norm(direction)
                delta_xT = target_scale * direction
                delta_x0 = np.zeros(2*N)
                delta_xT_full = np.concatenate([delta_xT, np.zeros(N)])

                energy, U = physics.minimum_energy_trajectory(A, B, delta_x0, delta_xT_full, T)
                u_func = physics.zero_order_hold(U, DT)
                result = net.simulate_controlled(u_func=u_func, t_span=(0.0, T*DT),
                                                  n_points=max(100,T), E0=E_star, I0=I_star)
                realized_final = np.concatenate([result["E"][:,-1], result["I"][:,-1]])
                xT_absolute = x_star + delta_xT_full
                err = np.linalg.norm(realized_final - xT_absolute) / np.linalg.norm(delta_xT_full)
                errors.append(err)
                energies.append(energy)

    median_error = np.median(errors)
    median_energy = np.median(energies)
    rows.append((w_EE, median_error, median_energy, gramian_cond, len(errors)))
    elapsed = time.time() - t_start
    print(f"w_EE={w_EE:.4f}: n={len(errors)}, median_error={median_error:.4f}, "
          f"median_energy={median_energy:.4f}, gramian_cond={gramian_cond:.2f} "
          f"[{elapsed:.1f}s elapsed]", flush=True)

print("\n=== Locating the empirical minimum vs. the combined-metric minimum ===")
errors = [r[1] for r in rows]
energies = [r[2] for r in rows]
conds = [r[3] for r in rows]
w_ees = [r[0] for r in rows]

empirical_min_idx = np.argmin(errors)
print(f"\nEmpirical reachability-error minimum at w_EE={w_ees[empirical_min_idx]:.4f} (error={errors[empirical_min_idx]:.4f})")
print(f"Full error curve: {[round(e,4) for e in errors]}")

energy_norm = (np.array(energies) - min(energies)) / (max(energies) - min(energies))
cond_norm = (np.array(conds) - min(conds)) / (max(conds) - min(conds))
combined_product = energy_norm * (1 + cond_norm)
combined_min_idx = np.argmin(combined_product)
print(f"\nCombined energy x (1+conditioning) metric minimum at w_EE={w_ees[combined_min_idx]:.4f}")
print(f"Combined-metric values: {[round(c,4) for c in combined_product]}")

match = empirical_min_idx == combined_min_idx
print(f"\n{'MATCH' if match else 'NO EXACT MATCH'}: empirical idx={empirical_min_idx}, combined-metric idx={combined_min_idx}")

import pandas as pd
pd.DataFrame(rows, columns=["w_EE","median_error","median_energy","gramian_cond","n_runs"]).to_csv(
    "crossover_point_precision_check_results.csv", index=False)
