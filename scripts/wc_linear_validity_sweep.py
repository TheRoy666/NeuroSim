#!/usr/bin/env python3
"""
Path A1 -- validity-regime sweep.

Systematically characterizes when linear control-energy predictions are
trustworthy vs. misleading, by driving the REAL nonlinear Wilson-Cowan
network with the linear-optimal control trajectory and measuring realized
reachability error, across three axes:

  1. Operating-point stability margin (how close to bifurcation/limit-cycle
     the fixed point sits -- varied via w_EE, the strongest driver of
     proximity to the oscillatory regime for this parameterization)
  2. Horizon T (in fine dt=2ms steps, per the dt-calibration finding --
     decoupled from real fMRI TR)
  3. Target distance (magnitude of the state perturbation being controlled)

Multiple random coupling matrices (seeds) and target directions are used at
each grid point for robustness, not a single toy example.

Run directly: `python3 path_a1_validity_sweep.py`
Writes results to path_a1_sweep_results.csv in the same directory.
"""
import time
import numpy as np
import pandas as pd
import simulation
import physics

# ---- Fixed settings ----
N = 10
DT = 2.0  # ms, per dt-calibration session (fine, decoupled from real TR)
BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                    tau_E=10.0, tau_I=20.0)

# ---- Sweep axes ----
W_EE_VALUES = [3.0, 3.426, 3.852, 4.278, 4.704]  # FINAL: validated across all
                                                   # four real cohort N (68, 400,
                                                   # 410, 450) -- boundary is
                                                   # ~4.80 at every N, spread=0.000,
                                                   # confirming a single shared
                                                   # grid works for all cohorts
T_VALUES = [5, 20, 50, 100, 200]              # dt-steps: 10ms - 400ms of neural time
TARGET_SCALES = [0.005, 0.01, 0.02, 0.05, 0.1]
N_COUPLING_SEEDS = 8
N_TARGET_DIRECTIONS_PER_SEED = 3


def build_network(w_EE, coupling_seed):
    rng = np.random.default_rng(coupling_seed)
    # mean-field coupling scale, N-independent -- without this, total input
    # per node grows with N, which caused a fixed-point convergence failure
    # at N=100 during compute-cost benchmarking (unscaled coupling that was
    # fine at N=10 pushed the system out of a stable regime at larger N).
    # Matches wc_linear_validity_sweep_parallel.py's convention exactly.
    C = rng.uniform(0, 0.15, (N, N)) / N * 20
    np.fill_diagonal(C, 0)
    params = dict(w_EE=w_EE, **BASE_PARAMS)
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)


def check_stable_fixed_point(net):
    """Returns (E_star, I_star, max_real_eig, is_stable) or None if the
    fixed-point solver itself fails to converge."""
    try:
        E_star, I_star = net.find_fixed_point()
    except RuntimeError:
        return None
    J = net.jacobian_at(E_star, I_star)
    max_real_eig = np.linalg.eigvals(J).real.max()
    return E_star, I_star, max_real_eig, (max_real_eig < 0)


def run_sweep():
    rows = []
    t_start = time.time()
    n_skipped_unstable = 0

    for coupling_seed in range(N_COUPLING_SEEDS):
        for w_EE in W_EE_VALUES:
            net = build_network(w_EE, coupling_seed)
            fp_result = check_stable_fixed_point(net)
            if fp_result is None:
                n_skipped_unstable += 1
                continue
            E_star, I_star, max_real_eig, is_stable = fp_result
            if not is_stable:
                n_skipped_unstable += 1
                continue

            J = net.jacobian_at(E_star, I_star)
            G = simulation.input_jacobian_at(net, E_star, I_star)
            A, B = simulation.discretize_system(J, G, DT)
            rho_A = np.max(np.abs(np.linalg.eigvals(A)))
            x_star = np.concatenate([E_star, I_star])

            for T in T_VALUES:
                for target_scale in TARGET_SCALES:
                    for dir_seed in range(N_TARGET_DIRECTIONS_PER_SEED):
                        drng = np.random.default_rng(
                            hash((coupling_seed, int(w_EE*10), T,
                                  int(target_scale*1000), dir_seed)) % (2**32)
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
                            "stability_margin": -max_real_eig,  # larger = more damped/stable
                            "rho_A": rho_A,
                            "T": T,
                            "target_scale": target_scale,
                            "dir_seed": dir_seed,
                            "energy": energy,
                            "rel_reachability_error": rel_err,
                        })

    elapsed = time.time() - t_start
    print(f"Sweep complete: {len(rows)} runs in {elapsed:.1f}s "
          f"({n_skipped_unstable} (seed, w_EE) combos skipped as unstable/non-convergent)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_sweep()
    df.to_csv("path_a1_sweep_results.csv", index=False)
    print(f"Saved {len(df)} rows to path_a1_sweep_results.csv")
