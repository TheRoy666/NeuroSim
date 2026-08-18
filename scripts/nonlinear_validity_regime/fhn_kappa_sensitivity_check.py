#!/usr/bin/env python3
"""
Coupling-scale sensitivity check for the FHN toy-scale result. The
opposite-direction finding (error INCREASING approaching the Hopf
boundary, kappa=0.1) was flagged as needing this check before being
trusted -- WC itself needed exactly this kind of check (real coupling
revealed the true, more complex shape of a toy-scale finding that looked
clean at one specific coupling convention).

For each kappa: recalibrate the boundary (a different coupling scale can
shift where the Hopf bifurcation sits, same as WC), build a safe I_ext
grid below it, run the full toy sweep, report the sign and significance
of the bifurcation-proximity correlation.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fhn_simulation as fhn
import physics
from simulation import discretize_system

N = 10
N_SEEDS = 8
DT = 0.5
T_VALUES = [5, 20, 50, 100, 200]
TARGET_SCALES = [0.005, 0.01, 0.02, 0.05, 0.1]
N_TARGET_DIRECTIONS_PER_SEED = 3

KAPPA_VALUES = [0.02, 0.05, 0.1, 0.2, 0.4]


def find_boundary(kappa, C_ref, verbose=True):
    """Coarse-then-fine boundary search, same method as the original FHN
    boundary check, now parameterized by kappa."""
    v_guess, w_guess = None, None
    last_stable = None
    for I_ext in np.arange(0.0, 1.01, 0.05):
        params = dict(**fhn.FHN_DEFAULT_PARAMS)
        params["kappa"] = kappa
        net = fhn.FitzHughNagumoNetwork(n_regions=len(C_ref), C=C_ref, I_ext=I_ext, node_params=params)
        try:
            v_star, w_star = net.find_fixed_point(v_guess, w_guess)
            J = net.jacobian_at(v_star, w_star)
            max_eig = np.linalg.eigvals(J).real.max()
            if max_eig < 0:
                last_stable = I_ext
                v_guess, w_guess = v_star, w_star
            else:
                break
        except RuntimeError:
            break
    if last_stable is None:
        return None
    # Fine scan
    v_guess, w_guess = None, None
    last_stable_fine = None
    for I_ext in np.arange(max(0, last_stable - 0.05), last_stable + 0.051, 0.005):
        params = dict(**fhn.FHN_DEFAULT_PARAMS)
        params["kappa"] = kappa
        net = fhn.FitzHughNagumoNetwork(n_regions=len(C_ref), C=C_ref, I_ext=I_ext, node_params=params)
        try:
            v_star, w_star = net.find_fixed_point(v_guess, w_guess)
            J = net.jacobian_at(v_star, w_star)
            max_eig = np.linalg.eigvals(J).real.max()
            if max_eig < 0:
                last_stable_fine = I_ext
                v_guess, w_guess = v_star, w_star
            else:
                break
        except RuntimeError:
            break
    return last_stable_fine if last_stable_fine is not None else last_stable


def run_sweep_for_kappa(kappa, boundary):
    safe_max = boundary * 0.9  # safety margin, mirrors WC's 0.98 factor loosely
    I_EXT_VALUES = list(np.linspace(0.0, safe_max, 5))

    all_rows = []
    n_skipped = 0
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        C = rng.uniform(0, 0.15, (N, N)) / N * 20
        np.fill_diagonal(C, 0)

        for I_ext in I_EXT_VALUES:
            params = dict(**fhn.FHN_DEFAULT_PARAMS)
            params["kappa"] = kappa
            net = fhn.FitzHughNagumoNetwork(n_regions=N, C=C, I_ext=I_ext, node_params=params)
            try:
                v_star, w_star = net.find_fixed_point()
            except RuntimeError:
                n_skipped += 1
                continue
            J = net.jacobian_at(v_star, w_star)
            max_real_eig = np.linalg.eigvals(J).real.max()
            if max_real_eig >= 0:
                n_skipped += 1
                continue

            G = np.vstack([np.eye(N), np.zeros((N, N))])
            A, B = discretize_system(J, G, DT)
            x_star = np.concatenate([v_star, w_star])

            for T in T_VALUES:
                for target_scale in TARGET_SCALES:
                    for dir_seed in range(N_TARGET_DIRECTIONS_PER_SEED):
                        drng = np.random.default_rng(
                            hash((seed, int(I_ext * 1000), T, int(target_scale * 1000), dir_seed)) % (2**32)
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

                        all_rows.append({
                            "kappa": kappa, "seed": seed, "I_ext": I_ext,
                            "T": T, "target_scale": target_scale,
                            "rel_reachability_error": rel_err,
                        })
    return pd.DataFrame(all_rows), n_skipped


if __name__ == "__main__":
    rng_ref = np.random.default_rng(0)
    C_ref = rng_ref.uniform(0, 0.15, (N, N)) / N * 20
    np.fill_diagonal(C_ref, 0)

    summary_rows = []
    t_start = time.time()

    for kappa in KAPPA_VALUES:
        print(f"\n=== kappa={kappa} ===")
        boundary = find_boundary(kappa, C_ref)
        if boundary is None:
            print(f"  No stable boundary found for kappa={kappa} -- skipping")
            continue
        print(f"  Boundary (reference seed): I_ext={boundary:.3f}")

        df, n_skipped = run_sweep_for_kappa(kappa, boundary)
        if len(df) == 0:
            print(f"  All combos skipped for kappa={kappa}")
            continue

        rho, p = spearmanr(df['I_ext'], df['rel_reachability_error'])
        medians = df.groupby('I_ext')['rel_reachability_error'].median()
        print(f"  n_skipped={n_skipped}, rows={len(df)}")
        print(f"  Medians by I_ext:\n{medians.to_string()}")
        print(f"  Spearman: rho={rho:.4f}, p={p:.2e}, sign={'POSITIVE (worse near boundary)' if rho > 0 else 'NEGATIVE (better near boundary)'}")

        summary_rows.append({"kappa": kappa, "boundary": boundary, "rho": rho, "p": p, "n_rows": len(df)})

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"SUMMARY -- sign consistency across coupling scales ({elapsed:.1f}s total)")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    signs = np.sign(summary_df['rho'])
    if (signs == signs.iloc[0]).all():
        print(f"\n=> SIGN CONSISTENT across all {len(summary_df)} kappa values tested "
              f"({'positive/worse-near-boundary' if signs.iloc[0] > 0 else 'negative/better-near-boundary'}). "
              f"The opposite-direction finding does NOT appear to be a coupling-scale artifact.")
    else:
        print(f"\n=> SIGN FLIPS across kappa values tested. The direction of the "
              f"bifurcation-proximity effect IS sensitive to coupling scale -- "
              f"cannot report a single direction as general without specifying "
              f"which regime it holds in.")
    summary_df.to_csv("fhn_kappa_sensitivity_summary.csv", index=False)
