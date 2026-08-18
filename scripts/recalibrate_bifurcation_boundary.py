#!/usr/bin/env python3
"""
Recalibrate the w_EE bifurcation boundary under the corrected coupling
convention (N-independent mean-field scaling: uniform(0,0.15)/N*20).

The original boundary (~5.6-5.75) was found under the OLD, unscaled
coupling. That fix made coupling 2x stronger at N=10, shifting the
boundary to a lower w_EE (confirmed: w_EE=5.0+ now all fail to converge
to a stable fixed point, where they used to succeed).

Uses the continuation method (each step warm-starts from the previous
converged solution) across multiple seeds, so the result isn't an
artifact of a poor initial guess -- same approach used to find the
original boundary, just rerun under the corrected coupling.

Run directly: `python3 recalibrate_bifurcation_boundary.py [--n-regions N]`
"""
import argparse
import numpy as np
import simulation

BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                    tau_E=10.0, tau_I=20.0)
N_SEEDS = 4  # matches the original boundary-finding approach


def build_network(N, w_EE, coupling_seed):
    rng = np.random.default_rng(coupling_seed)
    # N-independent mean-field coupling scale -- the corrected convention,
    # matching wc_linear_validity_sweep.py / _parallel.py
    C = rng.uniform(0, 0.15, (N, N)) / N * 20
    np.fill_diagonal(C, 0)
    params = dict(w_EE=w_EE, **BASE_PARAMS)
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)


def find_boundary_for_seed(N, coupling_seed, w_EE_scan, verbose=True):
    """Continuation method: warm-start each w_EE from the previous
    converged solution. Returns the largest w_EE at which a stable fixed
    point was still found, and the smallest at which it failed."""
    E_guess, I_guess = None, None
    last_stable_w_EE = None
    boundary_lo, boundary_hi = None, None

    if verbose:
        print(f"--- coupling seed {coupling_seed} ---")

    for w_EE in w_EE_scan:
        net = build_network(N, w_EE, coupling_seed)
        try:
            E_star, I_star = net.find_fixed_point(E_guess, I_guess)
            J = net.jacobian_at(E_star, I_star)
            max_eig = np.linalg.eigvals(J).real.max()
            is_stable = max_eig < 0
            status = "STABLE" if is_stable else "UNSTABLE (repelling fixed pt)"
            if verbose:
                print(f"  w_EE={w_EE:>6.3f}: max_real_eig={max_eig:>10.6f}  [{status}]")
            if is_stable:
                last_stable_w_EE = w_EE
                E_guess, I_guess = E_star, I_star
            else:
                if boundary_hi is None:
                    boundary_lo, boundary_hi = last_stable_w_EE, w_EE
        except RuntimeError:
            if verbose:
                print(f"  w_EE={w_EE:>6.3f}: solver did not converge (lost the branch)")
            if boundary_hi is None:
                boundary_lo, boundary_hi = last_stable_w_EE, w_EE
            E_guess, I_guess = None, None  # reset, lost the branch

    return last_stable_w_EE, boundary_lo, boundary_hi


def recommend_grid(boundary_estimates, n_points=5, low_start=3.0):
    """Given the boundary estimates across seeds, propose a w_EE grid that
    spans from a well-damped low point up to just below the most
    conservative (lowest) boundary found across all seeds."""
    valid = [b for b in boundary_estimates if b is not None]
    if not valid:
        return None
    safe_upper = min(valid) * 0.98  # small margin below the most conservative boundary
    return list(np.round(np.linspace(low_start, safe_upper, n_points), 4))


REAL_COHORT_N = {
    "UNAM": 68,
    "ADNI-Schaefer400": 400,
    "HCP-AUD": 410,
    "ADNI-Schaefer400+TianS3": 450,
}


def run_full_recalibration(N, label=""):
    """Runs the coarse-then-fine boundary search for one N value, returns
    the recommended grid and per-seed boundary estimates."""
    print(f"\n{'#'*70}\n{label} (N={N})\n{'#'*70}")

    coarse_scan = np.arange(3.0, 6.01, 0.5)
    print(f"=== Coarse scan ===")
    boundaries_coarse = []
    for seed in range(N_SEEDS):
        last_stable, lo, hi = find_boundary_for_seed(N, seed, coarse_scan)
        boundaries_coarse.append((lo, hi))
        print(f"  seed {seed}: last stable w_EE={last_stable}, "
              f"boundary between {lo} and {hi}\n")

    all_lo = [b[0] for b in boundaries_coarse if b[0] is not None]
    all_hi = [b[1] for b in boundaries_coarse if b[1] is not None]
    fine_lo = min(all_lo) if all_lo else 3.0
    fine_hi = max(all_hi) if all_hi else 6.0
    fine_scan = np.arange(fine_lo, fine_hi + 0.051, 0.05)

    print(f"=== Fine scan, range {fine_lo} to {fine_hi} ===")
    final_boundaries = []
    for seed in range(N_SEEDS):
        last_stable, lo, hi = find_boundary_for_seed(N, seed, fine_scan)
        final_boundaries.append(last_stable)
        print(f"  seed {seed}: boundary between {lo} and {hi}\n")

    recommended = recommend_grid(final_boundaries)
    print(f"--- {label} (N={N}) summary ---")
    print(f"Last-stable w_EE per seed: {final_boundaries}")
    print(f"Recommended W_EE_VALUES grid: {recommended}\n")
    return {"N": N, "label": label, "boundaries": final_boundaries,
            "recommended_grid": recommended}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-regions", type=int, default=None,
                         help="Single N to check. If omitted, checks all "
                              "four real cohort N values (68, 400, 410, 450).")
    args = parser.parse_args()

    if args.n_regions is not None:
        run_full_recalibration(args.n_regions, label="custom")
    else:
        all_results = []
        for label, N in REAL_COHORT_N.items():
            result = run_full_recalibration(N, label=label)
            all_results.append(result)

        print(f"\n{'='*70}\nFINAL SUMMARY -- all four real cohort N values\n{'='*70}")
        for r in all_results:
            print(f"  {r['label']:>28} (N={r['N']:>3}): "
                  f"boundaries={r['boundaries']}, grid={r['recommended_grid']}")

        # Flag whether a single shared grid would work across all cohorts,
        # or whether per-cohort grids are needed
        all_min_boundaries = [min(r['boundaries']) for r in all_results if r['boundaries']]
        if all_min_boundaries:
            most_conservative = min(all_min_boundaries)
            spread = max(all_min_boundaries) - most_conservative
            print(f"\nMost conservative boundary across all cohorts: {most_conservative:.3f}")
            print(f"Spread of boundaries across cohorts: {spread:.3f}")
            if spread < 0.3:
                print("=> Boundaries are close across cohorts; a SINGLE shared "
                      "W_EE_VALUES grid (based on the most conservative boundary) "
                      "should work for all four.")
            else:
                print("=> Boundaries differ meaningfully across cohorts; consider "
                      "PER-COHORT W_EE_VALUES grids rather than one shared grid.")
