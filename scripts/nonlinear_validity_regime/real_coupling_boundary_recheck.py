#!/usr/bin/env python3
"""
Real-coupling loader + boundary recheck for Path A1.

Loads real per-subject SC matrices (same file-discovery convention as
Path B's run_ec_bootstrap_batch.py -- --sc-dir/--sc-suffix) and uses them as
the WC network's coupling matrix C, in place of the synthetic
random-uniform placeholder.

IMPORTANT -- why this is a recheck script, not a straight-to-sweep script:
real SC has very different structure from random-uniform coupling (heavy-
tailed degree distribution, hub topology, individual variation) and an
entirely different raw scale (streamline counts or SIFT2 weights, not the
[0, 0.15] synthetic range). Naively plugging it into the already-
calibrated W_EE_VALUES grid risks silently invalidating that calibration
-- the same class of mistake caught earlier when the coupling-scale bug
shifted the bifurcation boundary. This script reruns the SAME boundary-
finding continuation method used for the synthetic-coupling recalibration,
now with real coupling, before any full sweep should be trusted.

Coupling scale note: raw SC is rescaled so its mean off-diagonal edge
weight matches the synthetic placeholder's expected mean
(uniform(0,0.15)/N*20 -> mean ~0.075/N*20 per off-diagonal entry before
self-excitation). This is a defensible starting point, NOT a validated
choice -- that is exactly what this recheck script exists to verify.

Run directly:
    python3 real_coupling_boundary_recheck.py \\
        --sc-dir /path/to/SC_matrices --sc-suffix _SC_SIFT2_410.csv \\
        --n-subjects 4
"""
import argparse
import os
import numpy as np
import simulation

BASE_PARAMS = dict(w_IE=4.0, w_EI=3.0, w_II=2.0, c_E=-2.0, c_I=-2.0,
                    tau_E=10.0, tau_I=20.0)
DT = 2.0


def discover_sc_files(sc_dir, sc_suffix):
    files = {f[:-len(sc_suffix)]: os.path.join(sc_dir, f)
             for f in os.listdir(sc_dir) if f.endswith(sc_suffix)}
    return files


def load_and_scale_real_coupling(sc_path, verbose=True, log_transform=True):
    """Load a raw SC matrix and rescale it to a target total input per
    node (row sum). Real SC's extreme hub-driven skew (a few edges 100s-
    1000s of times the mean) was found to break the fixed-point solver
    via numerical overflow in the sigmoid, EVEN when the mean row-sum is
    correctly matched -- confirmed via diagnostic row-sum-ratio output
    and a direct RuntimeWarning: overflow encountered in exp. Log-
    transforming the raw weights first (standard practice in structural
    connectivity work for exactly this kind of skew) compresses the
    dynamic range before scaling, addressing the actual mechanism rather
    than just the symptom."""
    SC = np.loadtxt(sc_path, delimiter=",")
    N = SC.shape[0]
    np.fill_diagonal(SC, 0)

    off_diag = SC[SC > 0]
    row_sums_raw = SC.sum(axis=1)
    if verbose:
        print(f"    raw SC: mean_edge={off_diag.mean():.4f}, "
              f"max_edge={off_diag.max():.4f} ({off_diag.max()/off_diag.mean():.0f}x mean), "
              f"row_sum mean={row_sums_raw.mean():.2f}, max={row_sums_raw.max():.2f} "
              f"({row_sums_raw.max()/row_sums_raw.mean():.1f}x mean)")

    if log_transform:
        SC_work = np.where(SC > 0, np.log1p(SC), 0)
        if verbose:
            log_row_sums = SC_work.sum(axis=1)
            print(f"    after log1p: row_sum mean={log_row_sums.mean():.2f}, "
                  f"max={log_row_sums.max():.2f} "
                  f"({log_row_sums.max()/log_row_sums.mean():.1f}x mean, "
                  f"was {row_sums_raw.max()/row_sums_raw.mean():.1f}x before)")
    else:
        SC_work = SC

    row_sums_work = SC_work.sum(axis=1)
    target_row_sum_mean = (0.075 / N) * 20 * (N - 1)
    scale_factor = target_row_sum_mean / row_sums_work.mean()
    SC_scaled = SC_work * scale_factor

    row_sums_scaled = SC_scaled.sum(axis=1)
    if verbose:
        print(f"    scaled : row_sum mean={row_sums_scaled.mean():.4f}, "
              f"max={row_sums_scaled.max():.4f} "
              f"({row_sums_scaled.max()/row_sums_scaled.mean():.1f}x mean)")
    return SC_scaled, N


def build_network_real(w_EE, C):
    params = dict(w_EE=w_EE, **BASE_PARAMS)
    N = C.shape[0]
    return simulation.WilsonCowanNetwork(n_regions=N, C=C, node_params=params)


def find_boundary_for_matrix(C, w_EE_scan, label, verbose=True):
    """Returns (first_boundary_lo, boundary_lo, boundary_hi) -- NOT
    'last stable w_EE seen anywhere in the scan'. That quantity is
    unreliable: once continuation is lost at a failure point, the next
    attempt starts from a generic guess rather than warm-starting from
    the lost branch, and can land on a DIFFERENT, unrelated stable point
    (re-entrant stability on a different branch, not confirmation the
    original branch survived) -- confirmed directly: a subject can fail
    to converge across a whole middle range and then show 'STABLE' again
    at higher w_EE, contradicting the real boundary found by the FIRST
    failure. boundary_lo (the last point on the ORIGINAL continued
    branch before the first failure) is the well-defined, safe quantity;
    it is not subject to this ambiguity."""
    E_guess, I_guess = None, None
    last_stable_on_original_branch = None
    boundary_lo, boundary_hi = None, None
    if verbose:
        print(f"--- {label} ---")
    for w_EE in w_EE_scan:
        net = build_network_real(w_EE, C)
        try:
            E_star, I_star = net.find_fixed_point(E_guess, I_guess)
            J = net.jacobian_at(E_star, I_star)
            max_eig = np.linalg.eigvals(J).real.max()
            is_stable = max_eig < 0
            status = "STABLE" if is_stable else "UNSTABLE"
            if verbose:
                print(f"  w_EE={w_EE:>6.3f}: max_real_eig={max_eig:>10.6f}  [{status}]")
            if is_stable and boundary_hi is None:
                # Still on the original continued branch -- update.
                last_stable_on_original_branch = w_EE
                E_guess, I_guess = E_star, I_star
            elif not is_stable and boundary_hi is None:
                boundary_lo, boundary_hi = last_stable_on_original_branch, w_EE
            # else: boundary already found: ANY further "STABLE" results
            # are on a different branch (re-entrant stability), reported
            # in the trace above but NOT used to update the boundary.
        except RuntimeError:
            if verbose:
                print(f"  w_EE={w_EE:>6.3f}: solver did not converge")
            if boundary_hi is None:
                boundary_lo, boundary_hi = last_stable_on_original_branch, w_EE
            E_guess, I_guess = None, None
    return last_stable_on_original_branch, boundary_lo, boundary_hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", default="_SC_SIFT2_410.csv")
    ap.add_argument("--n-subjects", type=int, default=4,
                     help="Number of real subjects to check (matches the "
                          "N_SEEDS=4 used for the original synthetic-"
                          "coupling recalibration)")
    ap.add_argument("--random-sample", action="store_true",
                     help="Select --n-subjects randomly from the full "
                          "cohort instead of taking the first N by sort "
                          "order. Use this when checking whether an "
                          "earlier first-N recheck was representative of "
                          "the whole cohort, not just its earliest-sorted "
                          "subjects.")
    ap.add_argument("--random-seed", type=int, default=42,
                     help="Seed for --random-sample, so the specific "
                          "subjects checked can be reported and "
                          "reproduced, not just described as 'random'.")
    args = ap.parse_args()

    files = discover_sc_files(args.sc_dir, args.sc_suffix)
    all_subjects = sorted(files.keys())
    if args.random_sample:
        import random
        rng = random.Random(args.random_seed)
        subjects = sorted(rng.sample(all_subjects, min(args.n_subjects, len(all_subjects))))
        print(f"Found {len(files)} SC files; checking {len(subjects)} "
              f"RANDOMLY sampled subjects (seed={args.random_seed}): {subjects}\n")
    else:
        subjects = all_subjects[:args.n_subjects]
        print(f"Found {len(files)} SC files; checking {len(subjects)} "
              f"subjects (first-N by sort order, NOT random)\n")

    coarse_scan = np.arange(3.0, 6.01, 0.5)
    coarse_results = []
    for sub in subjects:
        print(f"Loading and scaling {sub}...")
        C, N = load_and_scale_real_coupling(files[sub])
        last_stable, lo, hi = find_boundary_for_matrix(
            C, coarse_scan, label=f"subject {sub} (N={N}) -- coarse")
        coarse_results.append((sub, N, C, last_stable, lo, hi))
        print(f"  -> last stable w_EE={last_stable}, boundary in [{lo}, {hi}]\n")

    n_failed = sum(1 for r in coarse_results if r[3] is None)
    if n_failed > 0:
        print(f"*** {n_failed}/{len(coarse_results)} subjects had NO stable "
              f"fixed point anywhere in the coarse scan -- stopping before "
              f"fine scan. Coupling scale needs further rework. ***")
    else:
        # Fine scan, same two-stage approach as the original synthetic-
        # coupling recalibration -- narrow window around the coarse
        # boundary, 0.05 resolution.
        all_lo = [r[4] for r in coarse_results if r[4] is not None]
        all_hi = [r[5] for r in coarse_results if r[5] is not None]
        fine_lo = min(all_lo) if all_lo else 3.0
        fine_hi = max(all_hi) if all_hi else 6.0
        fine_scan = np.arange(fine_lo, fine_hi + 0.051, 0.05)

        print(f"=== Fine scan, range {fine_lo} to {fine_hi} ===\n")
        fine_bounds = []
        for sub, N, C, _, _, _ in coarse_results:
            last_stable, lo, hi = find_boundary_for_matrix(
                C, fine_scan, label=f"subject {sub} (N={N}) -- fine")
            fine_bounds.append(last_stable)
            print(f"  -> boundary in [{lo}, {hi}]\n")

        print("=" * 70)
        print("SUMMARY -- real-coupling boundary, fine resolution")
        print("=" * 70)
        print(f"Synthetic-coupling boundary (existing grid): ~4.80\n")
        for sub, last_stable in zip(subjects, fine_bounds):
            print(f"  subject {sub}: last stable w_EE={last_stable}")

        valid = [b for b in fine_bounds if b is not None]
        if valid:
            spread = max(valid) - min(valid)
            most_conservative = min(valid)
            print(f"\nSpread across subjects: {spread:.3f}")
            print(f"Most conservative real-coupling boundary: {most_conservative:.3f}")
            safe_upper = most_conservative * 0.98
            recommended_grid = list(np.round(np.linspace(3.0, safe_upper, 5), 4))
            print(f"\nRecommended REAL-COUPLING W_EE_VALUES grid: {recommended_grid}")
            print(f"(Existing synthetic-coupling grid's top point, 4.704, "
                  f"is {'ABOVE' if 4.704 > safe_upper else 'below'} this "
                  f"safe boundary -- {'do NOT reuse it for real coupling' if 4.704 > safe_upper else 'may still be usable'}.)")
