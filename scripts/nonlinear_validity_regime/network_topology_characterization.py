#!/usr/bin/env python3
"""
Network topology characterization -- quantifies precisely what this
project has so far only characterized indirectly (edge-weight ratios,
rewiring difficulty) about HCP's connectomes being more hub-concentrated
than either ADNI atlas.

Two complementary metrics per subject:

1. Node-strength heterogeneity (coefficient of variation of the
   weighted degree / row-sum sequence). Simple, standard, no null model
   needed -- a direct answer to "how unevenly is total connection
   weight distributed across nodes."

2. Normalized weighted rich-club coefficient (Opsahl et al. 2008): for
   a range of degree thresholds k, measures whether the edges among
   high-degree nodes carry disproportionately large weight, normalized
   against a degree-preserving null ensemble so the result isn't just
   restating the degree sequence itself. phi_norm(k) > 1 means genuine
   rich-club organization -- hubs connect to each other more strongly
   than degree alone would predict. phi_norm(k) ~ 1 means the apparent
   richness is fully explained by degree sequence alone.

The null ensemble for (2) reuses null_model_rewiring.py's
rewire_and_redistribute_weights directly, not a separate
implementation -- the same degree-preserving, weight-redistributing
null model already built and verified for the interior-minimum
comparison. Deliberately uses a MUCH lower n_swaps_per_edge (2, with
many quick independent realizations averaged together) than that
comparison did, since rich-club normalization needs a null ensemble
that's adequately mixed, not the same rigor as a primary confirmatory
result -- using the full 5x/thorough target here would cost far more
than this descriptive check needs.

Run directly:
    python3 network_topology_characterization.py \\
        --sc-dir /path/to/SC_matrices --sc-suffix _SC_SIFT2_410.csv \\
        --n-workers 55
"""
import os
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

from null_model_rewiring import rewire_and_redistribute_weights
from wc_linear_validity_sweep_real_coupling import discover_sc_files

DEFAULT_N_NULL_REALIZATIONS = 20
DEFAULT_NULL_N_SWAPS_PER_EDGE = 2
DEFAULT_K_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def weighted_rich_club(SC, k_fractions):
    """
    phi_w(k) for a range of degree-rank thresholds, expressed as top
    fractions of nodes by degree (e.g. 0.10 = top 10% highest-degree
    nodes) rather than raw degree values, so results are comparable
    across cohorts/atlases with different N.
    """
    N = SC.shape[0]
    degree = (SC > 0).sum(axis=1)
    weights_sorted_desc = np.sort(SC[np.triu_indices(N, k=1)])[::-1]

    results = {}
    for frac in k_fractions:
        n_rich = max(2, int(round(frac * N)))
        rich_idx = np.argsort(degree)[-n_rich:]
        sub = SC[np.ix_(rich_idx, rich_idx)]
        W_gt_k = sub[np.triu_indices(n_rich, k=1)].sum()
        E_gt_k = (sub[np.triu_indices(n_rich, k=1)] > 0).sum()
        if E_gt_k == 0:
            results[frac] = np.nan
            continue
        W_ranked = weights_sorted_desc[:E_gt_k].sum()
        results[frac] = W_gt_k / W_ranked if W_ranked > 0 else np.nan
    return results


def process_one_subject(args):
    (subject_id, sc_path, k_fractions, n_null, null_n_swaps_per_edge,
     seed) = args
    try:
        SC_raw = np.loadtxt(sc_path, delimiter=",")
        N = SC_raw.shape[0]
        np.fill_diagonal(SC_raw, 0)

        strength = SC_raw.sum(axis=1)
        strength_cv = float(strength.std() / strength.mean())

        phi_real = weighted_rich_club(SC_raw, k_fractions)

        phi_null_all = {f: [] for f in k_fractions}
        for i in range(n_null):
            SC_null, _ = rewire_and_redistribute_weights(
                SC_raw, n_swaps_per_edge=null_n_swaps_per_edge,
                seed=(seed + i) if seed is not None else None,
                time_budget_s=60)
            phi_n = weighted_rich_club(SC_null, k_fractions)
            for f in k_fractions:
                phi_null_all[f].append(phi_n[f])

        row = {"subject_id": subject_id, "N": N, "strength_cv": strength_cv}
        for f in k_fractions:
            phi_null_mean = np.nanmean(phi_null_all[f])
            row[f"phi_real_top{int(f*100)}pct"] = phi_real[f]
            row[f"phi_null_mean_top{int(f*100)}pct"] = phi_null_mean
            row[f"phi_norm_top{int(f*100)}pct"] = (
                phi_real[f] / phi_null_mean if phi_null_mean and not np.isnan(phi_null_mean) else np.nan)
        row["status"] = "ok"
        return row
    except Exception as e:
        return {"subject_id": subject_id, "status": f"ERROR: {e}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--sc-suffix", default="_SC_SIFT2_410.csv")
    ap.add_argument("--n-subjects", type=int, default=0,
                     help="0 = all subjects found")
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--n-null-realizations", type=int, default=DEFAULT_N_NULL_REALIZATIONS,
                     help="Independent null rewirings averaged per subject "
                          "for rich-club normalization -- more realizations "
                          "reduce noise in the normalization but cost "
                          "proportionally more time.")
    ap.add_argument("--null-n-swaps-per-edge", type=int, default=DEFAULT_NULL_N_SWAPS_PER_EDGE,
                     help="Deliberately much lower than the interior-minimum "
                          "comparison's target -- this needs an adequately "
                          "mixed null, not maximal randomization, since "
                          "many quick realizations are averaged together.")
    ap.add_argument("--rewire-seed", type=int, default=None)
    ap.add_argument("--out", default="network_topology_characterization_results.csv")
    args = ap.parse_args()

    n_workers = args.n_workers or cpu_count()
    print(f"Running with {n_workers} parallel workers ({cpu_count()} cores detected)")

    files = discover_sc_files(args.sc_dir, args.sc_suffix)
    subjects = sorted(files.keys())
    if args.n_subjects:
        subjects = subjects[:args.n_subjects]
    print(f"Found {len(files)} SC files; using {len(subjects)} subjects")
    print(f"Rich-club thresholds (top % by degree): {DEFAULT_K_FRACTIONS}")
    print(f"Null ensemble: {args.n_null_realizations} realizations/subject, "
          f"n_swaps_per_edge={args.null_n_swaps_per_edge}\n")

    tasks = [
        (sub, files[sub], DEFAULT_K_FRACTIONS, args.n_null_realizations,
         args.null_n_swaps_per_edge,
         (args.rewire_seed + i) if args.rewire_seed is not None else None)
        for i, sub in enumerate(subjects)
    ]

    t_start = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(process_one_subject, tasks)
    elapsed = time.time() - t_start

    df = pd.DataFrame(results)
    n_ok = (df["status"] == "ok").sum() if "status" in df.columns else 0
    print(f"Done: {n_ok}/{len(subjects)} subjects succeeded in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")
    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")

    if n_ok > 0:
        ok_df = df[df["status"] == "ok"]
        print(f"\nstrength_cv: mean={ok_df['strength_cv'].mean():.3f}, "
              f"std={ok_df['strength_cv'].std():.3f}")
        for f in DEFAULT_K_FRACTIONS:
            col = f"phi_norm_top{int(f*100)}pct"
            if col in ok_df.columns:
                print(f"{col}: mean={ok_df[col].mean():.3f}, std={ok_df[col].std():.3f}")
