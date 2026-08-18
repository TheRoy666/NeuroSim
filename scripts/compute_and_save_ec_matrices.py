#!/usr/bin/env python3
"""
Compute and save real per-subject EC matrices (point estimate, not
bootstrapped) -- the recompute step needed for the Manjunatha in/out-hub
asymmetry check, since the original operator-divergence run never saved
its intermediate EC matrices to disk.

Mirrors run_ec_bootstrap_batch.py's proven structure (file discovery,
thread-capping, parallelization, incremental saving, timeout handling)
but simplified: one graphnet EC estimate per subject, no bootstrapping.

Usage (same file-discovery conventions as every other script in this
project -- --ts-dir/--sc-dir/--ts-suffix/--sc-suffix, subject ID derived
by stripping the suffix from the filename):

    python3 compute_and_save_ec_matrices.py \\
        --ts-dir /path/to/timeseries --sc-dir /path/to/SC_matrices \\
        --ts-suffix _native_timeseries.npy --sc-suffix _SC_SIFT2_410.csv \\
        --out-dir /path/to/output --limit 8 --n-workers 40
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np
import connectivity


def discover_subjects(ts_dir, sc_dir, ts_suffix, sc_suffix):
    """Identical logic to run_ec_bootstrap_batch.py's discover_subjects."""
    ts = {f[:-len(ts_suffix)]: os.path.join(ts_dir, f)
          for f in os.listdir(ts_dir) if f.endswith(ts_suffix)}
    sc = {f[:-len(sc_suffix)]: os.path.join(sc_dir, f)
          for f in os.listdir(sc_dir) if f.endswith(sc_suffix)}
    common = sorted(set(ts) & set(sc))
    missing_sc = sorted(set(ts) - set(sc))
    missing_ts = sorted(set(sc) - set(ts))
    if missing_sc:
        print(f"WARNING: {len(missing_sc)} subjects have timeseries but no SC "
              f"(skipped): {missing_sc[:5]}{'...' if len(missing_sc) > 5 else ''}")
    if missing_ts:
        print(f"WARNING: {len(missing_ts)} subjects have SC but no timeseries "
              f"(skipped): {missing_ts[:5]}{'...' if len(missing_ts) > 5 else ''}")
    return {sub: (ts[sub], sc[sub]) for sub in common}


def discover_subjects_shared_sc(ts_dir, ts_suffix):
    """For cohorts with NO individual SC (e.g. UNAM, which uses a single
    shared template SC across all subjects, not per-subject files) --
    discovers subjects from timeseries alone, pairs every one of them
    with the same shared SC matrix. Confirmed necessary, not assumed:
    the normal discover_subjects would silently find zero matches here,
    since there's no per-subject naming pattern to extract from a single
    shared SC filename like 'template_SC_dk68.csv'."""
    ts = {f[:-len(ts_suffix)]: os.path.join(ts_dir, f)
          for f in os.listdir(ts_dir) if f.endswith(ts_suffix)}
    return ts


def process_one_subject(args):
    sub, ts_path, sc_path, out_dir = args
    t0 = time.time()
    try:
        X = np.load(ts_path)
        SC = np.loadtxt(sc_path, delimiter=",")

        orientation_note = "as-is (N,T)"
        if X.shape[0] != SC.shape[0]:
            if X.shape[1] == SC.shape[0]:
                X = X.T
                orientation_note = "transposed from (T,N) to (N,T)"
            else:
                return {"subject_id": sub, "status": f"dimension mismatch: X={X.shape}, SC={SC.shape}",
                        "n_regions": None, "n_timepoints": None, "elapsed_seconds": None}

        EC = connectivity.graphnet_effective_connectivity(X, SC)
        np.save(os.path.join(out_dir, f"{sub}_EC.npy"), EC)
        elapsed = time.time() - t0

        return {"subject_id": sub, "status": "ok", "orientation": orientation_note,
                "n_regions": X.shape[0], "n_timepoints": X.shape[1],
                "elapsed_seconds": elapsed}
    except Exception as e:
        return {"subject_id": sub, "status": f"ERROR: {e}",
                "n_regions": None, "n_timepoints": None, "elapsed_seconds": None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--sc-dir", required=False, default=None,
                     help="Per-subject SC directory. Omit if using --shared-sc.")
    ap.add_argument("--ts-suffix", required=True)
    ap.add_argument("--sc-suffix", required=False, default=None,
                     help="Per-subject SC filename suffix. Omit if using --shared-sc.")
    ap.add_argument("--shared-sc", required=False, default=None,
                     help="Path to a SINGLE shared SC matrix used for every "
                          "subject, for cohorts with no individual SC (e.g. "
                          "UNAM, which uses one template SC across all "
                          "subjects, not per-subject files). If given, "
                          "--sc-dir/--sc-suffix are ignored.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="Number of subjects to process (default: all found). "
                          "8 recommended, matching this project's established "
                          "per-cohort convention -- see MASTER_TODO.md for the "
                          "reasoning (within-subject power from N regions is "
                          "high; the real question is cross-cohort "
                          "generalization, which a small consistent sample "
                          "tests as well as the full cohort would).")
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.shared_sc is not None:
        print(f"Shared-SC mode: every subject paired with {args.shared_sc}")
        subjects_ts = discover_subjects_shared_sc(args.ts_dir, args.ts_suffix)
        subjects = {sub: (path, args.shared_sc) for sub, path in subjects_ts.items()}
        print(f"Found {len(subjects)} subjects with timeseries (SC shared across all).")
    else:
        if args.sc_dir is None or args.sc_suffix is None:
            raise ValueError("Either --shared-sc, or both --sc-dir and --sc-suffix, must be given.")
        subjects = discover_subjects(args.ts_dir, args.sc_dir, args.ts_suffix, args.sc_suffix)
        print(f"Found {len(subjects)} subjects with both timeseries and SC.")

    subject_list = sorted(subjects.keys())
    if args.limit is not None:
        subject_list = subject_list[:args.limit]
    print(f"Processing {len(subject_list)}.")

    n_workers = args.n_workers or len(subject_list)
    print(f"Running with {n_workers} parallel workers.")

    tasks = [(sub, subjects[sub][0], subjects[sub][1], args.out_dir) for sub in subject_list]

    log_path = os.path.join(args.out_dir, "compute_ec_log.csv")
    fieldnames = ["subject_id", "status", "orientation", "n_regions", "n_timepoints", "elapsed_seconds"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        t_start = time.time()
        n_ok, n_failed = 0, 0
        with Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one_subject, tasks)):
                writer.writerow(result)
                f.flush()
                if result["status"] == "ok":
                    n_ok += 1
                else:
                    n_failed += 1
                print(f"  [{i+1}/{len(tasks)}] subject {result['subject_id']} "
                      f"finished: {result['status']} (t={time.time()-t_start:.0f}s so far)")

    elapsed = time.time() - t_start
    print(f"Done: {n_ok} subjects succeeded, {n_failed} failed, in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min) wall-clock")
    print(f"Log saved to {log_path}; EC matrices saved as {{subject}}_EC.npy in {args.out_dir}")
