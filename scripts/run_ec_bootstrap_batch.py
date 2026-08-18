#!/usr/bin/env python3
"""
EC-estimation uncertainty batch run. Moving-block bootstrap EC estimation +
driver-node rank-stability, on actual per-subject BOLD timeseries + SC
(block_length=15, n_boot starting at 200, both Kendall's tau and top-k
Jaccard reported).

File-discovery conventions match the existing project scripts exactly
(run_hcp_aud_batch.py / run_adni_nctn_batch.py) -- same --ts-dir/--sc-dir/
--ts-suffix/--sc-suffix pattern, subject ID derived by stripping the
suffix from the filename.

IMPORTANT -- thread-oversubscription fix: when running many worker
PROCESSES in parallel (via multiprocessing.Pool), NumPy's BLAS backend
(OpenBLAS/MKL) can ALSO try to multithread internally by default, causing
each of the N worker processes to compete for all available cores instead
of each using exactly one. This produces a uniform, dramatic slowdown
regardless of data size -- exactly the pattern found when real per-subject
runs took ~70-140x longer than synthetic-data benchmarks predicted, with
no dependence on N or T. Capping each process to one thread must happen
BEFORE numpy is imported anywhere, hence these lines are first in the file.

Usage (HCP-AUD style):
    python3 run_ec_bootstrap_batch.py \\
        --ts-dir /path/to/timeseries \\
        --sc-dir /path/to/SC_matrices \\
        --ts-suffix _native_timeseries.npy \\
        --sc-suffix _SC_SIFT2_410.csv \\
        --out-dir /path/to/output \\
        --n-boot 200 --block-length 15 \\
        --limit 3        # dry-run on 3 subjects first
        --n-workers 80    # parallel across subjects

Usage (ADNI style, per atlas):
    python3 run_ec_bootstrap_batch.py \\
        --ts-dir /path/to/timeseries \\
        --sc-dir /path/to/connectomes \\
        --ts-suffix _schaefer400_native_timeseries.npy \\
        --sc-suffix _schaefer400_connectome.csv \\
        --out-dir /path/to/output \\
        --n-boot 200 --block-length 15
"""
import os
# Must happen before numpy/scipy are imported anywhere -- caps each worker
# process to a single BLAS thread, so parallelism comes from the process
# pool (n_workers processes), not from each process also fighting for all
# cores internally. See module docstring above for why this matters.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from neurosim import connectivity
from neurosim import physics


def discover_subjects(ts_dir, sc_dir, ts_suffix, sc_suffix):
    """Identical logic to run_hcp_aud_batch.py's discover_subjects, reused
    here for consistency with the rest of the project."""
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


def ctrl_func(A):
    """Average controllability after normalisation -- matches the
    benchmark script's convention exactly."""
    A_norm = physics.normalise_matrix(A, target_rho=0.9)
    return physics.average_controllability(A_norm)


def process_one_subject_multi_boot(args):
    """Same pipeline as process_one_subject, but runs MULTIPLE n_boot
    values for one subject to check rank-stability convergence -- the
    convergence-check step that was never actually done ('plot rank-stability
    vs. B to check convergence before deciding whether to scale to
    500-1000'). Returns one row per (subject, n_boot) combination."""
    sub, ts_path, sc_path, n_boot_list, block_length, out_dir = args
    rows = []
    try:
        X = np.load(ts_path)
        SC = np.loadtxt(sc_path, delimiter=",")

        orientation_note = "as-is (N,T)"
        if X.shape[0] != SC.shape[0]:
            if X.shape[1] == SC.shape[0]:
                X = X.T
                orientation_note = f"transposed from (T,N) to (N,T)"
            else:
                return [{"subject_id": sub, "n_boot": None,
                         "status": f"dimension mismatch: X={X.shape}, SC={SC.shape}"}]

        T = X.shape[1]
        if block_length == "auto":
            effective_block_length = max(1, round(np.sqrt(T)))
        else:
            effective_block_length = block_length

        for n_boot in n_boot_list:
            t0 = time.time()
            EC_boot = connectivity.block_bootstrap_ec(
                X, SC, n_boot=n_boot, block_length=effective_block_length, seed=0)
            result = connectivity.driver_node_rank_stability(EC_boot, ctrl_func, top_k=5)
            elapsed = time.time() - t0
            rows.append({
                "subject_id": sub,
                "status": "ok",
                "orientation": orientation_note,
                "n_regions": X.shape[0],
                "n_timepoints": T,
                "block_length_used": effective_block_length,
                "n_boot": n_boot,
                "kendall_tau_mean": result["kendall_tau_mean"],
                "kendall_tau_std": result["kendall_tau_std"],
                "jaccard_topk_mean": result["jaccard_topk_mean"],
                "jaccard_topk_std": result["jaccard_topk_std"],
                "elapsed_seconds": elapsed,
            })
        return rows
    except Exception as e:
        return [{"subject_id": sub, "n_boot": None, "status": f"ERROR: {e}"}]


def process_one_subject(args):
    sub, ts_path, sc_path, n_boot, block_length, out_dir, save_ec_boot = args
    t0 = time.time()
    try:
        X = np.load(ts_path)
        SC = np.loadtxt(sc_path, delimiter=",")

        # Auto-detect (T,N) vs (N,T) orientation -- connectivity.py's
        # functions expect (N,T). Real HCP files were found to be stored
        # as (T,N) (e.g. (4800,410)), not the assumed (N,T). Transpose
        # rather than silently guess; log which orientation was found.
        orientation_note = "as-is (N,T)"
        if X.shape[0] != SC.shape[0]:
            if X.shape[1] == SC.shape[0]:
                X = X.T
                orientation_note = f"transposed from (T,N)={X.T.shape} to (N,T)={X.shape}"
            else:
                return {"subject_id": sub, "status": f"dimension mismatch: "
                        f"X={X.shape} (tried both orientations), SC={SC.shape}"}

        T = X.shape[1]
        # Adaptive block length: sqrt(T) rule (same logic that gave 15 for
        # ADNI's T~197-200), unless an explicit value was passed. HCP's
        # real T=4800 needs its own value (~69), not ADNI's 15 -- using a
        # fixed 15 for both cohorts under-blocks HCP's much longer,
        # presumably differently-autocorrelated series.
        if block_length == "auto":
            effective_block_length = max(1, round(np.sqrt(T)))
        else:
            effective_block_length = block_length

        EC_boot = connectivity.block_bootstrap_ec(
            X, SC, n_boot=n_boot, block_length=effective_block_length, seed=0)
        result = connectivity.driver_node_rank_stability(EC_boot, ctrl_func, top_k=5)

        if save_ec_boot:
            np.save(os.path.join(out_dir, f"{sub}_EC_boot.npy"), EC_boot)

        elapsed = time.time() - t0
        return {
            "subject_id": sub,
            "status": "ok",
            "orientation": orientation_note,
            "n_regions": X.shape[0],
            "n_timepoints": T,
            "block_length_used": effective_block_length,
            "n_boot": n_boot,
            "kendall_tau_mean": result["kendall_tau_mean"],
            "kendall_tau_std": result["kendall_tau_std"],
            "jaccard_topk_mean": result["jaccard_topk_mean"],
            "jaccard_topk_std": result["jaccard_topk_std"],
            "elapsed_seconds": elapsed,
        }
    except Exception as e:
        return {"subject_id": sub, "status": f"ERROR: {e}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--ts-suffix", default="_native_timeseries.npy")
    ap.add_argument("--sc-suffix", default="_SC_SIFT2_410.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--block-length", default="auto",
                     help="Block length in TRs for the moving-block "
                          "bootstrap. Default 'auto' uses sqrt(T) rounded "
                          "(the same rule that gave 15 for ADNI's T~197-200; "
                          "HCP's T=4800 gets ~69 automatically). Pass an "
                          "integer to override and use a fixed value for "
                          "every subject regardless of their T.")
    ap.add_argument("--limit", type=int, default=0,
                     help="Process only the first N subjects (0 = all). "
                          "Use --limit 3 for a dry-run first.")
    ap.add_argument("--subject-id", default=None,
                     help="Process only this specific subject ID (exact "
                          "match), regardless of sort order. Use this to "
                          "rerun one failed/timed-out subject rather than "
                          "--limit, which just grabs the first N by sort "
                          "order and may not include the one you want.")
    ap.add_argument("--append", action="store_true",
                     help="Append results to an existing output CSV rather "
                          "than overwriting it (no header rewritten). Use "
                          "when completing a set after a partial failure, "
                          "e.g. rerunning one timed-out subject to add back "
                          "into an otherwise-complete results file.")
    ap.add_argument("--n-workers", type=int, default=None,
                     help="Parallel workers (default: all detected cores)")
    ap.add_argument("--save-ec-boot", action="store_true",
                     help="Also save the full (n_boot,N,N) EC array per "
                          "subject -- large, off by default")
    ap.add_argument("--convergence-check", action="store_true",
                     help="Run the convergence check instead of the "
                          "normal batch: sweeps n_boot over --n-boot-list "
                          "for each subject, so rank-stability vs. B can be "
                          "plotted before deciding on a final B. Use on a "
                          "small subject set (--limit), not the full cohort.")
    ap.add_argument("--n-boot-list", default="50,100,200,400",
                     help="Comma-separated n_boot values for --convergence-check")
    ap.add_argument("--timeout-per-subject", type=int, default=14400,
                     help="Max seconds to wait for any single subject "
                          "(default 4 hours) before marking it TIMEOUT and "
                          "moving on -- prevents one pathological subject "
                          "from hanging the entire batch. Adjust based on "
                          "your actual per-subject n_boot=200 timing once "
                          "known.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.block_length != "auto":
        try:
            args.block_length = int(args.block_length)
        except ValueError:
            raise SystemExit(f"--block-length must be 'auto' or an integer, "
                              f"got: {args.block_length!r}")

    subjects = discover_subjects(args.ts_dir, args.sc_dir,
                                  args.ts_suffix, args.sc_suffix)
    subject_list = sorted(subjects.keys())
    if args.subject_id is not None:
        if args.subject_id not in subjects:
            raise SystemExit(f"--subject-id {args.subject_id!r} not found "
                              f"among the {len(subjects)} discovered subjects "
                              f"(with both timeseries and SC present).")
        subject_list = [args.subject_id]
    elif args.limit > 0:
        subject_list = subject_list[:args.limit]
    print(f"Found {len(subjects)} subjects with both timeseries and SC; "
          f"processing {len(subject_list)}.")

    from multiprocessing import cpu_count
    n_workers = args.n_workers or cpu_count()
    print(f"Running with {n_workers} parallel workers ({cpu_count()} cores detected)")

    if args.convergence_check:
        n_boot_list = [int(x) for x in args.n_boot_list.split(",")]
        print(f"CONVERGENCE CHECK mode -- sweeping n_boot over {n_boot_list} "
              f"per subject (run before committing to a final B)")
        tasks = [(sub, subjects[sub][0], subjects[sub][1], n_boot_list,
                  args.block_length, args.out_dir)
                 for sub in subject_list]

        t_start = time.time()
        with Pool(n_workers) as pool:
            results_nested = pool.map(process_one_subject_multi_boot, tasks)
        elapsed = time.time() - t_start
        results = [row for sublist in results_nested for row in sublist]

        df = pd.DataFrame(results)
        n_ok = (df["status"] == "ok").sum()
        print(f"\nDone: {n_ok}/{len(df)} (subject, n_boot) combinations "
              f"succeeded, in {elapsed:.1f}s ({elapsed/60:.1f} min) wall-clock")
        out_path = os.path.join(args.out_dir, "ec_bootstrap_convergence_check.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")
        print("\nNext: plot kendall_tau_mean / jaccard_topk_mean vs. n_boot "
              "per subject to check whether they've stabilized by B=200, or "
              "need to go higher (500-1000).")

    else:
        tasks = [(sub, subjects[sub][0], subjects[sub][1], args.n_boot,
                  args.block_length, args.out_dir, args.save_ec_boot)
                 for sub in subject_list]

        out_path = os.path.join(args.out_dir, "ec_bootstrap_batch_results.csv")
        # Incremental save: open in write mode now, header written once,
        # then append (with flush) after EVERY subject -- so if the
        # process dies partway through, everything completed so far is
        # already on disk, not lost.
        # --append: open in append mode instead, and treat the header as
        # already written (don't rewrite it) if the file already exists
        # and is non-empty -- for completing a set after a partial failure
        # without clobbering the existing rows.
        results = []
        append_mode = args.append and os.path.exists(out_path) and os.path.getsize(out_path) > 0
        csv_file = open(out_path, "a" if append_mode else "w", newline="")
        header_written = [True if append_mode else False]
        if append_mode:
            print(f"--append: adding to existing {out_path} "
                  f"(header not rewritten)")
        # Fixed schema, written for EVERY row regardless of success/failure --
        # bug found during testing: if a failed row (only subject_id+status)
        # got written before a successful one (12 columns), the CSV
        # structure corrupted and became unparseable. Reindexing every row
        # to this fixed column set (missing values become blank) prevents
        # that regardless of what order subjects finish in.
        CSV_COLUMNS = ["subject_id", "status", "orientation", "n_regions",
                       "n_timepoints", "block_length_used", "n_boot",
                       "kendall_tau_mean", "kendall_tau_std",
                       "jaccard_topk_mean", "jaccard_topk_std", "elapsed_seconds"]

        def write_row(row):
            row_df = pd.DataFrame([row]).reindex(columns=CSV_COLUMNS)
            row_df.to_csv(csv_file, index=False, header=not header_written[0],
                          lineterminator="\n")
            header_written[0] = True
            csv_file.flush()
            os.fsync(csv_file.fileno())

        t_start = time.time()
        with Pool(n_workers) as pool:
            # apply_async per task (not imap_unordered) so each result can
            # be fetched with an individual timeout -- one hung subject
            # can't block the rest of the batch or the final save.
            async_results = {sub: pool.apply_async(process_one_subject, (task,))
                              for sub, task in zip(subject_list, tasks)}

            for i, sub in enumerate(subject_list, 1):
                elapsed_so_far = time.time() - t_start
                try:
                    r = async_results[sub].get(timeout=args.timeout_per_subject)
                except Exception as e:
                    is_timeout = "TimeoutError" in type(e).__name__
                    status = (f"TIMEOUT after {args.timeout_per_subject}s"
                               if is_timeout else f"ERROR: {e}")
                    r = {"subject_id": sub, "status": status}

                status_note = r.get("status", "?")
                print(f"  [{i}/{len(tasks)}] subject {r.get('subject_id', sub)} "
                      f"finished: {status_note} "
                      f"(t={time.time() - t_start:.0f}s so far)", flush=True)
                results.append(r)
                write_row(r)

        csv_file.close()
        elapsed = time.time() - t_start

        df = pd.DataFrame(results)
        n_ok = (df["status"] == "ok").sum()
        n_failed = len(df) - n_ok
        print(f"\nDone: {n_ok} subjects succeeded, {n_failed} failed, "
              f"in {elapsed:.1f}s ({elapsed/60:.1f} min) wall-clock")
        if n_failed > 0:
            print("Failed subjects:")
            print(df[df["status"] != "ok"][["subject_id", "status"]].to_string(index=False))

        print(f"Saved results to {out_path} (written incrementally, one "
              f"subject at a time -- safe even if the run had been "
              f"interrupted partway through)")
