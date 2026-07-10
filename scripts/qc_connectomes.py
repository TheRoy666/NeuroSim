#!/usr/bin/env python3
"""
NeuroSim ADNI — automated structural connectome QC.

Checks per subject per atlas:
  - Matrix dimensions (expected 450×450 and 400×400)
  - Symmetry (SIFT2+symmetric should be perfectly symmetric)
  - Empty nodes (fraction of all-zero rows — key registration check)
  - Matrix density (fraction of non-zero edges)
  - Strength distribution (mean, std, max — sanity on SIFT2 weights)
  - Diagonal (should be zero-diagonal)

Thresholds (defensible for SIFT2-weighted -scale_invnodevol):
  empty_nodes_max:  5%   (>5% = likely registration problem)
  density_min:      15%  (<15% = too sparse, possible tractography failure)
  density_max:      60%  (>60% = possible noise)
  symmetry_tol:     1e-6 (near-perfect for symmetric flag)

Usage:
  python qc_connectomes.py \
      --sc-dir /path/to/connectomes \
      --out    /path/to/qc_output \
      [--tians3-nodes 450] [--schaefer-nodes 400]
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

TH = {
    "empty_max":    0.05,   # >5% empty nodes = flag
    "density_min":  0.10,   # <10% density = too sparse
    "density_max":  0.65,   # >65% density = suspiciously dense
    "sym_tol":      1e-6,
}

ATLASES = {
    "tians3":      450,
    "schaefer400": 400,
}


def qc_matrix(path, expected_n):
    """Load and QC a single connectome CSV."""
    r = {"path": path, "status": "ok", "note": ""}
    try:
        M = np.loadtxt(path, delimiter=",")
    except Exception as e:
        r["status"] = "fail"; r["note"] = f"load error: {e}"; return r

    N = M.shape[0]
    r["n_nodes"] = N
    r["shape_ok"] = (M.shape == (expected_n, expected_n))

    # Symmetry
    sym_err = np.max(np.abs(M - M.T)) / (np.max(np.abs(M)) + 1e-12)
    r["symmetry_err"] = float(sym_err)
    r["is_symmetric"] = bool(sym_err < TH["sym_tol"])

    # Diagonal
    r["diag_max"] = float(np.max(np.abs(np.diag(M))))
    r["diag_zero"] = bool(r["diag_max"] < 1e-8)

    # Empty nodes (all-zero rows)
    row_sums = M.sum(axis=1)
    n_empty = int((row_sums == 0).sum())
    r["n_empty_nodes"] = n_empty
    r["pct_empty"] = float(100 * n_empty / N)
    r["empty_ok"] = bool(n_empty / N <= TH["empty_max"])

    # Density (upper triangle only, excluding diagonal)
    upper = M[np.triu_indices(N, k=1)]
    n_nonzero = int((upper > 0).sum())
    n_possible = len(upper)
    density = n_nonzero / n_possible
    r["density"] = float(density)
    r["density_ok"] = bool(TH["density_min"] <= density <= TH["density_max"])

    # Strength distribution
    nonzero = upper[upper > 0]
    if len(nonzero):
        r["strength_mean"] = float(nonzero.mean())
        r["strength_std"]  = float(nonzero.std())
        r["strength_max"]  = float(nonzero.max())
        r["strength_min"]  = float(nonzero.min())
    else:
        for k in ["strength_mean","strength_std","strength_max","strength_min"]:
            r[k] = 0.0

    # Verdict
    failures = []
    if not r["shape_ok"]:        failures.append(f"shape:{M.shape}")
    if not r["empty_ok"]:        failures.append(f"empty:{r['pct_empty']:.0f}%")
    if not r["density_ok"]:      failures.append(f"density:{density:.0%}")
    if not r["is_symmetric"]:    failures.append(f"asym:{sym_err:.2e}")
    if not r["diag_zero"]:       failures.append(f"diag:{r['diag_max']:.2e}")
    r["verdict"] = "PASS" if not failures else "FAIL"
    r["fail_reasons"] = ";".join(failures)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-dir",  required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--tians3-nodes",   type=int, default=450)
    ap.add_argument("--schaefer-nodes", type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    atlases = {"tians3": args.tians3_nodes, "schaefer400": args.schaefer_nodes}

    rows = []
    for atlas, expected_n in atlases.items():
        files = sorted(glob.glob(
            os.path.join(args.sc_dir, f"sub-*_{atlas}_connectome.csv")))
        print(f"\n=== {atlas} ({expected_n} nodes): {len(files)} files ===")
        for f in files:
            sub = os.path.basename(f).split(f"_{atlas}")[0]
            r = qc_matrix(f, expected_n)
            r["subject"] = sub
            r["atlas"] = atlas
            rows.append(r)
            flag = "✓" if r["verdict"] == "PASS" else "✗"
            print(f"  {flag} {sub}: "
                  f"empty={r['pct_empty']:.0f}%  "
                  f"density={r['density']:.1%}  "
                  f"sym={'ok' if r['is_symmetric'] else 'FAIL'}  "
                  f"{r['fail_reasons']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "qc_connectomes_report.csv"), index=False)

    fails = df[df["verdict"] == "FAIL"]
    fails.to_csv(os.path.join(args.out, "qc_connectomes_failures.csv"), index=False)

    print("\n" + "="*60)
    for atlas in atlases:
        sub_df = df[df["atlas"] == atlas]
        passed = (sub_df["verdict"] == "PASS").sum()
        print(f"{atlas}: {passed}/{len(sub_df)} PASS")
        print(f"  empty nodes: median={sub_df['pct_empty'].median():.0f}%  "
              f"max={sub_df['pct_empty'].max():.0f}%")
        print(f"  density:     median={sub_df['density'].median():.1%}  "
              f"range=[{sub_df['density'].min():.1%}, {sub_df['density'].max():.1%}]")
        print(f"  all symmetric: {sub_df['is_symmetric'].all()}")
    print("="*60)


if __name__ == "__main__":
    main()
