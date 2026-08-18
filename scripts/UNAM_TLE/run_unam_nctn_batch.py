#!/usr/bin/env python3
"""
NeuroSim UNAM_TLE — NCT batch analysis (Healthy / Left-TLE / Right-TLE)

Runs the validated physics chain across all 62 UNAM_TLE subjects using a
SHARED group-average template SC (ENIGMA Desikan-Killiany, 68 cortical
regions) rather than individual DTI — no per-subject structural prior.

    from_arrays(X, SC=template_SC) → graphnet_EC → normalise → Gramian
    → minimum_energy → finite_vs_infinite_comparison

Groups: Healthy (34), Left-TLE (17), Right-TLE (11)
[Bi-TLE / Unknown-TLE subjects did not survive QC — 0 in final cohort]

This is TLE lateralization signal-check infrastructure: lateralization asymmetry (per-node L vs R
control energy) is the quantity the signal-check actually tests, computed here and
exported for the separate statistical design (binomial CI on lateralization
accuracy + hippocampal-volume-asymmetry correlation — NOT decided in this
script, see project notes).

Brain states: same variance-based proxy as HCP/ADNI (placeholder pending
Path-C-specific state definition — see open flag in project discussion).
    x_rest   = mean of lowest-variance 20% of timepoints
    x_demand = mean of highest-synchrony 20% of timepoints

Outputs (in --out-dir):
    unam_results.csv              per-subject summary (all metrics + covariates)
    unam_group_comparison.csv     Healthy vs Left vs Right, Kruskal-Wallis
    unam_finite_vs_infinite.csv   E_fin/E_inf sweep T=1..20, per subject
    unam_teleportation.csv        FC vs EC energy divergence
    unam_node_asymmetry.csv       per-node L/R control energy asymmetry
                                  (signal-check's primary input — NOT the final
                                  statistic, just the per-subject metric)
    unam_batch_failures.csv       any errors

Usage:
    python run_unam_nctn_batch.py \
        --sc-file    /path/to/template_SC_dk68.csv \
        --ts-dir     /path/to/derivatives/timeseries \
        --meta       /path/to/participants.tsv \
        --out-dir    ./unam_results \
        [--horizon   10]
        [--limit     0]
"""

import argparse
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from neurosim.loader import from_arrays
from neurosim.connectivity import (
    graphnet_effective_connectivity,
    functional_connectivity,
)
from neurosim.physics import (
    normalise_matrix,
    compute_gramian_doubling,
    minimum_energy,
    average_controllability,
    modal_controllability,
    finite_vs_infinite_comparison,
)

N_REGIONS = 68


# ── DK-68 hemisphere split (matches ENIGMA aparc label order: LH 0-33, RH 34-67) ──
LH_IDX = list(range(0, 34))
RH_IDX = list(range(34, 68))


def define_states(X):
    """Same variance-based proxy as HCP/ADNI batches (placeholder)."""
    N, T = X.shape
    n = max(1, T // 5)
    tvar = X.var(axis=0)
    idx = np.argsort(tvar)
    x_rest = X[:, idx[:n]].mean(axis=1)
    x_demand = X[:, idx[-n:]].mean(axis=1)
    x_rest /= (np.linalg.norm(x_rest) + 1e-8)
    x_demand /= (np.linalg.norm(x_demand) + 1e-8)
    return x_rest, x_demand


def node_asymmetry(A):
    """
    Per-node control energy asymmetry: for each of the 34 homologous
    L/R region pairs, compare average controllability.
    Returns a (34,) array: positive = L > R, negative = R > L.
    This is the signal-check's raw input signal, not a lateralization decision.
    """
    ac = average_controllability(A)
    ac_L = ac[LH_IDX]
    ac_R = ac[RH_IDX]
    return ac_L - ac_R  # (34,) per-region-pair asymmetry


def analyse_subject(sub, ts_path, A_template, meta_row, horizon):
    row = {
        "subject_id": sub,
        "Group": meta_row.get("Group", ""),
        "TLEside": meta_row.get("TLEside", ""),
        "hasMTS": meta_row.get("hasMTS", np.nan),
        "age": meta_row.get("age", np.nan),
        "gender": meta_row.get("gender", ""),
        "L_hipp_vol": meta_row.get("L.hipp Volume", np.nan),
        "R_hipp_vol": meta_row.get("R.hipp Volume", np.nan),
    }

    X_raw = np.load(ts_path)  # (T, 68)
    data = from_arrays(X=X_raw.T, SC=A_template, subject_id=sub, validate=True)
    X, SC = data["X"], data["SC"]
    N, T = data["N"], data["T"]
    row.update(N=N, T=T)

    EC = graphnet_effective_connectivity(X, SC, lambda_ridge=1.0, lambda_graph=1.0)
    A = normalise_matrix(EC, target_rho=0.9)
    row["ec_asymmetry"] = float(np.linalg.norm(EC - EC.T, "fro") / np.linalg.norm(EC, "fro"))
    row["raw_rho"] = float(np.max(np.abs(np.linalg.eigvals(EC))))
    row["final_rho"] = float(np.max(np.abs(np.linalg.eigvals(A))))

    B = np.eye(N)
    W_T = compute_gramian_doubling(A, B, horizon)
    eig = np.linalg.eigvalsh(W_T)
    row["gramian_min_eig"] = float(np.min(eig))
    row["gramian_cond"] = float(np.linalg.cond(W_T))

    x_rest, x_demand = define_states(X)
    E_r2d, _ = minimum_energy(A, B, x_rest, x_demand, T=horizon)
    E_d2r, _ = minimum_energy(A, B, x_demand, x_rest, T=horizon)
    row["E_rest_to_demand"] = float(E_r2d)
    row["E_demand_to_rest"] = float(E_d2r)
    row["E_asymmetry"] = float(E_r2d - E_d2r)

    FC = functional_connectivity(X)
    A_fc = normalise_matrix(FC, target_rho=0.9)
    E_fc, _ = minimum_energy(A_fc, B, x_rest, x_demand, T=horizon)
    row["E_fc"] = float(E_fc)
    row["E_ec"] = float(E_r2d)
    row["teleport_ratio"] = float(E_fc / (E_r2d + 1e-12))

    ac = average_controllability(A)
    mc = modal_controllability(A)
    row["avg_ctrl_mean"] = float(np.mean(ac))
    row["mod_ctrl_mean"] = float(np.mean(mc))

    fvi = finite_vs_infinite_comparison(A, B, x_rest, x_demand, T_range=list(range(1, 21)))
    fvi.insert(0, "subject_id", sub)
    fvi.insert(1, "Group", row["Group"])
    fvi.insert(2, "TLEside", row["TLEside"])

    asym = node_asymmetry(A)  # (34,) per-region-pair L-R asymmetry
    asym_row = {"subject_id": sub, "Group": row["Group"], "TLEside": row["TLEside"]}
    for i, v in enumerate(asym):
        asym_row[f"pair_{i:02d}_LminusR"] = float(v)

    return row, fvi, asym_row


def group_stats_3way(df, metric):
    groups = {g: df[df["Group_detail"] == g][metric].dropna() for g in
              df["Group_detail"].unique()}
    valid = {g: v for g, v in groups.items() if len(v) >= 3}
    if len(valid) < 2:
        return None
    H, p = stats.kruskal(*valid.values())
    result = {"metric": metric, "kruskal_H": float(H), "p_value": float(p)}
    for g, v in valid.items():
        result[f"{g}_n"] = len(v)
        result[f"{g}_median"] = float(v.median())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-file", required=True, help="template_SC_dk68.csv")
    ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--meta", required=True, help="participants.tsv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    A_template = np.loadtxt(args.sc_file, delimiter=",")
    print(f"Template SC loaded: {A_template.shape}")
    assert A_template.shape == (N_REGIONS, N_REGIONS), \
        f"Expected ({N_REGIONS},{N_REGIONS}), got {A_template.shape}"

    meta_df = pd.read_csv(args.meta, sep="\t")
    meta_df["_sub"] = meta_df["participant_id"].str.replace("sub-", "", regex=False)
    meta_dict = {r["_sub"]: r.to_dict() for _, r in meta_df.iterrows()}

    ts_files = sorted(f for f in os.listdir(args.ts_dir) if f.endswith("_dk68_native_timeseries.npy"))
    subjects = [f.split("_")[0].replace("sub-", "") for f in ts_files]
    if args.limit:
        subjects = subjects[:args.limit]
    print(f"Subjects to process: {len(subjects)}")

    results, fvi_all, asym_all, failures = [], [], [], []
    t0 = time.time()

    for i, sub in enumerate(subjects, 1):
        ts_path = os.path.join(args.ts_dir, f"sub-{sub}_dk68_native_timeseries.npy")
        meta_row = meta_dict.get(sub, {})
        try:
            row, fvi, asym_row = analyse_subject(sub, ts_path, A_template, meta_row, args.horizon)
            # Build 3-way group label: Healthy / Left-TLE / Right-TLE
            side = row["TLEside"]
            row["Group_detail"] = "Healthy" if row["Group"] == "Healthy" else side
            fvi["Group_detail"] = row["Group_detail"]
            asym_row["Group_detail"] = row["Group_detail"]

            results.append(row)
            fvi_all.append(fvi)
            asym_all.append(asym_row)
            print(f"[{i:2d}/{len(subjects)}] sub-{sub} ({row['Group_detail']:10s}) "
                  f"E_r2d={row['E_rest_to_demand']:.4f}  teleport={row['teleport_ratio']:.2f}x")
        except Exception as e:
            failures.append({"subject_id": sub, "error": str(e),
                             "trace": traceback.format_exc()[-400:]})
            print(f"[{i:2d}/{len(subjects)}] sub-{sub} FAILED: {e}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(args.out_dir, "unam_results.csv"), index=False)

    if fvi_all:
        pd.concat(fvi_all, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "unam_finite_vs_infinite.csv"), index=False)

    if asym_all:
        pd.DataFrame(asym_all).to_csv(
            os.path.join(args.out_dir, "unam_node_asymmetry.csv"), index=False)

    if not res_df.empty:
        comp_rows = []
        for metric in ["E_rest_to_demand", "E_demand_to_rest", "E_asymmetry",
                        "teleport_ratio", "avg_ctrl_mean", "mod_ctrl_mean"]:
            s = group_stats_3way(res_df, metric)
            if s:
                comp_rows.append(s)
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(args.out_dir, "unam_group_comparison.csv"), index=False)

        res_df[["subject_id", "Group_detail", "E_fc", "E_ec", "teleport_ratio"]].to_csv(
            os.path.join(args.out_dir, "unam_teleportation.csv"), index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(args.out_dir, "unam_batch_failures.csv"), index=False)

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)}/{len(subjects)} subjects, {len(failures)} failed, {dt/60:.1f} min")
    if not res_df.empty:
        for grp in res_df["Group_detail"].unique():
            sub_df = res_df[res_df["Group_detail"] == grp]
            m = sub_df["E_rest_to_demand"]
            print(f"  {grp:10s} (N={len(sub_df)}): E_rest→demand median={m.median():.4f}")
        tr = res_df["teleport_ratio"]
        print(f"\n  Teleportation: median={tr.median():.2f}x  100%>1={100*(tr>1).mean():.0f}%")
    print(f"{'='*60}")
    print("\nNOTE: unam_node_asymmetry.csv is the signal-check's raw per-subject input.")
    print("The actual lateralization statistic (binomial CI on classification")
    print("accuracy vs TLEside, + hippocampal-volume-asymmetry correlation)")
    print("is a SEPARATE analysis step — not run here. See project notes.")


if __name__ == "__main__":
    main()