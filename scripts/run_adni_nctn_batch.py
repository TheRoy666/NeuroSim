"""
NeuroSim ADNI — NCT batch analysis (CN vs MCI, two atlases)

Runs the validated physics chain across all 49 ADNI subjects:
    from_arrays → graphnet_EC → normalise → Gramian
    → minimum_energy → finite_vs_infinite_comparison

Two atlases in parallel:
    tians3     (450 nodes: Schaefer-400 + Tian S3 subcortical)
    schaefer400 (400 nodes: cortex only — cross-parcellation check)

Brain states (ADNI-specific):
    x_rest   = mean of lowest-variance 20% of timepoints
    x_demand = mean of highest-synchrony 20% of timepoints
              (proxy for DMN activation / default mode demand)

Primary comparison: E_rest_to_demand CN vs MCI
Midterm anchor figure: finite_vs_infinite_comparison on real ADNI subjects

Outputs (in --out-dir):
    adni_results_<atlas>.csv          per-subject summary
    adni_group_comparison_<atlas>.csv median ± IQR per group
    adni_finite_vs_infinite_<atlas>.csv E_fin/E_inf sweep T=1..20
    adni_teleportation_<atlas>.csv    FC vs EC divergence
    adni_batch_failures.csv           any errors

Usage:
    python run_adni_nctn_batch.py \
        --sc-dir    /path/to/connectomes \
        --ts-dir    /path/to/timeseries \
        --meta      /path/to/participants_qc.tsv \
        --extract   /path/to/extraction_report.csv \
        --out-dir   ./adni_results \
        [--atlas    tians3]     # run one or both; default: both
        [--horizon  10]
        [--limit    0]
"""

import argparse
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from neurosim.loader    import load_connectome, from_arrays
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

#  Atlas configuration 
ATLAS_CONFIG = {
    "tians3":      {"n_nodes": 450, "ts_suffix": "_tians3_native_timeseries.npy",
                    "sc_suffix": "_tians3_connectome.csv"},
    "schaefer400": {"n_nodes": 400, "ts_suffix": "_schaefer400_native_timeseries.npy",
                    "sc_suffix": "_schaefer400_connectome.csv"},
}


#  Brain state extraction
def define_states(X):
    """
    Rest   = mean of lowest-variance 20% of timepoints (quiet resting)
    Demand = mean of highest-synchrony 20% of timepoints (DMN activation)
    Both unit-normalised per NCT convention.
    """
    N, T = X.shape
    n = max(1, T // 5)
    tvar = X.var(axis=0)
    idx  = np.argsort(tvar)
    x_rest   = X[:, idx[:n]].mean(axis=1)
    x_demand = X[:, idx[-n:]].mean(axis=1)
    x_rest   /= (np.linalg.norm(x_rest)   + 1e-8)
    x_demand /= (np.linalg.norm(x_demand) + 1e-8)
    return x_rest, x_demand


#  Per-subject analysis ─
def analyse_subject(sub, ts_path, sc_path, meta_row, horizon):
    row = {
        "subject_id":    sub,
        "DX_Group":      meta_row.get("DX_Group", ""),
        "DX_Subtype":    meta_row.get("DX_Subtype", ""),
        "age":           meta_row.get("age", np.nan),
        "sex":           meta_row.get("sex", ""),
        "manufacturer":  meta_row.get("manufacturer", ""),
        "scanner_model": meta_row.get("scanner_model", ""),
        "site":          meta_row.get("site", ""),
        "T1_NonMPRAGE":  meta_row.get("T1_NonMPRAGE", ""),
    }

    # Load + validate
    X_raw  = np.load(ts_path)              # (T, N) from nilearn masker
    SC_raw = load_connectome(sc_path)
    data   = from_arrays(X=X_raw.T, SC=SC_raw,
                          subject_id=sub, validate=True)
    X, SC  = data["X"], data["SC"]
    N, T   = data["N"], data["T"]
    row.update(N=N, T=T)

    # EC + normalise
    EC = graphnet_effective_connectivity(X, SC,
                                          lambda_ridge=1.0, lambda_graph=1.0)
    A  = normalise_matrix(EC, target_rho=0.9)
    row["ec_asymmetry"] = float(
        np.linalg.norm(EC - EC.T, "fro") / np.linalg.norm(EC, "fro"))
    row["raw_rho"]   = float(np.max(np.abs(np.linalg.eigvals(EC))))
    row["final_rho"] = float(np.max(np.abs(np.linalg.eigvals(A))))

    # Gramian (B = I, full-rank)
    B   = np.eye(N)
    W_T = compute_gramian_doubling(A, B, horizon)
    eig = np.linalg.eigvalsh(W_T)
    row["gramian_min_eig"] = float(np.min(eig))
    row["gramian_cond"]    = float(np.linalg.cond(W_T))

    # Brain states
    x_rest, x_demand = define_states(X)

    # Transition energies
    E_r2d, _ = minimum_energy(A, B, x_rest,   x_demand, T=horizon)
    E_d2r, _ = minimum_energy(A, B, x_demand, x_rest,   T=horizon)
    row["E_rest_to_demand"] = float(E_r2d)
    row["E_demand_to_rest"] = float(E_d2r)
    row["E_asymmetry"]      = float(E_r2d - E_d2r)

    # Teleportation Error (FC vs EC on same transition)
    FC   = functional_connectivity(X)
    A_fc = normalise_matrix(FC, target_rho=0.9)
    E_fc, _ = minimum_energy(A_fc, B, x_rest, x_demand, T=horizon)
    row["E_fc"]           = float(E_fc)
    row["E_ec"]           = float(E_r2d)
    row["teleport_ratio"] = float(E_fc / (E_r2d + 1e-12))

    # Controllability profiles
    ac = average_controllability(A)
    mc = modal_controllability(A)
    row["avg_ctrl_mean"] = float(np.mean(ac))
    row["mod_ctrl_mean"] = float(np.mean(mc))

    # Finite-vs-infinite sweep (THE MIDTERM ANCHOR FIGURE)
    fvi = finite_vs_infinite_comparison(
        A, B, x_rest, x_demand, T_range=list(range(1, 21)))
    fvi.insert(0, "subject_id", sub)
    fvi.insert(1, "DX_Group",   row["DX_Group"])

    return row, fvi


#  Group comparison
def group_stats(res_df, metric):
    cn  = res_df[res_df["DX_Group"] == "CN"][metric].dropna()
    mci = res_df[res_df["DX_Group"] == "MCI"][metric].dropna()
    if len(cn) < 3 or len(mci) < 3:
        return {}
    U, p = stats.mannwhitneyu(cn, mci, alternative="two-sided")
    # Cliff's delta
    gt = sum(x > y for x in cn for y in mci)
    lt = sum(x < y for x in cn for y in mci)
    d  = (gt - lt) / (len(cn) * len(mci))
    return {
        "metric":     metric,
        "CN_n":       len(cn),  "CN_median":  float(cn.median()),
        "CN_q25":     float(cn.quantile(.25)), "CN_q75": float(cn.quantile(.75)),
        "MCI_n":      len(mci), "MCI_median": float(mci.median()),
        "MCI_q25":    float(mci.quantile(.25)),"MCI_q75": float(mci.quantile(.75)),
        "MannWhitney_U": float(U), "p_value": float(p),
        "cliffs_delta":  float(d),
    }


#  Run one atlas 
def run_atlas(atlas, subjects, meta_dict, extract_dict,
              sc_dir, ts_dir, out_dir, horizon, limit):
    cfg = ATLAS_CONFIG[atlas]
    print(f"\n{'='*60}")
    print(f"ATLAS: {atlas} ({cfg['n_nodes']} nodes)")
    print(f"{'='*60}")

    results, fvi_all, failures = [], [], []
    t0 = time.time()

    subs = subjects[:limit] if limit else subjects

    for i, sub in enumerate(subs, 1):
        ts_path = os.path.join(ts_dir, "sub-" + sub + cfg["ts_suffix"])
        sc_path = os.path.join(sc_dir, "sub-" + sub + cfg["sc_suffix"])

        if not os.path.exists(ts_path):
            failures.append({"subject_id": sub, "error": f"no timeseries: {ts_path}"})
            print(f"[{i:2d}/{len(subs)}] {sub}  SKIP (no timeseries)")
            continue
        if not os.path.exists(sc_path):
            failures.append({"subject_id": sub, "error": f"no SC: {sc_path}"})
            print(f"[{i:2d}/{len(subs)}] {sub}  SKIP (no SC matrix)")
            continue

        try:
            meta = meta_dict.get(sub, {})
            # Add n_censored from extraction report as covariate
            meta["n_censored"] = extract_dict.get(sub, {}).get("n_censored", np.nan)
            row, fvi = analyse_subject(sub, ts_path, sc_path, meta, horizon)
            results.append(row)
            fvi_all.append(fvi)
            print(f"[{i:2d}/{len(subs)}] {sub} ({row['DX_Group']:3s}) "
                  f"E_r2d={row['E_rest_to_demand']:.4f}  "
                  f"teleport={row['teleport_ratio']:.2f}x  "
                  f"fvi_T1={fvi[fvi['T']==1]['ratio'].values[0]:.2f}x")
        except Exception as e:
            failures.append({"subject_id": sub, "error": str(e),
                             "trace": traceback.format_exc()[-400:]})
            print(f"[{i:2d}/{len(subs)}] {sub}  FAILED: {e}")

    #  Save results
    tag = atlas
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, f"adni_results_{tag}.csv"), index=False)

    if fvi_all:
        pd.concat(fvi_all, ignore_index=True).to_csv(
            os.path.join(out_dir, f"adni_finite_vs_infinite_{tag}.csv"), index=False)

    if not res_df.empty:
        # Group comparison — the clinical result
        comp_rows = []
        for metric in ["E_rest_to_demand", "E_demand_to_rest",
                        "E_asymmetry", "teleport_ratio",
                        "avg_ctrl_mean", "mod_ctrl_mean"]:
            s = group_stats(res_df, metric)
            if s:
                comp_rows.append(s)
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(out_dir, f"adni_group_comparison_{tag}.csv"), index=False)

        # Teleportation distribution
        res_df[["subject_id", "DX_Group", "E_fc", "E_ec", "teleport_ratio"]].to_csv(
            os.path.join(out_dir, f"adni_teleportation_{tag}.csv"), index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(out_dir, "adni_batch_failures.csv"), index=False)

    #  Summary 
    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE [{atlas}]: {len(results)}/{len(subs)} subjects, "
          f"{len(failures)} failed, {dt/60:.1f} min")

    if not res_df.empty:
        for grp in ["CN", "MCI"]:
            sub_df = res_df[res_df["DX_Group"] == grp]
            if len(sub_df):
                m = sub_df["E_rest_to_demand"]
                print(f"  {grp} (N={len(sub_df)}): "
                      f"E_rest→demand median={m.median():.4f}  "
                      f"IQR=[{m.quantile(.25):.4f}, {m.quantile(.75):.4f}]")

        # Primary clinical result
        s = group_stats(res_df, "E_rest_to_demand")
        if s:
            print(f"\n  CN vs MCI (E_rest→demand):")
            print(f"    Mann-Whitney p={s['p_value']:.3f}  "
                  f"Cliff's δ={s['cliffs_delta']:.3f}")

        # Finite-vs-infinite headline
        if fvi_all:
            fvi_all_df = pd.concat(fvi_all, ignore_index=True)
            t1_ratio = fvi_all_df[fvi_all_df["T"] == 1]["ratio"]
            print(f"\n  Finite-vs-infinite at T=1: "
                  f"median={t1_ratio.median():.2f}x  "
                  f"IQR=[{t1_ratio.quantile(.25):.2f}, {t1_ratio.quantile(.75):.2f}]")

        # Teleportation
        tr = res_df["teleport_ratio"]
        print(f"  Teleportation: median={tr.median():.2f}x  "
              f"100%>{1} ({(tr>1).mean():.0%})")

    print(f"{'='*60}")
    return res_df


#  Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc-dir",   required=True)
    ap.add_argument("--ts-dir",   required=True)
    ap.add_argument("--meta",     required=True, help="participants_qc.tsv")
    ap.add_argument("--extract",  default=None,  help="extraction_report.csv")
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--atlas",    default="both",
                    choices=["tians3","schaefer400","both"])
    ap.add_argument("--horizon",  type=int, default=10)
    ap.add_argument("--limit",    type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load metadata
    meta_df = pd.read_csv(args.meta, sep="\t")
    id_col  = "participant_id" if "participant_id" in meta_df.columns \
              else meta_df.columns[0]
    # Strip sub- prefix to match file naming
    meta_df["_sub"] = meta_df[id_col].str.replace("sub-", "", regex=False)
    meta_dict = {r["_sub"]: r.to_dict() for _, r in meta_df.iterrows()}

    # Load censoring covariate
    extract_dict = {}
    if args.extract and os.path.exists(args.extract):
        ex = pd.read_csv(args.extract)
        for _, r in ex.iterrows():
            sub = str(r.get("subject", r.get("participant_id", ""))).replace("sub-","")
            extract_dict[sub] = {"n_censored": r.get("n_censored", np.nan)}

    # QC-passing subjects from metadata
    if "QC_Pass" in meta_df.columns:
        pass_df = meta_df[meta_df["QC_Pass"] == "PASS"]
    else:
        pass_df = meta_df
    subjects = sorted(pass_df["_sub"].tolist())
    print(f"Subjects to process: {len(subjects)}")
    print(f"Horizon T={args.horizon}, Atlas={args.atlas}")

    # Run
    atlases = (["tians3", "schaefer400"] if args.atlas == "both"
               else [args.atlas])

    for atlas in atlases:
        run_atlas(atlas, subjects, meta_dict, extract_dict,
                  args.sc_dir, args.ts_dir, args.out_dir,
                  args.horizon, args.limit)

    print(f"\nAll results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
