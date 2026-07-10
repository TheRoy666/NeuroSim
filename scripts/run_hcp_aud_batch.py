"""
NeuroSim — HCP S1200 AUD cohort batch analysis.

Cohort: 238 subjects (social drinkers + DSM-IV AUD cases, twin design)
  Severity=0: Social drinker (control)
  Severity=1: DSM-IV Alcohol Abuser
  Severity=2: DSM-IV Alcohol Dependent

Discordant MZ pairs:
  ~20 pairs: Severity=0 vs Severity=1 or 2 (control vs any AUD)
  ~1  pair:  Severity=1 vs Severity=2 (abuser vs dependent)

Scientific note: Prior analysis found no significant population-level
group difference (underpowered subcohorts, age distribution). This
pipeline is demonstrated exploratorily. The within-pair discordant
design is the strongest available causal estimate on this cohort.

Outputs (all in --out-dir):
  aud_cohort_results.csv        per-subject summary
  aud_group_comparison.csv      median E* per severity group
  aud_discordant_pairs.csv      within-pair energy differences
  aud_finite_vs_infinite.csv    E_fin/E_inf sweep T=1..20
  aud_teleportation.csv         FC vs EC energy per subject
  aud_batch_failures.csv        errors

Usage:
  python run_hcp_aud_batch.py \
      --ts-dir  /path/to/timeseries \
      --sc-dir  /path/to/SC_matrices \
      --meta    /path/to/MASTER_ROI_METRICS_DTI_FBA.csv \
      --out-dir ./hcp_aud_results \
      [--ts-suffix _native_timeseries.npy] \
      [--sc-suffix _SC_SIFT2_410.csv] \
      [--target-rho 0.9] [--horizon 10] [--limit 0]
"""

import argparse
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from neurosim.loader import load_connectome, from_arrays
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

# Severity labels — graded AUD scale (DSM-IV)
SEVERITY_LABEL = {0: "Social_Drinker", 1: "Abuse", 2: "Dependence"}


#  File discovery
def discover_subjects(ts_dir, sc_dir, ts_suffix, sc_suffix):
    ts = {f[:-len(ts_suffix)]: os.path.join(ts_dir, f)
          for f in os.listdir(ts_dir) if f.endswith(ts_suffix)}
    sc = {f[:-len(sc_suffix)]: os.path.join(sc_dir, f)
          for f in os.listdir(sc_dir) if f.endswith(sc_suffix)}
    return sorted(set(ts) & set(sc)), ts, sc


#  Brain state extraction from BOLD 
def define_states(X):
    """
    Approximate rest and craving states from BOLD timeseries.
    Rest    = mean across lowest-variance 20% of timepoints (quietest)
    Craving = mean across highest-synchrony 20% of timepoints (most active)
    Both unit-normalised per NCT convention.
    """
    N, T = X.shape
    n = max(1, T // 5)
    tvar = X.var(axis=0)
    idx  = np.argsort(tvar)
    x_rest    = X[:, idx[:n]].mean(axis=1)
    x_craving = X[:, idx[-n:]].mean(axis=1)
    x_rest    /= (np.linalg.norm(x_rest)    + 1e-8)
    x_craving /= (np.linalg.norm(x_craving) + 1e-8)
    return x_rest, x_craving


#  Per-subject analysis
def analyse_subject(sid, ts_path, sc_path, meta_row,
                    target_rho, horizon):
    severity = int(meta_row.get("Severity", 0))
    row = {
        "subject_id":      sid,
        "TwinPairID":      meta_row.get("TwinPairID", ""),
        "ZygosityGT1":     meta_row.get("ZygosityGT1", ""),
        "Severity":        severity,
        "Severity_Label":  SEVERITY_LABEL.get(severity, f"Sev{severity}"),
        "Age":             meta_row.get("Age_in_Yrs", np.nan),
        "Sex":             int(meta_row.get("Gender", -1)),  # 0=Female, 1=Male
    }

    # Load + validate
    X_raw  = np.load(ts_path)
    SC_raw = load_connectome(sc_path)
    data   = from_arrays(X=X_raw.T, SC=SC_raw,
                         subject_id=sid, validate=True)
    X, SC  = data["X"], data["SC"]
    N      = data["N"]

    # EC + normalise
    EC = graphnet_effective_connectivity(X, SC,
                                         lambda_ridge=1.0, lambda_graph=1.0)
    A  = normalise_matrix(EC, target_rho=target_rho)
    row["ec_asymmetry"] = float(
        np.linalg.norm(EC - EC.T, "fro") / np.linalg.norm(EC, "fro"))
    row["raw_rho"]      = float(np.max(np.abs(np.linalg.eigvals(EC))))
    row["final_rho"]    = float(np.max(np.abs(np.linalg.eigvals(A))))

    # Gramian (B=I)
    B   = np.eye(N)
    W_T = compute_gramian_doubling(A, B, horizon)
    eig = np.linalg.eigvalsh(W_T)
    row["gramian_min_eig"] = float(np.min(eig))
    row["gramian_cond"]    = float(np.linalg.cond(W_T))

    # AUD brain states
    x_rest, x_craving = define_states(X)

    # Transition energies
    E_crav2rest, _ = minimum_energy(A, B, x_craving, x_rest,    T=horizon)
    E_rest2crav, _ = minimum_energy(A, B, x_rest,    x_craving, T=horizon)
    row["E_craving_to_rest"] = float(E_crav2rest)
    row["E_rest_to_craving"] = float(E_rest2crav)
    row["E_asymmetry"]       = float(E_crav2rest - E_rest2crav)

    # Teleportation Error (FC vs EC on same transition)
    FC   = functional_connectivity(X)
    A_fc = normalise_matrix(FC, target_rho=target_rho)
    E_fc, _ = minimum_energy(A_fc, B, x_craving, x_rest, T=horizon)
    row["E_fc"]           = float(E_fc)
    row["E_ec"]           = float(E_crav2rest)
    row["teleport_ratio"] = float(E_fc / (E_crav2rest + 1e-12))

    # Controllability profiles
    row["avg_ctrl_mean"] = float(np.mean(average_controllability(A)))
    row["mod_ctrl_mean"] = float(np.mean(modal_controllability(A)))

    # Finite-vs-infinite sweep
    fvi = finite_vs_infinite_comparison(
        A, B, x_craving, x_rest, T_range=list(range(1, 21)))
    fvi.insert(0, "subject_id", sid)
    fvi.insert(1, "Severity", severity)
    fvi.insert(2, "Severity_Label", SEVERITY_LABEL.get(severity, ""))

    return row, fvi


#  Discordant MZ pair analysis 
def compute_discordant_pairs(res_df):
    """
    Identify ALL discordant MZ pairs regardless of severity contrast:
      Type A: Severity=0 vs Severity=1 (social drinker vs abuser)
      Type B: Severity=0 vs Severity=2 (social drinker vs dependent)
      Type C: Severity=1 vs Severity=2 (abuser vs dependent)

    Within-pair delta = higher-severity minus lower-severity member.
    Positive dE_craving → higher-severity member needs more energy
    to transition from craving to rest.
    """
    mz = res_df[res_df["ZygosityGT1"] == "MZ"]
    rows = []
    for pair_id, grp in mz.groupby("TwinPairID"):
        if len(grp) != 2:
            continue
        s = sorted(grp["Severity"].values)
        if s[0] == s[1]:
            continue   # concordant — skip
        lo = grp[grp["Severity"] == s[0]].iloc[0]
        hi = grp[grp["Severity"] == s[1]].iloc[0]
        contrast = f"Sev{s[0]}_vs_Sev{s[1]}"
        rows.append({
            "TwinPairID":        pair_id,
            "contrast":          contrast,
            "lo_severity":       s[0],
            "hi_severity":       s[1],
            "lo_subject":        lo["subject_id"],
            "hi_subject":        hi["subject_id"],
            "dE_craving_to_rest": float(
                hi["E_craving_to_rest"] - lo["E_craving_to_rest"]),
            "dE_asymmetry":      float(
                hi["E_asymmetry"]       - lo["E_asymmetry"]),
            "d_teleport_ratio":  float(
                hi["teleport_ratio"]    - lo["teleport_ratio"]),
            "d_avg_ctrl":        float(
                hi["avg_ctrl_mean"]     - lo["avg_ctrl_mean"]),
        })
    return pd.DataFrame(rows)


#  Group summary 
def compute_group_summary(res_df):
    """Median + IQR per severity group for key NCT metrics."""
    metrics = ["E_craving_to_rest", "E_rest_to_craving",
               "E_asymmetry", "teleport_ratio",
               "avg_ctrl_mean", "mod_ctrl_mean"]
    rows = []
    for sev, label in SEVERITY_LABEL.items():
        sub = res_df[res_df["Severity"] == sev]
        if sub.empty:
            continue
        r = {"Severity": sev, "Severity_Label": label, "N": len(sub)}
        for m in metrics:
            r[f"{m}_median"] = float(sub[m].median())
            r[f"{m}_q25"]    = float(sub[m].quantile(.25))
            r[f"{m}_q75"]    = float(sub[m].quantile(.75))
        rows.append(r)
    return pd.DataFrame(rows)


#  Main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-dir",     required=True)
    ap.add_argument("--sc-dir",     required=True)
    ap.add_argument("--meta",       required=True)
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--ts-suffix",  default="_native_timeseries.npy")
    ap.add_argument("--sc-suffix",  default="_SC_SIFT2_410.csv")
    ap.add_argument("--target-rho", type=float, default=0.9)
    ap.add_argument("--horizon",    type=int,   default=10)
    ap.add_argument("--limit",      type=int,   default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta      = pd.read_csv(args.meta)
    meta["Subject"] = meta["Subject"].astype(str)
    meta_dict = meta.set_index("Subject").to_dict(orient="index")

    subjects, ts_files, sc_files = discover_subjects(
        args.ts_dir, args.sc_dir, args.ts_suffix, args.sc_suffix)
    subjects = [s for s in subjects if s in meta_dict]
    if args.limit:
        subjects = subjects[:args.limit]

    # Cohort breakdown
    print(f"Subjects with files + metadata: {len(subjects)}")
    for sev, label in SEVERITY_LABEL.items():
        n = sum(1 for s in subjects
                if int(meta_dict[s].get("Severity", -1)) == sev)
        print(f"  Severity={sev} ({label}): {n}")

    results, fvi_all, failures = [], [], []
    t0 = time.time()

    for i, sid in enumerate(subjects, 1):
        try:
            row, fvi = analyse_subject(
                sid, ts_files[sid], sc_files[sid],
                meta_dict[sid], args.target_rho, args.horizon)
            results.append(row)
            fvi_all.append(fvi)
            print(f"[{i:3d}/{len(subjects)}] {sid} "
                  f"Sev={row['Severity']} ({row['Severity_Label']:14s}) "
                  f"E_crav→rest={row['E_craving_to_rest']:.4f}  "
                  f"teleport={row['teleport_ratio']:.2f}x")
        except Exception as e:
            failures.append({"subject_id": sid,
                             "error": str(e),
                             "trace": traceback.format_exc()[-500:]})
            print(f"[{i:3d}/{len(subjects)}] {sid}  FAILED: {e}")

    #  Save 
    res_df = pd.DataFrame(results)
    res_df.to_csv(
        os.path.join(args.out_dir, "aud_cohort_results.csv"), index=False)

    if not res_df.empty:
        compute_group_summary(res_df).to_csv(
            os.path.join(args.out_dir, "aud_group_comparison.csv"),
            index=False)

        compute_discordant_pairs(res_df).to_csv(
            os.path.join(args.out_dir, "aud_discordant_pairs.csv"),
            index=False)

        res_df[["subject_id", "Severity_Label",
                "E_fc", "E_ec", "teleport_ratio"]].to_csv(
            os.path.join(args.out_dir, "aud_teleportation.csv"), index=False)

    if fvi_all:
        pd.concat(fvi_all, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "aud_finite_vs_infinite.csv"),
            index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(args.out_dir, "aud_batch_failures.csv"), index=False)

    #  Summary
    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"DONE: {len(results)}/{len(subjects)} subjects, "
          f"{len(failures)} failed, {dt/60:.1f} min")

    if not res_df.empty:
        print("\nGroup medians (E_craving_to_rest):")
        for sev, label in SEVERITY_LABEL.items():
            sub = res_df[res_df["Severity"] == sev]
            if len(sub):
                print(f"  {label:16s} (N={len(sub):3d}): "
                      f"median={sub['E_craving_to_rest'].median():.4f}  "
                      f"IQR=[{sub['E_craving_to_rest'].quantile(.25):.4f}, "
                      f"{sub['E_craving_to_rest'].quantile(.75):.4f}]")

        disc = compute_discordant_pairs(res_df)
        if not disc.empty:
            print(f"\nDiscordant MZ pairs (N={len(disc)}):")
            for contrast, grp in disc.groupby("contrast"):
                sign_consistent = (grp["dE_craving_to_rest"] > 0).sum()
                print(f"  {contrast}: N={len(grp)}, "
                      f"mean_dE={grp['dE_craving_to_rest'].mean():.4f}, "
                      f"positive: {sign_consistent}/{len(grp)}")
            print("\n  NOTE: N is small — treat as exploratory.")

        print(f"\nTeleportation ratio (EC vs FC, craving→rest):")
        print(f"  overall median={res_df['teleport_ratio'].median():.2f}x  "
              f"IQR=[{res_df['teleport_ratio'].quantile(.25):.2f}, "
              f"{res_df['teleport_ratio'].quantile(.75):.2f}]")

    print("=" * 60)


if __name__ == "__main__":
    main()
