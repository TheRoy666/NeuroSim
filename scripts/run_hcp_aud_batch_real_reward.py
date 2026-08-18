"""
NeuroSim -- HCP S1200 AUD cohort batch analysis, REAL REWARD-NETWORK
VERSION (Phase 7 sensitivity check).

Modifies ONLY the craving-state definition from run_hcp_aud_batch.py's
original cross-regional-variance proxy to real reward-network signal
(NAc, caudate, putamen, OFC, vmPFC), using the atlas index mapping
confirmed via SCMatrixExtractor.py's own docstring/merge_parcellation
code AND independently verified empirically against a real subject's SC
matrix (NAc-core-rh's top connections matched textbook basal ganglia
reward circuitry almost exactly, including landing on OFC and two of the
three proposed vmPFC regions).

Deliberately keeps REST defined EXACTLY as the original script does (low
cross-regional variance) -- mirrors AUDPipeline's own already-tested
design (only craving depends on reward_indices, rest stays a general
baseline) and isolates the one variable this check is actually testing.
Everything else (EC estimation via GraphNet, Gramian computation, energy
calculations) is byte-for-byte identical to the original script, so any
difference in results can be cleanly attributed to the state definition
alone, not a confound from changing multiple things simultaneously.

Run this ALONGSIDE the original script (not instead of it) on the same
subjects -- a sensitivity check, not a silent replacement, per the same
discipline already used for UNAM's pre-registered-primary +
exploratory-secondary structure.
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

SEVERITY_LABEL = {0: "Social_Drinker", 1: "Abuser", 2: "Dependent"}

# Confirmed reward-network indices (0-indexed, HCP 410-region atlas:
# TianS3 subcortical 1-50, HCP-MMP RH cortical 51-230, LH cortical
# 231-410). Confirmed via SCMatrixExtractor.py's docstring/merge code
# AND empirically against a real subject's SC matrix.
REWARD_INDICES_DEFAULT = {
    "NAc":     [21, 22, 46, 47],
    "Caudate": [14, 15, 16, 17, 39, 40, 41, 42],
    "Putamen": [10, 11, 12, 13, 35, 36, 37, 38],
    "OFC":     [142, 322],
    # vmPFC (10r/10v/s32/p32/a24/25) -- a real methodological choice,
    # not a data-derived fact like the others. Flagged for explicit
    # sign-off, not silently assumed.
    "vmPFC":   [114, 137, 214, 113, 110, 213, 294, 317, 394, 293, 290, 393],
}


def flatten_reward_indices(reward_dict):
    return sorted({i for indices in reward_dict.values() for i in indices})


#  File discovery (identical to original)
def discover_subjects(ts_dir, sc_dir, ts_suffix, sc_suffix):
    ts = {f[:-len(ts_suffix)]: os.path.join(ts_dir, f)
          for f in os.listdir(ts_dir) if f.endswith(ts_suffix)}
    sc = {f[:-len(sc_suffix)]: os.path.join(sc_dir, f)
          for f in os.listdir(sc_dir) if f.endswith(sc_suffix)}
    return sorted(set(ts) & set(sc)), ts, sc


#  Brain state extraction -- REAL REWARD-NETWORK VERSION
def define_states_real_reward(X, reward_indices):
    """
    Rest    = mean across the 20% of timepoints with LOWEST cross-regional
              variance -- UNCHANGED from the original script's definition.

    Craving = mean across the 20% of timepoints with HIGHEST mean
              activity in the real reward network (NAc, caudate, putamen,
              OFC, vmPFC) -- REPLACES the original's high-cross-regional-
              variance proxy with a real, anatomically-grounded signal.

    Both states are unit-normalised per NCT convention, matching the
    original.
    """
    N, T = X.shape
    n = max(1, T // 5)

    # Rest: unchanged, low cross-regional variance
    tvar = X.var(axis=0)
    rest_idx = np.argsort(tvar)[:n]
    x_rest = X[:, rest_idx].mean(axis=1)

    # Craving: real reward-network signal, top 20% by activity
    reward_signal = X[reward_indices, :].mean(axis=0)
    craving_idx = np.argsort(reward_signal)[-n:]
    x_craving = X[:, craving_idx].mean(axis=1)

    x_rest    /= (np.linalg.norm(x_rest)    + 1e-8)
    x_craving /= (np.linalg.norm(x_craving) + 1e-8)
    return x_rest, x_craving


#  Per-subject analysis (identical to original except the define_states call)
def analyse_subject(sid, ts_path, sc_path, meta_row,
                    target_rho, horizon, reward_indices):
    severity = int(meta_row.get("Severity", 0))
    row = {
        "subject_id":      sid,
        "TwinPairID":      meta_row.get("TwinPairID", ""),
        "ZygosityGT1":     meta_row.get("ZygosityGT1", ""),
        "Severity":        severity,
        "Severity_Label":  SEVERITY_LABEL.get(severity, f"Sev{severity}"),
        "Age":             meta_row.get("Age_in_Yrs", np.nan),
        "Sex":             int(meta_row.get("Gender", -1)),
    }

    X_raw  = np.load(ts_path)
    SC_raw = load_connectome(sc_path)
    data   = from_arrays(X=X_raw.T, SC=SC_raw,
                         subject_id=sid, validate=True)
    X, SC  = data["X"], data["SC"]
    N      = data["N"]

    EC = graphnet_effective_connectivity(X, SC,
                                         lambda_ridge=1.0, lambda_graph=1.0)
    A  = normalise_matrix(EC, target_rho=target_rho)
    row["ec_asymmetry"] = float(
        np.linalg.norm(EC - EC.T, "fro") / np.linalg.norm(EC, "fro"))
    row["raw_rho"]      = float(np.max(np.abs(np.linalg.eigvals(EC))))
    row["final_rho"]    = float(np.max(np.abs(np.linalg.eigvals(A))))

    B   = np.eye(N)
    W_T = compute_gramian_doubling(A, B, horizon)
    eig = np.linalg.eigvalsh(W_T)
    row["gramian_min_eig"] = float(np.min(eig))
    row["gramian_cond"]    = float(np.linalg.cond(W_T))

    # AUD brain states -- REAL REWARD-NETWORK VERSION
    x_rest, x_craving = define_states_real_reward(X, reward_indices)

    E_crav2rest, _ = minimum_energy(A, B, x_craving, x_rest,    T=horizon)
    E_rest2crav, _ = minimum_energy(A, B, x_rest,    x_craving, T=horizon)
    row["E_craving_to_rest"] = float(E_crav2rest)
    row["E_rest_to_craving"] = float(E_rest2crav)
    row["E_asymmetry"]       = float(E_crav2rest - E_rest2crav)

    FC   = functional_connectivity(X)
    A_fc = normalise_matrix(FC, target_rho=target_rho)
    E_fc, _ = minimum_energy(A_fc, B, x_craving, x_rest, T=horizon)
    row["E_fc"]           = float(E_fc)
    row["E_ec"]           = float(E_crav2rest)
    row["teleport_ratio"] = float(E_fc / (E_crav2rest + 1e-12))

    row["avg_ctrl_mean"] = float(np.mean(average_controllability(A)))
    row["mod_ctrl_mean"] = float(np.mean(modal_controllability(A)))

    fvi = finite_vs_infinite_comparison(
        A, B, x_craving, x_rest, T_range=list(range(1, 21)))
    fvi.insert(0, "subject_id", sid)
    fvi.insert(1, "Severity", severity)
    fvi.insert(2, "Severity_Label", SEVERITY_LABEL.get(severity, ""))

    return row, fvi


#  Discordant MZ pair analysis (identical to original)
def compute_discordant_pairs(res_df):
    mz = res_df[res_df["ZygosityGT1"] == "MZ"]
    rows = []
    for pair_id, grp in mz.groupby("TwinPairID"):
        if len(grp) != 2:
            continue
        s = sorted(grp["Severity"].values)
        if s[0] == s[1]:
            continue
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


#  Group summary (identical to original)
def compute_group_summary(res_df):
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
    ap.add_argument("--limit",      type=int,   default=0,
                     help="Take the first N subjects by file order -- NOT "
                          "severity-aware. For a scientifically meaningful "
                          "small sample, use --subjects instead.")
    ap.add_argument("--subjects",   default=None,
                     help="Comma-separated explicit subject ID list, e.g. "
                          "for targeting specific discordant MZ pairs "
                          "rather than an arbitrary file-order slice. "
                          "Takes precedence over --limit if both given.")
    ap.add_argument("--exclude-vmpfc", action="store_true",
                     help="Run without the vmPFC region set, for a "
                          "version of this check that doesn't depend on "
                          "the one methodological choice in the ROI "
                          "list (vmPFC has no literal HCP-MMP label, "
                          "unlike NAc/caudate/putamen/OFC).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    reward_dict = dict(REWARD_INDICES_DEFAULT)
    if args.exclude_vmpfc:
        reward_dict.pop("vmPFC")
        print("Running WITHOUT vmPFC (--exclude-vmpfc set)")
    reward_indices = flatten_reward_indices(reward_dict)
    print(f"Reward network: {sum(len(v) for v in reward_dict.values())} "
          f"regions across {list(reward_dict.keys())}")
    print(f"Indices: {reward_indices}")

    meta      = pd.read_csv(args.meta)
    meta["Subject"] = meta["Subject"].astype(str)
    meta_dict = meta.set_index("Subject").to_dict(orient="index")

    subjects, ts_files, sc_files = discover_subjects(
        args.ts_dir, args.sc_dir, args.ts_suffix, args.sc_suffix)
    subjects = [s for s in subjects if s in meta_dict]
    if args.subjects:
        wanted = [s.strip() for s in args.subjects.split(",")]
        missing = [s for s in wanted if s not in subjects]
        if missing:
            print(f"WARNING: requested subjects not found (no files/metadata): {missing}")
        subjects = [s for s in wanted if s in subjects]
    elif args.limit:
        subjects = subjects[:args.limit]

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
                meta_dict[sid], args.target_rho, args.horizon, reward_indices)
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

    res_df = pd.DataFrame(results)
    res_df.to_csv(
        os.path.join(args.out_dir, "aud_cohort_results_real_reward.csv"),
        index=False)

    if not res_df.empty:
        compute_group_summary(res_df).to_csv(
            os.path.join(args.out_dir, "aud_group_comparison_real_reward.csv"),
            index=False)
        compute_discordant_pairs(res_df).to_csv(
            os.path.join(args.out_dir, "aud_discordant_pairs_real_reward.csv"),
            index=False)
        res_df[["subject_id", "Severity_Label",
                "E_fc", "E_ec", "teleport_ratio"]].to_csv(
            os.path.join(args.out_dir, "aud_teleportation_real_reward.csv"),
            index=False)

    if fvi_all:
        pd.concat(fvi_all, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "aud_finite_vs_infinite_real_reward.csv"),
            index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(args.out_dir, "aud_batch_failures_real_reward.csv"),
            index=False)

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"DONE: {len(results)}/{len(subjects)} subjects, "
          f"{len(failures)} failed, {dt/60:.1f} min")

    if not res_df.empty:
        print("\nGroup medians (E_craving_to_rest, REAL REWARD-NETWORK):")
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
    print("\nRun the ORIGINAL run_hcp_aud_batch.py on the same subjects "
          "and compare aud_cohort_results.csv against "
          "aud_cohort_results_real_reward.csv directly -- this script "
          "does not replace that comparison, it produces one half of it.")


if __name__ == "__main__":
    main()
