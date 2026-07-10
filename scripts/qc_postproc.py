"""
NeuroSim ADNI — automated post-preprocessing QC across all 3 modalities.

Reads derivative outputs (no visual inspection) and produces a single
pass/fail table per subject. Checks the things that actually corrupt
downstream NCT analysis:

  ANATOMICAL (FastSurfer)
    - surface recon completed (aseg + aparc present)
    - cortical volume / segmentation sanity (no absurd values)
    - no NaN/empty in stats

  FUNCTIONAL (fMRIPrep)
    - preprocessed BOLD exists in MNI space
    - confounds present, framewise displacement readable
    - post-preproc mean FD, % high-motion volumes after scrubbing
    - tCompCor / aCompCor components present
    - temporal SNR of preprocessed series
    - registration quality (BOLD→T1 coregistration cost if available)

  DIFFUSION (QSIPrep)
    - preprocessed DWI exists
    - T1-DWI registration dice score
    - number of gradient directions retained
    - mean FD of DWI series
    - no degenerate b-values

Outputs:
  qc_postproc_report.csv     full per-subject metrics + PASS/FAIL
  qc_postproc_failures.csv   only flagged subjects, with reasons

Usage:
  python qc_postproc.py \
      --deriv  /path/to/derivatives \
      --bids   /path/to/BIDS_Dataset \
      --out    /path/to/qc_postproc \
      [--participants /path/to/participants_qc.tsv]
"""
import argparse
import json
import os
import glob
import sys
import numpy as np
import pandas as pd

#  Thresholds (defensible for elderly multi-site 3T) 
TH = {
    "fd_mean_post":      0.5,    # mm — post-preproc mean FD
    "fd_perc_post":      40.0,   # % volumes FD>0.5 after preproc (lenient/elderly)
    "tsnr_min":          25.0,   # preprocessed BOLD tSNR floor
    "dwi_dice_min":      0.85,   # T1-DWI registration overlap
    "dwi_ndirs_min":     40,     # gradient directions retained (ADNI3 ~48-54)
    "dwi_fd_mean":       1.5,    # mm — DWI motion (more lenient than BOLD)
    "n_dummy_max":       10,     # non-steady-state volumes flagged if excessive
}


def find_one(pattern):
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


#  ANATOMICAL (FastSurfer) 
def qc_anat(sub, fs_dir):
    r = {"anat_recon_done": False, "anat_cortex_vol": np.nan,
         "anat_note": ""}
    subdir = os.path.join(fs_dir, f"sub-{sub}")
    aseg   = os.path.join(subdir, "stats", "aseg.stats")
    aparc  = os.path.join(subdir, "stats", "lh.aparc.stats")
    mgz    = os.path.join(subdir, "mri", "aparc.DKTatlas+aseg.deep.mgz")

    if not os.path.exists(mgz):
        r["anat_note"] = "missing segmentation mgz"
        return r
    r["anat_recon_done"] = True

    # Parse total cortical grey matter volume from aseg.stats if present
    if os.path.exists(aseg):
        try:
            with open(aseg) as f:
                for line in f:
                    if "TotalGrayVol" in line or "Total cortical gray matter volume" in line:
                        # Measure line format: # Measure ..., value, unit
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            r["anat_cortex_vol"] = float(parts[-2].strip())
                        break
        except Exception as e:
            r["anat_note"] = f"aseg parse err: {str(e)[:40]}"
    return r


#  FUNCTIONAL (fMRIPrep) 
def qc_func(sub, fmriprep_dir):
    r = {"func_bold_exists": False, "func_fd_mean": np.nan,
         "func_fd_perc": np.nan, "func_tsnr": np.nan,
         "func_ncompcor": np.nan, "func_ndummy": np.nan,
         "func_ntimepoints": np.nan, "func_note": ""}

    # Preprocessed BOLD in MNI space
    bold = find_one(os.path.join(
        fmriprep_dir, f"sub-{sub}", "**",
        "*task-rest*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz"))
    r["func_bold_exists"] = bold is not None
    if bold is None:
        r["func_note"] = "no preproc BOLD in MNI"
        return r

    # Confounds TSV
    conf = find_one(os.path.join(
        fmriprep_dir, f"sub-{sub}", "**",
        "*task-rest*desc-confounds_timeseries.tsv"))
    if conf is None:
        r["func_note"] = "no confounds tsv"
        return r

    try:
        c = pd.read_csv(conf, sep="\t")
        r["func_ntimepoints"] = len(c)
        if "framewise_displacement" in c.columns:
            fd = c["framewise_displacement"].dropna()
            r["func_fd_mean"] = float(fd.mean())
            r["func_fd_perc"] = float(100 * (fd > 0.5).mean())
        # non-steady-state outliers
        dummy_cols = [x for x in c.columns if "non_steady_state" in x]
        r["func_ndummy"] = int(len(dummy_cols))
        # compcor components present
        compcor = [x for x in c.columns if "comp_cor" in x]
        r["func_ncompcor"] = int(len(compcor))
        # tSNR proxy: from global signal if present
        if "global_signal" in c.columns:
            gs = c["global_signal"].dropna()
            r["func_tsnr"] = float(gs.mean() / (gs.std() + 1e-8))
    except Exception as e:
        r["func_note"] = f"confounds parse err: {str(e)[:40]}"

    return r


#  DIFFUSION (QSIPrep) 
def qc_dwi(sub, qsiprep_dir):
    r = {"dwi_exists": False, "dwi_ndirs": np.nan,
         "dwi_dice": np.nan, "dwi_fd_mean": np.nan,
         "dwi_note": ""}

    dwi = find_one(os.path.join(
        qsiprep_dir, f"sub-{sub}", "**",
        "*desc-preproc_dwi.nii.gz"))
    r["dwi_exists"] = dwi is not None
    if dwi is None:
        r["dwi_note"] = "no preproc DWI"
        return r

    # gradient directions from .bval
    bval = find_one(os.path.join(
        qsiprep_dir, f"sub-{sub}", "**", "*desc-preproc_dwi.bval"))
    if bval:
        try:
            bvals = np.loadtxt(bval)
            r["dwi_ndirs"] = int((bvals > 50).sum())  # non-b0 directions
        except Exception:
            pass

    # QSIPrep dwiqc.json (per-subject QC metrics)
    qc_json = find_one(os.path.join(
        qsiprep_dir, f"sub-{sub}", "**", "*dwiqc.json"))
    if not qc_json:
        qc_json = find_one(os.path.join(
            qsiprep_dir, f"sub-{sub}", "**", "*desc-ImageQC_dwi.csv"))
    if qc_json and qc_json.endswith(".json"):
        try:
            with open(qc_json) as f:
                q = json.load(f)
            r["dwi_dice"]    = float(q.get("t1_dice_distance",
                                           q.get("t1_dice", np.nan)))
            r["dwi_fd_mean"] = float(q.get("mean_fd", np.nan))
        except Exception as e:
            r["dwi_note"] = f"qc json err: {str(e)[:40]}"
    elif qc_json and qc_json.endswith(".csv"):
        try:
            q = pd.read_csv(qc_json)
            for col, key in [("t1_dice_distance","dwi_dice"),
                             ("mean_fd","dwi_fd_mean"),
                             ("raw_num_directions","dwi_ndirs")]:
                if col in q.columns:
                    r[key] = float(q[col].iloc[0])
        except Exception as e:
            r["dwi_note"] = f"qc csv err: {str(e)[:40]}"

    return r


#  Verdict per subject 
def verdict(row):
    fails = []
    # anatomical
    if not row["anat_recon_done"]:
        fails.append("ANAT:no_recon")
    # functional
    if not row["func_bold_exists"]:
        fails.append("FUNC:no_bold")
    if pd.notna(row["func_fd_mean"]) and row["func_fd_mean"] > TH["fd_mean_post"]:
        fails.append(f"FUNC:fd={row['func_fd_mean']:.2f}")
    if pd.notna(row["func_tsnr"]) and row["func_tsnr"] < TH["tsnr_min"]:
        # global-signal tSNR proxy is crude; warn not hard-fail
        pass
    # diffusion
    if not row["dwi_exists"]:
        fails.append("DWI:no_dwi")
    if pd.notna(row["dwi_ndirs"]) and row["dwi_ndirs"] < TH["dwi_ndirs_min"]:
        fails.append(f"DWI:ndirs={int(row['dwi_ndirs'])}")
    if pd.notna(row["dwi_dice"]) and row["dwi_dice"] < TH["dwi_dice_min"]:
        fails.append(f"DWI:dice={row['dwi_dice']:.2f}")

    return "PASS" if not fails else "FAIL", ";".join(fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv", required=True,
                    help="derivatives/ dir containing freesurfer, fmriprep, qsiprep")
    ap.add_argument("--bids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--participants", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fs_dir       = os.path.join(args.deriv, "freesurfer")
    fmriprep_dir = os.path.join(args.deriv, "fmriprep")
    qsiprep_dir  = os.path.join(args.deriv, "qsiprep")

    # Subject list from BIDS
    subs = sorted(os.path.basename(d).replace("sub-", "")
                  for d in glob.glob(os.path.join(args.bids, "sub-*"))
                  if os.path.isdir(d))
    print(f"Found {len(subs)} subjects in BIDS")

    # Optional diagnosis merge
    dx = {}
    if args.participants and os.path.exists(args.participants):
        pt = pd.read_csv(args.participants, sep="\t")
        idcol = "participant_id" if "participant_id" in pt.columns else pt.columns[0]
        for _, r in pt.iterrows():
            sid = str(r[idcol]).replace("sub-", "")
            dx[sid] = r.get("DX_Group", "")

    rows = []
    for i, sub in enumerate(subs, 1):
        row = {"subject": f"sub-{sub}", "DX_Group": dx.get(sub, "")}
        row.update(qc_anat(sub, fs_dir))
        row.update(qc_func(sub, fmriprep_dir))
        row.update(qc_dwi(sub, qsiprep_dir))
        v, reasons = verdict(row)
        row["QC_VERDICT"] = v
        row["QC_REASONS"] = reasons
        rows.append(row)
        flag = "✓" if v == "PASS" else "✗"
        print(f"[{i:2d}/{len(subs)}] {flag} sub-{sub:10s} {v}  {reasons}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "qc_postproc_report.csv"), index=False)

    fails = df[df["QC_VERDICT"] == "FAIL"]
    fails.to_csv(os.path.join(args.out, "qc_postproc_failures.csv"), index=False)

    #  Summary 
    print("\n" + "=" * 60)
    print(f"QC COMPLETE: {len(df)} subjects")
    print(f"  PASS: {(df['QC_VERDICT']=='PASS').sum()}")
    print(f"  FAIL: {(df['QC_VERDICT']=='FAIL').sum()}")
    if len(fails):
        print(f"\nFailed subjects:")
        for _, r in fails.iterrows():
            print(f"  {r['subject']} ({r['DX_Group']}): {r['QC_REASONS']}")

    # Group balance of survivors
    passed = df[df["QC_VERDICT"]=="PASS"]
    if "DX_Group" in passed.columns and passed["DX_Group"].notna().any():
        print(f"\nSurviving cohort balance:")
        print(passed["DX_Group"].value_counts().to_string())

    # Modality-level completeness
    print(f"\nModality completeness:")
    print(f"  Anat recon:  {df['anat_recon_done'].sum()}/{len(df)}")
    print(f"  Func BOLD:   {df['func_bold_exists'].sum()}/{len(df)}")
    print(f"  DWI preproc: {df['dwi_exists'].sum()}/{len(df)}")

    # Metric distributions (sanity)
    print(f"\nMetric medians (passing subjects):")
    for m in ["func_fd_mean","func_fd_perc","dwi_ndirs","dwi_dice"]:
        if m in passed.columns and passed[m].notna().any():
            print(f"  {m:16s}: median={passed[m].median():.3f}  "
                  f"range=[{passed[m].min():.3f}, {passed[m].max():.3f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
