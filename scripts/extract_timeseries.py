#!/usr/bin/env python3
"""
NeuroSim ADNI — BOLD confound regression + parcellation.

Takes fMRIPrep preprocessed BOLD (MNI space) and produces NeuroSim-format
parcellated timeseries (_native_timeseries.npy) for TWO atlases, matching
the HCP run's format exactly.

Confound strategy (LOCKED — methods-paper commitment):
  - 24 motion parameters (6 + derivatives + squares)
  - aCompCor: top 5 WM + 5 CSF components
  - Cosine-basis high-pass drift (fMRIPrep cosine_XX regressors)
  - FD>0.5mm spike censoring (scrubbing)
  - NO global signal regression (corrupts directed EC sign structure)
  - High-pass only (~0.01 Hz via cosine); NO aggressive low-pass

Output per subject per atlas:
  sub-<ID>_<atlas>_native_timeseries.npy   shape (T, N)

Usage:
  python extract_timeseries.py \
      --fmriprep /path/to/derivatives/fmriprep \
      --out      /path/to/derivatives/timeseries \
      --atlas-schaefer /path/Schaefer2018_400_MNI152.nii.gz \
      --atlas-410      /path/HCPMMP_Tian_410_MNI152.nii.gz \
      [--n-jobs 8] [--fd-thresh 0.5] [--test]
"""
import argparse
import glob
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from joblib import Parallel, delayed
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image as nimg


# Confound selection
def build_confound_matrix(conf_df, fd_thresh=0.5):
    """Assemble the 24P + aCompCor + cosine confound matrix from fMRIPrep TSV."""
    cols = []

    # 24 motion parameters
    motion_base = ["trans_x", "trans_y", "trans_z",
                   "rot_x", "rot_y", "rot_z"]
    for m in motion_base:
        for suffix in ["", "_derivative1", "_power2", "_derivative1_power2"]:
            col = m + suffix
            if col in conf_df.columns:
                cols.append(col)

    # aCompCor — top 5 WM + 5 CSF (fMRIPrep labels a_comp_cor_XX)
    acompcor = [c for c in conf_df.columns if c.startswith("a_comp_cor_")]
    cols.extend(acompcor[:10])   # first 10 (5 WM + 5 CSF in fMRIPrep order)

    # Cosine high-pass drift
    cosine = [c for c in conf_df.columns if c.startswith("cosine")]
    cols.extend(cosine)

    # Build matrix, fill NaN (first-row derivatives) with 0
    conf = conf_df[cols].fillna(0).values

    # Spike regressors for FD>thresh (scrubbing as regression)
    if "framewise_displacement" in conf_df.columns:
        fd = conf_df["framewise_displacement"].fillna(0).values
        spikes = np.where(fd > fd_thresh)[0]
        if len(spikes) > 0:
            spike_reg = np.zeros((len(fd), len(spikes)))
            for i, t in enumerate(spikes):
                spike_reg[t, i] = 1
            conf = np.hstack([conf, spike_reg])
        n_censored = len(spikes)
    else:
        n_censored = 0

    return conf, n_censored


def process_subject(sub, fmriprep_dir, out_dir, atlases, fd_thresh):
    """Extract parcellated, confound-cleaned timeseries for one subject."""
    result = {"subject": sub, "status": "ok", "note": ""}

    # Find preprocessed BOLD (MNI) and confounds
    bold = glob.glob(os.path.join(
        fmriprep_dir, f"sub-{sub}", "**",
        "*task-rest*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz"),
        recursive=True)
    conf = glob.glob(os.path.join(
        fmriprep_dir, f"sub-{sub}", "**",
        "*task-rest*desc-confounds_timeseries.tsv"), recursive=True)

    if not bold or not conf:
        result["status"] = "fail"
        result["note"] = "missing BOLD or confounds"
        return result

    bold, conf = bold[0], conf[0]
    conf_df = pd.read_csv(conf, sep="\t")
    confounds, n_censored = build_confound_matrix(conf_df, fd_thresh)
    result["n_timepoints"] = len(conf_df)
    result["n_censored"] = n_censored
    result["n_confounds"] = confounds.shape[1]

    for atlas_name, atlas_path in atlases.items():
        try:
            masker = NiftiLabelsMasker(
                labels_img=atlas_path,
                standardize="zscore_sample",
                detrend=True,
                low_pass=None,          # NO low-pass (preserve dynamics for NCT)
                high_pass=0.01,         # light high-pass
                t_r=3.0,                # ADNI3 TR
                memory=None,
                verbose=0,
            )
            ts = masker.fit_transform(bold, confounds=confounds)  # (T, N)

            out_path = os.path.join(
                out_dir, f"sub-{sub}_{atlas_name}_native_timeseries.npy")
            np.save(out_path, ts)
            result[f"{atlas_name}_shape"] = str(ts.shape)
            # sanity: any all-zero parcels (atlas/coverage mismatch)?
            zero_parcels = int((ts.std(axis=0) < 1e-8).sum())
            result[f"{atlas_name}_zero_parcels"] = zero_parcels
        except Exception as e:
            result["status"] = "partial"
            result["note"] += f" {atlas_name}:err:{str(e)[:50]}"

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas-schaefer", required=True,
                help="Schaefer-400+TianS3 (primary)")
    ap.add_argument("--atlas-410", required=True,
                help="Schaefer-400 cortex-only (secondary)")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--fd-thresh", type=float, default=0.5)
    ap.add_argument("--test", action="store_true",
                    help="Process first subject only")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    atlases = {"tians3": args.atlas_schaefer, "schaefer400": args.atlas_410}

    # Discover subjects with preprocessed BOLD
    subs = sorted({
        os.path.basename(p).split("_")[0].replace("sub-", "")
        for p in glob.glob(os.path.join(
            args.fmriprep, "sub-*", "**",
            "*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz"),
            recursive=True)
    })
    if args.test:
        subs = subs[:1]
    print(f"Found {len(subs)} subjects with preprocessed BOLD")
    print(f"Confound strategy: 24P + aCompCor(10) + cosine + FD-spikes, NO GSR")
    print(f"Filtering: high-pass 0.01Hz, no low-pass, TR=3.0s\n")

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_subject)(
            sub, args.fmriprep, args.out, atlases, args.fd_thresh)
        for sub in subs
    )

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(args.out, "extraction_report.csv"), index=False)

    print("\n" + "=" * 60)
    ok = (rdf["status"] == "ok").sum()
    print(f"DONE: {ok}/{len(subs)} clean, "
          f"{(rdf['status']=='partial').sum()} partial, "
          f"{(rdf['status']=='fail').sum()} failed")
    if "n_censored" in rdf.columns:
        print(f"Censored volumes: median={rdf['n_censored'].median():.0f}, "
              f"max={rdf['n_censored'].max():.0f}")
    # Flag subjects with zero parcels (coverage problems)
    for atlas in ["tians3", "schaefer400"]:
        col = f"{atlas}_zero_parcels"
        if col in rdf.columns:
            bad = rdf[rdf[col] > 0]
            if len(bad):
                print(f"\n{atlas}: {len(bad)} subjects with empty parcels:")
                for _, r in bad.iterrows():
                    print(f"  sub-{r['subject']}: {int(r[col])} empty")
    print("=" * 60)


if __name__ == "__main__":
    main()
