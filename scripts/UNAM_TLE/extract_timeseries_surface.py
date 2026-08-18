#!/usr/bin/env python3
"""
UNAM_TLE — surface-based DK parcellation, paired natively with ENIGMA's
'aparc' template SC (fsaverage5 space, per ENIGMA's own documented
parcel_to_surface(..., 'aparc_fsa5') convention).

Why surface instead of volumetric: ENIGMA built their SC template and
their surface labels together, so pairing is correct by construction.
No third-party volumetric atlas, no numeric-ID guessing, no space-
warping risk -- this is the class of bug that has cost real time twice
already this project (Schaefer ordering, DK 70-vs-68 mismatch).

Requires: fMRIPrep run with --output-spaces including fsaverage5,
so *_space-fsaverage5_hemi-{L,R}_bold.func.gii files exist per subject.

Region matching: FreeSurfer aparc.annot names (e.g. 'bankssts') are
matched against ENIGMA's 'L_bankssts'/'R_bankssts' by stripping the
hemisphere prefix -- exact string match required, not positional.
"""
import argparse
import glob
import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from enigmatoolbox.datasets import load_sc

warnings.filterwarnings("ignore")


def get_fsaverage5_annot_labels(hemi):
    """
    Load the standard FreeSurfer fsaverage5 aparc.annot for one hemisphere.
    Ships with every FreeSurfer/FastSurfer install.
    """
    import subprocess
    fs_home = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
    annot_path = os.path.join(
        fs_home, "subjects", "fsaverage5", "label", f"{hemi}.aparc.annot")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(
            f"fsaverage5 annot not found at {annot_path} -- "
            f"set FREESURFER_HOME or verify FreeSurfer/FastSurfer install")
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, names  # labels: per-vertex region index; names: region names


def build_enigma_match(names_lh, names_rh, enigma_labels):
    """
    Build index arrays mapping fsaverage5 region order -> ENIGMA order,
    by NAME matching (L_<name> / R_<name>), not position.
    """
    enigma_list = list(enigma_labels)
    lh_order, rh_order = [], []
    unmatched = []

    for i, name in enumerate(names_lh):
        target = f"L_{name}"
        if target in enigma_list:
            lh_order.append((i, enigma_list.index(target)))
        elif name not in ("unknown", "corpuscallosum"):
            unmatched.append(f"LH:{name}")

    for i, name in enumerate(names_rh):
        target = f"R_{name}"
        if target in enigma_list:
            rh_order.append((i, enigma_list.index(target)))
        elif name not in ("unknown", "corpuscallosum"):
            unmatched.append(f"RH:{name}")

    if unmatched:
        print(f"  WARNING unmatched regions (excluded, expected for "
              f"unknown/corpuscallosum): {unmatched}")

    return lh_order, rh_order


def process_subject(sub, fmriprep_dir, out_dir, lh_annot_idx, rh_annot_idx,
                    lh_map, rh_map, n_enigma):
    result = {"subject": sub, "status": "ok", "note": ""}

    lh_gii = glob.glob(os.path.join(
        fmriprep_dir, f"sub-{sub}", "**",
        "*hemi-L_space-fsaverage5_bold.func.gii"), recursive=True)
    rh_gii = glob.glob(os.path.join(
        fmriprep_dir, f"sub-{sub}", "**",
        "*hemi-R_space-fsaverage5_bold.func.gii"), recursive=True)

    if not lh_gii or not rh_gii:
        result["status"] = "fail"
        result["note"] = "missing fsaverage5 surface BOLD"
        return result

    lh_data = nib.load(lh_gii[0])
    rh_data = nib.load(rh_gii[0])
    lh_ts = np.array([d.data for d in lh_data.darrays])   # (T, n_vertices)
    rh_ts = np.array([d.data for d in rh_data.darrays])
    T = lh_ts.shape[0]
    result["n_timepoints"] = T

    # Parcellate: mean timeseries per region, placed in ENIGMA order
    ts_enigma = np.zeros((T, n_enigma))
    filled = np.zeros(n_enigma, dtype=bool)

    for annot_idx, enigma_idx in lh_map:
        vertex_mask = (lh_annot_idx == annot_idx)
        if vertex_mask.sum() > 0:
            ts_enigma[:, enigma_idx] = lh_ts[:, vertex_mask].mean(axis=1)
            filled[enigma_idx] = True

    for annot_idx, enigma_idx in rh_map:
        vertex_mask = (rh_annot_idx == annot_idx)
        if vertex_mask.sum() > 0:
            ts_enigma[:, enigma_idx] = rh_ts[:, vertex_mask].mean(axis=1)
            filled[enigma_idx] = True

    result["n_regions_filled"] = int(filled.sum())
    if not filled.all():
        result["status"] = "partial"
        result["note"] = f"{(~filled).sum()} regions unfilled"

    out_path = os.path.join(out_dir, f"sub-{sub}_dk68_native_timeseries.npy")
    np.save(out_path, ts_enigma)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmriprep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Load ENIGMA template + fsaverage5 annot, build the name-match map ──
    sc_ctx, enigma_labels, sc_sctx, sctx_labels = load_sc(parcellation='aparc')
    lh_idx, names_lh = get_fsaverage5_annot_labels("lh")
    rh_idx, names_rh = get_fsaverage5_annot_labels("rh")
    lh_map, rh_map = build_enigma_match(names_lh, names_rh, enigma_labels)

    print(f"ENIGMA regions: {len(enigma_labels)}")
    print(f"LH matched: {len(lh_map)}  RH matched: {len(rh_map)}  "
          f"Total: {len(lh_map)+len(rh_map)}")
    if len(lh_map) + len(rh_map) != len(enigma_labels):
        print("*** WARNING: matched count != ENIGMA region count. ***")
        print("Inspect unmatched list above before trusting results.")

    # Save the template SC once, in ENIGMA's own order (ground truth)
    np.savetxt(os.path.join(args.out, "template_SC_dk68.csv"),
               sc_ctx, delimiter=",")
    print(f"Template SC saved: {sc_ctx.shape}")

    subs = sorted({
        os.path.basename(p).split("_")[0].replace("sub-", "")
        for p in glob.glob(os.path.join(
            args.fmriprep, "sub-*", "**",
            "*hemi-L_space-fsaverage5_bold.func.gii"), recursive=True)
    })
    print(f"\nFound {len(subs)} subjects with fsaverage5 surface BOLD\n")

    if len(subs) == 0:
        print("*** ZERO subjects found. Stopping before writing empty output. ***")
        print("Check the actual filenames on disk, e.g.:")
        print(f"  find {args.fmriprep} -name '*fsaverage5*func.gii' | head -3")
        print("and confirm the glob pattern above matches the real entity order.")
        return

    results = [process_subject(s, args.fmriprep, args.out, lh_idx, rh_idx,
                               lh_map, rh_map, len(enigma_labels))
              for s in subs]

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(args.out, "extraction_report.csv"), index=False)
    print(f"\nDONE: {(rdf['status']=='ok').sum()}/{len(subs)} clean, "
          f"{(rdf['status']=='partial').sum()} partial, "
          f"{(rdf['status']=='fail').sum()} failed")


if __name__ == "__main__":
    main()