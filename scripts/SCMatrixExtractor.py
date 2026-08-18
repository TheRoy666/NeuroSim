#!/usr/bin/env python3
"""
per_subject_SC_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compute 410×410 structural connectivity matrices for all 238 HCP-S1200
twin subjects. Tractography parameters match the population template
pipeline: ACT + SIFT2, backtrack, crop_at_gmwmi, 10M streamlines.

Atlas label scheme (preserved in every output matrix row/column):
    1 –  50 : Tian S3 subcortical  (RH 1–25, LH 26–50)
   51 – 230 : HCP-MMP RH cortical  (R_V1_ROI = 51 … = 230)
  231 – 410 : HCP-MMP LH cortical  (L_V1_ROI = 231 … = 410)

Per-subject pipeline (N_WORKERS=4 subjects in parallel):
  Stage 1  Build 410-region parcellation in subject DWI space
           wb_command cifti-separate → ribbon-constrained label-to-volume-mapping
           applywarp subcortical MNI→acpc → mrtransform to DWI grid
  Stage 2  5TT tissue segmentation from T1 brain → mrtransform to DWI space
  Stage 3  GMWMI seed mask
  Stage 4  Parallel tckgen  (2 chunks × 5M = 10M; ThreadPool within ProcessPool)
           Flags: -backtrack -crop_at_gmwmi -act -seed_gmwmi -cutoff 0.06
  Stage 5  tcksift2
  Stage 6  tck2connectome → 410×410 CSV + streamline-count CSV
           Delete tractogram after extraction (~5–15 GB per subject)
  Stage 7  Shape + symmetry QC; per-subject log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
  python per_subject_SC_pipeline.py             # full batch
  python per_subject_SC_pipeline.py --dry-run   # first subject, serial
  python per_subject_SC_pipeline.py --subject 100610
"""

import os, sys, time, argparse, logging, traceback, subprocess
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import nibabel as nib

# ── CONFIGURATION ─────────────────────────────────────────────────────────
MRTRIX_BIN = Path('/home/khushbu-lab4/mrtrix3_new/build/bin')
WB_CMD     = '/home/khushbu-lab4/miniforge3/envs/dtiproject/bin/wb_command'

PROC_DIR    = Path('/home/khushbu-lab4/Dtiproject/processed')
STRUCT_ROOT = Path('/home/khushbu-lab4/HCPStructuralData/Raw')
ATLAS_PATH  = Path('/home/khushbu-lab4/HCPStructuralData/Atlas/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S3.dlabel.nii')
SUBJ_CSV    = Path('/home/khushbu-lab4/Dtiproject/twintables/Twins_240_beh_sheet_complete_all_vars - Sheet1.csv')

SC_OUT_DIR  = Path('/home/khushbu-lab4/Dtiproject/SC_matrices')
SC_WORK_DIR = Path('/home/khushbu-lab4/Dtiproject/SC_work')
SC_LOG_DIR  = Path('/home/khushbu-lab4/Dtiproject/SC_logs')
ATLAS_WORK  = Path('/home/khushbu-lab4/Dtiproject/SC_atlas_work')

N_WORKERS        = 4          # subjects in parallel
CHUNKS_PER_SUBJ  = 2          # parallel tckgen chunks per subject
TRACKS_PER_CHUNK = 5_000_000  # 2 × 5M = 10M total streamlines
THREADS_TCKGEN   = 10         # threads per tckgen chunk
THREADS_SIFT2    = 10         # threads for tcksift2
THREADS_CONN     = 4          # threads for tck2connectome

TCKGEN_CUTOFF = '0.06'
TCKGEN_MINLEN = '10'
TCKGEN_MAXLEN = '250'

for _d in [SC_OUT_DIR, SC_WORK_DIR, SC_LOG_DIR, ATLAS_WORK]:
    _d.mkdir(parents=True, exist_ok=True)

N_REGIONS = 410


# ── UTILITIES ─────────────────────────────────────────────────────────────
def mrt(tool: str) -> str:
    return str(MRTRIX_BIN / tool)


def run(cmd, step_name: str, log: logging.Logger, check=True):
    """Run a command list with timing. Raises RuntimeError on failure."""
    log.info(f'START  {step_name}')
    log.debug(f'CMD    {" ".join(str(c) for c in cmd)}')
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    elapsed = timedelta(seconds=int(time.time() - t0))
    if result.returncode != 0 and check:
        log.error(f'FAILED {step_name} [{elapsed}]')
        log.error(f'STDERR:\n{result.stderr[-3000:]}')
        raise RuntimeError(f'{step_name} failed — check log')
    log.info(f'DONE   {step_name} [{elapsed}]')
    return result


def get_logger(subj_id: str) -> logging.Logger:
    log_path = SC_LOG_DIR / f'{subj_id}_SC.log'
    log = logging.getLogger(f'SC.{subj_id}')
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        fh = logging.FileHandler(str(log_path), mode='w')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S'))
        log.addHandler(fh)
    return log


def subject_paths(subj_id: str, zyg: str, pair_id: str, gender: int) -> dict:
    """Return all relevant paths for one subject as a dict of Path objects."""
    sex    = 'F' if gender == 0 else 'M'   # Gender 0=Female(_F), 1=Male(_M)
    dwi    = PROC_DIR / zyg / pair_id / f'sub-{subj_id}_{sex}'
    struct = STRUCT_ROOT / subj_id / 'T1w'
    native = struct / 'Native'
    return {
        'dwi_dir':       dwi,
        'wmfod_norm':    dwi / 'wmfod_norm.mif',
        't1_brain':      struct / 'T1w_acpc_dc_restore_brain.nii.gz',
        'aparc_aseg':    struct / 'aparc+aseg.nii.gz',
        'xfm_std2acpc':  STRUCT_ROOT / subj_id / 'MNINonLinear' / 'xfms' / 'standard2acpc_dc.nii.gz',
        'midthick_lh':   native / f'{subj_id}.L.midthickness.native.surf.gii',
        'midthick_rh':   native / f'{subj_id}.R.midthickness.native.surf.gii',
        'white_lh':      native / f'{subj_id}.L.white.native.surf.gii',
        'white_rh':      native / f'{subj_id}.R.white.native.surf.gii',
        'pial_lh':       native / f'{subj_id}.L.pial.native.surf.gii',
        'pial_rh':       native / f'{subj_id}.R.pial.native.surf.gii',
    }


def preflight_check(paths: dict, log: logging.Logger):
    required = ['wmfod_norm', 't1_brain', 'aparc_aseg', 'xfm_std2acpc',
                'midthick_lh', 'midthick_rh', 'white_lh', 'white_rh',
                'pial_lh', 'pial_rh']
    for key in required:
        p = paths[key]
        if not p.exists():
            raise FileNotFoundError(f'Missing {key}: {p}')
    log.info('Preflight: all required files present ✓')


# ── ONE-TIME ATLAS SETUP ──────────────────────────────────────────────────
def setup_atlas_once(log: logging.Logger):
    """
    Run wb_command -cifti-separate once for the whole batch.
    Uses a sentinel file to avoid re-running. Not parallelised — must be
    called from the main process before ProcessPoolExecutor is launched.
    """
    done_flag = ATLAS_WORK / 'ATLAS_SEPARATED.done'
    if done_flag.exists():
        log.info('Atlas components already extracted — skipping cifti-separate')
        return

    log.info('Extracting atlas components from CIFTI (one-time setup)...')
    result = subprocess.run([
        WB_CMD, '-cifti-separate', str(ATLAS_PATH), 'COLUMN',
        '-label', 'CORTEX_LEFT',  str(ATLAS_WORK / 'lh_atlas.label.gii'),
        '-label', 'CORTEX_RIGHT', str(ATLAS_WORK / 'rh_atlas.label.gii'),
        '-volume-all', str(ATLAS_WORK / 'subcortical_mni.nii.gz'),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f'wb_command -cifti-separate failed:\n{result.stderr}')

    done_flag.touch()
    log.info('Atlas components extracted and cached ✓')


# ── PARCELLATION MERGE ────────────────────────────────────────────────────
def merge_parcellation(lh_path, rh_path, sub_path, ref_path, out_path):
    """
    Merge LH (231-410), RH (51-230), subcortical (1-50) label volumes.
    Subcortical labels win at any overlap with cortical ribbon.
    """
    ref = nib.load(str(ref_path))
    lh  = nib.load(str(lh_path)).get_fdata().astype(np.int32)
    rh  = nib.load(str(rh_path)).get_fdata().astype(np.int32)
    sub = nib.load(str(sub_path)).get_fdata().astype(np.int32)

    target_shape = ref.shape[:3]
    for name, arr in [('lh', lh), ('rh', rh), ('sub', sub)]:
        if arr.shape[:3] != target_shape:
            raise ValueError(
                f'Shape mismatch: {name} has {arr.shape[:3]}, ref has {target_shape}')

    merged           = np.zeros(target_shape, dtype=np.int32)
    merged[lh  > 0]  = lh [lh  > 0]
    merged[rh  > 0]  = rh [rh  > 0]
    merged[sub > 0]  = sub[sub > 0]   

    nib.save(nib.Nifti1Image(merged, ref.affine, ref.header), str(out_path))


# ── STAGE 1: PARCELLATION IN DWI SPACE ────────────────────────────────────
def stage1_parcellation(subj_id: str, paths: dict, work: Path, log: logging.Logger) -> Path:
    out_parc = work / 'parcellation_410_dwi.nii.gz'
    if out_parc.exists():
        log.info('Stage 1: parcellation already exists — skipping')
        return out_parc

    t1       = paths['t1_brain']
    xfm      = paths['xfm_std2acpc']
    lh_lbl   = ATLAS_WORK / 'lh_atlas.label.gii'
    rh_lbl   = ATLAS_WORK / 'rh_atlas.label.gii'
    sub_mni  = ATLAS_WORK / 'subcortical_mni.nii.gz'

    # ── 1a-b. Project 32k atlas → Native surface → Native volume ────────────
    # We must go 32k -> native -> volume, because HCP surfaces are non-isomorphic.
    
    # Define the 32k atlas source labels
    lh_atlas_32k = ATLAS_WORK / 'lh_atlas.label.gii'
    rh_atlas_32k = ATLAS_WORK / 'rh_atlas.label.gii'
    
    # 1. Map 32k labels to Native surface using the native sphere
    lh_native_lbl = work / 'lh_native.label.gii'
    rh_native_lbl = work / 'rh_native.label.gii'
    
    # Current sphere is native, we need the sphere.reg that matches 32k
    # HCP structural data stores these in Native folder as .sphere.reg.surf.gii
    lh_sphere = STRUCT_ROOT / subj_id / 'T1w' / 'Native' / f'{subj_id}.L.sphere.reg.native.surf.gii'
    rh_sphere = STRUCT_ROOT / subj_id / 'T1w' / 'Native' / f'{subj_id}.R.sphere.reg.native.surf.gii'
    
    # The template sphere is the fsaverage_LR 32k sphere
    template_sphere = Path('/home/khushbu-lab4/HCPStructuralData/Atlas/fsaverage_LR32k/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii') 
    # (Note: ensure you have the 32k reference spheres if not in your folder)

    run([WB_CMD, '-label-resample', lh_atlas_32k, template_sphere, lh_sphere, 'ADAP_BARY_AREA', lh_native_lbl], 'Resample 32k→Native LH', log)
    run([WB_CMD, '-label-resample', rh_atlas_32k, template_sphere, rh_sphere, 'ADAP_BARY_AREA', rh_native_lbl], 'Resample 32k→Native RH', log)

    # 2. Map Native surface labels → T1 volume
    lh_vol = work / 'lh_cortex_acpc.nii.gz'
    rh_vol = work / 'rh_cortex_acpc.nii.gz'

    run([WB_CMD, '-label-to-volume-mapping',
         lh_native_lbl, paths['midthick_lh'], t1, lh_vol,
         '-ribbon-constrained', paths['white_lh'], paths['pial_lh']],
        'wb label-to-vol LH', log)

    run([WB_CMD, '-label-to-volume-mapping',
         rh_lbl, paths['midthick_rh'], t1, rh_vol,
         '-ribbon-constrained', paths['white_rh'], paths['pial_rh']],
        'wb label-to-vol RH', log)

    # ── 1c. Warp subcortical labels MNI → T1 acpc (nearest-neighbour) ─────
    sub_acpc = work / 'subcortical_acpc.nii.gz'
    run(['applywarp',
         f'--ref={t1}',
         f'--in={sub_mni}',
         f'--warp={xfm}',
         f'--out={sub_acpc}',
         '--interp=nn'],
        'applywarp subcortical MNI→acpc', log)

    # ── 1d. Merge → parcellation_410_acpc.nii.gz ──────────────────────────
    parc_acpc = work / 'parcellation_410_acpc.nii.gz'
    merge_parcellation(lh_vol, rh_vol, sub_acpc, t1, parc_acpc)
    labels_acpc = np.unique(nib.load(str(parc_acpc)).get_fdata().astype(int))
    log.info(f'Stage 1d: {(labels_acpc > 0).sum()} non-zero labels in acpc space')

    # ── 1e. Resample parcellation T1 acpc → DWI space via header transform ──
    # HCP DWI is already registered to T1 acpc; mrtransform reads headers
    # directly — no b0 extraction or FLIRT needed.
    run([mrt('mrtransform'),
         str(parc_acpc),
         '-template', str(paths['wmfod_norm']),
         str(out_parc),
         '-interp', 'nearest',
         '-force'],
        'mrtransform parcellation → DWI space (header-based)', log)

    labels_dwi = np.unique(nib.load(str(out_parc)).get_fdata().astype(int))
    labels_dwi = labels_dwi[labels_dwi > 0]
    log.info(f'Stage 1 QC: {len(labels_dwi)}/410 labels present in DWI space')
    
    if len(labels_dwi) < 350:
        log.warning(f'Stage 1 WARNING: only {len(labels_dwi)} labels — check header alignment')

    return out_parc


# ── STAGE 2: 5TT IN DWI SPACE ─────────────────────────────────────────────
def stage2_5tt(paths: dict, work: Path, log: logging.Logger) -> Path:
    fivett_dwi = work / '5tt_dwi.nii.gz'
    if fivett_dwi.exists():
        log.info('Stage 2: 5TT already exists — skipping')
        return fivett_dwi

    fivett_acpc = work / '5tt_acpc.mif'

    run([mrt('5ttgen'), 'fsl',
         str(paths['t1_brain']), str(fivett_acpc),
         '-premasked', '-nocrop', '-force',
         '-nthreads', str(THREADS_TCKGEN)],
        '5ttgen fsl (T1 acpc, premasked)', log)

    # mrtransform handles 4D data; trilinear is correct for probability maps
    run([mrt('mrtransform'),
         str(fivett_acpc),
         '-template', str(paths['wmfod_norm']),
         str(fivett_dwi),
         '-force'],
        'mrtransform 5TT → DWI space (header-based)', log)

    return fivett_dwi


# ── STAGE 3: GMWMI ────────────────────────────────────────────────────────
def stage3_gmwmi(work: Path, fivett_dwi: Path, log: logging.Logger) -> Path:
    gmwmi = work / 'gmwmi.mif'
    if gmwmi.exists():
        log.info('Stage 3: GMWMI already exists — skipping')
        return gmwmi
    run([mrt('5tt2gmwmi'), str(fivett_dwi), str(gmwmi), '-force'],
        '5tt2gmwmi', log)
    return gmwmi


# ── STAGE 4: TRACTOGRAPHY ─────────────────────────────────────────────────
def _tckgen_chunk(chunk_id: int, work: Path, wmfod: Path,
                  fivett: Path, gmwmi: Path) -> str:
    chunk = work / f'chunk_{chunk_id}.tck'
    if chunk.exists() and chunk.stat().st_size > 1_000_000:
        return str(chunk)
    cmd = [
        mrt('tckgen'), str(wmfod), str(chunk),
        '-act',          str(fivett),
        '-seed_gmwmi',   str(gmwmi),
        '-select',       str(TRACKS_PER_CHUNK),
        '-cutoff',       TCKGEN_CUTOFF,
        '-minlength',    TCKGEN_MINLEN,
        '-maxlength',    TCKGEN_MAXLEN,
        '-backtrack',
        '-crop_at_gmwmi',
        '-nthreads',     str(THREADS_TCKGEN),
        '-force', '-quiet',
    ]
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'tckgen chunk {chunk_id} failed:\n{result.stderr[-2000:]}')
    return str(chunk)


def stage4_tractography(subj_id: str, paths: dict, work: Path,
                         fivett_dwi: Path, gmwmi: Path,
                         log: logging.Logger) -> Path:
    final_tck = work / f'{subj_id}_10M.tck'
    if final_tck.exists() and final_tck.stat().st_size > 10_000_000:
        log.info('Stage 4: tractogram already exists — skipping')
        return final_tck

    total = CHUNKS_PER_SUBJ * TRACKS_PER_CHUNK
    log.info(f'Stage 4: {CHUNKS_PER_SUBJ} chunks × {TRACKS_PER_CHUNK:,} '
             f'= {total:,} streamlines  '
             f'({CHUNKS_PER_SUBJ} × {THREADS_TCKGEN} threads)')

    wmfod = paths['wmfod_norm']
    chunk_files = []

    with ThreadPoolExecutor(max_workers=CHUNKS_PER_SUBJ) as pool:
        futures = {
            pool.submit(_tckgen_chunk, i, work, wmfod, fivett_dwi, gmwmi): i
            for i in range(CHUNKS_PER_SUBJ)
        }
        for fut in as_completed(futures):
            cid  = futures[fut]
            path = fut.result()
            chunk_files.append(path)
            sz   = os.path.getsize(path) / 1e9
            log.info(f'  Chunk {cid} complete ({sz:.2f} GB)')

    run([mrt('tckedit')] + chunk_files + [str(final_tck), '-force', '-quiet'],
        'tckedit merge', log)

    tinfo = subprocess.run([mrt('tckinfo'), str(final_tck)],
                           capture_output=True, text=True)
    for line in tinfo.stdout.splitlines():
        if 'count' in line.lower():
            log.info(f'  Merged tractogram: {line.strip()}')

    for f in chunk_files:
        if os.path.exists(f):
            os.remove(f)

    return final_tck


# ── STAGE 5: SIFT2 ────────────────────────────────────────────────────────
def stage5_sift2(subj_id: str, paths: dict, work: Path,
                  final_tck: Path, fivett_dwi: Path,
                  log: logging.Logger) -> tuple[Path, Path]:
    weights = work / 'sift2_weights.txt'
    mu_file = work / 'sift2_mu.txt'

    if weights.exists() and weights.stat().st_size > 1_000_000:
        log.info('Stage 5: SIFT2 weights already exist — skipping')
        return weights, mu_file

    run([mrt('tcksift2'),
         str(final_tck),
         str(paths['wmfod_norm']),
         str(weights),
         '-act',      str(fivett_dwi),
         '-out_mu',   str(mu_file),
         '-nthreads', str(THREADS_SIFT2),
         '-force'],
        'tcksift2', log)

    if mu_file.exists():
        log.info(f'  SIFT2 mu = {mu_file.read_text().strip()}  [record in Methods]')

    return weights, mu_file


# ── STAGE 6: CONNECTOME EXTRACTION + QC ──────────────────────────────────
def stage6_connectome(subj_id: str, work: Path, final_tck: Path,
                       parc_dwi: Path, sift2_weights: Path,
                       log: logging.Logger) -> Path:
    out_sift2 = SC_OUT_DIR / f'{subj_id}_SC_SIFT2_410.csv'
    out_count = SC_OUT_DIR / f'{subj_id}_SC_count_410.csv'

    if out_sift2.exists():
        log.info('Stage 6: SC matrix already exists — skipping extraction')
    else:
        run([mrt('tck2connectome'),
             str(final_tck), str(parc_dwi), str(out_sift2),
             '-tck_weights_in', str(sift2_weights),
             '-symmetric', '-zero_diagonal',
             '-nthreads', str(THREADS_CONN), '-force'],
            'tck2connectome SIFT2-weighted', log)

        run([mrt('tck2connectome'),
             str(final_tck), str(parc_dwi), str(out_count),
             '-symmetric', '-zero_diagonal',
             '-nthreads', str(THREADS_CONN), '-force'],
            'tck2connectome streamline count', log)

    mat = np.loadtxt(str(out_sift2), delimiter=',')
    assert mat.shape == (N_REGIONS, N_REGIONS), f'Wrong shape: {mat.shape}'
    assert np.allclose(mat, mat.T, atol=1e-5), 'Matrix asymmetric'
    n_nz    = int(np.sum(mat > 0))
    density = n_nz / (N_REGIONS * (N_REGIONS - 1))
    log.info(f'SC matrix: shape={mat.shape}  non-zero={n_nz}  '
             f'density={density:.3f} ✓')

    if final_tck.exists():
        final_tck.unlink()
        log.info('Tractogram deleted (disk space reclaimed)')

    return out_sift2


# ── PER-SUBJECT ENTRY POINT ───────────────────────────────────────────────
def process_subject(subj_id: str, zyg: str, pair_id: str, gender: int) -> tuple:
    out_csv = SC_OUT_DIR / f'{subj_id}_SC_SIFT2_410.csv'
    if out_csv.exists():
        return subj_id, 'SKIPPED', 0.0

    log   = get_logger(subj_id)
    paths = subject_paths(subj_id, zyg, pair_id, int(gender))
    work  = SC_WORK_DIR / subj_id
    work.mkdir(exist_ok=True)
    t0    = time.time()

    try:
        log.info(f'━━━ SC pipeline: {subj_id}  ({zyg} {pair_id}) ━━━')
        log.info(f'DWI : {paths["wmfod_norm"]}')
        log.info(f'T1  : {paths["t1_brain"]}')

        preflight_check(paths, log)

        parc = stage1_parcellation(subj_id, paths, work, log)
        fivett  = stage2_5tt(paths, work, log)
        gmwmi   = stage3_gmwmi(work, fivett, log)
        tck     = stage4_tractography(subj_id, paths, work, fivett, gmwmi, log)
        wts, _  = stage5_sift2(subj_id, paths, work, tck, fivett, log)
        stage6_connectome(subj_id, work, tck, parc, wts, log)

        elapsed = time.time() - t0
        log.info(f'━━━ DONE in {timedelta(seconds=int(elapsed))} ━━━')
        return subj_id, 'OK', elapsed

    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f'FAILED: {exc}\n{traceback.format_exc()}')
        return subj_id, f'FAILED: {exc}', elapsed


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='HCP-S1200 SC pipeline (410 nodes)')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Run first subject only')
    parser.add_argument('--subject', default=None,
                        help='Run one specific subject ID')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()])
    main_log = logging.getLogger('SC.main')

    # Subject table
    df = pd.read_csv(str(SUBJ_CSV))
    df = df[df['TwinPairID'] != 'Pair41'].dropna(subset=['Subject'])
    df['subj_id'] = df['Subject'].apply(lambda x: str(int(float(x))))
    df['gender']  = df['Gender'].astype(int)
    subjects = df[['subj_id', 'ZygosityGT1', 'TwinPairID', 'gender']].values.tolist()
    main_log.info(f'Loaded {len(subjects)} subjects')

    setup_atlas_once(main_log)

    if args.subject:
        subjects = [r for r in subjects if r[0] == args.subject]
        if not subjects:
            main_log.error(f'Subject {args.subject} not found in table')
            sys.exit(1)
    elif args.dry_run:
        subjects = subjects[:1]
        main_log.info(f'DRY RUN: {subjects[0][0]} only')

    already_done = [r for r in subjects
                    if (SC_OUT_DIR / f'{r[0]}_SC_SIFT2_410.csv').exists()]
    to_run = [r for r in subjects if r not in already_done]
    main_log.info(f'To run: {len(to_run)}   Already done: {len(already_done)}')

    results = []

    if len(to_run) <= 1:
        for row in to_run:
            r = process_subject(*row)
            results.append(r)
            main_log.info(f'{r[0]}  {r[1]}  '
                          f'{timedelta(seconds=int(r[2]))}')
    else:
        n_total = len(to_run)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(process_subject, *row): row[0]
                for row in to_run
            }
            for fut in as_completed(futures):
                subj_id, status, elapsed = fut.result()
                results.append((subj_id, status, elapsed))
                n_ok   = sum(1 for _, s, _ in results if s == 'OK')
                n_fail = sum(1 for _, s, _ in results if s.startswith('FAILED'))
                main_log.info(
                    f'[{len(results):>3}/{n_total}] {subj_id}  {status:<12}  '
                    f'{timedelta(seconds=int(elapsed))}  '
                    f'(ok={n_ok} fail={n_fail})')

    for row in already_done:
        results.append((row[0], 'SKIPPED', 0.0))

    # ── Audit ──────────────────────────────────────────────────────────────
    ok_r   = [r for r in results if r[1] == 'OK']
    skip_r = [r for r in results if r[1] == 'SKIPPED']
    fail_r = [r for r in results if r[1].startswith('FAILED')]

    main_log.info('─' * 65)
    main_log.info('BATCH AUDIT')
    main_log.info('─' * 65)
    main_log.info(f'  OK      : {len(ok_r)}')
    main_log.info(f'  Skipped : {len(skip_r)}')
    main_log.info(f'  Failed  : {len(fail_r)}')

    if ok_r:
        times = [r[2] for r in ok_r]
        main_log.info(
            f'  Time    : mean={timedelta(seconds=int(np.mean(times)))}  '
            f'total≈{timedelta(seconds=int(np.sum(times)/N_WORKERS))} wall')

    if fail_r:
        main_log.error('  FAILED:')
        for subj_id, status, _ in fail_r:
            main_log.error(f'    {subj_id}: {status}')

    bad = []
    for subj_id, status, _ in results:
        if status in ('OK', 'SKIPPED'):
            p = SC_OUT_DIR / f'{subj_id}_SC_SIFT2_410.csv'
            if p.exists():
                mat = np.loadtxt(str(p), delimiter=',')
                if mat.shape != (N_REGIONS, N_REGIONS):
                    bad.append((subj_id, mat.shape))
    n_checked = sum(1 for _, s, _ in results
                    if s in ('OK', 'SKIPPED') and
                    (SC_OUT_DIR / f'{_}_SC_SIFT2_410.csv').exists())

    if bad:
        main_log.error(f'  BAD SHAPES: {bad}')
    else:
        main_log.info(f'  All {n_checked} matrices are {N_REGIONS}×{N_REGIONS} ✓')

    main_log.info('─' * 65)
    main_log.info(f'Outputs → {SC_OUT_DIR}')
    main_log.info(f'Logs    → {SC_LOG_DIR}')
    main_log.info('Done.')

if __name__ == '__main__':
    main()