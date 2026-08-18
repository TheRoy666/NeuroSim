#!/usr/bin/env bash
###############################################################################
# NeuroSim UNAM_TLE — preprocessing orchestration (T1 + rs-fMRI only)
#
# NO DTI in this dataset -> NO QSIPrep, NO MRtrix stage. Structural prior
# comes from a fixed group-average template SC (ENIGMA 'aparc', classic
# Desikan-Killiany, 68 cortical regions) shared across all subjects.
#
# Output space: MNI152NLin6Asym (matches the neurodata/neuroparc DK atlas
# natively -- NO atlas-warping step needed, unlike the ADNI pipeline).
#
# Sternberg task fMRI is present in this dataset but NOT processed here
# (--task-id resting only). Task files are left untouched on disk, not
# deleted -- may be useful for a future task-evoked state definition.
#
# Stages:
#   STAGE 1  FastSurfer   surface recon (CPU)
#   STAGE 2  fMRIPrep     BOLD only, task-id=resting, MNI152NLin6Asym output
###############################################################################

set -uo pipefail

# ─────────────────────────── CONFIG ────────────────────────────────────────
# All paths must be supplied via environment variables.
BASE="${BASE:?Set BASE to your UNAM_TLE data root}"
BIDS="${BIDS:-$BASE/UNAM_TLE_raw}"
SIF_DIR="${SIF_DIR:?Set SIF_DIR to the directory containing your Singularity images}"
FS_LICENSE="${FS_LICENSE:?Set FS_LICENSE to the path of your FreeSurfer license.txt}"

FASTSURFER_SIF="$SIF_DIR/fastsurfer-latest.sif"
FMRIPREP_SIF="$SIF_DIR/fmriprep_23.2.3.sif"

FS_DIR="$BASE/derivatives/freesurfer"
FMRIPREP_OUT="$BASE/derivatives/fmriprep"
WORK="$BASE/work"
LOGS="$BASE/logs"

# 46 rs-fMRI-flagged UNAM_TLE subjects (fill in from participants.tsv
# rs_analysis_fc_conition==1 filter, per the earlier cohort audit)
SUBJECTS="${UNAM_SUBJECTS:?Set UNAM_SUBJECTS to the space-separated subject ID list}"

# ─────────────────── CONCURRENCY (overridable via env vars) ────────────────
# Defaults below are BATCH-safe (many subjects in parallel on 80c/256GB).
# For single-subject test runs, override to use most of the machine:
#   FS_THREADS=72 FMRIPREP_NThreads=72 FMRIPREP_MEM=230000 ./run_unam_pipeline.sh test
FS_JOBS="${FS_JOBS:-16}";              FS_THREADS="${FS_THREADS:-4}"
FMRIPREP_JOBS="${FMRIPREP_JOBS:-10}";  FMRIPREP_NThreads="${FMRIPREP_NThreads:-8}"
FMRIPREP_MEM="${FMRIPREP_MEM:-12000}"
# No DTI stage -> more headroom than ADNI's 8-job fMRIPrep concurrency.

mkdir -p "$FS_DIR" "$FMRIPREP_OUT" "$WORK" "$LOGS"

# ─────────────────────────── PRE-FLIGHT ────────────────────────────────────
preflight() {
  echo "=== PRE-FLIGHT ==="
  local ok=1
  [[ -f "$FASTSURFER_SIF" ]] && echo "  ✓ FastSurfer sif" || { echo "  ✗ MISSING: $FASTSURFER_SIF"; ok=0; }
  [[ -f "$FMRIPREP_SIF"   ]] && echo "  ✓ fMRIPrep sif"   || { echo "  ✗ MISSING: $FMRIPREP_SIF"; ok=0; }
  [[ -f "$FS_LICENSE"     ]] && echo "  ✓ FS license"     || { echo "  ✗ MISSING: $FS_LICENSE"; ok=0; }
  [[ -d "$BIDS"           ]] && echo "  ✓ BIDS dataset"   || { echo "  ✗ MISSING: $BIDS"; ok=0; }
  command -v parallel     >/dev/null && echo "  ✓ GNU parallel" || { echo "  ✗ missing parallel"; ok=0; }
  command -v singularity  >/dev/null && echo "  ✓ singularity"  || { echo "  ✗ missing singularity"; ok=0; }
  echo "  Subjects: $(echo $SUBJECTS | wc -w)"
  echo "  Output space: MNI152NLin6Asym (matches DK atlas natively)"
  echo "  Task filter: rest ONLY (Sternberg tfMRI left unprocessed, not deleted)"
  [[ $ok -eq 1 ]] || { echo "PRE-FLIGHT FAILED"; exit 1; }
  echo "=== OK ==="; echo
}

# ─────────────────────────── STAGE 1: FastSurfer ───────────────────────────
run_fastsurfer() {
  local sub=$1
  local done_flag="$FS_DIR/sub-${sub}/scripts/recon-all.done"
  [[ -f "$done_flag" ]] && { echo "[FS] sub-$sub already done, skip"; return 0; }

  echo "[FS] START sub-$sub  $(date +%H:%M:%S)"
  singularity exec --cleanenv \
    -B "$BIDS":/data:ro \
    -B "$FS_DIR":/output \
    -B "$FS_LICENSE":/fs_license/license.txt:ro \
    "$FASTSURFER_SIF" \
    /fastsurfer/run_fastsurfer.sh \
      --t1 /data/sub-${sub}/anat/sub-${sub}_T1w.nii.gz \
      --sid sub-${sub} --sd /output \
      --fs_license /fs_license/license.txt \
      --parallel --threads $FS_THREADS \
      --device cpu --3T \
    > "$LOGS/fastsurfer_sub-${sub}.log" 2>&1 \
    && echo "[FS] DONE  sub-$sub  $(date +%H:%M:%S)" \
    || echo "[FS] FAIL  sub-$sub  (see log)"
}
export -f run_fastsurfer
export FASTSURFER_SIF BIDS FS_DIR FS_LICENSE FS_THREADS LOGS

# ─────────────────────────── STAGE 2: fMRIPrep (BOLD only) ─────────────────
run_fmriprep() {
  local sub=$1
  local done_flag="$FMRIPREP_OUT/sub-${sub}.html"
  [[ -f "$done_flag" ]] && { echo "[FMRI] sub-$sub already done, skip"; return 0; }

  echo "[FMRI] START sub-$sub  $(date +%H:%M:%S)"
  singularity run --cleanenv \
    -B "$BIDS":/data:ro \
    -B "$FMRIPREP_OUT":/out \
    -B "$WORK":/work \
    -B "$FS_DIR":/fsdir \
    -B "$FS_LICENSE":/fs_license/license.txt:ro \
    "$FMRIPREP_SIF" \
    /data /out participant \
      --participant-label "$sub" \
      --task-id resting \
      --fs-subjects-dir /fsdir \
      --output-spaces MNI152NLin6Asym:res-2 fsaverage5 \
      --fs-license-file /fs_license/license.txt \
      --work-dir /work \
      --nthreads $FMRIPREP_NThreads \
      --mem-mb $FMRIPREP_MEM \
      --skip-bids-validation \
      --notrack \
      --stop-on-first-crash \
    > "$LOGS/fmriprep_sub-${sub}.log" 2>&1 \
    && echo "[FMRI] DONE  sub-$sub  $(date +%H:%M:%S)" \
    || echo "[FMRI] FAIL  sub-$sub  (see log)"
}
export -f run_fmriprep
export FMRIPREP_SIF BIDS FMRIPREP_OUT WORK FS_DIR FS_LICENSE \
       FMRIPREP_NThreads FMRIPREP_MEM LOGS

# ─────────────────────────── ORCHESTRATION ─────────────────────────────────
preflight

STAGE=${1:-all}   # usage: ./run_unam_pipeline.sh [all|fs|fmri|test]

case "$STAGE" in
  test)
    echo "### TEST — first subject only (both stages) ###"
    first_sub=$(echo $SUBJECTS | awk '{print $1}')
    run_fastsurfer "$first_sub"
    run_fmriprep "$first_sub"
    echo "Inspect: $FMRIPREP_OUT/sub-${first_sub}.html"
    echo "Check for fsaverage5 surface output:"
    echo "  find $FMRIPREP_OUT/sub-${first_sub} -name '*fsaverage5*func.gii'"
    exit 0
    ;;

  fs|all)
    echo "############ STAGE 1: FastSurfer ($FS_JOBS concurrent) ############"
    echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
      | parallel -j $FS_JOBS --bar --joblog "$LOGS/fastsurfer.joblog" \
        run_fastsurfer {}
    echo "STAGE 1 complete: $(date)"
    [[ "$STAGE" == "fs" ]] && exit 0
    ;;&

  fmri|all)
    echo "############ STAGE 2: fMRIPrep ($FMRIPREP_JOBS concurrent, task=rest only) ############"
    echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
      | parallel -j $FMRIPREP_JOBS --bar --joblog "$LOGS/fmriprep.joblog" \
        run_fmriprep {}
    echo "STAGE 2 complete: $(date)"
    ;;
  *)
    echo "Unknown stage: $STAGE  (use: all|fs|fmri)"; exit 1
    ;;
esac

echo
echo "############ COMPLETION REPORT ############"
n_fs=$(ls -d "$FS_DIR"/sub-*/scripts/recon-all.done 2>/dev/null | wc -l)
n_fmri=$(ls "$FMRIPREP_OUT"/sub-*.html 2>/dev/null | wc -l)
echo "FastSurfer complete: $n_fs / $(echo $SUBJECTS | wc -w)"
echo "fMRIPrep   complete: $n_fmri / $(echo $SUBJECTS | wc -w)"
echo "Done: $(date)"
# (MRIQC is run separately, see run_unam_mriqc.sh -- not part of this
#  orchestration since it should complete and be reviewed BEFORE fMRIPrep)