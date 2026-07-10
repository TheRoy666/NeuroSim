###############################################################################
# NeuroSim ADNI3 — full preprocessing orchestration (CPU-only, R740)
#
# Hardware: 80 cores / 160 threads, 256 GB RAM, NO GPU
# Subjects: 50 QC-passing (25 CN / 25 MCI)
#
# Pipeline (staged, in dependency order):
#   STAGE 1  FastSurfer   surface recon (CPU)        — anatomical
#   STAGE 2  fMRIPrep     BOLD, reuses FastSurfer    — --fs-no-reconall
#   STAGE 3  QSIPrep      DWI preprocessing          — parallel with Stage 2
###############################################################################

set -uo pipefail

# CONFIG
# All paths must be supplied via environment variables, e.g.:
#   BASE=/data/adni SIF_DIR=/opt/containers FS_LICENSE=/opt/fs/license.txt \
#       ./run_adni_pipeline.sh
BASE="${BASE:?Set BASE to your ADNI data root, e.g. /data/NeuroSim_ADNIData}"
BIDS="${BIDS:-$BASE/BIDS_Dataset}"
SIF_DIR="${SIF_DIR:?Set SIF_DIR to the directory containing your Singularity images}"

FASTSURFER_SIF=$SIF_DIR/fastsurfer-latest.sif
FMRIPREP_SIF=$SIF_DIR/fmriprep_23.2.3.sif
QSIPREP_SIF=$SIF_DIR/qsiprep-latest.sif

# Outputs
FS_DIR=$BASE/derivatives/freesurfer        # FastSurfer writes here
FMRIPREP_OUT=$BASE/derivatives/fmriprep
QSIPREP_OUT=$BASE/derivatives/qsiprep
WORK=$BASE/work
LOGS=$BASE/logs

# FreeSurfer license (REQUIRED by both FastSurfer and fMRIPrep)
FS_LICENSE="${FS_LICENSE:?Set FS_LICENSE to the path of your FreeSurfer license.txt}"

# Subject list (space-separated IDs without 'sub-')
SUBJECTS="002S0413 006S0498 006S0731 018S4313 018S4400 021S0337 029S4290 \
041S4200 041S4427 041S5100 041S5253 053S4578 057S0934 057S5292 068S0127 \
068S0210 068S0473 068S4340 068S4424 070S5040 099S4076 100S0069 128S0272 \
129S4396 135S4598 011S4893 013S4268 022S2263 031S2233 032S2119 094S2201 \
128S2130 129S2332 130S4415 141S4160 002S4229 002S4654 019S4293 041S4510 \
068S4061 126S4507 127S4197 003S1122 018S2180 037S4214 041S0679 126S4514 \
127S2234 128S0200 135S4489"


FS_JOBS=16;        FS_THREADS=4
FMRIPREP_JOBS=8;   FMRIPREP_NThreads=8;  FMRIPREP_MEM=12000   # MB per subject
QSIPREP_JOBS=6;    QSIPREP_NThreads=8;   QSIPREP_MEM=15

mkdir -p "$FS_DIR" "$FMRIPREP_OUT" "$QSIPREP_OUT" "$WORK" "$LOGS"

# PRE-FLIGHT CHECKS
preflight() {
  echo "=== PRE-FLIGHT CHECKS ==="
  local ok=1
  for sif in "$FASTSURFER_SIF" "$FMRIPREP_SIF" "$QSIPREP_SIF"; do
    [[ -f "$sif" ]] && echo "  ✓ $sif" || { echo "  ✗ MISSING: $sif"; ok=0; }
  done
  [[ -f "$FS_LICENSE" ]] && echo "  ✓ FS license: $FS_LICENSE" \
      || { echo "  ✗ MISSING FS license: $FS_LICENSE"; ok=0; }
  [[ -d "$BIDS" ]] && echo "  ✓ BIDS: $BIDS" || { echo "  ✗ MISSING BIDS"; ok=0; }
  command -v parallel >/dev/null && echo "  ✓ GNU parallel" \
      || { echo "  ✗ GNU parallel missing"; ok=0; }
  command -v singularity >/dev/null && echo "  ✓ singularity" \
      || { echo "  ✗ singularity missing"; ok=0; }
  echo "  Subjects: $(echo $SUBJECTS | wc -w)"
  echo "  Free RAM: $(free -g | awk '/^Mem:/{print $7}') GB available"
  [[ $ok -eq 1 ]] || { echo "PRE-FLIGHT FAILED. Fix above and re-run."; exit 1; }
  echo "=== PRE-FLIGHT OK ==="; echo
}

# STAGE 1: FastSurfer
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
      --t1 /data/sub-${sub}/ses-bl/anat/sub-${sub}_ses-bl_T1w.nii.gz \
      --sid sub-${sub} --sd /output \
      --fs_license /fs_license/license.txt \
      --parallel --threads $FS_THREADS \
      --device cpu \
      --3T \
    > "$LOGS/fastsurfer_sub-${sub}.log" 2>&1 \
    && echo "[FS] DONE  sub-$sub  $(date +%H:%M:%S)" \
    || echo "[FS] FAIL  sub-$sub  (see log)"
}
export -f run_fastsurfer
export FASTSURFER_SIF BIDS FS_DIR FS_LICENSE FS_THREADS LOGS

# STAGE 2: fMRIPrep
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
      --task-id rest \
      --fs-subjects-dir /fsdir \
      --fs-no-reconall \
      --output-spaces MNI152NLin2009cAsym:res-2 \
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

# STAGE 3: QSIPrep
run_qsiprep() {
  local sub=$1
  local done_flag="$QSIPREP_OUT/sub-${sub}.html"
  [[ -f "$done_flag" ]] && { echo "[QSI] sub-$sub already done, skip"; return 0; }

  echo "[QSI] START sub-$sub  $(date +%H:%M:%S)"
  singularity run --cleanenv \
    -B "$BIDS":/data:ro \
    -B "$QSIPREP_OUT":/out \
    -B "$WORK":/work \
    -B "$FS_LICENSE":/fs_license/license.txt:ro \
    "$QSIPREP_SIF" \
    /data /out participant \
      --participant-label "$sub" \
      --output-resolution 2.0 \
      --fs-license-file /fs_license/license.txt \
      --work-dir /work \
      --nthreads $QSIPREP_NThreads \
      --mem-mb $((QSIPREP_MEM * 1000)) \
      --skip-bids-validation \
      --notrack \
      --stop-on-first-crash \
    > "$LOGS/qsiprep_sub-${sub}.log" 2>&1 \
    && echo "[QSI] DONE  sub-$sub  $(date +%H:%M:%S)" \
    || echo "[QSI] FAIL  sub-$sub  (see log)"
}
export -f run_qsiprep
export QSIPREP_SIF BIDS QSIPREP_OUT WORK FS_LICENSE \
       QSIPREP_NThreads QSIPREP_MEM LOGS

# ORCHESTRATION
preflight

STAGE=${1:-all}   # usage: ./run_adni_pipeline.sh [all|fs|fmri|qsi|bold]

case "$STAGE" in
  fs|all)
    echo "############ STAGE 1: FastSurfer ($FS_JOBS concurrent) ############"
    echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
      | parallel -j $FS_JOBS --bar --joblog "$LOGS/fastsurfer.joblog" \
        run_fastsurfer {}
    echo "STAGE 1 complete: $(date)"
    [[ "$STAGE" == "fs" ]] && exit 0
    ;;&

  fmri|qsi|bold|all)
    # Stages 2 (BOLD) and 3 (DWI) run CONCURRENTLY — independent inputs.
    # Launch both parallel pools in the background, wait for both.
    echo "##### STAGE 2+3: fMRIPrep ($FMRIPREP_JOBS) + QSIPrep ($QSIPREP_JOBS) concurrent #####"

    if [[ "$STAGE" == "all" || "$STAGE" == "fmri" || "$STAGE" == "bold" ]]; then
      ( echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
          | parallel -j $FMRIPREP_JOBS --bar --joblog "$LOGS/fmriprep.joblog" \
            run_fmriprep {} ) &
      FMRI_PID=$!
    fi

    if [[ "$STAGE" == "all" || "$STAGE" == "qsi" ]]; then
      ( echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
          | parallel -j $QSIPREP_JOBS --bar --joblog "$LOGS/qsiprep.joblog" \
            run_qsiprep {} ) &
      QSI_PID=$!
    fi

    [[ -n "${FMRI_PID:-}" ]] && wait $FMRI_PID && echo "fMRIPrep pool done"
    [[ -n "${QSI_PID:-}"  ]] && wait $QSI_PID  && echo "QSIPrep pool done"
    echo "STAGE 2+3 complete: $(date)"
    ;;

  *)
    echo "Unknown stage: $STAGE  (use: all|fs|fmri|qsi)"; exit 1
    ;;
esac

# COMPLETION REPORT
echo
echo "############ COMPLETION REPORT ############"
n_fs=$(ls -d "$FS_DIR"/sub-*/scripts/recon-all.done 2>/dev/null | wc -l)
n_fmri=$(ls "$FMRIPREP_OUT"/sub-*.html 2>/dev/null | wc -l)
n_qsi=$(ls "$QSIPREP_OUT"/sub-*.html 2>/dev/null | wc -l)
echo "FastSurfer complete: $n_fs / 50"
echo "fMRIPrep   complete: $n_fmri / 50"
echo "QSIPrep    complete: $n_qsi / 50"
echo "Failures logged in: $LOGS/*.joblog (exitval != 0)"
echo "Done: $(date)"
