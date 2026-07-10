#!/usr/bin/env bash
###############################################################################
# NeuroSim ADNI — MRtrix3 structural connectome builder (v3, 5ttgen fsl fix)
#
# PIPELINE:
#   mrconvert → dwi2response tournier → dwi2fod csd (L_max=4) → mtnormalise
#   → 5ttgen fsl (ACPC T1w) → 5tt2gmwmi → tckgen ACT 10M
#   → tcksift2 → atlas warp (MNI→ACPC) → tck2connectome × 2 atlases
###############################################################################

set -uo pipefail

#  TOOLCHAIN
# Must be supplied via environment, e.g.: MRTRIX_BIN=/opt/mrtrix3/bin ./build_connectomes.sh
MRTRIX_BIN="${MRTRIX_BIN:?Set MRTRIX_BIN to your MRtrix3 bin directory (v3.0.8+)}"
export PATH="$MRTRIX_BIN:$PATH"

# CONFIG
BASE="${BASE:?Set BASE to your ADNI data root, e.g. /data/NeuroSim_ADNIData}"
QSIPREP="${QSIPREP:-$BASE/derivatives/qsiprep}"
SC_OUT="${SC_OUT:-$BASE/derivatives/connectomes}"
LOGS="${LOGS:-$BASE/logs/connectome}"

# Atlas paths: TianS3 (450-node, primary) and Schaefer400 (400-node, secondary).
# See README for atlas provenance and MNI152NLin2009cAsym space verification.
ATLAS_TIANS3="${ATLAS_TIANS3:?Set ATLAS_TIANS3 to the Schaefer400+TianS3 NIfTI path}"
ATLAS_SCHAEFER="${ATLAS_SCHAEFER:?Set ATLAS_SCHAEFER to the Schaefer400 NIfTI path (e.g. from templateflow)}"
TIANS3_NODES=450
SCHAEFER_NODES=400

LMAX=4
N_STREAMLINES=10000000
THREADS_PER_SUBJECT=12
N_JOBS=6

SUBJECTS="002S0413 002S4229 002S4654 003S1122 006S0498 006S0731 011S4893 \
013S4268 018S2180 018S4313 018S4400 019S4293 021S0337 022S2263 029S4290 \
032S2119 037S4214 041S0679 041S4200 041S4427 041S4510 041S5100 041S5253 \
053S4578 057S0934 057S5292 068S0127 068S0210 068S0473 068S4061 068S4340 \
068S4424 070S5040 094S2201 099S4076 100S0069 126S4507 126S4514 127S2234 \
127S4197 128S0200 128S0272 128S2130 129S2332 129S4396 130S4415 135S4489 \
135S4598 141S4160"

mkdir -p "$SC_OUT" "$LOGS"

#  PRE-FLIGHT
preflight() {
  echo "=== PRE-FLIGHT ==="
  local ok=1
  for tool in mrconvert dwi2response dwi2fod mtnormalise 5ttgen 5tt2gmwmi \
              tckgen tcksift2 tck2connectome mrcalc; do
    [[ -x "$MRTRIX_BIN/$tool" ]] && echo "  ✓ $tool" \
        || { echo "  ✗ MISSING: $tool"; ok=0; }
  done
  command -v antsApplyTransforms &>/dev/null && \
      echo "  ✓ antsApplyTransforms ($(which antsApplyTransforms))" \
      || { echo "  ✗ MISSING: antsApplyTransforms"; ok=0; }
  command -v fsl5.0-fast &>/dev/null || command -v fast &>/dev/null && \
      echo "  ✓ FSL (fast available)" \
      || echo "  ⚠ FSL not found on PATH — 5ttgen fsl will fail"
  for f in "$ATLAS_TIANS3" "$ATLAS_SCHAEFER"; do
    [[ -f "$f" ]] && echo "  ✓ atlas: $(basename $f)" \
        || { echo "  ✗ MISSING atlas: $f"; ok=0; }
  done
  [[ -d "$QSIPREP" ]] && echo "  ✓ QSIPrep" || { echo "  ✗ no QSIPrep"; ok=0; }
  echo "  /dev/shm free: $(df -h /dev/shm | awk 'NR==2{print $4}')"
  echo "  5ttgen: fsl (ACPC T1w — correct space)"
  echo "  L_max=$LMAX  N_STREAMLINES=$N_STREAMLINES  jobs=$N_JOBS×${THREADS_PER_SUBJECT}thr"
  [[ $ok -eq 1 ]] || { echo "PRE-FLIGHT FAILED"; exit 1; }
  echo "=== OK ==="; echo
}

# PER-SUBJECT
build_one() {
  local sub=$1
  local log="$LOGS/connectome_sub-${sub}.log"
  local done_flag="$SC_OUT/sub-${sub}_tians3_connectome.csv"
  local saved_tck="$SC_OUT/sub-${sub}_tracks_10M.tck"
  local saved_w="$SC_OUT/sub-${sub}_sift2_weights.txt"

  if [[ -f "$done_flag" && -f "$saved_tck" ]]; then
    echo "[SC] sub-$sub done, skip"; return 0
  fi

  local shm="/dev/shm/${sub}_sc"
  mkdir -p "$shm"
  echo "[SC] START sub-$sub $(date +%H:%M:%S)"

  {
    set -e

    # Locate QSIPrep outputs
    local dwi=$(find "$QSIPREP/sub-${sub}" -name "*desc-preproc_dwi.nii.gz" | head -1)
    local bval=$(find "$QSIPREP/sub-${sub}" -name "*desc-preproc_dwi.bval" | head -1)
    local bvec=$(find "$QSIPREP/sub-${sub}" -name "*desc-preproc_dwi.bvec" | head -1)
    local mask=$(find "$QSIPREP/sub-${sub}" -name "*desc-brain_mask.nii.gz" \
                 | grep -i dwi | head -1)
    [[ -z "$mask" ]] && \
        mask=$(find "$QSIPREP/sub-${sub}" -name "*desc-brain_mask.nii.gz" | head -1)

    # ACPC T1w — used by 5ttgen fsl
    local t1_acpc="$QSIPREP/sub-${sub}/anat/sub-${sub}_space-ACPC_desc-preproc_T1w.nii.gz"
    [[ -f "$t1_acpc" ]] || { echo "FAIL: no ACPC T1w"; exit 1; }

    # MNI→ACPC transform for atlas warping
    local xfm="$QSIPREP/sub-${sub}/anat/sub-${sub}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"
    [[ -f "$xfm" ]] || { echo "FAIL: no MNI→ACPC transform"; exit 1; }

    [[ -z "$dwi" ]] && { echo "FAIL: no preprocessed DWI"; exit 1; }

    # Reuse saved tracks if available
    if [[ -f "$saved_tck" && -f "$saved_w" ]]; then
      echo "  Reusing saved tracks (skipping tckgen/sift2)"
      cp "$saved_tck" "$shm/tracks_10M.tck"
      cp "$saved_w"   "$shm/sift2_weights.txt"
    else

      # 1. DWI → .mif
      mrconvert "$dwi" "$shm/dwi.mif" \
          -fslgrad "$bvec" "$bval" -force -nthreads $THREADS_PER_SUBJECT

      # 2. Response function (tournier: single-shell single-tissue)
      dwi2response tournier "$shm/dwi.mif" "$shm/wm.txt" \
          -mask "$mask" -force -nthreads $THREADS_PER_SUBJECT

      # 3. FOD
      dwi2fod csd "$shm/dwi.mif" "$shm/wm.txt" "$shm/wmfod.mif" \
          -lmax $LMAX -mask "$mask" -force -nthreads $THREADS_PER_SUBJECT

      # 4. Intensity normalisation
      mtnormalise "$shm/wmfod.mif" "$shm/wmfod_norm.mif" \
          -mask "$mask" -force

      # 5. 5TT from QSIPrep ACPC T1w via FSL FAST
      # FSL FAST segments tissue from T1w intensity → robust on elderly.
      5ttgen fsl "$t1_acpc" "$shm/5tt.mif" -force
      5tt2gmwmi "$shm/5tt.mif" "$shm/gmwmi.mif" -force

      # 6. Tractography ACT 10M
      tckgen "$shm/wmfod_norm.mif" "$shm/tracks_10M.tck" \
          -act "$shm/5tt.mif" -backtrack -crop_at_gmwmi \
          -seed_gmwmi "$shm/gmwmi.mif" \
          -select $N_STREAMLINES \
          -maxlength 250 -minlength 5 -cutoff 0.06 \
          -nthreads $THREADS_PER_SUBJECT -force

      # 7. SIFT2
      tcksift2 "$shm/tracks_10M.tck" "$shm/wmfod_norm.mif" \
          "$shm/sift2_weights.txt" \
          -act "$shm/5tt.mif" \
          -nthreads $THREADS_PER_SUBJECT -force

      cp "$shm/tracks_10M.tck"    "$saved_tck"
      cp "$shm/sift2_weights.txt" "$saved_w"

    fi  # end tractography block

    # 8. Atlas warp MNI → ACPC + connectome
    for pair in "tians3:$ATLAS_TIANS3:$TIANS3_NODES" \
                "schaefer400:$ATLAS_SCHAEFER:$SCHAEFER_NODES"; do
      name="${pair%%:*}"; rest="${pair#*:}"
      atlas="${rest%%:*}"; nodes="${rest##*:}"

      # Warp atlas MNI → ACPC (nearest-neighbour: never interpolate labels)
      antsApplyTransforms -d 3 \
          -i "$atlas" \
          -r "$t1_acpc" \
          -o "$shm/atlas_${name}_acpc.nii.gz" \
          -n NearestNeighbor \
          -t "$xfm"

      # Sanitize + cast uint32
      mrcalc "$shm/atlas_${name}_acpc.nii.gz" 0 -max $nodes -min \
          "$shm/atlas_${name}.mif" -force
      mrconvert "$shm/atlas_${name}.mif" "$shm/atlas_${name}_u32.mif" \
          -datatype uint32 -force

      # Connectome (atlas and tracks both in ACPC space)
      tck2connectome "$shm/tracks_10M.tck" "$shm/atlas_${name}_u32.mif" \
          "$SC_OUT/sub-${sub}_${name}_connectome.csv" \
          -tck_weights_in "$shm/sift2_weights.txt" \
          -scale_invnodevol -symmetric -zero_diagonal \
          -nthreads $THREADS_PER_SUBJECT -force
    done

    echo "[SC] DONE sub-$sub $(date +%H:%M:%S)"
  } > "$log" 2>&1

  rm -rf "$shm"

  [[ -f "$done_flag" ]] \
      && echo "[SC] ✓ sub-$sub" \
      || echo "[SC] ✗ sub-$sub FAILED (see $log)"
}
export -f build_one
export MRTRIX_BIN SC_OUT LOGS QSIPREP LMAX N_STREAMLINES \
       THREADS_PER_SUBJECT ATLAS_TIANS3 ATLAS_SCHAEFER \
       TIANS3_NODES SCHAEFER_NODES PATH

# RUN
preflight

MODE=${1:-all}

case "$MODE" in
  test)
    # Test on 021S0337 — a known bad subject, best stress-test of the fix
    echo "### TEST — sub-021S0337 (previously 64% empty with both hsvs+freesurfer) ###"
    build_one 021S0337
    echo ""
    echo "Check empty nodes:"
    grep "nodes do not have" "$LOGS/connectome_sub-021S0337.log" | head -2
    grep "seeds\|100%" "$LOGS/connectome_sub-021S0337.log" | tail -3
    ;;

  reconnect)
    echo "### RECONNECT — reusing tracks, rebuilding connectomes only ###"
    for sub in $SUBJECTS; do
      rm -f "$SC_OUT/sub-${sub}_tians3_connectome.csv" \
            "$SC_OUT/sub-${sub}_schaefer400_connectome.csv"
    done
    echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
      | parallel -j $N_JOBS --bar --joblog "$LOGS/reconnect.joblog" build_one {}
    ;;

  all)
    echo "### Building all: $N_JOBS × $THREADS_PER_SUBJECT threads, L_max=$LMAX ###"
    echo "$SUBJECTS" | tr ' ' '\n' | grep -v '^$' \
      | parallel -j $N_JOBS --bar --joblog "$LOGS/connectome.joblog" build_one {}
    echo
    echo "### COMPLETION ###"
    n_t=$(ls "$SC_OUT"/sub-*_tians3_connectome.csv 2>/dev/null | wc -l)
    n_s=$(ls "$SC_OUT"/sub-*_schaefer400_connectome.csv 2>/dev/null | wc -l)
    echo "TianS3 (450):   $n_t / $(echo $SUBJECTS | wc -w)"
    echo "Schaefer (400): $n_s / $(echo $SUBJECTS | wc -w)"
    echo "Failures: awk '\$7!=0' $LOGS/connectome.joblog"
    ;;

  *)
    echo "Usage: $0 [test|all|reconnect]"; exit 1
    ;;
esac
