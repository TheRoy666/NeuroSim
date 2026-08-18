#!/usr/bin/env bash
###############################################################################
# NeuroSim UNAM_TLE — MRIQC (T1w + rest BOLD only)
#
# Sternberg task fMRI is NOT QC'd here -- not used in this analysis.
# Files remain on disk untouched; QC them later if they're ever pulled in.
###############################################################################

set -uo pipefail

BASE="${BASE:?Set BASE to your UNAM_TLE data root}"
BIDS="${BIDS:-$BASE}"
SIF_DIR="${SIF_DIR:?Set SIF_DIR to the directory containing your Singularity images}"

MRIQC_SIF="$SIF_DIR/mriqc-24.0.2.sif"
MRIQC_OUT="$BASE/mriqc_out"
MRIQC_WORK="$BASE/mriqc_work"

mkdir -p "$MRIQC_OUT" "$MRIQC_WORK" "$BASE/logs"

[[ -f "$MRIQC_SIF" ]] || { echo "MISSING: $MRIQC_SIF"; exit 1; }
[[ -d "$BIDS" ]]      || { echo "MISSING BIDS: $BIDS"; exit 1; }

echo "=== Running MRIQC: T1w + rest BOLD only (tfMRI excluded) ==="
echo "Free RAM before start: $(free -g | awk '/^Mem:/{print $7}') GB"

singularity run --cleanenv \
  -B "$BIDS":/data:ro \
  -B "$MRIQC_OUT":/out \
  -B "$MRIQC_WORK":/work \
  "$MRIQC_SIF" \
  /data /out participant \
    --task-id resting \
    --nprocs 20 --omp-nthreads 4 --mem-gb 180 \
    --work-dir /work \
    --no-sub \
  2>&1 | tee "$BASE/logs/mriqc_run.log"

echo "=== Group report ==="
singularity run --cleanenv \
  -B "$BIDS":/data:ro \
  -B "$MRIQC_OUT":/out \
  -B "$MRIQC_WORK":/work \
  "$MRIQC_SIF" \
  /data /out group --no-sub \
  2>&1 | tee "$BASE/logs/mriqc_group.log"

echo "Done. Review: $MRIQC_OUT/group_bold.html and group_T1w.html"