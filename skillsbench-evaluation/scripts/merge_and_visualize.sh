#!/usr/bin/env bash
# merge_and_visualize.sh
#
# Merge per-model result files into combined results and generate heatmaps.
# Run AFTER run_experiment.sh and run_vllm_eval.sh (or just run_experiment.sh
# if skipping 72B).
#
# Usage:
#   cd kcg-ml-llm
#   bash skillsbench-evaluation/scripts/merge_and_visualize.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/shared-data/skilleval-experiment-4x10"
RESULTS_DIR="${DATA_DIR}/results"
HEATMAPS_DIR="${DATA_DIR}/heatmaps"

MODES=("singlecall" "stepwise" "skillselect")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

die() { log "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

# Add module paths to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/skillsbench-evaluation:${PYTHONPATH:-}"

log "Repo root: ${REPO_ROOT}"
log "Data directory:  ${DATA_DIR}"

[[ -d "$RESULTS_DIR" ]] || die "Results directory not found: ${RESULTS_DIR}"

mkdir -p "${HEATMAPS_DIR}"

# ===================================================================
# Phase 7a: Merge per-model results
# ===================================================================

log "=== Phase 7a: Merging per-model results ==="

for mode in "${MODES[@]}"; do
    MODE_DIR="${RESULTS_DIR}/${mode}"
    MERGED_FILE="${RESULTS_DIR}/results-${mode}-merged.json"

    if [[ ! -d "$MODE_DIR" ]]; then
        log "  [skip] No directory for mode: ${mode}"
        continue
    fi

    RESULT_FILES=("${MODE_DIR}"/results-*.json)
    if [[ ${#RESULT_FILES[@]} -eq 0 ]] || [[ ! -f "${RESULT_FILES[0]}" ]]; then
        log "  [skip] No result files in ${MODE_DIR}"
        continue
    fi

    FILE_COUNT=${#RESULT_FILES[@]}
    log "  Merging ${FILE_COUNT} result files for mode: ${mode}"

    python -c "
import json, glob, sys

files = sorted(glob.glob('${MODE_DIR}/results-*.json'))
all_episodes = []
models_seen = set()

for f in files:
    with open(f) as fh:
        episodes = json.load(fh)
        for ep in episodes:
            models_seen.add(ep.get('model', ''))
        all_episodes.extend(episodes)

print(f'  Merged {len(all_episodes)} episodes from {len(files)} files ({len(models_seen)} models)')

with open('${MERGED_FILE}', 'w') as fh:
    json.dump(all_episodes, fh, indent=2)
"
    log "  Created: ${MERGED_FILE}"
done

# ===================================================================
# Phase 7b: Generate heatmaps per mode
# ===================================================================

log "=== Phase 7b: Generating heatmaps ==="

for mode in "${MODES[@]}"; do
    MERGED_FILE="${RESULTS_DIR}/results-${mode}-merged.json"
    MODE_HEATMAP_DIR="${HEATMAPS_DIR}/${mode}"

    if [[ ! -f "$MERGED_FILE" ]]; then
        log "  [skip] No merged results for mode: ${mode}"
        continue
    fi

    mkdir -p "${MODE_HEATMAP_DIR}"

    log "  Generating heatmaps for mode: ${mode}"
    python -m skillsbench.visualization \
        --results "${MERGED_FILE}" \
        --output-dir "${MODE_HEATMAP_DIR}" \
        --type all --dpi 200 \
        2>&1
    log "  Done: ${MODE_HEATMAP_DIR}"
done

# ===================================================================
# Phase 7c: Generate combined heatmap (all modes)
# ===================================================================

log "=== Phase 7c: Combined heatmap (all modes) ==="

COMBINED_ARGS=()
for mode in "${MODES[@]}"; do
    MERGED_FILE="${RESULTS_DIR}/results-${mode}-merged.json"
    if [[ -f "$MERGED_FILE" ]]; then
        COMBINED_ARGS+=("${MERGED_FILE}")
    fi
done

if [[ ${#COMBINED_ARGS[@]} -gt 0 ]]; then
    mkdir -p "${HEATMAPS_DIR}/combined"

    log "  Merging ${#COMBINED_ARGS[@]} mode files for combined visualization"
    python -m skillsbench.visualization \
        --results "${COMBINED_ARGS[@]}" \
        --output-dir "${HEATMAPS_DIR}/combined" \
        --mode per-mode --type all --dpi 200 \
        2>&1
    log "  Done: ${HEATMAPS_DIR}/combined"
else
    log "  [skip] No merged result files found"
fi

# ===================================================================
# Summary
# ===================================================================

log "=== Merge & Visualize Complete ==="
log ""
log "Merged results:"
for mode in "${MODES[@]}"; do
    MERGED_FILE="${RESULTS_DIR}/results-${mode}-merged.json"
    if [[ -f "$MERGED_FILE" ]]; then
        COUNT=$(python -c "import json; print(len(json.load(open('${MERGED_FILE}'))))")
        log "  ${mode}: ${COUNT} episodes"
    fi
done
log ""
log "Heatmaps:"
find "${HEATMAPS_DIR}" -name "*.png" -type f | sort | while read -r f; do
    log "  ${f}"
done
log ""
log "Experiment pipeline complete."
