#!/usr/bin/env bash
# run_validation.sh
#
# Fast validation run of the skillmix pipeline.
# Uses a subset of skills and tasks from an existing pipeline run to quickly
# verify that skill injection produces positive or neutral delta across
# all models and modes.
#
# This script is the lightweight counterpart to run_full_pipeline.sh.
# It skips stages 1-4 (extraction) and runs only stages 5-7 (evaluation,
# visualization, traceability) on pre-existing data.
#
# Two input modes:
#   1. FROM EXISTING RUN: point --data-dir at a completed pipeline-run/
#   2. FROM EXPERIMENT-V2: uses llm-skills.shared-data/260323.skillmix-extraction-experiment/ (default)
#
# Prerequisites:
#   - Ollama running on localhost:11434
#   - ZHIPU_API_KEY set (optional, for GLM-5-turbo)
#   - Anthropic OAuth configured (for judge and claude-opus-4-6 eval)
#
# Usage:
#   cd kcg-ml-llm
#   bash skillsbench-evaluation/scripts/run_validation.sh
#   bash skillsbench-evaluation/scripts/run_validation.sh --data-dir llm-skills.shared-data/skillmix-pipeline-run

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Convert MSYS2 paths (/c/...) to Windows paths (C:/...) for Python compatibility
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "mingw"* || "$OSTYPE" == "cygwin" ]]; then
    to_pypath() { cygpath -m "$1"; }
else
    to_pypath() { echo "$1"; }
fi

# Parse arguments
DATA_DIR="${REPO_ROOT}/llm-skills.shared-data/260323.skillmix-extraction-experiment"
CLEAN=false
QUICK=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --clean) CLEAN=true; shift ;;
        --quick) QUICK=true; shift ;;
        *) echo "Unknown argument: $1. Usage: run_validation.sh [--data-dir DIR] [--clean] [--quick]"; exit 1 ;;
    esac
done

VALIDATION_DIR="${REPO_ROOT}/llm-skills.shared-data/260323.skilleval-validation-experiment"

# Skills to validate (subset for speed)
if [[ "$QUICK" == true ]]; then
    VALID_SKILLS=(
        "expose-definitional-incompatibility"
    )
else
    VALID_SKILLS=(
        "expose-definitional-incompatibility"
        "identify-recurring-structural-pattern"
        "evaluate-prediction-against-evidence"
    )
fi

# Config-driven model routing (primary path)
CONFIG_FILE="${REPO_ROOT}/llm-providers/configs/models.yaml"

# Legacy models (fallback only, used when CONFIG_FILE does not exist)
OLLAMA_URL="http://localhost:11434/v1"
OLLAMA_MODELS=("qwen2.5:3b" "qwen2.5:7b")

ZAI_MODEL="glm-5-turbo"
ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"

ANTHROPIC_MODEL="claude-opus-4-6"

# Evaluation modes
MODES=("singlecall" "stepwise" "guided")

# Judge
JUDGE_PROVIDER="anthropic"
JUDGE_MODEL="claude-opus-4-6"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
sanitize() { echo "$1" | sed 's|[/:]|-|g'; }

cd "$REPO_ROOT"

# Python-friendly paths (converts /c/... to C:/... on Windows)
PY_DATA_DIR="$(to_pypath "$DATA_DIR")"
PY_VALIDATION_DIR="$(to_pypath "$VALIDATION_DIR")"

# Add module paths to PYTHONPATH so python -m can find them
PY_REPO_ROOT="$(to_pypath "$REPO_ROOT")"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "mingw"* || "$OSTYPE" == "cygwin" ]]; then
    SEP=";"
else
    SEP=":"
fi
export PYTHONPATH="${PY_REPO_ROOT}/skillsbench-evaluation${SEP}${PY_REPO_ROOT}/llm-skills.text-extraction-pipeline${SEP}${PY_REPO_ROOT}/llm-skills.shared-data${SEP}${PY_REPO_ROOT}/llm-providers${SEP}${PYTHONPATH:-}"

if [[ "$CLEAN" == true ]]; then
    log "Cleaning old validation results in ${VALIDATION_DIR}"
    rm -f "${VALIDATION_DIR}"/results-*.json
    rm -rf "${VALIDATION_DIR}"/heatmaps
fi

mkdir -p "${VALIDATION_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Locate input data
# ---------------------------------------------------------------------------

log "=== Locating input data ==="

# Try stage-prefixed paths first (run_full_pipeline.sh output), then flat paths (experiment-v2)
if [[ -f "${DATA_DIR}/stage1-task-extraction/tasks.json" ]]; then
    TASKS_SOURCE="${DATA_DIR}/stage1-task-extraction/tasks.json"
    SKILLS_SOURCE="${DATA_DIR}/stage4-skill-verification/verified_skills.json"
elif [[ -f "${DATA_DIR}/tasks-all.json" ]]; then
    TASKS_SOURCE="${DATA_DIR}/tasks-all.json"
    SKILLS_SOURCE="${DATA_DIR}/verified_skills.json"
else
    echo "ERROR: No tasks file found in ${DATA_DIR}. Run the full pipeline first."
    exit 1
fi

log "  Tasks:  ${TASKS_SOURCE}"
log "  Skills: ${SKILLS_SOURCE}"

# Convert paths for Python
PY_TASKS_SOURCE="$(to_pypath "$TASKS_SOURCE")"
PY_SKILLS_SOURCE="$(to_pypath "$SKILLS_SOURCE")"

# ---------------------------------------------------------------------------
# Step 2: Filter to validation subset
# ---------------------------------------------------------------------------

log "=== Filtering to validation subset ==="

FILTERED_TASKS="${VALIDATION_DIR}/tasks-filtered.json"
FILTERED_SKILLS="${VALIDATION_DIR}/skills-filtered.json"
PY_FILTERED_TASKS="$(to_pypath "$FILTERED_TASKS")"
PY_FILTERED_SKILLS="$(to_pypath "$FILTERED_SKILLS")"

VALID_NAMES=$(printf '"%s",' "${VALID_SKILLS[@]}")
VALID_NAMES="[${VALID_NAMES%,}]"

python3 -c "
import json

valid_names = ${VALID_NAMES}
tasks = json.load(open('${PY_TASKS_SOURCE}'))
skills = json.load(open('${PY_SKILLS_SOURCE}'))

filtered_skills = [s for s in skills if s.get('name') in valid_names]

valid_task_uids = set()
for s in filtered_skills:
    for uid in s.get('source_task_uids', s.get('source_task_ids', [])):
        valid_task_uids.add(uid)

filtered_tasks = [t for t in tasks
                  if t.get('task_uid', t.get('task_id', '')) in valid_task_uids]

json.dump(filtered_tasks, open('${PY_FILTERED_TASKS}', 'w'), indent=2)
json.dump(filtered_skills, open('${PY_FILTERED_SKILLS}', 'w'), indent=2)

print(f'Filtered: {len(filtered_tasks)} tasks, {len(filtered_skills)} skills')
for s in filtered_skills:
    source_uids = s.get('source_task_uids', s.get('source_task_ids', []))
    matched = sum(1 for t in filtered_tasks
                  if t.get('task_uid', t.get('task_id', '')) in source_uids)
    print(f'  {s[\"name\"]}: {matched} tasks')
"

# ---------------------------------------------------------------------------
# Step 3: Evaluate all models x all modes
# ---------------------------------------------------------------------------

log "=== Running evaluation ==="

if [[ -f "${CONFIG_FILE}" ]]; then
    # ---- Config-driven path (primary) ----
    log "Using config: ${CONFIG_FILE}"

    for mode in "${MODES[@]}"; do
        log "--- Mode: ${mode} ---"
        RESULT_FILE="${VALIDATION_DIR}/results-${mode}.json"

        if [[ "$CLEAN" != true && -f "$RESULT_FILE" ]]; then
            log "  [skip] ${mode} (result exists)"
            continue
        fi

        log "  Running all models / ${mode}"
        python3 -m c3_skillsbench.corpus_harness \
            --tasks "${FILTERED_TASKS}" \
            --skills "${FILTERED_SKILLS}" \
            --config "${CONFIG_FILE}" \
            --mode "${mode}" \
            --output "${RESULT_FILE}" -v
    done
else
    # ---- Legacy fallback (per-provider blocks) ----
    log "WARNING: Config file not found at ${CONFIG_FILE}, using legacy per-provider routing"

    for mode in "${MODES[@]}"; do
        log "--- Mode: ${mode} ---"

        # Ollama models
        for model in "${OLLAMA_MODELS[@]}"; do
            SAFE=$(sanitize "${model}")
            RESULT_FILE="${VALIDATION_DIR}/results-${mode}-${SAFE}.json"

            if [[ -f "$RESULT_FILE" ]]; then
                log "  [skip] ${model} / ${mode}"
                continue
            fi

            log "  Running ${model} / ${mode}"
            python3 -m c3_skillsbench.corpus_harness \
                --tasks "${FILTERED_TASKS}" \
                --skills "${FILTERED_SKILLS}" \
                --models "${model}" \
                --mode "${mode}" \
                --provider openai \
                --base-url "${OLLAMA_URL}" \
                --judge-provider "${JUDGE_PROVIDER}" \
                --judge-model "${JUDGE_MODEL}" \
                --output "${RESULT_FILE}" -v
        done

        # Z.AI model (optional)
        if [[ -n "${ZAI_MODEL:-}" && -n "${ZHIPU_API_KEY:-}" ]]; then
            SAFE_ZAI=$(sanitize "${ZAI_MODEL}")
            RESULT_FILE="${VALIDATION_DIR}/results-${mode}-${SAFE_ZAI}.json"

            if [[ -f "$RESULT_FILE" ]]; then
                log "  [skip] ${ZAI_MODEL} / ${mode}"
            else
                log "  Running ${ZAI_MODEL} / ${mode}"
                OPENAI_API_KEY="${ZHIPU_API_KEY}" python3 -m c3_skillsbench.corpus_harness \
                    --tasks "${FILTERED_TASKS}" \
                    --skills "${FILTERED_SKILLS}" \
                    --models "${ZAI_MODEL}" \
                    --mode "${mode}" \
                    --provider openai \
                    --base-url "${ZAI_BASE_URL}" \
                    --judge-provider "${JUDGE_PROVIDER}" \
                    --judge-model "${JUDGE_MODEL}" \
                    --output "${RESULT_FILE}" -v
            fi
        fi

        # Anthropic model (optional)
        if [[ -n "${ANTHROPIC_MODEL:-}" ]]; then
            SAFE_ANTH=$(sanitize "${ANTHROPIC_MODEL}")
            RESULT_FILE="${VALIDATION_DIR}/results-${mode}-${SAFE_ANTH}.json"

            if [[ -f "$RESULT_FILE" ]]; then
                log "  [skip] ${ANTHROPIC_MODEL} / ${mode}"
            else
                log "  Running ${ANTHROPIC_MODEL} / ${mode}"
                python3 -m c3_skillsbench.corpus_harness \
                    --tasks "${FILTERED_TASKS}" \
                    --skills "${FILTERED_SKILLS}" \
                    --models "${ANTHROPIC_MODEL}" \
                    --mode "${mode}" \
                    --provider anthropic \
                    --judge-provider "${JUDGE_PROVIDER}" \
                    --judge-model "${JUDGE_MODEL}" \
                    --output "${RESULT_FILE}" -v
            fi
        fi
    done
fi

# ---------------------------------------------------------------------------
# Step 4: Summary table
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Validation Results: ${#VALID_SKILLS[@]} skills, ${#MODES[@]} modes"
echo "============================================================"

python3 -c "
import json, glob, os

validation_dir = '${PY_VALIDATION_DIR}'
files = sorted(glob.glob(os.path.join(validation_dir, 'results-*.json')))
files = [f for f in files if 'merged' not in f and 'filtered' not in f]

print(f'{\"Model\":<25} {\"Mode\":<12} {\"Baseline\":>8} {\"Curated\":>8} {\"Delta\":>8} {\"Verdict\":>8}')
print('-' * 75)

total_pass = 0
total_fail = 0
total_neutral = 0

for f in files:
    eps = json.load(open(f))
    if not eps:
        continue

    # Derive mode from filename
    basename = os.path.basename(f).replace('results-', '').replace('.json', '')
    parts = basename.split('-', 1)
    file_mode = parts[0]
    model = eps[0].get('model', '?')

    bl = [e for e in eps if e['condition'] == 'baseline']
    cu = [e for e in eps if e['condition'] == 'curated']

    bl_avg = sum(e['score'] for e in bl) / len(bl) if bl else 0
    cu_avg = sum(e['score'] for e in cu) / len(cu) if cu else 0
    delta = cu_avg - bl_avg

    if delta > 0.01:
        verdict = 'PASS'
        total_pass += 1
    elif delta < -0.01:
        verdict = 'FAIL'
        total_fail += 1
    else:
        verdict = 'NEUTRAL'
        total_neutral += 1

    print(f'{model:<25} {file_mode:<12} {bl_avg:>8.3f} {cu_avg:>8.3f} {delta:>+8.3f} {verdict:>8}')

print('-' * 75)
print(f'Totals: {total_pass} PASS, {total_neutral} NEUTRAL, {total_fail} FAIL out of {total_pass + total_neutral + total_fail} runs')
"

# ---------------------------------------------------------------------------
# Step 5: Merge results and generate visualizations
# ---------------------------------------------------------------------------

log "=== Generating visualizations ==="

HEATMAPS_DIR="${VALIDATION_DIR}/heatmaps"
mkdir -p "${HEATMAPS_DIR}"

# Per-mode heatmaps
for mode in "${MODES[@]}"; do
    MERGED="${VALIDATION_DIR}/results-${mode}-merged.json"
    PY_MERGED="$(to_pypath "$MERGED")"

    python3 -c "
import json, glob, os
files = sorted(glob.glob(os.path.join('${PY_VALIDATION_DIR}', 'results-${mode}-*.json')))
files = [f for f in files if 'merged' not in f and 'filtered' not in f]
merged = []
for f in files:
    with open(f) as fh:
        merged.extend(json.load(fh))
with open('${PY_MERGED}', 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'  Merged ${mode}: {len(merged)} episodes from {len(files)} files')
"

    mkdir -p "${HEATMAPS_DIR}/${mode}"
    python3 -m c3_skillsbench.visualization \
        --results "${MERGED}" \
        -o "${HEATMAPS_DIR}/${mode}" \
        --type all --dpi 200 \
        2>&1 || log "  WARNING: visualization failed for ${mode}"
done

# Cross-mode charts
ALL_MERGED="${VALIDATION_DIR}/results-all-merged.json"
PY_ALL_MERGED="$(to_pypath "$ALL_MERGED")"
python3 -c "
import json, glob, os
files = sorted(glob.glob(os.path.join('${PY_VALIDATION_DIR}', 'results-*.json')))
files = [f for f in files if 'merged' not in f and 'filtered' not in f]
merged = []
for f in files:
    with open(f) as fh:
        merged.extend(json.load(fh))
with open('${PY_ALL_MERGED}', 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'  All merged: {len(merged)} episodes from {len(files)} files')
"

mkdir -p "${HEATMAPS_DIR}/cross-mode"
python3 -m c3_skillsbench.visualization \
    --results "${ALL_MERGED}" \
    -o "${HEATMAPS_DIR}/cross-mode" \
    --type charts --dpi 200 \
    2>&1 || log "  WARNING: cross-mode charts failed"

log "Visualizations written to ${HEATMAPS_DIR}"
