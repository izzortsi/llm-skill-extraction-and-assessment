#!/usr/bin/env bash
# run_experiment.sh
#
# Master script for the 4-domain x 9-model skill extraction & evaluation experiment.
# Phases: task extraction -> trace capture -> skill extraction -> eval (9 Ollama models)
#
# Prerequisites:
#   - Ollama running on localhost:11434
#   - ANTHROPIC_API_KEY set
#   - Python environment with: anthropic, openai, datasets, pyyaml
#
# Usage:
#   cd kcg-ml-llm
#   bash skillsbench-evaluation/scripts/run_experiment.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/llm-skills.shared-data/skilleval-experiment-4x10"
RESULTS_DIR="${DATA_DIR}/results"
LOGS_DIR="${DATA_DIR}/logs"

DOMAINS=("science" "history" "language" "mathematics")
DATASET="wikitext"
SUBSET="wikitext-2-raw-v1"
MAX_CHUNKS=5
CHUNK_SIZE=4000
TASKS_PER_CHUNK=1

TRACE_MODEL="qwen2.5:3b"
OLLAMA_URL="http://localhost:11434/v1"

OLLAMA_MODELS=(
    "qwen3:0.6b"
    "llama3.2:1b"
    "qwen2.5:1.5b"
    "smollm2:1.7b"
    "qwen2.5:3b"
    "phi3.5:3.8b"
    "mistral:7b"
    "llama3.1:8b"
    "qwen2.5:14b"
)

MODES=("singlecall" "stepwise" "skillselect")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

die() { log "FATAL: $*" >&2; exit 1; }

ensure_dir() { mkdir -p "$1"; }

# Sanitize model name for use as filename (replace / and : with -)
sanitize_model() {
    echo "$1" | sed 's|[/:]|-|g'
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

# Add module paths to PYTHONPATH
# NOTE: extraction exists in BOTH text-extraction-pipeline and task-skill-extraction-pipeline.
# Python only uses the first match, so we set PYTHONPATH per-phase below.
BASE_PYTHONPATH="${REPO_ROOT}/skillsbench-evaluation:${REPO_ROOT}/llm-skills.shared-data:${REPO_ROOT}/llm-providers:${PYTHONPATH:-}"
PYTHONPATH_TEXT="${REPO_ROOT}/llm-skills.text-extraction-pipeline:${BASE_PYTHONPATH}"
PYTHONPATH_SKILL="${REPO_ROOT}/llm-skills.task-skill-extraction-pipeline:${BASE_PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH_SKILL}"

log "Repo root: ${REPO_ROOT}"
log "Data directory:  ${DATA_DIR}"

[[ -n "${ANTHROPIC_API_KEY:-}" ]] || die "ANTHROPIC_API_KEY not set"

# Check Ollama is running
if ! curl -s "${OLLAMA_URL%/v1}/api/tags" > /dev/null 2>&1; then
    die "Ollama not reachable at ${OLLAMA_URL%/v1}. Start it with: ollama serve"
fi

ensure_dir "${DATA_DIR}/extraction"
ensure_dir "${RESULTS_DIR}"
ensure_dir "${LOGS_DIR}"
for mode in "${MODES[@]}"; do
    ensure_dir "${RESULTS_DIR}/${mode}"
done

# ===================================================================
# Phase 1: Task Extraction (Anthropic Opus) — 4 domains x 5 tasks = 20
# ===================================================================

log "=== Phase 1: Task Extraction ==="

for domain in "${DOMAINS[@]}"; do
    TASK_FILE="${DATA_DIR}/extraction/tasks-${domain}.json"
    if [[ -f "$TASK_FILE" ]]; then
        log "  [skip] ${domain} — ${TASK_FILE} already exists"
        continue
    fi
    log "  Extracting tasks for domain: ${domain}"
    PYTHONPATH="${PYTHONPATH_TEXT}" python -m extraction.task_extractor \
        --dataset "${DATASET}" --subset "${SUBSET}" \
        --domain "${domain}" \
        --max-chunks "${MAX_CHUNKS}" --chunk-size "${CHUNK_SIZE}" \
        --tasks-per-chunk "${TASKS_PER_CHUNK}" \
        --provider anthropic --model claude-opus-4-6 \
        --output "${TASK_FILE}" -v \
        2>&1 | tee "${LOGS_DIR}/extract-tasks-${domain}.log"
    log "  Done: ${TASK_FILE}"
done

# Merge per-domain task files into tasks-all.json
TASKS_ALL="${DATA_DIR}/tasks-all.json"
if [[ -f "$TASKS_ALL" ]]; then
    log "  [skip] tasks-all.json already exists"
else
    log "  Merging domain task files into tasks-all.json"
    python -c "
import json, glob, sys
tasks = []
for f in sorted(glob.glob('${DATA_DIR}/extraction/tasks-*.json')):
    with open(f) as fh:
        tasks.extend(json.load(fh))
print(f'Merged {len(tasks)} tasks from {len(glob.glob(\"${DATA_DIR}/extraction/tasks-*.json\"))} files')
with open('${TASKS_ALL}', 'w') as fh:
    json.dump(tasks, fh, indent=2)
"
    log "  Created: ${TASKS_ALL}"
fi

# Validate task count
TASK_COUNT=$(python -c "import json; print(len(json.load(open('${TASKS_ALL}'))))")
log "  Total tasks: ${TASK_COUNT}"
if [[ "$TASK_COUNT" -lt 20 ]]; then
    log "  WARNING: Expected 20 tasks, got ${TASK_COUNT}"
fi

# ===================================================================
# Phase 2: Trace Capture (Ollama qwen2.5:3b — free)
# ===================================================================

log "=== Phase 2: Trace Capture ==="

TRACES_FILE="${DATA_DIR}/traces.jsonl"
if [[ -f "$TRACES_FILE" ]]; then
    log "  [skip] traces.jsonl already exists"
else
    log "  Capturing traces with ${TRACE_MODEL} via Ollama"
    OPENAI_BASE_URL="${OLLAMA_URL}" python -m extraction.trace_runner \
        --tasks "${TASKS_ALL}" \
        --output "${TRACES_FILE}" \
        --provider openai --model "${TRACE_MODEL}" -v \
        2>&1 | tee "${LOGS_DIR}/trace-capture.log"
    log "  Done: ${TRACES_FILE}"
fi

# ===================================================================
# Phase 3: Skill Extraction + Verification (Anthropic Opus)
# ===================================================================

log "=== Phase 3: Skill Extraction & Verification ==="

SKILLS_FILE="${DATA_DIR}/skills.json"
VERIFIED_SKILLS="${DATA_DIR}/verified_skills.json"

if [[ -f "$SKILLS_FILE" ]]; then
    log "  [skip] skills.json already exists"
else
    log "  Extracting skills from traces"
    python -m extraction.skill_extractor \
        --traces "${TRACES_FILE}" \
        --output "${SKILLS_FILE}" \
        --max-skills 15 \
        --provider anthropic --model claude-opus-4-6 -v \
        2>&1 | tee "${LOGS_DIR}/skill-extraction.log"
    log "  Done: ${SKILLS_FILE}"
fi

if [[ -f "$VERIFIED_SKILLS" ]]; then
    log "  [skip] verified_skills.json already exists"
else
    log "  Verifying and revising skills"
    python -m extraction.skill_verifier \
        --skills "${SKILLS_FILE}" \
        --output "${VERIFIED_SKILLS}" \
        --revise --max-revisions 2 \
        --provider anthropic --model claude-opus-4-6 -v \
        2>&1 | tee "${LOGS_DIR}/skill-verification.log"
    log "  Done: ${VERIFIED_SKILLS}"
fi

SKILL_COUNT=$(python -c "import json; print(len(json.load(open('${VERIFIED_SKILLS}'))))")
log "  Verified skills: ${SKILL_COUNT}"

# ===================================================================
# Phase 4: Pull Ollama Models
# ===================================================================

log "=== Phase 4: Pulling Ollama Models ==="

for model in "${OLLAMA_MODELS[@]}"; do
    log "  Pulling ${model}"
    ollama pull "${model}" 2>&1 | tail -1
done

log "  All Ollama models pulled"

# ===================================================================
# Phase 5: Corpus Evaluation — 9 Ollama Models (per-model for recovery)
# ===================================================================

log "=== Phase 5: Corpus Evaluation (Ollama models) ==="

for mode in "${MODES[@]}"; do
    log "--- Mode: ${mode} ---"

    for model in "${OLLAMA_MODELS[@]}"; do
        SAFE_NAME=$(sanitize_model "${model}")
        RESULT_FILE="${RESULTS_DIR}/${mode}/results-${SAFE_NAME}.json"

        if [[ -f "$RESULT_FILE" ]]; then
            log "  [skip] ${model} / ${mode} — result file exists"
            continue
        fi

        log "  Running ${model} / ${mode}"
        python -m skillsbench.corpus_harness \
            --tasks "${TASKS_ALL}" \
            --skills "${VERIFIED_SKILLS}" \
            --models "${model}" \
            --mode "${mode}" \
            --provider openai \
            --base-url "${OLLAMA_URL}" \
            --judge-provider anthropic \
            --judge-model claude-opus-4-6 \
            --output "${RESULT_FILE}" -v \
            2>&1 | tee "${LOGS_DIR}/eval-${mode}-${SAFE_NAME}.log"
        log "  Done: ${RESULT_FILE}"
    done
done

# ===================================================================
# Summary
# ===================================================================

log "=== Experiment Complete ==="
log "Results directory: ${RESULTS_DIR}"
log ""
log "Next step: bash scripts/merge_and_visualize.sh"
