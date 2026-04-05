#!/usr/bin/env bash
# run_experiment_v2.sh
#
# Experiment v2: 4 domains x 4 models, 2 skills per domain.
# Domains: language-skills, rhetoric, logic-and-reasoning, law
# Source: Wikipedia (wikimedia/wikipedia, 20231101.en)
# Models: 2 Ollama (qwen3:0.6b, qwen2.5:3b) + GLM-5-turbo (Z.AI) + claude-opus-4-6 (Anthropic)
#
# Prerequisites:
#   - Ollama running on localhost:11434
#   - Anthropic OAuth configured (anthropic-oauth library) OR ANTHROPIC_API_KEY set
#   - ZHIPU_API_KEY set to ZhipuAI key (for GLM-5-turbo via Z.AI)
#   - Python environment with: anthropic, anthropic_oauth, openai, datasets
#
# Usage:
#   cd kcg-ml-llm
#   bash skillsbench-evaluation/scripts/run_experiment_v2.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/shared-data/260323.skillmix-extraction-experiment"
RESULTS_DIR="${DATA_DIR}/results"
LOGS_DIR="${DATA_DIR}/logs"

DOMAINS=("language-skills" "rhetoric" "logic-and-reasoning" "law")
DATASET="wikimedia/wikipedia"
SUBSET="20231101.en"
MAX_CHUNKS=4
CHUNK_SIZE=4000
TASKS_PER_CHUNK=1

TRACE_MODEL="claude-opus-4-6"
OLLAMA_URL="http://localhost:11434/v1"

OLLAMA_MODELS=(
    "qwen3:0.6b"
    "qwen2.5:3b"
    "qwen2.5:7b"
)

ZAI_MODEL="glm-5-turbo"
ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"

ANTHROPIC_MODEL="claude-opus-4-6"

MODES=("singlecall" "stepwise" "guided")

MAX_SKILLS=8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

die() { log "FATAL: $*" >&2; exit 1; }

ensure_dir() { mkdir -p "$1"; }

sanitize_model() {
    echo "$1" | sed 's|[/:]|-|g'
}

# Count items in a JSON array file
json_count() {
    python -c "import json; print(len(json.load(open('$1'))))" 2>/dev/null || echo "0"
}

# Count episodes in a result JSON file
episode_summary() {
    python -c "
import json
eps = json.load(open('$1'))
bl = [e for e in eps if e['condition']=='baseline']
cu = [e for e in eps if e['condition']=='curated']
bl_avg = sum(e['score'] for e in bl)/len(bl) if bl else 0
cu_avg = sum(e['score'] for e in cu)/len(cu) if cu else 0
print(f'{len(eps)} episodes ({len(bl)} baseline, {len(cu)} curated) | baseline={bl_avg:.2f} curated={cu_avg:.2f} delta={cu_avg-bl_avg:+.2f}')
" 2>/dev/null || echo "error reading file"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

# Add module paths to PYTHONPATH
# NOTE: extraction exists in BOTH text-extraction-pipeline and task-skill-extraction-pipeline.
# Python only uses the first match, so we set PYTHONPATH per-phase below.
BASE_PYTHONPATH="${REPO_ROOT}/skillsbench-evaluation:${REPO_ROOT}/skillmix-evaluation:${REPO_ROOT}/shared-data:${REPO_ROOT}/llm-providers:${PYTHONPATH:-}"
# Phase 1 (task extraction) needs text-extraction-pipeline
PYTHONPATH_TEXT="${REPO_ROOT}/llm-skills.text-extraction-pipeline:${BASE_PYTHONPATH}"
# Phases 2-4 (traces, skills, verification) need task-skill-extraction-pipeline
PYTHONPATH_SKILL="${REPO_ROOT}/llm-skills.task-skill-extraction-pipeline:${BASE_PYTHONPATH}"
# Phases 5+ (evaluation) need skillsbench (already in BASE)
export PYTHONPATH="${PYTHONPATH_SKILL}"

log "Repo root: ${REPO_ROOT}"
log "Data directory:  ${DATA_DIR}"

# Check Ollama is running
if ! curl -s "${OLLAMA_URL%/v1}/api/tags" > /dev/null 2>&1; then
    die "Ollama not reachable at ${OLLAMA_URL%/v1}. Start it with: ollama serve"
fi

# Check Z.AI API key
[[ -n "${ZHIPU_API_KEY:-}" ]] || log "WARNING: ZHIPU_API_KEY not set — GLM-5-turbo eval will fail"

ensure_dir "${DATA_DIR}/extraction"
ensure_dir "${RESULTS_DIR}"
ensure_dir "${LOGS_DIR}"
for mode in "${MODES[@]}"; do
    ensure_dir "${RESULTS_DIR}/${mode}"
done

# ===================================================================
# Phase 1: Task Extraction — 4 domains x ~4 tasks = ~16 tasks
# ===================================================================

log "=== Phase 1: Task Extraction ==="

DOMAIN_IDX=0
for domain in "${DOMAINS[@]}"; do
    DOMAIN_IDX=$((DOMAIN_IDX + 1))
    TASK_FILE="${DATA_DIR}/extraction/tasks-${domain}.json"
    if [[ -f "$TASK_FILE" ]]; then
        COUNT=$(json_count "${TASK_FILE}")
        log "  [skip] [${DOMAIN_IDX}/${#DOMAINS[@]}] ${domain} — ${COUNT} tasks already extracted"
        continue
    fi
    log "  [${DOMAIN_IDX}/${#DOMAINS[@]}] Extracting tasks for domain: ${domain}"
    PYTHONPATH="${PYTHONPATH_TEXT}" python -m extraction.task_extractor \
        --dataset "${DATASET}" --subset "${SUBSET}" \
        --domain "${domain}" \
        --max-chunks "${MAX_CHUNKS}" --chunk-size "${CHUNK_SIZE}" \
        --tasks-per-chunk "${TASKS_PER_CHUNK}" \
        --provider anthropic --model claude-opus-4-6 \
        --output "${TASK_FILE}" -v \
        2>&1 | tee "${LOGS_DIR}/extract-tasks-${domain}.log"
    COUNT=$(json_count "${TASK_FILE}")
    log "  [${DOMAIN_IDX}/${#DOMAINS[@]}] Done: ${domain} — ${COUNT} tasks extracted"
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

TASK_COUNT=$(python -c "import json; print(len(json.load(open('${TASKS_ALL}'))))")
log "  Total tasks: ${TASK_COUNT}"

# ===================================================================
# Phase 2: Trace Capture (Ollama qwen2.5:3b — free)
# ===================================================================

log "=== Phase 2: Trace Capture ==="

TRACES_FILE="${DATA_DIR}/traces.jsonl"
if [[ -f "$TRACES_FILE" ]]; then
    TRACE_COUNT=$(wc -l < "${TRACES_FILE}" | tr -d ' ')
    log "  [skip] traces.jsonl already exists (${TRACE_COUNT} traces)"
else
    log "  Capturing ${TASK_COUNT} traces with ${TRACE_MODEL} via Anthropic"
    python -m extraction.trace_runner \
        --tasks "${TASKS_ALL}" \
        --output "${TRACES_FILE}" \
        --provider anthropic --model "${TRACE_MODEL}" -v \
        2>&1 | tee "${LOGS_DIR}/trace-capture.log"
    TRACE_COUNT=$(wc -l < "${TRACES_FILE}" | tr -d ' ')
    log "  Done: ${TRACE_COUNT} traces captured"
fi

# ===================================================================
# Phase 3: Skill Extraction (Anthropic Opus — atomic emphasis)
# ===================================================================

log "=== Phase 3: Skill Extraction ==="

SKILLS_FILE="${DATA_DIR}/skills.json"
if [[ -f "$SKILLS_FILE" ]]; then
    SKILL_RAW=$(json_count "${SKILLS_FILE}")
    log "  [skip] skills.json already exists (${SKILL_RAW}/${MAX_SKILLS} skills)"
else
    log "  Extracting skills from traces (max ${MAX_SKILLS}, atomic emphasis)"
    python -m extraction.skill_extractor \
        --traces "${TRACES_FILE}" \
        --output "${SKILLS_FILE}" \
        --max-skills "${MAX_SKILLS}" \
        --provider anthropic --model claude-opus-4-6 -v \
        2>&1 | tee "${LOGS_DIR}/skill-extraction.log"
    SKILL_RAW=$(json_count "${SKILLS_FILE}")
    log "  Done: ${SKILL_RAW}/${MAX_SKILLS} skills extracted"
fi

# ===================================================================
# Phase 4: Skill Verification (Anthropic Opus — revise up to 2 rounds)
# ===================================================================

log "=== Phase 4: Skill Verification ==="

VERIFIED_SKILLS="${DATA_DIR}/verified_skills.json"
if [[ -f "$VERIFIED_SKILLS" ]]; then
    log "  [skip] verified_skills.json already exists"
else
    log "  Verifying and revising ${SKILL_RAW} skills (max 2 revision rounds)"
    python -m extraction.skill_verifier \
        --skills "${SKILLS_FILE}" \
        --output "${VERIFIED_SKILLS}" \
        --revise --max-revisions 2 \
        --provider anthropic --model claude-opus-4-6 -v \
        2>&1 | tee "${LOGS_DIR}/skill-verification.log"
fi

SKILL_COUNT=$(json_count "${VERIFIED_SKILLS}")
VALID_COUNT=$(python -c "import json; v=json.load(open('${VERIFIED_SKILLS}')); print(sum(1 for x in v if x.get('is_valid',False)))" 2>/dev/null || echo "?")
log "  Verified skills: ${VALID_COUNT}/${SKILL_COUNT} valid"

# ===================================================================
# Phase 4b: Rename Task IDs to Match Skill Names
# ===================================================================

log "=== Phase 4b: Rename Task IDs ==="

RENAME_MARKER="${DATA_DIR}/.tasks-renamed"
if [[ -f "$RENAME_MARKER" ]]; then
    log "  [skip] task IDs already renamed"
else
    log "  Renaming task IDs to match verified skill names"
    python -c "
import json

tasks = json.load(open('${TASKS_ALL}'))
skills = json.load(open('${VERIFIED_SKILLS}'))

# Build mapping: old_task_id -> skill_name
# A task may appear in multiple skills; pick the first match
task_to_skill = {}
for skill in skills:
    for tid in skill.get('source_task_ids', []):
        if tid not in task_to_skill:
            task_to_skill[tid] = skill['name']

# Build new task IDs: {skill-name}-{NN} for matched tasks,
# keep original ID for unmatched tasks
skill_counters = {}
old_to_new = {}
for task in tasks:
    old_id = task['task_id']
    skill_name = task_to_skill.get(old_id)
    if skill_name:
        skill_counters[skill_name] = skill_counters.get(skill_name, 0) + 1
        new_id = f'{skill_name}-{skill_counters[skill_name]:02d}'
        old_to_new[old_id] = new_id
        task['task_id'] = new_id
    else:
        old_to_new[old_id] = old_id  # unchanged

# Update source_task_ids in verified_skills.json
for skill in skills:
    skill['source_task_ids'] = [
        old_to_new.get(tid, tid) for tid in skill.get('source_task_ids', [])
    ]

# Write updated files
with open('${TASKS_ALL}', 'w') as f:
    json.dump(tasks, f, indent=2)

with open('${VERIFIED_SKILLS}', 'w') as f:
    json.dump(skills, f, indent=2)

# Also update traces.jsonl task_ids
updated_traces = 0
trace_lines = []
with open('${TRACES_FILE}', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        trace = json.loads(line)
        old_tid = trace.get('task_id', '')
        if old_tid in old_to_new and old_to_new[old_tid] != old_tid:
            trace['task_id'] = old_to_new[old_tid]
            updated_traces += 1
        trace_lines.append(json.dumps(trace, ensure_ascii=False))
with open('${TRACES_FILE}', 'w') as f:
    f.write('\n'.join(trace_lines) + '\n')

# Also update skills.json (raw, pre-verification) source_task_ids
raw_skills = json.load(open('${SKILLS_FILE}'))
for skill in raw_skills:
    skill['source_task_ids'] = [
        old_to_new.get(tid, tid) for tid in skill.get('source_task_ids', [])
    ]
with open('${SKILLS_FILE}', 'w') as f:
    json.dump(raw_skills, f, indent=2)

renamed = sum(1 for old, new in old_to_new.items() if old != new)
print(f'Renamed {renamed}/{len(tasks)} task IDs to match skill names')
print(f'Updated {updated_traces} trace entries')
for old, new in sorted(old_to_new.items()):
    if old != new:
        print(f'  {old} -> {new}')
" 2>&1 | tee "${LOGS_DIR}/rename-task-ids.log"
    touch "${RENAME_MARKER}"
    log "  Done: task IDs renamed"
fi

# Refresh task count after rename
TASK_COUNT=$(json_count "${TASKS_ALL}")

# ===================================================================
# Phase 5: Pull Ollama Models
# ===================================================================

log "=== Phase 5: Pulling Ollama Models ==="

PULL_IDX=0
for model in "${OLLAMA_MODELS[@]}"; do
    PULL_IDX=$((PULL_IDX + 1))
    log "  [${PULL_IDX}/${#OLLAMA_MODELS[@]}] Pulling ${model}"
    ollama pull "${model}" 2>&1 | tail -1
done

log "  All ${#OLLAMA_MODELS[@]} Ollama models pulled"

# ===================================================================
# Phase 6: SkillsBench Eval — Ollama Models (2 Qwen × 3 modes)
# ===================================================================

ALL_MODELS=("${OLLAMA_MODELS[@]}" "${ZAI_MODEL}" "${ANTHROPIC_MODEL}")
TOTAL_EVALS=$(( ${#ALL_MODELS[@]} * ${#MODES[@]} ))
EVAL_IDX=0

log "=== Phase 6: SkillsBench Evaluation (Ollama models) ==="
log "  ${TASK_COUNT} tasks, ${SKILL_COUNT} skills, ${#ALL_MODELS[@]} models, ${#MODES[@]} modes = ${TOTAL_EVALS} eval runs"

for mode in "${MODES[@]}"; do
    log "--- Mode: ${mode} ---"

    for model in "${OLLAMA_MODELS[@]}"; do
        EVAL_IDX=$((EVAL_IDX + 1))
        SAFE_NAME=$(sanitize_model "${model}")
        RESULT_FILE="${RESULTS_DIR}/${mode}/results-${SAFE_NAME}.json"

        if [[ -f "$RESULT_FILE" ]]; then
            log "  [${EVAL_IDX}/${TOTAL_EVALS}] [skip] ${model} / ${mode} — $(episode_summary "${RESULT_FILE}")"
            continue
        fi

        log "  [${EVAL_IDX}/${TOTAL_EVALS}] Running ${model} / ${mode}"
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
        log "  [${EVAL_IDX}/${TOTAL_EVALS}] Done: ${model} / ${mode} — $(episode_summary "${RESULT_FILE}")"
    done
done

# ===================================================================
# Phase 7: SkillsBench Eval — GLM-5-turbo (Z.AI, OpenAI-compatible)
# ===================================================================

log "=== Phase 7: SkillsBench Evaluation (GLM-5-turbo via Z.AI) ==="

SAFE_ZAI=$(sanitize_model "${ZAI_MODEL}")

for mode in "${MODES[@]}"; do
    EVAL_IDX=$((EVAL_IDX + 1))
    RESULT_FILE="${RESULTS_DIR}/${mode}/results-${SAFE_ZAI}.json"

    if [[ -f "$RESULT_FILE" ]]; then
        log "  [${EVAL_IDX}/${TOTAL_EVALS}] [skip] ${ZAI_MODEL} / ${mode} — $(episode_summary "${RESULT_FILE}")"
        continue
    fi

    log "  [${EVAL_IDX}/${TOTAL_EVALS}] Running ${ZAI_MODEL} / ${mode}"
    OPENAI_API_KEY="${ZHIPU_API_KEY}" python -m skillsbench.corpus_harness \
        --tasks "${TASKS_ALL}" \
        --skills "${VERIFIED_SKILLS}" \
        --models "${ZAI_MODEL}" \
        --mode "${mode}" \
        --provider openai \
        --base-url "${ZAI_BASE_URL}" \
        --judge-provider anthropic \
        --judge-model claude-opus-4-6 \
        --output "${RESULT_FILE}" -v \
        2>&1 | tee "${LOGS_DIR}/eval-${mode}-${SAFE_ZAI}.log"
    log "  [${EVAL_IDX}/${TOTAL_EVALS}] Done: ${ZAI_MODEL} / ${mode} — $(episode_summary "${RESULT_FILE}")"
done

# ===================================================================
# Phase 8: SkillsBench Eval — claude-opus-4-6 (Anthropic OAuth)
# ===================================================================

log "=== Phase 8: SkillsBench Evaluation (claude-opus-4-6 via Anthropic) ==="

SAFE_ANTHROPIC=$(sanitize_model "${ANTHROPIC_MODEL}")

for mode in "${MODES[@]}"; do
    EVAL_IDX=$((EVAL_IDX + 1))
    RESULT_FILE="${RESULTS_DIR}/${mode}/results-${SAFE_ANTHROPIC}.json"

    if [[ -f "$RESULT_FILE" ]]; then
        log "  [${EVAL_IDX}/${TOTAL_EVALS}] [skip] ${ANTHROPIC_MODEL} / ${mode} — $(episode_summary "${RESULT_FILE}")"
        continue
    fi

    log "  [${EVAL_IDX}/${TOTAL_EVALS}] Running ${ANTHROPIC_MODEL} / ${mode}"
    python -m skillsbench.corpus_harness \
        --tasks "${TASKS_ALL}" \
        --skills "${VERIFIED_SKILLS}" \
        --models "${ANTHROPIC_MODEL}" \
        --mode "${mode}" \
        --provider anthropic \
        --judge-provider anthropic \
        --judge-model claude-opus-4-6 \
        --output "${RESULT_FILE}" -v \
        2>&1 | tee "${LOGS_DIR}/eval-${mode}-${SAFE_ANTHROPIC}.log"
    log "  [${EVAL_IDX}/${TOTAL_EVALS}] Done: ${ANTHROPIC_MODEL} / ${mode} — $(episode_summary "${RESULT_FILE}")"
done

# ===================================================================
# Phase 9: SkillMix Eval — All Models
# ===================================================================

log "=== Phase 9: SkillMix Evaluation ==="

SKILLMIX_DIR="${DATA_DIR}/skillmix"
ensure_dir "${SKILLMIX_DIR}"

TOTAL_SKILLMIX=${#ALL_MODELS[@]}
SM_IDX=0

# Ollama models
for model in "${OLLAMA_MODELS[@]}"; do
    SM_IDX=$((SM_IDX + 1))
    SAFE_NAME=$(sanitize_model "${model}")
    SKILLMIX_FILE="${SKILLMIX_DIR}/skillmix-${SAFE_NAME}.json"

    if [[ -f "$SKILLMIX_FILE" ]]; then
        log "  [${SM_IDX}/${TOTAL_SKILLMIX}] [skip] SkillMix ${model} — result file exists"
        continue
    fi

    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] SkillMix: ${model}"
    OPENAI_BASE_URL="${OLLAMA_URL}" python -m skillmix.runner \
        --tasks "${TASKS_ALL}" \
        --provider openai --models "${model}" \
        --judge-provider anthropic --judge-model claude-opus-4-6 \
        --output-dir "${SKILLMIX_DIR}" -v \
        2>&1 | tee "${LOGS_DIR}/skillmix-${SAFE_NAME}.log"
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] Done: SkillMix ${model}"
done

# GLM-5-turbo
SM_IDX=$((SM_IDX + 1))
SKILLMIX_ZAI="${SKILLMIX_DIR}/skillmix-${SAFE_ZAI}.json"
if [[ -f "$SKILLMIX_ZAI" ]]; then
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] [skip] SkillMix ${ZAI_MODEL} — result file exists"
else
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] SkillMix: ${ZAI_MODEL}"
    OPENAI_API_KEY="${ZHIPU_API_KEY}" python -m skillmix.runner \
        --tasks "${TASKS_ALL}" \
        --provider openai --models "${ZAI_MODEL}" \
        --base-url "${ZAI_BASE_URL}" \
        --judge-provider anthropic --judge-model claude-opus-4-6 \
        --output-dir "${SKILLMIX_DIR}" -v \
        2>&1 | tee "${LOGS_DIR}/skillmix-${SAFE_ZAI}.log"
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] Done: SkillMix ${ZAI_MODEL}"
fi

# claude-opus-4-6
SM_IDX=$((SM_IDX + 1))
SKILLMIX_ANTH="${SKILLMIX_DIR}/skillmix-${SAFE_ANTHROPIC}.json"
if [[ -f "$SKILLMIX_ANTH" ]]; then
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] [skip] SkillMix ${ANTHROPIC_MODEL} — result file exists"
else
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] SkillMix: ${ANTHROPIC_MODEL}"
    python -m skillmix.runner \
        --tasks "${TASKS_ALL}" \
        --provider anthropic --models "${ANTHROPIC_MODEL}" \
        --judge-provider anthropic --judge-model claude-opus-4-6 \
        --output-dir "${SKILLMIX_DIR}" -v \
        2>&1 | tee "${LOGS_DIR}/skillmix-${SAFE_ANTHROPIC}.log"
    log "  [${SM_IDX}/${TOTAL_SKILLMIX}] Done: SkillMix ${ANTHROPIC_MODEL}"
fi

# ===================================================================
# Phase 10: Merge & Visualize
# ===================================================================

log "=== Phase 10: Merge & Visualize ==="

HEATMAPS_DIR="${DATA_DIR}/heatmaps"
ensure_dir "${HEATMAPS_DIR}"

for mode in "${MODES[@]}"; do
    MERGED="${RESULTS_DIR}/results-${mode}-merged.json"
    if [[ -f "$MERGED" ]]; then
        log "  [skip] ${mode} merged file exists"
    else
        log "  Merging ${mode} results"
        python -c "
import json, glob
files = sorted(glob.glob('${RESULTS_DIR}/${mode}/results-*.json'))
merged = []
for f in files:
    with open(f) as fh:
        merged.extend(json.load(fh))
with open('${MERGED}', 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'Merged {len(merged)} episodes from {len(files)} files -> ${MERGED}')
"
    fi

    ensure_dir "${HEATMAPS_DIR}/${mode}"
    log "  Generating heatmaps for ${mode}"
    python -m skillsbench.visualization \
        --results "${MERGED}" \
        -o "${HEATMAPS_DIR}/${mode}" \
        --dpi 200 \
        2>&1 | tee "${LOGS_DIR}/viz-${mode}.log"
done

# ===================================================================
# Summary
# ===================================================================

log "=== Experiment v2 Complete ==="
log "Data directory:    ${DATA_DIR}"
log "Results directory:  ${RESULTS_DIR}"
log "Heatmaps:          ${HEATMAPS_DIR}"
log "SkillMix:          ${SKILLMIX_DIR}"
log ""
log "Models evaluated: ${#OLLAMA_MODELS[@]} Ollama + ${ZAI_MODEL} (Z.AI) + ${ANTHROPIC_MODEL} (Anthropic)"
log "Domains: ${DOMAINS[*]}"
log "Modes: ${MODES[*]}"
