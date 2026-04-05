# extraction-pipeline

unified extraction library for the llm-skills pipeline. handles stages 1a through 4b
(text extraction, task generation, trace capture, skill extraction, verification, and
composition) plus stage 7 (traceability report and CSV export).

## what this project does

extracts procedural skills from text corpora through a multi-stage pipeline:
1. extract text passages from datasets (Wikipedia, etc.)
2. generate evaluation tasks from passages using an LLM
3. capture reasoning traces by running tasks through an LLM
4. extract reusable procedural skills from traces
5. verify skill quality via rule-based checks, then LLM revision of defective skills
6. compose atomic skills into multi-skill combinations (k=2..5, operators: seq/par/cond)
7. generate traceability reports linking passages to tasks to skills

## commands

```bash
cd extraction-pipeline
python -m cli.main extract-passages  --help   # stage 1a
python -m cli.main extract-tasks     --help   # stage 1b
python -m cli.main capture-traces    --help   # stage 2
python -m cli.main extract-skills    --help   # stage 3
python -m cli.main verify-skills     --help   # stage 4
python -m cli.main compose-skills    --help   # stage 4b
python -m cli.main traceability-report --help  # stage 7
python -m cli.main export-csv        --help   # stage 7
python -m cli.main format            --help   # JSON/markdown converter
```

## output structure

```
data/pipeline-runs/default-profile/
    stage1-task-extraction/
        passages.json, tasks.json, tasks-md/
    stage2-trace-capture/
        traces.jsonl
    stage3-skill-extraction/
        skills.json, skills-md/
    stage4-skill-verification/
        verified_skills.json, verified-skills-md/
    stage4b-skill-composition/
        atomic-skills-md/, k2/, k3/, k4/, k5/ (composed skills by size)
    stage5-corpus-evaluation/
        {singlecall,stepwise,guided}/results-all.json
    stage6-visualization/
        {singlecall,stepwise,guided}/, cross-mode/
    stage8-skillmix-evaluation/
        episodes.json, summary.json, report.txt
    stage9-skillmix-visualization/
        score_by_k.png, operator_heatmap.png, uplift_heatmap.png,
        k_operator_heatmap.png, baseline_vs_skill.png, win_loss.png
    traceability-report.txt
    csv/
        skills.csv, skill_instances.csv
    logs/
```

## structure

```
utils/          shared utilities (uid, text_utils, data_structures)
schemas/          data contracts (ExtractedTask, ExtractedSkill)
tools/          file loaders, skill registry, JSON/markdown formatter
extraction/     passage, task, trace, skill extraction + verification
composition/    skill composition operators (seq, par, cond, sem)
cli/            main.py with 11 commands
```

## dependencies

- anthropic (task/skill extraction, trace capture, skill revision)
- openai (Ollama model access)
- datasets (HuggingFace dataset loading)
- pyyaml (skill markdown parsing)
- llm-providers (LLM provider abstraction, used by stage 4 --revise)
