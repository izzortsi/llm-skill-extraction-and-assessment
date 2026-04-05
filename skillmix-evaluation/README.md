# skillmix-evaluation

SkillMix composition quality evaluation. measures whether composed skills (multi-skill
combinations) improve LLM performance compared to baseline (no skill). stage 8 of the
llm-skills pipeline.

## what this project does

runs the SkillMix benchmark: evaluates models on tasks using composed skills (sequential,
parallel, conditional compositions at k=2..5). compares composed-skill performance against
baseline and atomic-skill conditions. an LLM judge scores each model response against
acceptance criteria. generates per-episode results, summary reports, and visualization
charts broken down by composition size (k) and operator type (seq/par/cond).

## judging

the LLM judge (LLMJudgeEvaluator from skillsbench-evaluation) evaluates each
model response by comparing the response text against the task's passage, challenge, and
acceptance_criteria. the judge model is configured via models.yaml (defaults to
claude-opus-4-6). when --config is provided, the judge is resolved from the config's
`judge:` section. without --config, --judge-provider and --judge-model flags are used.

## commands

```bash
cd skillmix-evaluation
python -m cli.main run-skillmix --help  # run SkillMix experiment (stage 8)
python -m cli.main report       --help  # generate summary report (stage 8)
python -m cli.main visualize    --help  # generate charts (stage 9)
```

## examples

```bash
# run evaluation with config-driven model routing
python -m cli.main run-skillmix \
    --tasks ../extraction-pipeline/data/pipeline-runs/default-profile/stage1-task-extraction/tasks.json \
    --skills-dir ../extraction-pipeline/data/pipeline-runs/default-profile/stage4b-skill-composition \
    --models qwen2.5-3b,qwen2.5-7b \
    --config ../llm-providers/configs/models.yaml \
    --output-dir results/ -v

# generate text report
python -m cli.main report \
    --results-dir results/ \
    --output results/report.txt

# generate charts
python -m cli.main visualize \
    --results-dir results/ \
    --output-dir charts/ --dpi 200
```

## output files

```
stage8-skillmix-evaluation/
    episodes.json           per-episode results (task, skill, model, condition, score)
    summary.json            per-model aggregate statistics (baseline_mean, skill_mean, delta)
    report.txt              text summary report

stage9-skillmix-visualization/
    score_by_k.png          line: mean score by composition size (k), one line per model
    operator_heatmap.png    heatmap: operator type (seq/par/cond/atomic) x model
    uplift_heatmap.png      heatmap: skill x model, diverging delta from baseline
    k_operator_heatmap.png  heatmap: (k, operator) combinations x model
    baseline_vs_skill.png   grouped bar: baseline vs skill-injected per model
    win_loss.png            stacked bar: win/tie/loss per model
```

the visualizer parses composition metadata from skill names:
- operator type: `seq-`, `par-`, `cond-` prefix (no prefix = atomic)
- k-value: number of atomic skills in the composition (counted from separators)

## structure

```
analytics/      summary statistics, visualization charts
skillmix/       harness, runner, report
cli/            main.py, run_skillmix.py, report.py, visualize.py
```

## dependencies

- llm-providers (LLM provider abstraction, model config, judge config)
- skillsbench-evaluation (shared evaluation types: LLMJudgeEvaluator, skill_injection)
- matplotlib (chart generation)
