# llm-skills.skillmix-evaluation

SkillMix composition quality evaluation. measures whether composed skills (multi-skill
combinations) improve LLM performance compared to baseline (no skill). stage 8 of the
llm-skills pipeline.

## what this project does

runs the SkillMix benchmark: evaluates models on tasks using composed skills (sequential,
parallel, conditional, semantic compositions). compares composed-skill performance against
baseline and single-skill conditions. generates per-episode results, summary reports, and
visualization charts.

## commands

```bash
cd llm-skills.skillmix-evaluation
python -m c4_cli.main run-skillmix --help  # run SkillMix experiment
python -m c4_cli.main report       --help  # generate summary report
python -m c4_cli.main visualize    --help  # generate charts from results
```

## examples

```bash
# run evaluation with Ollama models
python -m c4_cli.main run-skillmix \
    --tasks ../llm-skills.extraction-pipeline/pipeline-runs/default/stage1-task-extraction/tasks.json \
    --skills-dir ../llm-skills.extraction-pipeline/pipeline-runs/default/stage4b-skill-composition/atomic-skills-md \
    --models qwen2.5-3b,qwen2.5-7b \
    --base-url http://localhost:11434/v1 \
    --output-dir results/ -v

# generate text report
python -m c4_cli.main report \
    --results-dir results/ \
    --output results/report.txt

# generate charts
python -m c4_cli.main visualize \
    --results-dir results/ \
    --output-dir results/charts/ --dpi 200
```

## output files

```
stage8-skillmix-evaluation/
    episodes.json           per-episode results (task, skill, model, condition, score)
    summary.json            per-model aggregate statistics (baseline_mean, skill_mean, delta)
    report.txt              text summary report
    charts/
        baseline_vs_skill.png   grouped bar: baseline vs skill-injected per model
        delta_by_model.png      bar chart: uplift (delta) per model
        skill_heatmap.png       heatmap: mean score per skill x model
        win_loss.png            stacked bar: win/tie/loss per model
```

## structure

```
c2_analytics/      summary statistics, visualization charts
c3_skillmix/       harness, runner, report
c4_cli/            main.py, run_skillmix.py, report.py, visualize.py
```

## dependencies

- llm-skills.llm-providers (LLM provider abstraction)
- llm-skills.skillsbench-evaluation (shared evaluation types: LLMJudgeEvaluator, skill_injection)
- matplotlib (chart generation)
