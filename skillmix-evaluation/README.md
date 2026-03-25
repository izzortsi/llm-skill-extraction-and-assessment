# llm-skills.skillmix-evaluation

SkillMix composition quality evaluation. measures whether composed skills (multi-skill
combinations) improve LLM performance compared to atomic skills.

## what this project does

runs the SkillMix benchmark: evaluates models on tasks using composed skills (sequential,
parallel, conditional, semantic compositions). compares composed-skill performance against
baseline and single-skill conditions. generates per-episode results and summary reports.

## commands

```bash
cd llm-skills.skillmix-evaluation
python -m c4_cli.main run-skillmix --help  # run SkillMix experiment
python -m c4_cli.main report       --help  # generate summary report
```

## structure

```
c2_analytics/      summary statistics
c3_skillmix/       harness, runner, report
c4_cli/            main.py, run_skillmix.py, report.py
```

## dependencies

- llm-skills.llm-providers (LLM provider abstraction)
- llm-skills.skillsbench-evaluation (shared evaluation types)
