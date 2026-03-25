# llm-skills.skillsbench-evaluation

corpus-based evaluation measuring whether skill injection improves LLM task performance.
stages 5-6 of the llm-skills pipeline.

## what this project does

runs each task against multiple models in two conditions: baseline (no skill) and curated
(skill injected into prompt). an LLM judge (configured via models.yaml) scores each model
response against acceptance criteria. supports three scaffolding modes: singlecall (one LLM
call), stepwise (multi-turn, one action primitive per step), and guided (multi-turn, following
the skill's procedure steps).

generates heatmaps and charts comparing baseline vs curated pass rates across models and modes.

## judging

the LLM judge (LLMJudgeEvaluator) evaluates each model response by comparing the response
text against the task's passage, challenge, and acceptance_criteria. the judge model is
configured in models.yaml under the `judge:` section (defaults to claude-opus-4-6). the
judge produces a score (0.0-1.0), a pass/fail boolean, and a rationale string explaining
the assessment.

## commands

```bash
cd llm-skills.skillsbench-evaluation
python -m c4_cli.main run-skillsbench --help  # run corpus evaluation (stage 5)
python -m c4_cli.main heatmaps       --help  # generate visualizations (stage 6)
```

## examples

```bash
# evaluate with config-driven model routing
python -m c4_cli.main run-skillsbench \
    --tasks ../llm-skills.extraction-pipeline/data/pipeline-runs/default-profile/stage1-task-extraction/tasks.json \
    --skills ../llm-skills.extraction-pipeline/data/pipeline-runs/default-profile/stage4-skill-verification/verified_skills.json \
    --config ../llm-skills.llm-providers/configs/models.yaml \
    --mode singlecall \
    --output results.json -v

# generate heatmaps from results
python -m c4_cli.main heatmaps \
    --results results.json \
    -o heatmaps/ --type all --dpi 200
```

## structure

```
c0_config/         experiment_config, skill_injection, trial_result
c2_evaluation/     llm_judge, effectiveness, proof_verifier
c3_skillsbench/    corpus_harness, visualization
c4_cli/            main.py, run_skillsbench.py
scripts/           run_experiment.sh, run_experiment_v2.sh, run_validation.sh
tests/             test_litellm_provider, test_model_config
```

## dependencies

- llm-skills.llm-providers (LLM provider abstraction, model config, judge config)
- llm-skills.extraction-pipeline (ExtractedTask, ExtractedSkill types via c1_types/)
