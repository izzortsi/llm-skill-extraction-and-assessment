# llm-skills.skillsbench-evaluation

corpus-based evaluation measuring whether skill injection improves LLM task performance.
stages 5-6 of the llm-skills pipeline.

## what this project does

runs each task against multiple models in two conditions: baseline (no skill) and curated
(skill injected into prompt). an LLM judge scores responses. supports three scaffolding modes:
singlecall (one LLM call), stepwise (multi-turn, one action primitive per step), and guided
(multi-turn, following the skill's procedure steps).

generates heatmaps and charts comparing baseline vs curated pass rates across models and modes.

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
    --tasks ../llm-skills.extraction-pipeline/pipeline-runs/default/stage1-task-extraction/tasks.json \
    --skills ../llm-skills.extraction-pipeline/pipeline-runs/default/stage4-skill-verification/verified_skills.json \
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

- llm-skills.llm-providers (LLM provider abstraction, model config)
- llm-skills.extraction-pipeline (ExtractedTask, ExtractedSkill types via c1_types/)
