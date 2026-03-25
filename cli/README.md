# llm-skills.cli

unified CLI for the llm-skills pipeline suite. orchestrates all 11 pipeline stages
(1a, 1b, 2-8) across 5 separate pipeline projects via subprocess isolation, with
Rich-powered terminal UI, YAML experiment profiles, and crash recovery.

## quickstart

```bash
cd llm-skills.cli

# minimal end-to-end run (fewest API calls)
python3 -m c4_cli.main run --stages all --minimal --clean

# check environment before running
python3 -m c4_cli.main setup

# see what completed
python3 -m c4_cli.main status
```

## commands

### run

execute pipeline stages with profile-based configuration.

```bash
python3 -m c4_cli.main run [options]

options:
  --stages RANGE       all, 1-4, 5-8, extraction, evaluation, skillsbench, skillmix, 1a,3,5
  --profile NAME       named profile from profiles/ directory
  --minimal            1 chunk, 2 tasks/chunk, 2 skills, k=2, singlecall, 2 models
  --clean              wipe previous output and re-run from scratch
  --interactive        build profile via guided prompts before running
  --run-dir PATH       override output directory
  --quiet              suppress stage subprocess output
```

examples:

```bash
# full pipeline with defaults (5 chunks, all modes, all models)
python3 -m c4_cli.main run --stages all

# extraction only (stages 1a through 4)
python3 -m c4_cli.main run --stages extraction

# evaluation only (stages 5-9, requires prior extraction output)
python3 -m c4_cli.main run --stages evaluation

# skillsbench only (stages 5-6)
python3 -m c4_cli.main run --stages skillsbench

# skillmix only (stages 8-9, requires stages 1b + 4b output)
python3 -m c4_cli.main run --stages skillmix

# single stage
python3 -m c4_cli.main run --stages 3

# custom profile
python3 -m c4_cli.main run --profile my-experiment --stages all --clean

# interactive mode: prompts for provider, model, dataset, modes
python3 -m c4_cli.main run --interactive
```

### config

manage experiment profiles stored as YAML files in `profiles/`.

```bash
python3 -m c4_cli.main config list                  # list saved profiles
python3 -m c4_cli.main config show default           # display profile config
python3 -m c4_cli.main config create my-experiment   # create with defaults
python3 -m c4_cli.main config create --interactive   # create via prompts
python3 -m c4_cli.main config delete my-experiment   # delete a profile
```

### status

show which stages have completed output in the run directory.

```bash
python3 -m c4_cli.main status                       # default run directory
python3 -m c4_cli.main status --profile my-experiment
python3 -m c4_cli.main status --run-dir /path/to/run
```

### setup

pre-flight environment checks: provider connectivity, API keys, Python packages.

```bash
python3 -m c4_cli.main setup                        # check default profile
python3 -m c4_cli.main setup --profile my-experiment
```

## stage range syntax

the `--stages` flag accepts these formats:

```
all           all 11 stages (1a, 1b, 2, 3, 4, 4b, 5, 6, 7, 8)
extraction    stages 1a through 4b (text + skill extraction + composition)
evaluation    stages 5 through 9 (eval + viz + traceability + skillmix)
skillsbench   stages 5-6 (skillsbench eval + visualization)
skillmix      stages 8-9 (skillmix eval + report + visualization)
1-4           range from stage 1a to stage 4
5-9           range from stage 5 to stage 9
1a,1b,5       comma-separated specific stages
3             single stage
```

## pipeline stages

```
stage  name                pipeline project                      commands
-----  ------------------  ------------------------------------  ----------------------------
1a     extract-passages    llm-skills.extraction-pipeline        extract-passages
1b     extract-tasks       llm-skills.extraction-pipeline        extract-tasks
2      capture-traces      llm-skills.extraction-pipeline        capture-traces
3      extract-skills      llm-skills.extraction-pipeline        extract-skills
4      verify-skills       llm-skills.extraction-pipeline        verify-skills
4b     compose-skills      llm-skills.extraction-pipeline        compose-skills
5      corpus-evaluation   llm-skills.skillsbench-evaluation     run-skillsbench
6      visualization       llm-skills.skillsbench-evaluation     heatmaps
7      traceability        llm-skills.extraction-pipeline        traceability-report, export-csv
8      skillmix-evaluation  llm-skills.skillmix-evaluation        run-skillmix, report
9      skillmix-viz         llm-skills.skillmix-evaluation        visualize
```

each stage runs as a subprocess inside its pipeline directory:

    (cd <pipeline-dir> && python3 -m c4_cli.main <command> <args>)

subprocess isolation prevents Python module shadowing between pipelines that share
package names (c0_utils, c1_tools, c2_extraction).

## profiles

profiles are YAML files in `llm-skills.cli/profiles/`. each profile stores all
configuration needed for a pipeline run:

```yaml
profile_name: my-experiment
run_dir: llm-skills.extraction-pipeline/data/pipeline-runs/my-experiment-profile

# source data (stage 1a)
dataset: wikimedia/wikipedia
subset: 20231101.en
domain: language-skills
max_chunks: 5
chunk_size: 4000
tasks_per_chunk: 2

# extraction model (stages 1b, 3)
extraction_provider: anthropic
extraction_model: claude-opus-4-6

# trace capture (stage 2)
trace_provider: anthropic
trace_model: claude-opus-4-6

# evaluation (stages 5, 8)
config_file: llm-skills.llm-providers/configs/models.yaml
modes:
  - singlecall
  - stepwise
  - guided

# judge (stages 5, 8)
judge_provider: anthropic
judge_model: claude-opus-4-6

# skill extraction (stage 3)
max_skills: 8
```

the `--minimal` flag overrides any profile with: 1 chunk, 2 tasks/chunk, 2 skills,
k=2 compositions only, singlecall mode, 2 Ollama models (qwen2.5-3b, qwen2.5-7b),
no ZAI or Anthropic eval.

## crash recovery

each stage checks if its output file already exists before running. if the pipeline
fails mid-run, re-running the same command skips completed stages and resumes from
the first incomplete stage. use `--clean` to force a full re-run.

## structure

```
llm-skills.cli/
    c0_config/
        pipeline_stage.py       stage metadata (id, name, pipeline_dir, commands, dependencies)
        pipeline_profile.py     experiment profile dataclass + minimal overrides
        stage_registry.py       registry of all 11 stages with range parser
    c1_tools/
        profile_loader.py       YAML load/save for profiles
        stage_runner.py         subprocess execution of individual stages
        output_inspector.py     run directory inspection and crash recovery
        provider_checker.py     pre-flight provider connectivity checks
    c2_orchestration/
        pipeline_executor.py    multi-stage orchestration with dependency resolution
        stage_output_wirer.py   map profile config to per-stage CLI arguments
    c4_cli/
        main.py                 entry point: routes run/config/status/setup commands
        command_run.py          run command implementation
        command_config.py       config command implementation
        command_status.py       status command implementation
        command_setup.py        setup command implementation
        interactive.py          Rich-powered interactive prompts
        rich_ui.py              Rich UI components (panels, tables, indicators)
    profiles/
        default.yaml            default experiment profile
    conftest.py                 pytest bootstrap
```

## dependencies

- Python 3.8+
- rich (optional, for colored terminal output; degrades to plain text if absent)
- pyyaml (for profile loading)

the pipeline projects (llm-skills.extraction-pipeline, llm-skills.skillsbench-evaluation,
llm-skills.skillmix-evaluation) have their own dependencies (anthropic, openai, datasets,
matplotlib). run `setup` to check.
