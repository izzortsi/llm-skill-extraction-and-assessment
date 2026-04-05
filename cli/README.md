# cli

unified CLI for the llm-skills pipeline suite. orchestrates all 11 pipeline stages
(1a, 1b, 2-8) across 5 separate pipeline projects via subprocess isolation, with
Rich-powered terminal UI, YAML experiment profiles, and crash recovery.

## quickstart

```bash
cd cli

# zero-setup run using Claude Code CLI (no API keys or servers needed)
python3 -m c4_cli.main run --stages all --minimal --claude-code --clean

# same with a specific model tier
python3 -m c4_cli.main run --stages all --minimal --claude-code sonnet --clean

# minimal end-to-end run via lmproxy/Ollama (fewest API calls)
python3 -m c4_cli.main run --stages all --minimal --clean

# interactive setup: discovers providers, prompts for model selection
python3 -m c4_cli.main config create --interactive

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
  --stages RANGE       all, 1-4, 5-9, extraction, evaluation, skillsbench, skillmix, 1a,3,5
  --clean-stages       wipe only the requested stages' output (not the whole run dir)
  --profile NAME       named profile from profiles/ directory
  --minimal            1 chunk, 2 tasks/chunk, 2 skills, k=2, singlecall, 2 models
  --clean              wipe previous output and re-run from scratch
  --interactive        build profile via guided prompts before running
  --claude-code [TIER] use Claude Code CLI as provider (haiku/sonnet/opus, default: haiku)
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

# zero-setup with Claude Code (default: haiku)
python3 -m c4_cli.main run --stages all --minimal --claude-code --clean

# zero-setup with Claude Code opus
python3 -m c4_cli.main run --stages all --minimal --claude-code opus --clean

# custom profile
python3 -m c4_cli.main run --profile my-experiment --stages all --clean

# interactive mode: discovers providers, prompts for model selection
python3 -m c4_cli.main run --interactive
```

### config

manage experiment profiles stored as YAML files in `profiles/`.

```bash
python3 -m c4_cli.main config list                  # list saved profiles
python3 -m c4_cli.main config show default           # display profile config
python3 -m c4_cli.main config create my-experiment   # create with defaults
python3 -m c4_cli.main config create --interactive   # create via prompts (discovers providers)
python3 -m c4_cli.main config delete my-experiment   # delete a profile
```

after creating a profile, run the pipeline with it:

```bash
# run all stages with a named profile
python3 -m c4_cli.main run --profile my-experiment --stages all --clean

# run extraction stages only
python3 -m c4_cli.main run --profile my-experiment --stages extraction

# run with minimal overrides (reduces chunks, skills, models)
python3 -m c4_cli.main run --profile my-experiment --stages all --minimal --clean
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
all           all 11 stages (1a, 1b, 2, 3, 4, 4b, 5, 6, 7, 8, 9)
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
1a     extract-passages    extraction-pipeline        extract-passages
1b     extract-tasks       extraction-pipeline        extract-tasks
2      capture-traces      extraction-pipeline        capture-traces
3      extract-skills      extraction-pipeline        extract-skills
4      verify-skills       extraction-pipeline        verify-skills --revise
4b     compose-skills      extraction-pipeline        compose-skills
5      corpus-evaluation   skillsbench-evaluation     run-skillsbench
6      visualization       skillsbench-evaluation     heatmaps
7      traceability        extraction-pipeline        traceability-report, export-csv
8      skillmix-evaluation  skillmix-evaluation        run-skillmix, report
9      skillmix-viz         skillmix-evaluation        visualize
```

### LLM usage by stage

- **stage 1b** (extract-tasks): extraction LLM generates tasks from passages
- **stage 2** (capture-traces): trace LLM runs tasks to capture reasoning traces
- **stage 3** (extract-skills): extraction LLM extracts skills from traces
- **stage 4** (verify-skills): rule-based checks first, then extraction LLM revises defective skills using b0.standards/ language rules (up to 2 revision attempts per skill)
- **stage 5** (corpus-evaluation): LLM judge scores model responses against acceptance criteria
- **stage 8** (skillmix-evaluation): LLM judge scores composed-skill responses against acceptance criteria

each stage runs as a subprocess inside its pipeline directory:

    (cd <pipeline-dir> && python3 -m c4_cli.main <command> <args>)

subprocess isolation prevents Python module shadowing between pipelines that share
package names (c0_utils, c1_tools, c2_extraction).

## profiles

profiles are YAML files in `cli/profiles/`. each profile stores all
configuration needed for a pipeline run:

```yaml
profile_name: my-experiment
run_dir: extraction-pipeline/data/pipeline-runs/my-experiment-profile
lmproxy_base_url: "http://localhost:8080/v1"

# source data (stage 1a)
dataset: wikimedia/wikipedia
subset: 20231101.en
domain: language-skills
max_chunks: 5
chunk_size: 4000
tasks_per_chunk: 2

# extraction model (stages 1b, 3)
extraction_provider: lmproxy
extraction_model: claude-opus-4-6

# trace capture (stage 2)
trace_provider: lmproxy
trace_model: claude-opus-4-6

# evaluation (stages 5, 8)
eval_models:
  - provider: lmproxy
    model: qwen2.5-3b
  - provider: ollama
    model: llama3.2:1b
modes:
  - singlecall
  - stepwise
  - guided

# judge (stages 5, 8)
judge_provider: lmproxy
judge_model: claude-opus-4-6

# skill extraction (stage 3)
max_skills: 8

# skill composition (stage 4b)
compose_k_values:
  - 2
  - 3
```

valid provider values: `lmproxy`, `ollama`, `anthropic`, `anthropic-oauth`, `openai`, `claude-code`.

the `--minimal` flag overrides any profile with: 1 chunk, 2 tasks/chunk, 2 skills,
k=2 compositions only, singlecall mode, 2 models.

the `--claude-code` flag overrides all 4 roles (extraction, trace, judge, eval) to
use Claude Code CLI as the provider. accepts a model tier: `haiku` (default),
`sonnet`, or `opus`. requires `claude` CLI on PATH. no API keys or servers needed.

## crash recovery

each stage checks if its output file already exists before running. if the pipeline
fails mid-run, re-running the same command skips completed stages and resumes from
the first incomplete stage. use `--clean` to force a full re-run.

## structure

```
cli/
    c0_config/
        pipeline_stage.py       stage metadata (id, name, pipeline_dir, commands, dependencies)
        pipeline_profile.py     experiment profile dataclass + minimal overrides
        stage_registry.py       registry of all 11 stages with range parser
    c1_tools/
        profile_loader.py       YAML load/save for profiles
        stage_runner.py         subprocess execution of individual stages
        output_inspector.py     run directory inspection and crash recovery
        provider_checker.py     pre-flight provider connectivity checks (lmproxy, ollama, anthropic, zai, iosys, lm-studio, anthropic-oauth)
        provider_discovery.py   runtime provider/model discovery (probes all providers, checks API keys and OAuth tokens)
        claude_code_provider.py ClaudeCodeProvider adapter (claude -p subprocess wrapper)
    c2_orchestration/
        pipeline_executor.py    multi-stage orchestration with dependency resolution + lmproxy session
        stage_output_wirer.py   map profile config to per-stage CLI arguments + provider routing
        config_generator.py     generate models.yaml from profile eval_models
    c4_cli/
        main.py                 entry point: routes run/config/status/setup commands
        command_run.py          run command implementation (--claude-code flag)
        command_config.py       config command implementation (--interactive with discovery)
        command_status.py       status command implementation
        command_setup.py        setup command implementation
        interactive.py          InquirerPy-powered interactive prompts with provider discovery
        rich_ui.py              Rich UI components (panels, tables, indicators)
    profiles/
        default.yaml            default experiment profile
    conftest.py                 pytest bootstrap
```

## providers

the CLI supports 7 LLM provider types:

| provider | what it is | requires |
|----------|-----------|----------|
| `lmproxy` | centralized API gateway (default) | lmproxy running at `lmproxy_base_url` |
| `ollama` | local model server | Ollama running at `ollama_url` |
| `anthropic` | direct Anthropic API | `ANTHROPIC_API_KEY` env var |
| `anthropic-oauth` | Anthropic via Claude Pro/Max subscription (no API cost) | `anthropic-oauth` package + valid OAuth tokens |
| `openai` | direct OpenAI API | `OPENAI_API_KEY` env var |
| `claude-code` | Claude Code CLI subprocess | `claude` on PATH |
| `iosys` | iosys LLM inference API | iosys server at `iosys_base_url`, optional `IOSYS_API_KEY` |
| `lm-studio` | LM Studio local server | LM Studio running at `lm_studio_url` |
| `zai` | Z.AI (Zhipu) API | `ZHIPU_API_KEY` env var |

the interactive setup (`config create --interactive`) probes all providers at startup
and shows which are reachable. `--claude-code` is a convenience flag that sets all
roles to use Claude Code with no additional configuration.

to use `anthropic-oauth`, install the package (`pip install anthropic-oauth[anthropic]`)
and run `anthropic-oauth` once to authenticate via browser. after that, select
`anthropic-oauth` as provider in the interactive config -- no API key needed.

## dependencies

- Python 3.8+
- rich (optional, for colored terminal output; degrades to plain text if absent)
- InquirerPy (optional, for arrow-key selection in interactive mode; falls back to input())
- pyyaml (for profile loading)
- requests (for provider discovery probes)

the pipeline projects (extraction-pipeline, skillsbench-evaluation,
skillmix-evaluation) have their own dependencies (anthropic, openai, datasets,
matplotlib). run `setup` to check.
