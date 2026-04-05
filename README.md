# llm-skill-extraction-and-assessment

an 11-stage pipeline suite for extracting verifiable skills from text corpora
and evaluating LLMs against those skills. extracted as a standalone monorepo
from [kcg-ml-llm](https://github.com/kk-digital/kcg-ml-llm).

## structure

```
llm-skill-extraction-and-assessment/
├── cli/                     unified CLI orchestrator (see cli/README.md)
├── extraction-pipeline/     stages 1a-4b: passages -> tasks -> traces -> skills -> compositions
├── skillsbench-evaluation/  stages 5-6: corpus-judge eval + heatmaps
├── skillmix-evaluation/     stages 8-9: composed-skill eval + reports
└── llm-providers/           shared LLM provider adapters (lmproxy, ollama, anthropic,
                             openai, claude-code, anthropic-oauth, iosys, lm-studio, zai)
```

## quickstart

```bash
cd cli
python3 -m cli.main run --stages all --minimal --claude-code --clean
```

see [cli/README.md](cli/README.md) for full CLI docs: profiles, stage ranges,
provider configuration, crash recovery.

## pipeline stages

| stage | name               | subproject              |
|-------|--------------------|-------------------------|
| 1a    | extract-passages   | extraction-pipeline     |
| 1b    | extract-tasks      | extraction-pipeline     |
| 2     | capture-traces     | extraction-pipeline     |
| 3     | extract-skills     | extraction-pipeline     |
| 4     | verify-skills      | extraction-pipeline     |
| 4b    | compose-skills     | extraction-pipeline     |
| 5     | corpus-evaluation  | skillsbench-evaluation  |
| 6     | visualization      | skillsbench-evaluation  |
| 7     | traceability       | extraction-pipeline     |
| 8     | skillmix-evaluation | skillmix-evaluation    |
| 9     | skillmix-viz       | skillmix-evaluation     |

## dependencies

python 3.8+; see [cli/requirements.txt](cli/requirements.txt). each subproject
may have additional runtime dependencies (datasets, matplotlib, openai,
litellm, anthropic).

## layout conventions

each subproject uses conventional python package names:

| folder         | purpose                                    |
|----------------|--------------------------------------------|
| `config/`      | experiment configuration + stage metadata  |
| `utils/`       | shared utility modules                     |
| `tools/`       | reusable tool modules                      |
| `schemas/`     | data-type definitions (ExtractedSkill, ExtractedTask) |
| `providers/`   | LLM provider adapters                      |
| `extraction/`  | extraction-stage implementations           |
| `composition/` | skill-composition operators                |
| `evaluation/`  | evaluation harness + metrics               |
| `analytics/`   | analysis + aggregation                     |
| `skillsbench/` | corpus-judge scoring                       |
| `skillmix/`    | skillmix composed-skill evaluation         |
| `orchestration/` | multi-stage pipeline orchestration       |
| `cli/`         | CLI entry point + commands                 |

subprocess isolation is used between pipelines that share package names —
each stage runs in its own subprocess with only that subproject on sys.path.
