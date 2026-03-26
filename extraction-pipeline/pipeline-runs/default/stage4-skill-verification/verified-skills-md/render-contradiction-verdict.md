---
name: render-contradiction-verdict
description: Use when When a factual comparison between a source claim and an external
  claim has been completed and a final judgment must be stated.
skill_uid: db9a-92fd-1fef-c110
source_task_uids:
- 82ea-9d89-c843-4163
extraction_method: opus-4-6-skill-extraction-v1
---

# render-contradiction-verdict

Produces a final correctness judgment by applying the identified contradiction (or consistency) to evaluate the external claim against the authoritative source.

## When to Use

- When a factual comparison between a source claim and an external claim has been completed and a final judgment must be stated.

## Procedure

1. The analyst applies the result of the factual comparison to render a verdict on whether the student's or external claim is correct or incorrect according to the source passage.

## Constraints

- The verdict must be grounded solely in the source passage, not external knowledge.
- The verdict must explicitly state what the correct information is according to the source.
