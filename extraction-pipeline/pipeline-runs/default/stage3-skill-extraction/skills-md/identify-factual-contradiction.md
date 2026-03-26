---
name: identify-factual-contradiction
description: Use when When two parallel claims on the same topic have been extracted
  and the analyst needs to determine whether they agree or conflict.
skill_uid: 740d-3be6-71ad-f0dd
source_task_uids:
- 82ea-9d89-c843-4163
extraction_method: opus-4-6-skill-extraction-v1
---

# identify-factual-contradiction

Determines whether two extracted claims on the same topic contradict each other by distinguishing the key differing elements and comparing them.

## When to Use

- When two parallel claims on the same topic have been extracted and the analyst needs to determine whether they agree or conflict.

## Procedure

1. The analyst distinguishes the specific factual element in the student's claim from the corresponding factual element in the source passage (e.g., a name, date, or category).
2. The analyst compares the two factual elements to determine whether they are consistent or contradictory.

## Constraints

- The comparison must be between claims on the same attribute or relationship.
- A contradiction requires that the two claims cannot both be true simultaneously.
