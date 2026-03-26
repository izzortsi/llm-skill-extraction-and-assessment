---
name: extract-parallel-claims
description: Use when When a student's or external claim needs to be checked against
  a source passage and the relevant statements must first be isolated before any comparison
  can occur.
skill_uid: d7be-47bf-73d7-e2f6
source_task_uids:
- 82ea-9d89-c843-4163
extraction_method: opus-4-6-skill-extraction-v1
---

# extract-parallel-claims

Locates and isolates the specific claim made in a source passage and the corresponding claim made by a student or external party on the same topic.

## When to Use

- When a student's or external claim needs to be checked against a source passage and the relevant statements must first be isolated before any comparison can occur.

## Procedure

1. The analyst observes the source passage to locate the specific factual claim relevant to the topic in question.
2. The analyst observes the student's or external party's statement to locate their corresponding claim on the same topic.

## Constraints

- Both claims must address the same topic or entity for extraction to be meaningful.
- The source passage must contain an explicit statement on the topic, not merely an implication.
