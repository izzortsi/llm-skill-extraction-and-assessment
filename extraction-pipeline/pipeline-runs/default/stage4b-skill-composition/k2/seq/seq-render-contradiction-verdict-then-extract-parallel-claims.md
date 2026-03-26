---
name: seq-render-contradiction-verdict-then-extract-parallel-claims
description: Apply skills in sequence: Use when When a factual comparison between a source claim and an external claim has been completed and a final judgment must be stated. -> Use when When a student's or external claim needs to be checked against a source passage and the relevant statements must first be isolated before any comparison can occur.. Use when multiple capabilities are needed in order.
---

# Seq Render Contradiction Verdict Then Extract Parallel Claims

## When to Use

- When a factual comparison between a source claim and an external claim has been completed and a final judgment must be stated.
- When a student's or external claim needs to be checked against a source passage and the relevant statements must first be isolated before any comparison can occur.

## Procedure

1. The analyst applies the result of the factual comparison to render a verdict on whether the student's or external claim is correct or incorrect according to the source passage.
2. Apply Extract Parallel Claims: The analyst observes the source passage to locate the specific factual claim relevant to the topic in question.
3. Apply Extract Parallel Claims: The analyst observes the student's or external party's statement to locate their corresponding claim on the same topic.

## Constraints

- The verdict must be grounded solely in the source passage, not external knowledge.
- The verdict must explicitly state what the correct information is according to the source.
- Both claims must address the same topic or entity for extraction to be meaningful.
- The source passage must contain an explicit statement on the topic, not merely an implication.

### Example 1: Sequential application of Render Contradiction Verdict -> Extract Parallel Claims

**Input:**
[Input requiring this skill]

**Process:**
Apply the following skills in sequence:
1. Apply Render Contradiction Verdict
2. Apply Extract Parallel Claims

**Output:**
[Output from Extract Parallel Claims]

