---
name: seq-identify-factual-contradiction-then-render-contradiction-verdict-then-extract-parallel-claims
description: Apply skills in sequence: Use when When two parallel claims on the same topic have been extracted and the analyst needs to determine whether they agree or conflict. -> Use when When a factual comparison between a source claim and an external claim has been completed and a final judgment must be stated. -> Use when When a student's or external claim needs to be checked against a source passage and the relevant statements must first be isolated before any comparison can occur.. Use when multiple capabilities are needed in order.
---

# Seq Identify Factual Contradiction Then Render Contradiction Verdict Then Extract Parallel Claims

## When to Use

- When two parallel claims on the same topic have been extracted and the analyst needs to determine whether they agree or conflict.
- When a factual comparison between a source claim and an external claim has been completed and a final judgment must be stated.
- When a student's or external claim needs to be checked against a source passage and the relevant statements must first be isolated before any comparison can occur.

## Procedure

1. The analyst distinguishes the specific factual element in the student's claim from the corresponding factual element in the source passage (e.g., a name, date, or category).
2. The analyst compares the two factual elements to determine whether they are consistent or contradictory.
3. Apply Render Contradiction Verdict: The analyst applies the result of the factual comparison to render a verdict on whether the student's or external claim is correct or incorrect according to the source passage.
4. Apply Extract Parallel Claims: The analyst observes the source passage to locate the specific factual claim relevant to the topic in question.
5. Apply Extract Parallel Claims: The analyst observes the student's or external party's statement to locate their corresponding claim on the same topic.

## Constraints

- The comparison must be between claims on the same attribute or relationship.
- A contradiction requires that the two claims cannot both be true simultaneously.
- The verdict must be grounded solely in the source passage, not external knowledge.
- The verdict must explicitly state what the correct information is according to the source.
- Both claims must address the same topic or entity for extraction to be meaningful.
- The source passage must contain an explicit statement on the topic, not merely an implication.

### Example 1: Sequential application of Identify Factual Contradiction -> Render Contradiction Verdict -> Extract Parallel Claims

**Input:**
[Input requiring this skill]

**Process:**
Apply the following skills in sequence:
1. Apply Identify Factual Contradiction
2. Apply Render Contradiction Verdict
3. Apply Extract Parallel Claims

**Output:**
[Output from Extract Parallel Claims]

