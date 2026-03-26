---
name: cond-render-contradiction-verdict-then-extract-parallel-claims
description: Conditional skill: if Render Contradiction Verdict, then apply Extract Parallel Claims. Use when skill application depends on a condition.
---

# Cond Render Contradiction Verdict Then Extract Parallel Claims

## When to Use

- When Render Contradiction Verdict determines the path
- When a factual comparison between a source claim and an external claim has been completed and a final judgment must be stated.

## Procedure

1. Evaluate the condition:
2. 1. Check Render Contradiction Verdict
3. 2. If condition is TRUE:
4.    1. Apply Extract Parallel Claims
5. 3. If condition is FALSE:
6.    No action (or apply default behavior)

## Constraints

- Condition must be evaluable: Render Contradiction Verdict
- Only one branch (then or else) is executed
- The verdict must be grounded solely in the source passage, not external knowledge.
- The verdict must explicitly state what the correct information is according to the source.

### Example 1: Condition is true

**Input:**
Input where Render Contradiction Verdict evaluates to TRUE

**Process:**
1. Render Contradiction Verdict evaluates to TRUE
2. Apply then-skills: Extract Parallel Claims

**Output:**
Output from applying Extract Parallel Claims

