---
name: cond-identify-factual-contradiction-then-render-contradiction-verdict-and-extract-parallel-claims
description: Conditional skill: if Identify Factual Contradiction, then apply Render Contradiction Verdict -> Extract Parallel Claims. Use when skill application depends on a condition.
---

# Cond Identify Factual Contradiction Then Render Contradiction Verdict And Extract Parallel Claims

## When to Use

- When Identify Factual Contradiction determines the path
- When two parallel claims on the same topic have been extracted and the analyst needs to determine whether they agree or conflict.

## Procedure

1. Evaluate the condition:
2. 1. Check Identify Factual Contradiction
3. 2. If condition is TRUE:
4.    1. Apply Render Contradiction Verdict
5.    2. Apply Extract Parallel Claims
6. 3. If condition is FALSE:
7.    No action (or apply default behavior)

## Constraints

- Condition must be evaluable: Identify Factual Contradiction
- Only one branch (then or else) is executed
- The comparison must be between claims on the same attribute or relationship.
- A contradiction requires that the two claims cannot both be true simultaneously.

### Example 1: Condition is true

**Input:**
Input where Identify Factual Contradiction evaluates to TRUE

**Process:**
1. Identify Factual Contradiction evaluates to TRUE
2. Apply then-skills: Render Contradiction Verdict -> Extract Parallel Claims

**Output:**
Output from applying Render Contradiction Verdict -> Extract Parallel Claims

