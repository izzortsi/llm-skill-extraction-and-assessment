"""
proof_verifier.py

Structural verifier for intuitionistic propositional calculus (IPC) proofs.
Standalone version for skills-bench -- no dependency on grpt-training-data-pipeline.

Tier 1: structural rule checking (deterministic, no LLM)
  - rule citation validity against known IPC rules
  - classical contamination detection (excluded middle, DNE, etc.)
  - scope tracking for discharged assumptions
  - conclusion verification against target formula

Tier 2: llm judge (called externally via judge_with_llm(), not during verify())

Scoring:
  each valid step: +1
  each scope violation: -2
  each classical contamination: -3
  missing conclusion: -5
  normalized to [0, 1]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


class ProofVerifier:
    """Structural verifier for intuitionistic propositional calculus proofs."""

    VALID_RULES: set[str] = {
        "assumption", "assume", "premise", "hypothesis", "given", "hyp",
        "conjunction-introduction", "and-intro", "\u2227I", "\u2227i", "\u2227-intro", "conj-intro",
        "conjunction-elimination", "and-elim", "\u2227E", "\u2227e", "\u2227-elim", "conj-elim",
        "simplification",  # common alias for \u2227E
        "disjunction-introduction", "or-intro", "\u2228I", "\u2228i", "\u2228-intro", "disj-intro",
        "disjunction-elimination", "or-elim", "\u2228E", "\u2228e", "\u2228-elim", "disj-elim", "proof-by-cases",
        "implication-introduction", "impl-intro", "\u21d2I", "\u21d2i", "\u21d2-intro", "deduction-theorem",
        "\u2192I", "\u2192i", "\u2192-intro",  # -> variants (U+2192)
        "implication-elimination", "modus-ponens", "mp", "\u21d2E", "\u21d2e", "\u21d2-elim",
        "\u2192E", "\u2192e", "\u2192-elim",  # -> variants
        "modus tollens", "mt",
        "hypothetical syllogism", "hs",
        "transitivity",  # alias for hypothetical syllogism
        "negation-introduction", "neg-intro", "\u00acI", "\u00aci", "\u00ac-intro", "refutation",
        "negi", "\u00ac-i", "\u00ac-intro",
        "negation-elimination", "neg-elim", "\u00acE", "\u00ace", "\u00ac-elim",
        "explosion", "ex-falso", "exfalso", "explode", "\u22a5E", "\u22a5e", "\u22a5-elim", "efq",
        "\u22a5I", "\u22a5i", "\u22a5-intro", "\u22a5r",  # bottom-I, bottom-i, bottom-R (contradiction intro)
        "contradiction",  # common alias for bottom-I / deriving bottom
        "discharge",
        "contrapositive", "contraposition",
        "biconditional-introduction", "biconditional-elimination",
        "\u2194I", "\u2194i", "\u2194E", "\u2194e", "iff-intro", "iff-elim",
        "subproof", "subproof introduction",
        "reiteration", "reit",
        "prem",  # abbreviation for premise
        "disjunctive syllogism", "ds",  # derived rule: A\/B, not-A |- B
        "conjunction", "disjunction",  # bare rule names (ambiguous but valid)
        "dn", "double negation", "double negation introduction", "dni",  # DNI is valid in IPC
        # Coq/Lean-style abbreviations
        "andi", "ande", "ori", "ore", "impi", "impe", "noti",
        "equivi", "equive",  # biconditional intro/elim (Coq-style)
        # Additional aliases
        "reflexivity", "rfl",  # identity: A entails A
        "axiom",  # generic axiom citation
        "\u22a5d", "\u22a5f", "exfalsum",  # bottom-destruction / falsum variants
        "\u00ac\u00aci",  # double negation introduction (symbol form)
        "proof by assumption",
        "imp", "cp",  # implication / conditional proof abbreviations
        "right-simplification", "left-simplification",  # conjunction-elimination variants
        "disjintro", "disjelim", "conjintro", "conjelim",  # CamelCase variants
        # v5 additions: missing aliases found in failing episodes
        "conji", "conje",  # conjunction intro/elim abbreviations
        "neg-e",  # negation elimination with hyphen
        "ass",  # abbreviation for assumption
        "new assumption", "new premise",  # synonyms for assumption/premise
    }

    CLASSICAL_PATTERNS: list[tuple[str, str]] = [
        (r"excluded\s+middle|tertium\s+non\s+datur|lem\b|law\s+of\s+excluded\s+middle", "excluded middle"),
        (r"double\s+negation\s+elimination|dne\b", "double negation elimination"),
        (r"peirce.*law", "Peirce's law"),
        (r"reverse\s+contrapositi", "reverse contraposition"),
        (r"\braa\b|reductio\s+ad\s+absurdum", "reductio ad absurdum (classical)"),
        (r"\bde\s+morgan", "De Morgan's law (classical)"),
        (r"proof\s+by\s+contradiction(?!\s+\(constructive\))", "classical proof by contradiction"),
        (r"assume\s+\u00ac\w+.*derive.*contradiction.*therefore\s+\w+(?!\s*\u21d2\s*\u22a5)", "classical reductio"),
    ]

    # Aliases for rule_selection mode
    RULE_ALIASES: dict[str, set[str]] = {
        "mp": {"mp", "modus-ponens", "modus ponens", "\u21d2e", "\u21d2-elim", "implication-elimination"},
        "\u2227i": {"\u2227i", "and-intro", "conjunction-introduction", "conj-intro", "\u2227-intro"},
        "\u2227e": {"\u2227e", "and-elim", "conjunction-elimination", "conj-elim", "\u2227-elim"},
        "\u2228i": {"\u2228i", "or-intro", "disjunction-introduction", "disj-intro", "\u2228-intro"},
        "\u2228e": {"\u2228e", "or-elim", "disjunction-elimination", "disj-elim", "\u2228-elim"},
        "\u21d2i": {"\u21d2i", "impl-intro", "implication-introduction", "\u21d2-intro", "deduction-theorem"},
        "\u21d2e": {"\u21d2e", "mp", "modus-ponens", "modus ponens", "implication-elimination", "\u21d2-elim"},
        "\u00aci": {"\u00aci", "neg-intro", "negation-introduction", "\u00ac-intro", "refutation"},
        "\u00ace": {"\u00ace", "neg-elim", "negation-elimination", "\u00ac-elim"},
        "\u22a5e": {"\u22a5e", "explosion", "ex-falso", "efq", "\u22a5-elim"},
    }

    # Long-form rule names (models often write these in explanations or brackets)
    VALID_RULES_LONG: set[str] = {
        "negation introduction", "negation elimination",
        "implication introduction", "implication elimination",
        "conjunction introduction", "conjunction elimination",
        "disjunction introduction", "disjunction elimination",
        "assumption reuse", "ex falso quodlibet",
        "modus ponens", "modus tollens",
        "hypothetical syllogism",
        "biconditional introduction", "biconditional elimination",
        "double negation introduction",
        "and introduction", "and elimination",
        "or introduction", "or elimination",
        "arrow introduction", "arrow elimination",
        "conditional introduction", "conditional elimination",
        "negation intro", "negation elim",
        "implication intro", "implication elim",
        "conjunction intro", "conjunction elim",
        "disjunction intro", "disjunction elim",
        "ex falso",
        "conjunction i", "conjunction e",  # spaced rule+direction
        "disjunction i", "disjunction e",
        "implication i", "implication e",
        "negation i", "negation e",
        "disjunctive syllogism",
        "double negation",
        "rule of conjunction introduction", "rule of conjunction elimination",
        "rule of disjunction introduction", "rule of disjunction elimination",
        "rule of implication introduction", "rule of implication elimination",
        "rule of negation introduction", "rule of negation elimination",
        "conditional proof",
        "proof by cases",
        # v5 additions: missing long-form aliases
        "disjunct introduction", "disjunct elimination",
        "conjunct introduction", "conjunct elimination",
        "elimination of disjunction", "elimination of conjunction",
        "elimination of implication", "elimination of negation",
        "eliminate disjunction", "eliminate conjunction",
        "eliminate implication", "eliminate negation",
        "introduce conjunction", "introduce disjunction",
        "introduce implication", "introduce negation",
    }

    # Patterns to ignore when extracting rule citations from brackets/parens
    ANNOTATION_PATTERNS: list[str] = [
        r"^line\s+\d+",
        r"^lines?\s+\d+",
        r"^\d+[-\u2013]\d+$",
        r"^for\s+(left|right)\s+case",
        r"^(left|right)\s+case",
        r"^discharg(e|ing)\s+",
        r"^modus ponens with",
        r"^shorthand for",
        r"^which is",
        r"^a\s+contradiction",
        r"^\u22a5$",
        r"^IPC$",
        r"^Ex Falso$",
        # Single letters / formula fragments (not rule names)
        r"^[a-z]$",  # single lowercase letter
        r"^[a-z]\d*$",  # single letter + digits (p0, p1, etc.)
        r"^\u00ac[a-z]",  # negated variable
        r"^[a-z]\s*[\u21d2\u2192\u2227\u2228\u2194]\s*[a-z]",  # formula like "a => b", "c & a"
        r"^[a-z]\s*[\u21d2\u2192\u2227\u2228\u2194]",  # partial formula
        r"^[\u21d2\u2192\u2227\u2228\u2194]\s*[a-z]",  # partial formula starting with operator
        # Annotations and meta-text
        r"^conclusion",
        r"^given$",
        r"^rule$",
        r"^proof\s+step",
        r"^line\s*refs?",
        r"^qed$",
        r"^then$",
        r"^rewrite",
        r"^substitution",
        r"^removal$",
        r"^introduction$",  # bare "introduction" without rule type
        r"^elimination$",  # bare "elimination" without rule type
        r"^universal\s+instantiation",
        r"^necessitation",
        r"^nevigation",  # misspelling of "navigation"
        r"^equivalence$",
        r"^deduction$",  # bare "deduction" (not "deduction-theorem")
        r"^assuming\b",
        r"^omitted",
        r"^default$",
        r"^output$",
        r"^proof$",
        r"^n/a$",
        r"^from\s+\d+",   # "FROM 1", "from 2, 3"
        r"^tip$",
        r"^combine\b",    # "combine a => b and b => a"
        r"^lemma",
        r"^generalization$",
        r"^ref\s+\d+",    # "ref 7"
        # Additional bare words / meta-text
        r"^intro$",
        r"^elim$",
        r"^right$",
        r"^left$",
        r"^step\b",
        r"^proved$",
        r"^none$",
        r"^rules$",
        r"^tactic\b",
        r"^tac$",
        r"^def\b",
        r"^definition\b",
        r"^cases$",
        r"^concl",
        r"^consequent",
        r"^antecedent",
        r"^result$",
        r"^x/[a-z]",      # substitution notation like "x/e"
        r"^copy\s+of\s+\d+",  # reiteration reference (e.g., "copy of 2")
    ]

    def __init__(self, target_formula: str, premises: list[str] = None, mode: str = "proof"):
        self.target_formula = target_formula
        self.premises = premises or []
        self.mode = mode

    @staticmethod
    def _extract_proof_lines(proof_text: str) -> str:
        """Extract only numbered proof lines from the response.

        Matches lines like:
          1. P [premise]
          2. P & Q [&I, 1, 2]
          - P [assumption]

        Returns just the proof lines joined, stripping explanatory text.
        """
        lines = []
        for line in proof_text.split("\n"):
            stripped = line.strip()
            # Remove markdown bold markers
            stripped = re.sub(r"\*\*", "", stripped)
            # Match numbered proof lines: "1." or "1:" or "- " at start
            if re.match(r"^\d+[\.\):]\s+", stripped) or re.match(r"^-\s+\S", stripped):
                lines.append(stripped)
        return "\n".join(lines) if lines else proof_text

    def verify_text(self, proof_text: str) -> dict[str, Any]:
        """Verify proof text directly (no filesystem dependency)."""
        if self.mode == "rule_selection":
            return self._verify_rule_selection(proof_text)

        # Extract only proof lines for rule/contamination checking
        proof_lines = self._extract_proof_lines(proof_text)

        checks = {
            "rule_citations": self._check_rule_citations(proof_lines),
            "classical_contamination": self._check_classical_contamination(proof_lines),
            "scope_violations": self._check_scope_violations(proof_text),
            "conclusion_reached": self._check_conclusion(proof_text),
        }

        valid_rules = checks["rule_citations"]["valid_count"]
        invalid_rules = checks["rule_citations"]["invalid_count"]
        contaminations = len(checks["classical_contamination"]["found"])
        scope_violations = checks["scope_violations"]["count"]
        has_conclusion = checks["conclusion_reached"]["found"]

        raw_score = valid_rules - (invalid_rules * 1) - (contaminations * 3) - (scope_violations * 2)
        if not has_conclusion:
            raw_score -= 5

        max_possible = max(valid_rules + invalid_rules, 1)
        score = max(0.0, min(1.0, raw_score / max_possible))

        passed = (
            has_conclusion
            and contaminations == 0
            and scope_violations == 0
            and invalid_rules == 0
            and valid_rules > 0
        )

        parts: list[str] = []
        if passed:
            parts.append(f"valid IPC proof ({valid_rules} steps)")
        else:
            if contaminations > 0:
                names = [c["type"] for c in checks["classical_contamination"]["found"]]
                parts.append(f"classical contamination: {', '.join(names)}")
            if scope_violations > 0:
                parts.append(f"{scope_violations} scope violation(s)")
            if invalid_rules > 0:
                parts.append(f"{invalid_rules} unrecognized rule(s)")
            if not has_conclusion:
                parts.append(f"target '{self.target_formula}' not derived")

        detail = "; ".join(parts) if parts else "no proof found"

        return {
            "pass": passed,
            "score": score,
            "detail": detail,
            "target_formula": self.target_formula,
            "valid_rules": valid_rules,
            "invalid_rules": invalid_rules,
            "classical_contaminations": contaminations,
            "scope_violations": scope_violations,
            "conclusion_reached": has_conclusion,
            "checks": checks,
        }

    def verify_file(self, workdir: Path) -> dict[str, Any]:
        """Read proof.txt from workdir and verify."""
        proof_path = workdir / "proof.txt"
        if proof_path.exists():
            proof_text = proof_path.read_text(encoding="utf-8")
        else:
            proof_text = ""
        return self.verify_text(proof_text)

    def _verify_rule_selection(self, proof_text: str) -> dict[str, Any]:
        answer = proof_text.strip().lower()
        target = self.target_formula.strip().lower()
        target_group = self.RULE_ALIASES.get(target, {target})
        passed = answer in target_group or target in answer

        return {
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "detail": f"correct rule: {answer}" if passed else f"expected '{self.target_formula}', got '{proof_text.strip()}'",
            "target_formula": self.target_formula,
            "valid_rules": 1 if passed else 0,
            "invalid_rules": 0 if passed else 1,
            "classical_contaminations": 0,
            "scope_violations": 0,
            "conclusion_reached": passed,
            "checks": {"mode": "rule_selection", "answer": proof_text.strip(), "expected": self.target_formula},
        }

    def _is_annotation(self, text: str) -> bool:
        """Check if a captured string is an annotation, not a rule name."""
        stripped = text.strip()
        for pattern in self.ANNOTATION_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
        return False

    def _check_rule_citations(self, proof_text: str) -> dict[str, Any]:
        # Primary: extract rules from bracket justifications [rule, lines]
        cited = re.findall(
            r"\[([a-zA-Z\u2227\u2228\u21d2\u00ac\u22a5]"
            r"[a-zA-Z0-9\-\u2227\u2228\u21d2\u00ac\u22a5/ ]*?)(?:,\s*\d+(?:[\u2013\-]\d+)?)*\]",
            proof_text,
        )
        # Secondary: "by X" or "using X" patterns
        cited += re.findall(
            r"(?:by|rule:|using|via|apply(?:ing)?)\s+"
            r"([a-zA-Z\u2227\u2228\u21d2\u00ac\u22a5][a-zA-Z0-9\-\u2227\u2228\u21d2\u00ac\u22a5 ]*?)"
            r"(?:[.,;)\]\n]|$)",
            proof_text, re.IGNORECASE,
        )

        valid_count = 0
        invalid_count = 0
        invalid_rules: list[str] = []

        for rule_name in cited:
            normalized = rule_name.strip().lower()
            # Skip annotations that aren't rule names
            if self._is_annotation(normalized):
                continue
            # Strip directional modifiers
            normalized = re.sub(r"\s+(?:left|right)\s*$", "", normalized)
            # Strip trailing "reuse" (e.g., "assumption reuse")
            normalized = re.sub(r"\s+reuse\s*$", "", normalized)
            # Strip trailing line refs (e.g., "=>I, 1, discharge 1" -> just the rule part)
            normalized = re.sub(r",?\s*discharge\s+.*$", "", normalized)
            # Normalize -> to => (models use both interchangeably)
            normalized = normalized.replace("\u2192", "\u21d2")
            # Strip "rule" suffix (e.g., "assumption rule" -> "assumption")
            normalized = re.sub(r"\s+rule\s*$", "", normalized)
            # Handle spaced variants: "& I" -> "&I", "=> E" -> "=>E"
            normalized = re.sub(r"([\u2227\u2228\u21d2\u00ac\u22a5])\s+([IiEe])\b", r"\1\2", normalized)
            # Handle "neg i" -> "negi", "conj intro" -> "conj-intro"
            normalized = re.sub(r"^(neg|conj|disj|impl?)\s+(intro|elim|introduction|elimination)$",
                                r"\1-\2", normalized)
            # Handle "ipc-modus ponens" -> "modus ponens"
            normalized = re.sub(r"^ipc[- ]", "", normalized)
            # Handle "step via X" / "rule of X" -> extract X
            normalized = re.sub(r"^(?:step\s+via|rule\s+of)\s+", "", normalized)
            if (normalized in self.VALID_RULES
                    or normalized in self.VALID_RULES_LONG
                    or any(normalized.startswith(v.lower()) for v in self.VALID_RULES)):
                valid_count += 1
            else:
                invalid_count += 1
                invalid_rules.append(rule_name.strip())

        return {"valid_count": valid_count, "invalid_count": invalid_count, "invalid_rules": invalid_rules}

    def _check_classical_contamination(self, proof_text: str) -> dict[str, Any]:
        found: list[dict[str, Any]] = []
        for pattern, name in self.CLASSICAL_PATTERNS:
            matches = re.findall(pattern, proof_text, re.IGNORECASE)
            if matches:
                found.append({"type": name, "count": len(matches)})
        return {"found": found}

    def _check_scope_violations(self, proof_text: str) -> dict[str, Any]:
        discharged = re.findall(
            r"discharge\s+(\w+)|discharging\s+(\w+)|close\s+subproof.*?(\w+)",
            proof_text, re.IGNORECASE,
        )
        discharged_names: set[str] = set()
        for groups in discharged:
            for g in groups:
                if g:
                    discharged_names.add(g.strip())

        violations: list[str] = []
        for name in discharged_names:
            discharge_pos = -1
            for m in re.finditer(
                rf"discharge\s+{re.escape(name)}|discharging\s+{re.escape(name)}",
                proof_text, re.IGNORECASE,
            ):
                discharge_pos = m.end()
            if discharge_pos > 0:
                remainder = proof_text[discharge_pos:]
                if re.search(
                    rf"\b(?:cite|use|by assumption)\s+{re.escape(name)}\b",
                    remainder, re.IGNORECASE,
                ):
                    violations.append(name)

        return {"count": len(violations), "violated_assumptions": violations}

    @staticmethod
    def _normalize_formula(text: str) -> str:
        """Normalize a formula for comparison: unify arrows, strip whitespace."""
        # Normalize arrow variants: -> (U+2192) to => (U+21D2)
        text = text.replace("\u2192", "\u21d2")
        # Strip backticks (models sometimes wrap formulas in them)
        text = text.replace("`", "")
        # Collapse whitespace
        text = re.sub(r"\s+", "", text)
        return text

    @staticmethod
    def _strip_outer_parens(formula: str) -> str:
        """Strip matched outer parentheses: '( p0 )' -> 'p0'."""
        s = formula.strip()
        while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            inner = s[1:-1]
            # Verify the parens are truly outer (balanced inside)
            depth = 0
            balanced = True
            for ch in inner:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth < 0:
                    balanced = False
                    break
            if balanced and depth == 0:
                s = inner.strip()
            else:
                break
        return s

    def _check_conclusion(self, proof_text: str) -> dict[str, Any]:
        target = self.target_formula.strip()

        # Build a list of target variants to check (most specific first)
        targets = [target]
        stripped = self._strip_outer_parens(target)
        if stripped != target:
            targets.append(stripped)

        # Normalize arrow symbols in proof text for matching
        proof_arrow_normalized = proof_text.replace("\u2192", "\u21d2")

        for t in targets:
            conclusion_patterns = [
                rf"(?:therefore|conclude|QED|proved|thus|hence|\u2234)\s*:?\s*.*{re.escape(t)}",
                rf"{re.escape(t)}\s*(?:QED|\u25a0|\u25a1|\u220e)",
                rf"(?:we have|we get|deriving|derived)\s*:?\s*{re.escape(t)}",
            ]

            for pattern in conclusion_patterns:
                if re.search(pattern, proof_arrow_normalized, re.IGNORECASE):
                    return {"found": True}

            # Exact substring match (with arrow normalization)
            if t in proof_arrow_normalized:
                return {"found": True}

            # Whitespace-collapsed match (with arrow normalization)
            norm_t = self._normalize_formula(t)
            norm_p = self._normalize_formula(proof_arrow_normalized)
            if norm_t in norm_p:
                return {"found": True}

        return {"found": False}
