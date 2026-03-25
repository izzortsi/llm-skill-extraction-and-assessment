"""
operators.py

Composition operators for skills based on skill-calculus framework.

Implements four composition types:
- seq: Sequential composition (skills applied in order)
- par: Parallel composition (skills applied simultaneously)
- cond: Conditional composition (skills applied conditionally)
- sem: Semantic composition (LLM-based fusion)

Usage:
    python -m c2_skill_composition.operators --skills-dir skills/ --max-k 3
    python -m c2_skill_composition.operators --skills-dir skills/ --max-k 3 --semantic --provider anthropic --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict


from c1_tools.skill_registry import Skill, SkillRegistry


@dataclass
class ComposedSkill:
    """A skill composed from atomic skills using composition operators."""

    # YAML frontmatter
    name: str
    description: str

    # Body sections
    when_to_use: List[str]
    procedure: List[str]
    constraints: List[str]
    examples: List[Dict[str, str]]
    related_skills: List[str]

    # Composition metadata
    composition_type: str  # "seq", "par", "cond", or "sem"
    source_skills: List[str]  # names of atomic skills used
    k_value: int  # number of atomic skills composed

    def to_markdown(self) -> str:
        """Convert composed skill to skill-creator template format."""
        # YAML frontmatter
        frontmatter = f"""---
name: {self.name}
description: {self.description}
---

# {format_title(self.name)}

"""

        # When to Use
        frontmatter += "## When to Use\n\n"
        for trigger in self.when_to_use:
            frontmatter += f"- {trigger}\n"
        frontmatter += "\n"

        # Procedure
        frontmatter += "## Procedure\n\n"
        for i, step in enumerate(self.procedure, 1):
            frontmatter += f"{i}. {step}\n"
        frontmatter += "\n"

        # Constraints
        if self.constraints:
            frontmatter += "## Constraints\n\n"
            for constraint in self.constraints:
                frontmatter += f"- {constraint}\n"
            frontmatter += "\n"

        # Examples
        if self.examples:
            for i, example in enumerate(self.examples, 1):
                title = example.get("title", f"Example {i}")
                frontmatter += f"### Example {i}: {title}\n\n"
                frontmatter += f"**Input:**\n{example['input']}\n\n"
                frontmatter += f"**Process:**\n{example['process']}\n\n"
                frontmatter += f"**Output:**\n{example['output']}\n\n"

        # Related Skills
        if self.related_skills:
            frontmatter += "## Related Skills\n\n"
            for skill in self.related_skills:
                frontmatter += f"- {skill}\n"

        return frontmatter


def format_title(name: str) -> str:
    """Convert hyphen-case name to Title Case."""
    return name.replace("-", " ").title()


def compose_sequential_examples(skills: List[Skill]) -> List[Dict[str, str]]:
    """Generate examples for sequential composition by chaining examples.

    Takes examples from source skills and shows how outputs flow between them.
    """
    if not skills:
        return []

    examples = []

    # Try to get at least one example from each skill
    source_examples = []
    for skill in skills:
        if skill.examples:
            source_examples.append(skill.examples[0])
        else:
            # Create placeholder if skill has no examples
            source_examples.append({
                "title": f"Using {format_title(skill.name)}",
                "input": "[Input requiring this skill]",
                "process": f"Follow the procedure for {format_title(skill.name)}",
                "output": f"[Output from {format_title(skill.name)}]",
            })

    if len(source_examples) == 1:
        return [{
            "title": source_examples[0].get("title", f"Using {format_title(skills[0].name)}"),
            "input": source_examples[0].get("input", "[Input]"),
            "process": source_examples[0].get("process", f"Apply {format_title(skills[0].name)}"),
            "output": source_examples[0].get("output", "[Output]"),
        }]

    # Create chained example showing sequential flow
    # Input from first skill
    combined_input = source_examples[0].get("input", "[Input]")

    # Process shows the sequence of skills
    process_lines = [f"Apply the following skills in sequence:"]
    for i, (skill, src_ex) in enumerate(zip(skills, source_examples)):
        step_num = i + 1
        skill_title = format_title(skill.name)
        if src_ex.get("process") and src_ex["process"] != f"Follow the procedure for {skill_title}":
            # Use actual process content if available
            process_lines.append(f"{step_num}. {skill_title}: {src_ex['process'][:100]}...")
        else:
            process_lines.append(f"{step_num}. Apply {skill_title}")

    combined_process = "\n".join(process_lines)

    # Output from last skill
    combined_output = source_examples[-1].get("output", f"[Output from {format_title(skills[-1].name)}]")

    examples.append({
        "title": f"Sequential application of {' -> '.join([format_title(s.name) for s in skills])}",
        "input": combined_input,
        "process": combined_process,
        "output": combined_output,
    })

    return examples


def compose_parallel_examples(skills: List[Skill]) -> List[Dict[str, str]]:
    """Generate examples for parallel composition."""
    if not skills:
        return []

    # Get first example from each skill
    skill_descriptions = []
    for skill in skills:
        if skill.examples and skill.examples[0].get("input"):
            skill_descriptions.append(f"- {format_title(skill.name)}: {skill.examples[0]['input'][:50]}...")
        else:
            skill_descriptions.append(f"- {format_title(skill.name)}")

    return [{
        "title": "Parallel skill application",
        "input": "Input that requires analysis from multiple skill perspectives simultaneously",
        "process": f"Apply the following skills in parallel to the same input:\n" + "\n".join(skill_descriptions) + "\n\nThen integrate the results from all skills.",
        "output": "Combined analysis integrating outputs from all parallel skills",
    }]


def compose_conditional_examples(
    condition_skill: Skill,
    then_skills: List[Skill],
    else_skills: Optional[List[Skill]] = None,
) -> List[Dict[str, str]]:
    """Generate examples for conditional composition."""
    examples = []

    condition_title = format_title(condition_skill.name)

    # True branch example
    then_desc = " -> ".join([format_title(s.name) for s in then_skills])
    examples.append({
        "title": "Condition is true",
        "input": f"Input where {condition_title} evaluates to TRUE",
        "process": f"1. {condition_title} evaluates to TRUE\n2. Apply then-skills: {then_desc}",
        "output": f"Output from applying {then_desc}",
    })

    # False branch example (if else_skills provided)
    if else_skills:
        else_desc = " -> ".join([format_title(s.name) for s in else_skills])
        examples.append({
            "title": "Condition is false",
            "input": f"Input where {condition_title} evaluates to FALSE",
            "process": f"1. {condition_title} evaluates to FALSE\n2. Apply else-skills: {else_desc}",
            "output": f"Output from applying {else_desc}",
        })

    return examples


def compose_seq(skills: List[Skill], registry: SkillRegistry) -> ComposedSkill:
    """Sequential composition: apply skills in order.

    alpha_seq(S1, S2, ..., Sn) = S1 -> S2 -> ... -> Sn

    The output of S1 feeds into S2, etc.
    """
    if not skills:
        raise ValueError("Cannot compose empty skill list")

    if len(skills) == 1:
        return single_skill_wrapper(skills[0], "seq")

    k = len(skills)
    skill_names = [s.name for s in skills]

    # Generate name
    name = f"seq-{'-then-'.join(skill_names)}"

    # Combine descriptions
    descriptions = [s.description for s in skills]
    description = f"Apply skills in sequence: {' -> '.join(descriptions)}. Use when multiple capabilities are needed in order."

    # Combine When to Use triggers
    when_to_use = []
    for skill in skills:
        when_to_use.extend(skill.when_to_use)

    # Remove duplicates while preserving order
    seen = set()
    unique_when = []
    for trigger in when_to_use:
        if trigger not in seen:
            seen.add(trigger)
            unique_when.append(trigger)
    when_to_use = unique_when

    # Compose procedures sequentially
    procedure = []
    step_counter = 1

    for i, skill in enumerate(skills):
        skill_intro = f"Apply {format_title(skill.name)}"

        # Add procedure steps from this skill
        for step in skill.procedure:
            # Reference the original step in context
            if i > 0:
                procedure.append(f"{skill_intro}: {step}")
            else:
                procedure.append(step)
            step_counter += 1

    # Combine constraints
    constraints = []
    for skill in skills:
        constraints.extend(skill.constraints)

    # Generate composed examples
    examples = compose_sequential_examples(skills)

    # Combine related skills (union)
    related_skills = []
    seen = set(skill_names)
    for skill in skills:
        for ref in skill.related_skills:
            if ref not in seen:
                seen.add(ref)
                related_skills.append(ref)

    return ComposedSkill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=examples,
        related_skills=related_skills,
        composition_type="seq",
        source_skills=skill_names,
        k_value=k,
    )


def compose_par(skills: List[Skill], registry: SkillRegistry) -> ComposedSkill:
    """Parallel composition: apply skills simultaneously.

    alpha_par(S1, S2, ..., Sn) = S1 & S2 & ... & Sn

    All skills are applied to the same input; outputs are combined.
    """
    if not skills:
        raise ValueError("Cannot compose empty skill list")

    if len(skills) == 1:
        return single_skill_wrapper(skills[0], "par")

    k = len(skills)
    skill_names = [s.name for s in skills]

    # Generate name
    name = f"par-{'-and-'.join(skill_names)}"

    # Combine descriptions
    descriptions = [s.description for s in skills]
    description = f"Apply skills in parallel: {' & '.join(descriptions)}. Use when multiple capabilities are needed simultaneously."

    # Combine When to Use triggers (intersection-ish)
    # Only keep triggers common to most skills or explicitly about parallelism
    when_to_use = [
        f"Multiple skills are needed simultaneously: {', '.join([format_title(s.name) for s in skills])}",
    ]
    # Add unique triggers from each skill
    for skill in skills:
        when_to_use.extend(skill.when_to_use)

    # Compose procedures in parallel (as simultaneous steps)
    procedure = []
    procedure.append("Apply the following skills in parallel:")

    for skill in skills:
        procedure.append(f"  - {format_title(skill.name)}: {skill.description}")

    # Then add sequential integration
    procedure.append("Integrate the outputs from all parallel skills:")
    procedure.append("  - Identify complementary results across skills")
    procedure.append("  - Resolve any conflicts between skill outputs")
    procedure.append("  - Synthesize a combined result")

    # Combine constraints
    constraints = []
    constraints.append("All parallel skills must receive the same input context")
    constraints.extend([s.constraints[0] if s.constraints else "" for s in skills])

    # Generate composed examples
    examples = compose_parallel_examples(skills)

    # Combine related skills (union)
    related_skills = []
    seen = set(skill_names)
    for skill in skills:
        for ref in skill.related_skills:
            if ref not in seen:
                seen.add(ref)
                related_skills.append(ref)

    return ComposedSkill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=examples,
        related_skills=related_skills,
        composition_type="par",
        source_skills=skill_names,
        k_value=k,
    )


def compose_cond(
    condition_skill: Skill,
    then_skills: List[Skill],
    else_skills: Optional[List[Skill]] = None,
    registry: SkillRegistry = None,
) -> ComposedSkill:
    """Conditional composition: apply skills based on condition.

    alpha_cond(S_cond, S_then1, ..., S_then_m, S_else1, ..., S_else_n)

    Apply then_skills if condition is met, else apply else_skills.
    """
    if not then_skills:
        raise ValueError("Must provide at least one 'then' skill")

    all_skills = [condition_skill] + then_skills + (else_skills or [])
    k = len(all_skills)

    # Generate name
    then_names = [s.name for s in then_skills]
    else_names = [s.name for s in else_skills] if else_skills else []

    if else_names:
        name = f"cond-{condition_skill.name}-then-{'-and-'.join(then_names)}-else-{'-and-'.join(else_names)}"
    else:
        name = f"cond-{condition_skill.name}-then-{'-and-'.join(then_names)}"

    # Build description
    description = (
        f"Conditional skill: if {format_title(condition_skill.name)}, "
        f"then apply {' -> '.join([format_title(s.name) for s in then_skills])}"
    )
    if else_skills:
        description += f", else apply {' -> '.join([format_title(s.name) for s in else_skills])}"
    description += ". Use when skill application depends on a condition."

    # When to Use
    when_to_use = [f"When {format_title(condition_skill.name)} determines the path"]
    when_to_use.extend(condition_skill.when_to_use)

    # Procedure
    procedure = []
    procedure.append("Evaluate the condition:")
    procedure.append(f"1. Check {format_title(condition_skill.name)}")
    procedure.append("2. If condition is TRUE:")
    for i, skill in enumerate(then_skills, 3):
        procedure.append(f"   {i-2}. Apply {format_title(skill.name)}")
    procedure.append("3. If condition is FALSE:")
    if else_skills:
        for i, skill in enumerate(else_skills, 3):
            procedure.append(f"   {i-2}. Apply {format_title(skill.name)}")
    else:
        procedure.append("   No action (or apply default behavior)")

    # Constraints
    constraints = [
        f"Condition must be evaluable: {format_title(condition_skill.name)}",
        "Only one branch (then or else) is executed",
    ]
    constraints.extend(condition_skill.constraints)

    # Generate composed examples
    examples = compose_conditional_examples(condition_skill, then_skills, else_skills)

    # Related skills
    all_names = [s.name for s in all_skills]
    related_skills = []
    seen = set(all_names)
    for skill in all_skills:
        for ref in skill.related_skills:
            if ref not in seen:
                seen.add(ref)
                related_skills.append(ref)

    return ComposedSkill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=examples,
        related_skills=related_skills,
        composition_type="cond",
        source_skills=all_names,
        k_value=k,
    )


def single_skill_wrapper(skill: Skill, composition_type: str) -> ComposedSkill:
    """Wrap a single skill as a composition (k=1 edge case)."""
    name = f"{composition_type}-{skill.name}"

    return ComposedSkill(
        name=name,
        description=skill.description,
        when_to_use=skill.when_to_use,
        procedure=skill.procedure,
        constraints=skill.constraints,
        examples=skill.examples,
        related_skills=skill.related_skills,
        composition_type=composition_type,
        source_skills=[skill.name],
        k_value=1,
    )


def generate_all_compositions(
    registry: SkillRegistry,
    max_k: int = 5,
) -> Dict[str, List[ComposedSkill]]:
    """Generate all mechanical compositions up to max_k.

    Returns:
        Dict with keys 'seq', 'par', 'cond' mapping to lists of ComposedSkill
    """
    skill_names = list(registry.skills.keys())

    compositions = {"seq": [], "par": [], "cond": []}

    # Generate sequential and parallel compositions for k=2 to max_k
    for k in range(2, max_k + 1):
        for i in range(len(skill_names)):
            for j in range(i + 1, min(i + k, len(skill_names))):
                selected = skill_names[i : j + 1]

                if len(selected) == k:
                    # Sequential
                    skills = [registry.skills[name] for name in selected]
                    compositions["seq"].append(compose_seq(skills, registry))

                    # Parallel (same skills)
                    compositions["par"].append(compose_par(skills, registry))

    # For conditional, use first skill as condition, rest as then-branch
    for k in range(2, max_k + 1):
        for i in range(len(skill_names) - 1):
            condition = registry.skills[skill_names[i]]
            then_names = skill_names[i + 1 : i + k]

            if len(then_names) < k - 1:
                continue

            then_skills = [registry.skills[name] for name in then_names[: k - 1]]
            compositions["cond"].append(compose_cond(condition, then_skills, None, registry))

    return compositions


# ---------------------------------------------------------------------------
# Semantic composition (alpha_sem) -- LLM-based fusion
# ---------------------------------------------------------------------------


@dataclass
class SemanticCompositionConfig:
    """Configuration for LLM-based semantic composition."""

    provider: str = "anthropic"
    model: str = "claude-opus-4-6"


class SemanticCompositor:
    """LLM-based semantic skill composition operator (alpha_sem).

    Uses the provider abstraction from c1_providers.providers for LLM calls.
    """

    def __init__(self, config: Optional[SemanticCompositionConfig] = None):
        """Initialize semantic compositor.

        Args:
            config: Configuration for provider and model
        """
        self.config = config or SemanticCompositionConfig()
        self._provider = None

    def _get_provider(self):
        """Lazily create the LLM provider."""
        if self._provider is None:
            # TODO: import from llm-skills.llm-providers
            from c1_providers.providers import create_provider  # noqa: requires llm-skills.llm-providers on sys.path
            self._provider = create_provider(self.config.provider, self.config.model)
        return self._provider

    def compose_semantic(
        self,
        skills: List[Skill],
        registry: SkillRegistry,
        fusion_type: str = "auto"
    ) -> ComposedSkill:
        """Semantically compose skills using LLM.

        Args:
            skills: List of atomic skills to compose
            registry: Skill registry for context
            fusion_type: Type of fusion ("auto", "sequential", "parallel", "conditional")

        Returns:
            ComposedSkill with semantically fused content
        """
        if not skills:
            raise ValueError("Cannot compose empty skill list")

        if len(skills) == 1:
            return self._single_skill_wrapper(skills[0], "sem")

        k = len(skills)
        skill_names = [s.name for s in skills]

        # Build prompt for LLM
        prompt = self._build_fusion_prompt(skills, fusion_type)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse LLM response
        composed = self._parse_llm_response(response, skills, fusion_type)

        return composed

    def _build_fusion_prompt(
        self,
        skills: List[Skill],
        fusion_type: str
    ) -> str:
        """Build prompt for LLM-based skill fusion.

        The prompt asks the LLM to:
        1. Understand each input skill's purpose and mechanism
        2. Identify semantic relationships between skills
        3. Create a coherent composite skill
        """
        skill_descriptions = []
        for i, skill in enumerate(skills, 1):
            skill_desc = f"""
## Skill {i}: {format_title(skill.name)}

**Description:** {skill.description}

**When to Use:**
{chr(10).join(f"- {t}" for t in skill.when_to_use[:5])}

**Procedure (summary):**
{chr(10).join(f"{i}. {s}" for i, s in enumerate(skill.procedure[:5], 1))}

**Key Constraints:**
{chr(10).join(f"- {c}" for c in skill.constraints[:3])}
"""
            skill_descriptions.append(skill_desc)

        fusion_guidance = {
            "auto": "automatically determine the best way to combine these skills",
            "sequential": "combine these skills so the output of each feeds into the next",
            "parallel": "combine these skills to be applied simultaneously to the same input",
            "conditional": "combine these skills where one acts as a condition for others",
        }

        prompt = f"""You are a skill composition expert. Your task is to create a NEW composite skill by semantically fusing the following input skills.

# Input Skills
{chr(10).join(skill_descriptions)}

# Your Task
Create a composite skill that {fusion_guidance.get(fusion_type, fusion_guidance["auto"])}.

The composite skill should:
1. Have a clear, descriptive name in hyphen-case
2. Include a comprehensive description explaining when and why to use this composite
3. Specify "When to Use" triggers that indicate when this composite is appropriate
4. Provide a step-by-step Procedure that shows how to apply the composite skill
5. List any Constraints or limitations
6. Include 2-3 concrete Examples with Input/Process/Output
7. Reference Related Skills

# Output Format
Return your response as a JSON object with this exact structure:

```json
{{
  "name": "composite-skill-name",
  "description": "Clear description of what this composite skill does and when to use it",
  "when_to_use": [
    "trigger condition 1",
    "trigger condition 2",
    "trigger condition 3"
  ],
  "procedure": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "constraints": [
    "constraint 1",
    "constraint 2"
  ],
  "examples": [
    {{
      "title": "Example title",
      "input": "Example input",
      "process": "Step-by-step process description",
      "output": "Example output"
    }}
  ],
  "fusion_rationale": "Brief explanation of how and why these skills were combined"
}}
```

# Important Guidelines
- The composite should be GREATER than the sum of its parts - show how skills synergize
- Avoid mechanical concatenation - create semantic understanding
- The Procedure should integrate the skills, not just list them separately
- Examples should demonstrate the composite skill in action
- Keep descriptions concise but informative

Now create the composite skill:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM provider with prompt.

        Uses the provider abstraction from c1_providers.providers.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        try:
            provider = self._get_provider()
            result = provider.chat([{"role": "user", "content": prompt}])

            # Extract text from result.message
            content = result.message.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "".join(parts)
            return str(content)
        except Exception:
            return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when LLM is unavailable.

        Extracts skill names from prompt and returns basic fusion.
        """
        # Extract skill names from prompt
        names = re.findall(r"## Skill \d+: (.+)", prompt)
        if not names:
            names = ["skill-a", "skill-b"]

        hyphen_names = [n.lower().replace(" ", "-") for n in names]

        fallback_json = {
            "name": f"sem-{'-'.join(hyphen_names)}",
            "description": f"Semantic fusion of: {', '.join(names)}. This composite skill integrates the capabilities of its component skills through semantic understanding.",
            "when_to_use": [
                f"When multiple capabilities are needed: {', '.join(names)}",
                "When a integrated approach is more effective than applying skills separately",
                "When the input requires understanding from multiple perspectives",
            ],
            "procedure": [
                f"1. Analyze the input using {names[0] if names else 'first skill'}",
                "2. Integrate insights with complementary skills",
                f"3. Synthesize results using {names[-1] if len(names) > 1 else 'the skill'}",
            ],
            "constraints": [
                "All component skills must be applicable to the input",
                "Skill outputs must be semantically compatible",
            ],
            "examples": [
                {
                    "title": f"Integrated application of {' and '.join(names[:2])}",
                    "input": "Input that requires multiple skill perspectives",
                    "process": f"Apply {names[0] if names else 'skill'} first, then integrate with {names[1] if len(names) > 1 else 'other skills'}",
                    "output": "Integrated result combining insights from all skills",
                }
            ],
            "fusion_rationale": f"Skills {', '.join(names)} are fused because they address complementary aspects of the same domain.",
        }

        return json.dumps(fallback_json, indent=2)

    def _parse_llm_response(
        self,
        response: str,
        skills: List[Skill],
        fusion_type: str
    ) -> ComposedSkill:
        """Parse LLM response into ComposedSkill.

        Args:
            response: LLM response text
            skills: Original input skills
            fusion_type: Type of fusion used

        Returns:
            ComposedSkill object
        """
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_str = json_match.group(0) if json_match else response

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: create minimal composed skill
            skill_names = [s.name for s in skills]
            return ComposedSkill(
                name=f"sem-{'-'.join(skill_names)}",
                description=f"Semantic fusion of {len(skills)} skills",
                when_to_use=["When integrated skill application is needed"],
                procedure=["Apply skills with semantic understanding"],
                constraints=[],
                examples=[],
                related_skills=[],
                composition_type="sem",
                source_skills=skill_names,
                k_value=len(skills),
            )

        # Extract data
        name = data.get("name", f"sem-{'-'.join(s.name for s in skills)}")
        description = data.get("description", "")
        when_to_use = data.get("when_to_use", [])
        procedure = data.get("procedure", [])
        constraints = data.get("constraints", [])
        examples_data = data.get("examples", [])

        # Convert examples to expected format
        examples = []
        for ex in examples_data:
            examples.append({
                "title": ex.get("title", "Example"),
                "input": ex.get("input", ""),
                "process": ex.get("process", ""),
                "output": ex.get("output", ""),
            })

        # Related skills (union of source skills)
        skill_names = [s.name for s in skills]
        related_skills = []
        seen = set(skill_names)
        for skill in skills:
            for ref in skill.related_skills:
                if ref not in seen:
                    seen.add(ref)
                    related_skills.append(ref)

        return ComposedSkill(
            name=name,
            description=description,
            when_to_use=when_to_use,
            procedure=procedure,
            constraints=constraints,
            examples=examples,
            related_skills=related_skills,
            composition_type="sem",
            source_skills=skill_names,
            k_value=len(skills),
        )

    def _single_skill_wrapper(self, skill: Skill, composition_type: str) -> ComposedSkill:
        """Wrap a single skill as a composition (k=1 edge case)."""
        name = f"{composition_type}-{skill.name}"

        return ComposedSkill(
            name=name,
            description=skill.description,
            when_to_use=skill.when_to_use,
            procedure=skill.procedure,
            constraints=skill.constraints,
            examples=skill.examples,
            related_skills=skill.related_skills,
            composition_type=composition_type,
            source_skills=[skill.name],
            k_value=1,
        )


def generate_semantic_compositions(
    registry: SkillRegistry,
    max_k: int = 5,
    min_k: int = 2,
    config: Optional[SemanticCompositionConfig] = None,
    fusion_type: str = "auto",
    output_dir: Optional[Path] = None,
) -> List[ComposedSkill]:
    """Generate semantic compositions using LLM.

    Args:
        registry: Skill registry with atomic skills
        max_k: Maximum number of skills to compose
        min_k: Minimum number of skills to compose (avoids redundant regeneration)
        config: LLM configuration
        fusion_type: Type of semantic fusion
        output_dir: If provided, write each skill to disk immediately after generation

    Returns:
        List of semantically composed skills
    """
    compositor = SemanticCompositor(config)
    skill_names = list(registry.skills.keys())

    compositions = []

    # Generate compositions for min_k to max_k
    for k in range(min_k, max_k + 1):
        count = 0
        for i in range(len(skill_names)):
            for j in range(i + 1, min(i + k, len(skill_names))):
                selected = skill_names[i : j + 1]

                if len(selected) == k:
                    skills = [registry.skills[name] for name in selected]
                    print(f"    sem k={k} [{count+1}]: {', '.join(s.name for s in skills)}")
                    composed = compositor.compose_semantic(skills, registry, fusion_type)
                    compositions.append(composed)
                    count += 1

                    # Write immediately if output_dir provided
                    if output_dir is not None:
                        type_dir = output_dir / f"k{k}" / "sem"
                        type_dir.mkdir(parents=True, exist_ok=True)
                        filepath = type_dir / f"{composed.name}.md"
                        filepath.write_text(composed.to_markdown(), encoding="utf-8")

        print(f"    sem k={k}: {count} compositions generated")

    return compositions


def main() -> None:
    """CLI entry point for composition operators."""
    parser = argparse.ArgumentParser(
        description="Generate skill compositions using composition operators"
    )
    parser.add_argument(
        "--skills-dir", type=Path, required=True,
        help="Directory with atomic skill .md files",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("composed-skills"),
        help="Output directory for composed skills",
    )
    parser.add_argument(
        "--max-k", type=int, default=5,
        help="Maximum composition depth (k value)",
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="Include semantic (LLM-based) compositions",
    )
    parser.add_argument(
        "--provider", type=str, default="anthropic",
        help="Provider for semantic composition (anthropic, openai, mock)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-6",
        help="Model for semantic composition",
    )
    args = parser.parse_args()

    # Load skills
    registry = SkillRegistry.from_directory(args.skills_dir)
    print(f"Loaded {len(registry.skills)} atomic skills")

    # Generate mechanical compositions
    print("Generating mechanical compositions...")
    all_comps = generate_all_compositions(registry, max_k=args.max_k)

    counts = {}
    for comp_type, skills_list in all_comps.items():
        for composed in skills_list:
            type_dir = args.output_dir / f"k{composed.k_value}" / comp_type
            type_dir.mkdir(parents=True, exist_ok=True)
            filepath = type_dir / f"{composed.name}.md"
            filepath.write_text(composed.to_markdown(), encoding="utf-8")
            counts[comp_type] = counts.get(comp_type, 0) + 1

    for t, c in counts.items():
        print(f"  {t}: {c} compositions")

    # Semantic compositions
    if args.semantic:
        print("Generating semantic compositions...")
        config = SemanticCompositionConfig(provider=args.provider, model=args.model)
        sem_skills = generate_semantic_compositions(
            registry, max_k=args.max_k, config=config, output_dir=args.output_dir,
        )
        counts["sem"] = len(sem_skills)
        print(f"  sem: {len(sem_skills)} compositions")

    print(f"Total: {counts}")


if __name__ == "__main__":
    main()
