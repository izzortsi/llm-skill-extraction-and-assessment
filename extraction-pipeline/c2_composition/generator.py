"""
generator.py

Generate skill compositions and write to disk.

Usage:
    python -m c2_skill_composition.generator --atomic-dir skills/ --output-dir output/ --k 2 3
    python -m c2_skill_composition.generator --atomic-dir skills/ --output-dir output/ --k 2 3 --semantic --provider anthropic --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from c1_tools.skill_registry import SkillRegistry
from c2_composition.operators import (
    generate_all_compositions,
    generate_semantic_compositions,
    SemanticCompositionConfig,
)


def generate_and_save(
    atomic_dir: Path,
    output_dir: Path,
    k_values: List[int] = None,
    use_semantic: bool = False,
    semantic_config: Optional[SemanticCompositionConfig] = None,
    verbose: bool = False,
) -> dict:
    """Generate all compositions and save to disk."""
    if k_values is None:
        k_values = [2, 3, 4, 5]

    registry = SkillRegistry.from_directory(atomic_dir)
    if verbose:
        print(f"Loaded {len(registry.skills)} atomic skills")

    max_k = max(k_values)
    min_k = min(k_values)

    # Mechanical compositions
    all_comps = generate_all_compositions(registry, max_k=max_k)

    counts = {}
    for comp_type, skills_list in all_comps.items():
        for composed in skills_list:
            if composed.k_value not in k_values:
                continue
            type_dir = output_dir / f"k{composed.k_value}" / comp_type
            type_dir.mkdir(parents=True, exist_ok=True)
            filepath = type_dir / f"{composed.name}.md"
            filepath.write_text(composed.to_markdown(), encoding="utf-8")
            counts[comp_type] = counts.get(comp_type, 0) + 1

    if verbose:
        for t, c in counts.items():
            print(f"  {t}: {c} compositions")

    # Semantic compositions
    if use_semantic and semantic_config:
        sem_skills = generate_semantic_compositions(
            registry, max_k=max_k, min_k=min_k,
            config=semantic_config, output_dir=output_dir,
        )
        counts["sem"] = len(sem_skills)
        if verbose:
            print(f"  sem: {len(sem_skills)} compositions")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate skill compositions")
    parser.add_argument("--atomic-dir", type=Path, required=True, help="Directory with atomic skill .md files")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("composed-skills"), help="Output directory")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5], help="K values to generate")
    parser.add_argument("--semantic", action="store_true", help="Include semantic compositions")
    parser.add_argument("--provider", type=str, default="anthropic", help="Provider for semantic composition")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="Model for semantic composition")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    semantic_config = None
    if args.semantic:
        semantic_config = SemanticCompositionConfig(provider=args.provider, model=args.model)

    counts = generate_and_save(
        args.atomic_dir, args.output_dir, k_values=args.k,
        use_semantic=args.semantic, semantic_config=semantic_config,
        verbose=args.verbose,
    )
    print(f"Generated compositions: {counts}")


if __name__ == "__main__":
    main()
