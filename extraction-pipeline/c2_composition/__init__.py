"""c2_composition -- skill composition operators and generation."""

__all__: list[str] = []

try:
    from c2_composition.operators import ComposedSkill, compose_seq, compose_par, compose_cond
    __all__ += ["ComposedSkill", "compose_seq", "compose_par", "compose_cond"]
except ImportError:
    pass

try:
    from c2_composition.generator import generate_all_compositions, generate_and_save
    __all__ += ["generate_all_compositions", "generate_and_save"]
except ImportError:
    pass
