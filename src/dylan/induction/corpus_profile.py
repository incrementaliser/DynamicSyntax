"""Corpus-specific induction profiles (BabyDS vs CHILDES).

Centralises the coupled hard-coded maps that must stay aligned when extending
induction: TreeFilter node addresses, TypeLattice priority templates,
``get_filtered_abstractions`` DSType lists, and Tree static typeMap overrides.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

CorpusProfileName = Literal["babyds", "childes"]


@dataclass(frozen=True)
class InductionCorpusProfile:
    """Named presets for argument-position and abstraction templates."""

    name: CorpusProfileName
    tree_filter_map: dict[str, str]
    filtered_abstraction_templates_t: tuple[str, ...]
    filtered_abstraction_templates_cn: tuple[str, ...]
    priority_template_specs: tuple[str, ...]
    priority_field_specs: tuple[str, ...]
    static_type_map_overrides: dict[str, str]


BABYDS_PROFILE = InductionCorpusProfile(
    name="babyds",
    tree_filter_map={
        "00": "p==obj(e,x):t",
        "010": "p==obj(e,x):t",
        "0110": "p==ind_obj(e1,e2):t",
    },
    filtered_abstraction_templates_t=("e>(e>(e>t))", "e>(e>t)", "e>t"),
    filtered_abstraction_templates_cn=("cn>cn",),
    priority_template_specs=(
        "[e1:es|e2:es|x1:e|x2:e|p1==subj(e1,x1):t|p2==obj(e1,x2):t|p3==ind_obj(e1, e2):t]",
        "[e1:es|x1:e|x2:e|p1==subj(e1,x1):t|p2==obj(e1,x2):t]",
        "[e1:es|x1:e|p1==subj(e1,x1):t]",
    ),
    priority_field_specs=(
        "p==subj(e,x):t",
        "p==obj(e,x):t",
        "p==ind_obj(e1,e2):t",
    ),
    static_type_map_overrides={},
)

CHILDES_PROFILE = InductionCorpusProfile(
    name="childes",
    tree_filter_map={
        "00": "p==subj(e,x):t",
        "010": "p==obj(e,x):t",
        "0110": "p==ind_obj(e1,e2):t",
    },
    # Pre-BabyDS / 2013 template order (early-return Filtered uses first hit).
    filtered_abstraction_templates_t=(
        "e>(e>(e>t))",
        "es>(e>(e>t))",
        "e>(e>t)",
        "e>t",
    ),
    # Classic cn peels (BabyDS uses cn>cn only).
    filtered_abstraction_templates_cn=(
        "e>(es>cn)",
        "es>cn",
        "e>cn",
    ),
    priority_template_specs=(
        "[e1:es|x1:e|x2:e|p1==subj(e1,x1):t|p2==obj(e1,x2):t|p3==ind_obj(e1, e2):t]",
        "[e1:es|x1:e|x2:e|p1==subj(e1,x1):t|p2==obj(e1,x2):t]",
        "[e1:es|x1:e|p1==subj(e1,x1):t]",
    ),
    priority_field_specs=(
        "p==subj(e,x):t",
        "p==obj(e,x):t",
        "p==ind_obj(e1,e2):t",
    ),
    static_type_map_overrides={
        "e>t": "R1^(R1 ++ [e1:es|p==subj(e1,R1.head):t|head==e1:es])",
        "e>cn": "R1^(R1 ++ [head==R1.head:e|p:t])",
    },
)

_PROFILES: dict[CorpusProfileName, InductionCorpusProfile] = {
    "babyds": BABYDS_PROFILE,
    "childes": CHILDES_PROFILE,
}

_active_profile: ContextVar[InductionCorpusProfile] = ContextVar(
    "induction_corpus_profile",
    default=BABYDS_PROFILE,
)


def get_profile(name: CorpusProfileName | str) -> InductionCorpusProfile:
    """Return the named corpus profile."""
    key = str(name).strip().lower()
    if key not in _PROFILES:
        raise ValueError(f"Unknown induction corpus profile: {name!r}")
    return _PROFILES[key]  # type: ignore[index]


def get_active_profile() -> InductionCorpusProfile:
    """Return the currently active induction corpus profile."""
    return _active_profile.get()


def set_active_profile(name_or_profile: CorpusProfileName | InductionCorpusProfile | str) -> InductionCorpusProfile:
    """Activate *name_or_profile* and invalidate caches that depend on it."""
    profile = (
        name_or_profile
        if isinstance(name_or_profile, InductionCorpusProfile)
        else get_profile(name_or_profile)
    )
    _active_profile.set(profile)
    _invalidate_profile_caches()
    return profile


def _invalidate_profile_caches() -> None:
    """Clear class-level caches rebuilt from the active profile."""
    from dylan.dag.type_lattice import TypeLattice
    from dylan.induction.em_learner.tree_filter import TreeFilter
    from dylan.tree import underspecified_type_map as type_map_mod

    TypeLattice.priority_templates = []
    TreeFilter._node_field_map = {}
    type_map_mod._STATIC_TYPE_MAP_CACHE = None
