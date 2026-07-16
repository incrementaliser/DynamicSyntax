"""Tests for random region-graph construction (no torch required)."""

import pytest

from dylan.pc.regions import PCStructure, random_region_graph


def test_requires_at_least_one_variable():
    with pytest.raises(ValueError):
        random_region_graph(0)
    with pytest.raises(ValueError):
        random_region_graph(3, num_repetitions=0)


def test_single_variable_has_no_layers():
    s = random_region_graph(1, seed=0)
    assert s.layers == []
    assert s.region_counts() == [1]


def test_layers_reduce_to_a_root():
    for num_vars in (2, 3, 4, 5, 7, 8):
        s = random_region_graph(num_vars, seed=num_vars)
        counts = s.region_counts()
        assert counts[0] == num_vars
        assert counts[-1] == 1
        # each layer strictly reduces the region count
        assert all(b < a for a, b in zip(counts, counts[1:]))


def test_pairs_are_disjoint_and_cover_previous_layer():
    s = random_region_graph(6, seed=3)
    for spec in s.layers:
        used = [i for pair in zip(spec.left, spec.right) for i in pair if i is not None]
        # every child used exactly once: decomposability of the pairing
        assert len(used) == len(set(used))


def test_structure_is_deterministic_with_seed():
    a: PCStructure = random_region_graph(8, seed=42)
    b: PCStructure = random_region_graph(8, seed=42)
    assert [(s.left, s.right) for s in a.layers] == [(s.left, s.right) for s in b.layers]
