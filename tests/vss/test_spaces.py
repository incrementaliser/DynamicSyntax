"""Unit tests for the DS-VSS linear-algebraic substrate."""

import numpy as np
import pytest

from dylan.vss.spaces import (
    VSSValue,
    VectorSpace,
    contract,
    direct_sum,
    mu,
    plausibility,
    plausibility_space,
    unit_value,
)

W = VectorSpace("W", 4, ("infant", "nappy", "pitch", "goal"))
S = plausibility_space()


def test_space_validation():
    with pytest.raises(ValueError):
        VectorSpace("X", 0)
    with pytest.raises(ValueError):
        VectorSpace("X", 2, ("only-one",))
    assert VectorSpace("X", 2).basis == ("X0", "X1")


def test_plausibility_space_basis():
    assert S.dim == 2
    assert S.basis == ("⊤", "⊥")


def test_value_shape_checked():
    with pytest.raises(ValueError):
        VSSValue((W,), np.zeros(3))
    with pytest.raises(ValueError):
        VSSValue((W, S), np.zeros((4,)))


def test_unit_value():
    u = unit_value((W, S))
    assert u.array.shape == (4, 2)
    assert np.all(u.array == 1.0)
    assert unit_value(()).array.shape == ()


def test_contract_matrix_vector():
    mat = VSSValue((W, S), np.arange(8, dtype=float).reshape(4, 2))
    vec = VSSValue((W,), np.array([1.0, 0.0, 2.0, 0.0]))
    out = contract(mat, vec)
    assert out.space_names() == ("S",)
    assert np.allclose(out.array, vec.array @ mat.array)


def test_contract_cube_rightmost_axis_first():
    """T_ijk contracts with the object (rightmost W) before the subject."""
    cube = VSSValue((W, S, W), np.arange(4 * 2 * 4, dtype=float).reshape(4, 2, 4))
    obj = VSSValue((W,), np.array([0.0, 0.0, 0.0, 1.0]))
    mat = contract(cube, obj)
    assert mat.space_names() == ("W", "S")
    assert np.allclose(mat.array, cube.array[:, :, 3])
    subj = VSSValue((W,), np.array([1.0, 0.0, 0.0, 0.0]))
    sent = contract(mat, subj)
    assert sent.space_names() == ("S",)
    assert np.allclose(sent.array, cube.array[0, :, 3])
    # Direct einsum agrees: T_i^subj T_ijk T_k^obj
    assert np.allclose(sent.array, np.einsum("i,ijk,k->j", subj.array, cube.array, obj.array))


def test_contract_rejects_non_vector_argument():
    mat = VSSValue((W, S), np.ones((4, 2)))
    with pytest.raises(ValueError):
        contract(mat, mat)


def test_contract_missing_space():
    vec_w = VSSValue((W,), np.ones(4))
    vec_s = VSSValue((S,), np.ones(2))
    with pytest.raises(ValueError):
        contract(vec_w, vec_s)  # no S axis in the functor
    mat = VSSValue((W, S), np.ones((4, 2)))
    out = contract(mat, vec_s)  # contracts the S axis
    assert out.space_names() == ("W",)


def test_mu_map():
    a = VSSValue((S,), np.array([2.0, 4.0]))
    b = VSSValue((S,), np.array([3.0, 0.5]))
    out = mu(a, b)
    assert np.allclose(out.array, [6.0, 2.0])
    with pytest.raises(ValueError):
        mu(a, VSSValue((W,), np.ones(4)))


def test_plausibility_measure():
    assert plausibility(VSSValue((S,), np.array([430.0, 98.0]))) == pytest.approx(
        430.0 / 528.0
    )
    assert plausibility(VSSValue((S,), np.array([0.0, 0.0]))) == 0.0
    assert plausibility(VSSValue((), np.array(3.5))) == 3.5


def test_direct_sum_distributes():
    m1 = VSSValue((W, S), np.ones((4, 2)))
    m2 = VSSValue((W, S), 2 * np.ones((4, 2)))
    ds = direct_sum([m1, m2])
    vec = VSSValue((W,), np.array([1.0, 1.0, 0.0, 0.0]))
    out = ds.map_contract(vec)
    assert [tuple(v.array) for v in out.values] == [(2.0, 2.0), (4.0, 4.0)]
    assert out.plausibilities() == [0.5, 0.5]
