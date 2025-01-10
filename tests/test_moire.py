import numpy as np
import pytest

from twistHS.lib.moire import (
    deformation, moire_components, moire_lattice, nearest_lattice_vector,
    principal_strains, reciprocal_2d, reduce_basis, supercell_matrices,
)


def hexagonal(a):
    return a * np.array([[1, 0], [0.5, np.sqrt(3) / 2]])


def rotation(theta):
    t = np.radians(theta)
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


@pytest.mark.parametrize('theta', [1, 5, 12])
@pytest.mark.parametrize('bottom,scales', [
    (hexagonal(2.46), [2.504 / 2.46] * 2),
    (np.diag([3., 4.]), [1.02, 1.04]),
])
def test_moire_matches_hermann(theta, bottom, scales):
    bottom, _ = reduce_basis(bottom)
    top = np.diag(scales) @ bottom @ rotation(theta)
    q = np.linalg.norm(bottom[0]) / np.linalg.norm(bottom[1])
    p1, p2 = scales
    omega = np.arccos(np.dot(*bottom) / np.prod(np.linalg.norm(bottom, axis=1)))
    alpha = np.radians(theta)
    denominator = (1 + p1 * p2 - (p1 + p2) * np.cos(alpha)
                   + (p1 - p2) * np.cos(omega) * np.sin(alpha))
    hermann = np.array([
        [p1 * (np.sin(omega - alpha) - p2 * np.sin(omega)), p1 * q * np.sin(alpha)],
        [-p2 / q * np.sin(alpha), p2 * (np.sin(omega + alpha) - p1 * np.sin(omega))],
    ]) / (denominator * np.sin(omega))
    # Hermann's convention uses the opposite sign for both moiré vectors.
    np.testing.assert_allclose(moire_lattice(bottom, top), -hermann @ bottom, atol=1e-10)


@pytest.mark.parametrize('A', [hexagonal(2.46), [[3, 0], [0, 4]], [[2, 1], [1, 4]]])
@pytest.mark.parametrize('U', [[[1, 8], [0, 1]], [[2, 3], [1, 2]], [[0, 1], [-1, 0]]])
def test_reduce_basis(A, U):
    A, U = np.asarray(A), np.asarray(U)
    reduced, transform = reduce_basis(U @ A)
    np.testing.assert_allclose(reduced, transform @ U @ A, atol=1e-12)
    assert round(np.linalg.det(transform)) == 1
    assert np.issubdtype(transform.dtype, np.integer)
    assert np.linalg.norm(reduced[0]) <= np.linalg.norm(reduced[1]) + 1e-10
    assert abs(np.dot(*reduced)) <= np.dot(reduced[0], reduced[0]) / 2 + 1e-10
    np.testing.assert_allclose(reduced, reduce_basis(A)[0], atol=1e-12)


def test_reciprocal_duality():
    A = np.array([[2., 1.], [0.2, 4.]])
    np.testing.assert_allclose(A @ reciprocal_2d(A).T, 2 * np.pi * np.eye(2), atol=1e-14)


def test_nearest_vector_not_fractional_rounding():
    B = np.array([[1., 0.], [4.8, 0.3]])
    target = np.array([0.45, 0.14])
    n = nearest_lattice_vector(target, B)
    indices = np.array([(i, j) for i in range(-40, 41) for j in range(-8, 9)])
    expected = np.min(np.linalg.norm(indices @ B - target, axis=1))
    assert np.linalg.norm(n @ B - target) == pytest.approx(expected)
    assert np.linalg.norm(n @ B - target) < np.linalg.norm(np.rint(target @ np.linalg.inv(B)) @ B - target)


def test_principal_strains_stretch_rotation_and_basis():
    layer = np.array([[2., 0.1], [0.5, 3.]])
    F = rotation(27).T @ np.diag([1.03, 0.98])
    common = layer @ F.T
    np.testing.assert_allclose(principal_strains(layer, common), [0.03, -0.02], atol=1e-14)
    U = np.array([[1, 4], [0, 1]])
    np.testing.assert_allclose(principal_strains(U @ layer, U @ common), [0.03, -0.02], atol=1e-14)
    np.testing.assert_allclose(principal_strains(layer @ rotation(13), common @ rotation(13)), [0.03, -0.02], atol=1e-14)


@pytest.mark.parametrize('theta', [-47, 27])
def test_deformation_polar_angle(theta):
    layer = np.array([[2., 0.1], [0.5, 3.]])
    Q = rotation(17)
    stretch = Q.T @ np.diag([1.03, 0.98]) @ Q
    common = layer @ stretch @ rotation(theta)
    for basis in (np.eye(2), [[0, 1], [1, 5]]):
        strains, phi = deformation(basis @ layer, basis @ common)
        np.testing.assert_allclose(strains, [0.03, -0.02], atol=1e-14)
        assert phi == pytest.approx(theta)


def test_first_order_components_period_and_multiplicity():
    a_b, a_t, theta = 2.46, 2.504, 5
    components = moire_components(hexagonal(a_b), hexagonal(a_t) @ rotation(theta), 1)
    assert len(components) == 1
    c = components[0]
    expected = np.sqrt(3) / 2 * a_b * a_t / np.sqrt(a_b**2 + a_t**2 - 2 * a_b * a_t * np.cos(np.radians(theta)))
    assert c.period == pytest.approx(expected)
    assert c.order == pytest.approx(1)
    assert c.multiplicity == 3


def test_higher_index_sqrt_three_component():
    bottom = hexagonal(2.46)
    epsilon = 0.01
    top = np.sqrt(3) * (1 + epsilon) * bottom @ rotation(30)
    components = moire_components(bottom, top, 2)
    c = next(c for c in components if np.isclose(c.order, np.sqrt(3)))
    expected = 2 * np.pi * (1 + epsilon) / (epsilon * np.linalg.norm(reciprocal_2d(bottom)[0]))
    assert c.period == pytest.approx(expected)
    assert c.multiplicity == 3
    with pytest.raises(ValueError, match='one-to-one first-order pairing'):
        supercell_matrices(bottom, top)


@pytest.mark.parametrize('U,V', [
    ([[1, 5], [0, 1]], [[2, 3], [1, 2]]),
    ([[0, 1], [1, 0]], [[0, -1], [1, 0]]),
])
def test_components_basis_invariance(U, V):
    bottom = hexagonal(2.46)
    top = hexagonal(2.504) @ rotation(5)
    original = moire_components(bottom, top, 4)
    changed = moire_components(U @ bottom, V @ top, 4)
    assert len(original) == len(changed)
    for a, b in zip(original, changed):
        assert a.order == pytest.approx(b.order)
        assert a.period == pytest.approx(b.period)
        assert a.multiplicity == b.multiplicity
        np.testing.assert_allclose(b.q, np.array(b.n) @ reciprocal_2d(U @ bottom) - np.array(b.m) @ reciprocal_2d(V @ top), atol=1e-12)
        from math import gcd
        assert gcd(*b.n, *b.m) == 1


def test_coincident_components_have_infinite_period():
    components = moire_components(hexagonal(2.46), hexagonal(2.46), 2)
    assert len(components) == 2  # order 1 and √3; order 2 is a harmonic
    assert all(np.isinf(c.period) and np.isnan(c.angle_deg) for c in components)
    assert all(c.multiplicity == 3 for c in components)


@pytest.mark.parametrize('top', [hexagonal(2.46), np.diag([1, 1.02]) @ hexagonal(2.46)])
def test_no_finite_two_dimensional_moire(top):
    with pytest.raises(ValueError, match='singular'):
        moire_lattice(hexagonal(2.46), top)


def test_near_zero_twist_geometry():
    bottom = hexagonal(2.46)
    theta = 1e-6
    moire = moire_lattice(bottom, bottom @ rotation(theta))
    assert np.all(np.isfinite(moire))
    expected_length = 2.46 / (2 * np.sin(np.radians(theta) / 2))
    np.testing.assert_allclose(np.linalg.norm(moire, axis=1), expected_length, rtol=1e-8)


@pytest.mark.parametrize('max_order', [0, -1, float('inf')])
def test_invalid_order(max_order):
    with pytest.raises(ValueError, match='max_order'):
        moire_components(np.eye(2), 1.01 * np.eye(2), max_order)
