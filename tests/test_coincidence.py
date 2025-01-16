from contextlib import nullcontext
from itertools import product

import numpy as np
import pytest

from twistHS.lib.coincidence import Coincidence, _disk_mask, _matches, find_coincidence
from twistHS.lib.moire import nearest_lattice_vector


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    if request.param == "jax":
        pytest.importorskip("jax")
    return request.param


def test_lattice_vector_disk_against_nearest_oracle(backend):
    xp, precision = np, nullcontext
    if backend == "jax":
        from twistHS.lib._coincidence_jax import enable_x64, jnp

        xp, precision = jnp, enable_x64
    lattice = np.array([[1.0, 0], [0.8, 0.3]])
    coefficients = np.array(list(product(range(-5, 6), repeat=2)))
    vectors = coefficients @ lattice
    for target in (np.array([0.45, 0.14]), np.array([-0.57, 0.89])):
        nearest = nearest_lattice_vector(target, lattice)
        distance = np.linalg.norm(nearest @ lattice - target)
        for radius in (distance * 0.99, distance * 1.01, distance * 3):
            with precision():
                matches = np.asarray(_disk_mask(
                    xp.asarray(target), xp.asarray(vectors), radius / np.linalg.norm(target), xp))
            np.testing.assert_array_equal(matches, np.linalg.norm(vectors - target, axis=1) <= radius)
            assert bool(matches.any()) == (radius > distance)


def brute_metrics(layer, common):
    F = (np.linalg.inv(layer) @ common).swapaxes(-1, -2)
    U, singular, Vt = np.linalg.svd(F)
    R = U @ Vt
    return np.max(np.abs(singular - 1), axis=-1), np.degrees(np.arctan2(R[..., 1, 0], R[..., 0, 0]))


def brute_pairs(bottom, top, budget, tolerance, angle_tol):
    # For these test inputs, the stretch bounds give top coefficient magnitude
    # below 11 for every row of a bottom HNF of determinant at most 4.
    candidates = np.array(list(product(range(-10, 11), repeat=4))).reshape(-1, 2, 2)
    det_t = np.rint(np.linalg.det(candidates))
    for a, d in product(range(1, budget), repeat=2):
        if a * d >= budget:
            continue
        top_matrices = candidates[(det_t > 0) & (det_t + a * d <= budget)]
        T = top_matrices @ top
        for b in range(a):
            H = np.array([[a, 0], [b, d]])
            B = H @ bottom
            C = (B + T) / 2
            positive = np.linalg.det(C) > 0
            eps_b, phi_b = brute_metrics(B, C[positive])
            eps_t, phi_t = brute_metrics(T[positive], C[positive])
            eps = np.maximum(eps_b, eps_t)
            offsets = (phi_t - phi_b + 180) % 360 - 180
            valid = (eps <= tolerance) & (np.abs(offsets) <= angle_tol)
            for m in top_matrices[positive][valid]:
                yield H, m


def pair_key(bottom, top):
    return (*bottom.ravel(), *top.ravel())


@pytest.mark.parametrize('top,tolerance,angle_tol', [
    (np.eye(2) * 0.68, 0.20, 0.01),
    (np.array([[0.89, 0.12], [0.16, 1.02]]), 0.20, 12),
])
def test_complete_search_against_brute_force(top, tolerance, angle_tol, backend):
    bottom = np.eye(2)
    budget = 5
    expected = list(brute_pairs(bottom, top, budget, tolerance, angle_tol))
    kernels, precision = {}, nullcontext
    if backend == "jax":
        from twistHS.lib import _coincidence_jax

        kernels = {"boxes": _coincidence_jax.boxes, "evaluate": _coincidence_jax.evaluate}
        precision = _coincidence_jax.enable_x64
    with precision():
        actual = list(_matches(1, budget - 1, bottom, top, (np.eye(2, dtype=int),) * 2,
                               (1, 1), tolerance, angle_tol, budget, **kernels))

    # Convert each reduced bottom basis back to its HNF, applying the same
    # unimodular row transform to the top matrix before comparing pairs.
    hnfs = {tuple(h.ravel()): h for h, _ in expected}
    keys = set()
    for match in actual:
        for h in hnfs.values():
            change = h @ np.linalg.inv(match.bottom)
            integer = np.rint(change).astype(int)
            if np.array_equal(integer @ match.bottom, h) and round(np.linalg.det(integer)) == 1:
                keys.add(pair_key(h, integer @ match.top))
                break
        else:
            pytest.fail(f"Unexpected bottom lattice {match.bottom}")
    assert len(actual) == len(keys)
    assert keys == {pair_key(h, m) for h, m in expected}
    match = find_coincidence(bottom, top, (1, 1), tolerance=100 * tolerance,
                             angle_tol=angle_tol, max_atoms=budget, backend=backend)
    expected_count = min(round(np.linalg.det(h) + np.linalg.det(m)) for h, m in expected)
    assert match.atom_count == expected_count
    assert match.rank() == min(candidate.rank() for candidate in actual)
    if top[0, 0] == 0.68:
        # The nearest top vectors contract too much. The passing pair uses
        # longer, non-nearest top vectors in both rows.
        assert match.atom_count == 5
        np.testing.assert_array_equal(match.top, 2 * match.bottom)


def test_inclusive_search_bounds(backend):
    bottom = np.eye(2)
    top = 1.25 * bottom
    assert find_coincidence(bottom, top, (1, 1), tolerance=12.5, angle_tol=0, max_atoms=2, backend=backend)
    assert find_coincidence(bottom, top, (1, 1), tolerance=12.49, angle_tol=0, max_atoms=2, backend=backend) is None
    assert find_coincidence(bottom, top, (1, 1), tolerance=12.5, angle_tol=0, max_atoms=1, backend=backend) is None

    top = np.array([[1., 0.125], [-0.125, 1.]])
    match = find_coincidence(bottom, top, (1, 1), angle_tol=10, max_atoms=2, backend=backend)
    angle_tol = abs(match.angle_offset)
    assert find_coincidence(bottom, top, (1, 1), angle_tol=angle_tol, max_atoms=2, backend=backend)
    assert find_coincidence(bottom, top, (1, 1), angle_tol=angle_tol - 0.01, max_atoms=2, backend=backend) is None


def test_rank_breaks_ties_using_integer_matrices():
    identity = np.eye(2, dtype=int)
    preferred = Coincidence(-identity, -identity, 4, 0.001 + 1e-16, 359.95 - 1e-12)
    other = Coincidence(identity, identity, 4, 0.001, 0.05)
    assert preferred.rank() < other.rank()


@pytest.mark.parametrize('kwargs', [
    {'tolerance': -1}, {'angle_tol': np.inf}, {'max_atoms': 4.5}, {'backend': 'gpu'},
])
def test_invalid_search_limits(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        find_coincidence(np.eye(2), np.eye(2), (1, 1), **kwargs)
