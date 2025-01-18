from contextlib import nullcontext
from itertools import product
from math import gcd
from pathlib import Path

import ase.io
import numpy as np
import pytest

from twistHS.lib.coincidence import (
    Coincidence,
    _disk_mask,
    _evaluate,
    _matches,
    _moire_projection,
    _moire_rows,
    _search_bounds,
    find_coincidence,
)
from twistHS.lib.moire import _component_pairs, nearest_lattice_vector, reciprocal_2d, reduce_basis


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    if request.param == "jax":
        pytest.importorskip("jax")
    return request.param


@pytest.fixture
def evaluation(backend):
    if backend == "jax":
        from twistHS.lib._coincidence_jax import enable_x64, evaluate

        return evaluate, enable_x64
    return _evaluate, nullcontext


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


def reciprocal_pairs(bottom, top):
    def pair(source, target):
        basis, _ = reduce_basis(source)
        nearest = np.array([nearest_lattice_vector(g, target) @ target for g in basis])
        return basis, nearest

    G_b, G_t = reciprocal_2d(bottom), reciprocal_2d(top)
    return pair(G_b, G_t), pair(G_t, G_b)[::-1]


def deformed_difference(G_b, G_t, B, T):
    C = (B + T) / 2
    return (G_b @ (np.linalg.inv(C) @ B).swapaxes(-1, -2)
            - G_t @ (np.linalg.inv(C) @ T).swapaxes(-1, -2))


def higher_order_references(bottom, top, cutoff):
    if cutoff == 1:
        return np.empty((2, 0, 2))
    # The cells used here bound the length of a nearest vector by twice the
    # source cutoff (zero is always a candidate), so coefficients fit in ±12.
    indices = np.array(list(product(range(-12, 13), repeat=2)))
    G = [indices @ reciprocal_2d(layer) for layer in (bottom, top)]
    lengths = [np.linalg.norm(vectors, axis=1) for vectors in G]
    shortest = [length[length > 0].min() for length in lengths]
    orders = [length / unit for length, unit in zip(lengths, shortest)]
    pairs = set()
    for source, target in ((0, 1), (1, 0)):
        for i in np.flatnonzero((orders[source] > 0) & (orders[source] <= cutoff * (1 + 1e-10))):
            j = np.argmin(np.linalg.norm(G[target] - G[source][i], axis=1))
            if not 0 < orders[target][j] <= cutoff * (1 + 1e-10) or gcd(*indices[i], *indices[j]) > 1:
                continue
            if max(orders[source][i], orders[target][j]) <= 1 + 1e-10:
                continue
            if np.linalg.norm(G[source][i] - G[target][j]) <= 1e-12 * max(shortest):
                continue
            pair = (i, j) if source == 0 else (j, i)
            # The ordered grid puts a vector and its negative at opposite offsets.
            opposite = tuple(len(indices) - 1 - k for k in pair)
            pairs.add(min(pair, opposite))
    return np.array([(G[0][i], G[1][j]) for i, j in sorted(pairs)]).reshape(-1, 2, 2).swapaxes(0, 1)


def higher_order_errors(G_b, G_t, B, T):
    Q = G_b - G_t
    deformed = deformed_difference(G_b, G_t, B, T)
    return np.linalg.norm(deformed - Q, axis=-1) / np.linalg.norm(Q, axis=-1)


def brute_pairs(bottom, top, budget, tolerance, angle_tol, moire_tol, moire_order):
    # For these test inputs, the stretch bounds give top coefficient magnitude
    # below 11 for every row of a bottom HNF of determinant at most 4.
    candidates = np.array(list(product(range(-10, 11), repeat=4))).reshape(-1, 2, 2)
    det_t = np.rint(np.linalg.det(candidates))
    references = [(G_b, G_t, G_b - G_t) for G_b, G_t in reciprocal_pairs(bottom, top)]
    higher = higher_order_references(bottom, top, moire_order)
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
            valid_top = T[positive][valid]
            moire_ok = np.ones(len(valid_top), dtype=bool)
            for G_b, G_t, Q in references:
                deformed = deformed_difference(G_b, G_t, B, valid_top)
                error = np.linalg.norm(np.linalg.solve(Q, deformed - Q), ord=2, axis=(-2, -1))
                moire_ok &= error <= moire_tol
            moire_ok &= np.all(higher_order_errors(*higher, B, valid_top) <= moire_tol, axis=-1)
            for m in top_matrices[positive][valid][moire_ok]:
                yield H, m


def pair_key(bottom, top):
    return (*bottom.ravel(), *top.ravel())


@pytest.mark.parametrize('top,tolerance,angle_tol,moire_tol,moire_order,pair_count', [
    (np.eye(2) * 0.68, 0.20, 0.01, 0.90, 1, 1),
    (np.eye(2) * 0.68, 0.20, 0.01, 0.90, 1.5, 1),
    (np.eye(2) * 0.68, 0.20, 0.01, 0.90, 3, 0),
    (np.array([[0.505, 0.002], [-0.003, 0.507]]), 0.20, 12, 0.05, 1, 0),
    (np.array([[0.89, 0.12], [0.16, 1.02]]), 0.20, 12, 0.05, 1, 0),
])
def test_complete_search_against_brute_force(top, tolerance, angle_tol, moire_tol, moire_order, pair_count, backend):
    bottom = np.eye(2)
    budget = 5
    expected = list(brute_pairs(bottom, top, budget, tolerance, angle_tol, moire_tol, moire_order))
    assert len(expected) == pair_count
    kernels, precision = {}, nullcontext
    if backend == "jax":
        from twistHS.lib import _coincidence_jax

        kernels = {"boxes": _coincidence_jax.boxes, "evaluate": _coincidence_jax.evaluate}
        precision = _coincidence_jax.enable_x64
    projection = _moire_projection(bottom, top, moire_tol)
    rows = _moire_rows(bottom, top, moire_tol, moire_order)
    radius, area_bounds = _search_bounds(bottom, top, tolerance, angle_tol, projection)
    with precision():
        actual = list(_matches(1, budget - 1, bottom, top, (np.eye(2, dtype=int),) * 2,
                               (1, 1), tolerance, angle_tol, budget, radius, area_bounds,
                               projection, rows, **kernels))

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
                             angle_tol=angle_tol, moire_tol=100 * moire_tol,
                             moire_order=moire_order, max_atoms=budget, backend=backend)
    if not expected:
        assert match is None
        return
    expected_count = min(round(np.linalg.det(h) + np.linalg.det(m)) for h, m in expected)
    assert match.atom_count == expected_count
    assert match.rank() == min(candidate.rank() for candidate in actual)


def test_moire_search_is_invariant_under_layer_exchange(backend):
    hexagonal = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    small, large = 2.46 * hexagonal, 5.2 * hexagonal
    theta = np.radians(4)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, s], [-s, c]])
    kwargs = dict(max_atoms=3000, backend=backend)
    forward = find_coincidence(small, large @ rotation, (2, 2), **kwargs)
    reverse = find_coincidence(large, small @ rotation.T, (2, 2), **kwargs)
    assert forward is not None and reverse is not None
    assert forward.atom_count == reverse.atom_count
    assert forward.strain == pytest.approx(reverse.strain, abs=1e-12, rel=0)


def test_inclusive_search_bounds(backend):
    bottom = np.eye(2)
    scale = 1.2578125
    top = scale * bottom
    match = find_coincidence(bottom, top, (1, 1), max_atoms=41, backend=backend)
    tolerance = match.strain * 100
    moire_tol = 100 * abs((scale + 1) / (scale - 1) * (5 - 4 * scale) / (5 + 4 * scale))
    kwargs = dict(tolerance=tolerance, angle_tol=0, moire_tol=moire_tol, max_atoms=41, backend=backend)
    assert find_coincidence(bottom, top, (1, 1), **kwargs)
    for limit, decrease in [('tolerance', 0.01), ('moire_tol', 0.01), ('max_atoms', 1)]:
        assert find_coincidence(bottom, top, (1, 1), **(kwargs | {limit: kwargs[limit] - decrease})) is None

    c, s = np.cos(0.001), np.sin(0.001)
    top = scale * np.array([[c, s], [-s, c]])
    match = find_coincidence(bottom, top, (1, 1), max_atoms=41, backend=backend)
    angle_tol = abs(match.angle_offset)
    assert find_coincidence(bottom, top, (1, 1), angle_tol=angle_tol, max_atoms=41, backend=backend)
    assert find_coincidence(bottom, top, (1, 1), angle_tol=angle_tol - 0.01, max_atoms=41, backend=backend) is None


def test_zero_and_rank_one_moire(backend, evaluation):
    bottom = np.eye(2)
    match = find_coincidence(bottom, bottom, (1, 1), moire_tol=0, max_atoms=2, backend=backend)
    assert match.strain == match.angle_offset == 0

    top = np.diag([1.2578125, 1])
    match = find_coincidence(bottom, top, (1, 1), max_atoms=9, backend=backend)
    assert match.atom_count == 9
    for G_b, G_t in reciprocal_pairs(bottom, top):
        Q = G_b - G_t
        assert np.linalg.matrix_rank(Q) == 1
        deformed = deformed_difference(G_b, G_t, match.bottom @ bottom, match.top @ top)
        np.testing.assert_array_equal(deformed[~Q.any(axis=1)], [[0, 0]])

    # A small mismatch along y creates a wave vector where the reference has zero.
    evaluate, precision = evaluation
    M_b = np.array([np.diag([5, 100])])
    M_t = np.array([np.diag([4, 101])])
    projection = _moire_projection(bottom, top, 0.05)
    with precision():
        strain, offset, feasible = evaluate(M_b, M_t, bottom, top, 0.03, projection, np.empty((0, 2)))
    assert strain[0] < 0.01
    assert offset[0] == 0
    assert not feasible[0]


def test_all_directions_reject_strip_collapse(evaluation):
    bottom = np.eye(2)
    top = np.linalg.inv(np.eye(2) - np.ones((2, 2)) / 50 - np.diag([1e-4, 0])).T
    M_b = np.array([[[25, 25], [-1, 1]]])
    M_t = np.array([[[24, 24], [-1, 1]]])
    G_b, G_t = reciprocal_pairs(bottom, top)[0]
    Q = G_b - G_t
    deformed = deformed_difference(G_b, G_t, M_b @ bottom, M_t @ top)[0]
    row_errors = np.linalg.norm(deformed - Q, axis=1) / np.linalg.norm(Q, axis=1)
    assert np.all(row_errors < 0.004)
    assert np.linalg.svd(deformed, compute_uv=False)[-1] < 1e-13
    assert np.linalg.norm(np.linalg.solve(Q, deformed - Q), ord=2) > 0.99

    evaluate, precision = evaluation
    projection = _moire_projection(bottom, top, 0.05)
    with precision():
        strain, offset, feasible = evaluate(M_b, M_t, bottom, top, 0.03, projection, np.empty((0, 2)))
    assert strain[0] < 0.01
    assert abs(offset[0]) < 0.1
    assert not feasible[0]


def test_moire_search_is_invariant_under_primitive_basis_changes(backend):
    bottom = np.eye(2)
    top = np.array([[1.2578125, 0.002], [-0.003, 1.25]])
    U = np.array([[1, 4], [0, -1]])
    V = np.array([[2, 1], [1, 1]])
    kwargs = dict(tolerance=2, angle_tol=1, max_atoms=41, backend=backend)
    match = find_coincidence(bottom, top, (1, 1), **kwargs)
    changed = find_coincidence(U @ bottom, V @ top, (1, 1), **kwargs)
    assert changed.atom_count == match.atom_count
    assert changed.strain == pytest.approx(match.strain, abs=1e-12)
    assert changed.angle_offset == pytest.approx(match.angle_offset, abs=1e-12)
    np.testing.assert_array_equal(changed.bottom @ U, match.bottom)
    np.testing.assert_array_equal(changed.top @ V, match.top)


def test_rank_breaks_ties_using_integer_matrices():
    identity = np.eye(2, dtype=int)
    preferred = Coincidence(-identity, -identity, 4, 0.001 + 1e-16, 359.95 - 1e-12)
    other = Coincidence(identity, identity, 4, 0.001, 0.05)
    assert preferred.rank() < other.rank()


def test_small_angle_graphene_moire(backend):
    graphene = ase.io.read(Path(__file__).parent / "graphene.xsf")
    bottom = graphene.cell[:2, :2]
    for theta, budget in [(1, 124), (1.05, 12000)]:
        top_layer = graphene.copy()
        top_layer.rotate(theta, "z", rotate_cell=True)
        top = top_layer.cell[:2, :2]
        match = find_coincidence(bottom, top, (2, 2), max_atoms=budget, backend=backend)
        if theta == 1:
            assert match is None
        else:
            assert match.atom_count <= budget
            for G_b, G_t in reciprocal_pairs(bottom, top):
                Q = G_b - G_t
                deformed = deformed_difference(G_b, G_t, match.bottom @ bottom, match.top @ top)
                assert np.linalg.norm(np.linalg.solve(Q, deformed - Q), ord=2) <= 0.05
                assert np.linalg.matrix_rank(deformed) == 2


def test_higher_order_protection_omits_infinite_periods():
    bottom = np.eye(2)
    top = (2 / 3) * bottom
    higher = [c for c in _component_pairs(bottom, top, 3) if c.order > 1]
    assert any(np.isinf(c.period) for c in higher)
    rows = _moire_rows(bottom, top, 0.05, 3)
    assert rows.shape == (sum(np.isfinite(c.period) for c in higher), 2)


@pytest.mark.parametrize('theta,default_error', [(0, 0.08424), (5, 0.11000)])
def test_graphene_mos2_higher_order_protection(theta, default_error, backend):
    root = Path(__file__).parent
    graphene = ase.io.read(root / "graphene.xsf")
    mos2 = ase.io.read(root / "MoS2.xsf")
    mos2.rotate(theta, "z", rotate_cell=True)
    bottom, top = graphene.cell[:2, :2], mos2.cell[:2, :2]
    counts = (len(graphene), len(mos2))
    references = higher_order_references(bottom, top, 4)

    default = find_coincidence(bottom, top, counts, backend=backend)
    protected = find_coincidence(bottom, top, counts, moire_order=4, backend=backend)
    assert default is not None and protected is not None
    assert default.atom_count < protected.atom_count <= 12_000
    assert higher_order_errors(*references, default.bottom @ bottom, default.top @ top).max() == pytest.approx(
        default_error, abs=1e-5)
    assert np.all(higher_order_errors(*references, protected.bottom @ bottom, protected.top @ top) <= 0.05)


@pytest.mark.parametrize('kwargs', [
    {'tolerance': -1}, {'angle_tol': np.inf}, {'max_atoms': 4.5}, {'backend': 'gpu'},
    {'moire_tol': -1}, {'moire_tol': 100}, {'moire_tol': np.nan},
    {'moire_order': 0.99}, {'moire_order': np.inf}, {'moire_order': np.nan}, {'moire_order': 1 + 1j},
])
def test_invalid_search_limits(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        find_coincidence(np.eye(2), np.eye(2), (1, 1), **kwargs)
