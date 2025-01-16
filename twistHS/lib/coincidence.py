"""Bounded search for pairs of two-dimensional coincidence sublattices."""

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from math import isqrt
from operator import index

import numpy as np

from twistHS.lib.moire import _polar_metrics, reduce_basis

_BATCH = 4096
_BOUND_SLACK = 1e-12  # Expand prefilters for roundoff without relaxing the final limits.
DEFAULT_TOLERANCE = 1.0
DEFAULT_ANGLE_TOL = 0.1
DEFAULT_MAX_ATOMS = 12000
SEARCH_BACKENDS = ("numpy", "jax")


@dataclass(frozen=True)
class Coincidence:
    bottom: np.ndarray
    top: np.ndarray
    atom_count: int
    strain: float
    angle_offset: float

    def rank(self):
        offset = abs(self.angle_offset)
        return (self.atom_count, round(self.strain, 9), round(min(offset, 360 - offset), 9),
                *self.bottom.ravel(), *self.top.ravel())


def validate_tolerance(tolerance):
    if not np.isfinite(tolerance) or not 0 <= tolerance < 100:
        raise ValueError("tolerance must be a strain percentage in [0, 100)")


def validate_search_limits(angle_tol, max_atoms):
    if not np.isfinite(angle_tol) or not 0 <= angle_tol <= 180:
        raise ValueError("angle_tol must be in [0, 180] degrees")
    if isinstance(max_atoms, bool) or not isinstance(max_atoms, (int, np.integer)) or max_atoms < 1:
        raise ValueError("max_atoms must be a positive integer")


def validate_search_backend(backend):
    if backend not in SEARCH_BACKENDS:
        raise ValueError("backend must be 'numpy' or 'jax'")


def _det(matrix):
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _hnfs(first, last):
    """Yield lower-triangular row HNFs in order of determinant."""
    for n in range(first, last + 1):
        for factor in range(1, isqrt(n) + 1):
            if n % factor:
                continue
            other = n // factor
            divisors = ((factor, other),) if factor == other else ((factor, other), (other, factor))
            for a, d in divisors:
                for b in range(a):
                    yield ((a, 0), (b, d))


def _batches(items):
    while batch := list(islice(items, _BATCH)):
        yield np.asarray(batch, dtype=np.int64)


def _positive_basis(lattice):
    basis, transform = reduce_basis(lattice)
    if _det(basis) < 0:
        basis[1] *= -1
        transform[1] *= -1
    return basis, transform


def _numpy_while_loop(condition, step, state):
    while condition(state):
        state = step(state)
    return state


def _reduce_supercells(matrices, lattice, xp, while_loop):
    """Reduce the candidate rows together, retaining their integer coefficients."""
    lengths_sq = xp.sum((matrices @ lattice)**2, axis=-1)
    swap = lengths_sq[:, 1] < lengths_sq[:, 0] * (1 - 1e-14)
    swapped = xp.stack((matrices[:, 1], -matrices[:, 0]), axis=1)
    matrices = xp.where(swap[:, None, None], swapped, matrices)

    def step(state):
        current, active = state
        cells = current @ lattice
        first_sq = xp.sum(cells[:, 0]**2, axis=1)
        projection = xp.sum(cells[:, 0] * cells[:, 1], axis=1) / first_sq
        shift = xp.where(active, xp.rint(projection), 0)
        second = current[:, 1] - shift.astype(current.dtype)[:, None] * current[:, 0]
        second_cell = (second[:, None, :] @ lattice)[:, 0]
        swap = active & (xp.sum(second_cell**2, axis=1) < first_sq * (1 - 1e-14))
        reduced = xp.stack((current[:, 0], second), axis=1)
        swapped = xp.stack((second, -current[:, 0]), axis=1)
        return xp.where(swap[:, None, None], swapped, reduced), swap

    return while_loop(lambda state: xp.any(state[1]), step,
                      (matrices, xp.ones(len(matrices), dtype=bool)))[0]


def _boxes(matrices, bottom, top_inverse, relative_radius, *, xp=np, while_loop=_numpy_while_loop):
    matrices = _reduce_supercells(matrices, bottom, xp, while_loop)
    cells = matrices @ bottom
    center = cells @ top_inverse
    radii = relative_radius * xp.linalg.norm(cells, axis=-1)
    extent = radii[..., None] * xp.linalg.norm(top_inverse, axis=0)
    low = xp.ceil(center - extent).astype(matrices.dtype)
    high = xp.floor(center + extent).astype(matrices.dtype)
    return matrices, low, high


def _disk_mask(cells_b, cells_t, relative_radius, xp=np):
    return (xp.sum((cells_b - cells_t)**2, axis=-1)
            <= relative_radius**2 * xp.sum(cells_b**2, axis=-1))


def _evaluate(bottom_matrices, top_matrices, bottom, top, relative_radius, *, xp=np):
    B, T = bottom_matrices @ bottom, top_matrices @ top
    C = (B + T) / 2
    strains_b, phi_b = _polar_metrics(B, C, xp)
    strains_t, phi_t = _polar_metrics(T, C, xp)
    strain = xp.max(xp.maximum(xp.abs(strains_b), xp.abs(strains_t)), axis=-1)
    feasible = (_det(C) > 0) & xp.all(_disk_mask(B, T, relative_radius, xp), axis=-1)
    return strain, phi_t - phi_b, feasible


def _top_pairs(low, high, top_det_min, top_det_max):
    (x0, y0), (u0, v0) = low
    (x1, y1), (u1, v1) = high
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for u in range(u0, u1 + 1):
                for v in range(v0, v1 + 1):
                    if top_det_min <= x * v - y * u <= top_det_max:
                        yield ((x, y), (u, v))


def _matrix_pairs(hnfs, bottom, top_inverse, radius, area_bounds, counts, budget, boxes):
    matrices, low, high = boxes(hnfs, bottom, top_inverse, radius)
    possible = np.flatnonzero(np.all(low <= high, axis=(1, 2)))
    for i in possible:
        matrix = matrices[i]
        det_b = _det(matrix)
        min_t = max(1, int(np.ceil(det_b * area_bounds[0])))
        max_t = min((budget - det_b * counts[0]) // counts[1],
                    int(np.floor(det_b * area_bounds[1])))
        if min_t <= max_t:
            for match in _top_pairs(low[i], high[i], min_t, max_t):
                yield matrix, match


def _area_bounds(bottom, top, tolerance):
    # The area ratio is bounded by the products of the two principal
    # stretches of each layer.
    area_ratio = _det(bottom) / _det(top)
    stretch_ratio = (1 + tolerance) / (1 - tolerance)
    return (area_ratio / stretch_ratio**2 * (1 - _BOUND_SLACK),
            area_ratio * stretch_ratio**2 * (1 + _BOUND_SLACK))


def _matches(first, last, bottom, top, transforms, counts, tolerance, angle_tol, budget,
             *, boxes=_boxes, evaluate=_evaluate):
    """Yield every passing matrix pair for the given bottom determinant interval."""
    area_bounds = _area_bounds(bottom, top, tolerance)
    # T = B (F_t^-1 F_b)^T. The two stretches can differ by at most
    # 2*tolerance and the polar rotations by 2*sin(angle_tol/2) in operator norm.
    radius = ((2 * tolerance + 2 * np.sin(np.radians(angle_tol) / 2)) / (1 - tolerance)
              + _BOUND_SLACK)
    top_inverse = np.linalg.inv(top)
    for hnfs in _batches(_hnfs(first, last)):
        pairs = _matrix_pairs(hnfs, bottom, top_inverse, radius, area_bounds, counts, budget, boxes)
        for batch in _batches(pairs):
            M_b, M_t = batch[:, 0], batch[:, 1]
            strain, offset, feasible = evaluate(M_b, M_t, bottom, top, radius)
            angle_error = np.abs(offset)
            angle_error = np.minimum(angle_error, 360 - angle_error)
            valid = np.flatnonzero(feasible & (strain <= tolerance) & (angle_error <= angle_tol))
            for i in valid:
                yield Coincidence(M_b[i] @ transforms[0], M_t[i] @ transforms[1],
                                  int(_det(M_b[i]) * counts[0] + _det(M_t[i]) * counts[1]),
                                  float(strain[i]), float(offset[i]))


def find_coincidence(A_b, A_t, counts, *, tolerance=DEFAULT_TOLERANCE,
                     angle_tol=DEFAULT_ANGLE_TOL, max_atoms=DEFAULT_MAX_ATOMS, backend="numpy"):
    """Return the smallest accepted pair of integer transforms, or None.

    A_t includes the requested twist. Counts give atoms per primitive cell,
    tolerance is in percent, and angle_tol is in degrees. Select the optional
    JAX kernels with backend="jax".
    """
    validate_tolerance(tolerance)
    validate_search_limits(angle_tol, max_atoms)
    validate_search_backend(backend)
    counts = tuple(map(index, counts))
    if len(counts) != 2 or min(counts) < 1:
        raise ValueError("Both primitive cells must contain atoms")
    bottom, U_b = _positive_basis(A_b)
    top, U_t = _positive_basis(A_t)
    strain = tolerance / 100
    min_ratio, _ = _area_bounds(bottom, top, strain)

    boxes, evaluate, precision = _boxes, _evaluate, nullcontext
    if backend == "jax":
        try:
            from twistHS.lib import _coincidence_jax
        except ModuleNotFoundError as exc:
            if exc.name != "jax":
                raise
            raise ImportError('The JAX backend requires twistHS[jax]') from exc
        boxes = _coincidence_jax.boxes
        evaluate = _coincidence_jax.evaluate
        precision = _coincidence_jax.enable_x64

    best = None
    first, window = 1, 16
    with precision():
        while True:
            budget = max_atoms if best is None else best.atom_count
            last = min(first + window - 1, (budget - counts[1]) // counts[0],
                       int(budget / (counts[0] + min_ratio * counts[1])))
            if last < first:
                return best
            for match in _matches(first, last, bottom, top, (U_b, U_t), counts, strain, angle_tol,
                                  budget, boxes=boxes, evaluate=evaluate):
                if best is None or match.rank() < best.rank():
                    best = match
            first = last + 1
            window *= 2
