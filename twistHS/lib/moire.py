"""Two-dimensional moiré geometry with lattice vectors stored as rows.

The aligned-basis special case is Hermann, J. Phys.: Condens. Matter 24,
314210 (2012), doi:10.1088/0953-8984/24/31/314210. Higher-index reciprocal
components follow the construction discussed by Pham et al., npj 2D Mater.
Appl. 6, 48 (2022), doi:10.1038/s41699-022-00321-9.
"""

from dataclasses import dataclass, replace
from itertools import product
from math import gcd

import numpy as np


def _lattice(A):
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2) or not np.all(np.isfinite(A)):
        raise ValueError("Lattice must be a finite 2x2 matrix")
    if np.linalg.det(A) == 0:
        raise ValueError("Lattice must be nonsingular")
    return A


def reciprocal_2d(A):
    """Return the reciprocal row basis, including the factor 2π."""
    return 2 * np.pi * np.linalg.inv(_lattice(A)).T


def reduce_basis(A):
    """Lagrange-reduce A, returning (U @ A, U) with integer det(U) = +1.

    Equal-length choices use Cartesian directions to make the working basis
    independent of the supplied primitive basis, including hexagonal ties.
    The handedness of the input basis is retained.
    """
    A = _lattice(A)
    U = np.eye(2, dtype=int)
    R = A.copy()
    while True:
        if np.dot(R[1], R[1]) < np.dot(R[0], R[0]) * (1 - 1e-12):
            U = np.array([U[1], -U[0]])
            R = U @ A
        mu = int(np.rint(np.dot(R[0], R[1]) / np.dot(R[0], R[0])))
        if mu == 0:
            break
        U[1] -= mu * U[0]
        R = U @ A
        if np.dot(R[1], R[1]) >= np.dot(R[0], R[0]) * (1 - 1e-12):
            break

    # All tied shortest vectors and reduced partners lie in this small shell.
    indices = np.array([v for v in product(range(-1, 2), repeat=2) if any(v)])
    vectors = indices @ R
    lengths = np.linalg.norm(vectors, axis=1)
    shortest = np.isclose(lengths, lengths.min(), rtol=1e-10, atol=0)
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi)
    angles[np.isclose(angles, 2 * np.pi, atol=1e-10)] = 0
    first = min(np.flatnonzero(shortest), key=lambda i: angles[i])
    v = indices[first]
    partners = np.flatnonzero(v[0] * indices[:, 1] - v[1] * indices[:, 0] == 1)
    min_length = lengths[partners].min()
    partners = [i for i in partners if np.isclose(lengths[i], min_length, rtol=1e-10, atol=0)]
    second = min(partners, key=lambda i: angles[i])
    U = np.array([v, indices[second]]) @ U
    return U @ A, U


def nearest_lattice_vector(target, B):
    """Return integer coefficients of the exact nearest vector in row basis B.

    Reduction keeps the search small for skewed input bases. If the rounded
    point has error d0, any better integer point lies within 2*d0/σ_min of it.
    """
    R, U = reduce_basis(B)
    target = np.asarray(target, dtype=float)
    center = np.rint(target @ np.linalg.inv(R)).astype(int)
    d0 = np.linalg.norm(center @ R - target)
    radius = int(np.ceil(2 * d0 / np.linalg.svd(R, compute_uv=False)[-1]))
    offsets = np.array(list(product(range(-radius, radius + 1), repeat=2)))
    candidates = center + offsets
    errors = np.linalg.norm(candidates @ R - target, axis=1)
    return candidates[np.argmin(errors)] @ U


def _aligned_bases(A_b, A_t):
    bottom, U_b = reduce_basis(A_b)
    B_b = reciprocal_2d(bottom)
    B_t = reciprocal_2d(A_t)
    N = np.array([nearest_lattice_vector(b, B_t) for b in B_b])
    if abs(round(np.linalg.det(N))) != 1:
        raise ValueError("No one-to-one first-order pairing (|det N| must be 1)")
    U_t = np.rint(np.linalg.inv(N).T).astype(int)
    difference = B_b - N @ B_t
    scale = max(np.linalg.norm(B_b), np.linalg.norm(N @ B_t))
    if np.linalg.svd(difference, compute_uv=False)[-1] <= 1e-12 * scale:
        raise ValueError("No finite two-dimensional moiré lattice: reciprocal difference is singular (lattices may coincide)")
    return bottom, U_t @ A_t, U_b, U_t


def moire_lattice(A_b, A_t):
    """Return a first-order moiré row basis after nearest reciprocal pairing.

    A_t already includes the desired rotation. A finite 2D difference and a
    unimodular pairing are required; this is not a coincidence-cell search.
    """
    bottom, top, _, _ = _aligned_bases(A_b, A_t)
    return np.linalg.inv(np.linalg.inv(bottom) - np.linalg.inv(top))


def supercell_matrices(A_b, A_t):
    """Round in aligned working bases and map integer matrices to input bases."""
    bottom, top, U_b, U_t = _aligned_bases(A_b, A_t)
    moire = np.linalg.inv(np.linalg.inv(bottom) - np.linalg.inv(top))
    M_b = np.rint(moire @ np.linalg.inv(bottom)).astype(int) @ U_b
    M_t = np.rint(moire @ np.linalg.inv(top)).astype(int) @ U_t
    if round(np.linalg.det(M_b)) == 0 or round(np.linalg.det(M_t)) == 0:
        raise ValueError("Rounded supercell matrix is singular")
    if np.linalg.det(M_b) < 0:
        M_b[1] *= -1
        M_t[1] *= -1
    return M_b, M_t


def _polar_metrics(C_layer, C):
    F = np.swapaxes(np.linalg.solve(C_layer, C), -1, -2)
    strains = np.linalg.svd(F, compute_uv=False) - 1
    # In 2D, for F = R S with S symmetric positive definite, the skew
    # difference and trace of F equal sin(phi) and cos(phi) times tr(S).
    phi = np.degrees(np.arctan2(F[..., 1, 0] - F[..., 0, 1],
                               F[..., 0, 0] + F[..., 1, 1]))
    return strains, phi


def deformation(C_layer, C):
    """Return engineering principal strains and the signed CCW polar angle in degrees."""
    C_layer, C = _lattice(C_layer), _lattice(C)
    if np.linalg.det(C_layer) * np.linalg.det(C) < 0:
        raise ValueError("Deformation must preserve orientation")
    strains, phi = _polar_metrics(C_layer, C)
    return strains, float(phi)


def principal_strains(C_layer, C):
    """Return engineering principal strains of the deformation into common C."""
    return _polar_metrics(_lattice(C_layer), _lattice(C))[0]


@dataclass(frozen=True)
class MoireComponent:
    """A representative ± pair; multiplicity counts pairs of equal order/period.

    n and m refer to the original input reciprocal bases, q is Cartesian, and
    angle_deg is atan2(q_y, q_x) in [0, 360). A zero q has no direction (NaN).
    """

    order: float
    n: tuple[int, int]
    m: tuple[int, int]
    q: np.ndarray
    period: float
    angle_deg: float
    multiplicity: int = 1


def moire_components(A_b, A_t, max_order):
    """Analyze primitive nearest-pair components within reciprocal length cutoffs.

    Each layer's cutoff is max_order times its shortest reciprocal length.
    Harmonics and ± duplicates are removed. Equal-order, equal-period pairs
    are grouped with their multiplicity; this does not predict intensity.
    """
    if not np.isfinite(max_order) or max_order < 1:
        raise ValueError("max_order must be finite and at least 1")
    B_b, U_b = reduce_basis(reciprocal_2d(A_b))
    B_t, U_t = reduce_basis(reciprocal_2d(A_t))
    b_min, t_min = np.linalg.norm(B_b[0]), np.linalg.norm(B_t[0])
    cutoff = max_order * b_min
    radius = int(np.ceil(cutoff / np.linalg.svd(B_b, compute_uv=False)[-1]))
    components = []
    for index in product(range(-radius, radius + 1), repeat=2):
        n = np.array(index)
        if not np.any(n):
            continue
        G_b = n @ B_b
        # Choose one of ±G by its Cartesian direction, independent of basis.
        if G_b[0] < -1e-12 * b_min or (abs(G_b[0]) <= 1e-12 * b_min and G_b[1] < 0):
            continue
        if np.linalg.norm(G_b) > cutoff * (1 + 1e-10):
            continue
        m = nearest_lattice_vector(G_b, B_t)
        if not np.any(m):
            continue
        G_t = m @ B_t
        if np.linalg.norm(G_t) > max_order * t_min * (1 + 1e-10):
            continue
        if gcd(*map(int, np.concatenate((n, m)))) > 1:
            continue
        q = G_b - G_t
        q_length = np.linalg.norm(q)
        zero = q_length <= 1e-12 * max(b_min, t_min)
        component = MoireComponent(
            order=float(max(np.linalg.norm(G_b) / b_min, np.linalg.norm(G_t) / t_min)),
            n=tuple(map(int, n @ U_b)),
            m=tuple(map(int, m @ U_t)),
            q=q,
            period=float('inf') if zero else float(2 * np.pi / q_length),
            angle_deg=float('nan') if zero else float(np.degrees(np.arctan2(q[1], q[0])) % 360),
        )
        components.append(component)

    components.sort(key=lambda c: (round(c.order, 10), -c.period, c.angle_deg))
    grouped = []
    for component in components:
        for i, existing in enumerate(grouped):
            if (np.isclose(component.order, existing.order, rtol=1e-10, atol=0)
                    and np.isclose(component.period, existing.period, rtol=1e-10, atol=0)):
                representative = component if component.angle_deg < existing.angle_deg else existing
                grouped[i] = replace(representative, multiplicity=existing.multiplicity + 1)
                break
        else:
            grouped.append(component)
    return grouped
