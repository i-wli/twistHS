"""Periodic metal-plane separation and local stacking of 1H MX₂ bilayers."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def _split_layers(atoms, metal, chalcogen):
    """Return bottom/top indices in atom order, validating each MX₂ layer."""
    order = np.argsort(atoms.positions[:, 2])
    cut = np.argmax(np.diff(atoms.positions[order, 2])) + 1
    layers = tuple(np.sort(layer) for layer in np.split(order, [cut]))
    symbols = np.asarray(atoms.get_chemical_symbols())
    for layer in layers:
        n_metal = np.count_nonzero(np.isin(symbols[layer], np.atleast_1d(metal)))
        n_chalcogen = np.count_nonzero(np.isin(symbols[layer], np.atleast_1d(chalcogen)))
        if n_metal == 0 or n_chalcogen != 2 * n_metal or n_metal + n_chalcogen != len(layer):
            raise ValueError("Each layer must contain only metal and chalcogen in MX2 stoichiometry")
    return layers


def _wrapped_xy(atoms, index):
    cell = atoms.cell[:2, :2]
    return (atoms.positions[index, :2] @ np.linalg.inv(cell) % 1) @ cell


def _tile(xy, cell2d):
    shifts = np.array(list(product((-1, 0, 1), repeat=2))) @ cell2d
    return (xy[None, :, :] + shifts[:, None, :]).reshape(-1, 2)


def _layer_sites(atoms, layer, metal, chalcogen):
    symbols = np.asarray(atoms.get_chemical_symbols())[layer]
    M = _wrapped_xy(atoms, layer[np.isin(symbols, np.atleast_1d(metal))])
    X = _wrapped_xy(atoms, layer[np.isin(symbols, np.atleast_1d(chalcogen))])
    tiled = _tile(X, atoms.cell[:2, :2])
    _, neighbors = cKDTree(tiled).query(M, k=6)
    return M, X, tiled[neighbors] - M[:, None, :]


def interlayer_distance_map(atoms, metal, chalcogen):
    """Return wrapped top metal xy and separation from the bottom metal plane (Å)."""
    bottom, top = _split_layers(atoms, metal, chalcogen)
    symbols = np.asarray(atoms.get_chemical_symbols())
    bottom = bottom[np.isin(symbols[bottom], np.atleast_1d(metal))]
    top = top[np.isin(symbols[top], np.atleast_1d(metal))]
    xy = _wrapped_xy(atoms, top)
    plane = LinearNDInterpolator(
        _tile(_wrapped_xy(atoms, bottom), atoms.cell[:2, :2]),
        np.tile(atoms.positions[bottom, 2], 9),
    )
    return xy, atoms.positions[top, 2] - plane(xy)


@dataclass(frozen=True)
class StackingMap:
    """Local labels and top-metal site fractions for an R or H bilayer."""

    xy: np.ndarray
    labels: np.ndarray
    fractions: dict[str, float]
    family: str


def stacking_map(atoms, metal, chalcogen, boundary=None):
    """Assign nearest M/X/hollow registries, optionally within boundary Å."""
    bottom, top = _split_layers(atoms, metal, chalcogen)
    M_bottom, X_bottom, tau_bottom = _layer_sites(atoms, bottom, metal, chalcogen)
    xy, _, tau_top = _layer_sites(atoms, top, metal, chalcogen)
    p_bottom, p_top = [
        np.mean(np.exp(3j * np.arctan2(tau[..., 1], tau[..., 0])))
        for tau in (tau_bottom, tau_top)
    ]
    delta = np.degrees(np.angle(p_top * np.conj(p_bottom))) / 3
    family = "R" if abs(delta) < 30 else "H"
    names = ["Rhh", "RhX", "RhM"] if family == "R" else ["HhX", "Hhh", "HhM"]
    hollow = (M_bottom[:, None, :] - tau_bottom).reshape(-1, 2)
    distances = np.column_stack([
        cKDTree(_tile(points, atoms.cell[:2, :2])).query(xy)[0]
        for points in (M_bottom, X_bottom, hollow)
    ])
    labels = np.asarray(names, dtype="U8")[np.argmin(distances, axis=1)]
    if boundary is not None:
        labels[np.min(distances, axis=1) > boundary] = "boundary"
        names.append("boundary")
    fractions = {name: float(np.mean(labels == name)) for name in names}
    return StackingMap(xy, labels, fractions, family)
