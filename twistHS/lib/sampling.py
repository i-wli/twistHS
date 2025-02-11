"""Small monolayer and aligned homobilayer configurations for reference labeling."""

import numpy as np

from twistHS.lib.analysis import stacking_map


def _centered_layer(layer, super_z):
    base = layer.copy()
    z = base.positions[:, 2]
    z -= (z.min() + z.max()) / 2
    base.cell[2] = (0, 0, super_z)
    base.pbc = True
    return base


def _perturb(atoms, rng, sigma_atom, strain):
    e1, e2 = rng.uniform(-strain, strain), rng.uniform(-strain, strain)
    phi = rng.uniform(0, np.pi)
    c, s = np.cos(phi), np.sin(phi)
    rotation = np.array([[c, -s], [s, c]])
    deformation = np.eye(3)
    deformation[:2, :2] += rotation @ np.diag([e1, e2]) @ rotation.T
    atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)
    atoms.positions += rng.normal(size=(len(atoms), 3)) * (
        sigma_atom / np.sqrt(atoms.get_masses())
    )[:, None]
    atoms.wrap()
    return atoms


def sample_aligned_bilayers(layer, n_configs, seed, stacking, d_range, super_z,
                            supercell=3, sigma_atom=0.5, strain=0.0):
    """Sample R/H homobilayers with uniform translation and separation.

    d_range is the (min, max) separation of the layer midplanes in Å, which
    for 1H MX₂ is the metal-plane separation. strain is the half-width
    of each uniformly sampled in-plane principal strain. sigma_atom is in Å·u^½.
    Each frame contains the bottom layer followed by the top.
    """
    rng = np.random.default_rng(seed)
    if stacking not in ("R", "H"):
        raise ValueError("stacking must be 'R' or 'H'")
    base = _centered_layer(layer, super_z)
    top0 = base.copy()
    if stacking == "H":
        top0.rotate(60, "z")
    bottom = base.repeat((supercell, supercell, 1))
    frames = []
    for _ in range(n_configs):
        top = top0.copy()
        top.positions[:, :2] += rng.random(2) @ base.cell[:2, :2]
        d = rng.uniform(*d_range)
        top = top.repeat((supercell, supercell, 1))
        top.positions[:, 2] += d
        atoms = bottom + top
        atoms.positions[:, 2] += (super_z - d) / 2
        frames.append(_perturb(atoms, rng, sigma_atom, strain))
    return frames


def sample_monolayers(layer, n_configs, seed, super_z,
                      supercell=3, sigma_atom=0.5, strain=0.0):
    """Sample monolayers with uniform principal strains and mass-scaled displacements."""
    rng = np.random.default_rng(seed)
    base = _centered_layer(layer, super_z).repeat((supercell, supercell, 1))
    base.positions[:, 2] += super_z / 2
    return [_perturb(base.copy(), rng, sigma_atom, strain) for _ in range(n_configs)]


def high_symmetry_bilayers(layer, d, super_z):
    """Return the six high-symmetry 1×1 homobilayers keyed by stacking label."""
    base = _centered_layer(layer, super_z)
    metal, *chalcogens = base.get_chemical_symbols()
    tau = (base.positions[1] - base.positions[0])[:2]
    stackings = {}
    for family in "RH":
        for shift in (0, tau, -tau):
            top = base.copy()
            if family == "H":
                top.rotate(60, "z")
            top.positions[:, :2] += base.positions[0, :2] - top.positions[0, :2] + shift
            top.positions[:, 2] += d
            atoms = base + top
            atoms.positions[:, 2] += (super_z - d) / 2
            atoms.wrap()
            stackings[stacking_map(atoms, metal, chalcogens).labels[0]] = atoms
    if stackings.keys() != {"Rhh", "RhX", "RhM", "HhX", "Hhh", "HhM"}:
        raise ValueError(f"Expected six high-symmetry stackings, got {list(stackings)}")
    return stackings
