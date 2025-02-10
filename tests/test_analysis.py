from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from ase.geometry import find_mic

from twistHS.lib.algorithm import HSGenerator
from twistHS.lib.analysis import interlayer_distance_map, stacking_map


@pytest.fixture
def monolayer():
    return read(Path(__file__).parent / "MoS2.xsf")


def aligned_bilayer(unit, family, shift):
    top = unit.copy()
    if family == "H":
        top.rotate(60, "z", center=unit.positions[0])
        top.wrap()
    bottom = unit.repeat((3, 3, 1))
    top = top.repeat((3, 3, 1))
    top.positions[:, :2] += shift
    top.positions[:, 2] += 6.5
    atoms = bottom + top
    atoms.pbc = True
    return atoms


@pytest.mark.parametrize("family,labels", [
    ("R", ("Rhh", "RhX", "RhM")), ("H", ("HhX", "Hhh", "HhM")),
])
@pytest.mark.parametrize("shift_index", range(3))
def test_aligned_registries(monolayer, family, labels, shift_index):
    vectors = monolayer.positions[1:] - monolayer.positions[0]
    vectors[:, 2] = 0
    vectors, lengths = find_mic(vectors, monolayer.cell, pbc=(True, True, False))
    tau = vectors[np.argmin(lengths), :2]
    atoms = aligned_bilayer(monolayer, family, (np.zeros(2), tau, -tau)[shift_index])
    mapping = stacking_map(atoms, ["Mo", "W"], ["S", "Se"])
    assert mapping.family == family
    assert np.all(mapping.labels == labels[shift_index])
    assert mapping.fractions[labels[shift_index]] == 1
    assert all(type(key) is str for key in mapping.fractions)
    xy, d = interlayer_distance_map(atoms, "Mo", "S")
    np.testing.assert_allclose(xy, mapping.xy)
    np.testing.assert_allclose(d, 6.5, atol=1e-8)
    assert np.all(np.isfinite(d))


@pytest.mark.parametrize("theta,family", [(3.89, "R"), (56.11, "H")])
def test_rigid_twisted_fractions(monolayer, theta, family):
    atoms, strain, _ = HSGenerator(
        monolayer, monolayer, tolerance=1e-6, angle_tol=0.01, max_atoms=6000,
    ).gen_supercell(theta, super_z=30, delta_z=6.5)
    assert len(atoms) == 1302
    assert strain < 1e-9
    nearest = stacking_map(atoms, "Mo", "S")
    assert nearest.family == family
    np.testing.assert_allclose(list(nearest.fractions.values()), 1 / 3, atol=0.01)
    disc = stacking_map(atoms, "Mo", "S", boundary=0.15 * np.linalg.norm(monolayer.cell[0]))
    assert disc.family == family
    for key, fraction in disc.fractions.items():
        assert fraction == pytest.approx(0.75 if key == "boundary" else 0.0816, abs=0.02 if key == "boundary" else 0.01)
    assert sum(disc.fractions.values()) == pytest.approx(1)
    _, d = interlayer_distance_map(atoms, "Mo", "S")
    assert np.all(np.isfinite(d))
    np.testing.assert_allclose(d, 6.5, atol=1e-8)


@pytest.mark.parametrize("analyze", [stacking_map, interlayer_distance_map])
def test_invalid_layer_composition(monolayer, analyze):
    atoms = aligned_bilayer(monolayer, "R", np.zeros(2))
    del atoms[1]
    with pytest.raises(ValueError, match="MX2"):
        analyze(atoms, "Mo", "S")


def test_interpolated_corrugation_and_atom_order(monolayer):
    atoms = aligned_bilayer(monolayer, "R", np.zeros(2))
    metal = np.flatnonzero(atoms.numbers == 42)
    bottom, top = np.split(metal, 2)
    corrugation = np.linspace(-0.15, 0.15, len(bottom))
    atoms.positions[bottom, 2] += corrugation
    atoms.positions[top, 2] += corrugation
    _, d = interlayer_distance_map(atoms, "Mo", "S")
    np.testing.assert_allclose(d, 6.5, atol=1e-8)
    shuffled = atoms[np.random.default_rng(4).permutation(len(atoms))]
    xy, d = interlayer_distance_map(shuffled, "Mo", "S")
    top = np.flatnonzero((shuffled.numbers == 42) &
                        (shuffled.positions[:, 2] > monolayer.positions[0, 2] + 3.25))
    expected = (shuffled.positions[top, :2] @ np.linalg.inv(shuffled.cell[:2, :2]) % 1) @ shuffled.cell[:2, :2]
    np.testing.assert_allclose(xy, expected)
    np.testing.assert_allclose(d, 6.5, atol=1e-8)
