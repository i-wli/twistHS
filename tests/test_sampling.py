from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from twistHS.lib.analysis import stacking_map
from twistHS.lib.sampling import high_symmetry_bilayers, sample_aligned_bilayers, sample_monolayers


@pytest.fixture
def monolayer():
    return read(Path(__file__).parent / "MoS2.xsf")


def sample(unit, n=4, seed=12, stacking="R", d_range=(6.0, 7.0), **kwargs):
    return sample_aligned_bilayers(unit, n, seed, stacking, d_range, super_z=30, **kwargs)


def test_reproducibility_and_cells(monolayer):
    original = monolayer.copy()
    first, second, other = sample(monolayer), sample(monolayer), sample(monolayer, seed=13)
    assert len(first) == 4
    for a, b, c in zip(first, second, other):
        np.testing.assert_array_equal(a.positions, b.positions)
        assert not np.array_equal(a.positions, c.positions)
        assert len(a) == 54
        assert a.get_chemical_formula() == "Mo18S36"
        np.testing.assert_allclose(a.cell.lengths(), [3 * 3.11754, 3 * 3.11754, 30], atol=1e-5)
        assert np.all(a.pbc)
    np.testing.assert_array_equal(monolayer.positions, original.positions)
    np.testing.assert_array_equal(monolayer.cell, original.cell)


def test_separation_distribution(monolayer):
    frames = sample(monolayer, n=200, sigma_atom=0)
    separations = [a.positions[27::3, 2].mean() - a.positions[:27:3, 2].mean() for a in frames]
    assert all(6.0 <= d <= 7.0 for d in separations)
    assert np.mean(separations) == pytest.approx(6.5, abs=0.08)
    assert np.std(separations) == pytest.approx(0.289, abs=0.04)


def test_mass_scaled_displacements(monolayer):
    frames = sample(monolayer, n=200, d_range=(6.5, 6.5), sigma_atom=0.5)
    residuals = {"Mo": [], "S": []}
    for atoms in frames:
        for layer in (atoms[:27], atoms[27:]):
            for symbol in residuals:
                # Center each of the two S planes separately.
                for plane in ([0] if symbol == "Mo" else [1, 2]):
                    z = layer.positions[plane::3, 2]
                    residuals[symbol].extend(z - z.mean())
    for symbol, mass in (("Mo", 95.95), ("S", 32.06)):
        assert np.std(residuals[symbol]) == pytest.approx(0.5 / np.sqrt(mass), rel=0.15)


@pytest.mark.parametrize("family", ["R", "H"])
def test_sample_family(monolayer, family):
    for atoms in sample(monolayer, stacking=family, d_range=(6.5, 6.5), sigma_atom=0):
        assert stacking_map(atoms, "Mo", "S").family == family


def test_principal_strains(monolayer):
    cell0 = sample(monolayer, n=1)[0].cell
    frames = sample(monolayer, n=200, d_range=(6.5, 6.5), strain=0.02, sigma_atom=0)
    principal_strains = []
    for atoms in frames:
        deformation = np.linalg.solve(cell0, atoms.cell)[:2, :2]
        np.testing.assert_allclose(deformation, deformation.T, atol=1e-12, rtol=0)
        principal_strains.append(np.linalg.eigvalsh(deformation) - 1)
        np.testing.assert_array_equal(atoms.cell[2], cell0[2])
        d = atoms.positions[27::3, 2].mean() - atoms.positions[:27:3, 2].mean()
        assert d == pytest.approx(6.5)
    principal_strains = np.array(principal_strains)
    assert np.all(np.abs(principal_strains) <= 0.02)
    assert not np.allclose(principal_strains, principal_strains[0])
    assert np.any(np.prod(principal_strains, axis=1) < 0)


def test_sample_monolayers(monolayer):
    original = monolayer.copy()
    first = sample_monolayers(monolayer, 4, 12, 30, strain=0.02)
    second = sample_monolayers(monolayer, 4, 12, 30, strain=0.02)
    assert len(first) == len(second) == 4
    for atoms, repeated in zip(first, second):
        assert len(atoms) == 27
        assert atoms.get_chemical_formula() == "Mo9S18"
        np.testing.assert_array_equal(atoms.cell[2], [0, 0, 30])
        np.testing.assert_array_equal(atoms.positions, repeated.positions)
        np.testing.assert_array_equal(atoms.cell, repeated.cell)
    np.testing.assert_array_equal(monolayer.positions, original.positions)
    np.testing.assert_array_equal(monolayer.cell, original.cell)


def test_high_symmetry_bilayers(monolayer):
    stackings = high_symmetry_bilayers(monolayer, 6.5, 30)
    assert stackings.keys() == {"Rhh", "RhX", "RhM", "HhX", "Hhh", "HhM"}
    for atoms in stackings.values():
        assert len(atoms) == 6
        assert atoms[0].symbol == atoms[3].symbol == "Mo"
        assert atoms.positions[3, 2] - atoms.positions[0, 2] == pytest.approx(6.5)


def test_invalid_stacking(monolayer):
    with pytest.raises(ValueError, match="stacking"):
        sample(monolayer, stacking="X")
