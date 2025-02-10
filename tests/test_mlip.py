import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from twistHS.mlip.calculators import finetuned_calculator, zero_shot_calculator
from twistHS.mlip.finetune import validate_dataset
from twistHS.mlip.relax import InPlaneIsotropicCellFilter, relax


@pytest.fixture
def argon():
    atoms = bulk("Ar", "fcc", a=5.6, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, rng=np.random.default_rng(0))
    return atoms


def lj():
    return LennardJones(sigma=3.4, epsilon=0.0104, rc=8.5, smooth=True)


@pytest.mark.parametrize("shear", [0, 0.15])
def test_cell_relaxation_and_trajectory(argon, tmp_path, shear):
    skewed = argon.cell.copy()
    skewed[1, 0] = shear * skewed[0, 0]
    argon.set_cell(skewed, scale_atoms=True)
    positions, cell = argon.positions.copy(), argon.cell.copy()
    path = tmp_path / "r.traj"
    relaxed = relax(argon, lj(), steps=2000, trajectory=path)
    np.testing.assert_allclose(relaxed.cell[2], cell[2])
    np.testing.assert_array_equal(relaxed.cell[:2, 2], 0)
    assert not np.allclose(relaxed.cell[:2], cell[:2])
    assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 1e-3
    assert relaxed.calc is not None
    np.testing.assert_array_equal(argon.positions, positions)
    np.testing.assert_array_equal(argon.cell, cell)
    assert argon.calc is None
    assert path.exists()
    for frame in read(path, index=":"):
        deformation = np.linalg.solve(cell, frame.cell)
        scale = deformation[0, 0]
        np.testing.assert_allclose(deformation, np.diag([scale, scale, 1]), atol=1e-12)
    last = read(path, index=-1)
    np.testing.assert_allclose(last.positions, relaxed.positions)
    np.testing.assert_allclose(last.cell, relaxed.cell)


def test_in_plane_force_projection():
    atoms = Atoms("Ar", cell=[[4, 0, 0], [-2, 3, 0], [0, 0, 20]], pbc=True)
    atomic_forces = np.array([[0.2, -0.3, 0.4]])
    projected = []
    for zz in (7.0, 70.0):
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=atomic_forces,
            stress=np.array([2.0, 4.0, zz, 0.5, 0.6, 0.7]),
        )
        forces = InPlaneIsotropicCellFilter(atoms).get_forces()
        np.testing.assert_array_equal(forces[:len(atoms)], atomic_forces)
        expected = np.diag([-3 * atoms.get_volume(), -3 * atoms.get_volume(), 0])
        np.testing.assert_allclose(forces[len(atoms):], expected)
        projected.append(forces)
    np.testing.assert_array_equal(projected[0], projected[1])


def test_fixed_cell_relaxation(argon):
    relaxed = relax(argon, lj(), steps=2000, cell=False)
    np.testing.assert_array_equal(relaxed.cell, argon.cell)
    assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 1e-3


def test_nonconvergence(argon):
    with pytest.raises(RuntimeError, match="in 3 steps"):
        relax(argon, lj(), steps=3)


@pytest.mark.parametrize("missing", [None, "REF_energy", "REF_forces", "REF_stress"])
def test_reference_labels(tmp_path, missing):
    atoms = Atoms("MoS2", positions=[[0, 0, 0], [1, 0, 1], [1, 0, -1]], cell=[4, 4, 30])
    atoms.info.update(REF_energy=-5.0, REF_stress=np.eye(3) * 0.01)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    path = tmp_path / "labels.extxyz"
    frames = [atoms.copy(), atoms.copy()]
    if missing:
        del (frames[1].arrays if missing == "REF_forces" else frames[1].info)[missing]
    write(path, frames, format="extxyz")
    if missing:
        with pytest.raises(ValueError, match=rf"labels.extxyz: frame 1 missing {missing}"):
            validate_dataset([path])
    else:
        validate_dataset([path])
        result = read(path)
        np.testing.assert_array_equal(result.info["REF_stress"], atoms.info["REF_stress"])
        np.testing.assert_array_equal(result.arrays["REF_forces"], atoms.arrays["REF_forces"])
        assert result.info["REF_energy"] == atoms.info["REF_energy"]


def test_device_validation_before_mace_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "mace", None)
    with pytest.raises(ValueError, match="device"):
        zero_shot_calculator("mps")
    with pytest.raises(ValueError, match="device"):
        finetuned_calculator("x.model", "mps")
