import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from twistHS.lib.algorithm import HSGenerator, check_vectors

@pytest.fixture
def test_structures():
    """Create simple test structures for bottom and top layers"""
    a = 2.46  # graphene lattice constant
    bottom = Atoms('C2',
                  positions=[[0, 0, 0],
                           [a/2, a*np.sqrt(3)/2, 0]],
                  cell=[[a, 0, 0],
                        [-a/2, a*np.sqrt(3)/2, 0],
                        [0, 0, 15]])
    
    a = 2.504 # hBN
    top = Atoms(['B', 'N'],
                  positions=[[0, 0, 0],
                           [a/2, a*np.sqrt(3)/2, 0]],
                  cell=[[a, 0, 0],
                        [-a/2, a*np.sqrt(3)/2, 0],
                        [0, 0, 15]])
    return bottom, top

def test_gen_supercell_basic(test_structures):
    """Test basic functionality with valid inputs"""
    bottom, top = test_structures
    theta = 5.0  # degrees
    super_z = 15.0  # Å
    delta_z = 3.4  # Å
    
    twistSupercell = HSGenerator(bottom, top)
    result, delta = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    assert isinstance(result, Atoms)
    assert isinstance(delta, float)
    assert delta >= 0
    assert len(result) == 524
    assert result.cell[2][2] == pytest.approx(super_z, rel=1e-5)

def test_gen_supercell_layer_separation(test_structures):
    """Test if layers are properly separated by delta_z"""
    bottom, top = test_structures
    theta = 0.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top)
    result, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    # Get z-coordinates
    z_coords = result.positions[:, 2]
    unique_z = np.unique(z_coords)
    
    assert len(unique_z) == 2  # Should have exactly 2 distinct z-levels
    assert np.abs(unique_z[1] - unique_z[0]) == pytest.approx(delta_z, rel=1e-5)

def test_gen_supercell_rotation(test_structures):
    """Test if supercell is properly generated"""
    bottom, top = test_structures
    theta = 3.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top)
    result, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    # Number of atoms should increase - superlattice is generated
    assert len(result) > len(bottom) + len(top)

def test_gen_supercell_different_angles(test_structures):
    """Equivalent 60° and 120° graphene bases give the same approximant."""
    bottom, top = test_structures
    reference, reference_strain = HSGenerator(bottom, top).gen_supercell(5, 30, 3.4)
    equivalent = top.copy()
    equivalent.cell[1] += equivalent.cell[0]
    assert equivalent.cell.cellpar()[5] == pytest.approx(60)
    result, strain = HSGenerator(bottom, equivalent).gen_supercell(5, 30, 3.4)
    assert len(result) == len(reference)
    assert strain == pytest.approx(reference_strain, abs=1e-12)
    np.testing.assert_allclose(result.cell, reference.cell, atol=1e-12)

def test_gen_supercell_zero_angle(test_structures):
    """Test behavior with zero rotation angle"""
    bottom, top = test_structures
    theta = 0.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top)
    result, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)

    assert result.cell.cellpar()[3] == pytest.approx(bottom.cell.cellpar()[3], rel=1e-5)

@pytest.mark.parametrize('theta', [np.degrees(np.arccos(13 / 14)), 60 - np.degrees(np.arccos(13 / 14))])
def test_graphene_exact_commensurate(test_structures, theta):
    bottom, _ = test_structures
    result, strain = HSGenerator(bottom, bottom).gen_supercell(theta, 30, 3.4)
    assert len(result) == 28
    assert strain < 1e-9


@pytest.mark.parametrize('theta', [0, 5, 12])
def test_atom_count_from_determinants(test_structures, theta):
    from twistHS.lib.moire import supercell_matrices
    bottom, top = test_structures
    rotated = top.copy()
    rotated.rotate(theta, 'z', rotate_cell=True)
    M_b, M_t = supercell_matrices(bottom.cell[:2, :2], rotated.cell[:2, :2])
    result, _ = HSGenerator(bottom, top).gen_supercell(theta, 30, 3.4)
    expected = round(abs(np.linalg.det(M_b))) * len(bottom) + round(abs(np.linalg.det(M_t))) * len(top)
    assert len(result) == expected
    assert np.all(np.isfinite(result.positions))
    assert result.cell.volume > 0


@pytest.mark.parametrize('U,V', [
    ([[1, 5], [0, 1]], [[2, 3], [1, 2]]),
    ([[0, 1], [-1, 0]], [[1, 0], [1, 1]]),
    ([[0, 1], [1, 0]], [[1, 0], [0, 1]]),
])
def test_supercell_basis_invariance(test_structures, U, V):
    bottom, top = test_structures
    reference, expected_strain = HSGenerator(bottom, top).gen_supercell(5, 30, 3.4)
    bottom, top = bottom.copy(), top.copy()
    bottom.cell[:2, :2] = U @ bottom.cell[:2, :2]
    top.cell[:2, :2] = V @ top.cell[:2, :2]
    result, strain = HSGenerator(bottom, top).gen_supercell(5, 30, 3.4)
    assert len(result) == len(reference)
    assert strain == pytest.approx(expected_strain, abs=1e-12)
    assert result.cell.volume == pytest.approx(reference.cell.volume)
    # Positions describe the same atoms modulo the reference supercell.
    frac = result.positions @ np.linalg.inv(reference.cell)
    expected_frac = reference.positions @ np.linalg.inv(reference.cell)
    offsets = frac[:, None, :] - expected_frac[None, :, :]
    offsets -= np.rint(offsets)
    assert np.max(np.min(np.linalg.norm(offsets, axis=2), axis=1)) < 1e-10


def test_rectangular_anisotropic_strain():
    bottom = Atoms('C', positions=[[0, 0, 0]], cell=[3, 4, 15])
    top = Atoms('C', positions=[[0, 0, 0]], cell=[3.1, 4.3, 15])
    _, strain = HSGenerator(bottom, top).gen_supercell(0, 30, 3.4)
    assert strain == pytest.approx(8.94454382826e-4)


def test_zero_twist_homobilayer_raises(test_structures):
    bottom, _ = test_structures
    with pytest.raises(ValueError, match='singular'):
        HSGenerator(bottom, bottom).gen_supercell(0, 30, 3.4)


def test_scan_filters_inclusive_tolerance(test_structures):
    generator = HSGenerator(*test_structures)
    angles = np.arange(1, 7)
    strains = [generator.gen_supercell(a, 30, 3.4)[1] for a in angles]
    tolerance = sorted(strains)[2] * 100
    results = generator.scan(angles, 30, 3.4, tolerance)
    assert [angle for angle, _, _ in results] == [angle for angle, strain in zip(angles, strains) if strain * 100 <= tolerance]
    assert len(results) == 3
    assert generator.scan(angles, 30, 3.4, 0) == []


def test_components_use_rotated_top_without_supercell(test_structures):
    from twistHS.lib.moire import moire_components
    bottom, top = test_structures
    rotated = top.copy()
    rotated.rotate(5, 'z', rotate_cell=True)
    expected = moire_components(bottom.cell[:2, :2], rotated.cell[:2, :2], 2)
    actual = HSGenerator(bottom, top).moire_components(5, 2)
    np.testing.assert_allclose([c.q for c in actual], [c.q for c in expected])
    assert [c.multiplicity for c in actual] == [c.multiplicity for c in expected]


def test_graphene_mos2_scan():
    import ase.io
    bottom = ase.io.read(Path(__file__).parent / 'graphene.xsf')
    top = ase.io.read(Path(__file__).parent / 'MoS2.xsf')
    results = HSGenerator(bottom, top).scan(np.arange(10), 30, 7, 0.5)
    assert [angle for angle, _, _ in results] == list(range(2, 9))


if __name__ == "__main__":
    pytest.main(["-v", "test_algorithm.py"])
