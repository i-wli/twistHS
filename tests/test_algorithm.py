from pathlib import Path

import pytest
import numpy as np
import ase.io
from ase import Atoms
from twistHS.lib.algorithm import HSGenerator
from twistHS.lib.func import Pyatoms, check_vectors, get_supercell_atoms_pos, lattice_points_in_supercell

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
    
    twistSupercell = HSGenerator(bottom, top, method='first-order')
    result, delta, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    assert isinstance(result, Atoms)
    assert isinstance(delta, float)
    assert delta >= 0
    assert len(result) == 524
    assert result.cell[2][2] == pytest.approx(super_z, rel=1e-5)


@pytest.mark.parametrize('materials', ['graphene/hBN', 'graphene/MoS2'])
def test_gen_supercell_matches_legacy_atom_set(test_structures, materials, monkeypatch):
    if materials == 'graphene/hBN':
        bottom, top = test_structures
    else:
        root = Path(__file__).parent
        bottom = ase.io.read(root / 'graphene.xsf')
        top = ase.io.read(root / 'MoS2.xsf')
    generator = HSGenerator(bottom, top, method='first-order')
    actual, _, _ = generator.gen_supercell(5, 30, 7)

    def legacy_coincidence(self, common, cell, transform, unit):
        matrix = np.eye(3, dtype=int)
        matrix[:2, :2] = transform
        points = lattice_points_in_supercell(matrix) @ cell
        structure = Atoms(cell=cell, pbc=unit.pbc)
        for point in points:
            shifted = unit.copy()
            shifted.positions += point
            structure.extend(shifted)
        positions = (structure.positions @ np.linalg.inv(cell)) @ common
        return Pyatoms(common, positions, structure.numbers)

    monkeypatch.setattr(HSGenerator, '_coincidence', legacy_coincidence)
    expected, _, _ = generator.gen_supercell(5, 30, 7)

    def sorted_atoms(structure):
        atoms = np.column_stack((structure.numbers, structure.positions))
        return atoms[np.lexsort(atoms.round(9).T[::-1])]

    np.testing.assert_allclose(sorted_atoms(actual), sorted_atoms(expected), rtol=0, atol=1e-10)


def test_supercell_builder_tiles_per_atom_arrays():
    unit = Atoms('BN', positions=[[0, 0, 0], [1, 1, 0]], cell=[3, 4, 15],
                 pbc=[True, True, False], tags=[2, 3])
    unit.new_array('vectors', np.arange(12).reshape(2, 2, 3))
    result = get_supercell_atoms_pos(np.diag([6, 4, 15]), np.diag([2, 1]), unit)
    np.testing.assert_array_equal(result.numbers, np.tile(unit.numbers, 2))
    np.testing.assert_array_equal(result.positions, [[0, 0, 0], [1, 1, 0], [3, 0, 0], [4, 1, 0]])
    np.testing.assert_array_equal(result.pbc, unit.pbc)
    np.testing.assert_array_equal(result.get_tags(), [2, 3, 2, 3])
    np.testing.assert_array_equal(result.arrays['vectors'], np.tile(unit.arrays['vectors'], (2, 1, 1)))


def test_gen_supercell_layer_separation(test_structures):
    """Test if layers are properly separated by delta_z"""
    bottom, top = test_structures
    theta = 0.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top, method='first-order')
    result, _, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    # Get z-coordinates
    z_coords = result.positions[:, 2]
    unique_z = np.unique(z_coords)
    
    assert len(unique_z) == 2  # Should have exactly 2 distinct z-levels
    assert np.abs(unique_z[1] - unique_z[0]) == pytest.approx(delta_z, rel=1e-5)


@pytest.mark.parametrize('method,delta_z', [('search', 7), ('first-order', -7)])
def test_midplane_separation_with_different_input_origins(method, delta_z):
    root = Path(__file__).parent
    bottom = ase.io.read(root / 'graphene.xsf')
    top = ase.io.read(root / 'MoS2.xsf')
    result, _, _ = HSGenerator(bottom, top, method=method).gen_supercell(5, 30, delta_z)
    is_bottom = result.numbers == bottom.numbers[0]
    bottom_z = result.positions[is_bottom, 2]
    top_z = result.positions[~is_bottom, 2]
    top_mid = (top_z.min() + top_z.max()) / 2
    np.testing.assert_allclose(bottom_z, bottom.positions[0, 2], rtol=0, atol=1e-12)
    assert top_mid - bottom_z[0] == pytest.approx(delta_z)
    assert np.ptp(top_z) == pytest.approx(np.ptp(top.positions[:, 2]))


def test_check_vectors_requires_xy_layer_plane():
    atoms = Atoms('C', cell=[[0, 3, 0], [0, 0, 4], [15, 0, 0]])
    with pytest.raises(ValueError, match='xy plane'):
        check_vectors(atoms)
    atoms.cell = [[3, 0, 1e-6], [0, 4, -1e-6], [1e-6, -1e-6, 15]]
    check_vectors(atoms)
    for row, col in [(0, 2), (1, 2), (2, 0), (2, 1)]:
        tilted = atoms.copy()
        tilted.cell[row, col] = np.nextafter(tilted.cell[row, col], 2 * tilted.cell[row, col])
        with pytest.raises(ValueError, match='xy plane'):
            check_vectors(tilted)


def test_gen_supercell_rotation(test_structures):
    """Test if supercell is properly generated"""
    bottom, top = test_structures
    theta = 3.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top, method='first-order')
    result, _, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)
    
    # Number of atoms should increase - superlattice is generated
    assert len(result) > len(bottom) + len(top)

def test_gen_supercell_different_angles(test_structures):
    """Equivalent 60° and 120° graphene bases give the same approximant."""
    bottom, top = test_structures
    reference, reference_strain, _ = HSGenerator(bottom, top, method='first-order').gen_supercell(5, 30, 3.4)
    equivalent = top.copy()
    equivalent.cell[1] += equivalent.cell[0]
    assert equivalent.cell.cellpar()[5] == pytest.approx(60)
    result, strain, _ = HSGenerator(bottom, equivalent, method='first-order').gen_supercell(5, 30, 3.4)
    assert len(result) == len(reference)
    assert strain == pytest.approx(reference_strain, abs=1e-12)
    np.testing.assert_allclose(result.cell, reference.cell, atol=1e-12)

def test_gen_supercell_zero_angle(test_structures):
    """Test behavior with zero rotation angle"""
    bottom, top = test_structures
    theta = 0.0
    super_z = 15.0
    delta_z = 3.4
    
    twistSupercell = HSGenerator(bottom, top, method='first-order')
    result, _, _ = twistSupercell.gen_supercell(theta, super_z, delta_z)

    assert result.cell.cellpar()[3] == pytest.approx(bottom.cell.cellpar()[3], rel=1e-5)

@pytest.mark.parametrize('i,count', [(1, 28), (31, 11908)])
def test_graphene_exact_commensurate(test_structures, i, count):
    bottom, _ = test_structures
    theta = np.degrees(np.arccos(1 - 1 / (2 * (3 * i**2 + 3 * i + 1))))
    generator = HSGenerator(bottom, bottom, tolerance=1e-6, angle_tol=1e-6, max_atoms=count)
    result, strain, theta_eff = generator.gen_supercell(theta, 30, 3.4)
    assert len(result) == count
    assert strain < 1e-12
    assert theta_eff == pytest.approx(theta, abs=1e-10)
    assert np.all(np.isfinite(result.positions))


@pytest.mark.parametrize('theta', [0, 5, 12])
def test_atom_count_from_determinants(test_structures, theta):
    from twistHS.lib.moire import supercell_matrices
    bottom, top = test_structures
    rotated = top.copy()
    rotated.rotate(theta, 'z', rotate_cell=True)
    M_b, M_t = supercell_matrices(bottom.cell[:2, :2], rotated.cell[:2, :2])
    result, _, _ = HSGenerator(bottom, top, method='first-order').gen_supercell(theta, 30, 3.4)
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
    reference, expected_strain, _ = HSGenerator(bottom, top, method='first-order').gen_supercell(5, 30, 3.4)
    bottom, top = bottom.copy(), top.copy()
    bottom.cell[:2, :2] = U @ bottom.cell[:2, :2]
    top.cell[:2, :2] = V @ top.cell[:2, :2]
    result, strain, _ = HSGenerator(bottom, top, method='first-order').gen_supercell(5, 30, 3.4)
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
    _, strain, _ = HSGenerator(bottom, top, method='first-order').gen_supercell(0, 30, 3.4)
    assert strain == pytest.approx(8.94454382826e-4)


def test_zero_twist_homobilayer_raises(test_structures):
    bottom, _ = test_structures
    with pytest.raises(ValueError, match='singular'):
        HSGenerator(bottom, bottom, method='first-order').gen_supercell(0, 30, 3.4)


def test_scan_filters_inclusive_tolerance(test_structures):
    generator = HSGenerator(*test_structures, method='first-order')
    angles = np.arange(1, 7)
    strains = [generator.gen_supercell(a, 30, 3.4)[1] for a in angles]
    tolerance = sorted(strains)[2] * 100
    results = generator.scan(angles, 30, 3.4, tolerance)
    assert [angle for angle, _, _ in results] == [angle for angle, strain in zip(angles, strains) if strain * 100 <= tolerance]
    assert len(results) == 3
    assert generator.scan(angles, 30, 3.4, 0) == []


@pytest.mark.parametrize('theta', [0, 1, 2, 5, 12, 21, 29])
def test_first_order_v1_atom_count(test_structures, theta):
    bottom, top = test_structures
    alpha = np.radians(theta)
    omega = np.radians(120)
    p = 2.504 / 2.46
    # Historical Hermann construction for the equally scaled hexagonal axes.
    rotated = p / np.sin(omega) * np.array([
        [np.sin(omega - alpha), np.sin(alpha)],
        [-np.sin(alpha), np.sin(omega + alpha)],
    ])
    v1_bottom = (rotated - p**2 * np.eye(2)) / (1 + p**2 - 2 * p * np.cos(alpha))
    v1_top = np.linalg.solve(rotated, v1_bottom)
    expected = sum(len(layer) * round(abs(np.linalg.det(np.rint(matrix))))
                   for layer, matrix in [(bottom, v1_bottom), (top, v1_top)])
    result, _, _ = HSGenerator(bottom, top, method="first-order", max_atoms=1,
                               angle_tol=0, tolerance=0).gen_supercell(theta, 30, 7)
    assert len(result) == expected


def test_search_scan_effective_angle_and_no_match(test_structures):
    bottom, _ = test_structures
    generator = HSGenerator(bottom, bottom, max_atoms=28)
    entries = generator.scan([0.05, 21.84, 5], 30, 7)
    requested, structure, strain = entries[0]
    assert len(entries) == 1
    assert requested == 21.84
    assert structure.info["theta_eff"] == pytest.approx(21.7867893, abs=1e-7)
    assert len(structure) == 28
    assert 0 < strain < 1e-6
    with pytest.raises(ValueError, match='No commensurate supercell at a requested twist angle of 5'):
        generator.gen_supercell(5, 30, 7)
    with pytest.raises(ValueError, match='theta'):
        generator.scan([0, np.nan], 30, 7)


def test_components_use_rotated_top_without_supercell(test_structures):
    from twistHS.lib.moire import moire_components
    bottom, top = test_structures
    rotated = top.copy()
    rotated.rotate(5, 'z', rotate_cell=True)
    expected = moire_components(bottom.cell[:2, :2], rotated.cell[:2, :2], 2)
    actual = HSGenerator(bottom, top, method='first-order').moire_components(5, 2)
    np.testing.assert_allclose([c.q for c in actual], [c.q for c in expected])
    assert [c.multiplicity for c in actual] == [c.multiplicity for c in expected]


def test_graphene_mos2_scan():
    bottom = ase.io.read(Path(__file__).parent / 'graphene.xsf')
    top = ase.io.read(Path(__file__).parent / 'MoS2.xsf')
    results = HSGenerator(bottom, top, method='first-order').scan(np.arange(10), 30, 7, 0.5)
    assert [angle for angle, _, _ in results] == list(range(2, 9))


if __name__ == "__main__":
    pytest.main(["-v", "test_algorithm.py"])
