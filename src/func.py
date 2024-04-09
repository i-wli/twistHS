import sys
import numpy as np
import ase.io
from ase.atoms import Atoms
from dataclasses import dataclass

@dataclass
class Pyatoms:
    vectors: np.ndarray
    positions: np.ndarray
    atomic_numbers: np.ndarray

@dataclass
class theta:
    radian: float
    degree: float

def calc_vectors_dis(v1):
    """calculate the angle between two vectors"""
    return np.linalg.norm(v1)

def float_eq(f1, f2, prec=1e-6):
    """float equal"""
    return np.isclose(f1, f2, atol=prec)

def exit(context='[error] Unknown: Something goes wrong...'):
    """exit the program with some error msg."""
    sys.exit(context)
    
def ase_atoms_to_py_atoms(atoms: ase.atoms.Atoms) -> Pyatoms:
    vectors = np.array(atoms.cell.copy())
    positions = atoms.positions
    atomic_numbers = atoms.numbers
    atoms = Pyatoms(vectors, positions, atomic_numbers)
    return atoms

def py_atoms_to_ase_atoms(atoms: Pyatoms) -> ase.atoms.Atoms:
    vectors = [[j for j in k] for k in atoms.vectors]
    positions = [[j for j in k] for k in atoms.positions]
    numbers = [i for i in atoms.atomic_numbers]
    atoms = Atoms(
        numbers=numbers, positions=positions, cell=vectors, pbc=[True, True, True]
    )
    return atoms    
    
def check_vectors(atoms: ase.atoms.Atoms):
    len_ang = atoms.cell.cellpar()
    if not (float_eq(len_ang[3], 90.0) and
            float_eq(len_ang[4], 90.0)):
        exit('[error] Input structures: c axis must be in z direction')

def coord_cart2frac(cell_vecs, cart_vec):
    """Transfrom the cart coords to frac coords"""
    cell_vecs_inv = np.linalg.inv(cell_vecs)
    frac_vec = cart_vec @ cell_vecs_inv
    return frac_vec

def coord_frac2cart(cell_vecs, frac_vec):
    """Transfrom the frac coords to cart coords"""
    return frac_vec @ cell_vecs

def get_supercell_vecs(trans_2D, unit_vecs, super_z):
    """Get the supercell lattice"""
    super_2d = np.dot(trans_2D, unit_vecs[0:2,0:2])
    supercell_vecs = np.array([[super_2d[0,0], super_2d[0,1], 0],
                               [super_2d[1,0], super_2d[1,1], 0],
                               [0, 0, super_z]])
    return supercell_vecs

def lattice_points_in_supercell(supercell_matrix):
    """find all lattice points contained in a supercell."""
    d_points = np.dot(np.indices((2, 2, 2)).reshape(3, -1).T, supercell_matrix)
    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    grid_ranges = [slice(mins[dim], maxes[dim]) for dim in range(3)]
    all_points = np.mgrid[grid_ranges[0], grid_ranges[1], grid_ranges[2]].reshape(3, -1).T

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    valid_mask = (frac_points < 1 - 1e-10) & (frac_points >= -1e-10)
    tvects = frac_points[np.all(valid_mask, axis=1)]

    # Ensuring the number of points matches the expected value from the determinant
    expected_points = round(abs(np.linalg.det(supercell_matrix)))
    assert len(tvects) == expected_points, "Mismatch in expected and actual points count"

    return tvects

def get_supercell_atoms_pos(supercell_vecs: "np.array",
                            trans_2D: "np.array",
                            unit: ase.atoms.Atoms) -> ase.atoms.Atoms:
    trans_matrix = np.array([[trans_2D[0,0], trans_2D[0,1], 0],
                               [trans_2D[1,0], trans_2D[1,1], 0],
                               [0, 0, 1]])
    lattice_points_frac = lattice_points_in_supercell(trans_matrix)
    lattice_points = lattice_points_frac @ supercell_vecs
    superatoms = Atoms(cell=supercell_vecs, pbc=unit.pbc)
    for lp in lattice_points:
        shifted_atoms = unit.copy()
        shifted_atoms.positions += lp
        superatoms.extend(shifted_atoms)
    return superatoms