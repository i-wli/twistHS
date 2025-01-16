import numpy as np
from ase.atoms import Atoms
from dataclasses import dataclass

DEFAULT_PRECISION = 1e-6
DEFAULT_PBC = [True, True, True]

@dataclass(frozen=True)
class Pyatoms:
    vectors: np.ndarray
    positions: np.ndarray
    atomic_numbers: np.ndarray

    def __post_init__(self):
        """Validate the data structure."""
        if self.vectors.shape != (3, 3):
            raise ValueError("[err] Cell vectors must be 3x3 array")
        if self.positions.shape[1] != 3:
            raise ValueError("[err] Positions must be Nx3 array")
        if len(self.atomic_numbers) != len(self.positions):
            raise ValueError("[err] Number of atoms mismatch")

def calc_vector_norm(vector: np.ndarray) -> float:
    """Calculate the norm of a vector"""
    return np.linalg.norm(vector)

def float_eq(f1: float, f2: float, prec: float = DEFAULT_PRECISION) -> bool:
    """Float equal"""
    return np.isclose(f1, f2, atol=prec)

def ase_atoms_to_py_atoms(atoms: Atoms) -> Pyatoms:
    """Convert ASE Atoms object to Pyatoms."""
    try:
        return Pyatoms(
            vectors=atoms.cell.copy(),
            positions=atoms.positions.copy(),
            atomic_numbers=atoms.numbers.copy()
        )
    except AttributeError as e:
        raise ValueError(f"Invalid ASE Atoms object: {str(e)}")

def py_atoms_to_ase_atoms(atoms: Pyatoms) -> Atoms:
    """Convert Pyatoms to ASE Atoms"""
    return Atoms(
        numbers=atoms.atomic_numbers,
        positions=atoms.positions,
        cell=atoms.vectors,
        pbc=DEFAULT_PBC
    )    
    
def check_vectors(atoms: Atoms):
    cell = atoms.cell
    if not (np.all(np.abs(cell[:2, 2]) <= DEFAULT_PRECISION) and
            np.all(np.abs(cell[2, :2]) <= DEFAULT_PRECISION)):
        raise ValueError('[err] Input structures: a and b must lie in the xy plane and c along z')

def coord_cart2frac(cell_vecs: np.ndarray, cart_coords: np.ndarray) -> np.ndarray:
    """Transfrom the cart coords to frac coords"""
    cell_vecs_inv = np.linalg.inv(cell_vecs)
    frac_vec = cart_coords @ cell_vecs_inv
    return frac_vec

def coord_frac2cart(cell_vecs: np.ndarray, frac_coords: np.ndarray) -> np.ndarray:
    """Transfrom the frac coords to cart coords"""
    return frac_coords @ cell_vecs

def get_supercell_vecs(trans_2D: np.ndarray, unit_vecs: np.ndarray, super_z: float) -> np.ndarray:
    """Get the supercell lattice"""
    if super_z <= 0:
        raise ValueError("[err] Supercell z-dimension must be positive")
    
    super_2d = trans_2D @ unit_vecs[0:2,0:2]
    supercell_vecs = np.array([[super_2d[0,0], super_2d[0,1], 0],
                               [super_2d[1,0], super_2d[1,1], 0],
                               [0, 0, super_z]])
    return supercell_vecs

def lattice_points_in_supercell(supercell_matrix: np.ndarray) -> np.ndarray:
    """Find all lattice points contained in a supercell."""
    d_points = np.indices((2, 2, 2)).reshape(3, -1).T @ supercell_matrix
    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    grid_ranges = [slice(mins[dim], maxes[dim]) for dim in range(3)]
    all_points = np.mgrid[grid_ranges[0], grid_ranges[1], grid_ranges[2]].reshape(3, -1).T

    frac_points = all_points @ np.linalg.inv(supercell_matrix)

    valid_mask = (frac_points < 1 - 1e-10) & (frac_points >= -1e-10)
    tvects = frac_points[np.all(valid_mask, axis=1)]

    # Ensuring the number of points matches the expected value from the determinant
    expected_points = round(abs(np.linalg.det(supercell_matrix)))
    if len(tvects) != expected_points:
        raise ValueError(f"Found {len(tvects)} points, expected {expected_points}")

    return tvects

def get_supercell_atoms_pos(supercell_vecs: np.ndarray,
                            trans_2D: np.ndarray,
                            unit: Atoms) -> Atoms:
    trans_matrix = np.array([[trans_2D[0,0], trans_2D[0,1], 0],
                               [trans_2D[1,0], trans_2D[1,1], 0],
                               [0, 0, 1]])
    lattice_points = lattice_points_in_supercell(trans_matrix) @ supercell_vecs
    positions = (lattice_points[:, None, :] + unit.positions[None, :, :]).reshape(-1, 3)
    superatoms = Atoms(
        numbers=np.tile(unit.numbers, len(lattice_points)),
        positions=positions,
        cell=supercell_vecs,
        pbc=unit.pbc,
    )
    for name, values in unit.arrays.items():
        if name not in superatoms.arrays:
            repeats = (len(lattice_points),) + (1,) * (values.ndim - 1)
            superatoms.new_array(name, np.tile(values, repeats))
    return superatoms
