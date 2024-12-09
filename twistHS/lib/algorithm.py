import argparse
import logging
import numpy as np
import ase.io
from typing import Tuple
from twistHS.lib.func import *

@dataclass
class LayerParameters:
    """Parameters for getting transformation matrix
    
    References:
        J. Phys.: Condens. Matter 24 314210 (2012)
    """
    q: float  # Ro1/Ro2
    p1: float  # R'o1/Ro1
    p2: float  # R'o2/Ro2
    omega: float  # lattice angle (radian)


class HSGenerator:
    """Generates twisted vdW heterostructure supercells."""
    
    def __init__(self, bottom: Atoms, top: Atoms):
        self.bottom = bottom
        self.top = top
        self.logger = logging.getLogger(__name__)
        self._validate_inputs()
        self.params = self._calculate_parameters()

    def _validate_inputs(self) -> None:
        """Validate input atomic structures."""
        if not isinstance(self.bottom, Atoms) or not isinstance(self.top, Atoms):
            raise ValueError("[err] Bottom and top must be ASE Atoms objects")
        check_vectors(self.bottom)
        check_vectors(self.top)

    def _calculate_parameters(self) -> LayerParameters:
        """Calculate basic parameters for transformation matrix."""
        bottom_cell = self.bottom.cell.cellpar()
        top_cell = self.top.cell.cellpar()

        return LayerParameters(
            q=bottom_cell[0] / bottom_cell[1],
            p1=top_cell[0] / bottom_cell[0],
            p2=top_cell[1] / bottom_cell[1],
            omega=np.radians(bottom_cell[5])
        )

    def _coincidence(self, vectors_super: np.ndarray, 
                    vectors_indiv: np.ndarray,
                    trans_2D: np.ndarray, 
                    unit: Atoms) -> Pyatoms:
        superatoms = get_supercell_atoms_pos(vectors_indiv, trans_2D, unit)
        superatoms_frac = coord_cart2frac(vectors_indiv, superatoms.positions)
        superatoms_pos = coord_frac2cart(vectors_super, superatoms_frac)
        return Pyatoms(vectors_super, superatoms_pos, superatoms.numbers)

    def _get_trans_matrix_2d(self, alpha_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get transformation matrix with given twist angle:
        alpha_value: twist angle (radian)"""

        param = self.params
        Delta = 1 + param.p1*param.p2 - (param.p1 + param.p2)*np.cos(alpha_value) + (param.p1 - param.p2)*np.cos(param.omega)*np.sin(alpha_value)

        # Define the matrix that maps Ro to Ro'
        transformation_matrix_R_prime = np.array([param.p1*np.array([np.sin(param.omega - alpha_value), param.q*np.sin(alpha_value)]),
                                                param.p2*np.array([-1/param.q*np.sin(alpha_value), np.sin(param.omega + alpha_value)])])
        transformation_matrix_R_prime = 1 / np.sin(param.omega) * transformation_matrix_R_prime

        # Define the matrix that maps Ro to RM
        transformation_matrix_RM = np.array([param.p1*np.array([(np.sin(param.omega - alpha_value) - param.p2*np.sin(param.omega)), param.q*np.sin(alpha_value)]),
                                            param.p2*np.array([-1/param.q*np.sin(alpha_value), (np.sin(param.omega + alpha_value) - param.p1*np.sin(param.omega))])])
        transformation_matrix_RM = 1 / (Delta * np.sin(param.omega)) * transformation_matrix_RM

        # The mapping matrix from Ro' to RM
        mapping_matrix_R_prime_to_RM = np.linalg.inv(transformation_matrix_R_prime).dot(transformation_matrix_RM)
        return transformation_matrix_RM, mapping_matrix_R_prime_to_RM


    def gen_supercell(self, theta: float, super_z: float, delta_z: float) -> Tuple[Atoms, float]:
        """Generate twisted supercell
        
        Args:
            theta: twist angle in degrees
            super_z: supercell constant in z direction
            delta_z: distance between two layers            
        
        Returns:
            (Atoms, float): supercell structure and mismatch ratio
        """
        self._validate_twist_parameters(theta)

        theta_rad = np.radians(theta)
        tran_2D_b, tran_2D_t = self._get_trans_matrix_2d(theta_rad)
        tran_2D_b = -np.round(tran_2D_b)
        tran_2D_t = -np.round(tran_2D_t)

        """Get supercell lattice"""
        vectors_b = get_supercell_vecs(tran_2D_b, self.bottom.cell, super_z)
        top_rot = self.top.copy()
        top_rot.rotate(theta, 'z', rotate_cell=True)
        vectors_t = get_supercell_vecs(tran_2D_t, top_rot.cell, super_z)
        vectors = (vectors_b + vectors_t)/2
        Delta = calc_vector_norm(vectors_t[0,:2] - vectors_b[0,:2])/calc_vector_norm(vectors[0,:2])

        """Get atom position"""
        bottom_super = self._coincidence(vectors, vectors_b, tran_2D_b, self.bottom)
        top_super = self._coincidence(vectors, vectors_t, tran_2D_t, top_rot)
        top_super.positions[..., 2] = top_super.positions[..., 2] + delta_z
        positions = np.vstack((bottom_super.positions, top_super.positions))
        number = np.hstack((bottom_super.atomic_numbers, top_super.atomic_numbers))
        results = py_atoms_to_ase_atoms(Pyatoms(vectors, positions, number))
        return results, Delta

    def _validate_twist_parameters(self, theta: float) -> None:
        """Validate twist angle and structure parameters."""
        if not float_eq(self.bottom.cell.cellpar()[5], 
                       self.top.cell.cellpar()[5], 
                       prec=DEFAULT_PRECISION):
            raise ValueError("[err] The angles of two layers are different")
            
        if self.params.p1 == 1 and self.params.p2 == 1 and theta == 0:
            raise ValueError("[err] Cannot generate superlattice for homostructure at 0 degree")


def get_parser():
    parser = argparse.ArgumentParser(description="Generate twisted van der Waals heterostructures.")
    parser.add_argument('-b','--bottom', required=True, help='Path to lower layer, need to be recognized by ASE')
    parser.add_argument('-t','--top', required=True, help='Path to upper layer, need to be recognized by ASE')
    parser.add_argument('-a','--angle', help='Multi twist angle',nargs='+', type=float)
    parser.add_argument('-al','--alist', help='start end step',nargs='+', type=float)
    parser.add_argument('-tol','--tolerance', help='Delta/moire_lattice (%%), default = 1%%', type=float, default = 1.0)
    parser.add_argument('-z', help='superlattice constant along z direction, default = 30 A', type=float, default = 30.0)
    parser.add_argument('-d', help='distance of two layers, default = 7 A', type=float, default = 7.0)
    parser.add_argument('-w','--write', help='Path to write supercell, need to be recognized by ASE')
    parser.add_argument('-o','--outformat', help='output file-format, like lammps-data', default = None)
    return parser
