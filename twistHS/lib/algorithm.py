import argparse
import logging
import numpy as np
import ase.io
from typing import Tuple
from twistHS.lib.func import *
from twistHS.lib.moire import principal_strains, supercell_matrices

class HSGenerator:
    """Generates twisted vdW heterostructure supercells."""
    
    def __init__(self, bottom: Atoms, top: Atoms):
        self.bottom = bottom
        self.top = top
        self.logger = logging.getLogger(__name__)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input atomic structures."""
        if not isinstance(self.bottom, Atoms) or not isinstance(self.top, Atoms):
            raise ValueError("[err] Bottom and top must be ASE Atoms objects")
        check_vectors(self.bottom)
        check_vectors(self.top)

    def _coincidence(self, vectors_super: np.ndarray, 
                    vectors_indiv: np.ndarray,
                    trans_2D: np.ndarray, 
                    unit: Atoms) -> Pyatoms:
        superatoms = get_supercell_atoms_pos(vectors_indiv, trans_2D, unit)
        superatoms_frac = coord_cart2frac(vectors_indiv, superatoms.positions)
        superatoms_pos = coord_frac2cart(vectors_super, superatoms_frac)
        return Pyatoms(vectors_super, superatoms_pos, superatoms.numbers)

    def gen_supercell(self, theta: float, super_z: float, delta_z: float) -> Tuple[Atoms, float]:
        """Generate twisted supercell
        
        Args:
            theta: twist angle in degrees
            super_z: supercell constant in z direction
            delta_z: distance between two layers            
        
        Returns:
            (Atoms, float): structure and maximum absolute principal strain
        """
        top_rot = self.top.copy()
        top_rot.rotate(theta, 'z', rotate_cell=True)
        tran_2D_b, tran_2D_t = supercell_matrices(
            self.bottom.cell[:2, :2], top_rot.cell[:2, :2])
        vectors_b = get_supercell_vecs(tran_2D_b, self.bottom.cell, super_z)
        vectors_t = get_supercell_vecs(tran_2D_t, top_rot.cell, super_z)
        vectors = (vectors_b + vectors_t)/2
        Delta = float(max(
            np.max(np.abs(principal_strains(vectors_b[:2, :2], vectors[:2, :2]))),
            np.max(np.abs(principal_strains(vectors_t[:2, :2], vectors[:2, :2]))),
        ))

        """Get atom position"""
        bottom_super = self._coincidence(vectors, vectors_b, tran_2D_b, self.bottom)
        top_super = self._coincidence(vectors, vectors_t, tran_2D_t, top_rot)
        top_super.positions[..., 2] = top_super.positions[..., 2] + delta_z
        positions = np.vstack((bottom_super.positions, top_super.positions))
        number = np.hstack((bottom_super.atomic_numbers, top_super.atomic_numbers))
        results = py_atoms_to_ase_atoms(Pyatoms(vectors, positions, number))
        return results, Delta

    def scan(self, angles, super_z, delta_z, tolerance):
        """Return (angle, structure, strain) entries within tolerance in percent.

        Invalid geometry raises immediately; it is not silently omitted.
        """
        results = []
        for angle in angles:
            structure, strain = self.gen_supercell(angle, super_z, delta_z)
            if strain * 100 <= tolerance:
                results.append((angle, structure, strain))
        return results



def get_parser():
    parser = argparse.ArgumentParser(description="Generate twisted van der Waals heterostructures.")
    parser.add_argument('-b','--bottom', required=True, help='Path to lower layer, need to be recognized by ASE')
    parser.add_argument('-t','--top', required=True, help='Path to upper layer, need to be recognized by ASE')
    parser.add_argument('-a','--angle', help='Multi twist angle',nargs='+', type=float)
    parser.add_argument('-al','--alist', help='start end step',nargs='+', type=float)
    parser.add_argument('-tol','--tolerance', help='max principal strain of either layer (%%), default = 1%%', type=float, default = 1.0)
    parser.add_argument('-z', help='superlattice constant along z direction, default = 30 A', type=float, default = 30.0)
    parser.add_argument('-d', help='distance of two layers, default = 7 A', type=float, default = 7.0)
    parser.add_argument('-w','--write', help='Path to write supercell, need to be recognized by ASE')
    parser.add_argument('-o','--outformat', help='output file-format, like lammps-data', default = None)
    return parser
