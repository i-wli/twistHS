import argparse
import logging

import numpy as np
from ase import Atoms

from twistHS.lib.coincidence import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_MAX_ATOMS,
    DEFAULT_MOIRE_TOL,
    DEFAULT_TOLERANCE,
    SEARCH_BACKENDS,
    find_coincidence,
    validate_search_backend,
    validate_search_limits,
    validate_tolerance,
)
from twistHS.lib.func import (
    Pyatoms,
    check_vectors,
    get_supercell_atoms_pos,
    get_supercell_vecs,
    py_atoms_to_ase_atoms,
)
from twistHS.lib.moire import deformation, moire_components, supercell_matrices

class HSGenerator:
    """Generates twisted vdW heterostructure supercells."""
    
    def __init__(self, bottom: Atoms, top: Atoms, *, method="search",
                 tolerance=DEFAULT_TOLERANCE, angle_tol=DEFAULT_ANGLE_TOL,
                 moire_tol=DEFAULT_MOIRE_TOL, max_atoms=DEFAULT_MAX_ATOMS, backend="numpy"):
        self.bottom = bottom
        self.top = top
        self.logger = logging.getLogger(__name__)
        self._validate_inputs()
        if method not in ("search", "first-order"):
            raise ValueError("method must be 'search' or 'first-order'")
        validate_tolerance(tolerance)
        validate_search_limits(angle_tol, max_atoms, moire_tol=moire_tol)
        validate_search_backend(backend)
        self.method = method
        self.tolerance = tolerance
        self.angle_tol = angle_tol
        self.moire_tol = moire_tol
        self.max_atoms = max_atoms
        self.backend = backend

    def _validate_inputs(self) -> None:
        """Validate input atomic structures."""
        if not isinstance(self.bottom, Atoms) or not isinstance(self.top, Atoms):
            raise ValueError("[err] Bottom and top must be ASE Atoms objects")
        if not len(self.bottom) or not len(self.top):
            raise ValueError("[err] Both primitive cells must contain atoms")
        check_vectors(self.bottom)
        check_vectors(self.top)

    def _coincidence(self, vectors_super: np.ndarray, 
                    vectors_indiv: np.ndarray,
                    trans_2D: np.ndarray, 
                    unit: Atoms) -> Pyatoms:
        superatoms = get_supercell_atoms_pos(vectors_indiv, trans_2D, unit)
        deformation = np.linalg.solve(vectors_indiv, vectors_super)
        superatoms_pos = superatoms.positions @ deformation
        return Pyatoms(vectors_super, superatoms_pos, superatoms.numbers)

    def gen_supercell(self, theta: float, super_z: float, delta_z: float) -> tuple[Atoms, float, float]:
        """Return (structure, maximum absolute principal strain, effective angle in degrees)."""
        result = self._generate(theta, super_z, delta_z, self.tolerance)
        if result is None:
            raise ValueError(f"No coincidence cell at {theta:g}° within the search limits")
        return result

    def _generate(self, theta, super_z, delta_z, tolerance):
        if not np.isfinite(theta) or not np.isfinite(delta_z):
            raise ValueError("theta and delta_z must be finite")
        if not np.isfinite(super_z) or super_z <= 0:
            raise ValueError("super_z must be finite and positive")
        top_rot = self.top.copy()
        top_rot.rotate(theta, 'z', rotate_cell=True)
        bottom, top = self.bottom.cell[:2, :2], top_rot.cell[:2, :2]
        if self.method == "search":
            match = find_coincidence(
                bottom, top, (len(self.bottom), len(self.top)),
                tolerance=tolerance, angle_tol=self.angle_tol, moire_tol=self.moire_tol,
                max_atoms=self.max_atoms, backend=self.backend,
            )
            if match is None:
                return None
            tran_2D_b, tran_2D_t = match.bottom, match.top
        else:
            tran_2D_b, tran_2D_t = supercell_matrices(bottom, top)

        vectors_b = get_supercell_vecs(tran_2D_b, self.bottom.cell, super_z)
        vectors_t = get_supercell_vecs(tran_2D_t, top_rot.cell, super_z)
        vectors = (vectors_b + vectors_t) / 2
        if self.method == "search":
            strain = match.strain
            theta_eff = float(theta + match.angle_offset)
        else:
            strains_b, phi_b = deformation(vectors_b[:2, :2], vectors[:2, :2])
            strains_t, phi_t = deformation(vectors_t[:2, :2], vectors[:2, :2])
            strain = float(np.max(np.abs([strains_b, strains_t])))
            theta_eff = float(theta + phi_t - phi_b)

        bottom_super = self._coincidence(vectors, vectors_b, tran_2D_b, self.bottom)
        top_super = self._coincidence(vectors, vectors_t, tran_2D_t, top_rot)
        top_super.positions[..., 2] = top_super.positions[..., 2] + delta_z
        positions = np.vstack((bottom_super.positions, top_super.positions))
        number = np.hstack((bottom_super.atomic_numbers, top_super.atomic_numbers))
        results = py_atoms_to_ase_atoms(Pyatoms(vectors, positions, number))
        results.info["theta_eff"] = theta_eff
        return results, strain, theta_eff

    def scan(self, angles, super_z, delta_z, tolerance=None):
        """Return (requested angle, structure, strain) entries for matched angles.

        Each structure carries its effective angle as info["theta_eff"].
        """
        tolerance = self.tolerance if tolerance is None else tolerance
        validate_tolerance(tolerance)
        results = []
        for angle in angles:
            result = self._generate(angle, super_z, delta_z, tolerance)
            if result is None:
                continue
            structure, strain, _ = result
            if self.method == "search" or strain * 100 <= tolerance:
                results.append((angle, structure, strain))
        return results

    def moire_components(self, theta, max_order):
        """Analyze reciprocal components without constructing an atomic cell."""
        top_rot = self.top.copy()
        top_rot.rotate(theta, 'z', rotate_cell=True)
        return moire_components(self.bottom.cell[:2, :2], top_rot.cell[:2, :2], max_order)


def get_parser():
    parser = argparse.ArgumentParser(description="Generate twisted van der Waals heterostructures.")
    parser.add_argument('-b','--bottom', required=True, help='Path to lower layer, need to be recognized by ASE')
    parser.add_argument('-t','--top', required=True, help='Path to upper layer, need to be recognized by ASE')
    parser.add_argument('-a','--angle', help='Multi twist angle',nargs='+', type=float)
    parser.add_argument('-al','--alist', help='start end step',nargs='+', type=float)
    parser.add_argument('-tol','--tolerance', help='max principal strain of either layer (%%), default = 1%%', type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument('--method', choices=('search', 'first-order'), default='search')
    parser.add_argument('--backend', choices=SEARCH_BACKENDS, default='numpy',
                        help='coincidence search backend (default: %(default)s)')
    parser.add_argument('--angle-tol', type=float, default=DEFAULT_ANGLE_TOL,
                        help='maximum effective angle deviation in degrees for search (default: %(default)s)')
    parser.add_argument('--moire-tol', type=float, default=DEFAULT_MOIRE_TOL,
                        help='maximum relative moiré wave-vector change in any direction (%%, default: %(default)s)')
    parser.add_argument('--max-atoms', type=int, default=DEFAULT_MAX_ATOMS,
                        help='maximum total atom count for search (default: %(default)s)')
    parser.add_argument('-mo', '--max-order', type=int, help='reciprocal length cutoff for higher-order analysis (integer >= 1)')
    parser.add_argument('-z', help='superlattice constant along z direction, default = 30 A', type=float, default = 30.0)
    parser.add_argument('-d', help='distance of two layers, default = 7 A', type=float, default = 7.0)
    parser.add_argument('-w','--write', help='Path to write supercell, need to be recognized by ASE')
    parser.add_argument('-o','--outformat', help='output file-format, like lammps-data', default = None)
    return parser
