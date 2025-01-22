import argparse
import logging

import numpy as np
from ase import Atoms

from twistHS.lib.coincidence import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_MAX_ATOMS,
    DEFAULT_MOIRE_ORDER,
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

DEFAULT_SUPER_Z = 30.0
DEFAULT_DELTA_Z = 7.0
CONSTRUCTION_METHODS = ("search", "first-order")
DESCRIPTION = "Construct commensurate moiré supercells for twisted van der Waals bilayers."

class HSGenerator:
    """Construct commensurate moiré supercells from two monolayer primitive cells.

    Coincidence-lattice search minimizes the supercell atom count subject to
    strain, twist-angle deviation, and moiré wave-vector constraints.
    The strain limit constrains the maximum absolute principal engineering
    strain across both layers, expressed as a percentage. The moiré tolerance
    limits the relative change of any linear combination in either first-order reference
    set. A moiré cutoff above 1 also constrains individual finite-period
    higher-index wave vectors within that cutoff.
    """
    
    def __init__(self, bottom: Atoms, top: Atoms, *, method="search",
                 tolerance=DEFAULT_TOLERANCE, angle_tol=DEFAULT_ANGLE_TOL,
                 moire_tol=DEFAULT_MOIRE_TOL, moire_order=DEFAULT_MOIRE_ORDER,
                 max_atoms=DEFAULT_MAX_ATOMS, backend="numpy"):
        self.bottom = bottom
        self.top = top
        self.logger = logging.getLogger(__name__)
        self._validate_inputs()
        if method not in CONSTRUCTION_METHODS:
            raise ValueError("method must be 'search' or 'first-order'")
        validate_tolerance(tolerance)
        validate_search_limits(angle_tol, max_atoms, moire_tol=moire_tol, moire_order=moire_order)
        validate_search_backend(backend)
        self.method = method
        self.tolerance = tolerance
        self.angle_tol = angle_tol
        self.moire_tol = moire_tol
        self.moire_order = moire_order
        self.max_atoms = max_atoms
        self.backend = backend

    def _validate_inputs(self) -> None:
        """Validate input atomic structures."""
        if not isinstance(self.bottom, Atoms) or not isinstance(self.top, Atoms):
            raise ValueError("The lower- and upper-layer structures must be ASE Atoms objects.")
        if not len(self.bottom) or not len(self.top):
            raise ValueError("Both monolayer primitive cells must contain atoms.")
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
        """Return the supercell, maximum absolute principal engineering strain as
        a fraction, and effective twist angle in degrees.

        theta is the requested twist angle in degrees. super_z is the
        out-of-plane lattice parameter in Å. delta_z is the interlayer
        distance in Å: the z coordinate of the upper-layer midplane minus that
        of the lower-layer midplane, before periodic wrapping.
        """
        result = self._generate(theta, super_z, delta_z, self.tolerance)
        if result is None:
            raise ValueError(f"No commensurate supercell at a requested twist angle of {theta:g}° satisfies the specified constraints.")
        return result

    def _generate(self, theta, super_z, delta_z, tolerance):
        if not np.isfinite(theta) or not np.isfinite(delta_z):
            raise ValueError("The requested twist angle (theta) and interlayer distance (delta_z) must be finite.")
        if not np.isfinite(super_z) or super_z <= 0:
            raise ValueError("The out-of-plane lattice parameter (super_z) must be finite and positive.")
        top_rot = self.top.copy()
        top_rot.rotate(theta, 'z', rotate_cell=True)
        bottom, top = self.bottom.cell[:2, :2], top_rot.cell[:2, :2]
        if self.method == "search":
            match = find_coincidence(
                bottom, top, (len(self.bottom), len(self.top)),
                tolerance=tolerance, angle_tol=self.angle_tol, moire_tol=self.moire_tol,
                moire_order=self.moire_order, max_atoms=self.max_atoms, backend=self.backend,
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
        bottom_z = bottom_super.positions[:, 2]
        top_z = top_super.positions[:, 2]
        bottom_mid = (bottom_z.min() + bottom_z.max()) / 2
        top_mid = (top_z.min() + top_z.max()) / 2
        top_z += bottom_mid + delta_z - top_mid
        positions = np.vstack((bottom_super.positions, top_super.positions))
        number = np.hstack((bottom_super.atomic_numbers, top_super.atomic_numbers))
        results = py_atoms_to_ase_atoms(Pyatoms(vectors, positions, number))
        results.info["theta_eff"] = theta_eff
        return results, strain, theta_eff

    def scan(self, angles, super_z, delta_z, tolerance=None):
        """Return (requested twist angle, supercell, strain) for each accepted angle.

        Angles are in degrees and strain is the maximum absolute principal
        engineering strain as a fraction. Each supercell stores the effective
        twist angle in info["theta_eff"]. super_z and delta_z have the same
        meanings as in gen_supercell. An explicit tolerance overrides the
        configured strain limit in percent.
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
        """Return geometric moiré components at the requested twist in degrees.

        max_order scales each layer's shortest nonzero reciprocal-lattice
        vector to define the reciprocal cutoff.
        """
        top_rot = self.top.copy()
        top_rot.rotate(theta, 'z', rotate_cell=True)
        return moire_components(self.bottom.cell[:2, :2], top_rot.cell[:2, :2], max_order)


def get_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-b", "--bottom", required=True, metavar="PATH",
                        help="Lower-layer primitive cell in an ASE-readable file")
    parser.add_argument("-t", "--top", required=True, metavar="PATH",
                        help="Upper-layer primitive cell in an ASE-readable file")
    angles = parser.add_mutually_exclusive_group(required=True)
    angles.add_argument("-a", "--angle", nargs="+", type=float, metavar="ANGLE",
                        help="Requested twist angles (degrees), in scan order")
    angles.add_argument("-al", "--alist", nargs=3, metavar=("START", "STOP", "STEP"),
                        help="Requested twist-angle interval (degrees), including START and excluding STOP")
    parser.add_argument("-tol", "--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Maximum strain: largest absolute principal engineering strain in either layer "
                             "(%%, default: %(default)s)")
    parser.add_argument("--method", choices=CONSTRUCTION_METHODS, default="search",
                        help="Supercell construction method (default: %(default)s)")
    parser.add_argument("--backend", choices=SEARCH_BACKENDS, default="numpy",
                        help="Coincidence-lattice search backend (default: %(default)s)")
    parser.add_argument("--angle-tol", type=float, default=DEFAULT_ANGLE_TOL,
                        help="Maximum deviation between requested and effective twist angles in coincidence-lattice "
                             "search (degrees, default: %(default)s)")
    parser.add_argument("--moire-tol", type=float, default=DEFAULT_MOIRE_TOL,
                        help="Maximum relative change of first-order moiré wave-vector combinations and enabled "
                             "higher-index wave vectors (%%, default: %(default)s)")
    parser.add_argument("--moire-order", type=float, default=DEFAULT_MOIRE_ORDER,
                        help="Reciprocal cutoff for the moiré constraint, in multiples of the length of each layer's shortest "
                             "nonzero reciprocal-lattice vector. Use 1 for the first-order constraint (default: %(default)s)")
    parser.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS,
                        help="Maximum number of atoms per supercell in coincidence-lattice search (default: %(default)s)")
    parser.add_argument("-mo", "--max-order", type=int, metavar="ORDER",
                        help="Reciprocal cutoff for geometric moiré component analysis, in multiples of the length of each "
                             "layer's shortest nonzero reciprocal-lattice vector (positive integer)")
    parser.add_argument("-z", type=float, default=DEFAULT_SUPER_Z,
                        help="Out-of-plane supercell lattice parameter (Å, default: %(default)s)")
    parser.add_argument("-d", type=float, default=DEFAULT_DELTA_Z,
                        help="Interlayer distance from the lower- to the upper-layer midplane "
                             "(Å, default: %(default)s)")
    parser.add_argument("-w", "--write", metavar="PATH",
                        help="Supercell export path (default: structure_<requested angle to two decimal places>.xsf)")
    parser.add_argument("-o", "--outformat", metavar="FORMAT",
                        help="Explicit ASE output format, e.g. lammps-data (requires --write)")
    return parser
