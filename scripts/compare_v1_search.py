"""Compare private twistHS v1 TMDC starting cells (first-order) with coincidence search."""

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

from twistHS.lib.coincidence import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_MAX_ATOMS,
    DEFAULT_MOIRE_TOL,
    DEFAULT_TOLERANCE,
    find_coincidence,
)
from twistHS.lib.moire import deformation, reduce_basis


def archive_layer_strain(archive, reference, metal_type):
    """Measure the in-plane strain from the metal sites of one hexagonal layer."""
    types = archive.arrays["type"]
    layer = archive[(types >= metal_type) & (types < metal_type + 3)]
    if len(reference) != 3 or not np.array_equal(
        np.sort(layer.numbers), np.repeat(np.sort(reference.numbers), len(layer) // 3)
    ):
        raise ValueError("Archive layer composition does not match the reference monolayer")

    sites = archive[types == metal_type]
    if len(sites) == 0 or 3 * len(sites) != len(layer):
        raise ValueError("Expected one metal site per TMDC primitive cell")
    sites.pbc = (True, True, False)
    sites.positions[:, 2] = 0

    lattice, _ = reduce_basis(reference.cell[:2, :2])
    if np.linalg.det(lattice) < 0:
        lattice[1] *= -1
    if lattice[0] @ lattice[1] < 0:
        lattice[1] += lattice[0]
    length_sq = lattice[0] @ lattice[0]
    if not np.allclose(lattice @ lattice.T, length_sq * np.array([[1, 0.5], [0.5, 1]]),
                       rtol=1e-7, atol=0):
        raise ValueError("Expected a hexagonal reference monolayer")

    # The first triangular shell has six sites. The cutoff lies between
    # the first and second shells for the small strains in the v1 archive.
    centers, vectors = neighbor_list("iD", sites, 1.35 * np.sqrt(length_sq))
    shell = vectors[centers == 0, :2]
    if len(shell) != 6:
        raise ValueError("Expected six first-shell neighbors around the archived metal site")
    order = np.argsort(np.arctan2(shell[:, 1], shell[:, 0]))
    primitive = shell[order[:2]]
    area = abs(np.linalg.det(sites.cell[:2, :2]))
    if not np.isclose(np.linalg.det(primitive) * len(sites), area, rtol=1e-7, atol=0):
        raise ValueError("Recovered metal primitive cell does not tile the archived cell")
    strain, _ = deformation(lattice, primitive)
    return float(np.max(np.abs(strain)))


def compare(archive_root, system, bottom, top, *, tolerance, angle_tol, max_atoms,
            moire_tol=DEFAULT_MOIRE_TOL):
    files = sorted((archive_root / system).glob("*/lammps.lmp"),
                   key=lambda path: float(path.parent.name))
    if not files:
        raise ValueError(f"No archived lammps.lmp files found for {system}")

    for path in files:
        angle = float(path.parent.name)
        try:
            archive = read(path, format="lammps-data", units="metal", atom_style="atomic")
            strain = max(archive_layer_strain(archive, bottom, 1),
                         archive_layer_strain(archive, top, 4)) * 100
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error

        rotated = top.copy()
        rotated.rotate(angle, "z", rotate_cell=True)
        match = find_coincidence(
            bottom.cell[:2, :2], rotated.cell[:2, :2], (len(bottom), len(top)),
            tolerance=tolerance, angle_tol=angle_tol, moire_tol=moire_tol, max_atoms=max_atoms,
        )
        yield {
            "angle_deg": angle,
            "tolerance_pct": tolerance,
            "moire_tol_pct": moire_tol,
            "v1_atoms": len(archive),
            "v1_strain_pct": strain,
            "v1_within_tolerance": strain <= tolerance,
            "search_atoms": match.atom_count if match is not None else None,
            "search_strain_pct": match.strain * 100 if match is not None else None,
            "search_within_limits": match is not None,
            "search_theta_eff_deg": angle + match.angle_offset if match is not None else None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("system", help="System directory within the archive")
    parser.add_argument("bottom", type=Path, help="Unstrained lower monolayer")
    parser.add_argument("top", type=Path, help="Unstrained upper monolayer")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--angle-tol", type=float, default=DEFAULT_ANGLE_TOL)
    parser.add_argument("--moire-tol", type=float, default=DEFAULT_MOIRE_TOL)
    parser.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    args = parser.parse_args()

    rows = list(compare(args.archive_root, args.system, read(args.bottom), read(args.top),
                        tolerance=args.tolerance, angle_tol=args.angle_tol,
                        moire_tol=args.moire_tol, max_atoms=args.max_atoms))
    destination = Path("results") / f"{Path(args.system).name}_v1_search.csv"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} comparisons to {destination}")


if __name__ == "__main__":
    main()
