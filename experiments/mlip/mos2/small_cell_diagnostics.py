"""Compare small-cell MLIP diagnostics with the LAMMPS reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.constraints import FixCartesian
from ase.io import read

from twistHS.lib.sampling import high_symmetry_bilayers
from twistHS.mlip.calculators import finetuned_calculator, zero_shot_calculator
from twistHS.mlip.relax import relax

FMAX = 1e-4
STRAIN = 0.01


def elastic_moduli(layer, calculator, steps):
    stresses = {}
    for mode, axes in (("biaxial", [1, 1, 0]), ("shear", [1, -1, 0])):
        for sign in (-1, 1):
            strained = layer.copy()
            deformation = np.eye(3) + np.diag(np.asarray(axes) * sign * STRAIN)
            strained.set_cell(layer.cell @ deformation, scale_atoms=True)
            final = relax(strained, calculator, steps=steps, cell=False, fmax=FMAX)
            stresses[mode, sign] = final.get_stress(voigt=False) * final.cell[2, 2]
    biaxial = stresses["biaxial", 1] - stresses["biaxial", -1]
    shear = stresses["shear", 1] - stresses["shear", -1]
    return (biaxial[0, 0] + biaxial[1, 1]) / (4 * STRAIN), (
        shear[0, 0] - shear[1, 1]
    ) / (8 * STRAIN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monolayer", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    layer = read(args.monolayer)
    reference = json.loads(args.reference.read_text())
    calculator = (zero_shot_calculator(args.device) if args.model == "zero-shot"
                  else finetuned_calculator(args.model, args.device))
    model = Path(args.model).stem
    relaxed = relax(layer, calculator, steps=args.steps, fmax=FMAX)
    a = np.linalg.norm(relaxed.cell[0])
    distances, energies = {}, {}
    for label, atoms in high_symmetry_bilayers(layer, 6.5, 30).items():
        atoms.set_constraint(FixCartesian(range(6), mask=(True, True, False)))
        final = relax(atoms, calculator, steps=args.steps, cell=False, fmax=FMAX)
        distances[label] = final.positions[3, 2] - final.positions[0, 2]
        energies[label] = final.get_potential_energy()
    biaxial, shear = elastic_moduli(relaxed, calculator, args.steps)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"small_cells_{model}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "reference", "model"))
        writer.writerow(("a", reference["a"], a))
        for label, d in distances.items():
            writer.writerow((f"d_{label}", reference["stackings"][label]["d"], d))
        for label, energy in energies.items():
            writer.writerow((
                f"energy_{label}", reference["stackings"][label]["energy"], energy - energies["Rhh"],
            ))
        writer.writerow(("biaxial_modulus", reference["biaxial_modulus"], biaxial))
        writer.writerow(("shear_modulus", reference["shear_modulus"], shear))


if __name__ == "__main__":
    main()
