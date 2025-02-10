"""Append the input and MLIP-relaxed monolayer lattice constants to CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read

from twistHS.mlip.calculators import finetuned_calculator, zero_shot_calculator
from twistHS.mlip.relax import relax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monolayer", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atoms = read(args.monolayer)
    calculator = (zero_shot_calculator(args.device) if args.model == "zero-shot"
                  else finetuned_calculator(args.model, args.device))
    relaxed = relax(atoms, calculator, steps=args.steps)
    row = {
        "model": "zero-shot" if args.model == "zero-shot" else Path(args.model).stem,
        "a_input": np.linalg.norm(atoms.cell[0]),
        "a_model": np.linalg.norm(relaxed.cell[0]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "monolayer.csv"
    exists = destination.exists()
    with destination.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
