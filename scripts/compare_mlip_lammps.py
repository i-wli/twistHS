"""Compare a candidate bilayer relaxation with the LAMMPS reference."""

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory

from twistHS.lib.analysis import _split_layers, interlayer_distance_map, stacking_map
from twistHS.mlip.calculators import finetuned_calculator, zero_shot_calculator
from twistHS.mlip.relax import relax

RMSD_FRACTION = 0.10
D_MAE = 0.02
DISC = 0.02
SITES = ("M", "X", "hollow")


def _displacements(start, final, layers):
    ds = final.get_scaled_positions(wrap=False) - start.get_scaled_positions(wrap=False)
    ds -= np.round(ds)
    u = ds @ final.cell
    for layer in layers:
        u[layer] -= u[layer].mean(axis=0)
    return u


def _strain(atoms, n_metal, a_ref):
    a_cell = np.sqrt(abs(np.linalg.det(atoms.cell[:2, :2])) / (n_metal * np.sqrt(3) / 2))
    return a_cell / a_ref - 1


def _in_plane_rms(displacements):
    return np.sqrt(np.mean(np.sum(displacements[:, :2] ** 2, axis=1)))


def comparison_metrics(start, model_final, lammps_final, *, metal, chalcogen, a_model, a_lammps):
    layers = _split_layers(start, metal, chalcogen)
    reference = _displacements(start, lammps_final, layers)
    diff = _displacements(start, model_final, layers) - reference
    distances = [interlayer_distance_map(atoms, metal, chalcogen)[1]
                 for atoms in (model_final, lammps_final)]
    maps = [stacking_map(atoms, metal, chalcogen) for atoms in (model_final, lammps_final)]
    row = {"n_atoms": len(start), "family": maps[0].family}
    for name, layer in zip(("bottom", "top"), layers):
        row[f"rmsd_in_{name}"] = _in_plane_rms(diff[layer])
        row[f"rmsd_out_{name}"] = np.sqrt(np.mean(diff[layer, 2] ** 2))
        row[f"u_in_ref_{name}"] = _in_plane_rms(reference[layer])
    row.update(
        d_mae=np.mean(np.abs(distances[0] - distances[1])),
        d_mean_model=np.mean(distances[0]),
        d_mean_lammps=np.mean(distances[1]),
    )
    for prefix in ("frac", "disc"):
        for name, atoms, a_ref, nearest in zip(
            ("model", "lammps"), (model_final, lammps_final), (a_model, a_lammps), maps,
        ):
            mapping = (nearest if prefix == "frac" else stacking_map(
                atoms, metal, chalcogen, boundary=0.15 * a_ref,
            ))
            labels = ("Rhh", "RhX", "RhM") if mapping.family == "R" else ("HhX", "Hhh", "HhM")
            for site, label in zip(SITES, labels):
                row[f"{prefix}_{site}_{name}"] = mapping.fractions[label]
    symbols = np.asarray(start.get_chemical_symbols())
    n_metal = np.count_nonzero(np.isin(symbols[layers[0]], np.atleast_1d(metal)))
    row["strain_model"] = _strain(model_final, n_metal, a_model)
    row["strain_lammps"] = _strain(lammps_final, n_metal, a_lammps)
    row["pass_rmsd_in"] = all(
        row[f"rmsd_in_{name}"] <= RMSD_FRACTION * row[f"u_in_ref_{name}"]
        for name in ("bottom", "top")
    )
    row["pass_d_mae"] = bool(row["d_mae"] <= D_MAE)
    row["pass_disc"] = max(
        abs(row[f"disc_{site}_model"] - row[f"disc_{site}_lammps"]) for site in SITES
    ) <= DISC
    row["pass"] = all(row[key] for key in ("pass_rmsd_in", "pass_d_mae", "pass_disc"))
    return row


def _read_lammps(path, start=None):
    atoms = read(path, format="lammps-data", units="metal", atom_style="atomic")
    if start is not None and not np.array_equal(start.numbers, atoms.numbers):
        raise ValueError(f"{path} must have the same atoms in the same order as the starting structure")
    atoms.set_scaled_positions(atoms.get_scaled_positions(wrap=False) % 1)
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--lammps-final", type=Path, required=True)
    parser.add_argument("--label", required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--model")
    candidate.add_argument("--candidate", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--a-model", type=float, required=True)
    parser.add_argument("--a-lammps", type=float, required=True)
    parser.add_argument("--metal", required=True)
    parser.add_argument("--chalcogen", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.model is not None and None in (args.device, args.steps, args.fmax):
        parser.error("--model requires --device, --steps and --fmax")

    start = _read_lammps(args.start)
    lammps_final = _read_lammps(args.lammps_final, start)
    args.out.mkdir(parents=True, exist_ok=True)
    steps_used = f_rms = f_max = None
    if args.candidate is not None:
        model = args.candidate.stem
        model_final = _read_lammps(args.candidate, start)
    else:
        calculator = (zero_shot_calculator(args.device) if args.model == "zero-shot"
                      else finetuned_calculator(args.model, args.device))
        model = Path(args.model).stem
        trajectory = args.out / f"{args.label}_{model}.traj"
        model_final = relax(
            start, calculator, steps=args.steps, fmax=args.fmax, trajectory=trajectory,
        )
        write(args.out / f"{args.label}_{model}.extxyz", model_final, format="extxyz")
        with Trajectory(trajectory) as frames:
            steps_used = len(frames) - 1
        reference = lammps_final.copy()
        reference.calc = calculator
        forces = reference.get_forces()
        f_rms = np.sqrt(np.mean(forces ** 2))
        f_max = np.linalg.norm(forces, axis=1).max()

    row = {
        "label": args.label, "model": model, "steps_used": steps_used,
        "f_rms_at_lammps": f_rms, "f_max_at_lammps": f_max,
    }
    row.update(comparison_metrics(
        start, model_final, lammps_final, metal=args.metal, chalcogen=args.chalcogen,
        a_model=args.a_model, a_lammps=args.a_lammps,
    ))
    destination = args.out / "compare.csv"
    exists = destination.exists()
    with destination.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
