"""Run MoS2 SW/mod+KC LAMMPS labeling and reference relaxations."""

import argparse
import json
import re
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from ase import units
from ase.calculators.lammps.coordinatetransform import Prism
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
from ase.neighborlist import neighbor_list

from twistHS.lib.algorithm import HSGenerator
from twistHS.lib.analysis import _split_layers
from twistHS.lib.sampling import high_symmetry_bilayers, sample_aligned_bilayers
from twistHS.mlip.finetune import validate_dataset


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/mlip"
LOCAL = OUT / "lammps"
POTENTIALS = Path(__file__).resolve().parent / "potentials"
OBSERVABLES = ("pe", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz", "fmax", "fnorm")
TEST_CELLS = [
    ("R_5.09", 5.09, 762), ("R_3.89", 3.89, 1302),
    ("R_3.15", 3.15, 1986), ("R_2.45", 2.45, 3282),
    ("R_2.00", 2.00, 4902),
]


def atom_types(atoms, monolayer=False):
    symbols = np.asarray(atoms.get_chemical_symbols())
    layers = [np.arange(len(atoms))] if monolayer else _split_layers(atoms, "Mo", "S")
    types = np.zeros(len(atoms), dtype=int)
    for number, layer in enumerate(layers):
        metal = layer[symbols[layer] == "Mo"]
        sulfur = layer[symbols[layer] == "S"]
        assert len(sulfur) == 2 * len(metal)
        center = atoms.positions[metal, 2].mean()
        lower = sulfur[atoms.positions[sulfur, 2] < center]
        upper = sulfur[atoms.positions[sulfur, 2] > center]
        assert len(lower) == len(upper) == len(metal)
        types[metal] = 3 * number + 1
        types[lower] = 3 * number + 2
        types[upper] = 3 * number + 3
    assert np.all(types > 0)
    return types


def write_data(atoms, directory, monolayer=False):
    directory.mkdir(parents=True, exist_ok=True)
    prism = Prism(atoms.cell, pbc=atoms.pbc, reduce_cell=False)
    types = atom_types(atoms, monolayer)
    x, y, z, xy, xz, yz = prism.get_lammps_prism()
    species = ["Mo", "S", "S"] * (1 if monolayer else 2)
    positions = prism.vector_to_lammps(atoms.positions, wrap=True)
    with (directory / "lammps.lmp").open("w") as stream:
        stream.write(f"MoS2 SW/mod+KC\n\n{len(atoms)} atoms\n{len(species)} atom types\n\n")
        stream.write(f"0 {x:.17g} xlo xhi\n0 {y:.17g} ylo yhi\n0 {z:.17g} zlo zhi\n")
        stream.write(f"{xy:.17g} {xz:.17g} {yz:.17g} xy xz yz\n\nMasses\n\n")
        for i, symbol in enumerate(species, 1):
            stream.write(f"{i} {atomic_masses[atomic_numbers[symbol]]:.17g}\n")
        stream.write("\nAtoms # atomic\n\n")
        for i, (kind, position) in enumerate(zip(types, positions), 1):
            stream.write(f"{i} {kind} " + " ".join(f"{v:.17g}" for v in position) + "\n")
    restored = read(directory / "lammps.lmp", format="lammps-data", units="metal", atom_style="atomic")
    np.testing.assert_array_equal(restored.numbers, atoms.numbers)
    np.testing.assert_array_equal(restored.arrays["id"], np.arange(1, len(atoms) + 1))
    np.testing.assert_array_equal(restored.arrays["type"], types)
    np.testing.assert_allclose(prism.vector_to_ase(restored.cell), atoms.cell, atol=1e-10)
    ds = restored.get_scaled_positions() - atoms.get_scaled_positions()
    ds -= np.round(ds)
    np.testing.assert_allclose(ds, 0, atol=1e-10)
    return prism


def potential_commands(monolayer):
    sw = f'"{POTENTIALS / "tmd.sw"}"'
    kc = f'"{POTENTIALS / "MoWSSe_main.KC"}"'
    if monolayer:
        return f"pair_style sw/mod maxdelcs 0.25 0.35\npair_coeff * * {sw} Mo S S\n"
    return f"""pair_style hybrid/overlay sw/mod maxdelcs 0.25 0.35 sw/mod maxdelcs 0.25 0.35 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 zero 10.0
pair_coeff * * sw/mod 1 {sw} Mo S S NULL NULL NULL
pair_coeff * * sw/mod 2 {sw} NULL NULL NULL Mo S S
pair_coeff 1 5 kolmogorov/crespi/z 1 {kc} Mo NULL NULL NULL S NULL
pair_coeff 3 4 kolmogorov/crespi/z 2 {kc} NULL NULL S Mo NULL NULL
pair_coeff 3 5 kolmogorov/crespi/z 3 {kc} NULL NULL S NULL S NULL
pair_coeff 1 4 kolmogorov/crespi/z 4 {kc} Mo NULL NULL Mo NULL NULL
pair_coeff * * zero
compute sw1 all pair sw/mod 1
compute sw2 all pair sw/mod 2
compute kc1 all pair kolmogorov/crespi/z 1
compute kc2 all pair kolmogorov/crespi/z 2
compute kc3 all pair kolmogorov/crespi/z 3
compute kc4 all pair kolmogorov/crespi/z 4
"""


def setup_commands(monolayer):
    return """units metal
dimension 3
box tilt large
atom_style atomic
boundary p p p
read_data lammps.lmp
""" + potential_commands(monolayer) + """neighbor 2 bin
neigh_modify every 1 delay 0 check yes
"""


def run_lammps(directory, commands):
    (directory / "lammps.in").write_text(commands)
    with (directory / "driver.txt").open("w") as stream:
        subprocess.run(
            [str(lmp), "-in", "lammps.in", "-log", "log.lammps", "-screen", "none"],
            cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True,
        )


def stopping_criteria(directory):
    return re.findall(r"Stopping criterion = (.+)", (directory / "log.lammps").read_text())


def calculate(atoms, directory, *, minimize=False, monolayer=False):
    prism = write_data(atoms, directory, monolayer)
    commands = setup_commands(monolayer) + "thermo 1000\n"
    thermo = "thermo_style custom step pe fmax fnorm pxx pyy pzz pxy pxz pyz lx ly lz xy"
    if not monolayer:
        thermo += " c_sw1 c_sw2 c_kc1 c_kc2 c_kc3 c_kc4"
    commands += thermo + "\nthermo_modify format float %.17g\n"
    if minimize:
        commands += "min_style cg\nmin_modify line forcezero\nfix thermostat all nve\n"
        commands += """fix stress all box/relax x 0.0 y 0.0 couple xy
minimize 0.0 1.0e-4 100000 100000
unfix stress
minimize 0.0 1.0e-4 100000 100000
""" * 6
        commands += """write_data cg.lmp nocoeff
min_style fire
min_modify norm max
minimize 0.0 1.0e-5 200000 200000
velocity all set 0.0 0.0 0.0
write_data min.lmp nocoeff
"""
    commands += "run 0\n"
    commands += 'print "' + " ".join(f"$({name}:%.17g)" for name in OBSERVABLES)
    commands += '" file observables.txt screen no\n'
    commands += "write_dump all custom forces.dump id type x y z fx fy fz modify sort id format float %.17g\n"
    started = perf_counter()
    run_lammps(directory, commands)
    stats = dict(zip(OBSERVABLES, np.loadtxt(directory / "observables.txt")))
    stats["seconds"] = perf_counter() - started
    dump = np.loadtxt(directory / "forces.dump", skiprows=9)
    np.testing.assert_array_equal(dump[:, 0], np.arange(1, len(atoms) + 1))
    np.testing.assert_array_equal(dump[:, 1], atom_types(atoms, monolayer))
    forces = prism.vector_to_ase(dump[:, 5:8])
    pxx, pyy, pzz, pxy, pxz, pyz = (stats[name] for name in OBSERVABLES[1:7])
    pressure = np.array([[pxx, pxy, pxz], [pxy, pyy, pyz], [pxz, pyz, pzz]])
    stress = prism.tensor2_to_ase(-pressure * 1e5 * units.Pascal)
    assert np.isfinite([stats["pe"], *forces.ravel(), *stress.ravel()]).all()
    stats["max_force"] = float(np.linalg.norm(forces, axis=1).max())
    stats["mean_in_plane_pressure_bar"] = (pxx + pyy) / 2
    if minimize:
        criteria = stopping_criteria(directory)
        assert len(criteria) == 13, (directory, criteria)
        if any("max iterations" in reason or "max force evaluations" in reason for reason in criteria):
            raise RuntimeError(f"LAMMPS minimization exhausted its budget: {directory}")
        if criteria[-1] != "force tolerance":
            raise RuntimeError(f"LAMMPS FIRE stopped at {criteria[-1]}: {directory}")
        stats["stopping_criteria"] = criteria
        final = read(directory / "min.lmp", format="lammps-data", units="metal", atom_style="atomic")
        np.testing.assert_array_equal(final.numbers, atoms.numbers)
        np.testing.assert_array_equal(final.arrays["type"], atom_types(atoms, monolayer))
        final.cell = prism.update_cell(final.cell)
        final.positions = prism.vector_to_ase(dump[:, 2:5])
        deformation = np.linalg.solve(atoms.cell, final.cell)
        scale = deformation[0, 0]
        np.testing.assert_allclose(deformation, np.diag([scale, scale, 1]), atol=1e-10)
        stats["in_plane_scale"] = float(scale)
        stats["force_threshold_passed"] = stats["max_force"] < 1e-5
        stats["final_force_tolerance_reached"] = criteria[-1] == "force tolerance"
        write(directory / "final.extxyz", final, format="extxyz")
    (directory / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats, forces, stress


def relaxed_monolayer():
    layer = read(ROOT / "tests/MoS2.xsf")
    layer.cell[2] = (0, 0, 30)
    layer.center(axis=2)
    layer.pbc = True
    result, _, _ = calculate(layer, LOCAL / "monolayer", minimize=True, monolayer=True)
    result["a_lammps"] = float(np.linalg.norm(layer.cell[0]) * result["in_plane_scale"])
    return read(LOCAL / "monolayer/final.extxyz"), result


def relax_fixed_cell(atoms, directory, *, monolayer, z_only):
    prism = write_data(atoms, directory, monolayer)
    commands = setup_commands(monolayer)
    if z_only:
        commands += "fix freeze all setforce 0.0 0.0 NULL\n"
    commands += """min_style fire
min_modify norm max
minimize 0.0 1.0e-6 100000 100000
write_dump all custom final.dump id type x y z fx fy fz modify sort id format float %.17g
"""
    run_lammps(directory, commands)
    criterion = stopping_criteria(directory)[-1]
    if criterion != "force tolerance":
        raise RuntimeError(f"LAMMPS FIRE stopped at {criterion}: {directory}")
    dump = np.loadtxt(directory / "final.dump", skiprows=9)
    final = atoms.copy()
    final.positions = prism.vector_to_ase(dump[:, 2:5])
    return final


def small_cells():
    layer, stats = relaxed_monolayer()
    stackings = {}
    for label, atoms in high_symmetry_bilayers(layer, 6.5, 30).items():
        directory = LOCAL / "stackings" / label
        final = relax_fixed_cell(atoms, directory, monolayer=False, z_only=True)
        result, _, _ = calculate(final, directory / "final")
        stackings[label] = {"d": final.positions[3, 2] - final.positions[0, 2],
                            "energy": result["pe"]}
    e_rhh = stackings["Rhh"]["energy"]
    for stacking in stackings.values():
        stacking["energy"] -= e_rhh

    e = 0.01
    stresses = {}
    for mode, axes in (("biaxial", [1, 1, 0]), ("shear", [1, -1, 0])):
        for sign in (-1, 1):
            directory = LOCAL / "moduli" / f"{mode}_{sign}"
            atoms = layer.copy()
            deformation = np.eye(3) + np.diag(np.asarray(axes) * sign * e)
            atoms.set_cell(layer.cell @ deformation, scale_atoms=True)
            final = relax_fixed_cell(atoms, directory, monolayer=True, z_only=False)
            _, _, stress = calculate(final, directory / "final", monolayer=True)
            stresses[mode, sign] = stress * final.cell[2, 2]
    biaxial = stresses["biaxial", 1] - stresses["biaxial", -1]
    shear = stresses["shear", 1] - stresses["shear", -1]
    report = {
        "a": stats["a_lammps"], "stackings": stackings,
        "biaxial_modulus": (biaxial[0, 0] + biaxial[1, 1]) / (4 * e),
        "shear_modulus": (shear[0, 0] - shear[1, 1]) / (8 * e),
    }
    (LOCAL / "small_cells.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def smoke():
    layer, _ = relaxed_monolayer()
    atoms = sample_aligned_bilayers(
        layer, 1, 3, "R", (6.5, 6.5), 30, sigma_atom=0,
    )[0]
    atoms.rotate(23, "z", rotate_cell=True)
    stats, forces, stress = calculate(atoms, LOCAL / "checks/base")
    h = 1e-5
    displaced = []
    for sign in (-1, 1):
        perturbed = atoms.copy()
        perturbed.positions[0, 0] += sign * h
        result, _, _ = calculate(perturbed, LOCAL / f"checks/displacement_{sign}")
        displaced.append(result["pe"])
    numeric_force = -(displaced[1] - displaced[0]) / (2 * h)
    np.testing.assert_allclose(forces[0, 0], numeric_force, atol=1e-5, rtol=1e-4)
    strained = []
    for sign in (-1, 1):
        perturbed = atoms.copy()
        deformation = np.diag([1 + sign * h, 1, 1])
        perturbed.set_cell(atoms.cell @ deformation, scale_atoms=True)
        result, _, _ = calculate(perturbed, LOCAL / f"checks/strain_{sign}")
        strained.append(result["pe"])
    numeric_stress = (strained[1] - strained[0]) / (2 * h * atoms.get_volume())
    np.testing.assert_allclose(stress[0, 0], numeric_stress, atol=1e-7, rtol=1e-4)
    report = {
        "energy": stats["pe"], "force_x": float(forces[0, 0]),
        "finite_difference_force_x": float(numeric_force),
        "stress_xx": float(stress[0, 0]), "finite_difference_stress_xx": float(numeric_stress),
        "rotation_degrees": 23, "displacement_A": h, "strain": h,
    }
    (LOCAL / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def attach_labels(atoms, stats, forces, stress):
    atoms.info.update(REF_energy=stats["pe"], REF_stress=stress)
    atoms.new_array("REF_forces", forces)


def switching_angles(atoms, monolayer, cos_theta0):
    species = np.asarray(atoms.get_chemical_symbols())
    atom_layer = np.zeros(len(atoms), dtype=int)
    if not monolayer:
        _, top = _split_layers(atoms, "Mo", "S")
        atom_layer[top] = 1
    i, j, vectors = neighbor_list("ijD", atoms, 3.159)
    same_layer_bonds = (atom_layer[i] == atom_layer[j]) & (species[i] != species[j])
    bonds = [[] for _ in atoms]
    for center, vector in zip(i[same_layer_bonds], vectors[same_layer_bonds]):
        bonds[center].append(vector / np.linalg.norm(vector))

    switching = total = 0
    for symbol, neighbors in zip(species, bonds):
        for left, right in combinations(neighbors, 2):
            delta = abs(left @ right - cos_theta0[symbol])
            switching += int(0.25 < delta < 0.35)
            total += 1
    return switching, total


def labels():
    label_directory = LOCAL / "labels"
    if label_directory.exists():
        shutil.rmtree(label_directory)
    for name in ("sampled", "labeled", "train", "valid"):
        (OUT / f"{name}.extxyz").unlink(missing_ok=True)

    unlabeled = OUT / "unlabeled"
    frames = {name: read(unlabeled / f"{name}.extxyz", index=":")
              for name in ("train_512", "valid", "rigid")}
    subset_ids = {a.info["sample_id"] for a in read(unlabeled / "train_128.extxyz", index=":")}
    ids = [{a.info["sample_id"] for a in dataset} for dataset in frames.values()]
    assert all(left.isdisjoint(right) for left, right in combinations(ids, 2))
    assert subset_ids <= ids[0]
    frames["train_128"] = [atoms for atoms in frames["train_512"]
                            if atoms.info["sample_id"] in subset_ids]
    counts = {name: len(dataset) for name, dataset in frames.items()}
    assert counts == {"train_512": 640, "train_128": 160, "valid": 64, "rigid": 128}, counts

    cos_theta0 = {}
    triplets = {("Mo", "S", "S"), ("S", "Mo", "Mo")}
    for line in (POTENTIALS / "tmd.sw").read_text().splitlines():
        fields = line.split()
        if tuple(fields[:3]) in triplets:
            cos_theta0[fields[0]] = float(fields[8])

    angle_counts = {}
    started = perf_counter()
    for name in ("train_512", "valid", "rigid"):
        for atoms in frames[name]:
            sample_id = atoms.info["sample_id"]
            mono = atoms.info["stratum"].startswith("mono_")
            stats, forces, stress = calculate(
                atoms, label_directory / f"{sample_id:04d}", monolayer=mono,
            )
            attach_labels(atoms, stats, forces, stress)
            angle_counts[sample_id] = switching_angles(atoms, mono, cos_theta0)
        print(f"Labeled {name} in {perf_counter() - started:.1f} s", flush=True)

    paths = []
    by_dataset = {}
    for name, dataset in frames.items():
        path = OUT / f"{name}.extxyz"
        write(path, dataset)
        paths.append(path)
        by_stratum = {}
        for atoms in dataset:
            by_stratum.setdefault(atoms.info["stratum"], []).append(atoms)
        by_dataset[name] = {}
        for stratum, group in by_stratum.items():
            forces = np.concatenate([a.arrays["REF_forces"] for a in group])
            switching, total = np.sum([angle_counts[a.info["sample_id"]] for a in group], axis=0)
            by_dataset[name][stratum] = {
                "force_rms": float(np.sqrt(np.mean(forces**2))),
                "switching_angles": int(switching), "total_angles": int(total),
            }
    validate_dataset(paths)
    (LOCAL / "dataset.json").write_text(json.dumps({
        "frame_counts": counts, "strata": by_dataset,
    }, indent=2) + "\n")


def references():
    test_frames = read(OUT / "rigid.extxyz", index=":")
    layer, result = relaxed_monolayer()
    report = {
        "minimization": {"style": "cg", "line": "forcezero", "etol": 0.0,
                         "ftol": 1e-4, "maxiter": 100000, "maxeval": 100000,
                         "rounds": 6, "box_relax": "x 0.0 y 0.0 couple xy",
                         "final_stage": {"style": "fire", "norm": "max", "etol": 0.0,
                                         "ftol": 1e-5, "maxiter": 200000, "maxeval": 200000}},
        "monolayer": result, "test_cells": {},
    }
    (LOCAL / "references.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Monolayer a={result['a_lammps']:.10f} A", flush=True)
    generator = HSGenerator(layer, layer, tolerance=1e-6, angle_tol=0.01, max_atoms=6000)
    for label, angle, count in [("R_21.79", 21.79, 42)] + TEST_CELLS:
        atoms, strain, effective = generator.gen_supercell(angle, super_z=30, delta_z=6.5)
        assert len(atoms) == count and abs(strain) < 1e-9
        directory = LOCAL / "test-cells" / label
        start, start_forces, start_stress = calculate(atoms, directory / "start")
        write(directory / "start.extxyz", atoms, format="extxyz")
        result, forces, stress = calculate(atoms, directory, minimize=True)
        if label != "R_21.79":
            final = read(directory / "final.extxyz")
            for stage, structure, stats, frame_forces, frame_stress in (
                ("start", atoms, start, start_forces, start_stress),
                ("final", final, result, forces, stress),
            ):
                structure.info["config_type"] = f"{stage}_{label}"
                attach_labels(structure, stats, frame_forces, frame_stress)
                test_frames.append(structure)
        result.update(n_atoms=len(atoms), theta_eff=float(effective))
        report["test_cells"][label] = result
        (LOCAL / "references.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{label}: {count} atoms, fmax={result['max_force']:.3g}, "
              f"Pxy_mean={result['mean_in_plane_pressure_bar']:.3g} bar, "
              f"stop={result['stopping_criteria'][-1]}, "
              f"force_check={result['force_threshold_passed']}, {result['seconds']:.1f} s", flush=True)
    assert len(test_frames) == 128 + 2 * len(TEST_CELLS)
    write(OUT / "test.extxyz", test_frames)
    validate_dataset([OUT / "test.extxyz"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("smoke", "small-cells", "labels", "references"))
    parser.add_argument("--lmp", type=Path, required=True, help="path to the LAMMPS executable")
    args = parser.parse_args()
    lmp = args.lmp.resolve()
    {"smoke": smoke, "small-cells": small_cells, "labels": labels,
     "references": references}[args.task]()
