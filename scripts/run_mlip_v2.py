"""Run the frozen v2 training and comparison workflow in a two-GPU Slurm job."""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REFERENCE = Path("results/mlip/lammps")
LABELS = ("R_5.09", "R_3.89", "R_3.15", "R_2.45", "R_2.00")
DECISION_LABELS = LABELS[2:]
SEEDS = (1, 2)
SRUN = ("srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=8", "--gres=gpu:1", "--cpu-bind=cores")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run_gpu(args, log, *, check=True):
    env = dict(os.environ, MPLCONFIGDIR=str(log.parent / "matplotlib"))
    with log.open("w") as stream:
        result = subprocess.run(
            [*SRUN, sys.executable, *map(str, args)], env=env, stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    if check and result.returncode:
        raise RuntimeError(f"GPU step exited with {result.returncode}. See {log}")
    return result.returncode


def train_and_diagnose(out, budget, seed):
    name = f"MoS2_SW_KC_{budget}_s{seed}"
    directory = out / name
    directory.mkdir()
    finetune = directory / "finetune"
    run_gpu([
        "-m", "mace.cli.run_train", "--config", "experiments/mlip/finetune.yaml",
        "--name", name, "--seed", seed, "--train_file", f"results/mlip/train_{budget}.extxyz",
        "--work_dir", finetune, "--model_dir", finetune,
    ], directory / "train.log")
    model = finetune / f"{name}_stagetwo.model"
    run_gpu([
        "scripts/small_cell_diagnostics.py", "--monolayer", REFERENCE / "monolayer/final.extxyz",
        "--reference", REFERENCE / "small_cells.json", "--model", model,
        "--device", "cuda", "--steps", 5000, "--out", directory,
    ], directory / "diagnostics.log")
    table = {
        row["quantity"]: {key: float(row[key]) for key in ("reference", "model")}
        for row in read_rows(directory / f"small_cells_{model.stem}.csv")
    }
    energy = table["energy_RhX"]
    relative_error = abs(energy["model"] - energy["reference"]) / abs(energy["reference"])
    diagnostic = {
        "energy_RhX": energy,
        "relative_energy_error": relative_error,
        "pass": relative_error <= 0.03,
        "separations": {label: table[f"d_{label}"] for label in ("Rhh", "RhX", "RhM")},
    }
    return model, table["a"]["model"], diagnostic


def compare(model, a_model, a_lammps):
    directory = model.parent.parent
    results = {}
    for label in LABELS:
        cell = REFERENCE / "test-cells" / label
        destination = directory / label
        destination.mkdir()
        exit_code = run_gpu([
            "scripts/compare_mlip_lammps.py", "--start", cell / "lammps.lmp",
            "--lammps-final", cell / "min.lmp", "--label", label, "--model", model,
            "--device", "cuda", "--steps", 15000, "--fmax", 1e-4,
            "--a-model", a_model, "--a-lammps", a_lammps,
            "--metal", "Mo", "--chalcogen", "S", "--out", destination,
        ], destination / "relax.log", check=False)
        results[label] = {"exit_code": exit_code}
        if exit_code == 0:
            row, = read_rows(destination / "compare.csv")
            results[label].update(row)
    return results


def workflow(out, a_lammps):
    summary = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "budgets": {}, "dft_frames": None}
    path = out / "summary.json"

    def save():
        path.write_text(json.dumps(summary, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for budget, frames in ((512, 640), (128, 160)):
            report = summary["budgets"][str(budget)] = {"status": "training"}
            save()
            training = [pool.submit(train_and_diagnose, out, budget, seed) for seed in SEEDS]
            trained = [task.result() for task in training]
            report["diagnostics"] = {seed: model[2] for seed, model in zip(SEEDS, trained)}
            if not all(model[2]["pass"] for model in trained):
                report["status"] = "energy_gate_failed"
                save()
                break

            report["status"] = "comparing"
            save()
            comparisons = [pool.submit(compare, model, a, a_lammps) for model, a, _ in trained]
            report["comparisons"] = results = {seed: task.result()
                                               for seed, task in zip(SEEDS, comparisons)}
            decision = [rows[label] for rows in results.values() for label in DECISION_LABELS]
            if any(row["exit_code"] != 0 for row in decision):
                report["status"] = "decision_relaxation_failed"
                save()
                break
            if any(row["pass"] != "True" for row in decision):
                report["status"] = "decision_metrics_failed"
                save()
                break
            report["status"] = "passed"
            summary["dft_frames"] = frames
            save()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    a_lammps = json.loads((REFERENCE / "references.json").read_text())["monolayer"]["a_lammps"]
    workflow(out, a_lammps)


if __name__ == "__main__":
    main()
