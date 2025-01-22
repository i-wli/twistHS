"""Time coincidence search and full supercell generation on two fixed bilayers."""

import argparse
import csv
import json
import os
import platform
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from math import isclose
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

from ase.build import graphene

from twistHS.lib.algorithm import HSGenerator
from twistHS.lib.coincidence import find_coincidence

SYSTEMS = ("TBG", "graphene-hBN")
BACKENDS = ("numpy", "jax-cpu", "jax-gpu")
ENTRYPOINTS = ("find_coincidence", "gen_supercell")
FIELDS = (
    "case", "system", "angle_deg", "tolerance_pct", "angle_tol_deg", "max_atoms",
    "backend", "cores", "entrypoint", "phase", "trial", "seconds",
    "status", "atoms", "strain_pct", "theta_eff_deg", "bottom_matrix", "top_matrix",
)


@dataclass(frozen=True)
class Case:
    system: str
    angle: float
    tolerance: float = 1.0
    angle_tol: float = 0.1
    max_atoms: int = 12000

    @property
    def name(self):
        return f"{self.system}-{self.angle:g}"


CASES = {
    case.name: case
    for system in SYSTEMS
    for case in (
        *(Case(system, angle) for angle in (5, 2, 1, 0.5, 0.25)),
        Case(system, 0.1, tolerance=0.05, angle_tol=0.01),
    )
}


def prepare(case, entrypoint, backend):
    bottom = graphene(a=2.460, vacuum=7.5)
    top = bottom.copy() if case.system == "TBG" else graphene("BN", a=2.504, vacuum=7.5)
    limits = dict(tolerance=case.tolerance, angle_tol=case.angle_tol,
                  max_atoms=case.max_atoms, backend=backend)
    if entrypoint == "find_coincidence":
        top.rotate(case.angle, "z", rotate_cell=True)
        return partial(find_coincidence, bottom.cell[:2, :2], top.cell[:2, :2],
                       (len(bottom), len(top)), **limits)
    generator = HSGenerator(bottom, top, **limits)
    return partial(generator.gen_supercell, case.angle, super_z=30, delta_z=7)


def measure(call, entrypoint, angle):
    start = perf_counter()
    try:
        result = call()
    except ValueError as error:
        seconds = perf_counter() - start
        if (entrypoint != "gen_supercell"
                or str(error) != f"No coincidence cell at {angle:g}° within the search limits"):
            raise
        result = None
    else:
        seconds = perf_counter() - start
    if result is None:
        return dict(seconds=seconds, status="no_match")
    if entrypoint == "find_coincidence":
        return dict(seconds=seconds, status="match", atoms=result.atom_count,
                    strain_pct=100 * result.strain, theta_eff_deg=angle + result.angle_offset,
                    bottom_matrix=json.dumps(result.bottom.tolist()),
                    top_matrix=json.dumps(result.top.tolist()))
    atoms, strain, theta_eff = result
    return dict(seconds=seconds, status="match", atoms=len(atoms),
                strain_pct=100 * strain, theta_eff_deg=theta_eff)


def configure(backend, cores):
    if hasattr(os, "sched_getaffinity"):
        bound_cores = len(os.sched_getaffinity(0))
        if bound_cores != cores:
            raise RuntimeError(f"Expected {cores} bound CPUs, got {bound_cores}")
    packages = {name: version(name) for name in ("numpy", "scipy", "ase")}
    devices = []
    if backend != "numpy":
        target = "cpu" if backend == "jax-cpu" else "cuda"
        os.environ["JAX_PLATFORMS"] = target
        import jax

        jax.config.update("jax_enable_compilation_cache", False)
        active = jax.devices()
        expected = "cpu" if target == "cpu" else "gpu"
        if len(active) != 1 or active[0].platform != expected:
            raise RuntimeError(f"Expected one JAX {expected} device, got {active}")
        devices = [device.device_kind for device in active]
        packages.update(jax=jax.__version__, jaxlib=version("jaxlib"))
    print(json.dumps(dict(backend=backend, cores=cores, devices=devices,
                          python=platform.python_version(),
                          packages=packages)), flush=True)
    return "numpy" if backend == "numpy" else "jax"


def run(args):
    backend = configure(args.backend, args.cores)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, name in enumerate(args.cases):
            case = CASES[name]
            call = prepare(case, args.entrypoint, backend)
            common = dict(case=name, system=case.system, angle_deg=case.angle,
                          tolerance_pct=case.tolerance, angle_tol_deg=case.angle_tol,
                          max_atoms=case.max_atoms, backend=args.backend, cores=args.cores,
                          entrypoint=args.entrypoint)
            if index == 0:
                writer.writerow(dict(common, phase="first", trial=0,
                                     **measure(call, args.entrypoint, case.angle)))
            measure(call, args.entrypoint, case.angle)
            for trial in range(1, args.repeats + 1):
                writer.writerow(dict(common, phase="warm", trial=trial,
                                     **measure(call, args.entrypoint, case.angle)))
                stream.flush()
            print(f"{args.backend} {args.cores}c {args.entrypoint} {name} done", flush=True)


def profile(args):
    import nvtx

    backend = configure("jax-gpu", args.cores)
    case = CASES[f"{args.system}-5"]
    call = prepare(case, "find_coincidence", backend)
    call()
    range_id = nvtx.start_range("twist_search", domain="twistHS")
    try:
        match = call()
    finally:
        nvtx.end_range(range_id)
    if match is None:
        raise RuntimeError(f"No match for profiler case {case.name}")


def check_match(row, reference):
    if (row["status"] != reference["status"] or row["atoms"] != reference["atoms"]
            or (row["status"] == "match" and any(
                not isclose(float(row[key]), float(reference[key]), rel_tol=0, abs_tol=1e-9)
                for key in ("strain_pct", "theta_eff_deg")
            ))):
        raise ValueError(f"Result mismatch: {row} vs {reference}")


def summarize(args):
    warm = defaultdict(list)
    first = {}
    for path in args.inputs:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["case"], row["backend"], int(row["cores"]), row["entrypoint"])
                if row["phase"] == "first":
                    first[key] = row
                else:
                    warm[key].append(row)

    medians = {key: median(float(row["seconds"]) * 1000 for row in rows)
               for key, rows in warm.items()}
    pairs = sorted({key[:3] for key in warm}, key=lambda key: (
        list(CASES).index(key[0]), BACKENDS.index(key[1]), key[2],
    ))
    summary = []
    for name, backend, cores in pairs:
        reference = warm[name, "numpy", 1, "find_coincidence"][0]
        result = {key: reference[key] for key in (
            "case", "system", "angle_deg", "tolerance_pct", "angle_tol_deg", "max_atoms",
            "status", "atoms", "strain_pct", "theta_eff_deg",
        )}
        result.update(backend=backend, cores=cores)
        for label, entrypoint in zip(("search", "gen"), ENTRYPOINTS):
            key = (name, backend, cores, entrypoint)
            rows = warm[key]
            for row in rows + ([first[key]] if key in first else []):
                check_match(row, reference)
            times = [float(row["seconds"]) * 1000 for row in rows]
            q1, _, q3 = quantiles(times, n=4, method="inclusive") if len(times) > 1 else times * 3
            result[f"{label}_first_ms"] = float(first[key]["seconds"]) * 1000 if key in first else ""
            result[f"{label}_median_ms"] = medians[key]
            result[f"{label}_iqr_ms"] = q3 - q1
            baseline = medians[name, "numpy", 1, entrypoint]
            success = reference["status"] == "match"
            result[f"{label}_speedup_vs_numpy"] = baseline / medians[key] if success else ""
            if label == "gen":
                result["gen_saved_ms_vs_numpy"] = baseline - medians[key] if success else ""
                result["gen_saved_pct_vs_numpy"] = 100 * (1 - medians[key] / baseline) if success else ""
            cpu_key = (name, "jax-cpu", 4, entrypoint)
            result[f"{label}_gpu_speedup_vs_jax_cpu4"] = (
                medians[cpu_key] / medians[key] if success and backend == "jax-gpu" else ""
            )
            if label == "gen":
                result["gen_gpu_saved_ms_vs_jax_cpu4"] = (
                    medians[cpu_key] - medians[key] if success and backend == "jax-gpu" else ""
                )
        matrices = ("bottom_matrix", "top_matrix")
        search_rows = warm[name, backend, cores, "find_coincidence"]
        result.update({key: search_rows[0][key] for key in matrices})
        result["matrices_differ_from_numpy"] = (
            any(row[key] != reference[key] for row in search_rows for key in matrices)
            if success else ""
        )
        summary.append(result)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


def plot(args):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with args.input.open(newline="") as stream:
        matches = {
            (row["case"], row["backend"], int(row["cores"])): row
            for row in csv.DictReader(stream) if row["status"] == "match"
        }

    configs = (("numpy", 1), ("jax-cpu", 1), ("jax-gpu", 4))
    labels = ("NumPy · 1 CPU core", "JAX CPU · 1 CPU core", "JAX A100 · 4 host CPU cores")
    prefix, title = {
        "find_coincidence": ("search", "Coincidence search"),
        "gen_supercell": ("gen", "Supercell generation"),
    }[args.entrypoint]
    panels = (
        (CASES["TBG-5"], "Twisted bilayer graphene", 1, "ms", 1),
        (CASES["graphene-hBN-0.5"], "Graphene/hBN", 1000, "s", 2),
    )

    fig = Figure(figsize=(11, 3.2), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    fig.suptitle(f"{title} after warm-up", fontsize=16, fontweight="semibold")
    axes = fig.subplots(1, 2, sharey=True)
    positions = range(len(configs))
    for ax, (case, system, scale, unit, precision) in zip(axes, panels):
        rows = [matches[case.name, backend, cores] for backend, cores in configs]
        values = [float(row[f"{prefix}_median_ms"]) / scale for row in rows]
        atoms = int(rows[0]["atoms"])
        ax.set_title(
            f"{system}\n{case.angle:g}° · {atoms:,} atoms",
            loc="left", fontsize=12, pad=12,
        )
        ax.barh(positions, values, height=0.54, color=("#84909e", "#3772a5", "#007d71"))
        for y, value in zip(positions, values):
            speedup = values[0] / value
            ax.annotate(f"{value:.{precision}f} {unit}  ({speedup:.2f}×)", (value, y),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)
        ax.set(xlim=(0, max(values) * 1.43), xlabel=f"Median elapsed time ({unit})")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e9ed")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b5bec7")
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(positions, labels, fontsize=10)
    axes[0].invert_yaxis()
    fig.savefig(args.output, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    timing = commands.add_parser("run")
    timing.add_argument("--backend", choices=BACKENDS, required=True)
    timing.add_argument("--cores", type=int, required=True)
    timing.add_argument("--entrypoint", choices=ENTRYPOINTS, required=True)
    timing.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    timing.add_argument("--repeats", type=int, default=5)
    timing.add_argument("--output", type=Path, required=True)
    timing.set_defaults(action=run)
    profiling = commands.add_parser("profile")
    profiling.add_argument("--system", choices=SYSTEMS, required=True)
    profiling.add_argument("--cores", type=int, required=True)
    profiling.set_defaults(action=profile)
    report = commands.add_parser("summarize")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("inputs", type=Path, nargs="+")
    report.set_defaults(action=summarize)
    figure = commands.add_parser("plot", help="Plot two warm timing comparisons from a summary CSV")
    figure.add_argument("input", type=Path)
    figure.add_argument("--entrypoint", choices=ENTRYPOINTS, default="gen_supercell")
    figure.add_argument("--output", type=Path, required=True)
    figure.set_defaults(action=plot)
    args = parser.parse_args()
    if args.action is run and args.repeats < 1:
        timing.error("--repeats must be positive")
    args.action(args)


if __name__ == "__main__":
    main()
