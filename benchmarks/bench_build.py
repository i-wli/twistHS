"""Compare the original and vectorized atom builders on twisted bilayer graphene."""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from statistics import median
from time import perf_counter

import numpy as np
from ase import Atoms
from ase.build import graphene

from twistHS.lib.func import (
    get_supercell_atoms_pos,
    get_supercell_vecs,
    lattice_points_in_supercell,
)
from twistHS.lib.moire import supercell_matrices

ANGLES = (5, 2, 1, 0.5, 0.25, 0.1)
METHODS = ("legacy", "vectorized")


def legacy_build(vectors, transform, unit):
    """Reproduce the original copy-and-extend implementation."""
    matrix = np.eye(3, dtype=int)
    matrix[:2, :2] = transform
    lattice_points = lattice_points_in_supercell(matrix) @ vectors
    result = Atoms(cell=vectors, pbc=unit.pbc)
    for point in lattice_points:
        shifted = unit.copy()
        shifted.positions += point
        result.extend(shifted)
    return result


def measure(method, angle):
    bottom = graphene(vacuum=15)
    top = bottom.copy()
    top.rotate(angle, "z", rotate_cell=True)
    layers = (bottom, top)
    transforms = supercell_matrices(bottom.cell[:2, :2], top.cell[:2, :2])
    cells = [
        get_supercell_vecs(transform, layer.cell, 30)
        for transform, layer in zip(transforms, layers)
    ]
    build = legacy_build if method == "legacy" else get_supercell_atoms_pos

    start = perf_counter()
    structures = [
        build(cell, transform, layer)
        for cell, transform, layer in zip(cells, transforms, layers)
    ]
    seconds = perf_counter() - start
    peak_rss = getrusage(RUSAGE_SELF).ru_maxrss
    peak_rss_mib = peak_rss / (2**20 if sys.platform == "darwin" else 2**10)
    return sum(map(len, structures)), seconds, peak_rss_mib


def plot(rows, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for method, label in zip(METHODS, ("Legacy loop", "Vectorized")):
        trials = [row for row in rows if row["method"] == method]
        counts = sorted({row["atoms"] for row in trials})
        for ax, metric in zip(axes, ("seconds", "peak_rss_mib")):
            values = [
                median(row[metric] for row in trials if row["atoms"] == count)
                for count in counts
            ]
            ax.plot(counts, values, marker="o", label=label)

    for ax, label in zip(axes, ("Build time (s)", "Peak process RSS (MiB)")):
        ax.set(xscale="log", yscale="log", xlabel="Total bilayer atoms", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Twisted bilayer graphene supercell construction")
    fig.savefig(destination, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", nargs="+", type=float, default=ANGLES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (
        (args.out_dir / "build.csv").open("w", newline="") as csv_file,
        ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor,
    ):
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["angle_deg", "atoms", "method", "trial", "seconds", "peak_rss_mib"],
            lineterminator="\n",
        )
        writer.writeheader()
        for angle in args.angles:
            for trial in range(1, args.repeats + 1):
                for method in METHODS:
                    atoms, seconds, peak_rss_mib = executor.submit(measure, method, angle).result()
                    row = dict(
                        angle_deg=angle, atoms=atoms, method=method, trial=trial,
                        seconds=seconds, peak_rss_mib=peak_rss_mib,
                    )
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(
                        f"{angle:g}°  {atoms:,} atoms  {method} #{trial}  "
                        f"{seconds:.4g} s  {peak_rss_mib:.1f} MiB",
                        flush=True,
                    )
    plot(rows, args.out_dir / "build.png")


if __name__ == "__main__":
    main()
