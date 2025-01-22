"""Shared scan, visualization, and export for the CLI and GUI."""

from decimal import Decimal, InvalidOperation
from math import ceil, isfinite
from pathlib import Path
from typing import NamedTuple

from ase import Atoms
from ase.io import read
from ase.io.formats import UnknownFileTypeError
from ase.visualize.plot import plot_atoms

from twistHS.lib.algorithm import HSGenerator

NO_SUPERCELLS = "No commensurate supercell satisfies the specified constraints."
STRAIN_LABEL = "Maximum strain (%)"
MOIRE_HEADERS = (
    "Requested twist angle (°)", "Geometric order", "Lower-layer indices",
    "Upper-layer indices", "Moiré wave vector (Å⁻¹)", "Period (Å)",
    "Azimuth (°)", "Multiplicity",
)
INPUT_ERRORS = (ValueError, OSError, ImportError, UnknownFileTypeError)


class Supercell(NamedTuple):
    requested_angle: float
    atoms: Atoms
    strain: float

    @property
    def effective_angle(self):
        return self.atoms.info["theta_eff"]

    @property
    def label(self):
        return (f"Requested twist {self.requested_angle:g}° · "
                f"Effective twist {self.effective_angle:.4f}° · {len(self.atoms)} atoms")


def angle_range(start, stop, step):
    if any(value is None for value in (start, stop, step)):
        raise ValueError("Specify the start, stop, and step of the twist-angle interval.")
    try:
        start, stop, step = (Decimal(str(value)) for value in (start, stop, step))
    except InvalidOperation as error:
        raise ValueError("Twist-angle interval values must be numbers.") from error
    if not all(value.is_finite() for value in (start, stop, step)):
        raise ValueError("Twist-angle interval values must be finite.")
    if step == 0 or start == stop or (stop > start) != (step > 0):
        raise ValueError("The twist-angle interval must be nonempty and the step must point toward the stop angle.")
    return [float(start + i * step) for i in range(ceil((stop - start) / step))]


def positive_integer(value, label):
    if value is None or not isfinite(value) or value < 1 or int(value) != value:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def run_scan(bottom, top, angles, z, d, *, max_order=None, **search_options):
    if not all(isfinite(angle) for angle in angles):
        raise ValueError("Requested twist angles must be finite.")
    if z is None or not isfinite(z) or z <= 0:
        raise ValueError("The out-of-plane lattice parameter must be finite and positive.")
    if d is None or not isfinite(d):
        raise ValueError("The interlayer distance must be finite.")
    if max_order is not None:
        max_order = positive_integer(max_order, "The reciprocal cutoff for moiré component analysis")
    generator = HSGenerator(read(bottom), read(top), **search_options)
    rows = []
    if max_order is not None:
        for angle in angles:
            for component in generator.moire_components(angle, max_order):
                qx, qy = component.q
                rows.append((
                    str(angle), f"{component.order:.8g}", str(component.n), str(component.m),
                    f"({qx:.8g}, {qy:.8g})", f"{component.period:.8g}",
                    f"{component.angle_deg:.8g}", str(component.multiplicity),
                ))
    return [Supercell(*entry) for entry in generator.scan(angles, z, d)], rows


def plot_scan(entries, ax):
    scatter = ax.scatter(
        [entry.effective_angle for entry in entries],
        [len(entry.atoms) for entry in entries],
        c=[100 * entry.strain for entry in entries], picker=True,
    )
    ax.figure.colorbar(scatter, ax=ax, label=STRAIN_LABEL)
    ax.set(xlabel="Effective twist angle (°)", ylabel="Atoms per supercell",
           title="Commensurate moiré supercells")
    return scatter


def plot_structure(entry, ax):
    atoms = entry.atoms.copy()
    atoms.wrap()
    plot_atoms(atoms, ax, scale=2, radii=0.3)
    ax.set_title(f"{entry.label}\n{STRAIN_LABEL.removesuffix(' (%)')}: {100 * entry.strain:.2f}%")
    ax.set_axis_off()


def export_structure(entry, filename=None, outformat=None, *, directory=None):
    if outformat and not filename:
        raise ValueError("Specify a filename when selecting an ASE output format.")
    path = Path(filename or f"structure_{entry.requested_angle:.2f}.xsf")
    if directory is not None:
        if path.name != filename and filename:
            raise ValueError("Specify a download filename without a directory.")
        path = Path(directory) / path
    entry.atoms.write(path, format=outformat or None)
    return path
