"""ASE FIRE relaxation with optional in-plane cell optimization."""

from ase.filters import FrechetCellFilter
from ase.optimize import FIRE


class InPlaneIsotropicCellFilter(FrechetCellFilter):
    """Allow equal in-plane scaling with a fixed out-of-plane cell.

    The xx and yy cell forces are replaced by their mean. ASE's
    hydrostatic_strain=True is not equivalent: ASE 3.24.0 averages xx, yy
    and zz before applying the mask, which mixes out-of-plane stress into
    the in-plane cell forces.
    """

    def __init__(self, atoms):
        super().__init__(atoms, mask=[1, 1, 0, 0, 0, 0])

    def get_forces(self, **kwargs):
        forces = super().get_forces(**kwargs)
        cell_forces = forces[len(self.atoms):]
        mean = (cell_forces[0, 0] + cell_forces[1, 1]) / 2
        cell_forces[0, 0] = cell_forces[1, 1] = mean
        return forces


def relax(atoms, calculator, *, steps, fmax=1e-3, cell=True, trajectory=None):
    """Relax a copy, returning it with its calculator; raise on nonconvergence."""
    atoms = atoms.copy()
    atoms.calc = calculator
    target = InPlaneIsotropicCellFilter(atoms) if cell else atoms
    optimizer = FIRE(target, trajectory=None if trajectory is None else str(trajectory), logfile=None)
    if not optimizer.run(fmax=fmax, steps=steps):
        raise RuntimeError(f"Relaxation did not reach fmax={fmax} in {steps} steps")
    return atoms
