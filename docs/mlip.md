# MLIP relaxation and fine-tuning

## 1. MACE

The first zero-shot run downloads the upstream MACE foundation checkpoint.
The MLIP calculators support `cpu` and `cuda`. The zero-shot baseline is
MACE-MP-0 medium plus D3(BJ), using `mace_mp(model="medium", dispersion=True)`.
Fine-tuned models use double precision without additional D3.

## 2. Zero-shot relaxation and analysis

This complete example runs on CPU. It writes structures, the FIRE trajectory
and figures under `results/mlip-example/`. The optimizer
raises if it exhausts the step budget before convergence.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from twistHS.lib.algorithm import HSGenerator
from twistHS.lib.analysis import interlayer_distance_map, stacking_map
from twistHS.mlip.calculators import zero_shot_calculator
from twistHS.mlip.relax import relax

out = Path("results/mlip-example")
out.mkdir(parents=True, exist_ok=True)
u = read("tests/MoS2.xsf")
generator = HSGenerator(u, u, tolerance=1e-6, angle_tol=0.01)
atoms, strain, theta_eff = generator.gen_supercell(21.79, super_z=30, delta_z=6.5)
write(out / "R_21.79_start.extxyz", atoms)
calculator = zero_shot_calculator("cpu")
final = relax(atoms, calculator, steps=5000, trajectory=out / "R_21.79_zero-shot.traj")
write(out / "R_21.79_zero-shot.extxyz", final)

xy, d = interlayer_distance_map(final, "Mo", "S")
mapping = stacking_map(final, "Mo", "S")
fig, ax = plt.subplots()
points = ax.scatter(xy[:, 0], xy[:, 1], c=d)
fig.colorbar(points, ax=ax, label="Metal-plane separation (Å)")
ax.set(xlabel="x (Å)", ylabel="y (Å)", aspect="equal")
fig.savefig(out / "interlayer-distance.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots()
for label in np.unique(mapping.labels):
    sites = mapping.xy[mapping.labels == label]
    ax.scatter(sites[:, 0], sites[:, 1], label=label)
ax.set(xlabel="x (Å)", ylabel="y (Å)", aspect="equal")
ax.legend()
fig.savefig(out / "stacking.png", bbox_inches="tight")
plt.close(fig)
```

## 3. Analysis functions

These functions apply to 1H MX₂ bilayers, either homo- or heterobilayers.
`metal` and `chalcogen` each accept one element symbol or a sequence, for
example `metal=["Mo", "W"]`, `chalcogen=["S", "Se"]`. Layers split at the
largest adjacent z gap; each must have MX₂ stoichiometry. Keep each layer
contiguous along z, away from the periodic z boundary, before analysis.

`interlayer_distance_map(atoms, metal, chalcogen)` returns wrapped top-metal
xy coordinates in atom order and top-minus-bottom metal-plane separation in Å.
The bottom metal plane is linearly interpolated with periodic xy images.

`stacking_map(atoms, metal, chalcogen, boundary=None)` returns a frozen
`StackingMap` with `xy`, `labels`, `fractions` and `family`. Family R has
relative orientation within 30° of parallel; H is within 30° of antiparallel.
The top metal's nearest bottom-layer site determines its label:

| Bottom site | R label | H label |
|---|---|---|
| M (metal) | `Rhh` | `HhX` |
| X (chalcogen projection) | `RhX` | `Hhh` |
| hollow | `RhM` | `HhM` |

Fractions count top-metal sites as equal-area samples. With `boundary=None`,
every site belongs to the nearest of M, X and hollow. With `boundary=0.15*a`
(Å, using the reference monolayer lattice constant), only sites within that
radius count toward a registry; the rest are `boundary`. Include `boundary`
when summing disc fractions.

## 4. Relaxation settings

`relax(atoms, calculator, *, steps, fmax=1e-3, cell=True, trajectory=None)`
copies its input and returns the relaxed `Atoms` with its calculator attached.
With `cell=True`, FIRE scales both in-plane cell vectors by one factor and
keeps the out-of-plane vector fixed.

This matches the cell degrees of freedom and zero mean in-plane stress target
of LAMMPS `fix stress all box/relax x 0.0 y 0.0 couple xy`.

The default force threshold is 1e-3 eV/Å. The cell filter also applies
FIRE's convergence criterion to its generalized cell forces. Use `cell=False` for fixed-cell relaxation. To continue a run,
read the last trajectory frame and call `relax` again with a new step budget
and a new trajectory path.

## 5. Load your own checkpoint

Use a compatible MACE checkpoint with `finetuned_calculator` and pass the
calculator to `relax`. The calculator loads the model in double precision
without an added D3 correction.

```python
from twistHS.mlip.calculators import finetuned_calculator
from twistHS.mlip.relax import relax

calculator = finetuned_calculator("my-model.model", device="cpu")
final = relax(atoms, calculator, steps=5000)
```

Here `atoms` is the starting structure from the zero-shot example or your own
ASE `Atoms`.

## 6. Sample structures and fine-tune with your own labels

The sampling functions generate unlabeled 3×3 cells. Give them the monolayer
geometry, a random seed and the cell height. For a bilayer, select the R or H
family and give the metal-plane separation range in Å. The following example
uses the repository fixture and illustrative sampling parameters.

```python
from ase.io import read
from twistHS.lib.sampling import sample_aligned_bilayers, sample_monolayers

layer = read("tests/MoS2.xsf")
bilayers = sample_aligned_bilayers(
    layer, 4, 42, "R", d_range=(6.0, 7.0), super_z=30, sigma_atom=0.1, strain=0.01,
)
monolayers = sample_monolayers(layer, 4, 43, super_z=30, sigma_atom=0.1, strain=0.01)
```

Use a consistent reference method to label your structures, preserve atom
order and write separate training and validation extxyz files. Each frame
uses the following keys.

| Key | ASE location | Units and shape |
|---|---|---|
| `REF_energy` | `atoms.info` | eV, scalar total energy |
| `REF_forces` | `atoms.arrays` | eV/Å, N×3 rows in atom order |
| `REF_stress` | `atoms.info` | eV/Å³, 3×3 tensor with tensile stress positive |

Check that every frame contains the expected keys.

```python
from twistHS.mlip.finetune import validate_dataset

validate_dataset(["train.extxyz", "validation.extxyz"])
```

For this single-head MACE example, supply `train.extxyz`,
`validation.extxyz` and `e0s.json`. The JSON maps atomic numbers to
isolated-atom energies in eV from your labeling method.

```sh
mace_run_train --name my_model --work_dir results/mlip-example/training \
  --foundation_model medium --multiheads_finetuning false --default_dtype float64 \
  --train_file train.extxyz --valid_file validation.extxyz --E0s e0s.json \
  --energy_key REF_energy --forces_key REF_forces --stress_key REF_stress \
  --compute_stress true --loss stress --device cpu
```

The [complete MoS₂ case study](../experiments/mlip/mos2/README.md) tests the
small-cell sampling recipe with classical LAMMPS labels before DFT labeling,
and serves as a template for other MX₂ systems.
