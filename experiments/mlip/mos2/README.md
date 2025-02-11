# MoS₂ MLIP case study

This case tests the small-cell sampling recipe with inexpensive classical
labels in place of DFT: MACE fine-tuned on small MoS₂ cells should reproduce
local relaxation in larger twisted R-family MoS₂ homobilayers over the
sampled twist-angle range. Both the training labels and the reference
relaxations use classical LAMMPS potentials (SW/mod intralayer, KC
interlayer). H-family structures contribute to sampling and
held-out single-point evaluation. Agreement indicates that the small cells
cover the local environments of the relaxed twisted cells, so the recipe
can then be used with DFT labels.

## Requirements

Use LAMMPS `stable_29Aug2024_update1` built with the MANYBODY and INTERLAYER
packages. From a repository checkout, install twistHS and pin the tested
dependencies in a Python 3.12 environment.

```sh
python -m pip install '.[mlip]'
python -m pip install mace-torch==0.3.9 torch==2.5.1 torch-dftd==0.5.1 ase==3.24.0 numpy==1.26.4
```

`run_validation.py` uses the MACE 0.3.9 stage-two checkpoint
`<name>_stagetwo.model`. Training and validation use `srun` and require a
Slurm allocation with two GPUs and eight CPU cores per GPU.

Before submitting the Slurm job, run this once on a node with network
access and a home directory shared with the compute nodes. It caches the
checkpoint that MACE's `run_train --foundation_model medium` loads.

```sh
python -c "from mace.calculators import mace_mp; mace_mp(model='medium')"
```

Create `experiments/mlip/mos2/potentials/`, download `potentials.zip` from
the Zenodo record and extract `tmd.sw` and `MoWSSe_main.KC` directly into
that directory. The driver uses SW/mod for intralayer interactions and
`kolmogorov/crespi/z` for interlayer interactions with `hybrid/overlay`.
The potential files are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). When using them, cite the
[Zenodo dataset](https://doi.org/10.5281/zenodo.10979042) and
[Li, Brumme and Heine, npj 2D Mater. Appl. 8, 43 (2024)](https://doi.org/10.1038/s41699-024-00477-6).

## Run the case

Run the commands in order from the repository root. Set `LMP` to the path
to your LAMMPS executable. Run the last command inside the Slurm allocation
and pass a new output directory.

`lammps_reference.py` runs the 832 labeling calculations and the CG/FIRE
references sequentially, each as a single LAMMPS process. To use MPI, point
`--lmp` to a wrapper script that launches LAMMPS under `mpirun`.

```sh
case_dir=experiments/mlip/mos2
LMP=/path/to/lmp

python "$case_dir/lammps_reference.py" smoke --lmp "$LMP"
python "$case_dir/lammps_reference.py" small-cells --lmp "$LMP"
python "$case_dir/build_mlip_dataset.py" \
  --monolayer results/mlip/lammps/monolayer/final.extxyz \
  --small-cells results/mlip/lammps/small_cells.json \
  --out results/mlip/unlabeled
python "$case_dir/lammps_reference.py" labels --lmp "$LMP"
python "$case_dir/lammps_reference.py" references --lmp "$LMP"
python "$case_dir/run_validation.py" results/mlip/validation
```

| Step | Main outputs |
| --- | --- |
| `smoke` | `results/mlip/lammps/checks.json` with the force and stress finite-difference checks |
| `small-cells` | `results/mlip/lammps/monolayer/final.extxyz` and `results/mlip/lammps/small_cells.json` with the lattice constant, stacking energies and separations, and elastic moduli |
| `build_mlip_dataset.py` | `recipe.json` and unlabeled `train_512.extxyz`, `train_128.extxyz`, `valid.extxyz` and `rigid.extxyz` in `results/mlip/unlabeled/` |
| `labels` | Labeled versions of the four extxyz files in `results/mlip/` and a label report at `results/mlip/lammps/dataset.json` |
| `references` | `results/mlip/test.extxyz`, `results/mlip/lammps/references.json` and the start, CG and FIRE structures under `results/mlip/lammps/test-cells/` |
| `run_validation.py` | MACE checkpoints, small-cell diagnostics, twisted-cell relaxation trajectories and comparison CSVs, and `summary.json` in the chosen output directory |

## Reference labels and data splits

Build small cells from the reference-relaxed monolayer. Use the relaxed
high-symmetry `STACKINGS` and `MARGIN` in `build_mlip_dataset.py` to set
the sampling range for metal-plane separations. Sample relative lateral
translations, interlayer separation, in-plane strain (`STRAIN`) and mass-scaled displacements
(`STRATA`) in aligned R and H bilayers. Include strained and displaced
monolayers to constrain the intralayer response and unperturbed R
bilayers to sample lateral registry (`REGISTRY_COUNT` and
`REGISTRY_SUBSET_COUNT`). The script sets the cell height with `SUPER_Z`.

Label each frame with a LAMMPS single-point calculation.

Use independently sampled small cells for validation (`VALID`) and
nested training sets (`TRAIN` and `SUBSET`) to test the effect of label
count.

The `references` task builds each entry in `TEST_CELLS` with twistHS's
`HSGenerator.gen_supercell`, as in [the MLIP user guide, §2](../../../docs/mlip.md#2-zero-shot-relaxation-and-analysis).
For held-out energy, force and stress evaluation, `test.extxyz` combines
the rigid R- and H-family frames (`RIGID` in `build_mlip_dataset.py`) with
the start and FIRE endpoints of the five twisted R-family cells. To add
your own twisted cell, put its label, twist angle and expected atom count
in `TEST_CELLS` in `lammps_reference.py` and add its label to `LABELS` in
`run_validation.py`.

## Controls and acceptance

To measure the effect of relaxing the
structure and of refining the reference, use `compare_mlip_lammps.py` with
`--candidate` to compare the unrelaxed `lammps.lmp` and then the CG
endpoint `cg.lmp` with the FIRE endpoint `min.lmp`.

Run the controls from the repository root after `references`. Both
lattice constants come from `monolayer.a_lammps` in `references.json`.
The comparisons append rows to `results/mlip/controls/compare.csv`.

```sh
(
  set -e
  reference=results/mlip/lammps
  a=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["monolayer"]["a_lammps"])' \
    "$reference/references.json")
  labels=$(python -c 'from experiments.mlip.mos2.lammps_reference import TEST_CELLS; print(*(label for label, _, _ in TEST_CELLS), sep="\n")')

  while IFS= read -r label; do
    cell="$reference/test-cells/$label"
    for candidate in lammps.lmp cg.lmp; do
      python experiments/mlip/mos2/compare_mlip_lammps.py \
        --start "$cell/lammps.lmp" --candidate "$cell/$candidate" \
        --lammps-final "$cell/min.lmp" --label "$label" \
        --a-model "$a" --a-lammps "$a" --metal Mo --chalcogen S \
        --out results/mlip/controls
    done
  done <<EOF
$labels
EOF
)
```

Before running twisted-cell relaxations for a training set, compare the
fine-tuned models' RhX–Rhh stacking energy difference with the reference.
For both models, the relative error in this difference must be at most
`ENERGY_GATE` in `run_validation.py`. The small-cell diagnostics also record
stacking separations, the monolayer lattice constant and its elastic moduli.

Compare each converged model relaxation against the FIRE endpoint using
three criteria from `compare_mlip_lammps.py`.

- `RMSD_FRACTION` limits the in-plane displacement RMSD in each layer
  relative to that layer's reference RMS displacement. Measure from the
  common starting structure using periodic images and subtract each
  layer's mean displacement.
- `D_MAE` limits the mean absolute error in metal-plane separation at
  corresponding top-layer metal sites.
- `DISC` limits the maximum absolute error in stacking area fractions.
  Each fraction counts top-layer metal sites within the registry disc
  used by `comparison_metrics`.

Both training seeds (`SEEDS`) must pass the energy gate and all three
criteria on every cell in `DECISION_LABELS`. `run_validation.py`
evaluates the larger set first and stops at the first failure; in
`summary.json`, `dft_frames` gives the frame count of the smallest passing
set. The comparison records
residual MLIP forces at the reference endpoint and errors across moiré
periods.

## Adapting to another system

Copy this case directory and edit the hard-coded settings for your MX₂
system. Update the starting monolayer, element symbols and MACE `E0s`.
Replace the potentials and update `atom_types` and the `pair_coeff`
mappings in the reference driver. Adjust the sampling recipe constants
and corresponding label counts, the `TEST_CELLS` angles and atom counts,
the validation cell labels, and the acceptance thresholds. See the
[MLIP user guide](../../../docs/mlip.md) for the shared sampling,
relaxation and analysis APIs.
