# Twisted Heterostructure Generation Toolkit
A Python-based solution for generating twisted heterostructures through both a Command-Line Interface (CLI) and a Graphical User Interface (GUI). Whether you are a researcher, student, or hobbyist interested in the fascinating world of moiré patterns, this toolkit is designed to provide a flexible and user-friendly approach to exploring these patterns.

The default coincidence search runs. It enumerates bottom sublattices in Hermite normal form and
matches their basis vectors to the rotated top lattice. For each requested
angle, it returns the cell with the fewest atoms that satisfies all three
limits. Ties are resolved by maximum strain, effective angle deviation, and
the integer matrices.

- `--tolerance` limits the maximum absolute principal strain across both
  layers. The default is 1%.
- `--angle-tol` limits the difference between requested and effective twist.
  The default is 0.1°.
- `--max-atoms` limits the total number of atoms. The default is 12,000.

Both layers deform into their average cell `C`. The right polar decomposition
of `F = (C_layer^-1 C)^T` gives a rotation `phi` and a stretch whose principal
values minus one give the engineering strains. The effective twist is
`theta_eff = theta + phi_top - phi_bottom`. Both interfaces plot this angle.

`--method first-order` reproduces the v1 rounding approach for aligned
primitive bases. It pairs the reduced bottom reciprocal vectors with their
nearest top reciprocal vectors and requires a unimodular pairing. The
real-space moiré cell is `A_M = (A_b^-1 - A_t^-1)^-1` in the aligned bases.
This mode applies only the strain filter when scanning angles. Singular
moiré geometry, invalid pairing, and singular rounded cells raise errors.
[Hermann (2012)](https://doi.org/10.1088/0953-8984/24/31/314210)

`-mo K` / `--max-order K` additionally prints higher-index reciprocal components
for every requested angle. Each nonzero bottom vector is paired with its nearest
top vector; pairs with a zero top vector are discarded. Both vectors must lie
within `K` times their layer's shortest reciprocal length. Harmonics and ±
duplicates are removed. The table reports
input-basis indices, Cartesian `q = G_b - G_t`, `2π/|q|`, direction, geometric
order (1, √3, …), and multiplicity of equal-order, equal-period ± pairs. Zero `q`
has infinite period and undefined direction (`nan`). This analysis follows the
higher-index construction discussed by
[Pham et al. (2022)](https://doi.org/10.1038/s41699-022-00321-9); it computes no
Fourier amplitudes and generates no additional atomic structures. Multilayer
and supermoiré analysis are not included.

## Features

- CLI Tool (`twist_cli.py`): generate structures through your terminal or command prompt.
- GUI Tool (`twist_gui.py`): an user-friendly interface, allowing for interactive adjustments and real-time visualization.


## Installation

Use Python 3.12 or newer. To install the CLI and Python package from source:

```sh
git clone https://github.com/i-wli/twistHS.git
cd twistHS
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Install `".[gui]"` to add the Gradio interface.

For local development and verification:

```sh
python -m pip install -e ".[test,gui]"
python -m pip install ruff
pytest
ruff check
```

To build and publish a release manually from a clean checkout:

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## Usage

**Option 1. Using the CLI Tool (`twist_cli.py`)**

Open your command-line interface and run:
```
twist_cli [options]
```
For example, analyze reciprocal components through order 2 and visualize the
structures that pass a 0.5% strain tolerance:
```
twist_cli -b tests/graphene.xsf -t tests/MoS2.xsf -al 0 10 1 -tol 0.5 -mo 2
```
This should open a matplotlib interface:
![image](https://github.com/i-wli/twistHS/blob/main/docs/cli.gif)


**Option 2. Using the GUI Tool (`twist_gui.py`)**

To launch the GUI, simply run:
```
twist_gui
```

Once the GUI is open, you should see:
![image](https://github.com/i-wli/twistHS/blob/main/docs/gui.gif)

The GUI uses the requested angle to select or save an individual structure.
In the CLI, click a point to open its structure and save it with the default
XSF filename or a path supplied through `-w`. Use `-o` to specify the format
for that path.

**Python API**

```python
from ase.io import read
from twistHS.lib.algorithm import HSGenerator

generator = HSGenerator(read("tests/graphene.xsf"), read("tests/MoS2.xsf"),
                        tolerance=1.0, angle_tol=0.1, max_atoms=12_000)
atoms, strain, theta_eff = generator.gen_supercell(5, super_z=30, delta_z=7)
for requested_angle, atoms, strain in generator.scan([4, 5, 6], super_z=30, delta_z=7):
    print(requested_angle, atoms.info["theta_eff"], len(atoms), 100 * strain)
```

`gen_supercell` returns strain as a fraction and raises `ValueError` if search
finds no matching cell. `scan` omits requested angles with no match. Its optional
`tolerance` argument overrides the generator's strain limit in percent.

## Supercell construction benchmark

On an Apple M1 with 8 GB of RAM, vectorized atom placement cut the median
time to build both layers of a 1,316,056-atom twisted graphene supercell
from 614.75 s to 0.1545 s. Median peak process memory fell
from 499.0 MiB to 377.6 MiB.
The [benchmark report](docs/benchmarks.md) covers the 5° to 0.1° sweep and
provides the raw measurements and reproduction commands.

![Build time and peak process memory by atom count](docs/benchmark-build.png)
