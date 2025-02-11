"""Generate the small-cell MoS2 datasets for LAMMPS labeling."""

import argparse
import json
from pathlib import Path

from ase.io import read, write

from twistHS.lib.sampling import sample_aligned_bilayers, sample_monolayers

STRATA = (0.0, 0.1, 0.25, 0.5)
STRAIN = 0.02
MARGIN = 0.3
SUPER_Z = 30
TRAIN = (64, 16)
SUBSET = (16, 4)
VALID = (7, 2)
RIGID = 64
TRAIN_SEED, VALID_SEED, RIGID_SEED = 100, 200, 300
REGISTRY_SEED = 400
REGISTRY_COUNT, REGISTRY_SUBSET_COUNT = 64, 16
STACKINGS = ("Rhh", "RhX", "RhM", "HhX", "Hhh", "HhM")


def annotate(frames, config_type, stratum, first_id):
    for sample_id, atoms in enumerate(frames, first_id):
        atoms.info.update(config_type=config_type, stratum=stratum, sample_id=sample_id)


def stratified(layer, d_range, counts, config_type, seed_base, first_id):
    bilayer_count, monolayer_count = counts
    strata = {}
    seeds = {}
    for family in ("R", "H", "mono"):
        for sigma in STRATA:
            stratum = f"{family}_{sigma:g}"
            seed = seed_base + len(strata)
            options = dict(super_z=SUPER_Z, sigma_atom=sigma, strain=STRAIN)
            if family == "mono":
                frames = sample_monolayers(layer, monolayer_count, seed, **options)
            else:
                frames = sample_aligned_bilayers(layer, bilayer_count, seed, family, d_range, **options)
            annotate(frames, config_type, stratum, first_id)
            first_id += len(frames)
            strata[stratum] = frames
            seeds[stratum] = seed
    return strata, seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monolayer", type=Path, required=True)
    parser.add_argument("--small-cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    layer = read(args.monolayer)
    reference = json.loads(args.small_cells.read_text())
    distances = {label: reference["stackings"][label]["d"] for label in STACKINGS}
    d_min, d_max = min(distances.values()), max(distances.values())
    d_range = (d_min - MARGIN, d_max + MARGIN)

    train, train_seeds = stratified(layer, d_range, TRAIN, "train", TRAIN_SEED, 0)
    valid, valid_seeds = stratified(layer, d_range, VALID, "valid", VALID_SEED, 1000)
    rigid, rigid_seeds = [], {}
    for i, family in enumerate("RH"):
        stratum = f"rigid_{family}"
        seed = RIGID_SEED + i
        frames = sample_aligned_bilayers(
            layer, RIGID, seed, family, (d_min, d_max), SUPER_Z, sigma_atom=0,
        )
        annotate(frames, "rigid", stratum, 2000 + len(rigid))
        rigid.extend(frames)
        rigid_seeds[stratum] = seed

    bilayer_subset, monolayer_subset = SUBSET
    subset = []
    for stratum, frames in train.items():
        count = monolayer_subset if stratum.startswith("mono_") else bilayer_subset
        subset.extend(frames[:count])

    registry = sample_aligned_bilayers(
        layer, REGISTRY_COUNT, REGISTRY_SEED, "R", d_range, SUPER_Z, sigma_atom=0, strain=0,
    )
    annotate(registry, "registry", "registry_R", 3000)
    train["registry_R"] = registry
    train_seeds["registry_R"] = REGISTRY_SEED
    subset.extend(registry[:REGISTRY_SUBSET_COUNT])

    files = {
        "train_512": [atoms for frames in train.values() for atoms in frames],
        "train_128": subset,
        "valid": [atoms for frames in valid.values() for atoms in frames],
        "rigid": rigid,
    }
    counts = {f"{name}.extxyz": len(frames) for name, frames in files.items()}
    if tuple(counts.values()) != (640, 160, 64, 128):
        raise ValueError(f"Unexpected dataset frame counts: {counts}")

    recipe = {
        "d_range": d_range, "stacking_distances": distances,
        "sigma_atom": STRATA, "strain": STRAIN, "super_z": SUPER_Z,
        "seeds": {"train_512": train_seeds, "valid": valid_seeds, "rigid": rigid_seeds},
        "frame_counts": counts,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in files.items():
        write(args.out / f"{name}.extxyz", frames)
    (args.out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")


if __name__ == "__main__":
    main()
