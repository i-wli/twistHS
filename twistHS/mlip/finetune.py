"""Check reference label keys before MACE fine-tuning."""

from ase.io import read


def validate_dataset(files):
    """Require REF_energy, REF_stress and REF_forces in every extxyz frame."""
    for path in files:
        for frame, atoms in enumerate(read(path, index=":", format="extxyz")):
            missing = [key for key in ("REF_energy", "REF_stress") if key not in atoms.info]
            if "REF_forces" not in atoms.arrays:
                missing.append("REF_forces")
            if missing:
                raise ValueError(f"{path}: frame {frame} missing {', '.join(missing)}")
