"""Lazy MACE calculators for the zero-shot baseline and fine-tuned models."""


def zero_shot_calculator(device):
    """Load MACE-MP-0 medium with D3(BJ) in double precision."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    from mace.calculators import mace_mp

    return mace_mp(model="medium", dispersion=True, default_dtype="float64", device=device)


def finetuned_calculator(model_path, device):
    """Load a fine-tuned model in double precision, without added dispersion."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(model_path), device=device, default_dtype="float64")
