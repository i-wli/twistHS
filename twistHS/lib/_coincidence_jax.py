"""JIT-compiled kernels for the optional JAX coincidence backend."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from twistHS.lib.coincidence import _BATCH, _boxes, _evaluate

__all__ = ["boxes", "enable_x64", "evaluate"]

_jax_boxes = jax.jit(partial(_boxes, xp=jnp, while_loop=jax.lax.while_loop))
_jax_evaluate = jax.jit(partial(_evaluate, xp=jnp))


def _pad(batch):
    """Keep the JIT input shape fixed, including the last block."""
    return np.pad(batch, ((0, _BATCH - len(batch)), (0, 0), (0, 0)), mode="edge")


def boxes(matrices, bottom, top_inverse, radius):
    result = _jax_boxes(_pad(matrices), bottom, top_inverse, radius)
    return tuple(np.asarray(array)[:len(matrices)] for array in result)


def evaluate(bottom_matrices, top_matrices, bottom, top, radius, moire_projection, moire_rows):
    result = _jax_evaluate(_pad(bottom_matrices), _pad(top_matrices), bottom, top, radius, moire_projection, moire_rows)
    return tuple(np.asarray(array)[:len(bottom_matrices)] for array in result)
