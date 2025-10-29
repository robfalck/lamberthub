"""JAX-compatible implementation of Stumpff functions.

These functions are fully differentiable using JAX's automatic differentiation.
No custom JVP rules are needed - we use jnp.where for branching to maintain
differentiability, with careful handling to avoid NaN values.
"""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def c2(psi):
    r"""Second Stumpff function (JAX-compatible).

    For positive arguments:

    .. math::

        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}

    Parameters
    ----------
    psi : float or jax.numpy.ndarray
        Input value(s).

    Returns
    -------
    res : float or jax.numpy.ndarray
        Value of c2(psi).

    Notes
    -----
    This implementation uses jnp.where for conditional logic to maintain
    differentiability. JAX's automatic differentiation computes gradients.
    Small epsilon values are used to prevent division by zero.
    """
    eps_threshold = 1.0
    eps_safe = 1e-15  # Small value to prevent division by zero

    # For psi > eps: use (1 - cos(sqrt(psi))) / psi
    # Add eps_safe to psi to prevent division by zero in unused branches
    sqrt_psi_pos = jnp.sqrt(jnp.maximum(psi, 0.0) + eps_safe)
    large_positive = (1.0 - jnp.cos(sqrt_psi_pos)) / (psi + eps_safe)

    # For psi < -eps: use (cosh(sqrt(-psi)) - 1) / (-psi)
    sqrt_psi_neg = jnp.sqrt(jnp.maximum(-psi, 0.0) + eps_safe)
    large_negative = (jnp.cosh(sqrt_psi_neg) - 1.0) / (-psi + eps_safe)

    # For |psi| <= eps: use Taylor series
    # c2(psi) = 1/2 - psi/24 + psi^2/720 - psi^3/40320 + psi^4/3628800...
    near_zero = (1.0 / 2.0 - psi / 24.0 + psi**2 / 720.0 -
                 psi**3 / 40320.0 + psi**4 / 3628800.0)

    # Use jnp.where to select between cases (maintaining differentiability)
    res = jnp.where(
        psi > eps_threshold,
        large_positive,
        jnp.where(
            psi < -eps_threshold,
            large_negative,
            near_zero
        )
    )

    return res


@jit
def c3(psi):
    r"""Third Stumpff function (JAX-compatible).

    For positive arguments:

    .. math::

        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}

    Parameters
    ----------
    psi : float or jax.numpy.ndarray
        Input value(s).

    Returns
    -------
    res : float or jax.numpy.ndarray
        Value of c3(psi).

    Notes
    -----
    This implementation uses jnp.where for conditional logic to maintain
    differentiability. JAX's automatic differentiation computes gradients.
    Small epsilon values are used to prevent division by zero.
    """
    eps_threshold = 1.0
    eps_safe = 1e-15  # Small value to prevent division by zero

    # For psi > eps: use (sqrt(psi) - sin(sqrt(psi))) / (psi * sqrt(psi))
    sqrt_psi_pos = jnp.sqrt(jnp.maximum(psi, 0.0) + eps_safe)
    large_positive = (sqrt_psi_pos - jnp.sin(sqrt_psi_pos)) / ((psi + eps_safe) * sqrt_psi_pos)

    # For psi < -eps: use (sinh(sqrt(-psi)) - sqrt(-psi)) / (-psi * sqrt(-psi))
    sqrt_psi_neg = jnp.sqrt(jnp.maximum(-psi, 0.0) + eps_safe)
    large_negative = (jnp.sinh(sqrt_psi_neg) - sqrt_psi_neg) / ((-psi + eps_safe) * sqrt_psi_neg)

    # For |psi| <= eps: use Taylor series
    # c3(psi) = 1/6 - psi/120 + psi^2/5040 - psi^3/362880 + psi^4/39916800...
    near_zero = (1.0 / 6.0 - psi / 120.0 + psi**2 / 5040.0 -
                 psi**3 / 362880.0 + psi**4 / 39916800.0)

    # Use jnp.where to select between cases (maintaining differentiability)
    res = jnp.where(
        psi > eps_threshold,
        large_positive,
        jnp.where(
            psi < -eps_threshold,
            large_negative,
            near_zero
        )
    )

    return res
