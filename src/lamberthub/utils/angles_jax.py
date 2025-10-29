"""Utilities related to angles computations"""

import jax
import jax.numpy as np
from jax import jit

from jax.numpy.linalg import norm


@jax.tree_util.Partial(jit, static_argnames=['prograde'])
def get_transfer_angle(r1, r2, prograde):
    """Compute the transfer angle of the trajectory.

    Initial and final position vectors are required together with the direction
    of motion.

    Parameters
    ----------
    r1 : ~np.array
        Initial position vector.
    r2 : ~np.array
        Final position vector.
    prograde : bool
        ``True`` for prograde motion, ``False`` otherwise.

    Returns
    -------
    dtheta : float
        Transfer angle in radians.

    Notes
    -----
    This implementation uses jnp.where for all conditional logic to maintain
    differentiability throughout the function.

    """
    # Compute cross product and check collinearity using a small tolerance
    cross_r1_r2 = np.cross(r1, r2)
    cross_norm = norm(cross_r1_r2)

    # Use a small tolerance to detect collinearity (avoiding exact zero comparison)
    eps = 1e-10
    is_collinear = cross_norm < eps

    # For collinear case: check if vectors point in same or opposite directions
    # Use dot product sign: positive means same direction (angle=0), negative means opposite (angle=pi)
    r1_norm = norm(r1)
    r2_norm = norm(r2)
    cos_angle = np.dot(r1, r2) / (r1_norm * r2_norm)
    collinear_angle = np.where(cos_angle > 0, 0.0, np.pi)

    # Non-collinear case: compute the transfer angle normally
    # Solve for a unitary vector normal to the vector plane
    # Add small eps to denominator to avoid division by zero
    h = cross_r1_r2 / (cross_norm + eps)

    # Compute the projection of the normal vector onto the reference plane
    alpha = np.dot(np.array([0, 0, 1]), h)

    # Get the minimum angle (0 <= theta0 <= pi) between r1 and r2
    theta0 = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Fix the value of theta based on prograde/retrograde and alpha sign
    # For prograde: dtheta = theta0 if alpha > 0 else 2*pi - theta0
    # For retrograde: dtheta = theta0 if alpha < 0 else 2*pi - theta0
    if prograde:
        noncollinear_angle = np.where(alpha > 0, theta0, 2 * np.pi - theta0)
    else:
        noncollinear_angle = np.where(alpha < 0, theta0, 2 * np.pi - theta0)

    # Select between collinear and non-collinear cases
    dtheta = np.where(is_collinear, collinear_angle, noncollinear_angle)

    return dtheta


@jax.tree_util.Partial(jit, static_argnames=['prograde'])
def get_orbit_normal_vector(r1, r2, prograde):
    """
    Computes a unitary normal vector aligned with the specific angular momentum
    one of the orbit.

    Parameters
    ----------
    r1: np.array
        Initial position vector.
    r2: np.array
        Final position vector.
    prograde: bool
        If True, it assumes prograde motion, otherwise assumes retrograde.

    Returns
    -------
    i_h: np.array
        Unitary vector aligned with orbit specific angular momentum.

    Notes
    -----
    This implementation uses jnp.where for conditional logic to maintain
    differentiability throughout the function.

    """
    # Compute the normal vector and its projection onto the vertical axis
    cross_r1_r2 = np.cross(r1, r2)
    i_h = cross_r1_r2 / norm(cross_r1_r2)

    # Solve the projection onto the positive vertical direction of the
    # fundamental plane.
    alpha = np.dot(np.array([0, 0, 1]), i_h)

    # A prograde orbit always has a positive vertical component of its specific
    # angular momentum. Therefore, we just need to check for this condition
    # Use jnp.where to maintain differentiability
    if prograde:
        i_h = np.where(alpha > 0, i_h, -i_h)
    else:
        i_h = np.where(alpha < 0, i_h, -i_h)

    return i_h


@jax.tree_util.Partial(jit, static_argnames=['prograde'])
def get_orbit_inc_and_raan_from_position_vectors(r1, r2, prograde):
    """
    Computes the inclination of the orbit being known an initial and a final
    position vectors together with the sense of motion.

    Parameters
    ----------
    r1: np.array
        Initial position vector.
    r2: np.array
        Final position vector.
    prograde: bool
        If True, it assumes prograde motion, otherwise assumes retrograde.

    Returns
    -------
    inc: float
        Inclination of the orbit.
    raan: float
        Right ascension of the ascending node.

    Notes
    -----
    This implementation uses jnp.where for conditional logic to maintain
    differentiability throughout the function.

    """
    # Get a unitary vector aligned in direction and sense with the specific
    # angular momentum one.
    i_h = get_orbit_normal_vector(r1, r2, prograde)

    # Define the unitary vector along Z-axis of the fundamental plane
    i_K = np.array([0, 0, 1])

    # Check if the orbit is coplanar with fundamental plane using tolerance
    # Use atol and rtol instead of pure floating zero comparison
    eps = 1e-10
    is_coplanar = (np.abs(i_h[0]) < eps) & (np.abs(i_h[1]) < eps)

    # Non-coplanar case: compute inc and raan normally
    # Inclination is always bounded between [0, pi], so no correction is needed
    inc_noncoplanar = np.arccos(i_h[2] / norm(i_h))

    # Compute the RAAN using a vector in the direction and sense of the line
    # of nodes. Because RAAN is bounded between [0, 2pi], the arctan2
    # function is used.
    n = np.cross(i_K, i_h)
    raan_noncoplanar = np.arctan2(n[1], n[0]) % (2 * np.pi)

    # Select between coplanar and non-coplanar cases
    inc = np.where(is_coplanar, 0.0, inc_noncoplanar)
    raan = np.where(is_coplanar, 0.0, raan_noncoplanar)

    return inc, raan


@jit
def nu_to_E(nu, ecc):
    """
    Retrieves eccentric anomaly from true one.

    Parameters
    ----------
    nu: float
        True anomaly.
    ecc: float
        Eccentricity of the orbit.

    Returns
    -------
    E: float
        Eccentric anomaly.

    """
    E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(nu / 2))
    return E


@jit
def E_to_nu(E, ecc):
    """
    Retrieves true anomaly from eccentric one.

    Parameters
    ----------
    E: float
        Eccentric anomaly.
    ecc: float
        Eccentricity of the orbit.

    Returns
    -------
    nu: float
        True anomaly.

    """
    nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    return nu


@jit
def nu_to_B(nu):
    """
    Retrieves parabolic anomaly from true one.

    Parameters
    ----------
    nu: float
        True anomaly

    Returns
    -------
    B: float
        Parabolic anomaly

    Notes
    -----
    As explained in Vallado's [1], :math:`B` is used instead of :math:`P` just
    to not confuse with the orbital parameter.

    """
    B = np.tan(nu / 2)
    return B


@jit
def B_to_nu(B):
    """
    Retrieves the true anomaly from parabolic one.

    Parameters
    ----------
    B: float
        Parabolic anomaly

    Returns
    -------
    nu: float
        True anomaly

    Notes
    -----
    As explained in Vallado's [1], :math:`B` is used instead of :math:`P` just
    to not confuse with the orbital parameter.

    """
    nu = 2 * np.arctan(B)
    return nu


@jit
def nu_to_H(nu, ecc):
    """
    Retrieves hyperbolic anomaly from true one.

    Parameters
    ----------
    nu: float
        True anomaly
    ecc: float
        Eccentricity of the orbit

    Returns
    -------
    H: float
        Hyperbolic anomaly

    """
    H = 2 * np.arctanh(np.sqrt((ecc - 1) / (ecc + 1)) * np.tan(nu / 2))
    return H


@jit
def H_to_nu(H, ecc):
    """
    Retrieves hyperbolic anomaly from true one.

    Parameters
    ----------
    H: float
        Hyperbolic anomaly
    ecc: float
        Eccentricity of the orbit

    Returns
    -------
    nu: float
        True anomaly

    """
    nu = 2 * np.arctan(np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(H / 2))
    return nu
