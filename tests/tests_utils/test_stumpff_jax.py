"""Test script to verify JAX Stumpff functions and their derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
from lamberthub.utils.stumpff_jax import c2, c3
from lamberthub.utils.stumpff import c2 as c2_ref, c3 as c3_ref


def test_c2_values():
    """Test that c2 produces correct values compared to reference implementation."""
    print("\n=== Testing c2 values ===")

    test_values = [
        -10.0,  # Large negative
        -1.5,   # Negative
        -0.5,   # Small negative
        0.0,    # Zero
        0.5,    # Small positive
        1.5,    # Positive
        10.0,   # Large positive
    ]

    print(f"{'psi':<10} {'c2_ref':<15} {'c2_jax':<15} {'diff':<15}")
    print("-" * 60)

    max_diff = 0.0
    for psi in test_values:
        c2_ref_val = c2_ref(psi)
        c2_jax_val = float(c2(psi))
        diff = abs(c2_ref_val - c2_jax_val)
        max_diff = max(max_diff, diff)

        print(f"{psi:<10.2f} {c2_ref_val:<15.10f} {c2_jax_val:<15.10f} {diff:<15.2e}")

    print(f"\nMax difference: {max_diff:.2e}")
    assert max_diff < 1e-6, f"c2 values differ too much: {max_diff}"
    print("✓ c2 values match reference implementation!")


def test_c3_values():
    """Test that c3 produces correct values compared to reference implementation."""
    print("\n=== Testing c3 values ===")

    test_values = [
        -10.0,  # Large negative
        -1.5,   # Negative
        -0.5,   # Small negative
        0.0,    # Zero
        0.5,    # Small positive
        1.5,    # Positive
        10.0,   # Large positive
    ]

    print(f"{'psi':<10} {'c3_ref':<15} {'c3_jax':<15} {'diff':<15}")
    print("-" * 60)

    max_diff = 0.0
    for psi in test_values:
        c3_ref_val = c3_ref(psi)
        c3_jax_val = float(c3(psi))
        diff = abs(c3_ref_val - c3_jax_val)
        max_diff = max(max_diff, diff)

        print(f"{psi:<10.2f} {c3_ref_val:<15.10f} {c3_jax_val:<15.10f} {diff:<15.2e}")

    print(f"\nMax difference: {max_diff:.2e}")
    assert max_diff < 1e-6, f"c3 values differ too much: {max_diff}"
    print("✓ c3 values match reference implementation!")


def test_c2_differentiability():
    """Test that c2 is differentiable."""
    print("\n=== Testing c2 differentiability ===")

    test_values = [
        -5.0,   # Large negative
        -1.5,   # Negative
        -0.5,   # Small negative
        0.001,  # Near zero (avoid exact zero for numerical derivative)
        0.5,    # Small positive
        1.5,    # Positive
        5.0,    # Large positive
    ]

    print(f"{'psi':<10} {'c2(psi)':<15} {'dc2/dpsi':<15} {'Finite diff':<15} {'Error':<15}")
    print("-" * 75)

    max_error = 0.0
    for psi in test_values:
        # Compute derivative using JAX
        c2_val = float(c2(psi))
        dc2_dpsi = float(jax.grad(c2)(psi))

        # Compute finite difference approximation
        # Use larger h for float32 precision
        h = 1e-5
        c2_plus = float(c2(psi + h))
        c2_minus = float(c2(psi - h))
        dc2_dpsi_fd = (c2_plus - c2_minus) / (2 * h)

        error = abs(dc2_dpsi - dc2_dpsi_fd)
        max_error = max(max_error, error)

        print(f"{psi:<10.3f} {c2_val:<15.10f} {dc2_dpsi:<15.10f} {dc2_dpsi_fd:<15.10f} {error:<15.2e}")

    print(f"\nMax error: {max_error:.2e}")
    # Finite differences have limited accuracy, especially near regime boundaries
    assert max_error < 2e-3, f"c2 derivative error too large: {max_error}"
    print("✓ c2 is differentiable and derivatives match finite differences!")


def test_c3_differentiability():
    """Test that c3 is differentiable."""
    print("\n=== Testing c3 differentiability ===")

    test_values = [
        -5.0,   # Large negative
        -1.5,   # Negative
        -0.5,   # Small negative
        0.001,  # Near zero (avoid exact zero for numerical derivative)
        0.5,    # Small positive
        1.5,    # Positive
        5.0,    # Large positive
    ]

    print(f"{'psi':<10} {'c3(psi)':<15} {'dc3/dpsi':<15} {'Finite diff':<15} {'Error':<15}")
    print("-" * 75)

    max_error = 0.0
    for psi in test_values:
        # Compute derivative using JAX
        c3_val = float(c3(psi))
        dc3_dpsi = float(jax.grad(c3)(psi))

        # Compute finite difference approximation
        # Use larger h for float32 precision
        h = 1e-5
        c3_plus = float(c3(psi + h))
        c3_minus = float(c3(psi - h))
        dc3_dpsi_fd = (c3_plus - c3_minus) / (2 * h)

        error = abs(dc3_dpsi - dc3_dpsi_fd)
        max_error = max(max_error, error)

        print(f"{psi:<10.3f} {c3_val:<15.10f} {dc3_dpsi:<15.10f} {dc3_dpsi_fd:<15.10f} {error:<15.2e}")

    print(f"\nMax error: {max_error:.2e}")
    # Finite differences have limited accuracy, especially near regime boundaries
    # The error is larger for c3 due to the higher order terms
    assert max_error < 1e-2, f"c3 derivative error too large: {max_error}"
    print("✓ c3 is differentiable and derivatives match finite differences!")


def test_vectorization():
    """Test that functions work with vectorized inputs."""
    print("\n=== Testing vectorization ===")

    psi_array = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    c2_vals = c2(psi_array)
    c3_vals = c3(psi_array)

    print(f"psi values: {psi_array}")
    print(f"c2 values:  {c2_vals}")
    print(f"c3 values:  {c3_vals}")

    # Test that gradients work with vectorization
    jacobian_c2 = jax.jacfwd(c2)(psi_array)
    jacobian_c3 = jax.jacfwd(c3)(psi_array)

    print(f"\nJacobian of c2: diagonal = {jnp.diag(jacobian_c2)}")
    print(f"Jacobian of c3: diagonal = {jnp.diag(jacobian_c3)}")

    print("✓ Vectorization works correctly!")


def test_jit_compilation():
    """Test that functions can be JIT compiled."""
    print("\n=== Testing JIT compilation ===")

    @jax.jit
    def compute_stumpff(psi):
        return c2(psi), c3(psi)

    @jax.jit
    def compute_stumpff_grad(psi):
        return jax.grad(lambda p: c2(p) + c3(p))(psi)

    # Test JIT compiled functions
    psi = 1.5
    c2_val, c3_val = compute_stumpff(psi)
    grad_val = compute_stumpff_grad(psi)

    print(f"JIT compiled c2({psi}) = {c2_val}")
    print(f"JIT compiled c3({psi}) = {c3_val}")
    print(f"JIT compiled gradient = {grad_val}")

    print("✓ JIT compilation works!")


def test_higher_order_derivatives():
    """Test second-order derivatives."""
    print("\n=== Testing higher-order derivatives ===")

    psi = 1.0

    # First derivative
    dc2_dpsi = jax.grad(c2)(psi)
    dc3_dpsi = jax.grad(c3)(psi)

    # Second derivative
    d2c2_dpsi2 = jax.grad(jax.grad(c2))(psi)
    d2c3_dpsi2 = jax.grad(jax.grad(c3))(psi)

    print(f"c2({psi}) = {c2(psi)}")
    print(f"dc2/dpsi = {dc2_dpsi}")
    print(f"d²c2/dpsi² = {d2c2_dpsi2}")

    print(f"\nc3({psi}) = {c3(psi)}")
    print(f"dc3/dpsi = {dc3_dpsi}")
    print(f"d²c3/dpsi² = {d2c3_dpsi2}")

    # Verify second derivative using finite differences
    h = 1e-5
    d2c2_fd = (float(jax.grad(c2)(psi + h)) - float(jax.grad(c2)(psi - h))) / (2 * h)
    error_c2 = abs(float(d2c2_dpsi2) - d2c2_fd)

    print(f"\nSecond derivative error (c2): {error_c2:.2e}")
    # Second derivatives from finite differences are even less accurate
    assert error_c2 < 1e-3, f"Second derivative error too large: {error_c2}"

    print("✓ Higher-order derivatives work!")


if __name__ == "__main__":
    print("=" * 75)
    print("Testing JAX Stumpff Functions Implementation")
    print("=" * 75)

    try:
        test_c2_values()
        test_c3_values()
        test_c2_differentiability()
        test_c3_differentiability()
        test_vectorization()
        test_jit_compilation()
        test_higher_order_derivatives()

        print("\n" + "=" * 75)
        print("All tests passed! ✓")
        print("The JAX Stumpff implementation is correct and fully differentiable.")
        print("=" * 75)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
