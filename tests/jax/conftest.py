"""
Pytest configuration for JAX tests.
"""

import pytest

# Skip all tests in this directory if JAX is not available
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    pytest.skip("JAX not installed", allow_module_level=True)


@pytest.fixture
def simple_coords():
    """Simple test coordinates: origin and unit vectors."""
    return jnp.array(
        [
            [0.0, 0.0, 0.0],  # origin
            [1.0, 0.0, 0.0],  # x-axis
            [0.0, 1.0, 0.0],  # y-axis
            [0.0, 0.0, 1.0],  # z-axis
        ]
    )


@pytest.fixture
def xy_plane_ring():
    """A square ring in the XY plane (normal should be Z-axis)."""
    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )


@pytest.fixture
def two_triangles():
    """Two triangles for testing batch operations."""
    coords = jnp.array(
        [
            # Triangle 1 (indices 0, 1, 2)
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            # Triangle 2 (indices 3, 4, 5)
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [11.0, 11.0, 10.0],
        ]
    )
    indices = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
    return coords, indices
