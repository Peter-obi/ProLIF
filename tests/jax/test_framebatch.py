"""
Tests for frame-batched JAX helpers.

Run with: pytest tests/jax/test_framebatch.py -v
"""

import jax.numpy as jnp
from numpy.testing import assert_allclose


def test_pairwise_distances_frames_shapes_and_values():
    from prolif.interactions._jax.framebatch import pairwise_distances_frames

    # F=2, R=2, N=1, M=2
    lig = jnp.array([
        [[0.0, 0.0, 0.0]],  # frame 0
        [[1.0, 0.0, 0.0]],  # frame 1
    ])
    res = jnp.array([
        [  # frame 0, residues 0..1
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        [  # frame 1
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        ],
    ])

    d = pairwise_distances_frames(lig, res)
    assert d.shape == (2, 2, 1, 2)

    # Check a few entries
    assert_allclose(d[0, 0, 0, 0], 0.0, atol=1e-6)
    assert_allclose(d[0, 0, 0, 1], 1.0, atol=1e-6)
    assert_allclose(d[1, 1, 0, 1], jnp.sqrt(5.0), atol=1e-6)  # dist((1,0,0),(2,2,0))


def test_hbacceptor_frames_true_mask():
    from prolif.interactions._jax.framebatch import hbacceptor_frames

    # Construct geometry with D(0,0,0), H(1,0,0), A(2,0,0) → angle 180°, AD=2
    lig = jnp.array([
        [[2.0, 0.0, 0.0]],  # frame 0: A
        [[3.0, 0.0, 0.0]],  # frame 1: farther A but still within cutoff 3.5
    ])
    res = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    acc_idx = jnp.array([0])
    d_idx = jnp.array([0])
    h_idx = jnp.array([1])

    mask, dists, ang = hbacceptor_frames(lig, res, acc_idx, d_idx, h_idx)
    assert mask.shape == (2, 1, 1)
    assert dists.shape == (2, 1, 1)
    assert ang.shape == (2, 1, 1)
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[1, 0, 0]) is True
    assert_allclose(ang[0, 0, 0], 180.0, atol=1e-4)


def test_hbdonor_frames_true_then_false():
    from prolif.interactions._jax.framebatch import hbdonor_frames

    # Donor/H in ligand, acceptor in residue
    # frame0: D(0,0,0), H(1,0,0), A(2,0,0) → true
    # frame1: A(10,0,0) → false by distance
    lig = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])  # shape (F=2, Nd=2, 3); indices d=0, h=1
    res = jnp.array([
        [[2.0, 0.0, 0.0]],
        [[10.0, 0.0, 0.0]],
    ])
    d_idx = jnp.array([0])
    h_idx = jnp.array([1])
    acc_idx = jnp.array([0])

    mask, dists, ang = hbdonor_frames(lig, res, d_idx, h_idx, acc_idx)
    assert mask.shape == (2, 1, 1)
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[1, 0, 0]) is False
    assert_allclose(ang[0, 0, 0], 180.0, atol=1e-4)


def test_xbacceptor_frames_true_then_false():
    from prolif.interactions._jax.framebatch import xbacceptor_frames

    # A(1,0,0), R(1,1,0) on ligand; X(0,0,0), D(-1,0,0) on residue
    # A–X=1, A–X–D=180, X–A–R=90 → true
    lig = jnp.array([
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[10.0, 0.0, 0.0], [1.0, 1.0, 0.0]],  # far A → false by distance
    ])
    res = jnp.array([
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    ])
    a_idx = jnp.array([0])
    r_idx = jnp.array([1])
    x_idx = jnp.array([0])
    d_idx = jnp.array([1])

    mask, dists, axd, xar = xbacceptor_frames(lig, res, a_idx, r_idx, x_idx, d_idx)
    assert mask.shape == (2, 1, 1)
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[1, 0, 0]) is False
    assert_allclose(axd[0, 0, 0], 180.0, atol=1e-4)
    assert_allclose(xar[0, 0, 0], 90.0, atol=1e-4)


def test_xbdonor_frames_true_then_false():
    from prolif.interactions._jax.framebatch import xbdonor_frames

    # X(0,0,0), D(-1,0,0) on ligand; A(1,0,0), R(1,1,0) on residue
    lig = jnp.array([
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    ])
    res = jnp.array([
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[10.0, 0.0, 0.0], [1.0, 1.0, 0.0]],  # far A → false by distance
    ])
    x_idx = jnp.array([0])
    d_idx = jnp.array([1])
    a_idx = jnp.array([0])
    r_idx = jnp.array([1])

    mask, dists, axd, xar = xbdonor_frames(lig, res, x_idx, d_idx, a_idx, r_idx)
    assert mask.shape == (2, 1, 1)
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[1, 0, 0]) is False
    assert_allclose(axd[0, 0, 0], 180.0, atol=1e-4)
    assert_allclose(xar[0, 0, 0], 90.0, atol=1e-4)

