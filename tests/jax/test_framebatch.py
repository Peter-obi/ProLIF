"""
Tests for frame-batched JAX helpers.

Run with: pytest tests/jax/test_framebatch.py -v
"""

import jax.numpy as jnp
from numpy.testing import assert_allclose


def test_pairwise_distances_frames_shapes_and_values():
    """Validate shapes and a few distances for a small synthetic setup.

    The setup uses F=2 frames, R=2 residues, N=1 ligand atom, M=2 atoms per
    residue (padded not required here). Different coordinates across frames
    ensure the distance tensor varies in time.
    """
    from prolif.interactions._jax.framebatch import pairwise_distances_frames

    lig = jnp.array([
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
    ])
    res = jnp.array([
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        ],
    ])

    d = pairwise_distances_frames(lig, res)
    assert d.shape == (2, 2, 1, 2)

    assert_allclose(d[0, 0, 0, 0], 0.0, atol=1e-6)
    assert_allclose(d[0, 0, 0, 1], 1.0, atol=1e-6)
    assert_allclose(d[1, 1, 0, 1], jnp.sqrt(5.0), atol=1e-6)


def test_hbacceptor_frames_true_mask():
    """Construct a linear D–H–A configuration that passes HB acceptor criteria.

    Two frames are used with an acceptor shifted but still under cutoff; the
    D–H–A angle remains 180 degrees, and the A–D distance is within 3.5 Å.
    """
    from prolif.interactions._jax.framebatch import hbacceptor_frames

    lig = jnp.array([
        [[2.0, 0.0, 0.0]],
        [[3.0, 0.0, 0.0]],
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
    """Donor/H on ligand and acceptor on residue; first frame true, second false.

    The first frame satisfies distance and angle; the second moves the acceptor
    far enough to fail the distance criterion.
    """
    from prolif.interactions._jax.framebatch import hbdonor_frames

    lig = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])
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
    """Ligand as acceptor with neighbor R; residue as X–D donor.

    The first frame meets AX distance and both angle constraints; the second
    moves A far away to fail the distance criterion.
    """
    from prolif.interactions._jax.framebatch import xbacceptor_frames

    lig = jnp.array([
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[10.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
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
    """Ligand as X–D donor; residue as acceptor with neighbor R.

    The first frame meets XA distance and angle constraints; the second places
    A far away to fail by distance.
    """
    from prolif.interactions._jax.framebatch import xbdonor_frames

    lig = jnp.array([
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    ])
    res = jnp.array([
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[10.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
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
