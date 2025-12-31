"""
JAX-accelerated halogen bond detection.

XBAcceptor: ligand acceptor-R + residue donor-halogen
XBDonor: ligand donor-halogen + residue acceptor-R (inverted)
"""

import jax.numpy as jnp
from jax import lax

from .primitives import angle_at_vertex, pairwise_distances


def xbacceptor_contacts(
    acceptor_coords: jnp.ndarray,
    acceptor_neighbor_coords: jnp.ndarray,
    halogen_coords: jnp.ndarray,
    donor_coords: jnp.ndarray,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
    precomputed_distances: jnp.ndarray | None = None,
    gate_by_distance: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect XBAcceptor interactions (ligand acceptor + residue halogen donor).

    A halogen bond exists when:
        1. Distance(acceptor, halogen) <= distance_cutoff
        2. axd_angle_min <= A-X-D angle <= axd_angle_max
        3. xar_angle_min <= X-A-R angle <= xar_angle_max

    Args:
        acceptor_coords: (N, 3) acceptor positions A (ligand).
        acceptor_neighbor_coords: (N, 3) acceptor neighbor R (ligand).
        halogen_coords: (M, 3) halogen positions X (residue).
        donor_coords: (M, 3) donor positions D (residue).
        distance_cutoff: Max acceptor-halogen distance (default 3.5 Å).
        axd_angle_min/max: A-X-D angle range in degrees.
        xar_angle_min/max: X-A-R angle range in degrees.

    Args:
        acceptor_coords: (N, 3) acceptor positions A (ligand).
        acceptor_neighbor_coords: (N, 3) neighbor R (ligand).
        halogen_coords: (M, 3) halogen positions X (residue).
        donor_coords: (M, 3) donor positions D (residue).
        distance_cutoff: Maximum A–X distance (Å).
        axd_angle_min: Minimum A–X–D angle (degrees).
        axd_angle_max: Maximum A–X–D angle (degrees).
        xar_angle_min: Minimum X–A–R angle (degrees).
        xar_angle_max: Maximum X–A–R angle (degrees).
        precomputed_distances: Optional (N, M) matrix of A–X distances to reuse.

    Returns:
        contact_mask: (N, M) boolean array, True where X-bond exists.
        distances: (N, M) acceptor-halogen distances.
        axd_angles: (N, M) A-X-D angles in degrees.
        xar_angles: (N, M) X-A-R angles in degrees.
    """
    distances = (
        precomputed_distances
        if precomputed_distances is not None
        else pairwise_distances(acceptor_coords, halogen_coords)
    )
    distance_ok = distances <= distance_cutoff

    def _compute_angles(_):
        axd_angles_rad = angle_at_vertex(
            acceptor_coords[:, None, :],
            halogen_coords[None, :, :],
            donor_coords[None, :, :],
        )
        axd_angles = jnp.degrees(axd_angles_rad)
        xar_angles_rad = angle_at_vertex(
            halogen_coords[None, :, :],
            acceptor_coords[:, None, :],
            acceptor_neighbor_coords[:, None, :],
        )
        xar_angles = jnp.degrees(xar_angles_rad)
        axd_ok = (axd_angles >= axd_angle_min) & (axd_angles <= axd_angle_max)
        xar_ok = (xar_angles >= xar_angle_min) & (xar_angles <= xar_angle_max)
        mask = distance_ok & axd_ok & xar_ok
        return mask, distances, axd_angles, xar_angles

    def _skip_angles(_):
        zeros = jnp.zeros_like(distances)
        return jnp.zeros_like(distance_ok, dtype=bool), distances, zeros, zeros

    contact_mask, distances, axd_angles, xar_angles = (
        lax.cond(jnp.any(distance_ok) & gate_by_distance, _compute_angles, _skip_angles, operand=None)
        if gate_by_distance
        else _compute_angles(None)
    )

    return contact_mask, distances, axd_angles, xar_angles


def xbdonor_contacts(
    halogen_coords: jnp.ndarray,
    donor_coords: jnp.ndarray,
    acceptor_coords: jnp.ndarray,
    acceptor_neighbor_coords: jnp.ndarray,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
    precomputed_distances: jnp.ndarray | None = None,
    gate_by_distance: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect XBDonor interactions (ligand halogen donor + residue acceptor).

    This is the inverted form of xbacceptor_contacts.

    Args:
        halogen_coords: (N, 3) halogen positions X (ligand).
        donor_coords: (N, 3) donor positions D (ligand).
        acceptor_coords: (M, 3) acceptor positions A (residue).
        acceptor_neighbor_coords: (M, 3) acceptor neighbor R (residue).
        distance_cutoff: Max halogen-acceptor distance (default 3.5 Å).
        axd_angle_min/max: A-X-D angle range in degrees.
        xar_angle_min/max: X-A-R angle range in degrees.

    Args:
        halogen_coords: (N, 3) halogen positions X (ligand).
        donor_coords: (N, 3) donor positions D (ligand).
        acceptor_coords: (M, 3) acceptor positions A (residue).
        acceptor_neighbor_coords: (M, 3) neighbor R (residue).
        distance_cutoff: Maximum X–A distance (Å).
        axd_angle_min: Minimum A–X–D angle (degrees).
        axd_angle_max: Maximum A–X–D angle (degrees).
        xar_angle_min: Minimum X–A–R angle (degrees).
        xar_angle_max: Maximum X–A–R angle (degrees).
        precomputed_distances: Optional (N, M) matrix of X–A distances to reuse.

    Returns:
        contact_mask: (N, M) boolean array, True where X-bond exists.
        distances: (N, M) halogen-acceptor distances.
        axd_angles: (N, M) A-X-D angles in degrees.
        xar_angles: (N, M) X-A-R angles in degrees.
    """
    distances = (
        precomputed_distances
        if precomputed_distances is not None
        else pairwise_distances(halogen_coords, acceptor_coords)
    )
    distance_ok = distances <= distance_cutoff

    def _compute_angles(_):
        axd_angles_rad = angle_at_vertex(
            acceptor_coords[None, :, :],
            halogen_coords[:, None, :],
            donor_coords[:, None, :],
        )
        axd_angles = jnp.degrees(axd_angles_rad)
        xar_angles_rad = angle_at_vertex(
            halogen_coords[:, None, :],
            acceptor_coords[None, :, :],
            acceptor_neighbor_coords[None, :, :],
        )
        xar_angles = jnp.degrees(xar_angles_rad)
        axd_ok = (axd_angles >= axd_angle_min) & (axd_angles <= axd_angle_max)
        xar_ok = (xar_angles >= xar_angle_min) & (xar_angles <= xar_angle_max)
        mask = distance_ok & axd_ok & xar_ok
        return mask, distances, axd_angles, xar_angles

    def _skip_angles(_):
        zeros = jnp.zeros_like(distances)
        return jnp.zeros_like(distance_ok, dtype=bool), distances, zeros, zeros

    contact_mask, distances, axd_angles, xar_angles = (
        lax.cond(jnp.any(distance_ok) & gate_by_distance, _compute_angles, _skip_angles, operand=None)
        if gate_by_distance
        else _compute_angles(None)
    )

    return contact_mask, distances, axd_angles, xar_angles
