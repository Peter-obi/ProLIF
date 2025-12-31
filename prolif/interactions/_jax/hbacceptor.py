"""
JAX-accelerated hydrogen bond detection.

HBAcceptor: ligand acceptor + residue donor-hydrogen
HBDonor: ligand donor-hydrogen + residue acceptor (inverted)
"""

import jax.numpy as jnp

from .primitives import angle_at_vertex, pairwise_distances


def hbacceptor_contacts(
    acceptor_coords: jnp.ndarray,
    donor_coords: jnp.ndarray,
    hydrogen_coords: jnp.ndarray,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect HBAcceptor interactions (ligand acceptor + residue donor).

    A hydrogen bond exists when:
        1. Distance(acceptor, donor) <= distance_cutoff
        2. dha_angle_min <= D-H-A angle <= dha_angle_max

    Args:
        acceptor_coords: (N, 3) acceptor positions (ligand).
        donor_coords: (M, 3) donor positions (residue).
        hydrogen_coords: (M, 3) hydrogen positions (bonded to donor[i]).
        distance_cutoff: Max acceptor-donor distance (default 3.5 Å).
        dha_angle_min: Minimum D-H-A angle in degrees (default 130°).
        dha_angle_max: Maximum D-H-A angle in degrees (default 180°).

    Returns:
        contact_mask: (N, M) boolean array, True where H-bond exists.
        distances: (N, M) acceptor-donor distances.
        angles: (N, M) D-H-A angles in degrees.
    """
    distances = pairwise_distances(acceptor_coords, donor_coords)
    angles_rad = angle_at_vertex(
        donor_coords[None, :, :],
        hydrogen_coords[None, :, :],
        acceptor_coords[:, None, :],
    )
    angles_deg = jnp.degrees(angles_rad)
    distance_ok = distances <= distance_cutoff
    angle_ok = (angles_deg >= dha_angle_min) & (angles_deg <= dha_angle_max)
    contact_mask = distance_ok & angle_ok

    return contact_mask, distances, angles_deg


def hbdonor_contacts(
    donor_coords: jnp.ndarray,
    hydrogen_coords: jnp.ndarray,
    acceptor_coords: jnp.ndarray,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect HBDonor interactions (ligand donor + residue acceptor).

    This is the inverted form of hbacceptor_contacts.

    Args:
        donor_coords: (N, 3) donor positions (ligand).
        hydrogen_coords: (N, 3) hydrogen positions (bonded to donor[i]).
        acceptor_coords: (M, 3) acceptor positions (residue).
        distance_cutoff: Max donor-acceptor distance (default 3.5 Å).
        dha_angle_min: Minimum D-H-A angle in degrees (default 130°).
        dha_angle_max: Maximum D-H-A angle in degrees (default 180°).

    Returns:
        contact_mask: (N, M) boolean array, True where H-bond exists.
        distances: (N, M) donor-acceptor distances.
        angles: (N, M) D-H-A angles in degrees.
    """
    distances = pairwise_distances(donor_coords, acceptor_coords)
    angles_rad = angle_at_vertex(
        donor_coords[:, None, :],
        hydrogen_coords[:, None, :],
        acceptor_coords[None, :, :],
    )
    angles_deg = jnp.degrees(angles_rad)
    distance_ok = distances <= distance_cutoff
    angle_ok = (angles_deg >= dha_angle_min) & (angles_deg <= dha_angle_max)
    contact_mask = distance_ok & angle_ok

    return contact_mask, distances, angles_deg
