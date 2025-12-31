"""
Frame-batched JAX helpers for MD trajectories.

This module provides minimal, safe utilities to vectorize geometry across
many frames while keeping indices fixed (SMARTS done once per system).

Design goals:
- Avoid residue-batched complexity (no per-residue index threading/padding).
- Keep shapes stable across frames to enable JIT specialization.
- Only implement clear wins; leave angles to the integration layer for now.

API stability: experimental. Functions here are intentionally small and
focused to avoid future breakage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .primitives import pairwise_distances, angle_at_vertex
from .primitives import angle_between_vectors


def pairwise_distances_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
) -> jnp.ndarray:
    """Compute distances for each frame and residue in a trajectory.

    Args:
        lig_coords_f: (F, N, 3) ligand coordinates per frame.
        res_coords_f: (F, R, M, 3) residue coordinates per frame
            (padded to the same M across residues; masks handled upstream).

    Returns:
        (F, R, N, M) array of pairwise distances.

    Notes:
        - Shapes must be consistent across frames to avoid recompilation.
        - This function does not apply masks; callers should mask padded
          atoms as needed.
    """

    def _frame_distances(lig: jnp.ndarray, res_batch: jnp.ndarray) -> jnp.ndarray:
        # res_batch: (R, M, 3)
        # Map over residues for a single frame → (R, N, M)
        return jax.vmap(lambda rc: pairwise_distances(lig, rc))(res_batch)

    # Map over frames → (F, R, N, M)
    return jax.vmap(_frame_distances)(lig_coords_f, res_coords_f)


def hbacceptor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    acc_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched hydrogen bond (acceptor) geometry.

    Args:
        lig_coords_f: (F, N, 3) ligand coords per frame.
        res_coords_f: (F, M, 3) residue coords per frame.
        acc_idx: (Na,) ligand acceptor indices.
        d_idx: (K,) residue donor indices.
        h_idx: (K,) residue hydrogen indices (paired with donors).
        distance_cutoff: Max A–D distance.
        dha_angle_min/max: D–H–A angle limits (deg).

    Returns:
        mask: (F, Na, K) boolean
        distances: (F, Na, K) A–D distances
        angles: (F, Na, K) D–H–A angles (deg)
    """
    acc = lig_coords_f[:, acc_idx, :]            # (F, Na, 3)
    donors = res_coords_f[:, d_idx, :]           # (F, K, 3)
    hydrogens = res_coords_f[:, h_idx, :]        # (F, K, 3)

    # Distances A–D per frame: (F, Na, K)
    dvec = acc[:, :, None, :] - donors[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    # Angles D–H–A at H vertex, broadcast to (F, Na, K)
    ang = angle_at_vertex(
        donors[:, None, :, :],
        hydrogens[:, None, :, :],
        acc[:, :, None, :],
    )
    ang_deg = jnp.degrees(ang)
    ang_ok = (ang_deg >= dha_angle_min) & (ang_deg <= dha_angle_max)

    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def hbdonor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    d_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    acc_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched hydrogen bond (donor) geometry (inverted).

    Returns (mask, distances, angles) with shapes (F, Nd, Ka).
    """
    donors = lig_coords_f[:, d_idx, :]           # (F, Nd, 3)
    hydrogens = lig_coords_f[:, h_idx, :]        # (F, Nd, 3)
    acc = res_coords_f[:, acc_idx, :]            # (F, Ka, 3)

    dvec = donors[:, :, None, :] - acc[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)       # (F, Nd, Ka)
    dist_ok = dists <= distance_cutoff

    ang = angle_at_vertex(
        donors[:, :, None, :],
        hydrogens[:, :, None, :],
        acc[:, None, :, :],
    )
    ang_deg = jnp.degrees(ang)
    ang_ok = (ang_deg >= dha_angle_min) & (ang_deg <= dha_angle_max)
    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def xbacceptor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    a_idx: jnp.ndarray,
    r_idx: jnp.ndarray,
    x_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched halogen bond (acceptor) geometry.

    Returns (mask, distances[A–X], axd_angles, xar_angles) with shapes (F, Na, K).
    """
    acc = lig_coords_f[:, a_idx, :]              # (F, Na, 3)
    neigh = lig_coords_f[:, r_idx, :]            # (F, Na, 3)
    hal = res_coords_f[:, x_idx, :]              # (F, K, 3)
    don = res_coords_f[:, d_idx, :]              # (F, K, 3)

    dvec = acc[:, :, None, :] - hal[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)       # (F, Na, K)
    dist_ok = dists <= distance_cutoff

    # A–X–D at X vertex → (F, Na, K)
    axd = angle_at_vertex(
        acc[:, :, None, :],
        hal[:, None, :, :],
        don[:, None, :, :],
    )
    axd_deg = jnp.degrees(axd)
    axd_ok = (axd_deg >= axd_angle_min) & (axd_deg <= axd_angle_max)

    # X–A–R at A vertex → (F, Na, K)
    xar = angle_at_vertex(
        hal[:, None, :, :],
        acc[:, :, None, :],
        neigh[:, :, None, :],
    )
    xar_deg = jnp.degrees(xar)
    xar_ok = (xar_deg >= xar_angle_min) & (xar_deg <= xar_angle_max)

    mask = dist_ok & axd_ok & xar_ok
    return mask, dists, axd_deg, xar_deg


def xbdonor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    x_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    a_idx: jnp.ndarray,
    r_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched halogen bond (donor) geometry (inverted).

    Returns (mask, distances[X–A], axd_angles, xar_angles) with shapes (F, Nx, Ka).
    """
    hal = lig_coords_f[:, x_idx, :]              # (F, Nx, 3)
    don = lig_coords_f[:, d_idx, :]              # (F, Nx, 3)
    acc = res_coords_f[:, a_idx, :]              # (F, Ka, 3)
    neigh = res_coords_f[:, r_idx, :]            # (F, Ka, 3)

    dvec = hal[:, :, None, :] - acc[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    axd = angle_at_vertex(
        acc[:, None, :, :],
        hal[:, :, None, :],
        don[:, :, None, :],
    )
    axd_deg = jnp.degrees(axd)
    axd_ok = (axd_deg >= axd_angle_min) & (axd_deg <= axd_angle_max)

    xar = angle_at_vertex(
        hal[:, :, None, :],
        acc[:, None, :, :],
        neigh[:, None, :, :],
    )
    xar_deg = jnp.degrees(xar)
    xar_ok = (xar_deg >= xar_angle_min) & (xar_deg <= xar_angle_max)

    mask = dist_ok & axd_ok & xar_ok
    return mask, dists, axd_deg, xar_deg


# ---------------------------- Ring-based helpers ---------------------------- #

def _ring_centroids_normals_frames(
    coords_f: jnp.ndarray,
    rings: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ring centroids and normals across frames for a list of rings.

    Args:
        coords_f: (F, N, 3) coordinates per frame
        rings: list of index arrays (variable ring sizes allowed)

    Returns:
        centroids: (F, K, 3)
        normals: (F, K, 3) unit normals (cross of first two centroid vectors)
    """
    F = int(coords_f.shape[0])
    centroids_list = []
    normals_list = []
    for ring_idx in rings:
        rc = coords_f[:, ring_idx, :]  # (F, S, 3)
        centroid = rc.mean(axis=1)     # (F, 3)
        v1 = rc[:, 0, :] - centroid
        v2 = rc[:, 1, :] - centroid
        n = jnp.cross(v1, v2)
        n = n / jnp.clip(jnp.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        centroids_list.append(centroid)
        normals_list.append(n)
    if centroids_list:
        centroids = jnp.stack(centroids_list, axis=1)
        normals = jnp.stack(normals_list, axis=1)
    else:
        centroids = jnp.zeros((F, 0, 3))
        normals = jnp.zeros((F, 0, 3))
    return centroids, normals


def cationpi_frames(
    ring_coords_f: jnp.ndarray,
    ring_list: list[jnp.ndarray],
    cation_coords_f: jnp.ndarray,
    cation_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 4.5,
    angle_min: float = 0.0,
    angle_max: float = 90.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched Cation–Pi geometry for one orientation (ring vs cation).

    Returns:
        mask: (F, Kr, Kc)
        distances: (F, Kr, Kc) centroid–cation distances
        angles: (F, Kr, Kc) normal–centroid vector angles (deg)
    """
    F = int(ring_coords_f.shape[0])
    centroids, normals = _ring_centroids_normals_frames(ring_coords_f, ring_list)  # (F, Kr, 3)
    cations = cation_coords_f[:, cation_idx, :] if cation_idx.size else jnp.zeros((F, 0, 3))
    if centroids.shape[1] == 0 or cations.shape[1] == 0:
        z = jnp.zeros((F, centroids.shape[1], cations.shape[1]))
        return z.astype(bool), z, z

    # Distances
    vec = cations[:, None, :, :] - centroids[:, :, None, :]  # (F, Kr, Kc, 3)
    dists = jnp.linalg.norm(vec, axis=-1)
    dist_ok = dists <= distance_cutoff
    # Angle between ring normal and centroid→cation
    ang = angle_between_vectors(normals[:, :, None, :], vec)
    ang_deg = jnp.degrees(ang)
    ang_ok = (ang_deg >= angle_min) & (ang_deg <= angle_max)
    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def pistacking_frames(
    lig_coords_f: jnp.ndarray,
    lig_rings: list[jnp.ndarray],
    res_coords_f: jnp.ndarray,
    res_rings: list[jnp.ndarray],
    *,
    distance_cutoff: float = 6.5,
    plane_angle_min: float = 0.0,
    plane_angle_max: float = 30.0,
    ncc_angle_min: float = 0.0,
    ncc_angle_max: float = 60.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched Pi-stacking geometry for ring–ring pairs.

    Returns:
        mask: (F, Kl, Kr)
        distances: (F, Kl, Kr) centroid–centroid distances
        plane_angles: (F, Kl, Kr) normals plane angle (deg)
        ncc_angles: (F, Kl, Kr) min of the two normal→centroid angles (deg)
    """
    F = int(lig_coords_f.shape[0])
    lc, ln = _ring_centroids_normals_frames(lig_coords_f, lig_rings)  # (F, Kl, 3)
    rc, rn = _ring_centroids_normals_frames(res_coords_f, res_rings)  # (F, Kr, 3)
    if lc.shape[1] == 0 or rc.shape[1] == 0:
        z = jnp.zeros((F, lc.shape[1], rc.shape[1]))
        return z.astype(bool), z, z, z

    # Distances
    cc_vec = rc[:, None, :, :] - lc[:, :, None, :]  # (F, Kl, Kr, 3)
    dists = jnp.linalg.norm(cc_vec, axis=-1)
    dist_ok = dists <= distance_cutoff

    # Plane angle (between normals)
    pa = angle_between_vectors(ln[:, :, None, :], rn[:, None, :, :])
    pa_deg = jnp.degrees(pa)
    pa_ok = (pa_deg >= plane_angle_min) & (pa_deg <= plane_angle_max)

    # Normal→centroid angles (either ring)
    n1 = angle_between_vectors(ln[:, :, None, :], cc_vec)
    n2 = angle_between_vectors(rn[:, None, :, :], -cc_vec)
    n1_deg = jnp.degrees(n1)
    n2_deg = jnp.degrees(n2)
    ncc_deg = jnp.minimum(n1_deg, n2_deg)
    ncc_ok = (ncc_deg >= ncc_angle_min) & (ncc_deg <= ncc_angle_max)

    mask = dist_ok & pa_ok & ncc_ok
    return mask, dists, pa_deg, ncc_deg
