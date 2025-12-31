"""
JAX-accelerated interaction detection for ProLIF.

This module provides GPU/CPU accelerated geometric calculations
for molecular interaction fingerprinting.

Usage:
    from prolif.interactions._jax import JAX_AVAILABLE

    if JAX_AVAILABLE:
        from prolif.interactions._jax import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic', 'HBAcceptor'])
        results = accel.compute_interactions(ligand, protein_residues)
"""

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    from .primitives import (
        pairwise_distances,
        batch_centroids,
        ring_normal,
        batch_ring_normals,
        angle_between_vectors,
        angle_at_vertex,
        point_to_plane_distance,
    )

    from .dispatch import (
        prepare_batch,
        run_all_interactions,
        unbatch_results,
    )

    from .accelerator import (
        JAXAccelerator,
        compute_hydrophobic_fast,
        compute_ionic_fast,
    )

    from .hydrophobic import hydrophobic_contacts
    from .cationic import cationic_contacts, anionic_contacts
    from .metal import metaldonor_contacts, metalacceptor_contacts
    from .vdwcontact import vdwcontact_contacts, vdwcontact_contacts_precomputed

    from .hbacceptor import hbacceptor_contacts, hbdonor_contacts

    from .xbacceptor import xbacceptor_contacts, xbdonor_contacts

    from .cationpi import cationpi_contacts, pication_contacts
    from .pistacking import (
        facetoface_contacts,
        edgetoface_contacts,
        pistacking_contacts,
    )

    __all__ = [
        "JAX_AVAILABLE",
        "pairwise_distances",
        "batch_centroids",
        "ring_normal",
        "batch_ring_normals",
        "angle_between_vectors",
        "angle_at_vertex",
        "point_to_plane_distance",
        "prepare_batch",
        "run_all_interactions",
        "unbatch_results",
        "JAXAccelerator",
        "compute_hydrophobic_fast",
        "compute_ionic_fast",
        "hydrophobic_contacts",
        "cationic_contacts",
        "anionic_contacts",
        "metaldonor_contacts",
        "metalacceptor_contacts",
        "vdwcontact_contacts",
        "vdwcontact_contacts_precomputed",
        "hbacceptor_contacts",
        "hbdonor_contacts",
        "xbacceptor_contacts",
        "xbdonor_contacts",
        "cationpi_contacts",
        "pication_contacts",
        "facetoface_contacts",
        "edgetoface_contacts",
        "pistacking_contacts",
    ]
else:
    __all__ = ["JAX_AVAILABLE"]
