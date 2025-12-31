"""
JAX-accelerated helpers for ProLIF.

Expose only the minimal, stable API:
- JAX availability flag
- pairwise_distances primitive
- simple integration helpers that use JAX for distances
"""

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    from .primitives import pairwise_distances
    from .integration import compute_distances_batch, has_interaction_batch
    from .framebatch import (
        pairwise_distances_frames,
        hbacceptor_frames,
        hbdonor_frames,
        xbacceptor_frames,
        xbdonor_frames,
        cationpi_frames,
        pistacking_frames,
    )

    __all__ = [
        "JAX_AVAILABLE",
        "pairwise_distances",
        "compute_distances_batch",
        "has_interaction_batch",
        "pairwise_distances_frames",
        "hbacceptor_frames",
        "hbdonor_frames",
        "xbacceptor_frames",
        "xbdonor_frames",
        "cationpi_frames",
        "pistacking_frames",
    ]
else:
    __all__ = ["JAX_AVAILABLE"]
