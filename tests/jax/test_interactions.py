"""
Tests for JAX interaction backends.

Run with: pytest tests/jax/test_interactions.py -v
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from prolif.interactions._jax.vdw import vdw_contacts


# =============================================================================
# Task 2.1: vdw_contacts tests
# =============================================================================
class TestVdWContacts:
    def test_atoms_in_contact(self):
        """Two atoms within vdW distance should be detected."""
        lig_coords = jnp.array([[0.0, 0.0, 0.0]])
        lig_radii = jnp.array([1.7])
        res_coords = jnp.array([[2.5, 0.0, 0.0]])
        res_radii = jnp.array([1.7])

        mask, dists = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii)

        assert mask.shape == (1, 1)
        assert dists.shape == (1, 1)
        assert mask[0, 0] == True  # 2.5 < 1.7 + 1.7 = 3.4
        assert_allclose(dists[0, 0], 2.5)

    def test_atoms_not_in_contact(self):
        """Two atoms beyond vdW distance should not be detected."""
        lig_coords = jnp.array([[0.0, 0.0, 0.0]])
        lig_radii = jnp.array([1.7])
        res_coords = jnp.array([[5.0, 0.0, 0.0]])
        res_radii = jnp.array([1.7])

        mask, dists = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii)

        assert mask[0, 0] == False  # 5.0 > 1.7 + 1.7 = 3.4
        assert_allclose(dists[0, 0], 5.0)

    def test_tolerance(self):
        """Tolerance should extend contact distance."""
        lig_coords = jnp.array([[0.0, 0.0, 0.0]])
        lig_radii = jnp.array([1.7])
        res_coords = jnp.array([[3.5, 0.0, 0.0]])  # Just beyond 3.4
        res_radii = jnp.array([1.7])

        # Without tolerance: no contact
        mask1, _ = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii, tolerance=0.0)
        assert mask1[0, 0] == False

        # With tolerance: contact
        mask2, _ = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii, tolerance=0.5)
        assert mask2[0, 0] == True  # 3.5 <= 3.4 + 0.5 = 3.9

    def test_multiple_atoms(self):
        """Test with multiple ligand and residue atoms."""
        lig_coords = jnp.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        lig_radii = jnp.array([1.7, 1.7])

        res_coords = jnp.array([
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])
        res_radii = jnp.array([1.7, 1.7, 1.7])

        mask, dists = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii)

        assert mask.shape == (2, 3)
        assert dists.shape == (2, 3)

        # First ligand atom contacts first two residue atoms
        assert mask[0, 0] == True   # dist 2.0 < 3.4
        assert mask[0, 1] == True   # dist 3.0 < 3.4
        assert mask[0, 2] == False  # dist 20.0 > 3.4

        # Second ligand atom contacts none
        assert mask[1, 0] == False  # dist 8.0 > 3.4
        assert mask[1, 1] == False  # dist 7.0 > 3.4
        assert mask[1, 2] == False  # dist 10.0 > 3.4

    def test_different_radii(self):
        """Test with different vdW radii per atom."""
        lig_coords = jnp.array([[0.0, 0.0, 0.0]])
        lig_radii = jnp.array([1.2])

        res_coords = jnp.array([
            [2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ])
        res_radii = jnp.array([1.0, 1.5])

        mask, _ = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii)

        # 1.2 + 1.0 = 2.2 < 2.5 → no contact
        assert mask[0, 0] == False
        # 1.2 + 1.5 = 2.7 > 2.5 → contact
        assert mask[0, 1] == True


# =============================================================================
# Task 2.2: hydrophobic_contacts tests
# =============================================================================
class TestHydrophobicContacts:
    def test_atoms_in_contact(self):
        """Two atoms within distance cutoff should be detected."""
        from prolif.interactions._jax.hydrophobic import hydrophobic_contacts

        lig_coords = jnp.array([[0.0, 0.0, 0.0]])
        res_coords = jnp.array([[4.0, 0.0, 0.0]])

        mask, dists = hydrophobic_contacts(lig_coords, res_coords, distance_cutoff=4.5)

        assert mask[0, 0] == True  # 4.0 <= 4.5
        assert_allclose(dists[0, 0], 4.0)

    def test_atoms_not_in_contact(self):
        """Two atoms beyond distance cutoff should not be detected."""
        from prolif.interactions._jax.hydrophobic import hydrophobic_contacts

        lig_coords = jnp.array([[0.0, 0.0, 0.0]])
        res_coords = jnp.array([[5.0, 0.0, 0.0]])

        mask, dists = hydrophobic_contacts(lig_coords, res_coords, distance_cutoff=4.5)

        assert mask[0, 0] == False  # 5.0 > 4.5

    def test_multiple_atoms(self):
        """Test with multiple atoms."""
        from prolif.interactions._jax.hydrophobic import hydrophobic_contacts

        lig_coords = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        res_coords = jnp.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        mask, _ = hydrophobic_contacts(lig_coords, res_coords, distance_cutoff=4.5)

        assert mask.shape == (2, 2)
        assert mask[0, 0] == True   # 3.0 <= 4.5
        assert mask[0, 1] == True   # 4.0 <= 4.5
        assert mask[1, 0] == False  # 7.0 > 4.5
        assert mask[1, 1] == False  # 6.0 > 4.5


# =============================================================================
# Task 2.3: hbond_contacts tests
# =============================================================================
class TestHBondContacts:
    def test_good_hbond(self):
        """Linear H-bond should be detected (180° angle)."""
        from prolif.interactions._jax.hbond import hbond_contacts

        # Linear arrangement: Acceptor ... H - Donor
        acceptor = jnp.array([[0.0, 0.0, 0.0]])
        hydrogen = jnp.array([[2.0, 0.0, 0.0]])
        donor = jnp.array([[3.0, 0.0, 0.0]])

        mask, dists, angles = hbond_contacts(acceptor, donor, hydrogen)

        assert mask[0, 0] == True
        assert_allclose(dists[0, 0], 3.0, atol=1e-5)  # acceptor-donor distance
        assert_allclose(angles[0, 0], 180.0, atol=1e-3)  # linear

    def test_bent_hbond_accepted(self):
        """H-bond with acceptable angle (150°) should be detected."""
        from prolif.interactions._jax.hbond import hbond_contacts

        # Acceptor at angle from D-H axis
        acceptor = jnp.array([[0.0, 0.0, 0.0]])
        hydrogen = jnp.array([[1.5, 0.0, 0.0]])
        # Donor placed so D-H-A is ~150°
        donor = jnp.array([[2.5, 0.5, 0.0]])

        mask, _, angles = hbond_contacts(acceptor, donor, hydrogen, angle_min=130.0)

        # Angle should be around 150°, which is within 130-180
        assert angles[0, 0] > 130.0
        assert mask[0, 0] == True

    def test_bent_hbond_rejected(self):
        """H-bond with bad angle (<130°) should be rejected."""
        from prolif.interactions._jax.hbond import hbond_contacts

        # Very bent arrangement
        acceptor = jnp.array([[0.0, 0.0, 0.0]])
        hydrogen = jnp.array([[1.0, 0.0, 0.0]])
        donor = jnp.array([[1.0, 1.0, 0.0]])  # 90° angle

        mask, _, angles = hbond_contacts(acceptor, donor, hydrogen, angle_min=130.0)

        assert angles[0, 0] < 130.0  # Should be ~90°
        assert mask[0, 0] == False

    def test_too_far(self):
        """H-bond beyond distance cutoff should be rejected."""
        from prolif.interactions._jax.hbond import hbond_contacts

        acceptor = jnp.array([[0.0, 0.0, 0.0]])
        hydrogen = jnp.array([[4.0, 0.0, 0.0]])
        donor = jnp.array([[5.0, 0.0, 0.0]])

        mask, dists, _ = hbond_contacts(acceptor, donor, hydrogen, distance_cutoff=3.5)

        assert dists[0, 0] > 3.5
        assert mask[0, 0] == False


# =============================================================================
# Task 2.4: ionic_contacts tests
# =============================================================================
class TestIonicContacts:
    def test_ions_in_contact(self):
        """Ions within distance should be detected."""
        from prolif.interactions._jax.ionic import ionic_contacts

        cation = jnp.array([[0.0, 0.0, 0.0]])
        anion = jnp.array([[4.0, 0.0, 0.0]])

        mask, dists = ionic_contacts(cation, anion, distance_cutoff=4.5)

        assert mask[0, 0] == True
        assert_allclose(dists[0, 0], 4.0)

    def test_ions_not_in_contact(self):
        """Ions beyond distance should not be detected."""
        from prolif.interactions._jax.ionic import ionic_contacts

        cation = jnp.array([[0.0, 0.0, 0.0]])
        anion = jnp.array([[5.0, 0.0, 0.0]])

        mask, _ = ionic_contacts(cation, anion, distance_cutoff=4.5)

        assert mask[0, 0] == False


# =============================================================================
# Task 2.5: xbond_contacts tests
# =============================================================================
class TestXBondContacts:
    def test_good_xbond(self):
        """Proper halogen bond geometry should be detected."""
        from prolif.interactions._jax.xbond import xbond_contacts

        # Linear A...X-D arrangement with ~90° X-A-R
        acceptor = jnp.array([[0.0, 0.0, 0.0]])
        acceptor_neighbor = jnp.array([[0.0, 1.5, 0.0]])  # R perpendicular
        halogen = jnp.array([[3.0, 0.0, 0.0]])
        donor = jnp.array([[4.5, 0.0, 0.0]])

        mask, dists, axd_angles, xar_angles = xbond_contacts(
            acceptor, acceptor_neighbor, halogen, donor
        )

        assert_allclose(dists[0, 0], 3.0, atol=1e-5)
        assert_allclose(axd_angles[0, 0], 180.0, atol=1e-3)  # linear A-X-D
        assert_allclose(xar_angles[0, 0], 90.0, atol=1e-3)   # perpendicular X-A-R
        assert mask[0, 0] == True

    def test_xbond_bad_axd_angle(self):
        """X-bond with bad A-X-D angle should be rejected."""
        from prolif.interactions._jax.xbond import xbond_contacts

        acceptor = jnp.array([[0.0, 0.0, 0.0]])
        acceptor_neighbor = jnp.array([[0.0, 1.5, 0.0]])
        halogen = jnp.array([[2.0, 0.0, 0.0]])
        donor = jnp.array([[2.0, 1.5, 0.0]])  # 90° A-X-D angle

        mask, _, axd_angles, _ = xbond_contacts(acceptor, acceptor_neighbor, halogen, donor)

        assert axd_angles[0, 0] < 130.0
        assert mask[0, 0] == False


# =============================================================================
# Task 2.6: cation_pi_contacts tests
# =============================================================================
class TestCationPiContacts:
    def test_cation_above_ring(self):
        """Cation directly above ring center should be detected."""
        from prolif.interactions._jax.cation_pi import cation_pi_contacts

        # Benzene-like ring in XY plane
        ring_coords = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ])
        ring_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        # Cation 3.5 Å above center
        cation = jnp.array([[0.0, 0.0, 3.5]])

        mask, dists, angles = cation_pi_contacts(cation, ring_coords, ring_indices)

        assert mask.shape == (1, 1)
        assert_allclose(dists[0, 0], 3.5, atol=1e-5)
        assert angles[0, 0] < 30.0  # Should be ~0° (directly above)
        assert mask[0, 0] == True

    def test_cation_beside_ring(self):
        """Cation beside ring (not above) should be rejected."""
        from prolif.interactions._jax.cation_pi import cation_pi_contacts

        ring_coords = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ])
        ring_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        # Cation to the side (in XY plane)
        cation = jnp.array([[4.0, 0.0, 0.0]])

        mask, _, angles = cation_pi_contacts(cation, ring_coords, ring_indices)

        assert angles[0, 0] > 30.0  # Should be ~90°
        assert mask[0, 0] == False

    def test_cation_too_far(self):
        """Cation too far from ring should be rejected."""
        from prolif.interactions._jax.cation_pi import cation_pi_contacts

        ring_coords = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ])
        ring_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        # Cation 6 Å above (beyond 4.5 Å cutoff)
        cation = jnp.array([[0.0, 0.0, 6.0]])

        mask, dists, _ = cation_pi_contacts(cation, ring_coords, ring_indices)

        assert dists[0, 0] > 4.5
        assert mask[0, 0] == False


# =============================================================================
# Task 2.7: pi_stacking_contacts tests
# =============================================================================
class TestPiStackingContacts:
    def test_parallel_rings_stacked(self):
        """Parallel rings stacked above each other (face-to-face)."""
        from prolif.interactions._jax.pi_stacking import pi_stacking_contacts

        # Ring 1 in XY plane at z=0
        ring1_coords = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ])
        ring1_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        # Ring 2 parallel, 3.5 Å above
        ring2_coords = jnp.array([
            [1.0, 0.0, 3.5],
            [0.5, 0.866, 3.5],
            [-0.5, 0.866, 3.5],
            [-1.0, 0.0, 3.5],
            [-0.5, -0.866, 3.5],
            [0.5, -0.866, 3.5],
        ])
        ring2_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        mask, dists, plane_angles, ncc_angles = pi_stacking_contacts(
            ring1_coords, ring1_indices, ring2_coords, ring2_indices
        )

        assert mask.shape == (1, 1)
        assert_allclose(dists[0, 0], 3.5, atol=1e-5)
        assert plane_angles[0, 0] < 35.0  # Parallel
        assert ncc_angles[0, 0] < 33.0    # Stacked directly above
        assert mask[0, 0] == True

    def test_perpendicular_rings(self):
        """Perpendicular rings should not be detected as face-to-face."""
        from prolif.interactions._jax.pi_stacking import pi_stacking_contacts

        # Ring 1 in XY plane
        ring1_coords = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ])
        ring1_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        # Ring 2 in XZ plane
        ring2_coords = jnp.array([
            [1.0, 3.5, 0.0],
            [0.5, 3.5, 0.866],
            [-0.5, 3.5, 0.866],
            [-1.0, 3.5, 0.0],
            [-0.5, 3.5, -0.866],
            [0.5, 3.5, -0.866],
        ])
        ring2_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        mask, _, plane_angles, _ = pi_stacking_contacts(
            ring1_coords, ring1_indices, ring2_coords, ring2_indices,
            plane_angle_min=0.0, plane_angle_max=35.0  # face-to-face criteria
        )

        assert plane_angles[0, 0] > 35.0  # Perpendicular ~90°
        assert mask[0, 0] == False

    def test_rings_too_far(self):
        """Rings too far apart should not be detected."""
        from prolif.interactions._jax.pi_stacking import pi_stacking_contacts

        ring1_coords = jnp.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ])
        ring1_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        # Ring 2 parallel but 8 Å away
        ring2_coords = ring1_coords + jnp.array([0.0, 0.0, 8.0])
        ring2_indices = [jnp.array([0, 1, 2, 3, 4, 5])]

        mask, dists, _, _ = pi_stacking_contacts(
            ring1_coords, ring1_indices, ring2_coords, ring2_indices,
            distance_cutoff=5.5
        )

        assert dists[0, 0] > 5.5
        assert mask[0, 0] == False
