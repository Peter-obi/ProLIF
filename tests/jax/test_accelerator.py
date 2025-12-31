"""
Tests for JAX accelerator (Stage 4).

Run with: pytest tests/jax/test_accelerator.py -v
"""

import numpy as np
import pytest


# =============================================================================
# Helper class to simulate RDKit molecule
# =============================================================================
class MockMolecule:
    """Mock molecule class for testing without RDKit dependency."""

    def __init__(self, coords: np.ndarray, elements: list[str]):
        self._coords = coords
        self._elements = elements

    @property
    def xyz(self):
        return self._coords

    def GetNumAtoms(self):
        return len(self._elements)

    def GetAtomWithIdx(self, idx):
        class MockAtom:
            def __init__(self, symbol):
                self._symbol = symbol

            def GetSymbol(self):
                return self._symbol

        return MockAtom(self._elements[idx])


# =============================================================================
# JAXAccelerator tests
# =============================================================================
class TestJAXAccelerator:
    def test_init_default(self):
        """Test default initialization."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator()
        assert 'hydrophobic' in accel.interaction_types
        assert 'ionic' in accel.interaction_types

    def test_init_custom_interactions(self):
        """Test initialization with custom interactions."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])
        assert accel.interaction_types == ['hydrophobic']

    def test_extract_coords(self):
        """Test coordinate extraction from mock molecule."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator()
        mol = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            elements=['C', 'N']
        )

        coords, elements = accel.extract_coords(mol)

        assert coords.shape == (2, 3)
        assert elements == ['C', 'N']

    def test_compute_single_hydrophobic(self):
        """Test single residue hydrophobic computation."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )
        residue = MockMolecule(
            coords=np.array([[3.0, 0.0, 0.0]]),  # Within 4.5 Å
            elements=['C']
        )

        result = accel.compute_single(ligand, residue)

        assert 'hydrophobic' in result
        assert result['hydrophobic']['mask'].shape == (1, 1)
        assert result['hydrophobic']['mask'][0, 0] == True

    def test_compute_single_no_contact(self):
        """Test single residue with no contact."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )
        residue = MockMolecule(
            coords=np.array([[10.0, 0.0, 0.0]]),  # Beyond 4.5 Å
            elements=['C']
        )

        result = accel.compute_single(ligand, residue)

        assert result['hydrophobic']['mask'][0, 0] == False

    def test_compute_interactions_multiple_residues(self):
        """Test batch computation with multiple residues."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            elements=['C', 'C']
        )
        residues = [
            MockMolecule(
                coords=np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
                elements=['C', 'C']
            ),
            MockMolecule(
                coords=np.array([[20.0, 0.0, 0.0]]),
                elements=['C']
            ),
        ]

        results = accel.compute_interactions(ligand, residues)

        assert len(results) == 2
        # First residue should have contacts
        assert results[0]['hydrophobic']['mask'].any()
        # Second residue should have no contacts
        assert not results[1]['hydrophobic']['mask'].any()

    def test_has_interaction(self):
        """Test has_interaction convenience method."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )
        close_residue = MockMolecule(
            coords=np.array([[3.0, 0.0, 0.0]]),
            elements=['C']
        )
        far_residue = MockMolecule(
            coords=np.array([[10.0, 0.0, 0.0]]),
            elements=['C']
        )

        assert accel.has_interaction(ligand, close_residue, 'hydrophobic') == True
        assert accel.has_interaction(ligand, far_residue, 'hydrophobic') == False

    def test_has_interaction_invalid_type(self):
        """Test has_interaction with invalid interaction type."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])

        ligand = MockMolecule(coords=np.array([[0.0, 0.0, 0.0]]), elements=['C'])
        residue = MockMolecule(coords=np.array([[3.0, 0.0, 0.0]]), elements=['C'])

        with pytest.raises(ValueError, match="not in configured types"):
            accel.has_interaction(ligand, residue, 'hbond')


# =============================================================================
# Convenience function tests
# =============================================================================
class TestConvenienceFunctions:
    def test_compute_hydrophobic_fast(self):
        """Test fast hydrophobic computation with numpy arrays."""
        from prolif.interactions._jax.accelerator import compute_hydrophobic_fast

        ligand_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        residue_coords = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        mask, dists = compute_hydrophobic_fast(ligand_coords, residue_coords)

        assert mask.shape == (2, 3)
        assert dists.shape == (2, 3)

        # First ligand atom: 3.0, 4.0, 10.0 distances
        assert mask[0, 0] == True   # 3.0 <= 4.5
        assert mask[0, 1] == True   # 4.0 <= 4.5
        assert mask[0, 2] == False  # 10.0 > 4.5

    def test_compute_ionic_fast(self):
        """Test fast ionic computation with numpy arrays."""
        from prolif.interactions._jax.accelerator import compute_ionic_fast

        cation_coords = np.array([[0.0, 0.0, 0.0]])
        anion_coords = np.array([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

        mask, dists = compute_ionic_fast(cation_coords, anion_coords)

        assert mask.shape == (1, 2)
        assert mask[0, 0] == True   # 4.0 <= 4.5
        assert mask[0, 1] == False  # 5.0 > 4.5


# =============================================================================
# Integration tests
# =============================================================================
class TestAcceleratorIntegration:
    def test_full_workflow(self):
        """Test complete accelerator workflow."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic', 'ionic'])

        # Create a small "protein-ligand" system
        ligand = MockMolecule(
            coords=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]),
            elements=['C', 'C', 'N']
        )

        residues = [
            MockMolecule(
                coords=np.array([
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]),
                elements=['C', 'C']
            ),
            MockMolecule(
                coords=np.array([
                    [3.0, 3.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [5.0, 5.0, 0.0],
                ]),
                elements=['O', 'N', 'C']
            ),
            MockMolecule(
                coords=np.array([
                    [50.0, 0.0, 0.0],
                ]),
                elements=['C']
            ),
        ]

        results = accel.compute_interactions(ligand, residues)

        # Should have results for all 3 residues
        assert len(results) == 3

        # Each result should have both interaction types
        for r in results:
            assert 'hydrophobic' in r
            assert 'ionic' in r

        # Third residue (far away) should have no contacts
        assert not results[2]['hydrophobic']['mask'].any()
        assert not results[2]['ionic']['mask'].any()

    def test_varying_residue_sizes(self):
        """Test with residues of very different sizes."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )

        # Residues with sizes 1, 5, and 10 atoms
        residues = [
            MockMolecule(
                coords=np.array([[3.0, 0.0, 0.0]]),
                elements=['C']
            ),
            MockMolecule(
                coords=np.array([[3.0, i, 0.0] for i in range(5)]),
                elements=['C'] * 5
            ),
            MockMolecule(
                coords=np.array([[3.0, i, 0.0] for i in range(10)]),
                elements=['C'] * 10
            ),
        ]

        results = accel.compute_interactions(ligand, residues)

        assert len(results) == 3
        # Original shapes should be preserved after unbatching
        assert results[0]['hydrophobic']['mask'].shape == (1, 1)
        assert results[1]['hydrophobic']['mask'].shape == (1, 5)
        assert results[2]['hydrophobic']['mask'].shape == (1, 10)
