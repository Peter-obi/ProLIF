"""
Tests for JAX accelerator (Stage 4).

Run with: pytest tests/jax/test_accelerator.py -v
"""

import numpy as np
import pytest


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


class TestJAXAccelerator:
    def test_init_default(self):
        """Test default initialization."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator()
        assert 'Hydrophobic' in accel.interaction_types
        assert 'Cationic' in accel.interaction_types

    def test_init_custom_interactions(self):
        """Test initialization with custom interactions."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic'])
        assert accel.interaction_types == ['Hydrophobic']

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

        accel = JAXAccelerator(interactions=['Hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )
        residue = MockMolecule(
            coords=np.array([[3.0, 0.0, 0.0]]),
            elements=['C']
        )

        result = accel.compute_single(ligand, residue)

        assert 'Hydrophobic' in result
        assert result['Hydrophobic']['mask'].shape == (1, 1)
        assert result['Hydrophobic']['mask'][0, 0] == True

    def test_compute_single_no_contact(self):
        """Test single residue with no contact."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )
        residue = MockMolecule(
            coords=np.array([[10.0, 0.0, 0.0]]),
            elements=['C']
        )

        result = accel.compute_single(ligand, residue)

        assert result['Hydrophobic']['mask'][0, 0] == False

    def test_compute_interactions_multiple_residues(self):
        """Test batch computation with multiple residues."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic'])

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
        assert results[0]['Hydrophobic']['mask'].any()
        assert not results[1]['Hydrophobic']['mask'].any()

    def test_has_interaction(self):
        """Test has_interaction convenience method."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic'])

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

        assert accel.has_interaction(ligand, close_residue, 'Hydrophobic') == True
        assert accel.has_interaction(ligand, far_residue, 'Hydrophobic') == False

    def test_has_interaction_invalid_type(self):
        """Test has_interaction with invalid interaction type."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic'])

        ligand = MockMolecule(coords=np.array([[0.0, 0.0, 0.0]]), elements=['C'])
        residue = MockMolecule(coords=np.array([[3.0, 0.0, 0.0]]), elements=['C'])

        with pytest.raises(ValueError, match="not in configured types"):
            accel.has_interaction(ligand, residue, 'HBDonor')


class TestConvenienceFunctions:
    def test_compute_hydrophobic_fast(self):
        """Test fast hydrophobic computation with numpy arrays."""
        from prolif.interactions._jax.accelerator import compute_hydrophobic_fast

        ligand_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        residue_coords = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        mask, dists = compute_hydrophobic_fast(ligand_coords, residue_coords)

        assert mask.shape == (2, 3)
        assert dists.shape == (2, 3)

        assert mask[0, 0] == True
        assert mask[0, 1] == True
        assert mask[0, 2] == False

    def test_compute_ionic_fast(self):
        """Test fast ionic computation with numpy arrays."""
        from prolif.interactions._jax.accelerator import compute_ionic_fast

        cation_coords = np.array([[0.0, 0.0, 0.0]])
        anion_coords = np.array([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

        mask, dists = compute_ionic_fast(cation_coords, anion_coords)

        assert mask.shape == (1, 2)
        assert mask[0, 0] == True
        assert mask[0, 1] == False


class TestAcceleratorIntegration:
    def test_full_workflow(self):
        """Test complete accelerator workflow."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic', 'Cationic'])

        
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

        assert len(results) == 3

        for r in results:
            assert 'Hydrophobic' in r
            assert 'Cationic' in r

        assert not results[2]['Hydrophobic']['mask'].any()
        assert not results[2]['Cationic']['mask'].any()

    def test_varying_residue_sizes(self):
        """Test with residues of very different sizes."""
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['Hydrophobic'])

        ligand = MockMolecule(
            coords=np.array([[0.0, 0.0, 0.0]]),
            elements=['C']
        )

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
        assert results[0]['Hydrophobic']['mask'].shape == (1, 1)
        assert results[1]['Hydrophobic']['mask'].shape == (1, 5)
        assert results[2]['Hydrophobic']['mask'].shape == (1, 10)
