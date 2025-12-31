"""
Tests for JAX dispatch layer (Stage 3).

Run with: pytest tests/jax/test_dispatch.py -v
"""

import jax.numpy as jnp
import pytest


# =============================================================================
# Task 3.1: prepare_batch tests
# =============================================================================
class TestPrepareBatch:
    def test_single_residue(self):
        """Test batching with single residue."""
        from prolif.interactions._jax.dispatch import prepare_batch

        ligand_coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ligand_elements = ['C', 'C']
        residue_coords_list = [jnp.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])]
        residue_elements_list = [['C', 'N']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        assert 'ligand_coords' in batch
        assert 'residue_coords' in batch
        assert batch['residue_coords'].shape[0] == 1  # 1 residue

    def test_multiple_residues_different_sizes(self):
        """Test batching with residues of different atom counts."""
        from prolif.interactions._jax.dispatch import prepare_batch

        ligand_coords = jnp.array([[0.0, 0.0, 0.0]])
        ligand_elements = ['C']

        # Residue 1: 2 atoms, Residue 2: 4 atoms
        residue_coords_list = [
            jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        ]
        residue_elements_list = [['C', 'C'], ['C', 'C', 'N', 'O']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        # Should be padded to max size (4)
        assert batch['residue_coords'].shape == (2, 4, 3)
        assert 'valid_mask' in batch
        assert batch['valid_mask'].shape == (2, 4)
        # First residue: 2 valid, 2 padded
        assert batch['valid_mask'][0].sum() == 2
        # Second residue: 4 valid, 0 padded
        assert batch['valid_mask'][1].sum() == 4

    def test_preserves_original_sizes(self):
        """Test that original sizes are preserved for unbatching."""
        from prolif.interactions._jax.dispatch import prepare_batch

        ligand_coords = jnp.array([[0.0, 0.0, 0.0]])
        ligand_elements = ['C']
        residue_coords_list = [
            jnp.array([[1.0, 0.0, 0.0]]),
            jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        ]
        residue_elements_list = [['C'], ['C', 'C', 'C']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        assert batch['original_sizes'] == [1, 3]


# =============================================================================
# Task 3.2: run_all_interactions tests
# =============================================================================
class TestRunAllInteractions:
    def test_hydrophobic_only(self):
        """Test running hydrophobic interaction on batch."""
        from prolif.interactions._jax.dispatch import prepare_batch, run_all_interactions

        ligand_coords = jnp.array([[0.0, 0.0, 0.0]])
        ligand_elements = ['C']
        residue_coords_list = [
            jnp.array([[3.0, 0.0, 0.0]]),  # Within 4.5 Å
            jnp.array([[10.0, 0.0, 0.0]]),  # Beyond 4.5 Å
        ]
        residue_elements_list = [['C'], ['C']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        results = run_all_interactions(batch)

        assert 'hydrophobic' in results
        assert results['hydrophobic']['mask'][0].any()  # First residue has contact
        assert not results['hydrophobic']['mask'][1].any()  # Second residue too far

    def test_multiple_interactions(self):
        """Test running multiple interaction types."""
        from prolif.interactions._jax.dispatch import prepare_batch, run_all_interactions

        ligand_coords = jnp.array([[0.0, 0.0, 0.0]])
        ligand_elements = ['C']
        residue_coords_list = [jnp.array([[3.0, 0.0, 0.0]])]
        residue_elements_list = [['C']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic', 'vdw']
        )

        results = run_all_interactions(batch)

        assert 'hydrophobic' in results or 'vdw' in results


# =============================================================================
# Task 3.3: unbatch_results tests
# =============================================================================
class TestUnbatchResults:
    def test_unbatch_single_residue(self):
        """Test unbatching results for single residue."""
        from prolif.interactions._jax.dispatch import (
            prepare_batch, run_all_interactions, unbatch_results
        )

        ligand_coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ligand_elements = ['C', 'C']
        residue_coords_list = [jnp.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])]
        residue_elements_list = [['C', 'C']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        results = run_all_interactions(batch)
        unbatched = unbatch_results(results, batch)

        assert len(unbatched) == 1
        assert 'hydrophobic' in unbatched[0]

    def test_unbatch_multiple_residues(self):
        """Test unbatching results preserves original shapes."""
        from prolif.interactions._jax.dispatch import (
            prepare_batch, run_all_interactions, unbatch_results
        )

        ligand_coords = jnp.array([[0.0, 0.0, 0.0]])
        ligand_elements = ['C']
        residue_coords_list = [
            jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),  # 2 atoms
            jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),  # 3 atoms
        ]
        residue_elements_list = [['C', 'C'], ['C', 'C', 'C']]

        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        results = run_all_interactions(batch)
        unbatched = unbatch_results(results, batch)

        assert len(unbatched) == 2
        assert unbatched[0]['hydrophobic']['mask'].shape == (1, 2)
        assert unbatched[1]['hydrophobic']['mask'].shape == (1, 3)


# =============================================================================
# Integration tests
# =============================================================================
class TestDispatchIntegration:
    def test_full_pipeline(self):
        """Test complete prepare -> run -> unbatch pipeline."""
        from prolif.interactions._jax.dispatch import (
            prepare_batch, run_all_interactions, unbatch_results
        )

        # Setup: ligand near first residue, far from second
        ligand_coords = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        ligand_elements = ['C', 'C']

        residue_coords_list = [
            jnp.array([[3.0, 0.0, 0.0], [3.5, 0.0, 0.0]]),  # Close
            jnp.array([[20.0, 0.0, 0.0]]),  # Far
        ]
        residue_elements_list = [['C', 'C'], ['C']]

        # Run pipeline
        batch = prepare_batch(
            ligand_coords, ligand_elements,
            residue_coords_list, residue_elements_list,
            interaction_types=['hydrophobic']
        )

        results = run_all_interactions(batch)
        unbatched = unbatch_results(results, batch)

        # Verify results
        assert len(unbatched) == 2
        assert unbatched[0]['hydrophobic']['mask'].any()  # Close residue has contacts
        assert not unbatched[1]['hydrophobic']['mask'].any()  # Far residue has none
