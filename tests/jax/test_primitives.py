"""
Tests for JAX geometric primitives.

Run with: pytest tests/jax/test_primitives.py -v
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from prolif.interactions._jax.primitives import (
    angle_at_vertex,
    angle_between_vectors,
    batch_centroids,
    batch_ring_normals,
    pairwise_distances,
    point_to_plane_distance,
    ring_normal,
)


class TestPairwiseDistances:
    def test_same_points(self):
        """Distance from point to itself should be 0."""
        coords = jnp.array([[0.0, 0.0, 0.0]])
        result = pairwise_distances(coords, coords)
        assert result.shape == (1, 1)
        assert_allclose(result[0, 0], 0.0, atol=1e-6)

    def test_unit_distances(self):
        """Distance from origin to unit vectors should be 1."""
        origin = jnp.array([[0.0, 0.0, 0.0]])
        units = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = pairwise_distances(origin, units)
        assert result.shape == (1, 3)
        assert_allclose(result[0], [1.0, 1.0, 1.0], atol=1e-6)

    def test_2x2_distances(self):
        """Test 2x2 distance matrix."""
        coords1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = pairwise_distances(coords1, coords2)

        assert result.shape == (2, 2)
        assert_allclose(result[0, 0], 0.0, atol=1e-6)
        assert_allclose(result[0, 1], 1.0, atol=1e-6)
        assert_allclose(result[1, 0], 1.0, atol=1e-6)
        assert_allclose(result[1, 1], jnp.sqrt(2), atol=1e-6)

    def test_different_shapes(self):
        """Test with different N and M."""
        coords1 = jnp.array([[0.0, 0.0, 0.0]])
        coords2 = jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        result = pairwise_distances(coords1, coords2)
        assert result.shape == (1, 3)
        assert_allclose(result[0], [1.0, 1.0, 1.0], atol=1e-6)


class TestBatchCentroids:
    def test_single_point_centroid(self):
        """Centroid of a single point is the point itself."""
        coords = jnp.array([[1.0, 2.0, 3.0]])
        indices = [jnp.array([0])]
        result = batch_centroids(coords, indices)
        assert result.shape == (1, 3)
        assert_allclose(result[0], [1.0, 2.0, 3.0], atol=1e-6)

    def test_two_triangles(self, two_triangles):
        """Test centroids of two triangles."""
        coords, indices = two_triangles
        result = batch_centroids(coords, indices)

        assert result.shape == (2, 3)
        assert_allclose(result[0], [1.0, 1 / 3, 0.0], atol=1e-6)
        assert_allclose(result[1], [11.0, 31 / 3, 10.0], atol=1e-6)


class TestRingNormal:
    def test_xy_plane_ring(self, xy_plane_ring):
        """Ring in XY plane should have normal along Z-axis."""
        indices = jnp.array([0, 1, 2, 3])
        result = ring_normal(xy_plane_ring, indices)

        assert result.shape == (3,)
        assert_allclose(jnp.abs(result), [0.0, 0.0, 1.0], atol=1e-6)

    def test_xz_plane_ring(self):
        """Ring in XZ plane should have normal along Y-axis."""
        coords = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        indices = jnp.array([0, 1, 2])
        result = ring_normal(coords, indices)

        assert result.shape == (3,)
        assert_allclose(jnp.abs(result), [0.0, 1.0, 0.0], atol=1e-6)

    def test_normal_is_unit_vector(self, xy_plane_ring):
        """Normal vector should have unit length."""
        indices = jnp.array([0, 1, 2, 3])
        result = ring_normal(xy_plane_ring, indices)
        assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)


class TestBatchRingNormals:
    def test_two_rings(self):
        """Test normals for two rings in different planes."""
        coords = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        ring_indices = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
        result = batch_ring_normals(coords, ring_indices)

        assert result.shape == (2, 3)
        assert_allclose(jnp.abs(result[0]), [0.0, 0.0, 1.0], atol=1e-6)
        assert_allclose(jnp.abs(result[1]), [0.0, 1.0, 0.0], atol=1e-6)


class TestAngleBetweenVectors:
    def test_perpendicular_vectors(self):
        """90 degrees between perpendicular vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 1.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert_allclose(result, jnp.pi / 2, atol=1e-6)

    def test_parallel_vectors(self):
        """0 degrees between parallel vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([2.0, 0.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert_allclose(result, 0.0, atol=1e-6)

    def test_opposite_vectors(self):
        """180 degrees between opposite vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([-1.0, 0.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert_allclose(result, jnp.pi, atol=1e-6)

    def test_45_degrees(self):
        """45 degrees test."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([1.0, 1.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert_allclose(result, jnp.pi / 4, atol=1e-6)

    def test_batch_vectors(self):
        """Test with batch of vectors."""
        v1 = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v2 = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        result = angle_between_vectors(v1, v2)

        assert result.shape == (2,)
        assert_allclose(result[0], jnp.pi / 2, atol=1e-6)
        assert_allclose(result[1], 0.0, atol=1e-6)


class TestAngleAtVertex:
    def test_right_angle(self):
        """90 degree angle at origin."""
        p1 = jnp.array([1.0, 0.0, 0.0])
        vertex = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([0.0, 1.0, 0.0])
        result = angle_at_vertex(p1, vertex, p2)
        assert_allclose(result, jnp.pi / 2, atol=1e-6)

    def test_straight_line(self):
        """180 degree angle (straight line)."""
        p1 = jnp.array([-1.0, 0.0, 0.0])
        vertex = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([1.0, 0.0, 0.0])
        result = angle_at_vertex(p1, vertex, p2)
        assert_allclose(result, jnp.pi, atol=1e-6)

    def test_acute_angle(self):
        """60 degree angle (equilateral triangle vertex)."""
        p1 = jnp.array([1.0, 0.0, 0.0])
        vertex = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([0.5, jnp.sqrt(3) / 2, 0.0])
        result = angle_at_vertex(p1, vertex, p2)
        assert_allclose(result, jnp.pi / 3, atol=1e-6)

    def test_translated_vertex(self):
        """Test with vertex not at origin."""
        p1 = jnp.array([11.0, 10.0, 10.0])
        vertex = jnp.array([10.0, 10.0, 10.0])
        p2 = jnp.array([10.0, 11.0, 10.0])
        result = angle_at_vertex(p1, vertex, p2)
        assert_allclose(result, jnp.pi / 2, atol=1e-6)


class TestPointToPlaneDistance:
    def test_points_above_and_below(self):
        """Test points above, below, and on XY plane."""
        points = jnp.array(
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, -3.0],
                [1.0, 1.0, 0.0],
            ]
        )
        plane_point = jnp.array([0.0, 0.0, 0.0])
        plane_normal = jnp.array([0.0, 0.0, 1.0])

        result = point_to_plane_distance(points, plane_point, plane_normal)
        assert result.shape == (3,)
        assert_allclose(result, [5.0, -3.0, 0.0], atol=1e-6)

    def test_tilted_plane(self):
        """Test with a tilted plane (normal at 45 degrees)."""
        points = jnp.array([[1.0, 0.0, 1.0]])
        plane_point = jnp.array([0.0, 0.0, 0.0])
        plane_normal = jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2)

        result = point_to_plane_distance(points, plane_point, plane_normal)
        assert_allclose(result[0], jnp.sqrt(2), atol=1e-6)

    def test_single_point(self):
        """Test with single point."""
        points = jnp.array([[0.0, 0.0, 10.0]])
        plane_point = jnp.array([0.0, 0.0, 0.0])
        plane_normal = jnp.array([0.0, 0.0, 1.0])

        result = point_to_plane_distance(points, plane_point, plane_normal)
        assert result.shape == (1,)
        assert_allclose(result[0], 10.0, atol=1e-6)


class TestEdgeCases:
    def test_very_small_distances(self):
        """Test numerical stability with very small distances."""
        coords1 = jnp.array([[0.0, 0.0, 0.0]])
        coords2 = jnp.array([[1e-10, 0.0, 0.0]])
        result = pairwise_distances(coords1, coords2)
        assert result[0, 0] >= 0
        assert jnp.isfinite(result[0, 0])

    def test_angle_near_zero(self):
        """Test angle calculation for nearly parallel vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([1.0, 1e-10, 0.0])
        result = angle_between_vectors(v1, v2)
        assert jnp.isfinite(result)
        assert result >= 0
        assert result <= jnp.pi

    def test_angle_near_pi(self):
        """Test angle calculation for nearly opposite vectors."""
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([-1.0, 1e-10, 0.0])
        result = angle_between_vectors(v1, v2)
        assert jnp.isfinite(result)
        assert result >= 0
        assert result <= jnp.pi
