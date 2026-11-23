"""Orientation sets must match Dynamo's GPU engine exactly (counts it prints, float32 semantics)."""
import numpy as np

from pydynamo.engine.angles import ScanRange, fill_interval, orientation_set, rotation_matrices


def test_local_counts_match_dynamo_gpu_engine():
    rng = ScanRange(30, 10, 30, 10)
    assert len(orientation_set((0, 0, 0), rng)) == 21          # "multigrid step: 0 totaling 21 triplets"
    rng.shrink(2)
    assert len(orientation_set((0, 0, 0), rng)) == 95          # "multigrid step: 1 totaling 95 triplets"


def test_global_360_15_count_matches_dynamo():
    assert len(orientation_set((0, 0, 0), ScanRange(360, 15, 360, 15))) == 4633   # Dynamo engine log


def test_refinement_counts_are_seed_dependent_like_dynamo():
    rng = ScanRange(30, 10, 30, 10)
    rng.shrink(2)
    counts = [len(orientation_set(tuple(np.random.RandomState(i).uniform([-180, 0, -180], [180, 180, 180])), rng))
              for i in range(50)]
    assert min(counts) >= 85 and max(counts) == 95              # float32 truncation drops end points


def test_fill_interval_float32_truncation():
    assert len(fill_interval(0, 27.6923, 360 - 27.6923)) in (12, 13)
    assert fill_interval(0, 0, 10) == [0]


def test_rotation_matrices_are_orthonormal_and_zxz():
    M = rotation_matrices(np.array([[10.0, 20.0, 30.0], [0, 0, 0]])).reshape(-1, 3, 3).astype(np.float64)
    for m in M:
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(m), 1, atol=1e-6)
    assert np.allclose(M[1], np.eye(3))
    # Dynamo's matrix for (10, 20, 30) (dynamo_euler2matrix, checked in the MATLAB Runtime)
    ref = np.array([[0.7713, -0.6131, 0.1710], [0.6337, 0.7146, -0.2962], [0.0594, 0.3368, 0.9397]])
    assert np.allclose(M[0], ref, atol=1e-4)
