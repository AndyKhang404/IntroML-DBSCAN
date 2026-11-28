"""Smoke tests for package scaffolding, config loading, and stub interfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dbscan_repro import DEFAULT_SEED, load_config, set_seed
from dbscan_repro.clarans import CLARANS
from dbscan_repro.datasets import make_spatial_scaling_dataset, make_synthetic_db
from dbscan_repro.dbscan import NOISE_LABEL, DBSCANScratch
from dbscan_repro.kdist import compute_sorted_k_distances
from dbscan_repro.metrics import clustering_summary, measure_runtime_seconds


def test_load_default_config() -> None:
    cfg = load_config(Path("Source/experiments/configs/default.yaml"))
    assert cfg.seed == DEFAULT_SEED
    assert cfg.params["runtime"]["sample_sizes"] == [1252, 2503, 3910, 5213, 6256]


def test_set_seed_deterministic() -> None:
    rng1 = set_seed(123)
    rng2 = set_seed(123)
    assert np.allclose(rng1.random(5), rng2.random(5))


def test_dbscan_parameter_validation() -> None:
    with pytest.raises(ValueError):
        DBSCANScratch(eps=0.0)
    with pytest.raises(ValueError):
        DBSCANScratch(min_pts=0)


def test_dbscan_and_kdist_smoke() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]])
    expected = np.array([0, 0, 0, NOISE_LABEL], dtype=np.intp)
    for algorithm in ("brute", "kdtree"):
        labels = DBSCANScratch(eps=0.5, min_pts=3, algorithm=algorithm).fit_predict(X)
        np.testing.assert_array_equal(labels, expected)

    k_dist = compute_sorted_k_distances(X, k=1)
    assert k_dist.shape == (4,)
    assert np.all(np.diff(k_dist) <= 0.0)


def test_m3_interfaces_return_expected_shapes_and_metrics() -> None:
    X, y = make_synthetic_db("db1", n_samples=100, seed=123)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert np.all(np.isfinite(X))
    assert set(np.unique(y)).issubset({-1, 0, 1, 2, 3})
    assert make_spatial_scaling_dataset(25, seed=123).shape == (25, 2)

    model = CLARANS(n_clusters=2, max_neighbor=10, random_state=123).fit(X)
    assert model.labels_ is not None and model.labels_.shape == (100,)
    assert model.medoid_indices_ is not None and len(model.medoid_indices_) == 2

    summary = clustering_summary(y, y)
    assert summary["ari"] == pytest.approx(1.0)
    assert summary["nmi"] == pytest.approx(1.0)
    assert summary["noise_ratio"] == pytest.approx(float(np.mean(y == -1)))
    result, mean, std = measure_runtime_seconds(lambda: 7, repeats=2)
    assert result == 7
    assert mean >= 0.0 and std >= 0.0
