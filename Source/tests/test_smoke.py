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


def test_unimplemented_stubs_raise_not_implemented() -> None:
    X = np.zeros((4, 2))
    y = np.zeros(4, dtype=np.intp)
    with pytest.raises(NotImplementedError):
        CLARANS().fit(X)
    with pytest.raises(NotImplementedError):
        make_synthetic_db("db1")
    with pytest.raises(NotImplementedError):
        make_spatial_scaling_dataset(10)
    with pytest.raises(NotImplementedError):
        clustering_summary(y, y)
    with pytest.raises(NotImplementedError):
        measure_runtime_seconds(lambda: 1)
