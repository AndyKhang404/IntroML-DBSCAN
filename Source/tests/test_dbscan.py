"""Unit and scikit-learn equivalence tests for the from-scratch DBSCAN (M2)."""

from __future__ import annotations

import numpy as np
import pytest
from dbscan_repro.dbscan import NOISE_LABEL, DBSCANScratch
from dbscan_repro.kdist import (
    compute_sorted_k_distances,
    detect_kdist_valley_index,
    suggest_eps_from_noise_ratio,
)
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

ALGORITHMS = ("brute", "kdtree")


def _make_datasets() -> dict[str, np.ndarray]:
    blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.35, random_state=42)
    moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    circles, _ = make_circles(n_samples=300, factor=0.5, noise=0.04, random_state=42)
    rng = np.random.default_rng(42)
    uniform = rng.uniform(-3.0, 3.0, size=(200, 2))
    return {"blobs": blobs, "moons": moons, "circles": circles, "uniform": uniform}


DATASETS = _make_datasets()
DATASET_SETTINGS = {
    "blobs": (0.8, 5),
    "moons": (0.3, 5),
    "circles": (0.3, 5),
    "uniform": (0.3, 4),
}


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("eps", [0.2, 0.5])
def test_brute_and_kdtree_are_identical(name: str, eps: float) -> None:
    X = DATASETS[name]
    brute = DBSCANScratch(eps=eps, min_pts=4, algorithm="brute").fit(X)
    kdtree = DBSCANScratch(eps=eps, min_pts=4, algorithm="kdtree").fit(X)

    np.testing.assert_array_equal(brute.labels_, kdtree.labels_)
    np.testing.assert_array_equal(brute.core_sample_indices_, kdtree.core_sample_indices_)


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_matches_sklearn(name: str, algorithm: str) -> None:
    eps, min_pts = DATASET_SETTINGS[name]
    X = DATASETS[name]

    ours = DBSCANScratch(eps=eps, min_pts=min_pts, algorithm=algorithm).fit(X)
    ref = DBSCAN(eps=eps, min_samples=min_pts, algorithm="brute").fit(X)

    np.testing.assert_array_equal(ours.labels_, ref.labels_)
    np.testing.assert_array_equal(ours.core_sample_indices_, ref.core_sample_indices_)
    np.testing.assert_array_equal(ours.labels_ == NOISE_LABEL, ref.labels_ == NOISE_LABEL)


def test_border_point_assignment_depends_on_scan_order() -> None:
    left = [(-1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5)]
    border = [(0.0, 0.0)]
    right = [(1.0, 0.0), (1.0, 0.5), (1.0, -0.5)]
    X = np.array(left + border + right)

    # Forward order: the left cluster is discovered first, so it owns the
    # shared border point (KDD-96, Section 4.1).
    forward = DBSCANScratch(eps=1.0, min_pts=4, algorithm="brute").fit_predict(X)
    assert forward[3] == forward[0]
    assert forward[3] != forward[4]

    # Reverse order: the right cluster is discovered first and claims it.
    permutation = [4, 5, 6, 3, 0, 1, 2]
    backward_perm = DBSCANScratch(eps=1.0, min_pts=4, algorithm="brute").fit_predict(X[permutation])
    assert backward_perm[3] == backward_perm[0]
    assert backward_perm[3] != backward_perm[4]

    backward = np.empty_like(backward_perm)
    backward[permutation] = backward_perm
    assert adjusted_rand_score(forward, backward) < 1.0

    # Core points and noise are order independent, and both runs match sklearn.
    for data in (X, X[permutation]):
        ours = DBSCANScratch(eps=1.0, min_pts=4, algorithm="brute").fit(data)
        ref = DBSCAN(eps=1.0, min_samples=4, algorithm="brute").fit(data)
        np.testing.assert_array_equal(ours.labels_, ref.labels_)
        np.testing.assert_array_equal(ours.core_sample_indices_, ref.core_sample_indices_)


def test_all_points_noise_when_min_pts_exceeds_neighborhoods() -> None:
    X = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    labels = DBSCANScratch(eps=0.5, min_pts=2).fit_predict(X)
    np.testing.assert_array_equal(labels, NOISE_LABEL)


def test_min_pts_one_labels_every_density_chain() -> None:
    X = np.array([[0.0], [0.5], [1.0], [5.0]])
    labels = DBSCANScratch(eps=0.6, min_pts=1).fit_predict(X)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1])


def test_single_cluster_with_large_eps() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    model = DBSCANScratch(eps=10.0, min_pts=2).fit(X)

    np.testing.assert_array_equal(model.labels_, [0, 0, 0])
    np.testing.assert_array_equal(model.core_sample_indices_, [0, 1, 2])


def test_duplicate_points_are_clustered_together() -> None:
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    labels = DBSCANScratch(eps=0.1, min_pts=3).fit_predict(X)
    np.testing.assert_array_equal(labels, [0, 0, 0, NOISE_LABEL])


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        DBSCANScratch().fit(np.zeros(5))
    with pytest.raises(ValueError):
        DBSCANScratch().fit(np.empty((0, 2)))
    with pytest.raises(ValueError):
        DBSCANScratch().fit(np.array([[0.0, np.nan]]))
    with pytest.raises(ValueError):
        DBSCANScratch(algorithm="balltree")  # type: ignore[arg-type]


def test_fit_predict_matches_fit_labels() -> None:
    X = DATASETS["blobs"]
    model = DBSCANScratch(eps=0.8, min_pts=5)
    predicted = model.fit_predict(X)

    assert model.labels_ is not None
    np.testing.assert_array_equal(predicted, model.labels_)


def test_k_dist_excludes_the_point_itself() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k_dist = compute_sorted_k_distances(square, k=1)

    np.testing.assert_allclose(k_dist, np.ones(4))


def test_k_dist_sorted_descending_with_known_values() -> None:
    line = np.arange(10, dtype=np.float64).reshape(-1, 1)
    k_dist = compute_sorted_k_distances(line, k=2)

    np.testing.assert_allclose(k_dist, [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.all(np.diff(k_dist) <= 0.0)


def test_k_dist_validation() -> None:
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        compute_sorted_k_distances(X, k=0)
    with pytest.raises(ValueError):
        compute_sorted_k_distances(X, k=3)
    with pytest.raises(ValueError):
        compute_sorted_k_distances(np.zeros(3), k=1)


def test_suggest_eps_from_noise_ratio() -> None:
    curve = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    index, eps = suggest_eps_from_noise_ratio(curve, noise_ratio=0.2)
    assert index == 2
    assert eps == 8.0
    assert suggest_eps_from_noise_ratio(curve, noise_ratio=0.0) == (0, 10.0)

    with pytest.raises(ValueError):
        suggest_eps_from_noise_ratio(curve, noise_ratio=1.0)
    with pytest.raises(ValueError):
        suggest_eps_from_noise_ratio(np.array([]), noise_ratio=0.1)


def test_detect_kdist_valley_index() -> None:
    curve = np.array([5.0, 4.8, 4.6, 1.0, 0.8, 0.6, 0.4])

    index, eps = detect_kdist_valley_index(curve)
    assert index == 3
    assert eps == 1.0
    assert detect_kdist_valley_index(np.full(5, 2.0)) == (0, 2.0)

    with pytest.raises(ValueError):
        detect_kdist_valley_index(np.array([1.0, 0.5]))
