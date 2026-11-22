"""Testes dos algoritmos de clusterização semântica."""

import numpy as np
import pytest

from research_atlas.clustering.algorithms import (
    run_agglomerative,
    run_hdbscan,
    run_kmeans,
    summarize_clusters,
    validate_embeddings,
)


@pytest.fixture
def sample_embeddings():
    rng = np.random.default_rng(42)

    centers = np.eye(3, dtype=np.float32)

    groups = [
        center + rng.normal(0, 0.03, size=(30, 3))
        for center in centers
    ]

    matrix = np.vstack(groups).astype(np.float32)

    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_validate_embeddings(sample_embeddings):
    result = validate_embeddings(sample_embeddings)
    assert result.shape == (90, 3)


def test_reject_non_finite_embeddings(sample_embeddings):
    sample_embeddings[0, 0] = np.nan

    with pytest.raises(ValueError, match="não finitos"):
        validate_embeddings(sample_embeddings)


def test_reject_zero_embeddings(sample_embeddings):
    sample_embeddings[0] = 0

    with pytest.raises(ValueError, match="vetores nulos"):
        validate_embeddings(sample_embeddings)


def test_kmeans_reproducibility(sample_embeddings):
    labels_a, _ = run_kmeans(sample_embeddings, k=3, seed=42)
    labels_b, _ = run_kmeans(sample_embeddings, k=3, seed=42)

    np.testing.assert_array_equal(labels_a, labels_b)


def test_agglomerative_ward(sample_embeddings):
    labels, _ = run_agglomerative(
        sample_embeddings,
        k=3,
        linkage="ward",
        metric="euclidean",
    )

    assert len(labels) == 90
    assert len(np.unique(labels)) == 3


def test_agglomerative_rejects_invalid_ward_metric(sample_embeddings):
    with pytest.raises(ValueError, match="Ward exige"):
        run_agglomerative(
            sample_embeddings,
            k=3,
            linkage="ward",
            metric="cosine",
        )


def test_hdbscan_returns_labels(sample_embeddings):
    labels, model = run_hdbscan(
        sample_embeddings,
        min_cluster_size=10,
        min_samples=5,
    )

    assert len(labels) == 90
    assert set(labels).issubset(set(range(90)) | {-1})
    assert len(model.probabilities_) == 90


def test_summary_without_noise():
    labels = np.array([0, 0, 1, 1, 1])

    result = summarize_clusters(labels)

    assert result["n_clusters"] == 2
    assert result["coverage_pct"] == 100
    assert result["noise_articles"] == 0
    assert result["max_cluster_size"] == 3


def test_summary_with_noise():
    labels = np.array([-1, -1, 0, 0, 1])

    result = summarize_clusters(labels)

    assert result["n_clusters"] == 2
    assert result["grouped_articles"] == 3
    assert result["noise_articles"] == 2
    assert result["coverage_pct"] == 60
    assert result["noise_pct"] == 40


def test_summary_all_noise():
    result = summarize_clusters(np.array([-1, -1, -1]))

    assert result["n_clusters"] == 0
    assert result["coverage_pct"] == 0
    assert result["min_cluster_size"] == 0
