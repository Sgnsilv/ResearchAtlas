
"""Testes das métricas de avaliação de clusterização do ResearchAtlas."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from research_atlas.evaluation.metrics import evaluate_partition


@pytest.fixture
def blobs():
    """Gera dados sintéticos com três grupos."""
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        n_features=8,
        random_state=0,
    )
    return X, y


def test_valid_partition_returns_all_metrics(blobs):
    X, y = blobs

    result = evaluate_partition(X, y)

    assert result["valid"] is True
    assert result["scope"] == "full"
    assert result["n_clusters"] == 3
    assert result["noise_articles"] == 0
    assert result["reasons"] == []

    assert result["silhouette"] is not None
    assert result["davies_bouldin"] is not None
    assert result["calinski_harabasz"] is not None


def test_single_cluster_is_invalid(blobs):
    X, _ = blobs

    labels = np.zeros(len(X), dtype=int)
    result = evaluate_partition(X, labels)

    assert result["valid"] is False
    assert result["scope"] == "full"
    assert "menos_de_2_clusters" in result["reasons"]

    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None


def test_all_noise_is_invalid(blobs):
    X, _ = blobs

    labels = -np.ones(len(X), dtype=int)
    result = evaluate_partition(X, labels)

    assert result["valid"] is False
    assert result["scope"] == "grouped_only"
    assert result["n_clusters"] == 0
    assert result["grouped_articles"] == 0
    assert result["noise_articles"] == len(X)

    assert "sem_artigos_agrupados" in result["reasons"]

    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None


def test_noise_marks_scope_as_grouped_only(blobs):
    X, y = blobs

    labels = y.copy()
    labels[:30] = -1

    result = evaluate_partition(X, labels)

    assert result["valid"] is True
    assert result["scope"] == "grouped_only"
    assert result["noise_articles"] == 30
    assert result["grouped_articles"] == 270

    assert result["silhouette"] is not None
    assert result["davies_bouldin"] is not None
    assert result["calinski_harabasz"] is not None


def test_label_length_mismatch_raises(blobs):
    X, y = blobs

    with pytest.raises(
        ValueError,
        match="Rótulos incompatíveis",
    ):
        evaluate_partition(X, y[:-1])


def test_every_article_in_own_cluster_is_invalid(blobs):
    """Métricas não são válidas quando n_clusters == n_amostras."""
    X, _ = blobs

    labels = np.arange(len(X), dtype=int)
    result = evaluate_partition(X, labels)

    assert result["valid"] is False
    assert "clusters_insuficientes_para_metricas" in result["reasons"]

    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None


def test_singleton_cluster_generates_warning(blobs):
    """Um cluster unitário deve gerar aviso."""
    X, y = blobs

    labels = y.copy()
    labels[0] = 3

    result = evaluate_partition(X, labels)

    assert result["valid"] is True
    assert "cluster_unitario" in result["warnings"]


def test_dominant_cluster_generates_warning(blobs):
    """Um cluster com mais de 80% dos artigos gera aviso."""
    X, _ = blobs

    labels = np.zeros(len(X), dtype=int)
    labels[-20:] = 1

    result = evaluate_partition(X, labels)

    assert result["valid"] is True
    assert "cluster_dominante" in result["warnings"]


def test_invalid_partition_preserves_summary(blobs):
    """Mesmo sem métricas válidas, o resumo deve ser retornado."""
    X, _ = blobs

    labels = np.zeros(len(X), dtype=int)
    result = evaluate_partition(X, labels)

    assert result["n_articles"] == 300
    assert result["n_clusters"] == 1
    assert result["grouped_articles"] == 300
    assert result["coverage_pct"] == pytest.approx(100.0)
    assert result["noise_pct"] == pytest.approx(0.0)


def test_metrics_are_reproducible(blobs):
    """Mesma semente e amostragem devem produzir as mesmas métricas."""
    X, y = blobs

    first = evaluate_partition(X, y, sample_size=200, seed=42)
    second = evaluate_partition(X, y, sample_size=200, seed=42)

    assert first["silhouette"] == pytest.approx(
        second["silhouette"]
    )
    assert first["davies_bouldin"] == pytest.approx(
        second["davies_bouldin"]
    )
    assert first["calinski_harabasz"] == pytest.approx(
        second["calinski_harabasz"]
    )
