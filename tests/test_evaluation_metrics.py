import numpy as np
import pytest
from sklearn.datasets import make_blobs

from research_atlas.evaluation.metrics import evaluate_partition


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, n_features=8, random_state=0)
    return X, y


def test_valid_partition_returns_all_metrics(blobs):
    X, y = blobs
    result = evaluate_partition(X, y)

    assert result["valid"] is True
    assert result["scope"] == "full"
    assert result["silhouette"] is not None
    assert result["davies_bouldin"] is not None
    assert result["calinski_harabasz"] is not None


def test_single_cluster_is_invalid(blobs):
    X, _ = blobs
    result = evaluate_partition(X, np.zeros(len(X), dtype=int))
    # TODO: valid deve ser False, as 3 métricas None e "menos_de_2_clusters" em reasons
    assert result["valid"] is False
    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None
    assert "menos_de_2_clusters" in result["reasons"]


def test_all_noise_is_invalid(blobs):
    X, _ = blobs
    # TODO: todos os rótulos iguais a -1; o que esperar de "valid" e de "reasons"?
    result = evaluate_partition(X, -np.ones(len(X), dtype=int))
    assert result["valid"] is False
    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None
    assert "agrupados_insuficientes" in result["reasons"]


def test_noise_marks_scope_as_grouped_only(blobs):
    X, y = blobs
    y = y.copy()
    y[:30] = -1
    # TODO: scope deve ser "grouped_only" e as métricas devem existir
    result = evaluate_partition(X, y)
    assert result["scope"] == "grouped_only"
    assert result["silhouette"] is not None
    assert result["davies_bouldin"] is not None
    assert result["calinski_harabasz"] is not None


def test_label_length_mismatch_raises(blobs):
    X, y = blobs
    # TODO: use pytest.raises(ValueError) com rótulos de tamanho diferente
    with pytest.raises(ValueError):
        evaluate_partition(X, y[:-1])