
"""Testes das métricas de avaliação de clusterização do ResearchAtlas."""

import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from research_atlas.evaluation import metrics
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


LEGACY_KEYS = {
    "n_articles",
    "n_clusters",
    "grouped_articles",
    "noise_articles",
    "coverage_pct",
    "noise_pct",
    "min_cluster_size",
    "max_cluster_size",
    "scope",
    "valid",
    "reasons",
    "warnings",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
}


def _sample_misses_cluster(n_articles, sample_size, seed, n_hidden=2):
    """Rótulos em que a amostra perde o cluster pequeno."""
    permutation = np.random.RandomState(seed).permutation(n_articles)

    labels = np.zeros(n_articles, dtype=int)
    labels[permutation[-n_hidden:]] = 1

    assert len(np.unique(labels[permutation[:sample_size]])) == 1

    return labels


def test_legacy_keys_are_preserved(blobs):
    X, y = blobs

    assert LEGACY_KEYS <= set(evaluate_partition(X, y))


def test_result_is_json_serializable(blobs):
    X, y = blobs

    labels = y.copy()
    labels[:10] = -1

    json.dumps(evaluate_partition(X, labels, sample_size=100))
    json.dumps(evaluate_partition(X, np.zeros(len(X), dtype=int)))


def test_metric_geometry_is_recorded(blobs):
    X, y = blobs

    result = evaluate_partition(X, y)

    assert result["silhouette_metric"] == "cosine"
    assert result["davies_bouldin_geometry"] == "euclidean"
    assert result["calinski_harabasz_geometry"] == "euclidean"


def test_silhouette_uses_cosine_distance(blobs):
    X, y = blobs

    result = evaluate_partition(X, y)

    cosine = silhouette_score(X, y, metric="cosine")
    euclidean = silhouette_score(X, y, metric="euclidean")

    assert result["silhouette"] == pytest.approx(cosine)
    assert result["silhouette"] != pytest.approx(euclidean)


def test_full_evaluation_metadata(blobs):
    X, y = blobs

    result = evaluate_partition(X, y, sample_size=2000)

    assert result["n_evaluated"] == 300
    assert result["silhouette_sampling"] == "full"
    assert result["silhouette_sampled"] is False
    assert result["silhouette_sample_size"] == 300
    assert result["sample_size_requested"] == 2000
    assert result["seed"] == 42


def test_sampled_evaluation_matches_sklearn_protocol(blobs):
    X, y = blobs

    result = evaluate_partition(X, y, sample_size=120, seed=7)

    expected = silhouette_score(
        X,
        y,
        metric="cosine",
        sample_size=120,
        random_state=7,
    )

    assert result["silhouette_sampling"] == "sampled"
    assert result["silhouette_sampled"] is True
    assert result["silhouette_sample_size"] == 120
    assert result["silhouette"] == pytest.approx(expected)


def test_metadata_counts_only_grouped_articles(blobs):
    X, y = blobs

    labels = y.copy()
    labels[:30] = -1

    result = evaluate_partition(X, labels, sample_size=100)

    assert result["scope"] == "grouped_only"
    assert result["n_articles"] == 300
    assert result["n_evaluated"] == 270
    assert result["silhouette_sample_size"] == 100


def test_noise_is_excluded_from_every_metric(blobs):
    X, y = blobs

    labels = y.copy()
    labels[:30] = -1
    keep = labels != -1

    result = evaluate_partition(X, labels)
    reference = evaluate_partition(X[keep], y[keep])

    assert result["silhouette"] == pytest.approx(reference["silhouette"])
    assert result["davies_bouldin"] == pytest.approx(
        reference["davies_bouldin"]
    )
    assert result["calinski_harabasz"] == pytest.approx(
        reference["calinski_harabasz"]
    )


def test_invalid_partition_has_no_silhouette_metadata(blobs):
    X, _ = blobs

    result = evaluate_partition(X, np.zeros(len(X), dtype=int))

    assert result["silhouette_sampling"] is None
    assert result["silhouette_sample_size"] is None
    assert result["silhouette_sampled"] is None
    assert result["n_evaluated"] == 300


def test_noise_leaving_single_cluster_is_invalid(blobs):
    """HDBSCAN pode deixar apenas um cluster após remover o ruído."""
    X, y = blobs

    labels = y.copy()
    labels[labels != 0] = -1

    result = evaluate_partition(X, labels)

    assert result["valid"] is False
    assert result["scope"] == "grouped_only"
    assert "menos_de_2_clusters" in result["reasons"]
    assert result["silhouette"] is None


def test_invalid_sample_falls_back_to_all_articles(blobs):
    X, _ = blobs

    labels = _sample_misses_cluster(len(X), sample_size=10, seed=42)

    result = evaluate_partition(X, labels, sample_size=10, seed=42)

    assert result["valid"] is True
    assert result["silhouette_sampling"] == "full_fallback"
    assert result["silhouette_sampled"] is False
    assert result["silhouette_sample_size"] == len(X)
    assert (
        "silhouette_amostra_invalida_base_completa" in result["warnings"]
    )
    assert result["silhouette"] == pytest.approx(
        silhouette_score(X, labels, metric="cosine")
    )
    assert result["davies_bouldin"] is not None
    assert result["calinski_harabasz"] is not None


def test_invalid_sample_without_fallback_keeps_other_metrics(
    blobs,
    monkeypatch,
):
    X, _ = blobs

    monkeypatch.setattr(metrics, "SILHOUETTE_FULL_FALLBACK_MAX", 100)
    labels = _sample_misses_cluster(len(X), sample_size=10, seed=42)

    result = evaluate_partition(X, labels, sample_size=10, seed=42)

    assert result["valid"] is True
    assert result["silhouette"] is None
    assert (
        result["silhouette_sampling"]
        == "nao_calculada_amostra_invalida"
    )
    assert result["silhouette_sample_size"] is None
    assert (
        "silhouette_nao_calculada_amostra_invalida" in result["warnings"]
    )
    assert result["davies_bouldin"] is not None
    assert result["calinski_harabasz"] is not None


def test_sample_with_every_article_unique_label_is_not_used(blobs):
    """n_rótulos == n_amostras na amostra também é inválido."""
    X, _ = blobs

    permutation = np.random.RandomState(42).permutation(len(X))

    labels = np.zeros(len(X), dtype=int)
    labels[permutation[:2]] = [1, 2]

    result = evaluate_partition(X, labels, sample_size=3, seed=42)

    assert result["silhouette_sampling"] == "full_fallback"
    assert result["silhouette"] is not None


def test_numpy_integers_are_accepted(blobs):
    X, y = blobs

    expected = evaluate_partition(X, y, sample_size=150, seed=5)
    result = evaluate_partition(
        X,
        y,
        sample_size=np.int64(150),
        seed=np.int32(5),
    )

    assert result["silhouette"] == pytest.approx(expected["silhouette"])
    assert result["sample_size_requested"] == 150
    assert type(result["sample_size_requested"]) is int
    assert type(result["seed"]) is int


@pytest.mark.parametrize(
    "sample_size",
    [True, 1, 0, -5, 2.5, "100", None, np.float64(100.0)],
)
def test_invalid_sample_size_raises(blobs, sample_size):
    X, y = blobs

    with pytest.raises(ValueError, match="sample_size"):
        evaluate_partition(X, y, sample_size=sample_size)


@pytest.mark.parametrize(
    "seed",
    [True, -1, 2**32, 1.5, "42", None],
)
def test_invalid_seed_raises(blobs, seed):
    X, y = blobs

    with pytest.raises(ValueError, match="seed"):
        evaluate_partition(X, y, seed=seed)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0.0, 1.0] * 150),
        np.array([True, False] * 150),
        np.array(["a", "b"] * 150),
        np.array([0, 1, -2] * 100),
    ],
)
def test_invalid_label_types_raise(blobs, labels):
    X, _ = blobs

    with pytest.raises(ValueError):
        evaluate_partition(X, labels)


def test_coincident_centroids_are_invalid(blobs):
    """Clusters duplicados geram Davies-Bouldin sem sentido (~1e7)."""
    X, _ = blobs

    stacked = np.vstack([X, X])
    labels = np.repeat([0, 1], len(X))

    result = evaluate_partition(stacked, labels)

    assert result["valid"] is False
    assert "centroides_coincidentes" in result["reasons"]
    assert result["silhouette"] is None
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None


def test_zero_intracluster_dispersion_is_invalid():
    """Clusters formados por vetores idênticos não têm dispersão."""
    first = np.tile([1.0, 0.0, 0.0], (20, 1))
    second = np.tile([0.0, 1.0, 0.0], (20, 1))

    X = np.vstack([first, second])
    labels = np.repeat([0, 1], 20)

    result = evaluate_partition(X, labels)

    assert result["valid"] is False
    assert "dispersao_intracluster_nula" in result["reasons"]
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None


def test_constant_cluster_generates_warning(blobs):
    """Um único cluster sem dispersão não invalida a partição."""
    X, y = blobs

    X = X.copy()
    members = y == 0
    X[members] = X[members][0]

    result = evaluate_partition(X, y)

    assert result["valid"] is True
    assert "cluster_sem_dispersao" in result["warnings"]


def test_low_coverage_generates_warning(blobs):
    X, y = blobs

    labels = y.copy()
    labels[:200] = -1

    result = evaluate_partition(X, labels)

    assert result["valid"] is True
    assert result["scope"] == "grouped_only"
    assert result["coverage_pct"] == pytest.approx(100 / 3)
    assert "cobertura_baixa" in result["warnings"]


def test_high_coverage_does_not_generate_warning(blobs):
    X, y = blobs

    labels = y.copy()
    labels[:30] = -1

    assert "cobertura_baixa" not in evaluate_partition(X, labels)[
        "warnings"
    ]
    assert "cobertura_baixa" not in evaluate_partition(X, y)["warnings"]


def test_normalization_is_reported(blobs):
    X, y = blobs

    raw = evaluate_partition(X, y)
    normalized = evaluate_partition(
        X / np.linalg.norm(X, axis=1, keepdims=True),
        y,
    )

    assert raw["input_l2_normalized"] is False
    assert "embeddings_nao_normalizados" in raw["warnings"]

    assert normalized["input_l2_normalized"] is True
    assert "embeddings_nao_normalizados" not in normalized["warnings"]


def test_dense_valid_partition_has_no_unavailable_reasons(blobs):
    X, y = blobs

    result = evaluate_partition(X, y)

    assert result["silhouette_unavailable_reason"] is None
    assert result["davies_bouldin_unavailable_reason"] is None
    assert result["calinski_harabasz_unavailable_reason"] is None
    assert result["input_sparse"] is False


def test_invalid_partition_reports_unavailable_reasons(blobs):
    X, _ = blobs

    result = evaluate_partition(X, np.zeros(len(X), dtype=int))

    assert result["silhouette_unavailable_reason"] == "particao_invalida"
    assert result["davies_bouldin_unavailable_reason"] == "particao_invalida"
    assert (
        result["calinski_harabasz_unavailable_reason"]
        == "particao_invalida"
    )


def test_unavailable_silhouette_reason_after_invalid_sample(
    blobs,
    monkeypatch,
):
    X, _ = blobs

    monkeypatch.setattr(metrics, "SILHOUETTE_FULL_FALLBACK_MAX", 100)
    labels = _sample_misses_cluster(len(X), sample_size=10, seed=42)

    result = evaluate_partition(X, labels, sample_size=10, seed=42)

    assert (
        result["silhouette_unavailable_reason"]
        == "amostra_invalida_sem_fallback"
    )
    assert result["davies_bouldin_unavailable_reason"] is None


def test_sparse_input_computes_only_cosine_silhouette(blobs):
    X, y = blobs

    features = sparse.csr_matrix(np.abs(X) + 0.1)
    result = evaluate_partition(features, y)

    assert result["input_sparse"] is True
    assert result["valid"] is True
    assert result["silhouette"] == pytest.approx(
        silhouette_score(features, y, metric="cosine")
    )
    assert result["davies_bouldin"] is None
    assert result["calinski_harabasz"] is None
    assert result["davies_bouldin_geometry"] is None
    assert result["calinski_harabasz_geometry"] is None
    assert (
        result["davies_bouldin_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    )
    assert (
        result["calinski_harabasz_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    )


def test_sparse_input_is_never_densified(blobs, monkeypatch):
    X, y = blobs

    features = sparse.csr_matrix(np.abs(X) + 0.1)
    received = []
    original = metrics.silhouette_score

    def spy(matrix, labels, **kwargs):
        received.append(sparse.issparse(matrix))
        return original(matrix, labels, **kwargs)

    monkeypatch.setattr(metrics, "silhouette_score", spy)

    evaluate_partition(features, y, sample_size=100)

    assert received == [True]


def test_sparse_input_with_noise_keeps_grouped_only_scope(blobs):
    X, y = blobs

    features = sparse.csr_matrix(np.abs(X) + 0.1)
    labels = y.copy()
    labels[:30] = -1

    result = evaluate_partition(features, labels)

    assert result["scope"] == "grouped_only"
    assert result["n_evaluated"] == 270


def test_sparse_input_with_zero_row_raises(blobs):
    X, y = blobs

    dense = np.abs(X) + 0.1
    dense[0] = 0

    with pytest.raises(ValueError, match="vetores nulos"):
        evaluate_partition(sparse.csr_matrix(dense), y)


def test_sparse_input_with_non_finite_values_raises(blobs):
    X, y = blobs

    dense = np.abs(X) + 0.1
    dense[0, 0] = np.nan

    with pytest.raises(ValueError, match="não finitos"):
        evaluate_partition(sparse.csr_matrix(dense), y)


def test_sparse_label_mismatch_raises(blobs):
    X, y = blobs

    with pytest.raises(ValueError, match="Rótulos incompatíveis"):
        evaluate_partition(sparse.csr_matrix(np.abs(X) + 0.1), y[:-1])
