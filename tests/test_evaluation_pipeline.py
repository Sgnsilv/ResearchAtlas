"""Testes do pipeline de avaliação consolidada."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from research_atlas.evaluation import metrics
from research_atlas.evaluation.pipeline import (
    TABLE_COLUMNS,
    evaluate_bge,
    evaluate_tfidf,
    label_columns,
    parse_bge_config,
    parse_tfidf_config,
    sha256_file,
    validate_ids,
    validate_order,
    verify_sha256,
)


@pytest.fixture
def embeddings_and_labels():
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        n_features=8,
        random_state=0,
    )

    return normalize(X), y


@pytest.fixture
def assignments(embeddings_and_labels):
    _, y = embeddings_and_labels

    noisy_leaf = y.copy()
    noisy_leaf[:100] = -1

    noisy_eom = y.copy()
    noisy_eom[:150] = -1

    return pd.DataFrame(
        {
            "embedding_index": np.arange(len(y)),
            "openalex_id": [f"W{i}" for i in range(len(y))],
            "sampling_area": "Medicina",
            "kmeans_k3": y,
            "agglomerative_ward_k3": y,
            "hdbscan_leaf_mcs20_ms5": noisy_leaf,
            "hdbscan_eom_mcs50_ms10": noisy_eom,
        }
    )


@pytest.fixture
def sparse_features():
    rng = np.random.default_rng(0)

    dense = np.abs(rng.normal(size=(200, 50)))
    dense *= rng.random((200, 50)) < 0.2
    dense[:, 0] = 1.0

    return sparse.csr_matrix(normalize(dense))


def test_parse_bge_configurations():
    assert parse_bge_config("kmeans_k8") == {
        "algorithm": "kmeans",
        "k": 8,
        "linkage": None,
        "selection_method": None,
        "hdbscan_min_cluster_size": None,
        "hdbscan_min_samples": None,
        "clustering_distance": "euclidean",
    }

    ward = parse_bge_config("agglomerative_ward_k8")
    average = parse_bge_config("agglomerative_average_k8")

    assert ward["clustering_distance"] == "euclidean"
    assert average["clustering_distance"] == "cosine"
    assert ward["linkage"] == "ward"

    leaf = parse_bge_config("hdbscan_leaf_mcs20_ms5")

    assert leaf["algorithm"] == "hdbscan"
    assert leaf["selection_method"] == "leaf"
    assert leaf["hdbscan_min_cluster_size"] == 20
    assert leaf["hdbscan_min_samples"] == 5
    assert leaf["k"] is None


@pytest.mark.parametrize(
    "config_id",
    [
        "kmeans8",
        "kmeans_k8_extra",
        "agglomerative_single_k8",
        "hdbscan_median_mcs20_ms5",
        "tfidf_10k_k8",
    ],
)
def test_unknown_bge_configuration_raises(config_id):
    with pytest.raises(ValueError, match="Configuração desconhecida"):
        parse_bge_config(config_id)


def test_parse_tfidf_configuration():
    parsed = parse_tfidf_config("tfidf_20k_k13")

    assert parsed["vocabulary"] == "20k"
    assert parsed["k"] == 13
    assert parsed["algorithm"] == "kmeans"

    with pytest.raises(ValueError, match="Configuração desconhecida"):
        parse_tfidf_config("tfidf_30k_k5")


def test_label_columns_exclude_identifiers(assignments):
    assert label_columns(assignments) == [
        "kmeans_k3",
        "agglomerative_ward_k3",
        "hdbscan_leaf_mcs20_ms5",
        "hdbscan_eom_mcs50_ms10",
    ]


def test_label_columns_reject_non_integer_labels(assignments):
    assignments["kmeans_k3"] = assignments["kmeans_k3"].astype(float)

    with pytest.raises(ValueError, match="não inteiros"):
        label_columns(assignments)


def test_label_columns_require_at_least_one_configuration(assignments):
    with pytest.raises(ValueError, match="Nenhuma configuração"):
        label_columns(assignments[["embedding_index", "openalex_id"]])


def test_validate_order_rejects_shuffled_index(assignments):
    shuffled = assignments.iloc[::-1].reset_index(drop=True)
    shuffled["embedding_index"] = assignments["embedding_index"].to_numpy()[::-1]

    with pytest.raises(ValueError, match="fora da ordem"):
        validate_order(shuffled, len(assignments))

    with pytest.raises(ValueError, match="Quantidade"):
        validate_order(assignments, len(assignments) + 1)


def test_validate_ids_detects_divergence(assignments):
    ids = assignments["openalex_id"]

    assert validate_ids(ids, len(ids), ids).equals(ids)

    swapped = ids.copy()
    swapped.iloc[[0, 1]] = swapped.iloc[[1, 0]].to_numpy()

    with pytest.raises(ValueError, match="divergentes"):
        validate_ids(swapped, len(ids), ids)

    duplicated = ids.copy()
    duplicated.iloc[1] = duplicated.iloc[0]

    with pytest.raises(ValueError, match="duplicados"):
        validate_ids(duplicated, len(ids))


def test_verify_sha256(tmp_path):
    path = tmp_path / "artefato.bin"
    path.write_bytes(b"researchatlas")

    verify_sha256(path, sha256_file(path))

    with pytest.raises(ValueError, match="Checksum divergente"):
        verify_sha256(path, "0" * 64)


def test_bge_table_structure(embeddings_and_labels, assignments):
    X, _ = embeddings_and_labels

    table = evaluate_bge(X, assignments, sample_size=2000, seed=42)

    assert list(table.columns) == TABLE_COLUMNS
    assert table["config_id"].tolist() == label_columns(assignments)
    assert (table["representation"] == "bge-small-en-v1.5").all()
    assert (table["feature_space"] == "bge-small-en-v1.5:8d_l2").all()
    assert (table["silhouette_metric"] == "cosine").all()
    assert (table["davies_bouldin_geometry"] == "euclidean").all()
    assert (table["calinski_harabasz_geometry"] == "euclidean").all()
    assert table["input_l2_normalized"].all()
    assert not table["input_sparse"].any()
    metric_columns = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    assert table[metric_columns].notna().all().all()


def test_bge_scope_and_coverage_are_preserved(
    embeddings_and_labels,
    assignments,
):
    X, _ = embeddings_and_labels

    table = evaluate_bge(X, assignments).set_index("config_id")

    assert table.loc["kmeans_k3", "scope"] == "full"
    assert table.loc["kmeans_k3", "noise_articles"] == 0

    leaf = table.loc["hdbscan_leaf_mcs20_ms5"]

    assert leaf["scope"] == "grouped_only"
    assert leaf["noise_articles"] == 100
    assert leaf["grouped_articles"] == 200
    assert leaf["n_evaluated"] == 200
    assert leaf["coverage_pct"] == pytest.approx(200 / 3)
    assert leaf["selection_method"] == "leaf"


def test_comparability_groups_never_mix_scopes(
    embeddings_and_labels,
    assignments,
):
    X, _ = embeddings_and_labels

    table = evaluate_bge(X, assignments).set_index("config_id")
    groups = table["comparability_group"]

    assert groups["kmeans_k3"] == groups["agglomerative_ward_k3"]
    assert groups["hdbscan_leaf_mcs20_ms5"] != groups["hdbscan_eom_mcs50_ms10"]
    assert groups["kmeans_k3"] != groups["hdbscan_leaf_mcs20_ms5"]
    assert "grouped_only" in groups["hdbscan_leaf_mcs20_ms5"]


def test_sampling_metadata_is_preserved(embeddings_and_labels, assignments):
    X, y = embeddings_and_labels

    table = evaluate_bge(X, assignments, sample_size=120, seed=7)
    row = table.set_index("config_id").loc["kmeans_k3"]

    assert row["silhouette_sampling"] == "sampled"
    assert row["silhouette_sample_size"] == 120
    assert bool(row["silhouette_sampled"]) is True
    assert row["sample_size_requested"] == 120
    assert row["seed"] == 7
    assert row["silhouette"] == pytest.approx(
        silhouette_score(
            X,
            y,
            metric="cosine",
            sample_size=120,
            random_state=7,
        )
    )
    assert "sampled:120:seed7" in row["comparability_group"]


def test_bge_evaluation_is_deterministic(embeddings_and_labels, assignments):
    X, _ = embeddings_and_labels

    first = evaluate_bge(X, assignments, sample_size=100, seed=3)
    second = evaluate_bge(X, assignments, sample_size=100, seed=3)

    assert first.equals(second)


def test_invalid_partition_is_reported_without_error(
    embeddings_and_labels,
    assignments,
):
    X, _ = embeddings_and_labels

    assignments["hdbscan_eom_mcs50_ms10"] = -1
    table = evaluate_bge(X, assignments).set_index("config_id")
    row = table.loc["hdbscan_eom_mcs50_ms10"]

    assert bool(row["valid"]) is False
    assert row["reasons"] == "sem_artigos_agrupados"
    assert row["scope"] == "grouped_only"
    assert pd.isna(row["silhouette"])
    assert row["silhouette_unavailable_reason"] == "particao_invalida"
    assert row["davies_bouldin_unavailable_reason"] == "particao_invalida"
    assert row["calinski_harabasz_unavailable_reason"] == "particao_invalida"


def test_misaligned_labels_raise(embeddings_and_labels, assignments):
    X, _ = embeddings_and_labels

    with pytest.raises(ValueError, match="Rótulos incompatíveis"):
        evaluate_bge(X[:-1], assignments)


def test_tfidf_computes_only_silhouette(sparse_features):
    labels = np.arange(sparse_features.shape[0]) % 3
    assignments = pd.DataFrame({"tfidf_10k_k3": labels})

    table = evaluate_tfidf(
        {"10k": sparse_features},
        assignments,
        sample_size=2000,
        seed=42,
    )
    row = table.iloc[0]

    assert row["representation"] == "tfidf"
    assert row["feature_space"] == "tfidf:50f_sparse"
    assert row["n_features"] == 50
    assert bool(row["input_sparse"]) is True
    assert row["silhouette"] == pytest.approx(
        silhouette_score(sparse_features, labels, metric="cosine")
    )
    assert pd.isna(row["davies_bouldin"])
    assert pd.isna(row["calinski_harabasz"])
    assert (
        row["davies_bouldin_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    )
    assert (
        row["calinski_harabasz_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    )
    assert pd.isna(row["davies_bouldin_geometry"])
    assert pd.isna(row["calinski_harabasz_geometry"])
    assert pd.isna(row["silhouette_unavailable_reason"])


def test_tfidf_silhouette_receives_sparse_matrix(
    sparse_features,
    monkeypatch,
):
    received = []
    original = metrics.silhouette_score

    def spy(X, labels, **kwargs):
        received.append(sparse.issparse(X))
        return original(X, labels, **kwargs)

    monkeypatch.setattr(metrics, "silhouette_score", spy)

    labels = np.arange(sparse_features.shape[0]) % 3

    evaluate_tfidf(
        {"10k": sparse_features},
        pd.DataFrame({"tfidf_10k_k3": labels}),
        sample_size=80,
    )

    assert received == [True]


def test_tfidf_unknown_configuration_raises(sparse_features):
    labels = np.arange(sparse_features.shape[0]) % 3

    with pytest.raises(ValueError, match="Configuração desconhecida"):
        evaluate_tfidf(
            {"10k": sparse_features},
            pd.DataFrame({"tfidf_5k_k3": labels}),
        )

    with pytest.raises(KeyError):
        evaluate_tfidf(
            {"10k": sparse_features},
            pd.DataFrame({"tfidf_20k_k3": labels}),
        )
