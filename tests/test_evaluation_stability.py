"""Testes unitários de estabilidade entre sementes e concordância por ARI."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from research_atlas.evaluation.stability import (
    CONCORDANCE_COLUMNS,
    IDENTICAL_PAIR_COLUMNS,
    IDENTICAL_PARTITIONS_COLUMNS,
    build_concordance_table,
    build_full_ari_matrix,
    build_identical_partition_pairs,
    canonical_partition_signature,
    classify_comparison,
    compare_partitions,
    find_identical_partitions,
)


@pytest.fixture
def synthetic_frames():
    bge = pd.DataFrame(
        {
            "embedding_index": np.arange(12),
            "openalex_id": [f"W{i}" for i in range(12)],
            "sampling_area": "Medicina",
            "kmeans_k3": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "agglomerative_ward_k3": [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30],
            "hdbscan_eom_mcs20_ms5": [0, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1],
        }
    )
    tfidf = pd.DataFrame(
        {
            "embedding_index": np.arange(12),
            "openalex_id": [f"W{i}" for i in range(12)],
            "sampling_area": "Medicina",
            "tfidf_10k_k3": [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
        }
    )
    return bge, tfidf


def test_canonical_signature_is_invariant_to_label_permutation():
    first = np.array([0, 0, 1, 1, -1, 2, 2])
    permuted = np.array([5, 5, 2, 2, -1, 0, 0])
    noise_changed = np.array([0, 0, 1, 1, 3, -1, 2])

    assert canonical_partition_signature(first) == canonical_partition_signature(
        permuted
    )
    assert canonical_partition_signature(first) != canonical_partition_signature(
        noise_changed
    )


def test_compare_partitions_full_matches_sklearn_and_serializes():
    a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0])

    result = compare_partitions(a, b)

    assert result["valid"] is True
    assert result["comparison_scope"] == "full"
    assert result["n_intersection"] == 9
    assert result["intersection_coverage_pct"] == pytest.approx(100.0)
    assert result["ari"] == pytest.approx(adjusted_rand_score(a, b))
    assert result["ari_unavailable_reason"] is None
    assert result["reasons"] == []
    assert result["warnings"] == []
    json.dumps(result)


def test_compare_partitions_is_symmetric():
    a = np.array([0, 0, 0, -1, 1, 1, 2, 2, -1, 2])
    b = np.array([1, 1, -1, -1, 0, 0, 2, 0, 2, 2])

    ab = compare_partitions(a, b)
    ba = compare_partitions(b, a)

    assert ab["ari"] == pytest.approx(ba["ari"])
    assert ab["valid"] == ba["valid"]
    assert ab["n_intersection"] == ba["n_intersection"]
    assert ab["intersection_coverage_pct"] == pytest.approx(
        ba["intersection_coverage_pct"]
    )
    assert ab["n_grouped_a"] == ba["n_grouped_b"]
    assert ab["n_clusters_a_intersection"] == ba["n_clusters_b_intersection"]
    assert ab["reasons"] == ba["reasons"]
    assert ab["warnings"] == ba["warnings"]


def test_hdbscan_ari_uses_only_grouped_intersection():
    a = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, -1, -1, -1])
    b = np.array([10, 10, 20, 20, -1, -1, 30, 30, -1, -1, -1, -1])

    result = compare_partitions(a, b)

    assert result["valid"] is True
    assert result["comparison_scope"] == "grouped_intersection"
    assert result["n_grouped_a"] == 6
    assert result["n_grouped_b"] == 6
    assert result["n_intersection"] == 4
    assert result["intersection_coverage_pct"] == pytest.approx(100 * 4 / 12)
    assert result["n_clusters_a_total"] == 3
    assert result["n_clusters_a_intersection"] == 2
    assert result["n_clusters_b_total"] == 3
    assert result["n_clusters_b_intersection"] == 2
    assert result["ari"] == pytest.approx(1.0)
    assert result["identical_partition"] is False
    assert set(result["warnings"]) == {
        "intersecao_parcial",
        "cobertura_intersecao_baixa",
        "clusters_ausentes_na_intersecao",
    }


@pytest.mark.parametrize(
    ("a", "b", "expected_reason"),
    [
        (
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 1, 1, 2, 2],
            "intersecao_vazia",
        ),
        (
            [0, 0, 1, -1, -1, -1],
            [-1, -1, -1, 0, 1, 1],
            "intersecao_vazia",
        ),
        (
            [0, -1, -1, -1, -1, -1],
            [0, 0, 1, 1, -1, -1],
            "intersecao_insuficiente",
        ),
        (
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, -1, -1],
            "menos_de_2_clusters_na_intersecao",
        ),
        (
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            "menos_de_2_clusters_na_intersecao",
        ),
        (
            [0, 1, 2, 3, -1, -1],
            [0, 0, 1, 1, -1, -1],
            "clusters_insuficientes_na_intersecao",
        ),
    ],
)
def test_degenerate_or_insufficient_intersections_are_invalid(
    a,
    b,
    expected_reason,
):
    result = compare_partitions(np.array(a), np.array(b))

    assert result["valid"] is False
    assert result["ari"] is None
    assert result["ari_unavailable_reason"] == expected_reason
    assert expected_reason in result["reasons"]


def test_min_intersection_size_accepts_numpy_integer():
    a = np.array([0, 0, 1, 1, -1, -1, -1, -1])
    b = np.array([0, 0, 1, 1, -1, -1, -1, -1])

    result = compare_partitions(a, b, min_intersection_size=np.int64(5))

    assert result["valid"] is False
    assert result["ari"] is None
    assert result["ari_unavailable_reason"] == "intersecao_abaixo_do_minimo"


@pytest.mark.parametrize("min_size", [True, 1, 0, -3, 2.5, "5", None])
def test_invalid_min_intersection_size_raises(min_size):
    y = np.array([0, 0, 1, 1, 2, 2])

    with pytest.raises(ValueError, match="min_intersection_size"):
        compare_partitions(y, y, min_intersection_size=min_size)


def test_invalid_label_inputs_raise():
    valid = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="comprimentos incompatíveis"):
        compare_partitions(valid, valid[:-1])

    with pytest.raises(ValueError, match="ao menos 3"):
        compare_partitions(np.array([0, 1]), np.array([0, 1]))

    with pytest.raises(ValueError, match="inteiros"):
        compare_partitions(valid.astype(float), valid)

    with pytest.raises(ValueError, match="inválidos"):
        compare_partitions(np.array([0, 0, 1, -2]), valid)


def test_classify_comparison_distinguishes_concepts():
    km8 = {
        "config_id": "kmeans_k8",
        "representation": "bge-small-en-v1.5",
        "algorithm": "kmeans",
    }
    km5 = {
        "config_id": "kmeans_k5",
        "representation": "bge-small-en-v1.5",
        "algorithm": "kmeans",
    }
    ward = {
        "config_id": "agglomerative_ward_k8",
        "representation": "bge-small-en-v1.5",
        "algorithm": "agglomerative",
    }
    tfidf = {
        "config_id": "tfidf_10k_k8",
        "representation": "tfidf",
        "algorithm": "kmeans",
    }

    assert classify_comparison(km8, km8) == "self"
    assert classify_comparison(km8, km5) == "hyperparameter_sensitivity"
    assert classify_comparison(km8, ward) == "cross_algorithm"
    assert classify_comparison(km8, tfidf) == "cross_representation"


def test_find_identical_partitions_keeps_all_configs(synthetic_frames):
    bge, tfidf = synthetic_frames

    table = find_identical_partitions(bge, tfidf)

    assert list(table.columns) == IDENTICAL_PARTITIONS_COLUMNS
    assert len(table) == 4

    indexed = table.set_index("config_id")
    assert bool(indexed.loc["kmeans_k3", "is_duplicate_partition"]) is True
    assert bool(indexed.loc["agglomerative_ward_k3", "is_duplicate_partition"]) is True
    assert (
        indexed.loc["kmeans_k3", "identical_configs"]
        == "kmeans_k3;agglomerative_ward_k3"
    )
    assert bool(indexed.loc["tfidf_10k_k3", "is_duplicate_partition"]) is False

    pairs = build_identical_partition_pairs(bge, tfidf)
    assert list(pairs.columns) == IDENTICAL_PAIR_COLUMNS
    assert len(pairs) == 1
    assert pairs.loc[0, "config_a"] == "kmeans_k3"
    assert pairs.loc[0, "config_b"] == "agglomerative_ward_k3"
    assert bool(pairs.loc[0, "exact_label_match"]) is False
    assert pairs.loc[0, "ari"] == pytest.approx(1.0)


def test_concordance_table_and_full_matrix(synthetic_frames):
    bge, tfidf = synthetic_frames

    concordance = build_concordance_table(bge, tfidf)
    matrix = build_full_ari_matrix(bge, tfidf)

    assert list(concordance.columns) == CONCORDANCE_COLUMNS
    assert len(concordance) == 6
    assert set(concordance["comparison_type"]).isdisjoint(
        {"seed_initialization", "resampling"}
    )

    assert list(matrix.index) == [
        "kmeans_k3",
        "agglomerative_ward_k3",
        "tfidf_10k_k3",
    ]
    assert matrix.equals(matrix.T)
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert matrix.loc["kmeans_k3", "agglomerative_ward_k3"] == pytest.approx(1.0)


def test_misaligned_bge_and_tfidf_raise(synthetic_frames):
    bge, tfidf = synthetic_frames

    swapped_ids = tfidf.copy()
    swapped_ids.loc[[0, 1], "openalex_id"] = swapped_ids.loc[
        [1, 0], "openalex_id"
    ].to_numpy()

    with pytest.raises(ValueError, match="divergentes"):
        build_concordance_table(bge, swapped_ids)

    shuffled_index = tfidf.copy()
    shuffled_index["embedding_index"] = shuffled_index["embedding_index"].to_numpy()[
        ::-1
    ]

    with pytest.raises(ValueError, match="fora da ordem"):
        build_full_ari_matrix(bge, shuffled_index)

    with pytest.raises(ValueError, match="ausente"):
        build_concordance_table(bge, tfidf.drop(columns=["openalex_id"]))
