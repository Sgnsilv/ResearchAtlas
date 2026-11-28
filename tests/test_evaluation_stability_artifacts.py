"""Integração da análise de estabilidade e concordância com os artefatos reais."""

import time

import numpy as np
import pandas as pd
import pytest

from research_atlas.evaluation.pipeline import PROCESSED_DIR, sha256_file
from research_atlas.evaluation.stability import (
    CONCORDANCE_COLUMNS,
    IDENTICAL_PARTITIONS_COLUMNS,
    SEED_STABILITY_COLUMNS,
    load_concordance_analysis,
)
INPUT_FILES = (
    "dataset_selected.parquet",
    "embeddings/bge-small-en-v1.5-v1/embeddings.npy",
    "embeddings/bge-small-en-v1.5-v1/metadata.parquet",
    "clustering/v1/cluster_assignments.parquet",
    "clustering/v1/kmeans_metrics.csv",
    "clustering/v1/kmeans_stability.csv",
    "clustering/v1/agglomerative_metrics.csv",
    "clustering/v1/hdbscan_eom_metrics.csv",
    "clustering/v1/hdbscan_leaf_metrics.csv",
    "clustering/v1/manifest.json",
    "tfidf_baseline/v1/tfidf_assignments.parquet",
    "tfidf_baseline/v1/kmeans_k_sensitivity.csv",
    "tfidf_baseline/v1/tfidf_final_candidates.csv",
    "tfidf_baseline/v1/tfidf_parameter_sensitivity.csv",
    "tfidf_baseline/v1/tfidf_stability.csv",
    "tfidf_baseline/v1/manifest.json",
)


def _snapshot_evaluation_dir() -> dict[str, str] | None:
    eval_dir = PROCESSED_DIR / "evaluation"
    if not eval_dir.exists():
        return None
    return {
        str(p.relative_to(eval_dir)): sha256_file(p)
        for p in sorted(eval_dir.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def run():
    before = {name: sha256_file(PROCESSED_DIR / name) for name in INPUT_FILES}
    eval_before = _snapshot_evaluation_dir()

    started = time.perf_counter()
    analysis = load_concordance_analysis()
    elapsed = time.perf_counter() - started

    after = {name: sha256_file(PROCESSED_DIR / name) for name in INPUT_FILES}
    eval_after = _snapshot_evaluation_dir()

    return {
        **analysis,
        "before": before,
        "after": after,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "elapsed": elapsed,
    }


def test_artifacts_are_untouched_by_concordance_analysis(run):
    assert run["before"] == run["after"]
    assert run["eval_before"] == run["eval_after"]


def test_seed_stability_consolidates_bge_and_tfidf(run):
    table = run["seed_stability"]

    assert list(table.columns) == SEED_STABILITY_COLUMNS
    assert table["config_id"].tolist() == [
        "kmeans_k5",
        "kmeans_k7",
        "kmeans_k8",
        "kmeans_k9",
        "tfidf_10k_k5",
        "tfidf_10k_k8",
        "tfidf_10k_k13",
    ]
    assert (table["stability_type"] == "seed_initialization").all()
    assert (table["n_seeds"] == 5).all()
    assert (table["n_seed_pairs"] == 10).all()
    assert (table["seeds"] == "11;21;42;73;101").all()
    assert (
        table[["ari_mean", "ari_std", "ari_min", "ari_max"]].notna().all().all()
    )
    assert (table["ari_min"] <= table["ari_mean"]).all()
    assert (table["ari_mean"] <= table["ari_max"]).all()


def test_full_ari_matrix_is_symmetric_and_complete(run):
    matrix = run["full_ari_matrix"]

    expected = [
        "kmeans_k5",
        "kmeans_k7",
        "kmeans_k8",
        "kmeans_k9",
        "agglomerative_average_k8",
        "agglomerative_complete_k8",
        "agglomerative_ward_k8",
        "tfidf_10k_k5",
        "tfidf_10k_k8",
        "tfidf_10k_k13",
        "tfidf_20k_k5",
        "tfidf_20k_k8",
        "tfidf_20k_k13",
    ]

    assert list(matrix.index) == expected
    assert list(matrix.columns) == expected
    assert matrix.notna().all().all()
    assert matrix.equals(matrix.T)
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)

    assert matrix.loc["kmeans_k8", "agglomerative_ward_k8"] == pytest.approx(
        0.7139315, abs=1e-6
    )
    assert matrix.loc["kmeans_k8", "agglomerative_complete_k8"] == pytest.approx(
        0.6002380, abs=1e-6
    )
    assert matrix.loc["kmeans_k8", "agglomerative_average_k8"] == pytest.approx(
        0.0004051, abs=1e-6
    )
    assert matrix.loc["kmeans_k8", "tfidf_10k_k8"] == pytest.approx(
        0.4420, abs=1e-4
    )


def test_identical_partitions_identified_without_dropping_configs(run):
    identical = run["identical_partitions"]

    assert list(identical.columns) == IDENTICAL_PARTITIONS_COLUMNS
    assert len(identical) == 24

    duplicates = identical[identical["is_duplicate_partition"]]
    assert set(duplicates["identical_configs"]) == {
        "hdbscan_eom_mcs20_ms5;hdbscan_eom_mcs30_ms5",
        "hdbscan_eom_mcs50_ms10;hdbscan_eom_mcs100_ms10",
    }


def test_concordance_table_covers_all_pairs_and_flags_intersections(run):
    concordance = run["concordance"]

    assert list(concordance.columns) == CONCORDANCE_COLUMNS
    assert len(concordance) == 276
    assert int(concordance["valid"].sum()) == 249
    assert int((~concordance["valid"]).sum()) == 27

    invalid = concordance[~concordance["valid"]]
    assert (
        invalid["ari_unavailable_reason"] == "menos_de_2_clusters_na_intersecao"
    ).all()
    assert invalid["ari"].isna().all()

    assert set(concordance["comparison_type"]) == {
        "cross_algorithm",
        "cross_representation",
        "hyperparameter_sensitivity",
    }

    partial = concordance[
        concordance["comparison_scope"] == "grouped_intersection"
    ]
    assert (partial["n_intersection"] < 5000).all()
    assert partial["warnings"].str.contains("intersecao_parcial").all()

    pair = concordance[
        (concordance["config_a"] == "hdbscan_eom_mcs30_ms10")
        & (concordance["config_b"] == "hdbscan_eom_mcs50_ms10")
    ].iloc[0]

    assert pair["ari"] == pytest.approx(1.0)
    assert bool(pair["identical_partition"]) is False
    assert pair["n_intersection"] == 1298
    assert "cobertura_intersecao_baixa" in pair["warnings"]
    assert "clusters_ausentes_na_intersecao" in pair["warnings"]


def test_reference_reconciliation_matches_previous_csvs(run):
    reconciliation = run["reconciliation"]

    assert len(reconciliation) == 9
    assert reconciliation["match"].all()
    assert reconciliation["abs_diff"].max() < 1e-12


def test_concordance_analysis_is_reproducible_and_fast(run):
    second = load_concordance_analysis()

    assert second["seed_stability"].equals(run["seed_stability"])
    assert second["concordance"].equals(run["concordance"])
    assert second["full_ari_matrix"].equals(run["full_ari_matrix"])
    assert pd.notna(run["elapsed"]) and run["elapsed"] < 10
