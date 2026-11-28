"""Integração do pipeline de avaliação com os artefatos versionados."""

import time

import pandas as pd
import pytest

from research_atlas.evaluation.pipeline import (
    PROCESSED_DIR,
    TABLE_COLUMNS,
    build_evaluation_table,
    reconcile_with_reference,
    sha256_file,
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

METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz"]


@pytest.fixture(scope="module")
def run():
    before = {name: sha256_file(PROCESSED_DIR / name) for name in INPUT_FILES}

    started = time.perf_counter()
    table = build_evaluation_table()
    elapsed = time.perf_counter() - started

    after = {name: sha256_file(PROCESSED_DIR / name) for name in INPUT_FILES}

    return {
        "table": table,
        "indexed": table.set_index("config_id"),
        "reconciliation": reconcile_with_reference(table),
        "before": before,
        "after": after,
        "elapsed": elapsed,
    }


def test_input_artifacts_are_not_modified(run):
    assert run["before"] == run["after"]


def test_all_configurations_are_evaluated(run):
    table = run["table"]
    bge = pd.read_parquet(
        PROCESSED_DIR / "clustering/v1/cluster_assignments.parquet"
    )
    tfidf = pd.read_parquet(
        PROCESSED_DIR / "tfidf_baseline/v1/tfidf_assignments.parquet"
    )

    expected = [
        c for c in bge.columns if c.startswith(("kmeans", "agglom", "hdbscan"))
    ] + [c for c in tfidf.columns if c.startswith("tfidf")]

    assert len(table) == 24
    assert list(table.columns) == TABLE_COLUMNS
    assert table["config_id"].tolist() == expected
    assert table["config_id"].is_unique
    assert (table["n_articles"] == 5000).all()
    assert table["representation"].value_counts().to_dict() == {
        "bge-small-en-v1.5": 18,
        "tfidf": 6,
    }


def test_scope_distinguishes_full_from_grouped_only(run):
    indexed = run["indexed"]
    hdbscan = indexed.index.str.startswith("hdbscan")

    assert (indexed.loc[hdbscan, "scope"] == "grouped_only").all()
    assert (indexed.loc[~hdbscan, "scope"] == "full").all()
    assert (indexed.loc[hdbscan, "noise_articles"] > 0).all()
    assert (indexed.loc[~hdbscan, "noise_articles"] == 0).all()
    assert (
        indexed["n_evaluated"] == indexed["grouped_articles"]
    ).all()


def test_grouped_only_rows_have_unique_comparability_groups(run):
    indexed = run["indexed"]
    grouped = indexed[indexed["scope"] == "grouped_only"]

    assert grouped["comparability_group"].is_unique

    full = indexed[indexed["scope"] == "full"]
    shared = full.groupby("comparability_group").size().to_dict()

    assert shared == {
        "bge-small-en-v1.5:384d_l2|full|sampled:2000:seed42": 7,
        "tfidf:10000f_sparse|full|sampled:2000:seed42": 3,
        "tfidf:20000f_sparse|full|sampled:2000:seed42": 3,
    }


def test_bge_metrics_are_computed_for_all_valid_partitions(run):
    bge = run["table"][run["table"]["representation"] == "bge-small-en-v1.5"]

    assert bge["valid"].all()
    assert bge[METRICS].notna().all().all()
    assert (bge["silhouette_metric"] == "cosine").all()
    assert (bge["davies_bouldin_geometry"] == "euclidean").all()
    assert (bge["calinski_harabasz_geometry"] == "euclidean").all()
    assert bge["input_l2_normalized"].all()
    assert not bge["input_sparse"].any()


def test_tfidf_has_only_silhouette_with_explicit_reason(run):
    tfidf = run["table"][run["table"]["representation"] == "tfidf"]

    assert tfidf["silhouette"].notna().all()
    assert tfidf["davies_bouldin"].isna().all()
    assert tfidf["calinski_harabasz"].isna().all()
    assert tfidf["input_sparse"].all()
    assert (tfidf["silhouette_metric"] == "cosine").all()
    assert tfidf["davies_bouldin_geometry"].isna().all()
    assert (
        tfidf["davies_bouldin_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    ).all()
    assert (
        tfidf["calinski_harabasz_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    ).all()
    assert tfidf["silhouette_unavailable_reason"].isna().all()
    assert set(tfidf["n_features"]) == {10_000, 20_000}


def test_sampling_metadata_matches_original_protocol(run):
    indexed = run["indexed"]

    assert (indexed.loc["kmeans_k8", "silhouette_sampling"]) == "sampled"
    assert indexed.loc["kmeans_k8", "silhouette_sample_size"] == 2000

    leaf = indexed.loc["hdbscan_leaf_mcs20_ms5"]

    assert leaf["silhouette_sampling"] == "full"
    assert leaf["silhouette_sample_size"] == 840
    assert not leaf["silhouette_sampled"]

    eom = indexed.loc["hdbscan_eom_mcs20_ms5"]

    assert eom["silhouette_sampling"] == "sampled"
    assert eom["n_evaluated"] == 2840
    assert eom["silhouette_sample_size"] == 2000
    assert (indexed["seed"] == 42).all()


def test_warnings_flag_degenerate_distributions(run):
    indexed = run["indexed"]

    average = indexed.loc["agglomerative_average_k8", "warnings"].split(";")

    assert {"cluster_unitario", "cluster_dominante"} <= set(average)
    assert "cobertura_baixa" in indexed.loc[
        "hdbscan_leaf_mcs20_ms5", "warnings"
    ]
    assert indexed.loc["kmeans_k8", "warnings"] == ""


def test_results_match_previous_csvs(run):
    reconciliation = run["reconciliation"]

    assert len(reconciliation) == 119
    assert reconciliation["match"].all()
    assert reconciliation["source"].nunique() == 6
    assert reconciliation["abs_diff"].max() < 1e-4


def test_reconciliation_detects_divergence(run):
    table = run["table"].copy()
    table.loc[table["config_id"] == "kmeans_k8", "silhouette"] += 0.01

    reconciliation = reconcile_with_reference(table)
    failed = reconciliation[~reconciliation["match"]]

    assert failed[["config_id", "field"]].to_dict("records") == [
        {"config_id": "kmeans_k8", "field": "silhouette"}
    ]


def test_evaluation_is_reproducible(run):
    assert build_evaluation_table().equals(run["table"])


def test_pipeline_runs_in_reasonable_time(run):
    assert run["elapsed"] < 120
