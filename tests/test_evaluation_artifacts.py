"""Validação de schema, hashes e integridade dos artefatos em evaluation/v1."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_atlas.evaluation.export import (
    CORE_INPUT_FILES,
    DEFAULT_OUTPUT_DIR,
    EXPORTED_CSV_FILES,
    REFERENCE_INPUT_FILES,
    _compute_run_id,
)
from research_atlas.evaluation.pipeline import (
    PROCESSED_DIR,
    TABLE_COLUMNS,
    build_evaluation_table,
    sha256_file,
)
from research_atlas.evaluation.stability import (
    CONCORDANCE_COLUMNS,
    IDENTICAL_PAIR_COLUMNS,
    SEED_STABILITY_COLUMNS,
)

ARTIFACT_DIR = DEFAULT_OUTPUT_DIR
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.is_file()
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evaluation_metrics() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "evaluation_metrics.csv")


@pytest.fixture(scope="module")
def seed_stability() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "seed_stability.csv")


@pytest.fixture(scope="module")
def partition_concordance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "partition_concordance.csv")


@pytest.fixture(scope="module")
def full_ari_matrix() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "full_ari_matrix.csv", index_col=0)


@pytest.fixture(scope="module")
def identical_partitions() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "identical_partitions.csv")


def test_directory_contains_exact_expected_artifacts():
    present = {p.name for p in ARTIFACT_DIR.iterdir()}
    assert present == {*EXPORTED_CSV_FILES, "manifest.json"}


def test_manifest_schema_provenance_and_deterministic_run_id(manifest):
    assert manifest["version"] == "1"
    assert manifest["issue"] == "#08"
    assert "created_at_utc" not in manifest
    assert manifest["run_id_method"] == "sha256_canonical_payload"
    assert len(manifest["run_id"]) == 64

    recomputed = _compute_run_id(
        {
            "version": manifest["version"],
            "issue": manifest["issue"],
            "parameters": manifest["parameters"],
            "inputs_sha256": manifest["inputs_sha256"],
            "files_sha256": manifest["files_sha256"],
        }
    )
    assert manifest["run_id"] == recomputed

    assert manifest["dataset_articles"] == 5000
    assert manifest["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert manifest["representations"] == ["bge-small-en-v1.5", "tfidf"]

    params = manifest["parameters"]
    assert params["random_seed"] == 42
    assert params["silhouette_metric"] == "cosine"
    assert params["silhouette_sample_size"] == 2000
    assert params["silhouette_full_fallback_max"] == 10000
    assert params["low_coverage_pct_operational"] == 50.0
    assert params["tfidf_sparse_only_cosine_silhouette"] is True
    assert params["resampling_stability_executed"] is False
    assert params["hdbscan_ari_protocol"] == "grouped_intersection_only"

    summary = manifest["summary"]
    assert summary == {
        "evaluation_configurations": 24,
        "bge_configurations": 18,
        "tfidf_configurations": 6,
        "full_scope_configurations": 13,
        "grouped_only_configurations": 11,
        "seed_stability_configurations": 7,
        "partition_pairs_total": 276,
        "partition_pairs_valid": 249,
        "partition_pairs_invalid": 27,
        "full_ari_matrix_size": 13,
        "identical_partition_pairs": 2,
        "reference_metrics_reconciled": 119,
        "reference_stability_reconciled": 9,
    }

    limitations_text = " ".join(manifest["limitations"]).lower()
    assert "reamostragem" in limitations_text
    assert "matriz_esparsa_nao_densificada" in limitations_text
    assert "grouped_only" in limitations_text
    assert "interseção" in limitations_text


def test_manifest_input_and_output_checksums(manifest):
    expected_inputs = set(CORE_INPUT_FILES + REFERENCE_INPUT_FILES)
    assert set(manifest["inputs_sha256"]) == expected_inputs

    for rel_path, expected_hash in manifest["inputs_sha256"].items():
        path = PROCESSED_DIR / rel_path
        assert path.is_file(), f"Insumo ausente: {rel_path}"
        assert sha256_file(path) == expected_hash

    assert set(manifest["files_sha256"]) == set(EXPORTED_CSV_FILES)
    for filename, expected_hash in manifest["files_sha256"].items():
        path = ARTIFACT_DIR / filename
        assert path.is_file(), f"Artefato ausente: {filename}"
        assert sha256_file(path) == expected_hash


def test_evaluation_metrics_schema_missing_reasons_and_unrounded_precision(
    evaluation_metrics,
):
    assert list(evaluation_metrics.columns) == TABLE_COLUMNS
    assert len(evaluation_metrics) == 24
    assert evaluation_metrics["config_id"].is_unique

    tfidf_mask = evaluation_metrics["representation"] == "tfidf"
    bge_mask = evaluation_metrics["representation"] == "bge-small-en-v1.5"
    hdbscan_mask = evaluation_metrics["algorithm"] == "hdbscan"

    assert int(bge_mask.sum()) == 18
    assert int(tfidf_mask.sum()) == 6
    assert (evaluation_metrics.loc[hdbscan_mask, "scope"] == "grouped_only").all()
    assert (evaluation_metrics.loc[~hdbscan_mask, "scope"] == "full").all()

    assert evaluation_metrics["silhouette"].notna().all()
    assert (evaluation_metrics["silhouette_metric"] == "cosine").all()
    assert set(evaluation_metrics["silhouette_sampling"]) == {"sampled", "full"}

    assert evaluation_metrics.loc[bge_mask, "davies_bouldin"].notna().all()
    assert evaluation_metrics.loc[bge_mask, "calinski_harabasz"].notna().all()
    assert evaluation_metrics.loc[tfidf_mask, "davies_bouldin"].isna().all()
    assert evaluation_metrics.loc[tfidf_mask, "calinski_harabasz"].isna().all()
    assert (
        evaluation_metrics.loc[tfidf_mask, "davies_bouldin_unavailable_reason"]
        == "matriz_esparsa_nao_densificada"
    ).all()
    assert (
        evaluation_metrics.loc[
            tfidf_mask, "calinski_harabasz_unavailable_reason"
        ]
        == "matriz_esparsa_nao_densificada"
    ).all()

    fresh = build_evaluation_table()
    assert evaluation_metrics["config_id"].tolist() == fresh["config_id"].tolist()
    for col in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        mask = fresh[col].notna()
        np.testing.assert_allclose(
            evaluation_metrics.loc[mask, col].to_numpy(dtype=float),
            fresh.loc[mask, col].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        )


def test_seed_stability_preserves_reference_csvs(seed_stability):
    assert list(seed_stability.columns) == SEED_STABILITY_COLUMNS
    assert len(seed_stability) == 7
    assert (seed_stability["stability_type"] == "seed_initialization").all()

    bge_ref = pd.read_csv(PROCESSED_DIR / "clustering/v1/kmeans_stability.csv")
    tfidf_ref = pd.read_csv(
        PROCESSED_DIR / "tfidf_baseline/v1/tfidf_stability.csv"
    )
    indexed = seed_stability.set_index("config_id")

    for _, row in bge_ref.iterrows():
        cid = f"kmeans_k{int(row['k'])}"
        for field in ("ari_mean", "ari_std", "ari_min", "ari_max"):
            assert indexed.loc[cid, field] == pytest.approx(
                row[field], rel=0.0, abs=1e-15
            )

    for _, row in tfidf_ref.iterrows():
        cid = f"tfidf_10k_k{int(row['k'])}"
        for field in ("ari_mean", "ari_std", "ari_min", "ari_max"):
            assert indexed.loc[cid, field] == pytest.approx(
                row[field], rel=0.0, abs=1e-15
            )


def test_partition_concordance_validity_and_intersection_semantics(
    partition_concordance,
):
    assert list(partition_concordance.columns) == CONCORDANCE_COLUMNS
    assert len(partition_concordance) == 276

    assert set(partition_concordance["comparison_type"]) == {
        "cross_algorithm",
        "cross_representation",
        "hyperparameter_sensitivity",
    }

    valid_mask = partition_concordance["valid"]
    assert int(valid_mask.sum()) == 249
    assert int((~valid_mask).sum()) == 27

    assert partition_concordance.loc[valid_mask, "ari"].notna().all()
    assert (
        partition_concordance.loc[valid_mask, "ari_unavailable_reason"]
        .isna()
        .all()
    )
    assert partition_concordance.loc[~valid_mask, "ari"].isna().all()
    assert (
        partition_concordance.loc[~valid_mask, "ari_unavailable_reason"]
        .notna()
        .all()
    )

    involves_hdbscan = (partition_concordance["algorithm_a"] == "hdbscan") | (
        partition_concordance["algorithm_b"] == "hdbscan"
    )
    assert (
        partition_concordance.loc[involves_hdbscan, "comparison_scope"]
        == "grouped_intersection"
    ).all()
    assert (
        partition_concordance.loc[~involves_hdbscan, "comparison_scope"]
        == "full"
    ).all()

    partial_perfect = partition_concordance[
        (partition_concordance["config_a"] == "hdbscan_eom_mcs30_ms10")
        & (partition_concordance["config_b"] == "hdbscan_eom_mcs50_ms10")
    ].iloc[0]
    assert partial_perfect["ari"] == pytest.approx(1.0)
    assert partial_perfect["intersection_coverage_pct"] == pytest.approx(25.96)
    assert bool(partial_perfect["identical_partition"]) is False


def test_full_ari_matrix_matches_concordance_table(
    full_ari_matrix,
    partition_concordance,
    evaluation_metrics,
):
    expected_full = evaluation_metrics.loc[
        evaluation_metrics["scope"] == "full",
        "config_id",
    ].tolist()

    assert full_ari_matrix.index.name == "config_id"
    assert list(full_ari_matrix.index) == expected_full
    assert list(full_ari_matrix.columns) == expected_full
    assert full_ari_matrix.shape == (13, 13)
    assert full_ari_matrix.notna().all().all()
    np.testing.assert_allclose(
        full_ari_matrix.to_numpy(),
        full_ari_matrix.T.to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.diag(full_ari_matrix.to_numpy()),
        1.0,
        rtol=0.0,
        atol=1e-15,
    )

    full_pairs = partition_concordance[
        partition_concordance["comparison_scope"] == "full"
    ]
    assert len(full_pairs) == 13 * 12 // 2
    for _, row in full_pairs.iterrows():
        a, b = row["config_a"], row["config_b"]
        assert full_ari_matrix.loc[a, b] == pytest.approx(
            row["ari"], rel=0.0, abs=1e-15
        )
        assert full_ari_matrix.loc[b, a] == pytest.approx(
            row["ari"], rel=0.0, abs=1e-15
        )


def test_identical_partitions_csv_matches_global_duplicates(
    identical_partitions,
    partition_concordance,
):
    assert list(identical_partitions.columns) == IDENTICAL_PAIR_COLUMNS
    assert len(identical_partitions) == 2

    expected_pairs = [
        ("hdbscan_eom_mcs20_ms5", "hdbscan_eom_mcs30_ms5"),
        ("hdbscan_eom_mcs50_ms10", "hdbscan_eom_mcs100_ms10"),
    ]
    actual_pairs = list(
        zip(identical_partitions["config_a"], identical_partitions["config_b"])
    )
    assert actual_pairs == expected_pairs
    assert (identical_partitions["ari"] == 1.0).all()
    assert identical_partitions["partition_signature"].str.len().eq(64).all()

    flagged = partition_concordance[partition_concordance["identical_partition"]]
    flagged_pairs = list(zip(flagged["config_a"], flagged["config_b"]))
    assert flagged_pairs == expected_pairs
