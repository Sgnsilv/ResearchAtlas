"""Testes de integração dos artefatos de clusterização BGE."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT / "data/processed/clustering/v1"

EMBEDDING_DIR = (
    ROOT / "data/processed/embeddings/bge-small-en-v1.5-v1"
)

MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

EXPECTED_EMBEDDING_SHA256 = (
    "13fe24f37622790f6d6630fbdb5e91742060f95a8d7310597d3bd98f2a7f465c"
)

EXPECTED_FILES = {
    "cluster_assignments.parquet",
    "kmeans_metrics.csv",
    "kmeans_stability.csv",
    "agglomerative_metrics.csv",
    "hdbscan_eom_metrics.csv",
    "hdbscan_leaf_metrics.csv",
}


def sha256_file(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.is_file()

    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def assignments():
    return pd.read_parquet(
        ARTIFACT_DIR / "cluster_assignments.parquet"
    )


def test_manifest_parameters(manifest):
    """Verifica a proveniência dos experimentos."""
    assert manifest["version"] == "1"
    assert manifest["dataset_articles"] == 5000
    assert manifest["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert manifest["embedding_sha256"] == EXPECTED_EMBEDDING_SHA256
    assert manifest["random_seed"] == 42

    assert manifest["kmeans_k_values"] == list(range(2, 16))
    assert manifest["kmeans_stability_seeds"] == [
        11, 21, 42, 73, 101
    ]

    assert set(manifest["agglomerative_linkages"]) == {
        "average",
        "complete",
        "ward",
    }


def test_artifact_checksums(manifest):
    """Verifica a integridade dos seis arquivos de dados."""
    checksums = manifest["files_sha256"]

    assert set(checksums) == EXPECTED_FILES

    for filename, expected_hash in checksums.items():
        path = ARTIFACT_DIR / filename

        assert path.is_file(), f"Arquivo ausente: {filename}"
        assert sha256_file(path) == expected_hash, (
            f"Checksum inválido: {filename}"
        )


def test_embedding_reference():
    """Confirma a versão dos embeddings utilizados."""
    path = EMBEDDING_DIR / "embeddings.npy"

    assert path.is_file()
    assert sha256_file(path) == EXPECTED_EMBEDDING_SHA256


def test_assignments_integrity(assignments):
    """Valida quantidade, índices e unicidade dos registros."""
    assert len(assignments) == 5000
    assert assignments["openalex_id"].is_unique

    np.testing.assert_array_equal(
        assignments["embedding_index"].to_numpy(),
        np.arange(5000),
    )

    assert not assignments.isna().any().any()

    assert assignments.shape[1] == 21


def test_assignments_match_embedding_metadata(assignments):
    """Confirma a ordem dos artigos associada aos embeddings."""
    metadata = pd.read_parquet(
        EMBEDDING_DIR / "metadata.parquet"
    )

    assert (
        assignments["openalex_id"].tolist()
        == metadata["openalex_id"].tolist()
    )

    assert (
        assignments["sampling_area"].tolist()
        == metadata["sampling_area"].tolist()
    )


@pytest.mark.parametrize(
    ("column", "expected_clusters"),
    [
        ("kmeans_k5", 5),
        ("kmeans_k7", 7),
        ("kmeans_k8", 8),
        ("kmeans_k9", 9),
        ("agglomerative_average_k8", 8),
        ("agglomerative_complete_k8", 8),
        ("agglomerative_ward_k8", 8),
    ],
)
def test_partition_labels(assignments, column, expected_clusters):
    """Verifica rótulos de partições completas."""
    assert column in assignments.columns

    labels = assignments[column].to_numpy()

    assert np.issubdtype(labels.dtype, np.integer)
    assert len(np.unique(labels)) == expected_clusters
    assert labels.min() == 0
    assert labels.max() == expected_clusters - 1


@pytest.mark.parametrize(
    "prefix,configs",
    [
        (
            "hdbscan_eom",
            [
                (20, 5),
                (20, 10),
                (30, 5),
                (30, 10),
                (50, 5),
                (50, 10),
                (50, 20),
                (100, 5),
                (100, 10),
            ],
        ),
        (
            "hdbscan_leaf",
            [
                (20, 5),
                (50, 5),
            ],
        ),
    ],
)
def test_hdbscan_labels(assignments, prefix, configs):
    """Verifica as configurações e o tratamento de ruído."""
    for mcs, ms in configs:
        column = f"{prefix}_mcs{mcs}_ms{ms}"

        assert column in assignments.columns

        labels = assignments[column].to_numpy()

        assert np.issubdtype(labels.dtype, np.integer)
        assert labels.min() >= -1
        assert (labels == -1).any()
        assert (labels != -1).any()


def test_saved_metrics_match_assignments(assignments):
    """Confere os números de clusters e ruído exportados."""
    eom = pd.read_csv(
        ARTIFACT_DIR / "hdbscan_eom_metrics.csv"
    )

    leaf = pd.read_csv(
        ARTIFACT_DIR / "hdbscan_leaf_metrics.csv"
    )

    for method, table in [("eom", eom), ("leaf", leaf)]:
        for _, row in table.iterrows():
            mcs = int(row["min_cluster_size"])
            ms = int(row["min_samples"])

            column = f"hdbscan_{method}_mcs{mcs}_ms{ms}"
            labels = assignments[column].to_numpy()

            noise = int(np.sum(labels == -1))
            grouped = int(np.sum(labels != -1))
            clusters = len(set(labels) - {-1})

            assert noise + grouped == 5000
            assert clusters == int(row["n_clusters"])
            assert grouped == int(row["grouped_articles"])

            if "noise_articles" in table.columns:
                assert noise == int(row["noise_articles"])
            else:
                assert grouped / 5000 * 100 == pytest.approx(
                    row["coverage_pct"],
                    abs=1e-6,
                )
