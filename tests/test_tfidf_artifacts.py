
"""Testes de integração dos artefatos do baseline TF-IDF."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT / "data/processed/dataset_selected.parquet"
)

ARTIFACT_DIR = (
    ROOT / "data/processed/tfidf_baseline/v1"
)

MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

EXPECTED_DATASET_SHA256 = (
    "533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389"
)

EXPECTED_FILES = {
    "tfidf_assignments.parquet",
    "kmeans_k_sensitivity.csv",
    "tfidf_parameter_sensitivity.csv",
    "tfidf_stability.csv",
    "tfidf_final_candidates.csv",
}


def sha256_file(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
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
        ARTIFACT_DIR / "tfidf_assignments.parquet"
    )


def test_manifest_parameters(manifest):
    """Verifica a configuração principal do experimento."""
    assert manifest["version"] == "1"
    assert manifest["dataset_articles"] == 5000
    assert manifest["dataset_sha256"] == EXPECTED_DATASET_SHA256
    assert manifest["representation"] == "TF-IDF"

    config = manifest["primary_configuration"]

    assert config["max_features"] == 10000
    assert config["ngram_range"] == [1, 2]
    assert config["min_df"] == 3
    assert config["k"] == 8
    assert config["random_state"] == 42
    assert config["n_init"] == 10


def test_artifact_checksums(manifest):
    """Confere a integridade dos arquivos exportados."""
    checksums = manifest["files_sha256"]

    assert set(checksums) == EXPECTED_FILES

    for filename, expected_hash in checksums.items():
        path = ARTIFACT_DIR / filename

        assert path.is_file(), f"Arquivo ausente: {filename}"
        assert sha256_file(path) == expected_hash, (
            f"Checksum inválido: {filename}"
        )


def test_dataset_reference():
    """Garante que o dataset original não foi alterado."""
    assert DATASET_PATH.is_file()
    assert sha256_file(DATASET_PATH) == EXPECTED_DATASET_SHA256


def test_assignments_integrity(assignments):
    """Verifica quantidade, índices e unicidade dos artigos."""
    assert len(assignments) == 5000
    assert assignments["openalex_id"].is_unique

    np.testing.assert_array_equal(
        assignments["embedding_index"].to_numpy(),
        np.arange(5000),
    )

    assert not assignments.isna().any().any()


def test_assignments_match_dataset(assignments):
    """Confirma que os rótulos seguem a ordem do dataset."""
    dataset = pd.read_parquet(DATASET_PATH)

    assert (
        assignments["openalex_id"].tolist()
        == dataset["openalex_id"].tolist()
    )

    assert (
        assignments["sampling_area"].tolist()
        == dataset["sampling_area"].tolist()
    )


@pytest.mark.parametrize(
    ("column", "expected_clusters"),
    [
        ("tfidf_10k_k5", 5),
        ("tfidf_10k_k8", 8),
        ("tfidf_10k_k13", 13),
        ("tfidf_20k_k5", 5),
        ("tfidf_20k_k8", 8),
        ("tfidf_20k_k13", 13),
    ],
)
def test_cluster_labels(assignments, column, expected_clusters):
    """Valida as seis configurações exportadas."""
    assert column in assignments.columns

    labels = assignments[column].to_numpy()

    assert np.issubdtype(labels.dtype, np.integer)
    assert len(np.unique(labels)) == expected_clusters
    assert labels.min() == 0
    assert labels.max() == expected_clusters - 1
