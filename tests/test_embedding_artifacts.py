"""Validação dos artefatos de embeddings do ResearchAtlas."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "data/processed/embeddings/bge-small-en-v1.5-v1"
)

EXPECTED_DATASET_SHA256 = (
    "533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389"
)

EXPECTED_EMBEDDINGS_SHA256 = (
    "13fe24f37622790f6d6630fbdb5e91742060f95a8d7310597d3bd98f2a7f465c"
)


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de um arquivo."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def test_artifact_files_exist():
    assert (ARTIFACT_DIR / "embeddings.npy").is_file()
    assert (ARTIFACT_DIR / "metadata.parquet").is_file()
    assert (ARTIFACT_DIR / "manifest.json").is_file()


def test_embeddings_integrity():
    path = ARTIFACT_DIR / "embeddings.npy"

    assert sha256_file(path) == EXPECTED_EMBEDDINGS_SHA256

    embeddings = np.load(path, allow_pickle=False)

    assert embeddings.shape == (5000, 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_metadata_matches_dataset():
    metadata = pd.read_parquet(ARTIFACT_DIR / "metadata.parquet")
    dataset = pd.read_parquet(
        PROJECT_ROOT / "data/processed/dataset_selected.parquet"
    )

    assert len(metadata) == len(dataset) == 5000
    assert metadata["openalex_id"].is_unique
    assert metadata["embedding_index"].tolist() == list(range(5000))

    assert metadata["openalex_id"].tolist() == (
        dataset["openalex_id"].tolist()
    )


def test_manifest_integrity():
    manifest = json.loads(
        (ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["model"] == "BAAI/bge-small-en-v1.5"
    assert manifest["dataset_sha256"] == EXPECTED_DATASET_SHA256
    assert (
        manifest["embedding_file_sha256"]
        == EXPECTED_EMBEDDINGS_SHA256
    )
    assert manifest["article_count"] == 5000
    assert manifest["embedding_dimension"] == 384
    assert manifest["chunk_count"] == 5678
    assert manifest["chunk_overlap"] == 64
