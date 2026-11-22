from sklearn.metrics import silhouette_score

"""Testes do baseline TF-IDF + K-Means."""

import numpy as np
import pandas as pd
import pytest

from research_atlas.features.tfidf import (
    build_article_texts,
    build_tfidf,
    evaluate_clustering,
    run_kmeans,
)


@pytest.fixture
def sample_articles():
    topics = [
        "galaxy stars telescope astronomy",
        "protein genome genes biology",
        "robot motor control engineering",
    ]

    rows = []

    for topic in topics:
        for i in range(8):
            rows.append({
                "title": f"{topic} study {i}",
                "abstract": f"Research about {topic} using data analysis",
            })

    return pd.DataFrame(rows)


def test_build_texts_preserves_order(sample_articles):
    texts = build_article_texts(sample_articles)

    assert len(texts) == 24
    assert "galaxy" in texts[0]
    assert "protein" in texts[8]


def test_build_texts_rejects_missing_columns():
    with pytest.raises(ValueError, match="título ou resumo"):
        build_article_texts(pd.DataFrame({"title": ["Test"]}))


def test_tfidf_returns_sparse_matrix(sample_articles):
    texts = build_article_texts(sample_articles)

    matrix, vectorizer = build_tfidf(
        texts,
        max_features=100,
        min_df=1,
    )

    assert matrix.shape[0] == 24
    assert matrix.format == "csr"
    assert len(vectorizer.get_feature_names_out()) > 0


def test_kmeans_is_reproducible(sample_articles):
    texts = build_article_texts(sample_articles)
    matrix, _ = build_tfidf(texts, min_df=1)

    labels_a, _ = run_kmeans(matrix, k=3, seed=42)
    labels_b, _ = run_kmeans(matrix, k=3, seed=42)

    np.testing.assert_array_equal(labels_a, labels_b)


def test_invalid_k_raises_error(sample_articles):
    texts = build_article_texts(sample_articles)
    matrix, _ = build_tfidf(texts, min_df=1)

    with pytest.raises(ValueError, match="k deve"):
        run_kmeans(matrix, k=30)


def test_evaluation_returns_metrics(sample_articles):
    texts = build_article_texts(sample_articles)
    matrix, _ = build_tfidf(texts, min_df=1)

    labels, _ = run_kmeans(matrix, k=3)

    metrics = evaluate_clustering(
        matrix,
        labels,
        sample_size=24,
    )

    assert metrics["n_clusters"] == 3
    assert -1 <= metrics["silhouette"] <= 1
    assert metrics["min_cluster_size"] > 0
    assert metrics["largest_cluster_pct"] <= 100


def test_evaluation_rejects_single_cluster(sample_articles):
    texts = build_article_texts(sample_articles)
    matrix, _ = build_tfidf(texts, min_df=1)

    labels = np.zeros(matrix.shape[0], dtype=int)

    with pytest.raises(ValueError, match="Partição inválida"):
        evaluate_clustering(matrix, labels)

def test_evaluation_matches_sklearn_protocol(sample_articles):
    """Confirma equivalência com a avaliação usada no Colab."""
    texts = build_article_texts(sample_articles)

    matrix, _ = build_tfidf(
        texts,
        max_features=100,
        min_df=1,
    )

    labels, _ = run_kmeans(
        matrix,
        k=3,
        seed=42,
    )

    expected = silhouette_score(
        matrix,
        labels,
        metric="cosine",
        sample_size=18,
        random_state=42,
    )

    result = evaluate_clustering(
        matrix,
        labels,
        sample_size=18,
        seed=42,
    )

    assert result["silhouette"] == pytest.approx(
        expected,
        abs=1e-12,
    )
