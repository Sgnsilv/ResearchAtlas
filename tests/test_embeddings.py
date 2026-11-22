
"""Testes das funções de preparação de embeddings."""

import numpy as np
import pytest

from research_atlas.features.embeddings import (
    aggregate_chunk_embeddings,
    split_token_ids,
)


def test_short_text_has_one_chunk():
    chunks = split_token_ids(
        [10, 20, 30],
        cls_token_id=101,
        sep_token_id=102,
    )

    assert len(chunks) == 1
    assert chunks[0]["input_ids"] == [101, 10, 20, 30, 102]
    assert chunks[0]["token_weight"] == 3


def test_long_text_respects_token_limit():
    chunks = split_token_ids(
        list(range(1200)),
        cls_token_id=101,
        sep_token_id=102,
    )

    assert len(chunks) == 3
    assert all(len(chunk["input_ids"]) <= 512 for chunk in chunks)
    assert sum(chunk["token_weight"] for chunk in chunks) == 1200


def test_empty_tokens_raise_error():
    with pytest.raises(ValueError, match="sem tokens"):
        split_token_ids([], 101, 102)


def test_aggregation_preserves_article_order():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = aggregate_chunk_embeddings(
        chunk_embeddings=embeddings,
        article_indices=[0, 1, 0],
        token_weights=[2, 1, 2],
        article_count=2,
    )

    assert result.shape == (2, 2)
    np.testing.assert_allclose(
        result,
        [[1.0, 0.0], [0.0, 1.0]],
        atol=1e-6,
    )


def test_aggregation_normalizes_vectors():
    result = aggregate_chunk_embeddings(
        chunk_embeddings=np.array(
            [[3.0, 0.0], [0.0, 4.0]],
            dtype=np.float32,
        ),
        article_indices=[0, 0],
        token_weights=[1, 1],
        article_count=1,
    )

    np.testing.assert_allclose(
        np.linalg.norm(result, axis=1),
        [1.0],
        atol=1e-6,
    )


def test_missing_article_raises_error():
    with pytest.raises(ValueError, match="sem trechos"):
        aggregate_chunk_embeddings(
            np.array([[1.0, 0.0]]),
            article_indices=[0],
            token_weights=[1],
            article_count=2,
        )


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError, match="Sobreposição"):
        split_token_ids(
            [1, 2, 3],
            cls_token_id=101,
            sep_token_id=102,
            overlap=510,
        )
