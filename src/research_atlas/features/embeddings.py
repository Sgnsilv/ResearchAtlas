"""Preparação e agregação de embeddings semânticos do ResearchAtlas."""

import numpy as np


def split_token_ids(
    token_ids: list[int],
    cls_token_id: int,
    sep_token_id: int,
    max_length: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Segmenta IDs de tokens sem decodificar e retokenizar."""
    content_size = max_length - 2

    if not 0 <= overlap < content_size:
        raise ValueError("Sobreposição inválida")

    if not token_ids:
        raise ValueError("Texto sem tokens de conteúdo")

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + content_size, len(token_ids))

        input_ids = (
            [cls_token_id]
            + token_ids[start:end]
            + [sep_token_id]
        )

        new_tokens = end - start

        if start > 0:
            new_tokens -= overlap

        chunks.append(
            {
                "input_ids": input_ids,
                "token_weight": max(1, new_tokens),
            }
        )

        if end == len(token_ids):
            break

        start = end - overlap

    return chunks


def aggregate_chunk_embeddings(
    chunk_embeddings: np.ndarray,
    article_indices: list[int],
    token_weights: list[int],
    article_count: int,
) -> np.ndarray:
    """Agrega trechos por média ponderada e normalização L2."""
    if chunk_embeddings.ndim != 2:
        raise ValueError("Os embeddings devem ter duas dimensões")

    chunk_count, dimension = chunk_embeddings.shape

    if not (
        chunk_count
        == len(article_indices)
        == len(token_weights)
    ):
        raise ValueError("Quantidades de trechos incompatíveis")

    if article_count <= 0:
        raise ValueError("Quantidade de artigos inválida")

    if not np.isfinite(chunk_embeddings).all():
        raise ValueError("Embeddings possuem valores não finitos")

    if any(
        index < 0 or index >= article_count
        for index in article_indices
    ):
        raise ValueError("Índice de artigo inválido")

    weights = np.asarray(token_weights, dtype=np.float64)

    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Pesos devem ser positivos e finitos")

    normalized_chunks = chunk_embeddings.astype(
        np.float64
    )

    chunk_norms = np.linalg.norm(
        normalized_chunks,
        axis=1,
        keepdims=True,
    )

    if np.any(chunk_norms == 0):
        raise ValueError("Embedding de trecho nulo")

    normalized_chunks /= chunk_norms

    result = np.zeros(
        (article_count, dimension),
        dtype=np.float64,
    )

    article_weights = np.zeros(
        article_count,
        dtype=np.float64,
    )

    np.add.at(
        result,
        article_indices,
        normalized_chunks * weights[:, None],
    )

    np.add.at(
        article_weights,
        article_indices,
        weights,
    )

    if np.any(article_weights == 0):
        raise ValueError("Existem artigos sem trechos")

    result /= article_weights[:, None]

    norms = np.linalg.norm(
        result,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Agregação produziu vetor nulo")

    result /= norms

    return result.astype(np.float32)
