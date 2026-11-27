
"""Avaliação interna de partições de clusterização do ResearchAtlas."""

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from research_atlas.clustering.algorithms import (
    summarize_clusters,
    validate_embeddings,
)

SILHOUETTE_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42

# Limite para sinalização de clusters dominantes.
MAX_CLUSTER_SHARE = 0.8


def evaluate_partition(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Avalia uma partição de clusterização.

    Calcula Silhouette, Davies-Bouldin e Calinski-Harabasz
    quando a partição é válida.

    Para resultados do HDBSCAN, exclui artigos classificados
    como ruído (rótulo -1) do cálculo das métricas.

    Partições inválidas retornam métricas None e os motivos
    correspondentes em 'reasons', sem interromper o pipeline.

    Erros de entrada geram ValueError.
    """
    matrix = validate_embeddings(embeddings)
    labels = np.asarray(labels)

    # 1. Validação das entradas.
    if labels.ndim != 1 or len(labels) != len(matrix):
        raise ValueError("Rótulos incompatíveis com os embeddings")

    if not isinstance(sample_size, int) or sample_size < 2:
        raise ValueError("sample_size deve ser um inteiro >= 2")

    if not isinstance(seed, int):
        raise ValueError("seed deve ser um inteiro")

    # 2. Resumo da partição.
    summary = summarize_clusters(labels)

    # 3. Remoção de ruído para avaliação.
    mask = labels != -1
    X = matrix[mask]
    y = labels[mask]

    reasons = []
    warnings = []

    n_samples = summary["grouped_articles"]
    n_clusters = summary["n_clusters"]

    # 4. Identificação de condições inválidas.
    if n_samples == 0:
        reasons.append("sem_artigos_agrupados")

    elif n_clusters < 2:
        reasons.append("menos_de_2_clusters")

    elif n_clusters >= n_samples:
        reasons.append("clusters_insuficientes_para_metricas")

    # 5. Identificação de avisos.
    if summary["min_cluster_size"] == 1:
        warnings.append("cluster_unitario")

    if n_samples > 0:
        largest_share = summary["max_cluster_size"] / n_samples

        if largest_share > MAX_CLUSTER_SHARE:
            warnings.append("cluster_dominante")

    # 6. Cálculo das métricas.
    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None

    if not reasons:
        effective_sample_size = min(sample_size, n_samples)

        silhouette = float(
            silhouette_score(
                X,
                y,
                metric="cosine",
                sample_size=(
                    effective_sample_size
                    if effective_sample_size < n_samples
                    else None
                ),
                random_state=seed,
            )
        )

        davies_bouldin = float(
            davies_bouldin_score(X, y)
        )

        calinski_harabasz = float(
            calinski_harabasz_score(X, y)
        )

    return {
        **summary,
        "scope": (
            "full"
            if summary["noise_articles"] == 0
            else "grouped_only"
        ),
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }
