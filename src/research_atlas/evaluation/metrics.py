"""Avaliação interna de partições de clusterização do ResearchAtlas."""

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from research_atlas.clustering.algorithms import summarize_clusters

SILHOUETTE_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42

# Acima desta fração dos artigos agrupados em um único cluster,
# a partição é considerada degenerada (gera aviso, não erro).
MAX_CLUSTER_SHARE = 0.8


def evaluate_partition(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Calcula Silhouette, Davies-Bouldin e Calinski-Harabasz quando válidos."""
    labels = np.asarray(labels)

    # 1. Validação de formato: estas condições SIM lançam ValueError,
    #    pois indicam erro de uso e não uma partição ruim.
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("Rótulos incompatíveis com os embeddings")

    # 2. Reaproveita o resumo que já existe (cobertura, ruído, tamanhos).
    summary = summarize_clusters(labels)

    # 3. Seleciona apenas os artigos agrupados (fora do ruído).
    mask = labels != -1
    X, y = embeddings[mask], labels[mask]

    # 4. Condições de invalidez e avisos.
    reasons, warnings = [], []

    if summary["n_clusters"] < 2:
        reasons.append("menos_de_2_clusters")
    elif summary["grouped_articles"] <= 1:
        # a Silhouette exige 2 <= n_clusters <= n_amostras - 1
        reasons.append("agrupados_insuficientes")

    if summary["min_cluster_size"] == 1:
        warnings.append("cluster_unitario")

    if summary["grouped_articles"] > 0 and (
        summary["max_cluster_size"] / summary["grouped_articles"] > MAX_CLUSTER_SHARE
    ):
        warnings.append("cluster_dominante")

    # 5. Métricas só quando a partição é válida.
    silhouette = davies_bouldin = calinski_harabasz = None

    if not reasons:
        effective = min(sample_size, len(X))
        silhouette = float(
            silhouette_score(
                X,
                y,
                metric="euclidean",
                sample_size=effective if effective < len(X) else None,
                random_state=seed,
            )
        )
        davies_bouldin = float(davies_bouldin_score(X, y))
        calinski_harabasz = float(calinski_harabasz_score(X, y))

    return {
        **summary,
        "scope": "full" if summary["noise_articles"] == 0 else "grouped_only",
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }