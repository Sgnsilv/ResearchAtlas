"""Algoritmos de clusterização dos embeddings do ResearchAtlas."""

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans


def validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Valida a matriz de embeddings antes da clusterização."""
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings devem ser um array NumPy")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings devem ter duas dimensões")

    if embeddings.shape[0] < 3 or embeddings.shape[1] == 0:
        raise ValueError("Matriz de embeddings com formato inválido")

    if not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError("Embeddings devem conter valores numéricos")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contêm valores não finitos")

    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Embeddings contêm vetores nulos")

    return embeddings


def run_kmeans(
    embeddings: np.ndarray,
    *,
    k: int = 8,
    seed: int = 42,
    n_init: int = 10,
) -> tuple[np.ndarray, KMeans]:
    """Executa K-Means com inicialização reprodutível."""
    matrix = validate_embeddings(embeddings)

    if not isinstance(k, int) or not 2 <= k < len(matrix):
        raise ValueError("Número de clusters inválido")

    if not isinstance(n_init, int) or n_init < 1:
        raise ValueError("n_init deve ser positivo")

    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
    )

    labels = model.fit_predict(matrix)

    return labels, model


def run_agglomerative(
    embeddings: np.ndarray,
    *,
    k: int = 8,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[np.ndarray, AgglomerativeClustering]:
    """Executa agrupamento hierárquico aglomerativo."""
    matrix = validate_embeddings(embeddings)

    if not isinstance(k, int) or not 2 <= k < len(matrix):
        raise ValueError("Número de clusters inválido")

    valid_linkages = {"ward", "complete", "average", "single"}

    if linkage not in valid_linkages:
        raise ValueError("Método de ligação inválido")

    if metric not in {"euclidean", "cosine"}:
        raise ValueError("Métrica de distância inválida")

    if linkage == "ward" and metric != "euclidean":
        raise ValueError("Ward exige distância euclidiana")

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric,
    )

    labels = model.fit_predict(matrix)

    return labels, model


def run_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    selection_method: str = "eom",
    metric: str = "euclidean",
    core_dist_n_jobs: int = -1,
) -> tuple[np.ndarray, HDBSCAN]:
    """Executa HDBSCAN preservando artigos classificados como ruído."""
    matrix = validate_embeddings(embeddings)

    if (
        not isinstance(min_cluster_size, int)
        or not 2 <= min_cluster_size <= len(matrix)
    ):
        raise ValueError("min_cluster_size inválido")

    if not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples inválido")

    if selection_method not in {"eom", "leaf"}:
        raise ValueError("Método de seleção inválido")

    if metric != "euclidean":
        raise ValueError(
            "Este pipeline utiliza distância euclidiana no HDBSCAN"
        )

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=selection_method,
        core_dist_n_jobs=core_dist_n_jobs,
    )

    labels = model.fit_predict(matrix)

    return labels, model


def summarize_clusters(labels: np.ndarray) -> dict:
    """Resume cobertura, ruído e tamanhos dos clusters."""
    labels = np.asarray(labels)

    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Rótulos devem formar um vetor não vazio")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Rótulos devem ser inteiros")

    if np.any(labels < -1):
        raise ValueError("Rótulos inválidos")

    valid_mask = labels != -1
    cluster_ids, counts = np.unique(
        labels[valid_mask],
        return_counts=True,
    )

    grouped = int(valid_mask.sum())
    noise = int((~valid_mask).sum())
    total = len(labels)

    return {
        "n_articles": total,
        "n_clusters": int(len(cluster_ids)),
        "grouped_articles": grouped,
        "noise_articles": noise,
        "coverage_pct": 100 * grouped / total,
        "noise_pct": 100 * noise / total,
        "min_cluster_size": int(counts.min()) if counts.size else 0,
        "max_cluster_size": int(counts.max()) if counts.size else 0,
    }
