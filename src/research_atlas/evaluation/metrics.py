"""Avaliação interna de partições de clusterização do ResearchAtlas."""

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)

from research_atlas.clustering.algorithms import (
    summarize_clusters,
    validate_embeddings,
)

SILHOUETTE_SAMPLE_SIZE = 2000
DEFAULT_SEED = 42
MAX_SEED = 2**32 - 1

MAX_CLUSTER_SHARE = 0.8
LOW_COVERAGE_PCT = 50.0
SILHOUETTE_FULL_FALLBACK_MAX = 10_000
NORM_TOLERANCE = 1e-3
DEGENERACY_TOLERANCE = 1e-9

SILHOUETTE_METRIC = "cosine"
DAVIES_BOULDIN_GEOMETRY = "euclidean"
CALINSKI_HARABASZ_GEOMETRY = "euclidean"


def _as_integer(value, message: str) -> int:
    """Aceita inteiros Python ou NumPy; rejeita bool e outros tipos."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(message)

    return int(value)


def _is_valid_silhouette_sample(labels: np.ndarray) -> bool:
    """Exige 2 <= n_rótulos <= n_amostras - 1."""
    n_labels = len(np.unique(labels))

    return 2 <= n_labels <= len(labels) - 1


def _compute_silhouette(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> tuple[float | None, dict]:
    """
    Calcula a Silhouette cosseno com amostragem reprodutível.

    Se a amostra for inválida, usa todos os artigos avaliados quando
    viável; caso contrário, não calcula.
    """
    n_samples = len(y)
    warnings = []

    if sample_size >= n_samples:
        indices = None
        sampling = "full"
    else:
        candidate = np.random.RandomState(seed).permutation(n_samples)[
            :sample_size
        ]

        if _is_valid_silhouette_sample(y[candidate]):
            indices = candidate
            sampling = "sampled"
        elif n_samples <= SILHOUETTE_FULL_FALLBACK_MAX:
            indices = None
            sampling = "full_fallback"
            warnings.append("silhouette_amostra_invalida_base_completa")
        else:
            return None, {
                "silhouette_sampling": "nao_calculada_amostra_invalida",
                "silhouette_sample_size": None,
                "silhouette_sampled": None,
                "warnings": ["silhouette_nao_calculada_amostra_invalida"],
            }

    if indices is None:
        X_eval, y_eval = X, y
    else:
        X_eval, y_eval = X[indices], y[indices]

    score = float(
        silhouette_score(X_eval, y_eval, metric=SILHOUETTE_METRIC)
    )

    return score, {
        "silhouette_sampling": sampling,
        "silhouette_sample_size": int(len(y_eval)),
        "silhouette_sampled": sampling == "sampled",
        "warnings": warnings,
    }


def _centroid_diagnostics(X: np.ndarray, y: np.ndarray) -> dict:
    """Mede distância entre centroides e dispersão intracluster."""
    ids, inverse, counts = np.unique(
        y,
        return_inverse=True,
        return_counts=True,
    )

    order = np.argsort(inverse, kind="stable")
    sorted_X = np.asarray(X[order], dtype=np.float64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    centroids = (
        np.add.reduceat(sorted_X, starts, axis=0) / counts[:, None]
    )

    distances_to_centroid = np.linalg.norm(
        sorted_X - np.repeat(centroids, counts, axis=0),
        axis=1,
    )
    max_dispersion = np.maximum.reduceat(distances_to_centroid, starts)

    pairwise = pairwise_distances(centroids)
    min_centroid_distance = float(
        pairwise[np.triu_indices(len(ids), k=1)].min()
    )

    scale = max(1.0, float(np.linalg.norm(sorted_X, axis=1).max()))

    return {
        "tolerance": DEGENERACY_TOLERANCE * scale,
        "counts": counts,
        "max_dispersion": max_dispersion,
        "min_centroid_distance": min_centroid_distance,
    }


def evaluate_partition(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Avalia uma partição de clusterização.

    Calcula Silhouette (cosseno), Davies-Bouldin e Calinski-Harabasz
    (euclidianos) quando a partição é válida.

    Artigos de ruído (-1) são excluídos; nesse caso 'scope' é
    'grouped_only' e as métricas não são comparáveis às de partições
    'full'.

    Partições inválidas ou degeneradas retornam métricas None e os
    motivos em 'reasons'. 'valid' refere-se à partição; consulte
    'silhouette_sampling' para saber como a Silhouette foi obtida.

    Erros de entrada geram ValueError.
    """
    matrix = validate_embeddings(embeddings)
    labels = np.asarray(labels)

    if labels.ndim != 1 or len(labels) != len(matrix):
        raise ValueError("Rótulos incompatíveis com os embeddings")

    sample_size = _as_integer(
        sample_size,
        "sample_size deve ser um inteiro >= 2",
    )

    if sample_size < 2:
        raise ValueError("sample_size deve ser um inteiro >= 2")

    seed = _as_integer(
        seed,
        f"seed deve ser um inteiro entre 0 e {MAX_SEED}",
    )

    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed deve ser um inteiro entre 0 e {MAX_SEED}")

    summary = summarize_clusters(labels)

    mask = labels != -1
    X = matrix[mask]
    y = labels[mask]

    reasons = []
    warnings = []

    n_samples = summary["grouped_articles"]
    n_clusters = summary["n_clusters"]

    if n_samples == 0:
        reasons.append("sem_artigos_agrupados")

    elif n_clusters < 2:
        reasons.append("menos_de_2_clusters")

    elif n_clusters >= n_samples:
        reasons.append("clusters_insuficientes_para_metricas")

    if not reasons:
        diagnostics = _centroid_diagnostics(X, y)
        tolerance = diagnostics["tolerance"]

        if diagnostics["min_centroid_distance"] <= tolerance:
            reasons.append("centroides_coincidentes")

        if (diagnostics["max_dispersion"] <= tolerance).all():
            reasons.append("dispersao_intracluster_nula")

        elif (
            (diagnostics["max_dispersion"] <= tolerance)
            & (diagnostics["counts"] > 1)
        ).any():
            warnings.append("cluster_sem_dispersao")

    if summary["min_cluster_size"] == 1:
        warnings.append("cluster_unitario")

    if n_samples > 0:
        largest_share = summary["max_cluster_size"] / n_samples

        if largest_share > MAX_CLUSTER_SHARE:
            warnings.append("cluster_dominante")

    if (
        summary["noise_articles"] > 0
        and summary["coverage_pct"] < LOW_COVERAGE_PCT
    ):
        warnings.append("cobertura_baixa")

    norms = np.linalg.norm(matrix, axis=1)
    input_l2_normalized = bool(
        np.allclose(norms, 1.0, atol=NORM_TOLERANCE)
    )

    if not input_l2_normalized:
        warnings.append("embeddings_nao_normalizados")

    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None

    silhouette_info = {
        "silhouette_sampling": None,
        "silhouette_sample_size": None,
        "silhouette_sampled": None,
    }

    if not reasons:
        silhouette, info = _compute_silhouette(
            X,
            y,
            sample_size=sample_size,
            seed=seed,
        )

        warnings.extend(info.pop("warnings"))
        silhouette_info = info

        davies_bouldin = float(davies_bouldin_score(X, y))
        calinski_harabasz = float(calinski_harabasz_score(X, y))

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
        "n_evaluated": int(n_samples),
        "silhouette_metric": SILHOUETTE_METRIC,
        "davies_bouldin_geometry": DAVIES_BOULDIN_GEOMETRY,
        "calinski_harabasz_geometry": CALINSKI_HARABASZ_GEOMETRY,
        "input_l2_normalized": input_l2_normalized,
        "seed": seed,
        "sample_size_requested": sample_size,
        **silhouette_info,
    }
