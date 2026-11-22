
"""Pipeline reutilizável do baseline TF-IDF + K-Means."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def build_article_texts(articles: pd.DataFrame) -> list[str]:
    """Combina título e resumo preservando a ordem dos artigos."""
    required = {"title", "abstract"}

    if not required.issubset(articles.columns):
        raise ValueError("Dataset sem título ou resumo")

    if articles.empty:
        raise ValueError("Dataset vazio")

    texts = (
        articles["title"].fillna("").astype(str)
        + ". "
        + articles["abstract"].fillna("").astype(str)
    ).tolist()

    if any(not text.strip(" .") for text in texts):
        raise ValueError("Existem artigos sem texto")

    return texts


def build_tfidf(
    texts: list[str],
    *,
    max_features: int = 10000,
    min_df: int = 3,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[csr_matrix, TfidfVectorizer]:
    """Constrói a representação TF-IDF do corpus."""
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Corpus vazio ou com textos inválidos")

    if max_features <= 0 or min_df <= 0:
        raise ValueError("Parâmetros TF-IDF devem ser positivos")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(texts).tocsr()

    if matrix.shape[1] == 0 or not np.isfinite(matrix.data).all():
        raise ValueError("Matriz TF-IDF inválida")

    return matrix, vectorizer


def run_kmeans(
    matrix: csr_matrix,
    *,
    k: int = 8,
    seed: int = 42,
    n_init: int = 10,
) -> tuple[np.ndarray, KMeans]:
    """Executa K-Means sobre a representação textual."""
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Matriz de características inválida")

    if not np.isfinite(matrix.data).all():
        raise ValueError("Matriz contém valores não finitos")

    if not 2 <= k < matrix.shape[0]:
        raise ValueError("k deve estar entre 2 e n_artigos - 1")

    if n_init < 1:
        raise ValueError("n_init deve ser positivo")

    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
    )

    labels = model.fit_predict(matrix)

    return labels, model



def evaluate_clustering(
    matrix: csr_matrix,
    labels: np.ndarray,
    *,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict:
    """Avalia clusters usando o protocolo de Silhouette do Colab."""
    labels = np.asarray(labels)

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Matriz de características inválida")

    if not np.isfinite(matrix.data).all():
        raise ValueError("Matriz contém valores não finitos")

    if labels.ndim != 1 or len(labels) != matrix.shape[0]:
        raise ValueError("Rótulos incompatíveis com os artigos")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    if not 2 <= len(unique_labels) < len(labels):
        raise ValueError("Partição inválida para Silhouette")

    if sample_size < 2:
        raise ValueError("Amostra de avaliação inválida")

    effective_sample = min(sample_size, matrix.shape[0])

    # Mesmo protocolo utilizado nos experimentos do Colab.
    score = silhouette_score(
        matrix,
        labels,
        metric="cosine",
        sample_size=(
            effective_sample
            if effective_sample < matrix.shape[0]
            else None
        ),
        random_state=seed,
    )

    return {
        "n_clusters": int(len(unique_labels)),
        "silhouette": float(score),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "largest_cluster_pct": float(
            counts.max() / len(labels) * 100
        ),
    }
