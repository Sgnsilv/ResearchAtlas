"""Avaliação consolidada dos agrupamentos já gerados no ResearchAtlas.

Reutiliza embeddings, rótulos e manifestos versionados; não refaz
clusterizações. Cada configuração é avaliada isoladamente: a tabela
não ordena nem compara métricas entre configurações. A coluna
`comparability_group` indica apenas quais linhas compartilham espaço
vetorial, escopo e protocolo da Silhouette.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from research_atlas.evaluation.metrics import (
    DEFAULT_SEED,
    SILHOUETTE_SAMPLE_SIZE,
    evaluate_partition,
)
from research_atlas.features.tfidf import build_article_texts, build_tfidf

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data/processed"

BGE_REPRESENTATION = "bge-small-en-v1.5"
TFIDF_REPRESENTATION = "tfidf"

ID_COLUMNS = ("embedding_index", "openalex_id", "sampling_area")

TFIDF_VOCABULARY_SIZES = {"10k": 10_000, "20k": 20_000}

AGGLOMERATIVE_DISTANCE = {
    "average": "cosine",
    "complete": "cosine",
    "ward": "euclidean",
}

BGE_CONFIG_PATTERNS = (
    ("kmeans", re.compile(r"kmeans_k(?P<k>\d+)")),
    (
        "agglomerative",
        re.compile(
            r"agglomerative_(?P<linkage>average|complete|ward)_k(?P<k>\d+)"
        ),
    ),
    (
        "hdbscan",
        re.compile(
            r"hdbscan_(?P<selection_method>eom|leaf)"
            r"_mcs(?P<min_cluster_size>\d+)_ms(?P<min_samples>\d+)"
        ),
    ),
)

TFIDF_CONFIG_PATTERN = re.compile(r"tfidf_(?P<vocabulary>10k|20k)_k(?P<k>\d+)")

TABLE_COLUMNS = [
    "config_id",
    "representation",
    "feature_space",
    "n_features",
    "algorithm",
    "k",
    "linkage",
    "selection_method",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "clustering_distance",
    "n_articles",
    "n_clusters",
    "grouped_articles",
    "noise_articles",
    "coverage_pct",
    "noise_pct",
    "min_cluster_size",
    "max_cluster_size",
    "largest_cluster_pct_total",
    "scope",
    "valid",
    "reasons",
    "warnings",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "silhouette_unavailable_reason",
    "davies_bouldin_unavailable_reason",
    "calinski_harabasz_unavailable_reason",
    "silhouette_metric",
    "davies_bouldin_geometry",
    "calinski_harabasz_geometry",
    "input_l2_normalized",
    "input_sparse",
    "silhouette_sampling",
    "silhouette_sample_size",
    "silhouette_sampled",
    "sample_size_requested",
    "seed",
    "n_evaluated",
    "comparability_group",
]

INTEGER_COLUMNS = [
    "n_features",
    "k",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "n_articles",
    "n_clusters",
    "grouped_articles",
    "noise_articles",
    "min_cluster_size",
    "max_cluster_size",
    "silhouette_sample_size",
    "sample_size_requested",
    "seed",
    "n_evaluated",
]

BOOLEAN_COLUMNS = [
    "valid",
    "input_l2_normalized",
    "input_sparse",
    "silhouette_sampled",
]

EVALUATION_FIELDS = (
    "n_articles",
    "n_clusters",
    "grouped_articles",
    "noise_articles",
    "coverage_pct",
    "noise_pct",
    "min_cluster_size",
    "max_cluster_size",
    "scope",
    "valid",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "silhouette_unavailable_reason",
    "davies_bouldin_unavailable_reason",
    "calinski_harabasz_unavailable_reason",
    "silhouette_metric",
    "davies_bouldin_geometry",
    "calinski_harabasz_geometry",
    "input_l2_normalized",
    "input_sparse",
    "silhouette_sampling",
    "silhouette_sample_size",
    "silhouette_sampled",
    "sample_size_requested",
    "seed",
    "n_evaluated",
)

REFERENCE_TOLERANCE = {"rtol": 1e-6, "atol": 1e-7}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Checksum divergente: {Path(path).name}")


def parse_bge_config(config_id: str) -> dict:
    """Extrai algoritmo e parâmetros do nome de uma configuração BGE."""
    for algorithm, pattern in BGE_CONFIG_PATTERNS:
        match = pattern.fullmatch(config_id)

        if match is None:
            continue

        params = match.groupdict()
        linkage = params.get("linkage")

        return {
            "algorithm": algorithm,
            "k": int(params["k"]) if "k" in params else None,
            "linkage": linkage,
            "selection_method": params.get("selection_method"),
            "hdbscan_min_cluster_size": (
                int(params["min_cluster_size"])
                if "min_cluster_size" in params
                else None
            ),
            "hdbscan_min_samples": (
                int(params["min_samples"])
                if "min_samples" in params
                else None
            ),
            "clustering_distance": (
                AGGLOMERATIVE_DISTANCE[linkage]
                if linkage
                else "euclidean"
            ),
        }

    raise ValueError(f"Configuração desconhecida: {config_id}")


def parse_tfidf_config(config_id: str) -> dict:
    match = TFIDF_CONFIG_PATTERN.fullmatch(config_id)

    if match is None:
        raise ValueError(f"Configuração desconhecida: {config_id}")

    return {
        "vocabulary": match["vocabulary"],
        "algorithm": "kmeans",
        "k": int(match["k"]),
        "linkage": None,
        "selection_method": None,
        "hdbscan_min_cluster_size": None,
        "hdbscan_min_samples": None,
        "clustering_distance": "euclidean",
    }


def label_columns(assignments: pd.DataFrame) -> list[str]:
    columns = [c for c in assignments.columns if c not in ID_COLUMNS]

    if not columns:
        raise ValueError("Nenhuma configuração de rótulos encontrada")

    for column in columns:
        if not pd.api.types.is_integer_dtype(assignments[column]):
            raise ValueError(f"Rótulos não inteiros: {column}")

    return columns


def validate_order(frame: pd.DataFrame, n_articles: int) -> None:
    if len(frame) != n_articles:
        raise ValueError("Quantidade de artigos incompatível")

    if not (frame["embedding_index"].to_numpy() == np.arange(n_articles)).all():
        raise ValueError("embedding_index fora da ordem original")


def validate_ids(
    ids: pd.Series,
    n_articles: int,
    reference_ids: pd.Series | None = None,
) -> pd.Series:
    ids = ids.reset_index(drop=True)

    if len(ids) != n_articles:
        raise ValueError("Quantidade de artigos incompatível")

    if ids.duplicated().any():
        raise ValueError("Identificadores duplicados")

    if reference_ids is not None and not ids.equals(
        reference_ids.reset_index(drop=True)
    ):
        raise ValueError("Identificadores divergentes entre artefatos")

    return ids


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_bge_inputs(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Carrega embeddings e rótulos BGE verificando checksums e ordem."""
    processed_dir = Path(processed_dir)
    embedding_dir = processed_dir / "embeddings/bge-small-en-v1.5-v1"
    clustering_dir = processed_dir / "clustering/v1"

    embedding_manifest = _read_json(embedding_dir / "manifest.json")
    clustering_manifest = _read_json(clustering_dir / "manifest.json")

    embedding_path = embedding_dir / "embeddings.npy"
    assignments_path = clustering_dir / "cluster_assignments.parquet"

    verify_sha256(embedding_path, embedding_manifest["embedding_file_sha256"])
    verify_sha256(
        assignments_path,
        clustering_manifest["files_sha256"]["cluster_assignments.parquet"],
    )

    if (
        clustering_manifest["embedding_sha256"]
        != embedding_manifest["embedding_file_sha256"]
    ):
        raise ValueError("Embeddings divergentes entre manifestos")

    embeddings = np.load(embedding_path, allow_pickle=False)
    metadata = pd.read_parquet(embedding_dir / "metadata.parquet")
    assignments = pd.read_parquet(assignments_path)

    n_articles = len(embeddings)

    validate_order(metadata, n_articles)
    validate_order(assignments, n_articles)

    metadata_ids = validate_ids(metadata["openalex_id"], n_articles)
    validate_ids(assignments["openalex_id"], n_articles, metadata_ids)

    return embeddings, assignments


def load_tfidf_inputs(
    reference_ids: pd.Series,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[dict[str, csr_matrix], pd.DataFrame]:
    """
    Reconstrói as matrizes TF-IDF esparsas a partir do dataset versionado.

    A matriz não é versionada; os parâmetros vêm do manifesto do baseline.
    """
    processed_dir = Path(processed_dir)
    tfidf_dir = processed_dir / "tfidf_baseline/v1"
    dataset_path = processed_dir / "dataset_selected.parquet"

    manifest = _read_json(tfidf_dir / "manifest.json")
    assignments_path = tfidf_dir / "tfidf_assignments.parquet"

    verify_sha256(dataset_path, manifest["dataset_sha256"])
    verify_sha256(
        assignments_path,
        manifest["files_sha256"]["tfidf_assignments.parquet"],
    )

    dataset = pd.read_parquet(dataset_path)
    assignments = pd.read_parquet(assignments_path)

    n_articles = len(reference_ids)

    validate_ids(dataset["openalex_id"], n_articles, reference_ids)
    validate_order(assignments, n_articles)
    validate_ids(assignments["openalex_id"], n_articles, reference_ids)

    params = manifest["primary_configuration"]
    texts = build_article_texts(dataset)

    matrices = {}

    for name, max_features in TFIDF_VOCABULARY_SIZES.items():
        matrix, _ = build_tfidf(
            texts,
            max_features=max_features,
            min_df=params["min_df"],
            ngram_range=tuple(params["ngram_range"]),
        )

        if matrix.shape != (len(texts), max_features):
            raise ValueError(f"Vocabulário TF-IDF inesperado: {name}")

        matrices[name] = matrix

    return matrices, assignments


def comparability_group(
    feature_space: str,
    config_id: str,
    result: dict,
) -> str:
    """
    Identifica linhas que podem ser comparadas diretamente.

    Partições `grouped_only` avaliam subconjuntos distintos e nunca
    compartilham grupo.
    """
    if result["scope"] == "grouped_only":
        return f"{feature_space}|grouped_only|{config_id}"

    protocol = (
        f"{result['silhouette_sampling']}"
        f":{result['silhouette_sample_size']}"
        f":seed{result['seed']}"
    )

    return f"{feature_space}|full|{protocol}"


def _row(
    config_id: str,
    representation: str,
    feature_space: str,
    n_features: int,
    parameters: dict,
    result: dict,
) -> dict:
    return {
        "config_id": config_id,
        "representation": representation,
        "feature_space": feature_space,
        "n_features": n_features,
        **{
            key: parameters[key]
            for key in (
                "algorithm",
                "k",
                "linkage",
                "selection_method",
                "hdbscan_min_cluster_size",
                "hdbscan_min_samples",
                "clustering_distance",
            )
        },
        **{field: result[field] for field in EVALUATION_FIELDS},
        "reasons": ";".join(result["reasons"]),
        "warnings": ";".join(result["warnings"]),
        "largest_cluster_pct_total": (
            100 * result["max_cluster_size"] / result["n_articles"]
        ),
        "comparability_group": comparability_group(
            feature_space,
            config_id,
            result,
        ),
    }


def _to_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)

    for column in INTEGER_COLUMNS:
        table[column] = table[column].astype("Int64")

    for column in BOOLEAN_COLUMNS:
        table[column] = table[column].astype("boolean")

    return table


def evaluate_bge(
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Avalia cada configuração BGE com `evaluate_partition`."""
    feature_space = f"{BGE_REPRESENTATION}:{embeddings.shape[1]}d_l2"
    rows = []

    for config_id in label_columns(assignments):
        result = evaluate_partition(
            embeddings,
            assignments[config_id].to_numpy(),
            sample_size=sample_size,
            seed=seed,
        )

        rows.append(
            _row(
                config_id,
                BGE_REPRESENTATION,
                feature_space,
                embeddings.shape[1],
                parse_bge_config(config_id),
                result,
            )
        )

    return _to_table(rows)


def evaluate_tfidf(
    matrices: dict[str, csr_matrix],
    assignments: pd.DataFrame,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Avalia o TF-IDF esparso: apenas Silhouette cosseno."""
    rows = []

    for config_id in label_columns(assignments):
        parameters = parse_tfidf_config(config_id)
        matrix = matrices[parameters["vocabulary"]]

        result = evaluate_partition(
            matrix,
            assignments[config_id].to_numpy(),
            sample_size=sample_size,
            seed=seed,
        )

        rows.append(
            _row(
                config_id,
                TFIDF_REPRESENTATION,
                f"{TFIDF_REPRESENTATION}:{matrix.shape[1]}f_sparse",
                matrix.shape[1],
                parameters,
                result,
            )
        )

    return _to_table(rows)


def build_evaluation_table(
    processed_dir: Path = PROCESSED_DIR,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Avalia as configurações BGE e TF-IDF versionadas."""
    embeddings, bge_assignments = load_bge_inputs(processed_dir)
    matrices, tfidf_assignments = load_tfidf_inputs(
        bge_assignments["openalex_id"],
        processed_dir,
    )

    bge = evaluate_bge(
        embeddings,
        bge_assignments,
        sample_size=sample_size,
        seed=seed,
    )
    tfidf = evaluate_tfidf(
        matrices,
        tfidf_assignments,
        sample_size=sample_size,
        seed=seed,
    )

    return _to_table(
        pd.concat([bge, tfidf], ignore_index=True).to_dict("records")
    )


REFERENCE_SOURCES = (
    (
        "clustering/v1/kmeans_metrics.csv",
        lambda row: f"kmeans_k{int(row['k'])}",
        {
            "silhouette": "silhouette",
            "davies_bouldin": "davies_bouldin",
            "calinski_harabasz": "calinski_harabasz",
            "min_cluster_size": "min_cluster_size",
            "max_cluster_size": "max_cluster_size",
        },
    ),
    (
        "clustering/v1/agglomerative_metrics.csv",
        lambda row: f"agglomerative_{row['linkage']}_k8",
        {
            "silhouette": "silhouette",
            "davies_bouldin": "davies_bouldin",
            "calinski_harabasz": "calinski_harabasz",
            "min_cluster_size": "min_cluster_size",
            "max_cluster_size": "max_cluster_size",
        },
    ),
    (
        "clustering/v1/hdbscan_eom_metrics.csv",
        lambda row: (
            f"hdbscan_eom_mcs{int(row['min_cluster_size'])}"
            f"_ms{int(row['min_samples'])}"
        ),
        {
            "n_clusters": "n_clusters",
            "grouped_articles": "grouped_articles",
            "noise_articles": "noise_articles",
            "noise_pct": "noise_pct",
            "min_cluster_size": "min_cluster_size_found",
            "max_cluster_size": "max_cluster_size_found",
        },
    ),
    (
        "clustering/v1/hdbscan_leaf_metrics.csv",
        lambda row: (
            f"hdbscan_leaf_mcs{int(row['min_cluster_size'])}"
            f"_ms{int(row['min_samples'])}"
        ),
        {
            "n_clusters": "n_clusters",
            "grouped_articles": "grouped_articles",
            "coverage_pct": "coverage_pct",
            "min_cluster_size": "min_size",
            "max_cluster_size": "max_size",
            "silhouette": "silhouette_grouped_only",
        },
    ),
    (
        "tfidf_baseline/v1/tfidf_final_candidates.csv",
        lambda row: f"tfidf_10k_k{int(row['k'])}",
        {
            "silhouette": "silhouette",
            "min_cluster_size": "min_cluster_size",
            "max_cluster_size": "max_cluster_size",
        },
    ),
    (
        "tfidf_baseline/v1/kmeans_k_sensitivity.csv",
        lambda row: f"tfidf_20k_k{int(row['k'])}",
        {
            "silhouette": "silhouette",
            "min_cluster_size": "min_cluster_size",
            "max_cluster_size": "max_cluster_size",
        },
    ),
)


def reconcile_with_reference(
    table: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Confronta a tabela com os CSVs das Issues #06 e #07.

    Só são comparados valores presentes nos CSVs originais; configurações
    ou campos ausentes neles não são preenchidos.
    """
    processed_dir = Path(processed_dir)
    indexed = table.set_index("config_id")
    records = []

    for filename, config_of, fields in REFERENCE_SOURCES:
        reference = pd.read_csv(processed_dir / filename)

        for _, row in reference.iterrows():
            config_id = config_of(row)

            if config_id not in indexed.index:
                continue

            for field, column in fields.items():
                value = indexed.at[config_id, field]
                expected = row[column]

                records.append(
                    {
                        "config_id": config_id,
                        "source": filename,
                        "field": field,
                        "value": float(value),
                        "reference": float(expected),
                        "abs_diff": abs(float(value) - float(expected)),
                        "match": bool(
                            pd.notna(value)
                            and np.isclose(
                                float(value),
                                float(expected),
                                **REFERENCE_TOLERANCE,
                            )
                        ),
                    }
                )

    return pd.DataFrame(records)


def main() -> None:
    started = time.perf_counter()

    table = build_evaluation_table()
    reconciliation = reconcile_with_reference(table)

    elapsed = time.perf_counter() - started

    print(f"Configurações avaliadas: {len(table)}")
    print(table.groupby(["representation", "scope"]).size().to_string())
    print(
        "Métricas calculadas:",
        table[["silhouette", "davies_bouldin", "calinski_harabasz"]]
        .notna()
        .sum()
        .to_dict(),
    )
    print(
        f"Conferência com CSVs: {int(reconciliation['match'].sum())}"
        f"/{len(reconciliation)}; maior diferença absoluta: "
        f"{reconciliation['abs_diff'].max():.3g}"
    )
    print(f"Tempo: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
