"""Estabilidade entre sementes e concordância por ARI entre partições.

Distingue explicitamente:
- `seed_initialization`: variação entre sementes do mesmo algoritmo e espaço;
- `cross_algorithm`, `cross_representation` e `hyperparameter_sensitivity`:
  concordância estrutural entre partições distintas (não é estabilidade de
  inicialização nem reamostragem).

Para partições com ruído (`-1`), o ARI é calculado apenas na interseção dos
artigos agrupados por ambas as configurações.
"""

import hashlib
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from research_atlas.clustering.algorithms import summarize_clusters
from research_atlas.evaluation.metrics import LOW_COVERAGE_PCT, _as_integer
from research_atlas.evaluation.pipeline import (
    BGE_REPRESENTATION,
    PROCESSED_DIR,
    REFERENCE_TOLERANCE,
    TFIDF_REPRESENTATION,
    _read_json,
    label_columns,
    parse_bge_config,
    parse_tfidf_config,
    validate_ids,
    validate_order,
    verify_sha256,
)

DEFAULT_MIN_INTERSECTION_SIZE = 3

SEED_STABILITY_COLUMNS = [
    "config_id",
    "representation",
    "feature_space",
    "algorithm",
    "k",
    "stability_type",
    "n_seeds",
    "n_seed_pairs",
    "seeds",
    "ari_mean",
    "ari_std",
    "ari_min",
    "ari_max",
    "source_artifact",
]

CONCORDANCE_COLUMNS = [
    "config_a",
    "config_b",
    "representation_a",
    "representation_b",
    "algorithm_a",
    "algorithm_b",
    "scope_a",
    "scope_b",
    "comparison_type",
    "comparison_scope",
    "n_articles",
    "n_grouped_a",
    "n_grouped_b",
    "n_intersection",
    "intersection_coverage_pct",
    "n_clusters_a_total",
    "n_clusters_b_total",
    "n_clusters_a_intersection",
    "n_clusters_b_intersection",
    "identical_partition",
    "exact_label_match",
    "valid",
    "reasons",
    "warnings",
    "ari",
    "ari_unavailable_reason",
]

IDENTICAL_PARTITIONS_COLUMNS = [
    "config_id",
    "representation",
    "algorithm",
    "scope",
    "n_clusters",
    "grouped_articles",
    "partition_signature",
    "identical_group_id",
    "identical_group_size",
    "is_duplicate_partition",
    "identical_configs",
]

IDENTICAL_PAIR_COLUMNS = [
    "config_a",
    "config_b",
    "representation_a",
    "representation_b",
    "algorithm_a",
    "algorithm_b",
    "scope",
    "n_articles",
    "grouped_articles",
    "noise_articles",
    "coverage_pct",
    "n_clusters",
    "exact_label_match",
    "ari",
    "partition_signature",
]

CONCORDANCE_INTEGER_COLUMNS = [
    "n_articles",
    "n_grouped_a",
    "n_grouped_b",
    "n_intersection",
    "n_clusters_a_total",
    "n_clusters_b_total",
    "n_clusters_a_intersection",
    "n_clusters_b_intersection",
]

CONCORDANCE_BOOLEAN_COLUMNS = [
    "identical_partition",
    "exact_label_match",
    "valid",
]


def _validate_labels(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    array = np.asarray(labels)
    summary = summarize_clusters(array)

    if len(array) < 3:
        raise ValueError("Rótulos devem conter ao menos 3 artigos")

    return array, summary


def canonical_partition_signature(labels: np.ndarray) -> str:
    """Assinatura invariante a permutação de IDs dos clusters, preservando -1."""
    array, _ = _validate_labels(labels)
    canonical = np.full(len(array), -1, dtype=np.int64)
    mask = array != -1

    if mask.any():
        mapping: dict[int, int] = {}
        remapped = np.empty(int(mask.sum()), dtype=np.int64)

        for idx, raw in enumerate(array[mask]):
            value = int(raw)
            if value not in mapping:
                mapping[value] = len(mapping)
            remapped[idx] = mapping[value]

        canonical[mask] = remapped

    return hashlib.sha256(canonical.tobytes()).hexdigest()


def compare_partitions(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    min_intersection_size: int = DEFAULT_MIN_INTERSECTION_SIZE,
) -> dict:
    """
    Calcula o ARI entre duas partições alinhadas de artigos.

    Artigos com rótulo -1 são excluídos; quando presente em qualquer
    partição, o cálculo ocorre apenas na interseção agrupada.
    """
    min_intersection_size = _as_integer(
        min_intersection_size,
        "min_intersection_size deve ser um inteiro >= 2",
    )

    if min_intersection_size < 2:
        raise ValueError("min_intersection_size deve ser um inteiro >= 2")

    a, summary_a = _validate_labels(labels_a)
    b, summary_b = _validate_labels(labels_b)

    if len(a) != len(b):
        raise ValueError("Rótulos com comprimentos incompatíveis")

    n_articles = len(a)
    grouped_a = a != -1
    grouped_b = b != -1
    intersection_mask = grouped_a & grouped_b

    n_intersection = int(intersection_mask.sum())
    intersection_coverage_pct = 100.0 * n_intersection / n_articles

    ya = a[intersection_mask]
    yb = b[intersection_mask]

    clusters_a_int = int(len(np.unique(ya))) if n_intersection else 0
    clusters_b_int = int(len(np.unique(yb))) if n_intersection else 0

    has_noise = (
        summary_a["noise_articles"] > 0 or summary_b["noise_articles"] > 0
    )
    comparison_scope = "grouped_intersection" if has_noise else "full"

    reasons = []
    warnings = []

    if n_intersection == 0:
        reasons.append("intersecao_vazia")
    elif n_intersection < 2:
        reasons.append("intersecao_insuficiente")
    elif n_intersection < min_intersection_size:
        reasons.append("intersecao_abaixo_do_minimo")
    elif clusters_a_int < 2 or clusters_b_int < 2:
        reasons.append("menos_de_2_clusters_na_intersecao")
    elif clusters_a_int >= n_intersection or clusters_b_int >= n_intersection:
        reasons.append("clusters_insuficientes_na_intersecao")

    if comparison_scope == "grouped_intersection":
        warnings.append("intersecao_parcial")
        if intersection_coverage_pct < LOW_COVERAGE_PCT:
            warnings.append("cobertura_intersecao_baixa")

    if n_intersection > 0 and (
        clusters_a_int < summary_a["n_clusters"]
        or clusters_b_int < summary_b["n_clusters"]
    ):
        warnings.append("clusters_ausentes_na_intersecao")

    ari = float(adjusted_rand_score(ya, yb)) if not reasons else None

    return {
        "comparison_scope": comparison_scope,
        "n_articles": n_articles,
        "n_grouped_a": summary_a["grouped_articles"],
        "n_grouped_b": summary_b["grouped_articles"],
        "n_intersection": n_intersection,
        "intersection_coverage_pct": intersection_coverage_pct,
        "n_clusters_a_total": summary_a["n_clusters"],
        "n_clusters_b_total": summary_b["n_clusters"],
        "n_clusters_a_intersection": clusters_a_int,
        "n_clusters_b_intersection": clusters_b_int,
        "identical_partition": (
            canonical_partition_signature(a)
            == canonical_partition_signature(b)
        ),
        "exact_label_match": bool(np.array_equal(a, b)),
        "valid": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "ari": ari,
        "ari_unavailable_reason": reasons[0] if reasons else None,
    }


def classify_comparison(meta_a: dict, meta_b: dict) -> str:
    if meta_a["config_id"] == meta_b["config_id"]:
        return "self"

    if meta_a["representation"] != meta_b["representation"]:
        return "cross_representation"

    if meta_a["algorithm"] != meta_b["algorithm"]:
        return "cross_algorithm"

    return "hyperparameter_sensitivity"


def load_aligned_assignments(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega rótulos BGE e TF-IDF validando hashes, ordem e IDs."""
    processed_dir = Path(processed_dir)
    embedding_dir = processed_dir / "embeddings/bge-small-en-v1.5-v1"
    clustering_dir = processed_dir / "clustering/v1"
    tfidf_dir = processed_dir / "tfidf_baseline/v1"
    dataset_path = processed_dir / "dataset_selected.parquet"

    embedding_manifest = _read_json(embedding_dir / "manifest.json")
    clustering_manifest = _read_json(clustering_dir / "manifest.json")
    tfidf_manifest = _read_json(tfidf_dir / "manifest.json")

    if (
        clustering_manifest["embedding_sha256"]
        != embedding_manifest["embedding_file_sha256"]
    ):
        raise ValueError("Embeddings divergentes entre manifestos")

    if (
        tfidf_manifest["dataset_sha256"]
        != embedding_manifest["dataset_sha256"]
    ):
        raise ValueError("Dataset divergente entre manifestos")

    bge_path = clustering_dir / "cluster_assignments.parquet"
    tfidf_path = tfidf_dir / "tfidf_assignments.parquet"

    verify_sha256(dataset_path, tfidf_manifest["dataset_sha256"])
    verify_sha256(
        bge_path,
        clustering_manifest["files_sha256"]["cluster_assignments.parquet"],
    )
    verify_sha256(
        tfidf_path,
        tfidf_manifest["files_sha256"]["tfidf_assignments.parquet"],
    )

    metadata = pd.read_parquet(embedding_dir / "metadata.parquet")
    dataset = pd.read_parquet(dataset_path, columns=["openalex_id"])
    bge_assignments = pd.read_parquet(bge_path)
    tfidf_assignments = pd.read_parquet(tfidf_path)

    n_articles = int(clustering_manifest["dataset_articles"])

    validate_order(metadata, n_articles)
    validate_order(bge_assignments, n_articles)
    validate_order(tfidf_assignments, n_articles)

    reference_ids = validate_ids(metadata["openalex_id"], n_articles)
    validate_ids(dataset["openalex_id"], n_articles, reference_ids)
    validate_ids(bge_assignments["openalex_id"], n_articles, reference_ids)
    validate_ids(tfidf_assignments["openalex_id"], n_articles, reference_ids)

    return bge_assignments, tfidf_assignments


def load_seed_stability(
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Consolida os CSVs de estabilidade entre sementes do K-Means."""
    processed_dir = Path(processed_dir)
    clustering_dir = processed_dir / "clustering/v1"
    tfidf_dir = processed_dir / "tfidf_baseline/v1"

    clustering_manifest = _read_json(clustering_dir / "manifest.json")
    tfidf_manifest = _read_json(tfidf_dir / "manifest.json")

    bge_csv = clustering_dir / "kmeans_stability.csv"
    tfidf_csv = tfidf_dir / "tfidf_stability.csv"

    verify_sha256(
        bge_csv,
        clustering_manifest["files_sha256"]["kmeans_stability.csv"],
    )
    verify_sha256(
        tfidf_csv,
        tfidf_manifest["files_sha256"]["tfidf_stability.csv"],
    )

    bge_seeds = [int(s) for s in clustering_manifest["kmeans_stability_seeds"]]
    tfidf_seeds = [int(s) for s in tfidf_manifest["stability_seeds"]]
    tfidf_features = int(
        tfidf_manifest["primary_configuration"]["max_features"]
    )
    tfidf_vocab_tag = f"{tfidf_features // 1000}k"

    rows = []

    for _, row in pd.read_csv(bge_csv).iterrows():
        k = int(row["k"])
        rows.append(
            {
                "config_id": f"kmeans_k{k}",
                "representation": BGE_REPRESENTATION,
                "feature_space": f"{BGE_REPRESENTATION}:384d_l2",
                "algorithm": "kmeans",
                "k": k,
                "stability_type": "seed_initialization",
                "n_seeds": len(bge_seeds),
                "n_seed_pairs": comb(len(bge_seeds), 2),
                "seeds": ";".join(map(str, bge_seeds)),
                "ari_mean": float(row["ari_mean"]),
                "ari_std": float(row["ari_std"]),
                "ari_min": float(row["ari_min"]),
                "ari_max": float(row["ari_max"]),
                "source_artifact": "clustering/v1/kmeans_stability.csv",
            }
        )

    for _, row in pd.read_csv(tfidf_csv).iterrows():
        k = int(row["k"])
        rows.append(
            {
                "config_id": f"tfidf_{tfidf_vocab_tag}_k{k}",
                "representation": TFIDF_REPRESENTATION,
                "feature_space": f"{TFIDF_REPRESENTATION}:{tfidf_features}f_sparse",
                "algorithm": "kmeans",
                "k": k,
                "stability_type": "seed_initialization",
                "n_seeds": len(tfidf_seeds),
                "n_seed_pairs": comb(len(tfidf_seeds), 2),
                "seeds": ";".join(map(str, tfidf_seeds)),
                "ari_mean": float(row["ari_mean"]),
                "ari_std": float(row["ari_std"]),
                "ari_min": float(row["ari_min"]),
                "ari_max": float(row["ari_max"]),
                "source_artifact": "tfidf_baseline/v1/tfidf_stability.csv",
            }
        )

    table = pd.DataFrame(rows, columns=SEED_STABILITY_COLUMNS)

    for column in ("k", "n_seeds", "n_seed_pairs"):
        table[column] = table[column].astype("Int64")

    return table


def _validate_cross_alignment(
    bge_assignments: pd.DataFrame,
    tfidf_assignments: pd.DataFrame,
) -> None:
    if len(bge_assignments) != len(tfidf_assignments):
        raise ValueError("Quantidade de artigos incompatível")

    n_articles = len(bge_assignments)

    has_bge_index = "embedding_index" in bge_assignments.columns
    has_tfidf_index = "embedding_index" in tfidf_assignments.columns

    if has_bge_index != has_tfidf_index:
        raise ValueError("Coluna embedding_index ausente em um dos artefatos")

    if has_bge_index:
        validate_order(bge_assignments, n_articles)
        validate_order(tfidf_assignments, n_articles)

    has_bge_id = "openalex_id" in bge_assignments.columns
    has_tfidf_id = "openalex_id" in tfidf_assignments.columns

    if has_bge_id != has_tfidf_id:
        raise ValueError("Coluna openalex_id ausente em um dos artefatos")

    if has_bge_id:
        ref = validate_ids(bge_assignments["openalex_id"], n_articles)
        validate_ids(tfidf_assignments["openalex_id"], n_articles, ref)


def _collect_partitions(
    bge_assignments: pd.DataFrame,
    tfidf_assignments: pd.DataFrame | None = None,
) -> list[dict]:
    n_articles = len(bge_assignments)

    if "embedding_index" in bge_assignments.columns:
        validate_order(bge_assignments, n_articles)

    if "openalex_id" in bge_assignments.columns:
        validate_ids(bge_assignments["openalex_id"], n_articles)

    if tfidf_assignments is not None:
        _validate_cross_alignment(bge_assignments, tfidf_assignments)

    partitions = []

    for config_id in label_columns(bge_assignments):
        params = parse_bge_config(config_id)
        labels = bge_assignments[config_id].to_numpy()
        summary = summarize_clusters(labels)
        partitions.append(
            {
                "config_id": config_id,
                "representation": BGE_REPRESENTATION,
                "algorithm": params["algorithm"],
                "scope": (
                    "full"
                    if summary["noise_articles"] == 0
                    else "grouped_only"
                ),
                "labels": labels,
                "summary": summary,
                "signature": canonical_partition_signature(labels),
            }
        )

    if tfidf_assignments is not None:
        for config_id in label_columns(tfidf_assignments):
            params = parse_tfidf_config(config_id)
            labels = tfidf_assignments[config_id].to_numpy()
            summary = summarize_clusters(labels)

            if config_id in {p["config_id"] for p in partitions}:
                raise ValueError(f"Configuração duplicada: {config_id}")

            partitions.append(
                {
                    "config_id": config_id,
                    "representation": TFIDF_REPRESENTATION,
                    "algorithm": params["algorithm"],
                    "scope": (
                        "full"
                        if summary["noise_articles"] == 0
                        else "grouped_only"
                    ),
                    "labels": labels,
                    "summary": summary,
                    "signature": canonical_partition_signature(labels),
                }
            )

    return partitions


def find_identical_partitions(
    bge_assignments: pd.DataFrame,
    tfidf_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Identifica partições equivalentes sem remover nenhuma configuração."""
    partitions = _collect_partitions(bge_assignments, tfidf_assignments)
    groups: dict[str, list[str]] = {}
    group_ids: dict[str, int] = {}

    for item in partitions:
        sig = item["signature"]
        if sig not in groups:
            group_ids[sig] = len(group_ids) + 1
            groups[sig] = []
        groups[sig].append(item["config_id"])

    rows = []

    for item in partitions:
        sig = item["signature"]
        members = groups[sig]
        rows.append(
            {
                "config_id": item["config_id"],
                "representation": item["representation"],
                "algorithm": item["algorithm"],
                "scope": item["scope"],
                "n_clusters": item["summary"]["n_clusters"],
                "grouped_articles": item["summary"]["grouped_articles"],
                "partition_signature": sig,
                "identical_group_id": group_ids[sig],
                "identical_group_size": len(members),
                "is_duplicate_partition": len(members) > 1,
                "identical_configs": ";".join(members),
            }
        )

    table = pd.DataFrame(rows, columns=IDENTICAL_PARTITIONS_COLUMNS)

    for column in (
        "n_clusters",
        "grouped_articles",
        "identical_group_id",
        "identical_group_size",
    ):
        table[column] = table[column].astype("Int64")

    table["is_duplicate_partition"] = table["is_duplicate_partition"].astype(
        "boolean"
    )

    return table


def build_identical_partition_pairs(
    bge_assignments: pd.DataFrame,
    tfidf_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lista os pares de configurações com partições globalmente idênticas."""
    partitions = _collect_partitions(bge_assignments, tfidf_assignments)
    rows = []

    for idx_a, part_a in enumerate(partitions):
        for part_b in partitions[idx_a + 1 :]:
            if part_a["signature"] != part_b["signature"]:
                continue

            comparison = compare_partitions(part_a["labels"], part_b["labels"])
            summary = part_a["summary"]

            rows.append(
                {
                    "config_a": part_a["config_id"],
                    "config_b": part_b["config_id"],
                    "representation_a": part_a["representation"],
                    "representation_b": part_b["representation"],
                    "algorithm_a": part_a["algorithm"],
                    "algorithm_b": part_b["algorithm"],
                    "scope": part_a["scope"],
                    "n_articles": summary["n_articles"],
                    "grouped_articles": summary["grouped_articles"],
                    "noise_articles": summary["noise_articles"],
                    "coverage_pct": summary["coverage_pct"],
                    "n_clusters": summary["n_clusters"],
                    "exact_label_match": comparison["exact_label_match"],
                    "ari": comparison["ari"],
                    "partition_signature": part_a["signature"],
                }
            )

    table = pd.DataFrame(rows, columns=IDENTICAL_PAIR_COLUMNS)

    for column in (
        "n_articles",
        "grouped_articles",
        "noise_articles",
        "n_clusters",
    ):
        table[column] = table[column].astype("Int64")

    table["exact_label_match"] = table["exact_label_match"].astype("boolean")

    return table


def build_concordance_table(
    bge_assignments: pd.DataFrame,
    tfidf_assignments: pd.DataFrame | None = None,
    *,
    include_self: bool = False,
    min_intersection_size: int = DEFAULT_MIN_INTERSECTION_SIZE,
) -> pd.DataFrame:
    """Calcula a concordância por ARI entre os pares de partições."""
    partitions = _collect_partitions(bge_assignments, tfidf_assignments)
    rows = []

    for idx_a, part_a in enumerate(partitions):
        start_b = idx_a if include_self else idx_a + 1

        for part_b in partitions[start_b:]:
            result = compare_partitions(
                part_a["labels"],
                part_b["labels"],
                min_intersection_size=min_intersection_size,
            )

            rows.append(
                {
                    "config_a": part_a["config_id"],
                    "config_b": part_b["config_id"],
                    "representation_a": part_a["representation"],
                    "representation_b": part_b["representation"],
                    "algorithm_a": part_a["algorithm"],
                    "algorithm_b": part_b["algorithm"],
                    "scope_a": part_a["scope"],
                    "scope_b": part_b["scope"],
                    "comparison_type": classify_comparison(part_a, part_b),
                    **{
                        key: result[key]
                        for key in (
                            "comparison_scope",
                            "n_articles",
                            "n_grouped_a",
                            "n_grouped_b",
                            "n_intersection",
                            "intersection_coverage_pct",
                            "n_clusters_a_total",
                            "n_clusters_b_total",
                            "n_clusters_a_intersection",
                            "n_clusters_b_intersection",
                            "identical_partition",
                            "exact_label_match",
                            "valid",
                            "ari",
                            "ari_unavailable_reason",
                        )
                    },
                    "reasons": ";".join(result["reasons"]),
                    "warnings": ";".join(result["warnings"]),
                }
            )

    table = pd.DataFrame(rows, columns=CONCORDANCE_COLUMNS)

    for column in CONCORDANCE_INTEGER_COLUMNS:
        table[column] = table[column].astype("Int64")

    for column in CONCORDANCE_BOOLEAN_COLUMNS:
        table[column] = table[column].astype("boolean")

    return table


def build_full_ari_matrix(
    bge_assignments: pd.DataFrame,
    tfidf_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Matriz simétrica de ARI entre todas as partições completas (`full`)."""
    partitions = [
        item
        for item in _collect_partitions(bge_assignments, tfidf_assignments)
        if item["scope"] == "full"
    ]

    if not partitions:
        raise ValueError("Nenhuma partição completa encontrada")

    names = [item["config_id"] for item in partitions]
    matrix = np.full((len(partitions), len(partitions)), np.nan, dtype=float)

    for i, part_a in enumerate(partitions):
        self_result = compare_partitions(part_a["labels"], part_a["labels"])
        if self_result["valid"]:
            matrix[i, i] = self_result["ari"]

        for j in range(i + 1, len(partitions)):
            result = compare_partitions(
                part_a["labels"],
                partitions[j]["labels"],
            )
            if result["valid"]:
                matrix[i, j] = result["ari"]
                matrix[j, i] = result["ari"]

    return pd.DataFrame(matrix, index=names, columns=names)


def reconcile_stability_with_reference(
    seed_stability: pd.DataFrame,
    concordance: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Confronta ARI e estabilidade consolidados com os CSVs anteriores."""
    processed_dir = Path(processed_dir)
    records = []

    pair_lookup: dict[tuple[str, str], float] = {}
    for _, row in concordance.iterrows():
        if pd.notna(row["ari"]):
            val = float(row["ari"])
            pair_lookup[(row["config_a"], row["config_b"])] = val
            pair_lookup[(row["config_b"], row["config_a"])] = val

    agg_csv = "clustering/v1/agglomerative_metrics.csv"
    for _, row in pd.read_csv(processed_dir / agg_csv).iterrows():
        config_id = f"agglomerative_{row['linkage']}_k8"
        value = pair_lookup[(config_id, "kmeans_k8")]
        expected = float(row["ari_vs_kmeans"])
        records.append(
            {
                "config_id": config_id,
                "source": agg_csv,
                "field": "ari_vs_kmeans_k8",
                "value": value,
                "reference": expected,
                "abs_diff": abs(value - expected),
                "match": bool(
                    np.isclose(value, expected, **REFERENCE_TOLERANCE)
                ),
            }
        )

    stability_indexed = seed_stability.set_index("config_id")
    tfidf_csv = "tfidf_baseline/v1/tfidf_final_candidates.csv"
    for _, row in pd.read_csv(processed_dir / tfidf_csv).iterrows():
        config_id = f"tfidf_10k_k{int(row['k'])}"
        for field in ("ari_mean", "ari_min"):
            value = float(stability_indexed.at[config_id, field])
            expected = float(row[field])
            records.append(
                {
                    "config_id": config_id,
                    "source": tfidf_csv,
                    "field": field,
                    "value": value,
                    "reference": expected,
                    "abs_diff": abs(value - expected),
                    "match": bool(
                        np.isclose(value, expected, **REFERENCE_TOLERANCE)
                    ),
                }
            )

    return pd.DataFrame(records)


def load_concordance_analysis(
    processed_dir: Path = PROCESSED_DIR,
    *,
    min_intersection_size: int = DEFAULT_MIN_INTERSECTION_SIZE,
    verify_reference: bool = True,
) -> dict:
    """Executa a análise completa da Etapa 3A sem reexecutar algoritmos."""
    bge_assignments, tfidf_assignments = load_aligned_assignments(
        processed_dir
    )
    seed_stability = load_seed_stability(processed_dir)
    identical = find_identical_partitions(bge_assignments, tfidf_assignments)
    identical_pairs = build_identical_partition_pairs(
        bge_assignments,
        tfidf_assignments,
    )
    concordance = build_concordance_table(
        bge_assignments,
        tfidf_assignments,
        min_intersection_size=min_intersection_size,
    )
    full_matrix = build_full_ari_matrix(bge_assignments, tfidf_assignments)
    reconciliation = (
        reconcile_stability_with_reference(
            seed_stability,
            concordance,
            processed_dir,
        )
        if verify_reference
        else pd.DataFrame()
    )

    return {
        "seed_stability": seed_stability,
        "identical_partitions": identical,
        "identical_pairs": identical_pairs,
        "concordance": concordance,
        "full_ari_matrix": full_matrix,
        "reconciliation": reconciliation,
    }


def main() -> None:
    started = time.perf_counter()
    analysis = load_concordance_analysis()
    elapsed = time.perf_counter() - started

    concordance = analysis["concordance"]
    duplicates = analysis["identical_partitions"]
    duplicates = duplicates[duplicates["is_duplicate_partition"]]
    reconciliation = analysis["reconciliation"]

    print(
        f"Estabilidade entre sementes: {len(analysis['seed_stability'])} "
        "configurações"
    )
    print(f"Matriz ARI (full): {analysis['full_ari_matrix'].shape}")
    print(
        f"Pares avaliados: {len(concordance)} "
        f"(válidos: {int(concordance['valid'].sum())}, "
        f"inválidos: {int((~concordance['valid']).sum())})"
    )
    print(f"Configurações em grupos idênticos: {len(duplicates)}")
    print(
        f"Conferência com CSVs: {int(reconciliation['match'].sum())}"
        f"/{len(reconciliation)}"
    )
    print(f"Tempo: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
