"""Exportação determinística e atômica dos artefatos consolidados da Issue #08."""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd

from research_atlas.evaluation.metrics import (
    CALINSKI_HARABASZ_GEOMETRY,
    DAVIES_BOULDIN_GEOMETRY,
    DEFAULT_SEED,
    LOW_COVERAGE_PCT,
    MAX_CLUSTER_SHARE,
    SILHOUETTE_FULL_FALLBACK_MAX,
    SILHOUETTE_METRIC,
    SILHOUETTE_SAMPLE_SIZE,
)
from research_atlas.evaluation.pipeline import (
    BGE_REPRESENTATION,
    PROCESSED_DIR,
    TFIDF_REPRESENTATION,
    TFIDF_VOCABULARY_SIZES,
    _read_json,
    build_evaluation_table,
    reconcile_with_reference,
    sha256_file,
    verify_sha256,
)
from research_atlas.evaluation.stability import (
    DEFAULT_MIN_INTERSECTION_SIZE,
    load_concordance_analysis,
)

ARTIFACT_VERSION = "1"
ISSUE_ID = "#08"

DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "evaluation/v1"

EXPORTED_CSV_FILES = (
    "evaluation_metrics.csv",
    "seed_stability.csv",
    "partition_concordance.csv",
    "full_ari_matrix.csv",
    "identical_partitions.csv",
)

CORE_INPUT_FILES = (
    "dataset_selected.parquet",
    "embeddings/bge-small-en-v1.5-v1/embeddings.npy",
    "embeddings/bge-small-en-v1.5-v1/metadata.parquet",
    "embeddings/bge-small-en-v1.5-v1/manifest.json",
    "clustering/v1/cluster_assignments.parquet",
    "clustering/v1/kmeans_stability.csv",
    "clustering/v1/manifest.json",
    "tfidf_baseline/v1/tfidf_assignments.parquet",
    "tfidf_baseline/v1/tfidf_stability.csv",
    "tfidf_baseline/v1/manifest.json",
)

REFERENCE_INPUT_FILES = (
    "clustering/v1/kmeans_metrics.csv",
    "clustering/v1/agglomerative_metrics.csv",
    "clustering/v1/hdbscan_eom_metrics.csv",
    "clustering/v1/hdbscan_leaf_metrics.csv",
    "tfidf_baseline/v1/kmeans_k_sensitivity.csv",
    "tfidf_baseline/v1/tfidf_final_candidates.csv",
)

PROTECTED_SUBDIRS = (
    "embeddings",
    "clustering",
    "tfidf_baseline",
    "audit",
)

LIMITATIONS = [
    (
        "A estabilidade por reamostragem não foi executada nesta entrega; "
        "a análise de estabilidade restringe-se à variação entre sementes "
        "de inicialização do K-Means."
    ),
    (
        "Matrizes TF-IDF permanecem esparsas e avaliam apenas Silhouette "
        "cosseno; Davies-Bouldin e Calinski-Harabasz permanecem ausentes "
        "com justificativa explícita (matriz_esparsa_nao_densificada)."
    ),
    (
        "Métricas de escopo grouped_only (HDBSCAN) e de espaços vetoriais "
        "distintos (BGE vs TF-IDF) não são diretamente equivalentes a "
        "partições completas no mesmo espaço."
    ),
    (
        "O ARI envolvendo partições com ruído (HDBSCAN) é calculado "
        "exclusivamente na interseção dos artigos agrupados e não "
        "representa concordância global da base."
    ),
    (
        "LOW_COVERAGE_PCT (50.0) e SILHOUETTE_FULL_FALLBACK_MAX (10000) "
        "são convenções operacionais deste pipeline, não limiares "
        "científicos universais."
    ),
]


def _validate_output_dir(processed_dir: Path, output_dir: Path) -> None:
    resolved_processed = processed_dir.resolve()
    resolved_output = output_dir.resolve()

    if resolved_output == resolved_processed or (
        resolved_processed.is_relative_to(resolved_output)
    ):
        raise ValueError("Diretório de saída inválido")

    for subdir in PROTECTED_SUBDIRS:
        protected = (resolved_processed / subdir).resolve()
        if resolved_output == protected or resolved_output.is_relative_to(
            protected
        ):
            raise ValueError(f"Diretório de saída protegido: {subdir}")


def _hash_inputs(
    processed_dir: Path,
    *,
    include_reference: bool = True,
) -> dict[str, str]:
    relative_paths = (
        CORE_INPUT_FILES + REFERENCE_INPUT_FILES
        if include_reference
        else CORE_INPUT_FILES
    )
    hashes: dict[str, str] = {}

    for rel_path in relative_paths:
        path = processed_dir / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Insumo ausente: {rel_path}")
        hashes[rel_path] = sha256_file(path)

    return hashes


def _compute_run_id(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
    index_label: str | None = None,
) -> None:
    frame.to_csv(
        path,
        index=index,
        index_label=index_label,
        lineterminator="\n",
    )


def _write_staging_files(
    staging_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> dict[str, str]:
    _write_csv(
        tables["evaluation_metrics"],
        staging_dir / "evaluation_metrics.csv",
    )
    _write_csv(
        tables["seed_stability"],
        staging_dir / "seed_stability.csv",
    )
    _write_csv(
        tables["partition_concordance"],
        staging_dir / "partition_concordance.csv",
    )
    _write_csv(
        tables["full_ari_matrix"],
        staging_dir / "full_ari_matrix.csv",
        index=True,
        index_label="config_id",
    )
    _write_csv(
        tables["identical_partitions"],
        staging_dir / "identical_partitions.csv",
    )

    return {
        filename: sha256_file(staging_dir / filename)
        for filename in EXPORTED_CSV_FILES
    }


def _build_manifest(
    processed_dir: Path,
    tables: dict[str, pd.DataFrame],
    inputs_sha256: dict[str, str],
    files_sha256: dict[str, str],
    *,
    sample_size: int,
    seed: int,
    min_intersection_size: int,
    metric_reconciliation: pd.DataFrame,
    stability_reconciliation: pd.DataFrame,
) -> dict:
    embedding_manifest = _read_json(
        processed_dir / "embeddings/bge-small-en-v1.5-v1/manifest.json"
    )
    clustering_manifest = _read_json(
        processed_dir / "clustering/v1/manifest.json"
    )
    tfidf_manifest = _read_json(
        processed_dir / "tfidf_baseline/v1/manifest.json"
    )

    evaluation_metrics = tables["evaluation_metrics"]
    seed_stability = tables["seed_stability"]
    concordance = tables["partition_concordance"]
    full_matrix = tables["full_ari_matrix"]
    identical_pairs = tables["identical_partitions"]

    parameters = {
        "random_seed": int(seed),
        "silhouette_metric": SILHOUETTE_METRIC,
        "silhouette_sample_size": int(sample_size),
        "silhouette_full_fallback_max": SILHOUETTE_FULL_FALLBACK_MAX,
        "davies_bouldin_geometry": DAVIES_BOULDIN_GEOMETRY,
        "calinski_harabasz_geometry": CALINSKI_HARABASZ_GEOMETRY,
        "tfidf_sparse_only_cosine_silhouette": True,
        "low_coverage_pct_operational": LOW_COVERAGE_PCT,
        "max_cluster_share": MAX_CLUSTER_SHARE,
        "min_intersection_size": int(min_intersection_size),
        "resampling_stability_executed": False,
        "hdbscan_ari_protocol": "grouped_intersection_only",
        "ari_seed_vs_partition_distinction": (
            "seed_stability.csv registra estabilidade entre sementes "
            "(seed_initialization); partition_concordance.csv registra "
            "concordância estrutural entre partições distintas"
        ),
    }

    provenance = {
        "modules": [
            "research_atlas.evaluation.metrics",
            "research_atlas.evaluation.pipeline",
            "research_atlas.evaluation.stability",
            "research_atlas.evaluation.export",
        ],
        "upstream_artifacts": {
            "dataset": "dataset_selected.parquet",
            "embeddings": "embeddings/bge-small-en-v1.5-v1",
            "bge_clustering": "clustering/v1",
            "tfidf_baseline": "tfidf_baseline/v1",
        },
        "upstream_created_at_utc": {
            "embeddings": embedding_manifest.get("created_at_utc"),
            "bge_clustering": clustering_manifest.get("created_at_utc"),
            "tfidf_baseline": tfidf_manifest.get("created_at_utc"),
        },
        "tfidf_vocabulary_sizes": dict(TFIDF_VOCABULARY_SIZES),
        "tfidf_primary_configuration": tfidf_manifest[
            "primary_configuration"
        ],
        "kmeans_stability_seeds": clustering_manifest[
            "kmeans_stability_seeds"
        ],
    }

    summary = {
        "evaluation_configurations": len(evaluation_metrics),
        "bge_configurations": int(
            (evaluation_metrics["representation"] == BGE_REPRESENTATION).sum()
        ),
        "tfidf_configurations": int(
            (
                evaluation_metrics["representation"] == TFIDF_REPRESENTATION
            ).sum()
        ),
        "full_scope_configurations": int(
            (evaluation_metrics["scope"] == "full").sum()
        ),
        "grouped_only_configurations": int(
            (evaluation_metrics["scope"] == "grouped_only").sum()
        ),
        "seed_stability_configurations": len(seed_stability),
        "partition_pairs_total": len(concordance),
        "partition_pairs_valid": int(concordance["valid"].sum()),
        "partition_pairs_invalid": int((~concordance["valid"]).sum()),
        "full_ari_matrix_size": len(full_matrix),
        "identical_partition_pairs": len(identical_pairs),
        "reference_metrics_reconciled": len(metric_reconciliation),
        "reference_stability_reconciled": len(stability_reconciliation),
    }

    run_id = _compute_run_id(
        {
            "version": ARTIFACT_VERSION,
            "issue": ISSUE_ID,
            "parameters": parameters,
            "inputs_sha256": inputs_sha256,
            "files_sha256": files_sha256,
        }
    )

    return {
        "version": ARTIFACT_VERSION,
        "issue": ISSUE_ID,
        "run_id": run_id,
        "run_id_method": "sha256_canonical_payload",
        "dataset_articles": int(evaluation_metrics["n_articles"].iloc[0]),
        "embedding_model": embedding_manifest["model"],
        "representations": [BGE_REPRESENTATION, TFIDF_REPRESENTATION],
        "parameters": parameters,
        "provenance": provenance,
        "summary": summary,
        "inputs_sha256": inputs_sha256,
        "files_sha256": files_sha256,
        "limitations": list(LIMITATIONS),
    }


def _atomic_swap_directory(staging_dir: Path, output_dir: Path) -> None:
    backup_dir: Path | None = None

    if output_dir.exists():
        backup_dir = (
            output_dir.parent / f".{output_dir.name}_bak_{uuid4().hex}"
        )
        output_dir.replace(backup_dir)

    try:
        staging_dir.replace(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            backup_dir.replace(output_dir)
        raise
    else:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)


def export_evaluation_artifacts(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path | None = None,
    *,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    min_intersection_size: int = DEFAULT_MIN_INTERSECTION_SIZE,
    verify_reference: bool = True,
) -> dict:
    """Consolida e exporta de forma atômica os artefatos da Issue #08."""
    processed_dir = Path(processed_dir)
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else processed_dir / "evaluation/v1"
    )

    _validate_output_dir(processed_dir, target_dir)
    inputs_before = _hash_inputs(
        processed_dir,
        include_reference=verify_reference,
    )

    evaluation_metrics = build_evaluation_table(
        processed_dir,
        sample_size=sample_size,
        seed=seed,
    )
    metric_reconciliation = (
        reconcile_with_reference(evaluation_metrics, processed_dir)
        if verify_reference
        else pd.DataFrame()
    )
    if verify_reference and not bool(metric_reconciliation["match"].all()):
        raise ValueError("Divergência na reconciliação de métricas")

    concordance_bundle = load_concordance_analysis(
        processed_dir,
        min_intersection_size=min_intersection_size,
        verify_reference=verify_reference,
    )
    stability_reconciliation = concordance_bundle["reconciliation"]
    if verify_reference and not bool(stability_reconciliation["match"].all()):
        raise ValueError("Divergência na reconciliação de estabilidade/ARI")

    tables = {
        "evaluation_metrics": evaluation_metrics,
        "seed_stability": concordance_bundle["seed_stability"],
        "partition_concordance": concordance_bundle["concordance"],
        "full_ari_matrix": concordance_bundle["full_ari_matrix"],
        "identical_partitions": concordance_bundle["identical_pairs"],
    }

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}_tmp_",
            dir=target_dir.parent,
        )
    )

    try:
        files_sha256 = _write_staging_files(staging_dir, tables)
        manifest = _build_manifest(
            processed_dir,
            tables,
            inputs_before,
            files_sha256,
            sample_size=sample_size,
            seed=seed,
            min_intersection_size=min_intersection_size,
            metric_reconciliation=metric_reconciliation,
            stability_reconciliation=stability_reconciliation,
        )
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        for filename, expected_hash in files_sha256.items():
            verify_sha256(staging_dir / filename, expected_hash)

        _atomic_swap_directory(staging_dir, target_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    inputs_after = _hash_inputs(
        processed_dir,
        include_reference=verify_reference,
    )
    if inputs_after != inputs_before:
        raise RuntimeError("Insumos modificados durante a exportação")

    return {
        "output_dir": target_dir,
        "manifest": manifest,
        "tables": tables,
        "metric_reconciliation": metric_reconciliation,
        "stability_reconciliation": stability_reconciliation,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta os artefatos consolidados da Issue #08."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.perf_counter()
    result = export_evaluation_artifacts(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
    elapsed = time.perf_counter() - started

    manifest = result["manifest"]
    summary = manifest["summary"]
    print(f"Diretório exportado: {result['output_dir']}")
    print(f"Run ID: {manifest['run_id']}")
    print(
        f"evaluation_metrics.csv: {summary['evaluation_configurations']} "
        "configurações"
    )
    print(
        f"seed_stability.csv: {summary['seed_stability_configurations']} "
        "configurações"
    )
    print(
        f"partition_concordance.csv: {summary['partition_pairs_total']} "
        f"pares (válidos: {summary['partition_pairs_valid']}, "
        f"inválidos: {summary['partition_pairs_invalid']})"
    )
    print(
        f"full_ari_matrix.csv: {summary['full_ari_matrix_size']}x"
        f"{summary['full_ari_matrix_size']}"
    )
    print(
        f"identical_partitions.csv: {summary['identical_partition_pairs']} "
        "pares"
    )
    print(f"Tempo: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
