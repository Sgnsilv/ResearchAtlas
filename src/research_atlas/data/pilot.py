
"""Coleta piloto e análise de qualidade dos artigos do ResearchAtlas."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from research_atlas.data.openalex import OpenAlexClient
from research_atlas.data.preprocessing import (
    assess_abstract_quality,
    deduplicate_articles,
    normalize_article,
    validate_article,
)


AREAS = {
    "Medicina": ["2718", "2741"],
    "Astronomia": ["3103"],
    "Biologia": ["1311", "1315"],
    "Clima": ["1902", "2306"],
    "Engenharia": ["2207", "2210"],
}

SEARCH_TERMS = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural networks",
]

OUTPUT_DIR = Path("data/processed")
RESULTS_PER_QUERY = 20

NESTED_FIELDS = (
    "topics",
    "referenced_works",
    "collection_areas",
    "retrieval_terms",
    "metadata_conflicts",
    "quality_flags",
)


def evaluate_text_quality(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adiciona indicadores de qualidade textual aos artigos."""

    dataframe = dataframe.copy()

    if dataframe.empty:
        dataframe["abstract_word_count"] = pd.Series(dtype="int64")
        dataframe["quality_flags"] = pd.Series(dtype="object")
        dataframe["requires_review"] = pd.Series(dtype="bool")
        return dataframe

    quality = dataframe["abstract"].apply(assess_abstract_quality)

    dataframe["abstract_word_count"] = quality.apply(
        lambda result: result["word_count"]
    )

    dataframe["quality_flags"] = quality.apply(
        lambda result: result["quality_flags"]
    )

    dataframe["requires_review"] = quality.apply(
        lambda result: result["requires_review"]
    )

    return dataframe


def generate_quality_report(dataframe: pd.DataFrame) -> dict:
    """Calcula estatísticas sobre a qualidade dos resumos."""

    if dataframe.empty:
        return {
            "requires_review": 0,
            "short_abstracts": 0,
            "long_abstracts": 0,
            "suspicious_content": 0,
        }

    flags = dataframe["quality_flags"]

    return {
        "requires_review": int(
            dataframe["requires_review"].sum()
        ),
        "short_abstracts": int(
            flags.apply(
                lambda values: "short_abstract" in values
            ).sum()
        ),
        "long_abstracts": int(
            flags.apply(
                lambda values: "long_abstract" in values
            ).sum()
        ),
        "suspicious_content": int(
            flags.apply(
                lambda values: "suspicious_content" in values
            ).sum()
        ),
    }


def collect_pilot() -> tuple[pd.DataFrame, dict]:
    """Coleta, normaliza, deduplica e valida uma amostra exploratória."""

    raw_articles = []
    collection_counts = {}

    with OpenAlexClient() as client:
        for area, subfields in AREAS.items():
            print(f"\n=== {area.upper()} ===")

            for term in SEARCH_TERMS:
                works = list(
                    client.iter_works(
                        subfields=subfields,
                        search_term=term,
                        max_results=RESULTS_PER_QUERY,
                    )
                )

                collection_counts[f"{area} | {term}"] = len(works)

                for work in works:
                    article = normalize_article(
                        work,
                        collection_area=area,
                        retrieval_term=term,
                    )
                    raw_articles.append(article)

                print(f"{term}: {len(works)} registros")

    # Remove duplicatas e preserva áreas e termos de recuperação.
    unique_articles = deduplicate_articles(raw_articles)

    valid_articles = []
    exclusion_counts = Counter()

    # Validação estrutural dos registros.
    for article in unique_articles:
        valid, reasons = validate_article(article)

        if valid:
            valid_articles.append(article)
        else:
            exclusion_counts.update(reasons)

    dataframe = pd.DataFrame(valid_articles)

    # Avaliação textual sem exclusão automática.
    dataframe = evaluate_text_quality(dataframe)

    # Relatório da coleta.
    report = {
        "raw_records": len(raw_articles),
        "unique_records": len(unique_articles),
        "duplicate_occurrences": (
            len(raw_articles) - len(unique_articles)
        ),
        "valid_records": len(valid_articles),
        "invalid_records": (
            len(unique_articles) - len(valid_articles)
        ),
        "collection_counts": collection_counts,
        "exclusion_reasons": dict(exclusion_counts),
        "text_quality": generate_quality_report(dataframe),
        "sampling_method": "first_results_by_relevance",
        "records_per_query": RESULTS_PER_QUERY,
    }

    return dataframe, report


def save_results(
    dataframe: pd.DataFrame,
    report: dict,
) -> None:
    """Exporta artigos normalizados e relatório de qualidade."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_dataframe = dataframe.copy()

    # Serializa campos aninhados para armazenamento em Parquet.
    for field in NESTED_FIELDS:
        if field in export_dataframe.columns:
            export_dataframe[field] = export_dataframe[field].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
            )

    parquet_path = OUTPUT_DIR / "pilot_articles.parquet"
    report_path = OUTPUT_DIR / "pilot_report.json"

    export_dataframe.to_parquet(
        parquet_path,
        index=False,
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== ARQUIVOS GERADOS ===")
    print(f"Dataset: {parquet_path}")
    print(f"Relatório: {report_path}")


def main() -> None:
    """Executa o pipeline completo de coleta piloto."""

    dataframe, report = collect_pilot()

    save_results(dataframe, report)

    print("\n=== RELATÓRIO FINAL ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
