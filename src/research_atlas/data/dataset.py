
"""Construção do dataset científico ampliado do ResearchAtlas."""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research_atlas.data.openalex import OpenAlexClient
from research_atlas.data.pilot import (
    AREAS,
    NESTED_FIELDS,
    SEARCH_TERMS,
    evaluate_text_quality,
    generate_quality_report,
)
from research_atlas.data.preprocessing import (
    deduplicate_articles,
    normalize_article,
    validate_article,
)


YEARS = tuple(range(2020, 2026))

OUTPUT_DIR = Path("data/processed")
CACHE_DIR = Path("data/raw/openalex_cache")

DEFAULT_TARGET_PER_AREA = 1000
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 42


def stable_seed(
    base_seed: int,
    area: str,
    year: int,
    term: str,
    round_number: int,
) -> int:
    """Gera uma semente determinística para cada consulta."""
    value = f"{base_seed}|{area}|{year}|{term}|{round_number}"

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return int.from_bytes(digest[:4], "big")


def year_quotas(target_per_area: int) -> dict[int, int]:
    """Distribui a meta de artigos entre os seis anos."""
    if target_per_area < 1:
        raise ValueError("A meta por área deve ser positiva")

    base, remainder = divmod(
        target_per_area,
        len(YEARS),
    )

    return {
        year: base + (index < remainder)
        for index, year in enumerate(YEARS)
    }


def cached_sample(
    client: OpenAlexClient,
    *,
    area: str,
    subfields: list[str],
    year: int,
    term: str,
    sample_size: int,
    seed: int,
    cache_dir: Path = CACHE_DIR,
) -> list[dict[str, Any]]:
    """Consulta a OpenAlex ou recupera uma resposta salva.

    A chave do cache considera todos os parâmetros da consulta.
    Os arquivos são gravados atomicamente para evitar resultados
    parciais quando uma execução é interrompida.
    """
    parameters = {
        "area": area,
        "subfields": subfields,
        "year": year,
        "term": term,
        "sample_size": sample_size,
        "seed": seed,
    }

    cache_key = hashlib.sha256(
        json.dumps(
            parameters,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = cache_dir / f"{cache_key}.json"

    # Reutiliza uma consulta concluída anteriormente.
    if cache_path.exists():
        payload = json.loads(
            cache_path.read_text(encoding="utf-8")
        )

        if payload.get("parameters") != parameters:
            raise ValueError(
                f"Parâmetros incompatíveis no cache: {cache_path}"
            )

        results = payload.get("results")

        if not isinstance(results, list):
            raise ValueError(
                f"Resposta inválida no cache: {cache_path}"
            )

        return results

    # Consulta a API quando não há resultado salvo.
    results = client.sample_works(
        subfields=subfields,
        search_term=term,
        year=year,
        sample_size=sample_size,
        seed=seed,
    )

    payload = {
        "parameters": parameters,
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results,
    }

    # Evita deixar um JSON incompleto caso a escrita falhe.
    temporary_path = cache_path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            cache_path,
        )

    finally:
        temporary_path.unlink(missing_ok=True)

    return results


def serialize_nested(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepara campos estruturados para exportação em Parquet."""
    result = dataframe.copy()

    for field in NESTED_FIELDS:
        if field in result.columns:
            result[field] = result[field].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
            )

    return result


def collect_candidates(
    client: OpenAlexClient,
    sample_size: int,
    seed: int,
    rounds: int,
) -> tuple[list[dict], list[dict]]:
    """Coleta candidatos por área, ano e termo utilizando cache."""
    candidates = []
    query_log = []

    for area, subfields in AREAS.items():
        print(
            f"\n=== {area.upper()} ===",
            flush=True,
        )

        for year in YEARS:
            year_count = 0

            for term in SEARCH_TERMS:
                for round_number in range(rounds):
                    query_seed = stable_seed(
                        seed,
                        area,
                        year,
                        term,
                        round_number,
                    )

                    works = cached_sample(
                        client,
                        area=area,
                        subfields=subfields,
                        year=year,
                        term=term,
                        sample_size=sample_size,
                        seed=query_seed,
                    )

                    for work in works:
                        article = normalize_article(
                            work,
                            collection_area=area,
                            retrieval_term=term,
                        )

                        candidates.append(article)

                    query_log.append(
                        {
                            "area": area,
                            "year": year,
                            "term": term,
                            "round": round_number + 1,
                            "seed": query_seed,
                            "requested": sample_size,
                            "returned": len(works),
                        }
                    )

                    year_count += len(works)

            print(
                f"{year}: {year_count} registros brutos",
                flush=True,
            )

    return candidates, query_log


def select_balanced_articles(
    articles: list[dict],
    target_per_area: int = DEFAULT_TARGET_PER_AREA,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Seleciona artigos únicos respeitando cotas por área e ano.

    Cada artigo recebe uma área principal para contabilização.
    As demais áreas de recuperação continuam preservadas.
    """
    if target_per_area < 1:
        raise ValueError("A meta por área deve ser positiva")

    quotas = year_quotas(target_per_area)
    buckets = defaultdict(list)

    for article in articles:
        year = article.get("publication_year")

        if year not in YEARS:
            continue

        for area in article.get("collection_areas", []):
            if area in AREAS:
                buckets[(area, year)].append(article)

    selected = []
    used_ids = set()
    strata_report = []

    for area in AREAS:
        for year in YEARS:
            quota = quotas[year]
            pool = buckets[(area, year)].copy()

            rng = random.Random(
                stable_seed(
                    seed,
                    area,
                    year,
                    "selection",
                    0,
                )
            )

            rng.shuffle(pool)

            count = 0

            for article in pool:
                article_id = article["openalex_id"]

                # Evita contabilizar o mesmo artigo duas vezes.
                if article_id in used_ids:
                    continue

                selected_article = {
                    **article,
                    "sampling_area": area,
                }

                selected.append(selected_article)
                used_ids.add(article_id)
                count += 1

                if count == quota:
                    break

            strata_report.append(
                {
                    "area": area,
                    "year": year,
                    "target": quota,
                    "selected": count,
                    "shortfall": quota - count,
                    "available_candidates": len(pool),
                }
            )

    return selected, strata_report


def build_dataset(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    target_per_area: int = DEFAULT_TARGET_PER_AREA,
    seed: int = DEFAULT_SEED,
    rounds: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Executa a coleta, validação e seleção estratificada."""
    if not 1 <= sample_size <= 100:
        raise ValueError(
            "sample_size deve estar entre 1 e 100"
        )

    if target_per_area < 1:
        raise ValueError(
            "target_per_area deve ser positivo"
        )

    if rounds < 1:
        raise ValueError(
            "rounds deve ser positivo"
        )

    with OpenAlexClient() as client:
        raw_articles, query_log = collect_candidates(
            client=client,
            sample_size=sample_size,
            seed=seed,
            rounds=rounds,
        )

    # Identificadores inválidos são contabilizados antes
    # da deduplicação.
    invalid_ids = sum(
        article["openalex_id"] is None
        for article in raw_articles
    )

    identified = [
        article
        for article in raw_articles
        if article["openalex_id"] is not None
    ]

    # Consolida ocorrências repetidas, preservando áreas e termos.
    unique_articles = deduplicate_articles(
        identified
    )

    valid_articles = []
    exclusion_counts = Counter()

    for article in unique_articles:
        valid, reasons = validate_article(article)

        if valid:
            valid_articles.append(article)
        else:
            exclusion_counts.update(reasons)

    # Candidatos válidos, antes da seleção por estrato.
    candidates_df = evaluate_text_quality(
        pd.DataFrame(valid_articles)
    )

    # Amostragem balanceada por área e ano.
    selected_articles, strata_report = select_balanced_articles(
        valid_articles,
        target_per_area=target_per_area,
        seed=seed,
    )

    selected_df = evaluate_text_quality(
        pd.DataFrame(selected_articles)
    )

    per_area = {
        area: sum(
            article["sampling_area"] == area
            for article in selected_articles
        )
        for area in AREAS
    }

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": "docs/dataset.md",
        "sampling_method": (
            "openalex_seeded_random_stratified"
        ),
        "base_seed": seed,
        "sample_per_query": sample_size,
        "rounds_per_query": rounds,
        "target_per_area": target_per_area,
        "target_total": (
            target_per_area * len(AREAS)
        ),
        "raw_records": len(raw_articles),
        "invalid_identifiers": invalid_ids,
        "unique_candidates": len(unique_articles),
        "valid_candidates": len(valid_articles),
        "duplicate_occurrences": (
            len(identified) - len(unique_articles)
        ),
        "exclusion_reasons": dict(exclusion_counts),
        "selected_total": len(selected_articles),
        "selected_unique_ids": len(
            {
                article["openalex_id"]
                for article in selected_articles
            }
        ),
        "selected_per_area": per_area,
        "multi_area_candidates": sum(
            len(article["collection_areas"]) > 1
            for article in valid_articles
        ),
        "multi_area_selected": sum(
            len(article["collection_areas"]) > 1
            for article in selected_articles
        ),
        "text_quality": generate_quality_report(
            selected_df
        ),
        "strata": strata_report,
        "queries": query_log,
        "limitations": [
            (
                "Randomização condicionada aos filtros "
                "e à cobertura OpenAlex"
            ),
            (
                "Amostra balanceada não representa a "
                "distribuição natural da literatura"
            ),
            (
                "Relevância científica ainda depende "
                "de auditoria humana"
            ),
            (
                "Alertas textuais não implicam "
                "exclusão automática"
            ),
        ],
    }

    report["shortfall_total"] = sum(
        item["shortfall"]
        for item in strata_report
    )

    return candidates_df, selected_df, report


def save_dataset(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    report: dict,
) -> None:
    """Exporta os candidatos, os selecionados e o relatório."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialize_nested(candidates).to_parquet(
        OUTPUT_DIR / "dataset_candidates.parquet",
        index=False,
    )

    serialize_nested(selected).to_parquet(
        OUTPUT_DIR / "dataset_selected.parquet",
        index=False,
    )

    report_path = OUTPUT_DIR / "dataset_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Executa o pipeline a partir dos argumentos do terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Constrói o dataset ampliado do ResearchAtlas."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--target-per-area",
        type=int,
        default=DEFAULT_TARGET_PER_AREA,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    candidates, selected, report = build_dataset(
        sample_size=args.sample_size,
        target_per_area=args.target_per_area,
        seed=args.seed,
        rounds=args.rounds,
    )

    save_dataset(
        candidates,
        selected,
        report,
    )

    print("\n=== RESULTADO DA COLETA ===")
    print(f"Candidatos válidos: {len(candidates)}")
    print(f"Artigos selecionados: {len(selected)}")
    print(f"Meta: {report['target_total']}")
    print(f"Déficit: {report['shortfall_total']}")

    print("\n=== SELEÇÃO POR ÁREA ===")

    for area, count in report["selected_per_area"].items():
        print(f"{area}: {count}")

    print("\n=== QUALIDADE TEXTUAL ===")
    print(
        json.dumps(
            report["text_quality"],
            indent=2,
            ensure_ascii=False,
        )
    )

    if report["shortfall_total"]:
        print(
            "\nATENÇÃO: alguns estratos não atingiram suas cotas. "
            "Consulte dataset_report.json antes de utilizar "
            "o conjunto nos experimentos."
        )

    print(
        "\nArquivos gerados em:",
        OUTPUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
