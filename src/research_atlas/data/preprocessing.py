
"""Normalização, validação e deduplicação de artigos da OpenAlex."""

import re
from typing import Any


def normalize_openalex_id(value: str | None) -> str | None:
    """Padroniza identificadores de trabalhos da OpenAlex."""
    if not isinstance(value, str):
        return None

    identifier = value.strip().rstrip("/").split("/")[-1]

    if not re.fullmatch(r"W[1-9]\d*", identifier):
        return None

    return f"https://openalex.org/{identifier}"


def reconstruct_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    """Reconstrói um resumo a partir do índice invertido da OpenAlex."""
    if not inverted_index:
        return ""

    words = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]

    words.sort(key=lambda item: item[0])

    return " ".join(word for _, word in words)


def normalize_text(text: str | None) -> str:
    """Remove espaços redundantes sem alterar o conteúdo científico."""
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def normalize_article(
    work: dict[str, Any],
    collection_area: str,
    retrieval_term: str,
) -> dict[str, Any]:
    """Converte um registro bruto da OpenAlex em formato padronizado."""
    abstract = reconstruct_abstract(
        work.get("abstract_inverted_index")
    )

    primary_topic = work.get("primary_topic") or {}

    return {
        "openalex_id": normalize_openalex_id(work.get("id")),
        "doi": work.get("doi"),
        "title": normalize_text(work.get("title")),
        "abstract": normalize_text(abstract),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "language": work.get("language"),
        "publication_type": work.get("type"),
        "is_retracted": work.get("is_retracted", False),
        "primary_topic_id": primary_topic.get("id"),
        "primary_topic_name": primary_topic.get("display_name"),
        "topics": work.get("topics") or [],
        "referenced_works": work.get("referenced_works") or [],
        "cited_by_count": work.get("cited_by_count"),
        "collection_areas": [collection_area],
        "retrieval_terms": [retrieval_term],
        "metadata_conflicts": [],
    }


def validate_article(
    article: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Verifica os critérios estruturais de inclusão do protocolo."""
    reasons = []

    if not normalize_openalex_id(article.get("openalex_id")):
        reasons.append("Identificador OpenAlex ausente ou inválido")

    if not article.get("title"):
        reasons.append("Título ausente")

    if not article.get("abstract"):
        reasons.append("Resumo ausente")

    year = article.get("publication_year")

    if (
        not isinstance(year, int)
        or isinstance(year, bool)
        or not 2020 <= year <= 2025
    ):
        reasons.append("Ano fora do período definido")

    if article.get("language") != "en":
        reasons.append("Idioma diferente de inglês")

    if article.get("publication_type") != "article":
        reasons.append("Tipo de publicação inválido")

    if article.get("is_retracted") is True:
        reasons.append("Artigo retratado")

    if article.get("metadata_conflicts"):
        reasons.append("Conflitos de metadados pendentes de revisão")

    return len(reasons) == 0, reasons


def deduplicate_articles(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicatas preservando áreas, termos e metadados."""
    unique_articles: dict[str, dict[str, Any]] = {}

    for article in articles:
        article_id = normalize_openalex_id(
            article.get("openalex_id")
        )

        if not article_id:
            raise ValueError(
                "Não é possível deduplicar artigo "
                "sem identificador OpenAlex válido"
            )

        if article_id not in unique_articles:
            unique_articles[article_id] = {
                **article,
                "openalex_id": article_id,
                "collection_areas": list(
                    dict.fromkeys(
                        article.get("collection_areas") or []
                    )
                ),
                "retrieval_terms": list(
                    dict.fromkeys(
                        article.get("retrieval_terms") or []
                    )
                ),
                "metadata_conflicts": list(
                    dict.fromkeys(
                        article.get("metadata_conflicts") or []
                    )
                ),
            }
            continue

        existing = unique_articles[article_id]

        # Registra diferenças entre metadados de registros duplicados.
        conflict_fields = (
            "doi",
            "title",
            "publication_year",
            "publication_date",
            "language",
            "publication_type",
            "is_retracted",
        )

        conflicts = existing.setdefault("metadata_conflicts", [])

        for field in conflict_fields:
            old_value = existing.get(field)
            new_value = article.get(field)

            if old_value is not None and new_value is not None:
                if old_value != new_value and field not in conflicts:
                    conflicts.append(field)

        # Preserva conflitos que já estavam registrados.
        for conflict in article.get("metadata_conflicts") or []:
            if conflict not in conflicts:
                conflicts.append(conflict)

        # Combina as áreas e os termos de recuperação.
        for field in ("collection_areas", "retrieval_terms"):
            existing[field] = list(
                dict.fromkeys(
                    (existing.get(field) or [])
                    + (article.get(field) or [])
                )
            )

        # Completa metadados ausentes sem sobrescrever valores existentes.
        complementary_fields = (
            "doi",
            "title",
            "abstract",
            "publication_year",
            "publication_date",
            "language",
            "publication_type",
            "primary_topic_id",
            "primary_topic_name",
            "cited_by_count",
        )

        for field in complementary_fields:
            if existing.get(field) is None or existing.get(field) == "":
                new_value = article.get(field)

                if new_value is not None and new_value != "":
                    existing[field] = new_value

        # Combina tópicos sem repetir os mesmos identificadores.
        existing["topics"] = list(
            {
                topic["id"]: topic
                for topic in (
                    (existing.get("topics") or [])
                    + (article.get("topics") or [])
                )
                if isinstance(topic, dict) and topic.get("id")
            }.values()
        )

        # Combina referências bibliográficas sem duplicação.
        existing["referenced_works"] = list(
            dict.fromkeys(
                (existing.get("referenced_works") or [])
                + (article.get("referenced_works") or [])
            )
        )

    return list(unique_articles.values())
