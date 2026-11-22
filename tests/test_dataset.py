
"""Testes da construção do dataset definitivo."""

import pandas as pd
import pytest

from research_atlas.data.dataset import (
    stable_seed,
    year_quotas,
    select_balanced_articles,
)
from research_atlas.data.pilot import AREAS


def make_article(identifier, area, year):
    """Cria um artigo sintético para testar a seleção."""
    return {
        "openalex_id": f"https://openalex.org/W{identifier}",
        "title": f"Artigo {identifier}",
        "publication_year": year,
        "collection_areas": [area],
        "retrieval_terms": ["machine learning"],
    }


@pytest.fixture
def articles():
    """Gera 240 artigos: oito por combinação de área e ano."""
    result = []
    identifier = 1

    for area in AREAS:
        for year in range(2020, 2026):
            for _ in range(8):
                result.append(
                    make_article(identifier, area, year)
                )
                identifier += 1

    return result


def test_year_quotas():
    quotas = year_quotas(1000)

    assert sum(quotas.values()) == 1000
    assert max(quotas.values()) - min(quotas.values()) <= 1
    assert set(quotas) == set(range(2020, 2026))


def test_stable_seed():
    first = stable_seed(42, "Medicina", 2020, "machine learning", 0)
    second = stable_seed(42, "Medicina", 2020, "machine learning", 0)
    different = stable_seed(42, "Medicina", 2021, "machine learning", 0)

    assert first == second
    assert first != different


def test_balanced_selection(articles):
    selected, report = select_balanced_articles(
        articles,
        target_per_area=12,
        seed=42,
    )

    assert len(selected) == 60
    assert all(item["shortfall"] == 0 for item in report)

    counts = pd.DataFrame(selected).groupby(
        ["sampling_area", "publication_year"]
    ).size()

    assert (counts == 2).all()


def test_selection_has_unique_ids(articles):
    selected, _ = select_balanced_articles(
        articles,
        target_per_area=12,
    )

    identifiers = [
        article["openalex_id"] for article in selected
    ]

    assert len(identifiers) == len(set(identifiers))


def test_selection_is_reproducible(articles):
    first, _ = select_balanced_articles(
        articles, target_per_area=12, seed=42
    )
    second, _ = select_balanced_articles(
        articles, target_per_area=12, seed=42
    )

    assert [
        article["openalex_id"] for article in first
    ] == [
        article["openalex_id"] for article in second
    ]


def test_shortfall_is_reported(articles):
    selected, report = select_balanced_articles(
        articles,
        target_per_area=1000,
    )

    assert len(selected) == len(articles)
    assert sum(item["shortfall"] for item in report) > 0


def test_original_articles_are_preserved(articles):
    original = [
        {**article, "collection_areas": article["collection_areas"].copy()}
        for article in articles
    ]

    select_balanced_articles(articles, target_per_area=12)

    assert articles == original
