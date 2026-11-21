
"""Testes da amostragem para auditoria científica do ResearchAtlas."""

import pandas as pd
import pytest

from research_atlas.analysis.audit import select_stratified_sample


AREAS = [
    "Medicina",
    "Astronomia",
    "Biologia",
    "Clima",
    "Engenharia",
]

YEARS = range(2020, 2026)


@pytest.fixture
def sample_dataframe():
    """Cria 120 artigos sintéticos: 24 por área e 4 por ano."""
    articles = []
    identifier = 1

    for area in AREAS:
        for year in YEARS:
            for _ in range(4):
                articles.append(
                    {
                        "openalex_id": f"https://openalex.org/W{identifier}",
                        "title": f"Artigo {identifier}",
                        "publication_year": year,
                        "collection_areas": [area],
                    }
                )
                identifier += 1

    return pd.DataFrame(articles)


def test_sample_has_100_articles(sample_dataframe):
    result = select_stratified_sample(sample_dataframe)

    assert len(result) == 100


def test_sample_has_20_articles_per_area(sample_dataframe):
    result = select_stratified_sample(sample_dataframe)

    counts = result["audit_area"].value_counts()

    assert set(counts.index) == set(AREAS)
    assert (counts == 20).all()


def test_sampling_is_reproducible(sample_dataframe):
    first = select_stratified_sample(sample_dataframe)
    second = select_stratified_sample(sample_dataframe)

    pd.testing.assert_frame_equal(first, second)


def test_sample_has_no_duplicate_articles(sample_dataframe):
    result = select_stratified_sample(sample_dataframe)

    assert result["openalex_id"].is_unique


def test_sample_balances_years(sample_dataframe):
    result = select_stratified_sample(sample_dataframe)

    for area in AREAS:
        subset = result[result["audit_area"] == area]
        counts = subset["publication_year"].value_counts()

        assert set(counts.index) == set(YEARS)
        assert counts.max() - counts.min() <= 1


def test_insufficient_articles_raises_error(sample_dataframe):
    reduced = sample_dataframe[
        sample_dataframe["collection_areas"].apply(
            lambda areas: "Medicina" not in areas
        )
    ].copy()

    # Mantém Medicina na amostra, mas com apenas 10 registros.
    medicine = sample_dataframe[
        sample_dataframe["collection_areas"].apply(
            lambda areas: "Medicina" in areas
        )
    ].head(10)

    reduced = pd.concat([reduced, medicine], ignore_index=True)

    with pytest.raises(ValueError, match="Medicina"):
        select_stratified_sample(reduced)


def test_original_dataframe_is_not_modified(sample_dataframe):
    original = sample_dataframe.copy(deep=True)

    select_stratified_sample(sample_dataframe)

    pd.testing.assert_frame_equal(sample_dataframe, original)
