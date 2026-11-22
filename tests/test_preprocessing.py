
"""Testes do pré-processamento de artigos da OpenAlex."""

import sys

import pytest

sys.path.insert(0, "src")

from research_atlas.data.preprocessing import (
    reconstruct_abstract,
    normalize_text,
    normalize_article,
    validate_article,
    deduplicate_articles,
)


@pytest.fixture
def valid_work():
    return {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1234/test",
        "title": " Machine Learning   in Genomics ",
        "abstract_inverted_index": {
            "Machine": [0],
            "learning": [1],
            "improves": [2],
            "genomics": [3],
        },
        "publication_year": 2024,
        "publication_date": "2024-05-10",
        "language": "en",
        "type": "article",
        "is_retracted": False,
        "primary_topic": {
            "id": "T123",
            "display_name": "Computational Genomics",
        },
        "topics": [],
        "referenced_works": ["W10"],
        "cited_by_count": 15,
    }


def test_reconstruct_abstract():
    index = {
        "learning": [1],
        "Machine": [0],
        "works": [2],
    }

    assert reconstruct_abstract(index) == "Machine learning works"


def test_reconstruct_empty_abstract():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_normalize_text():
    assert normalize_text("  Machine   Learning \n") == "Machine Learning"
    assert normalize_text(None) == ""


def test_normalize_article(valid_work):
    article = normalize_article(
        valid_work, "Biologia", "machine learning"
    )

    assert article["openalex_id"] == valid_work["id"]
    assert article["title"] == "Machine Learning in Genomics"
    assert article["abstract"] == "Machine learning improves genomics"
    assert article["collection_areas"] == ["Biologia"]
    assert article["retrieval_terms"] == ["machine learning"]
    assert article["primary_topic_name"] == "Computational Genomics"


def test_validate_article_success(valid_work):
    article = normalize_article(
        valid_work, "Biologia", "machine learning"
    )

    valid, reasons = validate_article(article)

    assert valid is True
    assert reasons == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("openalex_id", None),
        ("title", ""),
        ("abstract", ""),
        ("publication_year", 2019),
        ("publication_year", None),
        ("language", "pt"),
        ("publication_type", "book"),
        ("is_retracted", True),
    ],
)
def test_validate_invalid_article(valid_work, field, value):
    article = normalize_article(
        valid_work, "Biologia", "machine learning"
    )
    article[field] = value

    valid, reasons = validate_article(article)

    assert valid is False
    assert len(reasons) >= 1


def test_deduplicate_articles():
    articles = [
        {
            "openalex_id": "W123",
            "collection_areas": ["Biologia"],
            "retrieval_terms": ["machine learning"],
            "topics": [],
            "referenced_works": ["W10"],
        },
        {
            "openalex_id": "W123",
            "collection_areas": ["Medicina"],
            "retrieval_terms": ["deep learning"],
            "topics": [],
            "referenced_works": ["W20"],
        },
    ]

    result = deduplicate_articles(articles)

    assert len(result) == 1
    assert result[0]["collection_areas"] == ["Biologia", "Medicina"]
    assert result[0]["retrieval_terms"] == ["machine learning", "deep learning"]
    assert result[0]["referenced_works"] == ["W10", "W20"]


def test_deduplicate_missing_id():
    with pytest.raises(ValueError):
        deduplicate_articles([{"title": "No identifier"}])


def test_deduplicate_empty_list():
    assert deduplicate_articles([]) == []
