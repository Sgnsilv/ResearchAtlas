
"""Testes do pipeline de qualidade do dataset piloto."""

import sys

import pandas as pd
import pytest

sys.path.insert(0, "src")

from research_atlas.data.pilot import (
    evaluate_text_quality,
    generate_quality_report,
)
from research_atlas.data.preprocessing import assess_abstract_quality


@pytest.fixture
def sample_dataframe():
    """Cria uma amostra sintética para os testes."""
    return pd.DataFrame(
        {
            "openalex_id": ["W1", "W2", "W3"],
            "abstract": [
                "Machine learning improves genomic prediction.",
                " ".join(["Scientific research methods"] * 60),
                "Preview this article: Previous page | Next page",
            ],
        }
    )


def test_short_abstract():
    result = assess_abstract_quality("Short scientific abstract.")

    assert result["word_count"] == 3
    assert "short_abstract" in result["quality_flags"]
    assert result["requires_review"] is True


def test_suspicious_content():
    result = assess_abstract_quality(
        "Preview this article: Previous page | Next page"
    )

    assert "suspicious_content" in result["quality_flags"]


def test_long_abstract():
    result = assess_abstract_quality("research " * 501)

    assert result["word_count"] == 501
    assert "long_abstract" in result["quality_flags"]


def test_valid_abstract():
    result = assess_abstract_quality(
        " ".join(["scientific"] * 150)
    )

    assert result["quality_flags"] == []
    assert result["requires_review"] is False


def test_evaluate_text_quality(sample_dataframe):
    result = evaluate_text_quality(sample_dataframe)

    assert "abstract_word_count" in result.columns
    assert "quality_flags" in result.columns
    assert "requires_review" in result.columns

    assert result["requires_review"].sum() == 2

    # O DataFrame original não deve ser modificado.
    assert "quality_flags" not in sample_dataframe.columns


def test_generate_quality_report(sample_dataframe):
    dataframe = evaluate_text_quality(sample_dataframe)
    report = generate_quality_report(dataframe)

    assert report["requires_review"] == 2
    assert report["short_abstracts"] == 2
    assert report["long_abstracts"] == 0
    assert report["suspicious_content"] == 1


def test_empty_dataframe():
    dataframe = evaluate_text_quality(pd.DataFrame())

    report = generate_quality_report(dataframe)

    assert dataframe.empty
    assert report["requires_review"] == 0
    assert report["short_abstracts"] == 0
    assert report["long_abstracts"] == 0
    assert report["suspicious_content"] == 0
