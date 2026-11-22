
"""Testes do cache de consultas OpenAlex."""

import json
from unittest.mock import Mock

import pytest

from research_atlas.data.dataset import cached_sample


@pytest.fixture
def parameters(tmp_path):
    return {
        "area": "Astronomia",
        "subfields": ["3103"],
        "year": 2024,
        "term": "machine learning",
        "sample_size": 20,
        "seed": 42,
        "cache_dir": tmp_path,
    }


def test_first_request_creates_cache(parameters):
    client = Mock()
    client.sample_works.return_value = [
        {"id": "https://openalex.org/W123"}
    ]

    result = cached_sample(client, **parameters)

    assert len(result) == 1
    client.sample_works.assert_called_once()
    assert len(list(parameters["cache_dir"].glob("*.json"))) == 1


def test_second_request_uses_cache(parameters):
    client = Mock()
    client.sample_works.return_value = [{"id": "W123"}]

    first = cached_sample(client, **parameters)
    second = cached_sample(client, **parameters)

    assert first == second
    assert client.sample_works.call_count == 1


def test_different_seed_creates_separate_cache(parameters):
    client = Mock()
    client.sample_works.return_value = [{"id": "W123"}]

    cached_sample(client, **parameters)
    cached_sample(client, **{**parameters, "seed": 43})

    assert client.sample_works.call_count == 2


def test_different_sample_size_creates_separate_cache(parameters):
    client = Mock()
    client.sample_works.return_value = [{"id": "W123"}]

    cached_sample(client, **parameters)
    cached_sample(
        client,
        **{**parameters, "sample_size": 30},
    )

    assert client.sample_works.call_count == 2


def test_invalid_cache_raises_error(parameters):
    client = Mock()
    client.sample_works.return_value = [{"id": "W123"}]

    cached_sample(client, **parameters)

    cache_path = next(
        parameters["cache_dir"].glob("*.json")
    )

    payload = json.loads(cache_path.read_text())
    payload["results"] = "invalid"

    cache_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Resposta inválida"):
        cached_sample(client, **parameters)
