
"""Testes automatizados do cliente OpenAlex."""

import sys

import httpx
import pytest

sys.path.insert(0, "src")

from research_atlas.data.openalex import OpenAlexClient


def test_search_works_success():
    """Verifica a construção e execução de uma consulta."""

    def handler(request):
        assert request.url.path == "/works"
        assert request.url.params["per_page"] == "3"

        filters = request.url.params["filter"]

        assert "topics.subfield.id:3103" in filters
        assert "title_and_abstract.search:machine learning" in filters

        return httpx.Response(
            200,
            json={
                "meta": {"count": 1, "next_cursor": None},
                "results": [{"id": "W1", "title": "Test Article"}],
            },
        )

    with OpenAlexClient(api_key="test-key") as client:
        client.client.close()
        client.client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=client.BASE_URL,
        )

        result = client.search_works(
            subfields=["3103"],
            search_term="machine learning",
            per_page=3,
        )

    assert result["meta"]["count"] == 1
    assert len(result["results"]) == 1


@pytest.mark.parametrize("per_page", [0, 101, -1])
def test_invalid_per_page(per_page):
    """Rejeita tamanhos de página inválidos."""

    with OpenAlexClient(api_key="test-key") as client:
        with pytest.raises(ValueError):
            client.search_works(
                subfields=["3103"],
                search_term="machine learning",
                per_page=per_page,
            )


def test_empty_subfields():
    """Exige pelo menos uma subárea."""

    with OpenAlexClient(api_key="test-key") as client:
        with pytest.raises(ValueError):
            client.search_works(
                subfields=[],
                search_term="machine learning",
            )


def test_cursor_pagination():
    """Verifica recuperação de múltiplas páginas."""

    requested_cursors = []

    def handler(request):
        cursor = request.url.params.get("cursor")
        requested_cursors.append(cursor)

        if cursor == "*":
            return httpx.Response(
                200,
                json={
                    "meta": {"next_cursor": "page-2"},
                    "results": [
                        {"id": "W1"},
                        {"id": "W2"},
                    ],
                },
            )

        return httpx.Response(
            200,
            json={
                "meta": {"next_cursor": None},
                "results": [{"id": "W3"}],
            },
        )

    with OpenAlexClient(api_key="test-key") as client:
        client.client.close()
        client.client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=client.BASE_URL,
        )

        articles = list(
            client.iter_works(
                subfields=["3103"],
                search_term="machine learning",
                max_results=3,
            )
        )

    assert len(articles) == 3
    assert requested_cursors == ["*", "page-2"]


def test_pagination_requires_api_key(monkeypatch):
    """Valida a exigência da chave para paginação."""

    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    with OpenAlexClient(api_key=None) as client:
        with pytest.raises(ValueError):
            list(
                client.iter_works(
                    subfields=["3103"],
                    search_term="machine learning",
                )
            )


def test_retry_after_rate_limit(monkeypatch):
    """Repete a requisição quando a API retorna HTTP 429."""
    monkeypatch.setattr(
        "research_atlas.data.openalex.time.sleep",
        lambda _: None,
    )

    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(429)

        return httpx.Response(
            200,
            json={"meta": {"count": 1}, "results": []},
        )

    with OpenAlexClient(api_key="test-key") as client:
        client.client.close()
        client.client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=client.BASE_URL,
        )

        result = client.search_works(
            subfields=["3103"],
            search_term="machine learning",
        )

    assert attempts == 2
    assert result["meta"]["count"] == 1


def test_invalid_request_does_not_retry():
    """Erros HTTP 400 não devem gerar novas tentativas."""
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(400)

    with OpenAlexClient(api_key="test-key") as client:
        client.client.close()
        client.client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=client.BASE_URL,
        )

        with pytest.raises(httpx.HTTPStatusError):
            client.search_works(
                subfields=["3103"],
                search_term="machine learning",
            )

    assert attempts == 1
