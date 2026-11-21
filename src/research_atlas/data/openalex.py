"""Cliente HTTP para consulta de artigos na OpenAlex."""

import os
import time
from typing import Iterator

import httpx


class OpenAlexClient:
    BASE_URL = "https://api.openalex.org"
    MAX_RETRIES = 3

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.client = httpx.Client(
            base_url=self.BASE_URL,
            timeout=30.0,
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _request(self, endpoint: str, params: dict) -> dict:
        """Executa uma requisição com tentativas em falhas transitórias."""
        params = params.copy()

        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.get(endpoint, params=params)

                if response.status_code in (429, 500, 502, 503, 504):
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == self.MAX_RETRIES - 1:
                    raise

            except httpx.HTTPStatusError as error:
                if (
                    error.response.status_code
                    not in (429, 500, 502, 503, 504)
                    or attempt == self.MAX_RETRIES - 1
                ):
                    raise

            time.sleep(2 ** attempt)

        raise RuntimeError("Falha inesperada na consulta OpenAlex")

    def search_works(
        self,
        subfields: list[str],
        search_term: str,
        per_page: int = 10,
        cursor: str | None = None,
    ) -> dict:
        """Busca artigos com filtros científicos do ResearchAtlas."""
        if not subfields:
            raise ValueError("Informe pelo menos uma subárea")

        if not 1 <= per_page <= 100:
            raise ValueError("per_page deve estar entre 1 e 100")

        filters = (
            "from_publication_date:2020-01-01,"
            "to_publication_date:2025-12-31,"
            "type:article,"
            "has_abstract:true,"
            "language:en,"
            f"topics.subfield.id:{'|'.join(subfields)},"
            f"title_and_abstract.search:{search_term}"
        )

        params = {
            "filter": filters,
            "per_page": per_page,
        }

        if cursor is not None:
            params["cursor"] = cursor

        return self._request("/works", params)

    def iter_works(
        self,
        subfields: list[str],
        search_term: str,
        max_results: int = 100,
    ) -> Iterator[dict]:
        """Percorre resultados utilizando paginação por cursor."""
        if max_results < 1:
            raise ValueError("max_results deve ser positivo")

        if not self.api_key:
            raise ValueError(
                "OPENALEX_API_KEY é necessária para paginação por cursor"
            )

        cursor = "*"
        collected = 0

        while collected < max_results:
            remaining = max_results - collected

            data = self.search_works(
                subfields=subfields,
                search_term=search_term,
                per_page=min(100, remaining),
                cursor=cursor,
            )

            results = data.get("results", [])

            if not results:
                break

            for article in results:
                yield article
                collected += 1

            next_cursor = data.get("meta", {}).get("next_cursor")

            if not next_cursor or next_cursor == cursor:
                break

            cursor = next_cursor
