
"""Análise exploratória do dataset piloto do ResearchAtlas."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px


DATASET_PATH = Path("data/processed/pilot_articles.parquet")
OUTPUT_DIR = Path("reports/eda")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carrega o dataset piloto e recupera campos estruturados."""
    dataframe = pd.read_parquet(path)

    for field in ("collection_areas", "retrieval_terms", "quality_flags"):
        dataframe[field] = dataframe[field].apply(json.loads)

    return dataframe


def generate_eda(dataframe: pd.DataFrame) -> dict:
    """Gera gráficos interativos e estatísticas exploratórias."""
    if dataframe.empty:
        raise ValueError("O dataset piloto está vazio.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Publicações por ano
    years = (
        dataframe["publication_year"]
        .value_counts()
        .sort_index()
        .rename_axis("Ano")
        .reset_index(name="Artigos")
    )

    fig = px.bar(
        years,
        x="Ano",
        y="Artigos",
        title="Distribuição temporal das publicações",
        text="Artigos",
    )
    fig.update_xaxes(type="category")
    fig.write_html(OUTPUT_DIR / "01_publicacoes_por_ano.html")

    # 2. Artigos por área
    areas = (
        dataframe.explode("collection_areas")["collection_areas"]
        .value_counts()
        .rename_axis("Área")
        .reset_index(name="Artigos")
    )

    fig = px.bar(
        areas,
        x="Área",
        y="Artigos",
        title="Artigos por área de coleta",
        text="Artigos",
    )
    fig.write_html(OUTPUT_DIR / "02_artigos_por_area.html")

    # 3. Distribuição do tamanho dos resumos
    fig = px.histogram(
        dataframe,
        x="abstract_word_count",
        nbins=40,
        title="Distribuição do tamanho dos resumos",
        labels={
            "abstract_word_count": "Número de palavras",
            "count": "Artigos",
        },
    )
    fig.update_layout(yaxis_title="Artigos")
    fig.write_html(OUTPUT_DIR / "03_tamanho_resumos.html")

    # 4. Indicadores de qualidade
    quality_counts = Counter(
        flag
        for flags in dataframe["quality_flags"]
        for flag in flags
    )

    quality = pd.DataFrame(
        {
            "Indicador": [
                "Resumo curto",
                "Conteúdo suspeito",
                "Resumo longo",
            ],
            "Artigos": [
                quality_counts.get("short_abstract", 0),
                quality_counts.get("suspicious_content", 0),
                quality_counts.get("long_abstract", 0),
            ],
        }
    )

    fig = px.bar(
        quality,
        x="Indicador",
        y="Artigos",
        title="Alertas de qualidade textual",
        text="Artigos",
    )
    fig.write_html(OUTPUT_DIR / "04_qualidade_textual.html")

    # 5. Estatísticas resumidas
    area_counts = dataframe["collection_areas"].apply(len)

    summary = {
        "total_articles": len(dataframe),
        "unique_openalex_ids": int(
            dataframe["openalex_id"].nunique()
        ),
        "publication_years": {
            str(row["Ano"]): int(row["Artigos"])
            for _, row in years.iterrows()
        },
        "collection_areas": {
            row["Área"]: int(row["Artigos"])
            for _, row in areas.iterrows()
        },
        "abstract_length": {
            "mean": float(
                dataframe["abstract_word_count"].mean()
            ),
            "median": float(
                dataframe["abstract_word_count"].median()
            ),
            "min": int(
                dataframe["abstract_word_count"].min()
            ),
            "max": int(
                dataframe["abstract_word_count"].max()
            ),
        },
        "multi_area_articles": int((area_counts > 1).sum()),
        "quality_flags": dict(quality_counts),
        "requires_review": int(
            dataframe["requires_review"].sum()
        ),
    }

    summary_path = OUTPUT_DIR / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    dataframe = load_dataset()
    summary = generate_eda(dataframe)

    print("\n=== ANÁLISE EXPLORATÓRIA ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nGráficos disponíveis em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
