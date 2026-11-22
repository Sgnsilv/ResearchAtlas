
"""Geração de amostras para auditoria científica do dataset piloto."""

import json
from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/processed/pilot_articles.parquet")
OUTPUT_DIR = Path("data/processed/audit")
SAMPLE_SIZE = 20
RANDOM_SEED = 42


def select_stratified_sample(
    dataframe: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Seleciona artigos equilibrando os anos disponíveis em cada área."""

    selected = []

    areas = sorted(
        {
            area
            for values in dataframe["collection_areas"]
            for area in values
        }
    )

    for area in areas:
        subset = dataframe[
            dataframe["collection_areas"].apply(
                lambda values: area in values
            )
        ].copy()

        if len(subset) < sample_size:
            raise ValueError(
                f"A área {area} possui apenas {len(subset)} artigos."
            )

        # Embaralha de maneira reproduzível dentro de cada ano.
        year_groups = {
            year: group.sample(
                frac=1,
                random_state=RANDOM_SEED,
            ).reset_index(drop=True)
            for year, group in subset.groupby("publication_year")
        }

        positions = {year: 0 for year in year_groups}
        area_sample = []

        # Seleção por rodadas para favorecer equilíbrio temporal.
        while len(area_sample) < sample_size:
            progress = False

            for year in sorted(year_groups):
                group = year_groups[year]
                position = positions[year]

                if position >= len(group):
                    continue

                area_sample.append(group.iloc[position].to_dict())
                positions[year] += 1
                progress = True

                if len(area_sample) == sample_size:
                    break

            if not progress:
                raise ValueError(
                    f"Não foi possível completar a amostra de {area}."
                )

        for article in area_sample:
            article["audit_area"] = area

        selected.extend(area_sample)

    return pd.DataFrame(selected)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(DATASET_PATH)

    dataframe["collection_areas"] = (
        dataframe["collection_areas"].apply(json.loads)
    )
    dataframe["quality_flags"] = (
        dataframe["quality_flags"].apply(json.loads)
    )

    sample = select_stratified_sample(dataframe)

    columns = [
        "openalex_id",
        "audit_area",
        "publication_year",
        "title",
        "abstract",
        "abstract_word_count",
        "quality_flags",
    ]

    sample = sample[columns].copy()

    sample["quality_flags"] = sample["quality_flags"].apply(
        lambda flags: "; ".join(flags)
    )

    sample.insert(
        0,
        "audit_id",
        [f"RA-{number:03d}" for number in range(1, len(sample) + 1)],
    )

    sample["relevance"] = ""
    sample["abstract_adequacy"] = ""
    sample["notes"] = ""

    for reviewer in ("reviewer_1", "reviewer_2"):
        path = OUTPUT_DIR / f"{reviewer}.csv"

        sample.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Gerado: {path}")

    print("\n=== AMOSTRA DE AUDITORIA ===")
    print(f"Total de artigos: {len(sample)}")

    print("\nPor área:")
    print(sample["audit_area"].value_counts().to_string())

    print("\nPor ano:")
    print(
        sample["publication_year"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()
