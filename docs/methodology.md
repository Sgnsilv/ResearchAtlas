# Metodologia — ResearchAtlas

## Objetivo

Identificar comunidades científicas interdisciplinares
em pesquisas que aplicam Inteligência Artificial.

## Dataset

- Fonte: OpenAlex
- Período: 2020–2025
- Amostra-alvo: aproximadamente 5.000 artigos
- Áreas: Medicina, Astronomia, Biologia,
  Clima/Meio Ambiente e Engenharia

## Experimentos planejados

| ID | Representação | Algoritmo |
|---|---|---|
| E1 | TF-IDF | K-Means |
| E2 | Sentence Transformers | K-Means |
| E3 | Sentence Transformers | HDBSCAN |
| E4 | Sentence Transformers | Agglomerative |
| E5 | Grafo de citações | Leiden |

## Avaliação

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Estabilidade dos agrupamentos
- Interpretação semântica dos clusters

## Infraestrutura

- Google Colab para experimentos
- Streamlit para visualização
- GitHub para versionamento