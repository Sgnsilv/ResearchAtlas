# Metodologia — ResearchAtlas

## Objetivo

Investigar, de forma exploratória e reproduzível, possíveis comunidades científicas e pontes interdisciplinares em artigos sobre aplicações de Inteligência Artificial, sem utilizar áreas científicas como rótulos de treinamento.

## Dataset de referência

- **Fonte:** OpenAlex (`docs/dataset.md`)
- **Período:** 2020–2025
- **Amostra selecionada (v1):** `5.000` artigos únicos (`1.000` por área em `30` estratos área × ano)
- **Áreas:** Medicina, Astronomia, Biologia, Clima/Meio Ambiente e Engenharia

## Representações e algoritmos

| ID | Representação | Algoritmo | Estado |
|---|---|---|---|
| E1 | TF-IDF esparso CSR (`10k` e `20k`, L2) | K-Means | Implementado (Issue #06) |
| E2 | `BAAI/bge-small-en-v1.5` (`384d`, L2) | K-Means | Implementado (Issue #07) |
| E3 | `BAAI/bge-small-en-v1.5` (`384d`, L2) | HDBSCAN (`eom` e `leaf`) | Implementado (Issue #07) |
| E4 | `BAAI/bge-small-en-v1.5` (`384d`, L2) | Agglomerative (`average`, `complete`, `ward`) | Implementado (Issue #07) |
| E5 | Grafo de similaridade / citações | Detecção de comunidades | Planejado |

## Protocolo de avaliação e concordância (Issue #08)

1. **Geometria e esparsidade:**
   - Silhouette calculada com distância cosseno (`sample_size=2000`, `seed=42`, ou cálculo exato quando `n_evaluated <= 2000`).
   - Davies-Bouldin e Calinski-Harabasz calculados em geometria euclidiana apenas sobre embeddings densos BGE.
   - Matrizes TF-IDF não são densificadas; DB e CH permanecem ausentes com justificativa estruturada (`matriz_esparsa_nao_densificada`).
2. **Escopo e comparabilidade:**
   - Partições completas (`full`) e partições parciais com ruído `-1` (`grouped_only`, HDBSCAN) nunca são comparadas diretamente como equivalentes.
   - Valores absolutos de Silhouette obtidos em espaços vetoriais distintos (BGE `384d` versus TF-IDF esparso) não comprovam superioridade científica.
3. **Estabilidade versus concordância por ARI:**
   - **Estabilidade por semente (`seed_initialization`):** ARI entre execuções do K-Means com sementes distintas (`11, 21, 42, 73, 101`) sobre o mesmo espaço e `k`.
   - **Concordância estrutural entre partições:** ARI entre algoritmos, hiperparâmetros ou representações distintas (`cross_algorithm`, `cross_representation`, `hyperparameter_sensitivity`) sobre artigos alinhados por `openalex_id`.
   - **Interseção agrupada (`grouped_intersection`):** quando há ruído (`-1`), o ARI é restrito aos artigos agrupados por ambas as partições e reportado junto à cobertura da interseção; interseções com menos de `2` clusters em qualquer lado são marcadas como inválidas (`menos_de_2_clusters_na_intersecao`), e ARI elevado em baixa cobertura não indica concordância global.
   - **Estabilidade sob reamostragem:** não executada nesta etapa, permanecendo registrada como limitação metodológica.
4. **Validação científica:**
   - Coocorrência de múltiplas áreas de amostragem em um mesmo cluster e valores favoráveis de métricas internas não comprovam interdisciplinaridade; a interpretação temática e a auditoria humana independente pertencem à Issue #09.

## Infraestrutura e reprodutibilidade

- Python `3.12` gerenciado com `uv` e suíte automatizada em `pytest`
- Google Colab (GPU T4) para geração de embeddings e cadernos exploratórios
- Artefatos versionados com manifestos determinísticos e verificação por SHA-256
- Streamlit para visualização interativa