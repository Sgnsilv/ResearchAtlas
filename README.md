# ResearchAtlas

**Exploração de comunidades científicas interdisciplinares por meio de clusterização semântica e análise de grafos.**

O ResearchAtlas é um projeto de pesquisa e uma plataforma interativa desenvolvida em Python e Streamlit para investigar como a Inteligência Artificial é aplicada a diferentes áreas do conhecimento científico.

A partir de metadados da OpenAlex e métodos de aprendizado não supervisionado, investigamos padrões lexicais e semânticos, comunidades de pesquisa e possíveis conexões interdisciplinares.

> Projeto acadêmico de Aprendizagem de Máquina, desenvolvido também como portfólio técnico e científico. A presença de artigos de diferentes áreas em um cluster não comprova, por si só, interdisciplinaridade.

## Pergunta científica

**É possível identificar comunidades de pesquisa interdisciplinares a partir da similaridade semântica de artigos sobre aplicações de Inteligência Artificial e descobrir conexões entre áreas científicas distintas?**

### Áreas investigadas

- Medicina e Saúde
- Astronomia
- Biologia e Bioinformática
- Clima e Meio Ambiente
- Engenharia

**Período:** 2020 a 2025.

## Estado atual

O ResearchAtlas possui um dataset científico balanceado, embeddings semânticos BGE, um baseline lexical TF-IDF + K-Means e experimentos comparativos com K-Means, Agglomerative Clustering e HDBSCAN.

As etapas de coleta, preparação, representação textual e experimentação inicial foram implementadas e testadas.

A avaliação aprofundada da qualidade e estabilidade dos agrupamentos, a interpretação científica e a auditoria humana permanecem em desenvolvimento.

| Indicador | Resultado |
|---|---:|
| Artigos selecionados | 5.000 |
| Identificadores únicos | 5.000 |
| Candidatos estruturalmente válidos | 11.210 |
| Artigos por área de amostragem | 1.000 |
| Estratos área × ano preenchidos | 30/30 |
| Déficit de amostragem | 0 |
| Artigos recuperados por múltiplas áreas | 7 |
| Artigos sinalizados para revisão textual | 195 |
| Embeddings BGE | 5.000 × 384 |
| Características do baseline TF-IDF selecionado | 10.000 |
| Clusters do baseline TF-IDF selecionado | 8 |
| Algoritmos de clusterização semântica | 3 |
| Configurações com rótulos BGE preservados | 18 |
| Testes automatizados | 106 aprovados |

O arquivo de referência dos artigos é:

`data/processed/dataset_selected.parquet`

**SHA-256 do dataset selecionado v1:**

```text
533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389
```

O hash identifica a versão exata do arquivo utilizada nos experimentos. Uma nova coleta na OpenAlex não é necessariamente idêntica.

## Tecnologias

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3.12 |
| Ambiente e dependências | uv |
| Fonte de dados | OpenAlex API |
| HTTP | HTTPX |
| Dados e armazenamento | pandas, NumPy, PyArrow, Parquet, JSON |
| Machine Learning | scikit-learn, HDBSCAN, TF-IDF, K-Means, Agglomerative Clustering |
| Representação semântica | BAAI/bge-small-en-v1.5 |
| Processamento dos embeddings | Sentence Transformers, Google Colab |
| Visualização e interface | Plotly, Streamlit, Matplotlib nos notebooks |
| Testes | pytest |
| Planejado | UMAP, análise de grafos e detecção de comunidades |

As dependências específicas de experimentos no Colab podem ser instaladas no próprio ambiente experimental. O ambiente principal utiliza as versões declaradas em `pyproject.toml` e `uv.lock`.

## Instalação

### Pré-requisitos

- Git
- uv
- Python 3.12
- Internet e chave da OpenAlex apenas para novas consultas à API

```bash
git clone https://github.com/Sgnsilv/ResearchAtlas.git
cd ResearchAtlas

uv python install 3.12
uv sync --dev
```

### Configuração da OpenAlex

Obtenha uma chave em:

https://openalex.org/settings/api

Crie o arquivo local de configuração:

```bash
cp .env.example .env
```

Preencha:

```dotenv
OPENALEX_API_KEY=sua_chave_aqui
```

Não versione o arquivo `.env` nem publique chaves de acesso.

Os comandos que consultam a API utilizam `uv run --env-file .env`.

## Estrutura do projeto

```text
ResearchAtlas/
├── app/
│   └── Home.py
│
├── src/
│   └── research_atlas/
│       ├── data/
│       │   ├── openalex.py
│       │   ├── preprocessing.py
│       │   ├── pilot.py
│       │   └── dataset.py
│       │
│       ├── analysis/
│       │   ├── eda.py
│       │   └── audit.py
│       │
│       ├── features/
│       │   ├── embeddings.py
│       │   └── tfidf.py
│       │
│       └── clustering/
│           └── algorithms.py
│
├── notebooks/
│   ├── 01_embeddings_semanticos.ipynb
│   ├── 02_clusterizacao.ipynb
│   └── 03_baseline_tfidf_kmeans.ipynb
│
├── data/
│   ├── raw/
│   │   └── openalex_cache/
│   │
│   └── processed/
│       ├── dataset_selected.parquet
│       ├── dataset_report.json
│       │
│       ├── embeddings/
│       │   └── bge-small-en-v1.5-v1/
│       │       ├── embeddings.npy
│       │       ├── metadata.parquet
│       │       └── manifest.json
│       │
│       ├── tfidf_baseline/
│       │   └── v1/
│       │       ├── tfidf_assignments.parquet
│       │       ├── kmeans_k_sensitivity.csv
│       │       ├── tfidf_parameter_sensitivity.csv
│       │       ├── tfidf_stability.csv
│       │       ├── tfidf_final_candidates.csv
│       │       └── manifest.json
│       │
│       └── clustering/
│           └── v1/
│               ├── cluster_assignments.parquet
│               ├── kmeans_metrics.csv
│               ├── kmeans_stability.csv
│               ├── agglomerative_metrics.csv
│               ├── hdbscan_eom_metrics.csv
│               ├── hdbscan_leaf_metrics.csv
│               └── manifest.json
│
├── reports/
│   ├── eda/
│   └── experiments/
│       ├── pilot_dataset.md
│       ├── final_dataset.md
│       ├── semantic_embeddings.md
│       ├── tfidf_baseline.md
│       └── clustering_comparison.md
│
├── docs/
│   └── dataset.md
│
├── tests/
│   ├── test_embeddings.py
│   ├── test_embedding_artifacts.py
│   ├── test_tfidf.py
│   ├── test_tfidf_artifacts.py
│   ├── test_clustering.py
│   └── test_clustering_artifacts.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

A árvore destaca os principais arquivos e diretórios. Outros módulos, testes e arquivos auxiliares não estão enumerados.

### Versionamento dos dados

O `.gitignore` ignora artefatos gerados por padrão.

Os arquivos de referência necessários à reprodução dos experimentos são adicionados explicitamente ao Git, incluindo:

- Dataset selecionado;
- Embeddings e metadados associados;
- Artefatos do baseline TF-IDF;
- Resultados de clusterização semântica;
- Manifestos com parâmetros e checksums.

Caches, arquivos temporários e saídas intermediárias não são necessariamente versionados.

## Funcionalidades implementadas

### 1. Protocolo científico

O documento `docs/dataset.md` registra:

- Pergunta científica;
- Objetivos e hipóteses;
- Áreas do conhecimento;
- Critérios de inclusão e exclusão;
- Período de publicação;
- Estratégia de consulta à OpenAlex;
- Decisões metodológicas.

A definição antecipada do protocolo busca reduzir alterações arbitrárias nos critérios de coleta e análise.

### 2. Cliente OpenAlex

Implementado em:

`src/research_atlas/data/openalex.py`

Funcionalidades:

- Autenticação por variável de ambiente;
- Recuperação de artigos científicos;
- Consultas filtradas por área e termos;
- Paginação por cursor;
- Amostragem com semente;
- Retentativas para falhas HTTP;
- Tratamento de respostas;
- Persistência de dados para processamento.

### 3. Pré-processamento

Implementado em:

`src/research_atlas/data/preprocessing.py`

Operações:

- Reconstrução de resumos;
- Normalização de identificadores;
- Normalização textual;
- Validação estrutural;
- Deduplicação;
- Sinalização de possíveis problemas de qualidade textual.

A sinalização de qualidade não implica exclusão automática de artigos.

### 4. Dataset piloto

Execução:

```bash
PYTHONPATH=src uv run --env-file .env python -m research_atlas.data.pilot
```

A execução documentada produziu:

| Indicador | Resultado |
|---|---:|
| Registros brutos | 400 |
| Artigos únicos | 398 |
| Artigos sinalizados | 13 |

O piloto ajudou a identificar possíveis problemas de resumos e viés temporal associado a consultas ordenadas por relevância.

Relatório:

`reports/experiments/pilot_dataset.md`

### 5. Análise exploratória

Implementada em:

`src/research_atlas/analysis/eda.py`

Execução:

```bash
PYTHONPATH=src uv run python -m research_atlas.analysis.eda
```

A análise exploratória contempla:

- Distribuição por ano;
- Distribuição por área;
- Comprimento dos resumos;
- Alertas de qualidade textual;
- Visualizações descritivas.

A análise original foi aplicada ao dataset piloto.

### 6. Preparação da auditoria humana

Implementada em:

`src/research_atlas/analysis/audit.py`

Execução:

```bash
PYTHONPATH=src uv run python -m research_atlas.analysis.audit
```

O módulo prepara dois arquivos CSV independentes com a mesma amostra de 100 artigos, sendo 20 por área.

A auditoria contempla critérios de:

- Relevância para Inteligência Artificial;
- Pertinência à área científica;
- Qualidade do resumo;
- Consistência dos metadados.

**Estado:** amostra preparada; revisão humana pendente.

### 7. Dataset científico v1

O módulo:

`src/research_atlas/data/dataset.py`

integra consultas amostradas, validação, deduplicação, seleção estratificada, relatórios e exportação.

Exemplo de execução:

```bash
PYTHONPATH=src uv run --env-file .env python \
  -m research_atlas.data.dataset \
  --sample-size 100 \
  --target-per-area 1000 \
  --seed 42 \
  --rounds 1
```

#### Resultado da coleta de referência

Foram obtidos 11.210 candidatos estruturalmente válidos, dos quais 5.000 artigos únicos foram selecionados para o dataset v1.

| Área de amostragem | Artigos |
|---|---:|
| Medicina | 1.000 |
| Astronomia | 1.000 |
| Biologia | 1.000 |
| Clima | 1.000 |
| Engenharia | 1.000 |
| **Total** | **5.000** |

Os 30 estratos de área × ano foram preenchidos, com aproximadamente 166 ou 167 artigos por estrato.

O dataset foi deliberadamente balanceado e não representa a distribuição natural da produção científica na OpenAlex.

#### Qualidade textual

| Indicador | Artigos |
|---|---:|
| Resumo com menos de 50 palavras | 72 |
| Resumo com mais de 500 palavras | 122 |
| Conteúdo suspeito | 8 |
| Pelo menos um alerta | 195 |

Os alertas podem se sobrepor.

Distribuição por área:

| Área | Artigos sinalizados |
|---|---:|
| Medicina | 55 |
| Astronomia | 22 |
| Biologia | 81 |
| Clima | 21 |
| Engenharia | 16 |

O cache das respostas é armazenado em:

`data/raw/openalex_cache/`

Uma nova coleta sem o mesmo cache pode retornar resultados diferentes devido à evolução da OpenAlex.

Relatório:

`reports/experiments/final_dataset.md`

### 8. Embeddings semânticos — Issue #05

Foram utilizados embeddings produzidos pelo modelo:

`BAAI/bge-small-en-v1.5`

Cada um dos 5.000 artigos foi representado em um espaço vetorial de 384 dimensões.

#### Estratégia de representação

| Parâmetro | Valor |
|---|---|
| Entrada | Título + resumo |
| Limite do modelo | 512 tokens |
| Janela | 510 tokens de conteúdo + tokens especiais |
| Sobreposição | 64 tokens |
| Agregação | Média ponderada dos trechos |
| Normalização | L2 |
| Ambiente | Google Colab, GPU Tesla T4 |

Foram processados 5.678 trechos, produzindo 5.000 vetores normalizados.

A implementação reutilizável de segmentação e agregação está disponível em:

`src/research_atlas/features/embeddings.py`

#### Artefatos

```text
data/processed/embeddings/bge-small-en-v1.5-v1/
├── embeddings.npy
├── metadata.parquet
└── manifest.json
```

**SHA-256 de `embeddings.npy`:**

```text
13fe24f37622790f6d6630fbdb5e91742060f95a8d7310597d3bd98f2a7f465c
```

Exemplo de carregamento:

```python
import numpy as np
import pandas as pd

base = "data/processed/embeddings/bge-small-en-v1.5-v1"

embeddings = np.load(
    f"{base}/embeddings.npy",
    allow_pickle=False,
)

metadata = pd.read_parquet(
    f"{base}/metadata.parquet"
)

print(embeddings.shape)  # (5000, 384)
print(metadata.shape)    # (5000, 9)
```

A coluna `embedding_index` preserva a correspondência entre vetores e artigos.

**Notebook:** `notebooks/01_embeddings_semanticos.ipynb`

**Relatório:** `reports/experiments/semantic_embeddings.md`

### 9. Baseline TF-IDF + K-Means — Issue #06

Foi implementado um baseline lexical de clusterização sobre os mesmos 5.000 artigos utilizados na representação semântica BGE.

A entrada é formada pela concatenação do título e do resumo.

As áreas científicas não são utilizadas como rótulos de treinamento.

#### Configuração de referência

| Parâmetro | Valor |
|---|---|
| Vetorização | TF-IDF |
| Normalização | L2 |
| Vocabulário máximo | 10.000 características |
| ngram_range | (1, 2) |
| min_df | 3 |
| max_df | 0,95 |
| stop_words | english |
| sublinear_tf | True |
| Algoritmo | K-Means |
| k | 8 |
| random_state | 42 |
| n_init | 10 |
| Silhouette | Distância cosseno, amostra de 2.000 artigos |

A matriz inicial de TF-IDF com 20.000 características apresentou:

- 5.000 linhas;
- 20.000 colunas;
- 648.816 valores não nulos;
- Densidade de 0,6488%;
- Representação esparsa CSR.

#### Experimentos

Foram realizados:

- Variação de `k` entre 2 e 15;
- Análise de sensibilidade dos parâmetros do TF-IDF;
- Comparação entre unigramas e unigramas + bigramas;
- Comparação de tamanhos de vocabulário;
- Análise de estabilidade entre inicializações;
- Inspeção exploratória dos temas dos clusters.

A estabilidade foi avaliada com as sementes 11, 21, 42, 73 e 101.

#### Resultados

| TF-IDF 10k | Silhouette | ARI médio | Maior cluster |
|---|---:|---:|---:|
| k=5 | 0,0130 | 0,6578 | 38,64% |
| **k=8** | **0,0146** | **0,6692** | **19,52%** |
| k=13 | 0,0156 | 0,5044 | 13,68% |

O valor `k=8` foi selecionado como compromisso entre separação, estabilidade e distribuição dos agrupamentos.

Essa configuração não representa necessariamente o número verdadeiro de comunidades científicas.

A Silhouette permaneceu baixa nas configurações avaliadas.

#### Interpretação temática preliminar

| Cluster | Artigos | Tema |
|---|---:|---|
| 0 | 772 | Controle, robótica e sistemas |
| 1 | 976 | Predição climática e meteorológica |
| 2 | 140 | COVID-19 e diagnóstico |
| 3 | 424 | Diagnóstico por imagens e câncer |
| 4 | 554 | Aplicações gerais de IA na saúde |
| 5 | 883 | Deep Learning e análise de imagens |
| 6 | 631 | Galáxias e classificação astronômica |
| 7 | 620 | Genômica, proteínas e genética |

Os temas foram inferidos a partir de termos de maior peso nos centroides e inspeção de títulos representativos.

O agrupamento de artigos de diferentes áreas por vocabulário compartilhado pode indicar similaridade metodológica, sem necessariamente comprovar interdisciplinaridade.

#### Implementação

- Notebook: `notebooks/03_baseline_tfidf_kmeans.ipynb`
- Módulo: `src/research_atlas/features/tfidf.py`
- Testes: `tests/test_tfidf.py`
- Integração: `tests/test_tfidf_artifacts.py`

#### Artefatos

```text
data/processed/tfidf_baseline/v1/
├── tfidf_assignments.parquet
├── kmeans_k_sensitivity.csv
├── tfidf_parameter_sensitivity.csv
├── tfidf_stability.csv
├── tfidf_final_candidates.csv
└── manifest.json
```

Os artefatos preservam rótulos de seis configurações selecionadas, tabelas de sensibilidade, métricas e parâmetros experimentais.

O manifesto registra a versão do dataset e checksums dos arquivos.

**Relatório:** [Baseline TF-IDF + K-Means](reports/experiments/tfidf_baseline.md)

**Estado:** implementação, experimentos, testes e documentação concluídos e integrados à `dev` por meio do PR #22.

### 10. Clusterização semântica — Issue #07

Foram implementadas e comparadas três abordagens de clusterização não supervisionada utilizando os mesmos 5.000 embeddings BGE normalizados:

1. K-Means;
2. Agglomerative Clustering;
3. HDBSCAN.

O objetivo é analisar como diferentes algoritmos organizam os artigos no espaço semântico e quais diferenças surgem em relação à quantidade de clusters, distribuição dos agrupamentos e identificação de ruído.

As áreas científicas não foram utilizadas no treinamento dos algoritmos.

#### 10.1. K-Means

Foram avaliados valores de `k` entre 2 e 15.

Configuração:

- `random_state=42`;
- `n_init=10`;
- Embeddings normalizados;
- Avaliação por Silhouette com distância cosseno;
- Amostra de 2.000 artigos na Silhouette.

Resultados:

| Indicador | Resultado |
|---|---:|
| Maior Silhouette | 0,1472 |
| k da maior Silhouette | 8 |
| Menor Davies-Bouldin | 2,9904 |
| k do menor Davies-Bouldin | 9 |
| Maior Calinski-Harabasz | 361,87 |
| k do maior Calinski-Harabasz | 2 |

Os diferentes indicadores favoreceram configurações distintas.

O valor `k=8` foi mantido como candidato para análise temática.

##### Estabilidade entre inicializações

Foram avaliadas cinco sementes: 11, 21, 42, 73 e 101.

| k | ARI médio |
|---|---:|
| 5 | 0,9979 |
| 7 | 0,9471 |
| 8 | 0,8621 |
| 9 | 0,8926 |

O valor `k=5` apresentou maior consistência entre inicializações.

O ARI mede a concordância entre partições e não representa uma medida direta de interdisciplinaridade.

#### 10.2. Agglomerative Clustering

Foram comparados três métodos de ligação, mantendo oito clusters.

| Linkage | Distância | Silhouette | Maior cluster |
|---|---|---:|---:|
| Average | Cosseno | 0,1082 | 4.983 |
| Complete | Cosseno | 0,1112 | 1.008 |
| Ward | Euclidiana | 0,1142 | 949 |

A configuração Average + cosseno gerou uma partição extremamente concentrada.

A configuração Ward + euclidiana apresentou uma distribuição mais equilibrada dos artigos e a maior Silhouette entre as alternativas hierárquicas testadas.

O ARI entre Agglomerative Ward e K-Means, ambos com `k=8`, foi **0,7139**.

A configuração Ward foi mantida como principal referência hierárquica para análises posteriores.

#### 10.3. HDBSCAN

O HDBSCAN foi utilizado para investigar a identificação de comunidades densas sem exigir a definição prévia do número de clusters.

Foram avaliadas duas estratégias:

- **EOM:** Excess of Mass;
- **Leaf:** seleção de agrupamentos nas folhas da hierarquia.

Os experimentos utilizaram distância euclidiana sobre os embeddings normalizados.

##### HDBSCAN — EOM

| min_cluster_size | min_samples | Clusters | Cobertura | Ruído |
|---|---|---:|---:|---:|
| 20 | 5 | 2 | 56,80% | 43,20% |
| 20 | 10 | 8 | 22,36% | 77,64% |
| 30 | 10 | 4 | 26,84% | 73,16% |
| 50 | 5 | 6 | 26,70% | 73,30% |
| 50 | 10 | 3 | 25,96% | 74,04% |
| 100 | 5 | 4 | 25,26% | 74,74% |

A configuração (20, 5) apresentou maior cobertura, mas concentrou quase todos os artigos agrupados em um único cluster.

A configuração (50, 5) identificou seis clusters com cobertura de 26,70%.

##### HDBSCAN — Leaf

| min_cluster_size | min_samples | Clusters | Cobertura | Silhouette dos agrupados |
|---|---|---:|---:|---:|
| 20 | 5 | 18 | 16,80% | 0,2764 |
| 50 | 5 | 9 | 19,10% | 0,2471 |

A seleção Leaf produziu agrupamentos menores e mais específicos, porém com cobertura reduzida.

Os artigos marcados como `-1` são classificados como ruído pelo algoritmo.

**A classificação como ruído não significa que o artigo não possui relevância científica.**

As Silhouettes do HDBSCAN Leaf foram calculadas somente sobre os artigos agrupados, não sendo diretamente comparáveis às Silhouettes das partições completas.

#### 10.4. Comparação dos métodos

| Critério | K-Means | Agglomerative | HDBSCAN |
|---|---|---|---|
| Exige número de clusters | Sim | Sim | Não |
| Partição completa | Sim | Sim | Não necessariamente |
| Identifica ruído | Não | Não | Sim |
| Depende de densidade | Não | Não | Sim |
| Sensível a parâmetros | Sim | Sim | Sim |
| Aplicação principal explorada | Organização global | Estrutura hierárquica | Núcleos densos |

O K-Means apresentou a maior Silhouette entre as principais partições completas avaliadas.

O Agglomerative Ward ofereceu uma alternativa hierárquica, com distribuição relativamente equilibrada.

O HDBSCAN identificou subconjuntos densos, mas apresentou cobertura reduzida nas configurações testadas.

Esses resultados não estabelecem superioridade universal de um algoritmo.

#### 10.5. Interpretação temática preliminar

A inspeção exploratória dos clusters K-Means com `k=8` indicou temas relacionados a:

- Manutenção preditiva industrial;
- Estruturas moleculares e criomicroscopia;
- Astronomia e classificação de objetos celestes;
- Controle, robótica e otimização;
- Genômica e biologia molecular;
- Modelagem e previsão climática;
- Diagnóstico por imagens médicas;
- Aplicações gerais de IA na saúde.

Um dos agrupamentos apresentou artigos provenientes das cinco áreas de amostragem, com predominância de Medicina.

A presença de múltiplas áreas no mesmo cluster não constitui, isoladamente, evidência suficiente de interdisciplinaridade.

Uma amostra exploratória de 25 artigos foi preparada para revisão humana independente.

A análise temática e a validação das possíveis conexões interdisciplinares serão aprofundadas na Issue #09.

#### 10.6. Implementação

Módulo reutilizável:

`src/research_atlas/clustering/algorithms.py`

Funções disponíveis:

- `validate_embeddings()`: valida a matriz de entrada;
- `run_kmeans()`: executa K-Means;
- `run_agglomerative()`: executa Agglomerative Clustering;
- `run_hdbscan()`: executa HDBSCAN;
- `summarize_clusters()`: resume clusters, cobertura e ruído.

Exemplo:

```python
import numpy as np

from research_atlas.clustering.algorithms import (
    run_kmeans,
    run_agglomerative,
    run_hdbscan,
    summarize_clusters,
)

embeddings = np.load(
    "data/processed/embeddings/"
    "bge-small-en-v1.5-v1/embeddings.npy",
    allow_pickle=False,
)

kmeans_labels, _ = run_kmeans(
    embeddings,
    k=8,
    seed=42,
)

ward_labels, _ = run_agglomerative(
    embeddings,
    k=8,
    linkage="ward",
    metric="euclidean",
)

hdbscan_labels, _ = run_hdbscan(
    embeddings,
    min_cluster_size=50,
    min_samples=5,
    selection_method="eom",
)

print(summarize_clusters(kmeans_labels))
print(summarize_clusters(ward_labels))
print(summarize_clusters(hdbscan_labels))
```

#### 10.7. Artefatos

```text
data/processed/clustering/v1/
├── cluster_assignments.parquet
├── kmeans_metrics.csv
├── kmeans_stability.csv
├── agglomerative_metrics.csv
├── hdbscan_eom_metrics.csv
├── hdbscan_leaf_metrics.csv
└── manifest.json
```

O arquivo `cluster_assignments.parquet` preserva os rótulos de 18 configurações experimentais para os 5.000 artigos.

Os arquivos CSV registram métricas e parâmetros das execuções.

O `manifest.json` contém informações de proveniência e hashes SHA-256.

Os testes de integração verificam:

- Presença e integridade dos artefatos;
- Correspondência com os embeddings;
- Ordem dos 5.000 artigos;
- Quantidade de clusters;
- Consistência dos rótulos;
- Tratamento do ruído;
- Correspondência com as métricas exportadas.

#### 10.8. Limitações

- A exploração de hiperparâmetros não foi exaustiva.
- As métricas internas não comprovam interdisciplinaridade.
- O K-Means requer definição prévia de `k`.
- O Agglomerative depende da estratégia de ligação.
- O HDBSCAN apresentou elevada proporção de ruído em várias configurações.
- A Silhouette calculada apenas sobre artigos agrupados deve ser interpretada considerando a cobertura.
- A estabilidade entre inicializações não substitui a avaliação sob reamostragem.
- A interpretação temática demanda avaliação humana.
- O desempenho computacional não foi comparado sistematicamente.

**Notebook:** `notebooks/02_clusterizacao.ipynb`

**Relatório:** [Comparação de algoritmos de clusterização](reports/experiments/clustering_comparison.md)

**Estado:** experimentos, implementação reutilizável, artefatos e testes concluídos na branch da Issue #07; integração à `dev` pendente.

## Comparação preliminar entre TF-IDF e BGE

As Issues #06 e #07 utilizaram os mesmos 5.000 artigos, permitindo uma comparação exploratória entre representações lexicais e semânticas.

| Representação | Algoritmo | k | Silhouette | ARI médio |
|---|---|---:|---:|---:|
| TF-IDF 10k | K-Means | 8 | 0,0146 | 0,6692 |
| BGE 384d | K-Means | 8 | 0,1472 | 0,8621 |

O BGE apresentou maior Silhouette e maior estabilidade entre inicializações nos experimentos registrados.

Entretanto, as representações ocupam espaços vetoriais diferentes e apresentam distribuições de distância distintas.

**Não é metodologicamente suficiente concluir que uma representação é superior apenas comparando os valores absolutos de Silhouette.**

Uma comparação mais aprofundada será realizada na Issue #08.

## Testes

Execute a suíte completa:

```bash
uv run pytest -q
```

Última execução local registrada:

```text
106 passed in 1.36s
```

### Testes do baseline TF-IDF

```bash
uv run pytest \
  tests/test_tfidf.py \
  tests/test_tfidf_artifacts.py -v
```

### Testes da clusterização BGE

```bash
uv run pytest \
  tests/test_clustering.py \
  tests/test_clustering_artifacts.py -v
```

A suíte contempla:

- Cliente OpenAlex;
- Pré-processamento;
- Amostragem;
- Validação de dados;
- Embeddings;
- Baseline TF-IDF;
- K-Means;
- Agglomerative Clustering;
- HDBSCAN;
- Integridade e consistência dos artefatos.

Os testes automatizados não dependem de novas chamadas reais à API OpenAlex.

## Interface

Execute:

```bash
uv run streamlit run app/Home.py
```

Acesse:

http://localhost:8501

A interface científica completa, os dashboards interativos e as visualizações de comunidades ainda estão em desenvolvimento.

## Reprodutibilidade

As etapas experimentais utilizam versões identificáveis dos dados e parâmetros explícitos.

### Dataset

O dataset v1 possui 5.000 artigos selecionados, identificados por SHA-256.

A ordem dos registros é preservada para garantir a correspondência entre textos, embeddings e agrupamentos.

### Embeddings

A representação BGE possui:

- Modelo identificado;
- 384 dimensões;
- Normalização L2;
- Estratégia de segmentação documentada;
- Manifesto;
- Hash SHA-256.

### Baseline TF-IDF

Os experimentos preservam:

- Parâmetros do vetorizador;
- Sementes;
- Quantidade de clusters;
- Métricas;
- Rótulos das configurações selecionadas;
- Manifesto e checksums.

As matrizes TF-IDF são mantidas em formato esparso CSR.

### Clusterização semântica

Os experimentos da Issue #07 preservam:

- Identificação dos embeddings;
- Parâmetros dos algoritmos;
- Configurações avaliadas;
- Rótulos por artigo;
- Métricas;
- Cobertura e ruído;
- Manifesto e checksums.

Os notebooks documentam as execuções experimentais realizadas no Google Colab.

### Verificação de integridade

É possível verificar a versão do dataset com:

```bash
sha256sum data/processed/dataset_selected.parquet
```

E a versão dos embeddings com:

```bash
sha256sum \
  data/processed/embeddings/\
bge-small-en-v1.5-v1/embeddings.npy
```

Novas coletas, mudanças no pré-processamento, alterações de dependências e diferentes configurações de execução podem produzir resultados distintos.

## Limitações científicas

O ResearchAtlas é uma investigação exploratória baseada em aprendizado não supervisionado.

As seguintes limitações devem ser consideradas:

1. A categorização temática da OpenAlex pode apresentar imprecisões.
2. A presença de termos relacionados à IA não garante aplicação efetiva de IA.
3. A auditoria humana de relevância científica permanece pendente.
4. O dataset é artificialmente balanceado por área e ano.
5. Alertas de qualidade textual não foram utilizados como critérios automáticos de exclusão.
6. O TF-IDF depende de características lexicais e pode agrupar artigos por vocabulário metodológico compartilhado.
7. Embeddings semânticos podem preservar similaridades não relacionadas à interdisciplinaridade.
8. Algoritmos de clusterização apresentam comportamentos distintos diante de parâmetros e distribuições dos dados.
9. A presença de artigos de diferentes áreas em um cluster não comprova interdisciplinaridade.
10. A estabilidade entre inicializações não garante estabilidade sob reamostragem.
11. Métricas internas não substituem inspeção científica ou validação humana.
12. A busca de hiperparâmetros realizada não foi exaustiva.

As conclusões devem ser interpretadas como hipóteses e evidências exploratórias, não como demonstrações definitivas de comunidades científicas interdisciplinares.

## Próximas etapas

### Issue #08 — Avaliação de métricas e estabilidade

Consolidar e comparar:

- Silhouette;
- Davies-Bouldin;
- Calinski-Harabasz;
- Adjusted Rand Index;
- Distribuição dos clusters;
- Cobertura e ruído;
- Estabilidade;
- Sensibilidade aos hiperparâmetros.

O objetivo é estabelecer uma análise metodológica consistente e justificar a seleção das configurações que serão utilizadas nas etapas seguintes.

### Issue #09 — Interpretação e validação interdisciplinar

Investigar:

- Termos e conceitos representativos;
- Artigos próximos aos centroides;
- Distribuição por área científica;
- Composição dos clusters;
- Possíveis pontes entre áreas;
- Amostras para auditoria humana;
- Concordância entre revisores;
- Evidências e limitações das possíveis conexões interdisciplinares.

### Etapas posteriores

- Visualização de embeddings em baixa dimensão;
- Construção de grafos de similaridade;
- Análise de comunidades;
- Investigação temporal;
- Desenvolvimento de dashboards interativos;
- Integração das descobertas à interface Streamlit;
- Consolidação do relatório científico final.

## Organização do desenvolvimento

O projeto utiliza Git e GitHub com branches por issue.

### Branches

- `main`: versão estável;
- `dev`: integração das funcionalidades;
- Branches específicas para desenvolvimento das issues.

### Commits

As mensagens seguem o padrão Conventional Commits, com descrições em português.

Exemplos:

```text
feat: implementa algoritmos de clusterizacao semantica

data: adiciona resultados de referencia da clusterizacao

docs: documenta comparacao dos algoritmos de clusterizacao
```

### Fluxo de desenvolvimento

1. Criar branch a partir de `dev`.
2. Implementar e testar.
3. Versionar artefatos de referência necessários.
4. Documentar os experimentos.
5. Executar a suíte completa de testes.
6. Abrir Pull Request para `dev`.
7. Revisar e integrar.
8. Manter `main` como referência estável.

## Fonte científica

**OpenAlex:** https://openalex.org/

Os dados são utilizados conforme as condições de acesso e o protocolo científico documentado no repositório.

---

**Status:** dataset v1, embeddings BGE, baseline TF-IDF + K-Means e clusterização semântica implementados. As próximas etapas concentram-se na avaliação aprofundada dos agrupamentos, validação temática e identificação de possíveis conexões interdisciplinares.
