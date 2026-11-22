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

O dataset v1, a representação semântica BGE e o baseline lexical TF-IDF + K-Means estão preparados. A interpretação científica dos agrupamentos e a auditoria humana ainda não foram concluídas.

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

O arquivo de referência dos artigos é `data/processed/dataset_selected.parquet`.

**SHA-256 do dataset selecionado v1:**

```text
533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389
```

O hash identifica a versão exata do arquivo usada nos experimentos; uma nova coleta na OpenAlex não é necessariamente idêntica.

## Tecnologias

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3.12 |
| Ambiente e dependências | uv |
| Fonte de dados | OpenAlex API |
| HTTP | HTTPX |
| Dados e armazenamento | pandas, NumPy, PyArrow, Parquet, JSON |
| Machine Learning implementado | scikit-learn (TF-IDF, K-Means), BGE-small-en-v1.5 via Sentence Transformers no Colab |
| Experimentação exploratória de clusterização | Agglomerative Clustering, HDBSCAN |
| Visualização e interface | Plotly, Streamlit, Matplotlib nos notebooks |
| Testes | pytest |
| Planejado | UMAP, análise de grafos e detecção de comunidades |

As dependências exigidas por notebooks experimentais podem ser instaladas no respectivo ambiente Colab e não precisam integrar as dependências de execução da aplicação.

## Instalação

### Pré-requisitos

- Git, uv e Python 3.12.
- Internet e chave da OpenAlex **apenas para executar novas consultas à API**.

```bash
git clone https://github.com/Sgnsilv/ResearchAtlas.git
cd ResearchAtlas
uv python install 3.12
uv sync --dev
```

O uv utiliza as dependências declaradas em `pyproject.toml` e resolvidas em `uv.lock`.

### Configuração da OpenAlex

Obtenha uma chave em https://openalex.org/settings/api e crie o arquivo local:

```bash
cp .env.example .env
```

```dotenv
OPENALEX_API_KEY=sua_chave_aqui
```

Não versione `.env` nem publique chaves. Os comandos que consultam a API utilizam `uv run --env-file .env`.

## Estrutura do projeto

```text
ResearchAtlas/
├── app/
│   └── Home.py
├── src/research_atlas/
│   ├── data/
│   │   ├── openalex.py
│   │   ├── preprocessing.py
│   │   ├── pilot.py
│   │   └── dataset.py
│   ├── analysis/
│   │   ├── eda.py
│   │   └── audit.py
│   └── features/
│       ├── embeddings.py
│       └── tfidf.py
├── notebooks/
│   ├── 01_embeddings_semanticos.ipynb
│   ├── 02_clusterizacao.ipynb
│   └── 03_baseline_tfidf_kmeans.ipynb
├── data/
│   ├── raw/openalex_cache/
│   └── processed/
│       ├── dataset_selected.parquet
│       ├── dataset_report.json
│       ├── embeddings/bge-small-en-v1.5-v1/
│       │   ├── embeddings.npy
│       │   ├── metadata.parquet
│       │   └── manifest.json
│       └── tfidf_baseline/v1/
│           ├── tfidf_assignments.parquet
│           ├── kmeans_k_sensitivity.csv
│           ├── tfidf_parameter_sensitivity.csv
│           ├── tfidf_stability.csv
│           ├── tfidf_final_candidates.csv
│           └── manifest.json
├── reports/
│   ├── eda/
│   └── experiments/
│       ├── pilot_dataset.md
│       ├── final_dataset.md
│       ├── semantic_embeddings.md
│       └── tfidf_baseline.md
├── docs/dataset.md
├── tests/
│   ├── test_embeddings.py
│   ├── test_embedding_artifacts.py
│   ├── test_tfidf.py
│   └── test_tfidf_artifacts.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

A árvore destaca os arquivos principais; outros módulos, testes e diretórios não estão enumerados. O notebook `02_clusterizacao.ipynb` registra investigações exploratórias associadas ao escopo das Issues #07–#09 e deve ser separado da entrega final da Issue #06 antes do merge.

**Versionamento dos dados:** o `.gitignore` ignora artefatos gerados por padrão. Alguns arquivos de referência são adicionados explicitamente ao Git para reprodução; caches, candidatos e saídas intermediárias não são necessariamente versionados. Os artefatos TF-IDF exportados localmente na branch #06 precisam ser adicionados explicitamente ao commit.

## Funcionalidades implementadas

### 1. Protocolo científico

`docs/dataset.md` documenta pergunta científica, critérios de seleção, áreas, filtros da OpenAlex e decisões metodológicas.

### 2. Cliente OpenAlex

`src/research_atlas/data/openalex.py`: autenticação por ambiente, recuperação filtrada, consultas por subárea e termo, cursor, amostragem com semente e retentativas de falhas HTTP.

### 3. Pré-processamento

`src/research_atlas/data/preprocessing.py`: reconstrução de resumos, normalização de identificadores e texto, validação estrutural, deduplicação e sinalização de qualidade textual.

### 4. Dataset piloto

```bash
PYTHONPATH=src uv run --env-file .env python -m research_atlas.data.pilot
```

A execução documentada recuperou 400 registros brutos, 398 artigos únicos e 13 sinalizados para revisão textual. O piloto evidenciou possíveis problemas nos resumos e viés temporal de consultas por relevância.

Relatório: `reports/experiments/pilot_dataset.md`.

### 5. Análise exploratória

```bash
PYTHONPATH=src uv run python -m research_atlas.analysis.eda
```

`src/research_atlas/analysis/eda.py` gera gráficos de distribuição por ano, área, tamanho dos resumos e qualidade textual. A análise original foi aplicada ao piloto.

### 6. Amostragem para auditoria humana

```bash
PYTHONPATH=src uv run python -m research_atlas.analysis.audit
```

`src/research_atlas/analysis/audit.py` prepara dois CSVs independentes, contendo a mesma amostra de 100 artigos (20 por área), para revisão de relevância de IA, pertinência científica e qualidade do resumo.

**Estado:** amostra preparada, revisão humana pendente.

### 7. Dataset ampliado

`src/research_atlas/data/dataset.py` integra consultas amostradas por área/ano/termo, sementes determinísticas, cache, validação, deduplicação, seleção estratificada, relatórios e exportação.

```bash
PYTHONPATH=src uv run --env-file .env python -m research_atlas.data.dataset \
  --sample-size 100 \
  --target-per-area 1000 \
  --seed 42 \
  --rounds 1
```

A execução de referência produziu 11.210 candidatos válidos e selecionou 5.000 artigos únicos, sem déficit nos 30 estratos de área × ano.

| Área de amostragem | Artigos |
|---|---:|
| Medicina | 1.000 |
| Astronomia | 1.000 |
| Biologia | 1.000 |
| Clima | 1.000 |
| Engenharia | 1.000 |
| **Total** | **5.000** |

Cada estrato tem aproximadamente 166 ou 167 artigos. Essa amostra deliberadamente balanceada **não representa a distribuição natural das publicações**.

#### Qualidade textual

| Indicador | Artigos |
|---|---:|
| Resumo com menos de 50 palavras | 72 |
| Resumo com mais de 500 palavras | 122 |
| Conteúdo suspeito | 8 |
| Pelo menos um alerta | 195 |

Os alertas podem se sobrepor; não implicam exclusão automática.

| Área | Artigos sinalizados |
|---|---:|
| Medicina | 55 |
| Astronomia | 22 |
| Biologia | 81 |
| Clima | 21 |
| Engenharia | 16 |

O cache das respostas fica em `data/raw/openalex_cache/`. Uma nova coleta sem cache pode retornar resultados diferentes devido à evolução da OpenAlex.

Relatório: `reports/experiments/final_dataset.md`.

### 8. Embeddings semânticos — Issue #05

Utilizamos `BAAI/bge-small-en-v1.5` para representar cada um dos 5.000 artigos em 384 dimensões.

| Parâmetro | Valor |
|---|---|
| Entrada | Título + resumo |
| Limite do modelo | 512 tokens |
| Janela | 510 tokens de conteúdo + tokens especiais |
| Sobreposição | 64 tokens |
| Agregação | Média ponderada dos trechos e normalização L2 |
| Ambiente experimental | Google Colab, GPU Tesla T4 |

Foram processados 5.678 trechos, gerando 5.000 vetores normalizados. A implementação reutilizável de segmentação/agregação está em `src/research_atlas/features/embeddings.py`.

Artefatos de referência:

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

```python
import numpy as np
import pandas as pd

base = "data/processed/embeddings/bge-small-en-v1.5-v1"
embeddings = np.load(f"{base}/embeddings.npy", allow_pickle=False)
metadata = pd.read_parquet(f"{base}/metadata.parquet")

print(embeddings.shape)  # (5000, 384)
print(metadata.shape)    # (5000, 9)
```

A coluna `embedding_index` preserva a correspondência entre vetores e artigos.

Notebook: `notebooks/01_embeddings_semanticos.ipynb`.

Relatório: `reports/experiments/semantic_embeddings.md`.

### 9. Baseline TF-IDF + K-Means — Issue #06

Implementamos um baseline lexical de clusterização sobre os mesmos 5.000 artigos. A entrada é a concatenação de título e resumo, sem uso das áreas científicas como rótulos de treinamento.

**Configuração de referência:**

| Parâmetro | Valor |
|---|---|
| Vetorização | TF-IDF, norma L2 |
| Vocabulário máximo | 10.000 características |
| ngram_range | (1, 2) |
| min_df / max_df | 3 / 0,95 |
| stop_words | english |
| sublinear_tf | True |
| Algoritmo | K-Means |
| k | 8 |
| random_state | 42 |
| n_init | 10 |
| Silhouette | Distância cosseno, amostra de 2.000, seed 42 |

O baseline inicial com 20.000 características produziu uma matriz CSR de **5.000 × 20.000**, com **648.816 valores não nulos** (densidade de **0,6488%**).

#### Experimentos e resultados

- Sensibilidade a `k`: 2 a 15, com a representação inicial de 20.000 características.
- Sensibilidade do TF-IDF: baseline, apenas unigramas, vocabulário de 10.000 características e `min_df=5` (com `k=8`).
- Estabilidade entre inicializações: sementes 11, 21, 42, 73 e 101, para `k=5`, `k=8` e `k=13` usando TF-IDF 10k.

| TF-IDF 10k | Silhouette (cosseno) | ARI médio | Maior cluster |
|---|---:|---:|---:|
| k=5 | 0,0130 | 0,6578 | 38,64% |
| **k=8 (referência)** | **0,0146** | **0,6692** | **19,52%** |
| k=13 | 0,0156 | 0,5044 | 13,68% |

`k=8` foi selecionado como compromisso entre estabilidade, separação e distribuição dos grupos, não como número verdadeiro de comunidades. A Silhouette é baixa em todas as configurações analisadas.

#### Interpretação exploratória dos oito clusters

| Cluster | Artigos | Tema preliminar |
|---|---:|---|
| 0 | 772 | Controle, robótica e sistemas |
| 1 | 976 | Predição climática e meteorológica |
| 2 | 140 | COVID-19 e diagnóstico |
| 3 | 424 | Diagnóstico por imagens e câncer |
| 4 | 554 | Aplicações gerais de IA na saúde |
| 5 | 883 | Deep Learning e análise de imagens |
| 6 | 631 | Galáxias e classificação astronômica |
| 7 | 620 | Genômica, proteínas e genética |

Os temas provêm de termos de maior peso nos centroides e da inspeção de títulos representativos. O cluster 5 reúne artigos de várias áreas que compartilham vocabulário ligado a imagens e redes neurais; isso sugere **similaridade metodológica**, não prova interdisciplinaridade.

#### Implementação e artefatos

- Notebook: `notebooks/03_baseline_tfidf_kmeans.ipynb`.
- Módulo reutilizável: `src/research_atlas/features/tfidf.py`.
- Testes: `tests/test_tfidf.py` e `tests/test_tfidf_artifacts.py`.
- Relatório: `reports/experiments/tfidf_baseline.md`.

```text
data/processed/tfidf_baseline/v1/
├── tfidf_assignments.parquet
├── kmeans_k_sensitivity.csv
├── tfidf_parameter_sensitivity.csv
├── tfidf_stability.csv
├── tfidf_final_candidates.csv
└── manifest.json
```

O manifesto registra parâmetros, versão do dataset e checksums dos artefatos. O Parquet contém os rótulos de seis configurações selecionadas (TF-IDF 10k e 20k, cada uma para `k=5`, `8` e `13`). As outras configurações constam nas tabelas de sensibilidade, sem necessariamente ter seus rótulos preservados.

#### Comparação preliminar com BGE

| Representação (k=8) | Silhouette | ARI médio entre inicializações |
|---|---:|---:|
| TF-IDF 10k | 0,0146 | 0,6692 |
| BGE, 384 dimensões | 0,1472 | 0,8621 |

Ambas utilizam os mesmos artigos, mas têm espaços vetoriais e distribuições de distância diferentes. **Não se deve inferir superioridade apenas da diferença entre Silhouettes**. O ARI mede estabilidade entre inicializações sobre o mesmo dataset, não estabilidade sob reamostragem.

**Estado da Issue #06:** experimentos, implementação reutilizável, testes e exportação dos resultados concluídos localmente; documentação e integração por PR dependem do commit/merge.

## Testes

Execute a suíte completa:

```bash
uv run pytest -q
```

Testes direcionados à Issue #06:

```bash
uv run pytest tests/test_tfidf.py tests/test_tfidf_artifacts.py -v
```

A suíte também contempla cliente OpenAlex, pré-processamento, amostragem, qualidade textual, embeddings e integridade de artefatos. Testes automatizados não devem depender de chamadas reais à API.

## Interface

```bash
uv run streamlit run app/Home.py
```

Acesse `http://localhost:8501`. Os dashboards científicos completos e a visualização interativa das comunidades são etapas futuras.

## Reprodutibilidade

- Dataset v1 identificado por SHA-256 e preservação da ordem dos artigos.
- Embeddings BGE identificados por hash e manifesto.
- Baseline TF-IDF identificado por parâmetros, sementes, rótulos, métricas e checksums.
- Matrizes TF-IDF mantidas esparsas (CSR), evitando conversão desnecessária para matriz densa.
- Experimentos no Colab documentados em notebooks próprios.
- Silhouette do baseline: cosseno, amostragem de 2.000 artigos com `random_state=42`.
- A estabilidade por ARI compara execuções com diferentes inicializações sobre os mesmos 5.000 artigos.

Verifique o dataset original com:

```bash
sha256sum data/processed/dataset_selected.parquet
```

Novas coletas, alterações no pré-processamento, mudanças de dependências ou outros métodos de amostragem podem produzir resultados diferentes.

## Limitações científicas

- A categorização temática da OpenAlex pode ser imprecisa.
- A presença de termos de IA não confirma aplicação efetiva de IA no artigo.
- A auditoria humana de relevância científica ainda está pendente.
- O dataset é artificialmente balanceado por área e ano.
- Alertas textuais não foram utilizados como critérios automáticos de exclusão.
- O TF-IDF depende do vocabulário e pode agrupar artigos por termos metodológicos compartilhados.
- Silhouette próxima de zero sugere baixa separação dos clusters TF-IDF segundo cosseno.
- Avaliar diferentes representações apenas pelos valores absolutos de Silhouette é insuficiente.
- A interpretação dos temas e a identificação de interdisciplinaridade exigem validação adicional.
- Os parâmetros testados não constituem uma busca exaustiva.

## Próximas etapas

1. Concluir a auditoria humana do dataset e definir o tratamento dos resumos sinalizados.
2. Integrar e revisar o baseline da Issue #06 na branch `dev`.
3. Organizar os experimentos com embeddings em sua própria issue e comparar K-Means, Agglomerative e HDBSCAN (**#07**).
4. Consolidar métricas, cobertura, ruído e estabilidade com critérios metodológicos consistentes (**#08**).
5. Investigar termos, artigos representativos e possíveis pontes interdisciplinares, com validação humana (**#09**).
6. Construir visualizações, dashboards, grafos de pesquisa e análises temporais.

## Organização do desenvolvimento

- `main`: versão estável.
- `dev`: integração das funcionalidades.
- Branches por issue, com alterações e commits de escopo delimitado.

Os commits seguem o padrão Conventional Commits, com mensagens em português.

## Fonte científica

[OpenAlex](https://openalex.org/)

Os dados são utilizados conforme as condições de acesso e o protocolo científico do projeto.

---

**Status:** dataset v1 e embeddings BGE disponíveis; baseline TF-IDF + K-Means implementado e validado localmente; auditoria científica e integração final pendentes.
