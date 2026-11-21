
# ResearchAtlas

**Exploração de comunidades científicas interdisciplinares por meio de clusterização semântica e análise de grafos.**

O ResearchAtlas é uma plataforma interativa desenvolvida em Python e Streamlit para investigar como a Inteligência Artificial é aplicada a diferentes áreas do conhecimento científico.

O projeto utiliza metadados de publicações da OpenAlex e técnicas de aprendizado não supervisionado para descobrir padrões semânticos, comunidades de pesquisa e possíveis conexões interdisciplinares.

> Projeto acadêmico da disciplina de Aprendizagem de Máquina, desenvolvido também como portfólio de Machine Learning.

## Objetivo

Investigar a seguinte pergunta científica:

**É possível identificar comunidades de pesquisa interdisciplinares a partir da similaridade semântica de artigos sobre aplicações de Inteligência Artificial e descobrir conexões entre áreas científicas distintas?**

### Áreas investigadas

- Medicina e Saúde
- Astronomia
- Biologia e Bioinformática
- Clima e Meio Ambiente
- Engenharia

O recorte considera artigos científicos publicados entre 2020 e 2025.

A meta para o dataset definitivo é de aproximadamente **5.000 artigos**, distribuídos entre as cinco áreas. Esse volume ainda não foi coletado.

## Tecnologias

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3.12 |
| Gerenciamento de dependências | uv |
| Interface | Streamlit |
| Visualização | Plotly |
| Fonte de dados | OpenAlex API |
| Manipulação de dados | pandas, NumPy, PyArrow |
| Requisições HTTP | HTTPX |
| Testes | pytest |
| Machine Learning planejado | scikit-learn, Sentence Transformers, HDBSCAN |
| Redução de dimensionalidade planejada | UMAP |
| Análise de grafos planejada | NetworkX, Leiden |
| Ambiente experimental planejado | Google Colab |

As bibliotecas destinadas aos experimentos de Machine Learning serão adicionadas conforme a implementação.

## Instalação

### Pré-requisitos

- Git
- uv
- Acesso à internet
- Chave de API da OpenAlex

### 1. Clonar o repositório

```bash
git clone https://github.com/Sgnsilv/ResearchAtlas.git
cd ResearchAtlas
```

### 2. Instalar Python e dependências

```bash
uv python install 3.12
uv sync --dev
```

O uv gerencia o ambiente virtual `.venv` e instala as dependências do projeto.

Não é necessário ativar manualmente o ambiente virtual ao utilizar `uv run`.

### 3. Configurar a OpenAlex

Obtenha sua chave de API em:

https://openalex.org/settings/api

Na raiz do projeto:

```bash
cp .env.example .env
```

Configure o arquivo `.env`:

```dotenv
OPENALEX_API_KEY=sua_chave_aqui
```

O cliente OpenAlex lê essa variável do ambiente. Os comandos que acessam a API utilizam `uv run --env-file .env`.

**Nunca envie a chave de API ao GitHub.**

## Estrutura do projeto

```text
ResearchAtlas/
├── app/
│   └── Home.py
├── src/
│   └── research_atlas/
│       ├── data/
│       │   ├── openalex.py
│       │   ├── preprocessing.py
│       │   └── pilot.py
│       ├── analysis/
│       │   ├── eda.py
│       │   └── audit.py
│       ├── features/
│       ├── clustering/
│       ├── evaluation/
│       └── visualization/
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── eda/
│   ├── figures/
│   └── experiments/
│       └── pilot_dataset.md
├── docs/
│   └── dataset.md
├── tests/
│   ├── test_openalex.py
│   ├── test_pilot.py
│   └── test_audit.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

Alguns diretórios estão reservados para funcionalidades futuras.

Datasets, modelos e artefatos gerados automaticamente não devem ser versionados diretamente no Git.

## Funcionalidades implementadas

### 1. Protocolo científico

O protocolo de construção do dataset está documentado em `docs/dataset.md`.

Ele estabelece o recorte temporal, as áreas científicas, os critérios de seleção e a estratégia de coleta.

### 2. Cliente OpenAlex

O módulo `src/research_atlas/data/openalex.py` oferece:

- Consultas à API de artigos científicos.
- Filtros por subárea científica e termo de busca.
- Recuperação paginada.
- Tratamento de falhas temporárias.
- Retentativas para erros HTTP recuperáveis.
- Autenticação por variável de ambiente.

### 3. Pré-processamento

O módulo `src/research_atlas/data/preprocessing.py` implementa:

- Reconstrução de resumos em índice invertido.
- Normalização dos identificadores OpenAlex.
- Normalização textual.
- Validação estrutural dos artigos.
- Deduplicação e preservação de metadados.
- Identificação de possíveis problemas de qualidade textual.

A validação estrutural e a avaliação de qualidade textual são procedimentos separados.

### 4. Construção do dataset piloto

O módulo `src/research_atlas/data/pilot.py` integra coleta, normalização, validação e avaliação textual.

Para executar:

```bash
uv run --env-file .env python -m research_atlas.data.pilot
```

O processo gera:

```text
data/processed/
├── pilot_articles.parquet
└── pilot_report.json
```

O arquivo Parquet contém os artigos processados e indicadores de qualidade textual.

O arquivo JSON registra estatísticas da coleta, deduplicação, validação e alertas.

#### Resultados da execução piloto

| Indicador | Resultado |
|---|---:|
| Registros brutos | 400 |
| Artigos únicos | 398 |
| Ocorrências duplicadas | 2 |
| Artigos estruturalmente válidos | 398 |
| Artigos estruturalmente inválidos | 0 |
| Artigos sinalizados para revisão textual | 13 |

A seleção recupera os primeiros resultados de cada consulta, ordenados pela relevância da fonte. Portanto, o dataset piloto não deve ser considerado uma amostra representativa da literatura científica.

Os valores correspondem à execução documentada e podem variar em novas consultas.

### 5. Análise exploratória de dados

O módulo `src/research_atlas/analysis/eda.py` permite analisar:

- Distribuição de publicações por ano.
- Distribuição de artigos por área de coleta.
- Comprimento dos resumos.
- Frequência de alertas de qualidade textual.
- Registros recuperados por múltiplas áreas.

Para executar:

```bash
uv run python -m research_atlas.analysis.eda
```

Os resultados são gerados em:

```text
reports/eda/
├── 01_publicacoes_por_ano.html
├── 02_artigos_por_area.html
├── 03_tamanho_resumos.html
├── 04_qualidade_textual.html
└── eda_summary.json
```

Os gráficos HTML são interativos e podem ser abertos no navegador.

Na execução piloto, observou-se concentração de publicações nos anos de 2020 e 2021, indicando a necessidade de uma estratégia mais equilibrada para a coleta definitiva.

### 6. Auditoria científica

O módulo `src/research_atlas/analysis/audit.py` gera uma amostra reproduzível para revisão humana.

Para executar:

```bash
uv run python -m research_atlas.analysis.audit
```

São gerados dois arquivos:

```text
data/processed/audit/
├── reviewer_1.csv
└── reviewer_2.csv
```

A amostra contém 100 registros, sendo 20 artigos por área.

Cada avaliadora deve analisar independentemente:

- A relevância científica do artigo.
- A adequação do resumo para análise semântica.
- Os casos duvidosos ou inadequados.

Após o preenchimento das planilhas, serão avaliadas a concordância entre as classificações e as divergências, incluindo o coeficiente Kappa de Cohen.

**Estado atual:** amostra gerada; avaliação humana pendente.

## Qualidade dos dados

A qualidade do dataset é avaliada em duas dimensões.

### Validação estrutural

Verifica identificador, título, resumo, período, idioma, tipo de publicação, retratação e conflitos de metadados.

### Qualidade textual

Sinaliza possíveis problemas nos resumos:

| Indicador | Condição |
|---|---|
| `short_abstract` | Menos de 50 palavras |
| `long_abstract` | Mais de 500 palavras |
| `suspicious_content` | Presença de padrões textuais suspeitos |

Os alertas não provocam exclusão automática de registros.

A relevância científica deverá ser verificada separadamente por auditoria humana.

## Resultados exploratórios

A execução piloto recuperou 398 artigos únicos:

| Área | Artigos |
|---|---:|
| Medicina | 80 |
| Astronomia | 80 |
| Biologia | 80 |
| Clima | 80 |
| Engenharia | 78 |

A mediana dos resumos foi de 194,5 palavras.

Foram identificados 13 artigos com pelo menos um alerta de qualidade textual.

Nenhum artigo foi recuperado por múltiplas áreas de coleta. Esse resultado não demonstra ausência de interdisciplinaridade científica, que será investigada posteriormente por técnicas semânticas.

O relatório metodológico completo está disponível em:

`reports/experiments/pilot_dataset.md`

## Executando a aplicação

Para iniciar a interface Streamlit:

```bash
uv run streamlit run app/Home.py
```

Acesse:

http://localhost:8501

A interface apresenta a página inicial do projeto. Os dashboards científicos ainda serão desenvolvidos.

## Executando os testes

Para executar a suíte completa:

```bash
uv run pytest -v
```

A execução de referência da Issue #04 concluiu **39 testes com sucesso**.

Os testes verificam as funcionalidades de coleta, pré-processamento, indicadores de qualidade textual e amostragem científica, sem necessidade de realizar novas consultas à API.

## Reprodutibilidade

Para reproduzir as etapas implementadas:

```bash
# 1. Instalar dependências
uv sync --dev

# 2. Construir o dataset piloto
uv run --env-file .env python -m research_atlas.data.pilot

# 3. Gerar a análise exploratória
uv run python -m research_atlas.analysis.eda

# 4. Preparar a auditoria
uv run python -m research_atlas.analysis.audit

# 5. Executar os testes
uv run pytest -v
```

A amostragem para auditoria utiliza uma semente fixa para garantir reprodutibilidade sobre um mesmo conjunto de entrada.

Entretanto, novas consultas à OpenAlex podem recuperar resultados diferentes caso os registros ou a ordenação da fonte sejam atualizados.

Os experimentos mais pesados serão realizados preferencialmente no Google Colab. Os artefatos serão exportados para a aplicação Streamlit, evitando processamento intensivo durante a utilização do dashboard.

## Próximas etapas

- Concluir a auditoria humana de relevância científica.
- Implementar a coleta definitiva, estratificada por área e ano.
- Construir o dataset científico de aproximadamente 5.000 artigos.
- Implementar representações TF-IDF e embeddings semânticos.
- Desenvolver o baseline TF-IDF + K-Means.
- Comparar algoritmos como HDBSCAN e Agglomerative Clustering.
- Avaliar qualidade, estabilidade e interpretação dos clusters.
- Visualizar os agrupamentos usando redução de dimensionalidade.
- Desenvolver dashboards interativos em Streamlit.
- Explorar redes de citações e comunidades científicas.
- Investigar a evolução temporal dos agrupamentos.

## Organização do desenvolvimento

O projeto utiliza:

- `main`: versão estável.
- `dev`: integração das funcionalidades.
- Branches específicas por issue, quando necessário.

Os commits seguem Conventional Commits, com descrições em português.

Cada issue deve incluir testes e atualização da documentação correspondente.

## Fonte de dados

[OpenAlex](https://openalex.org/)

Os dados são obtidos respeitando os termos de utilização da fonte e os critérios científicos documentados no projeto.

---

**Status:** em desenvolvimento — pipeline piloto, EDA e amostragem de auditoria implementados.
