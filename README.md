
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

O recorte inicial considera aproximadamente 5.000 artigos científicos publicados entre 2020 e 2025.

Esse volume é uma meta de coleta, não um dataset já disponível.

## Tecnologias

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3.12 |
| Gerenciamento de dependências | uv |
| Interface | Streamlit, Plotly |
| Fonte de dados | OpenAlex API |
| Manipulação de dados | pandas, NumPy, PyArrow |
| Requisições HTTP | HTTPX |
| Testes | pytest |
| Machine Learning planejado | scikit-learn, Sentence Transformers, HDBSCAN |
| Visualização planejada | UMAP |
| Análise de grafos planejada | NetworkX, Leiden |
| Ambiente experimental planejado | Google Colab |

As bibliotecas de Machine Learning serão adicionadas conforme a implementação dos experimentos.

## Instalação

### Pré-requisitos

- Git
- uv
- Acesso à internet
- Chave de API da OpenAlex para consultas paginadas

### 1. Clonar o repositório

```bash
git clone https://github.com/Sgnsilv/ResearchAtlas.git
cd ResearchAtlas
```

### 2. Instalar o Python e as dependências

O projeto utiliza Python 3.12.

```bash
uv python install 3.12
uv sync --dev
```

O uv gerencia o ambiente virtual `.venv` e instala as dependências registradas no `pyproject.toml` e no `uv.lock`.

Não é necessário ativar manualmente o ambiente virtual para utilizar `uv run`.

### 3. Configurar a API OpenAlex

Obtenha sua chave em:

https://openalex.org/settings/api

Na raiz do projeto, copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Abra `.env` e configure:

```dotenv
OPENALEX_API_KEY=sua_chave_aqui
```

**Importante:** nunca envie sua chave para o GitHub. O arquivo `.env` está incluído no `.gitignore`.

O cliente OpenAlex lê a variável `OPENALEX_API_KEY` do ambiente. Para carregar o arquivo `.env` nos comandos, utilize `uv run --env-file .env`.

## Executando a aplicação

Para iniciar o Streamlit:

```bash
uv run streamlit run app/Home.py
```

A aplicação estará disponível em:

http://localhost:8501

Atualmente, a interface apresenta a página inicial do projeto. Os dashboards científicos serão desenvolvidos nas próximas issues.

## Executando os testes

Para executar todos os testes automatizados:

```bash
uv run pytest -v
```

Os testes utilizam dados simulados e mocks para validar a lógica sem depender de requisições reais à OpenAlex.

## Estrutura do projeto

```text
ResearchAtlas/
├── app/                     # Aplicação Streamlit
│   └── Home.py
├── src/
│   └── research_atlas/
│       ├── data/            # Coleta e pré-processamento
│       ├── features/        # Representações e embeddings
│       ├── clustering/      # Algoritmos de clusterização
│       ├── evaluation/      # Métricas e avaliação
│       └── visualization/   # Visualizações
├── notebooks/               # Experimentos no Google Colab
├── data/
│   ├── raw/                 # Dados brutos
│   └── processed/           # Dados processados
├── models/                  # Artefatos de modelos
├── reports/
│   ├── figures/             # Gráficos e figuras
│   └── experiments/         # Resultados experimentais
├── docs/                    # Documentação científica e técnica
├── tests/                   # Testes automatizados
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

Alguns diretórios estão reservados para funcionalidades futuras.

Datasets completos, embeddings e modelos pesados não devem ser adicionados diretamente ao histórico Git.

## Próximas etapas

- Construção e análise exploratória do dataset piloto.
- Geração de embeddings semânticos.
- Implementação do baseline TF-IDF + K-Means.
- Comparação com HDBSCAN e Agglomerative Clustering.
- Avaliação e interpretação dos agrupamentos.
- Desenvolvimento dos dashboards no Streamlit.
- Análise de comunidades em grafos de citações.
- Estudo da evolução temporal das pesquisas.

## Metodologia e reprodutibilidade

Os experimentos serão realizados preferencialmente no Google Colab, utilizando GPU quando necessário. Os resultados serão exportados para utilização na aplicação Streamlit, evitando a execução de treinamentos pesados durante o acesso ao dashboard. As decisões metodológicas, os parâmetros experimentais, as métricas e as limitações serão documentados ao longo do desenvolvimento.

## Organização do desenvolvimento

O projeto utiliza as seguintes branches:

- `main`: versão estável.
- `dev`: integração das funcionalidades.
- Branches por issue: desenvolvimento de funcionalidades específicas.

Os commits seguem o padrão Conventional Commits, com descrições em português.

Cada issue deve incluir testes e atualização da documentação correspondente antes de ser considerada concluída.

## Fonte de dados

[OpenAlex](https://openalex.org/)

Os dados científicos serão obtidos respeitando os termos de utilização da fonte e os critérios de seleção documentados no projeto.

---

**Status:** em desenvolvimento.
