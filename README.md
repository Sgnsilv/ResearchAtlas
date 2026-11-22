
# ResearchAtlas

**Exploração de comunidades científicas interdisciplinares por meio de clusterização semântica e análise de grafos.**

O ResearchAtlas é um projeto de pesquisa e uma plataforma interativa desenvolvida em Python e Streamlit para investigar como a Inteligência Artificial é aplicada a diferentes áreas do conhecimento científico.

Utilizando metadados da OpenAlex e técnicas de aprendizado não supervisionado, buscamos identificar padrões semânticos, comunidades de pesquisa e possíveis conexões interdisciplinares.

> Projeto acadêmico de Aprendizagem de Máquina, desenvolvido também como portfólio técnico e científico.

## Pergunta científica

**É possível identificar comunidades de pesquisa interdisciplinares a partir da similaridade semântica de artigos sobre aplicações de Inteligência Artificial e descobrir conexões entre áreas científicas distintas?**

### Áreas investigadas

- Medicina e Saúde
- Astronomia
- Biologia e Bioinformática
- Clima e Meio Ambiente
- Engenharia

O período de investigação é de **2020 a 2025**.

## Estado atual

A etapa de coleta ampliada está concluída.

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

**Importante:** o conjunto foi validado quantitativa e estruturalmente. A auditoria de relevância científica ainda está pendente.

O dataset de referência é `data/processed/dataset_selected.parquet`.

### Identificação da versão

**Dataset selecionado v1**

SHA-256:

```text
533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389
```

A identificação acima corresponde ao arquivo efetivamente produzido na execução documentada.

## Tecnologias

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3.12 |
| Dependências e ambiente | uv |
| Fonte científica | OpenAlex API |
| Consultas HTTP | HTTPX |
| Manipulação de dados | pandas, NumPy, PyArrow |
| Armazenamento | Parquet, JSON |
| Visualizações | Plotly |
| Interface | Streamlit |
| Testes | pytest |
| Machine Learning planejado | scikit-learn, Sentence Transformers, HDBSCAN |
| Redução de dimensionalidade planejada | UMAP |
| Grafos planejados | NetworkX, Leiden |
| Experimentos pesados planejados | Google Colab |

As dependências de Machine Learning serão adicionadas conforme a implementação da pipeline experimental.

## Instalação

### Pré-requisitos

- Git
- uv
- Python 3.12
- Acesso à internet
- Chave de API da OpenAlex

### Clonar o repositório

```bash
git clone https://github.com/Sgnsilv/ResearchAtlas.git
cd ResearchAtlas
```

### Instalar as dependências

```bash
uv python install 3.12
uv sync --dev
```

O uv configura o ambiente virtual e instala as dependências presentes no `pyproject.toml` e no `uv.lock`.

### Configurar a OpenAlex

Obtenha uma chave em:

https://openalex.org/settings/api

Crie o arquivo de configuração:

```bash
cp .env.example .env
```

Defina a variável:

```dotenv
OPENALEX_API_KEY=sua_chave_aqui
```

Nunca compartilhe sua chave ou versione o arquivo `.env`.

Os comandos que consultam a API utilizam `uv run --env-file .env`.

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
│       │   ├── pilot.py
│       │   └── dataset.py
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
│   │   └── openalex_cache/
│   └── processed/
│       ├── pilot_articles.parquet
│       ├── pilot_report.json
│       ├── dataset_candidates.parquet
│       ├── dataset_selected.parquet
│       ├── dataset_report.json
│       └── audit/
├── models/
├── reports/
│   ├── eda/
│   ├── figures/
│   └── experiments/
│       ├── pilot_dataset.md
│       └── final_dataset.md
├── docs/
│   └── dataset.md
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

Alguns diretórios são reservados para futuras funcionalidades.

Os arquivos de dados e artefatos gerados são locais e não integram o repositório Git.

## Funcionalidades implementadas

### 1. Protocolo científico

O arquivo `docs/dataset.md` documenta a pergunta científica, os critérios de seleção, as áreas investigadas, os filtros da OpenAlex e as decisões metodológicas.

### 2. Cliente OpenAlex

`src/research_atlas/data/openalex.py`

Funcionalidades:

- Autenticação por variável de ambiente.
- Recuperação filtrada de artigos científicos.
- Consultas por subárea e termo de busca.
- Paginação por cursor.
- Amostragem aleatória com semente.
- Tratamento e retentativas de falhas HTTP temporárias.

### 3. Pré-processamento

`src/research_atlas/data/preprocessing.py`

Funcionalidades:

- Reconstrução dos resumos da OpenAlex.
- Normalização de texto e identificadores.
- Validação estrutural.
- Deduplicação por OpenAlex ID.
- Preservação de áreas e termos de recuperação.
- Identificação de possíveis problemas de qualidade textual.

### 4. Dataset piloto

`src/research_atlas/data/pilot.py`

Executar:

```bash
PYTHONPATH=src uv run --env-file .env python -m research_atlas.data.pilot
```

A execução documentada recuperou:

- 400 registros brutos.
- 398 artigos únicos.
- 13 artigos sinalizados para revisão textual.

O piloto permitiu identificar limitações na qualidade dos resumos e um desequilíbrio temporal provocado pela recuperação dos primeiros resultados por relevância.

Relatório: `reports/experiments/pilot_dataset.md`.

### 5. Análise exploratória

`src/research_atlas/analysis/eda.py`

Executar:

```bash
PYTHONPATH=src uv run python -m research_atlas.analysis.eda
```

A análise gera gráficos interativos em `reports/eda/`:

- Distribuição por ano.
- Artigos por área de coleta.
- Distribuição do tamanho dos resumos.
- Indicadores de qualidade textual.

A EDA implementada originalmente foi aplicada ao dataset piloto.

### 6. Amostragem para auditoria

`src/research_atlas/analysis/audit.py`

Executar:

```bash
PYTHONPATH=src uv run python -m research_atlas.analysis.audit
```

São gerados dois CSVs independentes para avaliação humana:

```text
data/processed/audit/
├── reviewer_1.csv
└── reviewer_2.csv
```

A amostra contém 100 artigos, com 20 por área científica.

A avaliação deve considerar relevância da aplicação de IA, pertinência científica e adequação do resumo.

**Estado:** amostra preparada; avaliação manual pendente.

### 7. Dataset ampliado

`src/research_atlas/data/dataset.py`

O pipeline integra:

1. Consultas aleatórias da OpenAlex por área, ano e termo.
2. Sementes determinísticas por consulta.
3. Cache local das respostas.
4. Normalização e deduplicação.
5. Validação estrutural.
6. Avaliação de qualidade textual.
7. Seleção estratificada por área e ano.
8. Relatório de rastreabilidade.
9. Exportação em Parquet.

#### Coleta ampliada

```bash
PYTHONPATH=src uv run --env-file .env python -m research_atlas.data.dataset \
  --sample-size 100 \
  --target-per-area 1000 \
  --seed 42 \
  --rounds 1
```

A coleta produz:

```text
data/processed/
├── dataset_candidates.parquet
├── dataset_selected.parquet
└── dataset_report.json
```

O arquivo `dataset_candidates.parquet` contém os candidatos estruturalmente válidos.

O arquivo `dataset_selected.parquet` contém os 5.000 artigos selecionados, com um identificador único por artigo e uma área principal de amostragem.

O relatório registra os parâmetros e os resultados da seleção.

#### Cache e retomada

As respostas de cada consulta são armazenadas em:

```text
data/raw/openalex_cache/
```

Executar novamente uma coleta com os mesmos parâmetros permite reutilizar as respostas disponíveis no cache.

Isso evita repetir consultas já concluídas em caso de interrupção.

Alterações nos parâmetros da consulta podem produzir novos arquivos de cache.

O mecanismo não garante que uma nova coleta sem o cache produza os mesmos registros, pois a cobertura da OpenAlex pode mudar.

## Distribuição da amostra ampliada

| Área de amostragem | Artigos |
|---|---:|
| Medicina | 1.000 |
| Astronomia | 1.000 |
| Biologia | 1.000 |
| Clima | 1.000 |
| Engenharia | 1.000 |
| **Total** | **5.000** |

Cada combinação de área e ano recebeu aproximadamente 166 ou 167 artigos, conforme a distribuição das cotas.

A amostragem foi projetada para comparações equilibradas. Ela não representa a frequência natural de publicações na literatura.

## Qualidade textual da amostra ampliada

| Indicador | Artigos |
|---|---:|
| Resumo com menos de 50 palavras | 72 |
| Resumo com mais de 500 palavras | 122 |
| Conteúdo textual suspeito | 8 |
| Pelo menos um alerta | 195 |

Os indicadores podem ocorrer simultaneamente e não são motivos automáticos de exclusão.

### Alertas por área

| Área | Artigos sinalizados |
|---|---:|
| Medicina | 55 |
| Astronomia | 22 |
| Biologia | 81 |
| Clima | 21 |
| Engenharia | 16 |

A Biologia apresentou a maior proporção de artigos sinalizados.

Esses registros precisarão de análise adicional antes da definição da política de entrada dos modelos de embeddings.

O relatório completo está disponível em:

`reports/experiments/final_dataset.md`

## Testes

Execute:

```bash
uv run pytest -v
```

A suíte cobre o cliente OpenAlex, pré-processamento, coleta piloto, qualidade textual, amostragem para auditoria, seleção estratificada e cache das consultas.

Os testes não devem depender de consultas reais à API.

## Interface

Inicie o Streamlit:

```bash
uv run streamlit run app/Home.py
```

Acesse:

http://localhost:8501

Os dashboards científicos e a visualização dos clusters serão desenvolvidos nas próximas etapas.

## Reprodutibilidade

A construção ampliada utiliza:

- Período fixo de 2020 a 2025.
- Cinco áreas de amostragem.
- Quatro termos de busca.
- Semente-base 42.
- Consultas aleatórias segmentadas.
- Deduplicação global pelo OpenAlex ID.
- Cotas por área e ano.
- Cache local.
- Relatório de parâmetros.
- Hash SHA-256 do dataset selecionado.

Para confirmar a integridade do arquivo de referência:

```bash
sha256sum data/processed/dataset_selected.parquet
```

O valor esperado para a versão v1 é:

```text
533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389
```

Datasets e caches não são versionados diretamente no Git. É necessário preservar cópias externas dos arquivos da coleta para permitir a reprodução exata dos experimentos.

## Limitações científicas

- A classificação temática da OpenAlex pode apresentar erros.
- A presença de termos relacionados à IA não confirma aplicação efetiva dessas técnicas.
- A auditoria humana ainda não foi concluída.
- A amostra é balanceada artificialmente e não representa a distribuição natural das publicações.
- Resumos curtos, longos ou inadequados podem afetar embeddings.
- A identificação de múltiplas áreas na coleta não equivale à descoberta de interdisciplinaridade semântica.
- Atualizações da fonte podem afetar novas coletas.

## Próximas etapas

1. Concluir a auditoria humana de relevância científica.
2. Definir a política de tratamento dos resumos sinalizados.
3. Implementar o pipeline de Machine Learning.
4. Criar o baseline TF-IDF + K-Means.
5. Gerar embeddings semânticos.
6. Comparar algoritmos de clusterização.
7. Avaliar qualidade, estabilidade e interpretação dos agrupamentos.
8. Produzir visualizações e dashboards científicos.
9. Explorar conexões em grafos de citações e evolução temporal.

## Organização do desenvolvimento

- `main`: versão estável.
- `dev`: integração das funcionalidades.
- Branches por issue, quando necessário.

Os commits seguem o padrão Conventional Commits, com mensagens em português.

## Fonte científica

[OpenAlex](https://openalex.org/)

O projeto utiliza os dados observando as condições de acesso e os critérios científicos definidos no protocolo.

---

**Status:** coleta ampliada concluída; validação científica pendente; pipeline de Machine Learning em preparação.