# Dataset ampliado — ResearchAtlas

## 1. Objetivo

Construir um conjunto de 5.000 artigos científicos sobre aplicações de Inteligência Artificial em Medicina, Astronomia, Biologia, Clima e Engenharia, destinado aos experimentos de clusterização semântica do ResearchAtlas.

## 2. Metodologia

**Fonte:** OpenAlex API.

**Período:** 2020–2025.

**Critérios estruturais:** artigos científicos, idioma inglês, título e resumo disponíveis, período válido e ausência de retratação.

**Termos de recuperação:**
- `machine learning`
- `deep learning`
- `artificial intelligence`
- `neural networks`

A coleta combinou filtros por subáreas científicas e termos presentes no título e resumo. Foram utilizadas consultas aleatórias com sementes determinísticas, distribuídas por área, ano e termo.

Após a coleta, os registros passaram por normalização, deduplicação pelo identificador OpenAlex, validação estrutural e seleção estratificada.

A seleção utilizou uma única área de amostragem principal por artigo, preservando todas as áreas identificadas durante a recuperação.

## 3. Resultados da coleta

| Indicador | Resultado |
|---|---:|
| Candidatos estruturalmente válidos | 11.210 |
| Artigos selecionados | 5.000 |
| Identificadores únicos selecionados | 5.000 |
| Meta total | 5.000 |
| Déficit de amostragem | 0 |
| Artigos selecionados recuperados por múltiplas áreas | 7 |

### 3.1 Distribuição por área

| Área | Artigos |
|---|---:|
| Medicina | 1.000 |
| Astronomia | 1.000 |
| Biologia | 1.000 |
| Clima | 1.000 |
| Engenharia | 1.000 |

Os 30 estratos de área e ano atingiram suas cotas, com aproximadamente 166 a 167 artigos por estrato.

### 3.2 Termos de recuperação

| Termo | Artigos associados |
|---|---:|
| Artificial intelligence | 1.180 |
| Deep learning | 1.339 |
| Machine learning | 1.327 |
| Neural networks | 1.385 |

Um artigo pode estar associado a mais de um termo. Portanto, os valores não representam conjuntos mutuamente exclusivos.

## 4. Qualidade textual

Foram sinalizados 195 artigos para revisão, correspondentes a 3,9% do dataset.

| Indicador | Artigos |
|---|---:|
| Resumo curto | 72 |
| Resumo longo | 122 |
| Conteúdo suspeito | 8 |
| Pelo menos um alerta | 195 |

Os indicadores podem se sobrepor.

### 4.1 Distribuição de alertas por área

| Área | Artigos sinalizados | Percentual |
|---|---:|---:|
| Astronomia | 22 | 2,2% |
| Biologia | 81 | 8,1% |
| Clima | 21 | 2,1% |
| Engenharia | 16 | 1,6% |
| Medicina | 55 | 5,5% |

Os registros sinalizados não foram excluídos automaticamente.

## 5. Rastreabilidade

**Arquivo principal:** `data/processed/dataset_selected.parquet`

**Conjunto de candidatos:** `data/processed/dataset_candidates.parquet`

**Relatório da coleta:** `data/processed/dataset_report.json`

**Cache das consultas:** `data/raw/openalex_cache/`

### SHA-256 do dataset selecionado

`533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389`

Esse identificador corresponde à versão utilizada como referência para os futuros experimentos.

## 6. Limitações

- A amostragem é balanceada, mas não representa a distribuição natural da produção científica.
- A classificação temática da OpenAlex pode conter imprecisões.
- A presença de termos relacionados à IA não garante aplicação efetiva dessas técnicas.
- Os resumos sinalizados exigem revisão antes de decisões definitivas de inclusão ou exclusão.
- A auditoria humana de relevância científica ainda não foi concluída.
- Os resultados dependem da cobertura e do estado dos registros da OpenAlex no momento da coleta.

## 7. Estado do dataset

**Coleta e integridade quantitativa:** concluídas.

**Validação científica de relevância:** pendente.

**Preparação dos dados para embeddings:** próxima etapa.

O dataset selecionado será mantido como referência reproduzível para o pipeline de Machine Learning. Qualquer versão filtrada ou transformada deverá ser identificada separadamente, preservando o arquivo original e sua rastreabilidade.