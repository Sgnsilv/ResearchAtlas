# Protocolo de Coleta de Dados — ResearchAtlas

**Versão:** 1.0 — Protocolo de coleta ampliada executado  
**Fonte de dados:** OpenAlex  
**Status:** Coleta quantitativa concluída; auditoria de relevância científica pendente

## 1. Objetivo

Construir um conjunto de dados de publicações científicas sobre aplicações de Inteligência Artificial em diferentes disciplinas, permitindo investigar comunidades semânticas e conexões interdisciplinares por meio de aprendizado não supervisionado.

### Pergunta científica

É possível identificar comunidades de pesquisa interdisciplinares a partir da similaridade semântica de artigos sobre aplicações de Inteligência Artificial e descobrir conexões entre áreas científicas distintas?

## 2. Fonte de dados

Utilizaremos a API REST da OpenAlex.

- Endpoint: `https://api.openalex.org/works`
- Documentação: https://developers.openalex.org/
- Identificador primário: OpenAlex Work ID
- Formato de coleta: JSON
- Formato de armazenamento processado: Parquet

A coleta deverá ser reproduzível, registrando parâmetros, data de execução e critérios de seleção.

## 3. Delimitação do dataset

| Parâmetro | Definição |
|---|---|
| Período | 2020–2025 |
| Idioma | Inglês |
| Tipo | Artigos científicos (`article`) |
| Volume-alvo | 5.000 artigos |
| Amostra piloto | 100–200 artigos por área |
| Áreas | Medicina, Astronomia, Biologia, Clima e Engenharia |
| Texto obrigatório | Título e resumo utilizável |

A quantidade definitiva poderá ser ajustada conforme a disponibilidade e qualidade dos registros.

## 4. Estratégia de busca

A seleção será realizada a partir da combinação de dois componentes:

**Componente A — Inteligência Artificial**

Termos candidatos:

- artificial intelligence
- machine learning
- deep learning
- neural networks
- transformer models
- reinforcement learning
- computer vision

**Componente B — Área científica de aplicação**

| Área | Termos candidatos |
|---|---|
| Medicina | medical imaging, clinical prediction, disease diagnosis |
| Astronomia | astrophysics, galaxies, exoplanets, astronomical images |
| Biologia | genomics, protein structure, bioinformatics |
| Clima | climate modeling, weather forecasting, environmental monitoring |
| Engenharia | predictive maintenance, industrial robotics, structural monitoring |

Os termos não constituem ainda consultas finais. Avaliaremos combinações de busca textual e filtros por tópicos científicos da OpenAlex.

Os identificadores dos tópicos deverão ser verificados na API antes de sua utilização.

### 4.1. Medicina e Saúde

**Identificador da área na OpenAlex:** `27` (Medicine).

**Estratégia candidata:** combinar a classificação temática da OpenAlex com evidências textuais de aplicação de Inteligência Artificial.

**Tópicos candidatos:**
- Medical Image Segmentation Techniques
- AI in Cancer Detection
- Radiomics and Machine Learning in Medical Imaging

**Filtro de área a avaliar:** `topics.field.id:27`.

**Cuidados metodológicos:**
- Não considerar todo artigo médico como aplicação de IA.
- Evitar concentrar a amostra exclusivamente em diagnóstico por imagem.
- Investigar aplicações de IA também em predição clínica e outras especialidades.
- Verificar a relevância dos resultados por meio de auditoria manual.

**Status:** tópicos candidatos identificados; IDs específicos e consultas definitivas pendentes de validação.

### 4.2. Astronomia

**Field:** `31` — Physics and Astronomy.

**Subfield candidata:**
- `3103` — Astronomy and Astrophysics.

**Aplicações de IA investigadas:**
- Classificação de galáxias.
- Detecção e caracterização de exoplanetas.
- Análise de imagens astronômicas.
- Identificação de eventos e objetos astronômicos.

**Termos candidatos:** `machine learning`, `deep learning`, `galaxy classification`, `exoplanet detection`, `astronomical image analysis`.

**Cuidados:** selecionar aplicações efetivas de IA e não confundir artigos gerais de astrofísica com trabalhos de aprendizado de máquina.

### 4.3. Biologia e Bioinformática

**Field:** `13` — Biochemistry, Genetics and Molecular Biology.

**Subfields candidatas:**
- `1311` — Genetics.
- `1315` — Structural Biology.

**Aplicações de IA investigadas:**
- Predição de estruturas de proteínas.
- Análise de sequências genéticas.
- Genômica computacional.
- Identificação de padrões moleculares.

**Termos candidatos:** `machine learning`, `deep learning`, `protein structure prediction`, `genomics`, `bioinformatics`.

**Cuidados:** diferenciar aplicações de IA de métodos tradicionais de bioinformática que não utilizam aprendizado de máquina.

### 4.4. Clima e Meio Ambiente

**Fields:**
- `19` — Earth and Planetary Sciences.
- `23` — Environmental Science.

**Subfields candidatas:**
- `1902` — Atmospheric Science.
- `2306` — Global and Planetary Change.

**Aplicações de IA investigadas:**
- Modelagem climática.
- Previsão meteorológica.
- Detecção de eventos climáticos extremos.
- Monitoramento ambiental.
- Análise de dados de sensoriamento remoto.

**Termos candidatos:** `machine learning`, `deep learning`, `climate modeling`, `weather forecasting`, `remote sensing`.

**Cuidados:** evitar artigos de monitoramento ambiental que não utilizem IA e avaliar a sobreposição com Engenharia.

### 4.5. Engenharia

**Field:** `22` — Engineering.

**Subfields candidatas:**
- `2207` — Control and Systems Engineering.
- `2210` — Mechanical Engineering.

**Aplicações de IA investigadas:**
- Manutenção preditiva.
- Robótica inteligente.
- Otimização industrial.
- Diagnóstico de falhas.
- Controle de sistemas.

**Termos candidatos:** `machine learning`, `deep learning`, `predictive maintenance`, `industrial robotics`, `fault detection`.

**Cuidados:** distinguir técnicas clássicas de controle e otimização de aplicações efetivas de aprendizado de máquina.

### 4.6. Estratégia de filtragem

A seleção inicial combinará três condições:

1. Publicação dentro dos critérios gerais de período, idioma, tipo e disponibilidade de resumo.
2. Pertencimento a uma das subáreas científicas selecionadas.
3. Evidência textual ou temática de utilização de Inteligência Artificial.

A OpenAlex permite consultar publicações por `topics.subfield.id` e `topics.field.id`.

Por exemplo, a expressão:

`topics.subfield.id:3103`

seleciona artigos associados à subárea Astronomy and Astrophysics.

A utilização de `topics.subfield.id` será priorizada em relação a `primary_topic.subfield.id` na fase exploratória, pois permite recuperar artigos com tópicos secundários relacionados à área investigada.

A estratégia definitiva será definida após a avaliação de cobertura, precisão temática e sobreposição entre áreas.

**Status:** identificadores candidatos definidos. Consultas combinadas e relevância dos resultados ainda não validadas.
## 5. Critérios de inclusão

Um artigo será elegível quando:

1. Estiver publicado entre 2020 e 2025.
2. Possuir título e resumo utilizáveis.
3. Estiver classificado como artigo científico.
4. Tiver idioma identificado como inglês.
5. Apresentar aplicação relevante de uma técnica de IA.
6. Estiver relacionado a pelo menos uma das áreas investigadas.

## 6. Critérios de exclusão

Serão excluídos:

- Registros duplicados.
- Publicações retratadas.
- Trabalhos sem resumo utilizável.
- Resultados sem relação demonstrável com a pergunta científica.
- Trabalhos que apenas mencionam IA incidentalmente.
- Registros com metadados essenciais inválidos.

Artigos associados a múltiplas áreas poderão manter essa informação, mas serão contabilizados apenas uma vez no dataset final.

## 7. Campos a recuperar

| Campo | Finalidade |
|---|---|
| `id` | Identificador único |
| `doi` | Identificador bibliográfico complementar |
| `title` | Conteúdo textual |
| `abstract_inverted_index` | Reconstrução do resumo |
| `publication_year` | Análise temporal |
| `publication_date` | Data da publicação |
| `language` | Validação do idioma |
| `type` | Validação do tipo |
| `primary_topic` | Classificação temática principal |
| `topics` | Classificações temáticas adicionais |
| `keywords` | Interpretação dos resultados |
| `referenced_works` | Construção do grafo de citações |
| `cited_by_count` | Metadado bibliométrico |
| `is_retracted` | Exclusão de retratações |

A disponibilidade de cada campo será verificada na coleta piloto.

## 8. Amostragem

A amostra inicial terá aproximadamente 100–200 artigos por área.

Após validar a qualidade dos resultados, a meta será expandir para aproximadamente 1.000 artigos por área.

A amostragem deverá considerar:

- Representação dos diferentes anos.
- Diversidade temática.
- Prevenção de duplicatas.
- Identificação de artigos interdisciplinares.
- Controle de vieses de relevância e popularidade.

Não utilizaremos a contagem de citações como critério principal de seleção.

## 9. Validação da qualidade

A qualidade da amostra será avaliada por:

1. Percentual de registros com campos obrigatórios completos.
2. Quantidade de duplicatas.
3. Distribuição por ano e área.
4. Tamanho dos títulos e resumos.
5. Auditoria manual de relevância temática.
6. Verificação da presença efetiva de aplicações de IA.

Na auditoria manual, registraremos o motivo das exclusões e divergências de classificação.

## 10. Reprodutibilidade

Cada execução deverá registrar:

- Data da coleta.
- Versão do protocolo.
- Consultas utilizadas.
- Parâmetros e filtros da API.
- Volume retornado e volume aceito.
- Sementes aleatórias, quando aplicáveis.
- Critérios e quantidade de exclusões.
- Versões das bibliotecas utilizadas.

## 11. Limitações previstas

- Incompletude ou ausência de resumos.
- Erros de classificação temática da OpenAlex.
- Vieses de indexação e idioma.
- Dificuldade de distinguir aplicações efetivas de simples menções à IA.
- Representatividade limitada de uma amostra estratificada.
- Cobertura incompleta de citações entre os artigos selecionados.

## 12. Próximas etapas

1. Investigar identificadores de tópicos da OpenAlex.
2. Validar combinações de termos e filtros.
3. Executar consultas exploratórias.
4. Auditar a relevância dos resultados.
5. Revisar e aprovar o protocolo antes da coleta definitiva.

## 13. Validação exploratória — Astronomia

**Subárea OpenAlex:** `3103` — Astronomy and Astrophysics.

**Período:** 2020–2025.

### Resultados das consultas

| Estratégia | Artigos encontrados |
|---|---:|
| A — Apenas classificação temática | 175.456 |
| B — Classificação + machine learning | 11.378 |
| C — Classificação + múltiplos termos de IA | 15.107 |

A estratégia C apresentou cobertura aproximadamente 32,8% superior à estratégia B.

### Análise preliminar

Foram examinados os títulos dos cinco primeiros resultados das consultas B e C.

Ambas apresentaram artigos aparentemente relevantes para aplicações de IA em Astronomia.

A estratégia C recuperou maior diversidade aparente de temas, incluindo exoplanetas, astroquímica, ondas gravitacionais e matéria escura.

Entretanto, a avaliação foi limitada aos títulos e aos primeiros resultados das buscas.

### Decisões provisórias

1. Manter a estratégia C como candidata principal.
2. Verificar a interpretação dos operadores booleanos pela API.
3. Avaliar a relevância dos resumos, não apenas dos títulos.
4. Evitar amostragem baseada exclusivamente na ordenação por relevância.
5. Identificar artigos de revisão para análise de sensibilidade.
6. Validar uma amostra mais representativa antes de escolher a estratégia definitiva.

**Status:** consulta exploratória concluída; protocolo de seleção ainda não aprovado.

## 14. Validação exploratória — Biologia

### Comparação das consultas

| Estratégia | Artigos encontrados |
|---|---:|
| Busca geral ampliada (C) | 33.877 |
| Busca por "machine learning" no título e resumo | 4.602 |
| Busca por "deep learning" no título e resumo | 2.470 |

As buscas restritas ao título e resumo apresentaram resultados preliminares mais alinhados ao objetivo da pesquisa.

### Auditoria preliminar

Foi selecionada uma amostra de cinco artigos utilizando `sample=5` e `seed=42`.

A avaliação inicial dos títulos identificou:
- 2 artigos claramente relacionados à aplicação de IA em Biologia.
- 3 artigos que exigem análise dos resumos para confirmação.

Esses resultados não constituem uma estimativa confiável da precisão da estratégia de busca.

### Decisões metodológicas

1. Priorizar buscas restritas a títulos e resumos.
2. Registrar a área utilizada na coleta separadamente das áreas de aplicação identificadas.
3. Permitir que um artigo pertença a múltiplas áreas científicas.
4. Eliminar duplicatas pelo identificador OpenAlex.
5. Verificar a aplicação efetiva de IA antes de aceitar um artigo.
6. Realizar uma auditoria maior e estratificada antes da coleta definitiva.

**Status:** validação exploratória concluída; estratégia definitiva pendente de aprovação.

## 15. Comparação de cobertura — Busca por título e resumo

Foram realizadas consultas à OpenAlex utilizando quatro expressões relacionadas à Inteligência Artificial, combinadas com os filtros temáticos e temporais definidos no protocolo.

### Resultados

| Área | Machine Learning | Deep Learning | Artificial Intelligence | Neural Networks |
|---|---:|---:|---:|---:|
| Medicina | 27.974 | 37.320 | 46.947 | 24.960 |
| Astronomia | 2.983 | 1.669 | 385 | 2.760 |
| Biologia | 4.602 | 2.470 | 1.333 | 2.337 |
| Clima | 11.065 | 5.316 | 2.083 | 7.205 |
| Engenharia | 29.015 | 23.528 | 14.293 | 45.142 |

**Data da consulta:** 21/11/2026.

### Interpretação

Os resultados indicam diferenças expressivas na cobertura dos termos entre as disciplinas.

A adoção de um único termo de busca poderia introduzir vieses relacionados à terminologia utilizada em cada campo científico.

As contagens não representam conjuntos mutuamente exclusivos e não devem ser somadas diretamente.

### Estratégia candidata

1. Executar consultas separadas para os quatro termos de IA.
2. Combinar os resultados em um conjunto de candidatos.
3. Remover duplicatas pelo identificador OpenAlex.
4. Preservar as áreas e os termos responsáveis pela recuperação de cada artigo.
5. Aplicar amostragem estratificada por área e ano, utilizando semente fixa.
6. Validar a relevância temática e a aplicação efetiva de IA.
7. Registrar os critérios de seleção e exclusão.

### Plano de amostragem

- Meta: aproximadamente 5.000 artigos únicos.
- Distribuição-alvo: 1.000 artigos por área.
- Período: 2020 a 2025.
- Distribuição anual: aproximadamente uniforme dentro de cada área, quando viável.
- Artigos multiarea: identificados e preservados, sem duplicação global.
- Semente aleatória: 42.

A amostra balanceada não deve ser interpretada como representativa da distribuição real da produção científica.

### Pendências

- Confirmar a semântica das buscas compostas.
- Auditar amostras maiores utilizando títulos e resumos.
- Determinar uma regra operacional para resolver sobreposições entre estratos.
- Verificar a cobertura de termos adicionais, quando necessário.

**Status:** estratégia candidata selecionada; aprovação final depende da auditoria de relevância.

## 16. Protocolo de auditoria de relevância

### Objetivo

Avaliar a precisão temática da estratégia de coleta antes da construção do dataset definitivo.

### Amostragem para auditoria

Será selecionada inicialmente uma amostra de 100 artigos, distribuídos igualmente entre as cinco áreas científicas.

- 20 artigos por área.
- Seleção aleatória reproduzível.
- Semente aleatória: 42.
- Preservação das consultas, dos identificadores e das áreas de origem.
- Distribuição entre termos de busca e anos, sempre que possível.

Artigos recuperados por múltiplas consultas serão identificados e não contabilizados como registros independentes.

### Critérios de avaliação

**Critério A — Aplicação efetiva de IA**

- Sim: o estudo utiliza ou investiga uma técnica de IA aplicada ao problema científico.
- Não: a IA é apenas mencionada ou não participa da investigação.
- Incerto: as informações disponíveis são insuficientes.

**Critério B — Pertinência científica**

- Sim: o problema investigado pertence à área de aplicação avaliada.
- Não: a aplicação principal não pertence à área.
- Incerto: não é possível determinar a pertinência com segurança.

Artigos de revisão serão identificados separadamente para posterior análise de sensibilidade.

### Procedimento

1. Selecionar a amostra conforme os critérios estabelecidos.
2. Analisar título e resumo de cada artigo.
3. Registrar as classificações em tabela padronizada.
4. Realizar uma avaliação independente pelos dois integrantes em uma subamostra comum.
5. Comparar classificações e resolver divergências por consenso.
6. Registrar os motivos de exclusão.
7. Calcular a proporção observada de artigos relevantes por área, acompanhada de sua incerteza amostral.
8. Revisar a estratégia de busca caso sejam identificados problemas sistemáticos.

### Campos da auditoria

- `openalex_id`
- `title`
- `collection_area`
- `retrieval_terms`
- `ai_application`
- `scientific_relevance`
- `secondary_areas`
- `publication_type`
- `decision`
- `exclusion_reason`
- `reviewer`

### Critério de aprovação

A estratégia será considerada adequada quando apresentar relevância temática satisfatória nas cinco áreas, sem evidências de erros sistemáticos importantes.

Como referência inicial, buscaremos pelo menos 80% de artigos classificados como relevantes em cada estrato. Esse valor é uma meta operacional provisória, não um limiar estatisticamente validado.

Caso a relevância seja insuficiente, os filtros serão revisados e uma nova amostra independente será avaliada.

### Limitações

A auditoria inicial possui tamanho reduzido e não comprova a representatividade da amostra definitiva.

A avaliação de títulos e resumos também pode não identificar todas as aplicações efetivas de IA.

A capacidade de recuperar artigos relevantes que não foram selecionados (recall) não será estimada diretamente por esse procedimento.

**Execução da auditoria:** Issue #04 — Dataset piloto e análise exploratória.

**Status:** protocolo de auditoria definido; execução pendente.


## 17. Execução da coleta ampliada — Dataset v1

### 17.1. Objetivo

Executar a estratégia de amostragem estratificada definida no protocolo, produzindo um conjunto de aproximadamente 5.000 artigos científicos para os experimentos de aprendizado de máquina do ResearchAtlas.

### 17.2. Estratégia executada

A coleta utilizou os filtros:

- Período: 2020–2025.
- Idioma: inglês.
- Tipo de publicação: artigo científico.
- Disponibilidade de resumo na OpenAlex.
- Subáreas científicas definidas nas etapas exploratórias.
- Busca no título e resumo por termos relacionados à IA.

Foram utilizadas as expressões:

- `machine learning`
- `deep learning`
- `artificial intelligence`
- `neural networks`

A seleção de candidatos utilizou amostragem aleatória com sementes determinísticas, segmentada por área, ano e termo de recuperação.

Parâmetros da execução:

| Parâmetro | Valor |
|---|---|
| Semente-base | 42 |
| Amostra solicitada por consulta | 100 |
| Rodadas por consulta | 1 |
| Meta por área | 1.000 |
| Meta global | 5.000 |
| Número de áreas | 5 |
| Número de anos | 6 |
| Número de termos | 4 |
| Estratos de seleção | 30 |
| Consultas planejadas | 120 |

A coleta utilizou cache JSON local para preservar as respostas e permitir retomada das consultas concluídas.

### 17.3. Processamento e seleção

Os registros recuperados passaram pelas seguintes etapas:

1. Reconstrução dos resumos.
2. Normalização de identificadores e textos.
3. Consolidação de duplicatas por OpenAlex Work ID.
4. Preservação dos termos e áreas de recuperação.
5. Validação estrutural dos registros.
6. Avaliação exploratória da qualidade textual.
7. Seleção estratificada por área e ano.
8. Exportação e geração do relatório de rastreabilidade.

Cada artigo foi contabilizado em uma única área principal de amostragem (`sampling_area`), mantendo a lista de todas as áreas de recuperação (`collection_areas`).

A seleção não utilizou a contagem de citações como critério de ordenação.

### 17.4. Resultados

| Indicador | Resultado |
|---|---:|
| Candidatos estruturalmente válidos | 11.210 |
| Artigos selecionados | 5.000 |
| Identificadores únicos | 5.000 |
| Déficit de amostragem | 0 |
| Artigos recuperados por múltiplas áreas | 7 |

Distribuição por área:

| Área | Artigos |
|---|---:|
| Medicina | 1.000 |
| Astronomia | 1.000 |
| Biologia | 1.000 |
| Clima | 1.000 |
| Engenharia | 1.000 |

Todos os 30 estratos de área e ano atingiram suas cotas de seleção, com aproximadamente 166 ou 167 artigos por estrato.

### 17.5. Qualidade textual

Foram sinalizados 195 artigos, correspondentes a 3,9% do conjunto selecionado.

| Indicador | Ocorrências |
|---|---:|
| Resumos curtos | 72 |
| Resumos longos | 122 |
| Conteúdo suspeito | 8 |

As categorias podem se sobrepor.

Os artigos sinalizados permaneceram no arquivo selecionado para permitir avaliação posterior, sem exclusão automática.

A distribuição de alertas por área foi:

| Área | Artigos sinalizados |
|---|---:|
| Medicina | 55 |
| Astronomia | 22 |
| Biologia | 81 |
| Clima | 21 |
| Engenharia | 16 |

A maior incidência em Biologia deverá ser investigada antes do tratamento definitivo dos textos.

### 17.6. Identificação e rastreabilidade

Arquivo de referência:

`data/processed/dataset_selected.parquet`

SHA-256:

`533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389`

O hash identifica os bytes exatos do arquivo selecionado nessa execução.

Também foram produzidos:

- `data/processed/dataset_candidates.parquet`
- `data/processed/dataset_report.json`
- Cache das consultas em `data/raw/openalex_cache/`

O relatório metodológico está disponível em:

`reports/experiments/final_dataset.md`

### 17.7. Limitações e validações pendentes

A conclusão quantitativa não equivale à aprovação científica de todos os artigos.

Permanecem pendentes:

- Auditoria humana da pertinência dos artigos.
- Verificação da aplicação efetiva de IA.
- Avaliação dos resumos inadequados.
- Decisão documentada sobre eventuais exclusões.
- Definição do subconjunto textual adequado aos embeddings.

A amostragem balanceada é apropriada para experimentos comparativos entre estratos, mas não representa a distribuição natural da produção científica.

A reprodutibilidade exata depende da preservação dos arquivos coletados, pois a OpenAlex pode atualizar seus registros.

### 17.8. Encaminhamento

O conjunto selecionado v1 será preservado como referência imutável para os experimentos.

Caso a auditoria ou o processamento textual resultem em exclusões, será produzida uma nova versão derivada, com critérios, contagens e identificador próprios, sem sobrescrever a v1.

**Estado:** coleta e integridade quantitativa concluídas; validação científica pendente.