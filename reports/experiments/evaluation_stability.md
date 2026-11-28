# Avaliação consolidada de métricas, estabilidade e concordância — ResearchAtlas

**Issue:** #08 — Criar pipeline de avaliação e estabilidade  
**Dataset:** ResearchAtlas v1 (`5.000` artigos)  
**Representações avaliadas:** `BAAI/bge-small-en-v1.5` (`384d`, L2) e `TF-IDF` esparso (`10.000` e `20.000` características, L2)  
**Artefatos de referência:** `data/processed/evaluation/v1/`  
**Status:** pipeline implementado, artefatos determinísticos exportados e validados por `250` testes automatizados

## 1. Objetivo

Consolidar, padronizar e auditar a avaliação quantitativa e a análise de concordância das `24` configurações de clusterização previamente versionadas nas Issues #06 e #07, sem reexecutar algoritmos de agrupamento nem alterar artefatos anteriores.

O pipeline estabelece regras explícitas de geometria, amostragem, escopo (`full` versus `grouped_only`), tratamento de partições degeneradas ou inválidas e distinção rigorosa entre estabilidade por semente de inicialização e concordância estrutural entre partições distintas.

## 2. Metodologia e protocolo de avaliação

### 2.1. Insumos e alinhamento

A avaliação reutiliza integralmente os artefatos versionados:

- Dataset selecionado (`data/processed/dataset_selected.parquet`, SHA-256 `533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389`);
- Embeddings BGE (`data/processed/embeddings/bge-small-en-v1.5-v1/embeddings.npy`, SHA-256 `13fe24f37622790f6d6630fbdb5e91742060f95a8d7310597d3bd98f2a7f465c`);
- Rótulos de `18` configurações semânticas em `data/processed/clustering/v1/cluster_assignments.parquet`;
- Rótulos de `6` configurações lexicais em `data/processed/tfidf_baseline/v1/tfidf_assignments.parquet`, com reconstrução determinística das matrizes TF-IDF esparsas CSR a partir dos parâmetros do manifesto da Issue #06;
- Tabelas de estabilidade por semente em `clustering/v1/kmeans_stability.csv` e `tfidf_baseline/v1/tfidf_stability.csv`.

Antes de qualquer cálculo, o pipeline verifica os hashes SHA-256 dos insumos, a ordenação contínua de `embedding_index` (`0` a `4.999`) e a igualdade posicional estrita de `openalex_id` entre dataset, metadados de embeddings e tabelas de atribuição de clusters. Os `119` valores de referência das Issues #06 e #07 e os `9` indicadores cruzados de estabilidade/ARI foram integralmente reconciliados dentro da tolerância numérica (`rtol=1e-6`, `atol=1e-7`).

### 2.2. Métricas internas, geometria e amostragem

| Métrica | Geometria | BGE denso (`384d`) | TF-IDF esparso (`CSR`) | Direção formal |
|---|---|---|---|---|
| Silhouette | Distância cosseno | Calculada | Calculada | Maior valor indica maior coesão/separação na geometria cosseno |
| Davies-Bouldin (DB) | Distância euclidiana | Calculada | Não calculada (`matriz_esparsa_nao_densificada`) | Menor valor indica menor razão dispersão/separação euclidiana |
| Calinski-Harabasz (CH) | Distância euclidiana | Calculada | Não calculada (`matriz_esparsa_nao_densificada`) | Maior valor indica maior razão de variância entre/intra-clusters |

Decisões metodológicas aplicadas em `src/research_atlas/evaluation/metrics.py` e `pipeline.py`:

1. **Preservação de esparsidade no TF-IDF:** matrizes `5.000 × 10.000` e `5.000 × 20.000` não são convertidas para formato denso. Apenas a Silhouette cosseno é calculada diretamente sobre a matriz esparsa; DB e CH permanecem ausentes com justificativa estruturada (`matriz_esparsa_nao_densificada`).
2. **Protocolo da Silhouette:** utiliza `metric="cosine"`, `sample_size=2000` e `seed=42`. Quando o número de artigos avaliados (`n_evaluated`) é menor ou igual a `2.000`, o cálculo é exato sobre todo o subconjunto avaliado (`silhouette_sampling="full"`, `silhouette_sampled=False`). Caso uma amostra aleatória colapse para menos de `2` clusters ou coincida apenas em pontos idênticos, o módulo recorre ao cálculo exato se `n_evaluated <= 10.000` (`fallback_full`) ou registra indisponibilidade estruturada (`amostra_sem_clusters_validos`).
3. **Separação de escopos e comparabilidade (`comparability_group`):**
   - Partições sem ruído são avaliadas em escopo `full` (`5.000` artigos).
   - Partições com ruído (`-1`, HDBSCAN) são avaliadas em escopo `grouped_only`, restrito aos artigos agrupados. Como cada configuração HDBSCAN seleciona um subconjunto distinto de artigos, cada linha `grouped_only` recebe um `comparability_group` próprio e **não** pode ser comparada diretamente às partições `full` nem a outras partições `grouped_only` como se cobrissem a mesma base.
   - Da mesma forma, valores de Silhouette obtidos em espaços vetoriais distintos (`bge-small-en-v1.5:384d_l2` versus `tfidf:10000f_sparse` e `tfidf:20000f_sparse`) pertencem a grupos de comparabilidade distintos.

### 2.3. Tratamento de partições inválidas, degenerescências e convenções operacionais

Uma partição interna é marcada como `valid=False` (com métricas ausentes e motivo explícito em `reasons` e `*_unavailable_reason`) quando ocorre qualquer uma das condições:

- Nenhum artigo agrupado (`sem_artigos_agrupados`, ex.: todos os rótulos `-1`);
- Menos de dois clusters efetivos (`menos_de_2_clusters`);
- Número de clusters maior ou igual ao número de artigos agrupados (`clusters_nao_reduzem_dados`);
- Vetores avaliados todos coincidentes dentro da tolerância numérica (`pontos_identicos`).

Para partições formalmente válidas, mas estruturalmente assimétricas ou parciais, o pipeline registra avisos em `warnings`:

- `cluster_unitario`: presença de ao menos um cluster com apenas `1` artigo;
- `cluster_dominante`: maior cluster concentrando mais de `80%` dos artigos agrupados (`MAX_CLUSTER_SHARE = 0,8`);
- `cobertura_baixa`: cobertura inferior a `50%` da base (`LOW_COVERAGE_PCT = 50,0`).

Os parâmetros `LOW_COVERAGE_PCT = 50,0` e `SILHOUETTE_FULL_FALLBACK_MAX = 10.000` são **convenções operacionais deste pipeline** (sinalização descritiva de cobertura reduzida e teto computacional para fallback exato), e não limiares científicos universais.

Nas `24` configurações exportadas em `evaluation_metrics.csv`, todas as partições individuais são formalmente válidas (`valid=True`), enquanto `12` apresentam avisos estruturais (`1` com `cluster_unitario;cluster_dominante`, `2` com `cluster_dominante` e `9` com `cobertura_baixa`). Já nas comparações pareadas de concordância (`partition_concordance.csv`), `27` dos `276` pares são **inválidos** (`valid=False`, `ari` ausente) porque uma das partições colapsa para apenas `1` cluster dentro da interseção avaliada (`menos_de_2_clusters_na_intersecao`).

## 3. Avaliação interna das 24 configurações

### 3.1. Partições completas (`full`) — BGE e TF-IDF

Todas as `13` partições completas cobrem `100%` dos `5.000` artigos (`0%` de ruído) e utilizam amostragem determinística da Silhouette (`sampled`, `2.000` artigos, `seed=42`).

| Configuração (`config_id`) | Representação | Algoritmo | Clusters | Maior cluster (% total) | Silhouette (cosseno) | Davies-Bouldin | Calinski-Harabasz | Avisos |
|---|---|---|---:|---:|---:|---:|---:|---|
| `kmeans_k5` | BGE 384d | K-Means | 5 | 1.132 (22,64%) | 0,1360 | 3,2154 | 255,21 | — |
| `kmeans_k7` | BGE 384d | K-Means | 7 | 987 (19,74%) | 0,1424 | 3,0094 | 210,31 | — |
| `kmeans_k8` | BGE 384d | K-Means | 8 | 993 (19,86%) | 0,1472 | 3,2192 | 192,45 | — |
| `kmeans_k9` | BGE 384d | K-Means | 9 | 967 (19,34%) | 0,1189 | 2,9904 | 175,68 | — |
| `agglomerative_average_k8` | BGE 384d | Agglomerative | 8 | 4.983 (99,66%) | 0,1082 | 1,6661 | 2,09 | `cluster_unitario;cluster_dominante` |
| `agglomerative_complete_k8` | BGE 384d | Agglomerative | 8 | 1.008 (20,16%) | 0,1112 | 3,4904 | 150,12 | — |
| `agglomerative_ward_k8` | BGE 384d | Agglomerative | 8 | 949 (18,98%) | 0,1142 | 3,4579 | 173,67 | — |
| `tfidf_10k_k5` | TF-IDF 10k | K-Means | 5 | 1.932 (38,64%) | 0,0130 | Ausente* | Ausente* | — |
| `tfidf_10k_k8` | TF-IDF 10k | K-Means | 8 | 976 (19,52%) | 0,0146 | Ausente* | Ausente* | — |
| `tfidf_10k_k13` | TF-IDF 10k | K-Means | 13 | 684 (13,68%) | 0,0156 | Ausente* | Ausente* | — |
| `tfidf_20k_k5` | TF-IDF 20k | K-Means | 5 | 1.845 (36,90%) | 0,0099 | Ausente* | Ausente* | — |
| `tfidf_20k_k8` | TF-IDF 20k | K-Means | 8 | 888 (17,76%) | 0,0124 | Ausente* | Ausente* | — |
| `tfidf_20k_k13` | TF-IDF 20k | K-Means | 13 | 1.015 (20,30%) | 0,0140 | Ausente* | Ausente* | — |

*\*Ausente com justificativa `matriz_esparsa_nao_densificada`.*

Observações sobre as partições completas:

- No espaço BGE equilibrado, nenhuma métrica interna aponta isoladamente um único `k` ótimo: entre os candidatos com rótulos preservados, `kmeans_k8` apresenta a maior Silhouette (`0,1472`), `kmeans_k9` o menor Davies-Bouldin entre partições equilibradas (`2,9904`) e `kmeans_k5` o maior Calinski-Harabasz (`255,21`; na varredura `k=2..15` da Issue #07, CH foi máximo em `k=2`).
- O caso `agglomerative_average_k8` ilustra por que métricas internas não devem ser lidas sem diagnóstico estrutural: apesar de exibir o menor Davies-Bouldin global (`1,6661`), agrupa `4.983` de `5.000` artigos (`99,66%`) em um único cluster e isola outliers unitários, sendo imediatamente sinalizado por `cluster_unitario;cluster_dominante` e `CH = 2,09`.
- Entre os métodos hierárquicos em `k=8`, `agglomerative_ward_k8` mantém a distribuição mais equilibrada (`350` a `949` artigos) e a maior Silhouette (`0,1142`).

### 3.2. Cobertura, ruído e métricas do HDBSCAN (`grouped_only`)

As `11` configurações HDBSCAN (nove com seleção `eom` e duas com seleção `leaf`) foram avaliadas exclusivamente sobre os artigos agrupados (`scope="grouped_only"`). Para `(20, 5)` e `(30, 5)` em `eom` (`2.840` agrupados), a Silhouette utilizou amostra de `2.000` pontos (`sampled`); nas outras nove configurações (`840` a `1.342` agrupados), o cálculo foi exato sobre todos os agrupados (`full`).

| Configuração (`config_id`) | Seleção | Clusters | Agrupados | Cobertura | Ruído (`-1`) | Maior cluster (% total) | Amostragem Sil. | Silhouette (`grouped_only`) | DB | CH | Avisos |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| `hdbscan_eom_mcs20_ms5` | EOM | 2 | 2.840 | 56,80% | 43,20% | 2.809 (56,18%) | `sampled` (2.000) | 0,0950 | 2,3441 | 17,81 | `cluster_dominante` |
| `hdbscan_eom_mcs20_ms10` | EOM | 8 | 1.118 | 22,36% | 77,64% | 725 (14,50%) | `full` (1.118) | 0,2096 | 1,8655 | 47,30 | `cobertura_baixa` |
| `hdbscan_eom_mcs30_ms5` | EOM | 2 | 2.840 | 56,80% | 43,20% | 2.809 (56,18%) | `sampled` (2.000) | 0,0950 | 2,3441 | 17,81 | `cluster_dominante` |
| `hdbscan_eom_mcs30_ms10` | EOM | 4 | 1.342 | 26,84% | 73,16% | 725 (14,50%) | `full` (1.342) | 0,1899 | 2,3934 | 87,66 | `cobertura_baixa` |
| `hdbscan_eom_mcs50_ms5` | EOM | 6 | 1.335 | 26,70% | 73,30% | 488 (9,76%) | `full` (1.335) | 0,2063 | 2,2862 | 91,00 | `cobertura_baixa` |
| `hdbscan_eom_mcs50_ms10` | EOM | 3 | 1.298 | 25,96% | 74,04% | 725 (14,50%) | `full` (1.298) | 0,1938 | 2,5542 | 113,57 | `cobertura_baixa` |
| `hdbscan_eom_mcs50_ms20` | EOM | 3 | 1.113 | 22,26% | 77,74% | 671 (13,42%) | `full` (1.113) | 0,2063 | 2,4717 | 98,35 | `cobertura_baixa` |
| `hdbscan_eom_mcs100_ms5` | EOM | 4 | 1.263 | 25,26% | 74,74% | 488 (9,76%) | `full` (1.263) | 0,2045 | 2,4434 | 114,21 | `cobertura_baixa` |
| `hdbscan_eom_mcs100_ms10` | EOM | 3 | 1.298 | 25,96% | 74,04% | 725 (14,50%) | `full` (1.298) | 0,1938 | 2,5542 | 113,57 | `cobertura_baixa` |
| `hdbscan_leaf_mcs20_ms5` | Leaf | 18 | 840 | 16,80% | 83,20% | 93 (1,86%) | `full` (840) | 0,2764 | 2,0423 | 41,62 | `cobertura_baixa` |
| `hdbscan_leaf_mcs50_ms5` | Leaf | 9 | 955 | 19,10% | 80,90% | 270 (5,40%) | `full` (955) | 0,2471 | 2,1922 | 64,74 | `cobertura_baixa` |

Observações sobre cobertura e ruído:

- Há um compromisso direto entre cobertura e coesão interna aparente: as únicas configurações com cobertura acima de `50%` (`mcs20_ms5` e `mcs30_ms5`, `56,80%`) concentram `98,91%` dos artigos agrupados (`2.809 / 2.840`) em um único macrocluster (`cluster_dominante`, `Silhouette = 0,0950`).
- Configurações com múltiplos clusters equilibrados (`hdbscan_eom_mcs50_ms5` com `6` clusters; `hdbscan_leaf_mcs20_ms5` com `18` clusters; `hdbscan_leaf_mcs50_ms5` com `9` clusters) retêm apenas entre `16,80%` e `26,70%` do dataset, deixando `73,30%` a `83,20%` dos artigos marcados como `-1`.
- Silhouettes mais altas em `grouped_only` (como `0,2764` em `hdbscan_leaf_mcs20_ms5`) refletem exclusivamente a compacidade dos núcleos densos após descartar `83,20%` do corpus; **não** indicam partição global superior ao K-Means (`0,1472` sobre `100%` dos artigos).
- O rótulo `-1` indica apenas baixa densidade local segundo os parâmetros do HDBSCAN, não ausência de relevância científica nem ausência de relação interdisciplinar.

## 4. Estabilidade entre sementes do K-Means

O artefato `seed_stability.csv` consolida os experimentos de sensibilidade à inicialização (`stability_type="seed_initialization"`), calculados sobre `5` sementes (`11, 21, 42, 73, 101`, totalizando `10` pares de sementes por configuração).

| Configuração (`config_id`) | Representação | `k` | ARI médio | Desvio-padrão | ARI mínimo | ARI máximo | Fonte |
|---|---|---:|---:|---:|---:|---:|---|
| `kmeans_k5` | BGE 384d | 5 | 0,9979 | 0,0012 | 0,9956 | 1,0000 | `clustering/v1/kmeans_stability.csv` |
| `kmeans_k7` | BGE 384d | 7 | 0,9471 | 0,0594 | 0,8735 | 1,0000 | `clustering/v1/kmeans_stability.csv` |
| `kmeans_k8` | BGE 384d | 8 | 0,8621 | 0,0915 | 0,7518 | 0,9928 | `clustering/v1/kmeans_stability.csv` |
| `kmeans_k9` | BGE 384d | 9 | 0,8926 | 0,0628 | 0,8236 | 0,9961 | `clustering/v1/kmeans_stability.csv` |
| `tfidf_10k_k5` | TF-IDF 10k | 5 | 0,6578 | 0,1822 | 0,4742 | 0,9167 | `tfidf_baseline/v1/tfidf_stability.csv` |
| `tfidf_10k_k8` | TF-IDF 10k | 8 | 0,6692 | 0,0836 | 0,5701 | 0,8818 | `tfidf_baseline/v1/tfidf_stability.csv` |
| `tfidf_10k_k13` | TF-IDF 10k | 13 | 0,5044 | 0,0588 | 0,4440 | 0,6329 | `tfidf_baseline/v1/tfidf_stability.csv` |

Conclusões de estabilidade entre inicializações:

- No espaço BGE, `kmeans_k5` é quase invariante à semente (`ARI médio = 0,9979`, `ARI mínimo = 0,9956`), enquanto `kmeans_k8` apresenta variação moderada entre ótimos locais (`ARI médio = 0,8621`, `ARI mínimo = 0,7518`).
- No baseline TF-IDF 10k, a sensibilidade à semente é mais acentuada em todos os valores testados (`ARI médio` entre `0,5044` e `0,6692`), sendo `k=8` o mais estável entre os três candidatos lexicais (`ARI médio = 0,6692`, `ARI mínimo = 0,5701`).
- **Distinção conceitual obrigatória:** estes valores medem apenas reprodutibilidade de inicialização do K-Means sobre o corpus fixo de `5.000` artigos. Eles **não** avaliam estabilidade sob reamostragem (perturbação do conjunto de artigos), que não foi executada nesta entrega.

## 5. Concordância estrutural entre partições (ARI)

O módulo `stability.py` separa explicitamente o ARI entre inicializações (`seed_stability.csv`) do ARI entre configurações distintas (`partition_concordance.csv`, `276` pares classificados em `hyperparameter_sensitivity`, `cross_algorithm` e `cross_representation`).

| Tipo de comparação (`comparison_type`) | Escopo (`comparison_scope`) | Pares válidos | Pares inválidos | Total |
|---|---|---:|---:|---:|
| `hyperparameter_sensitivity` | `full` | 24 | 0 | 24 |
| `hyperparameter_sensitivity` | `grouped_intersection` | 39 | 16 | 55 |
| `cross_algorithm` | `full` | 12 | 0 | 12 |
| `cross_algorithm` | `grouped_intersection` | 66 | 11 | 77 |
| `cross_representation` | `full` | 42 | 0 | 42 |
| `cross_representation` | `grouped_intersection` | 66 | 0 | 66 |
| **Total** | — | **249** | **27** | **276** |

### 5.1. Matriz de concordância entre partições completas (`full_ari_matrix.csv`)

O artefato `full_ari_matrix.csv` contém a matriz simétrica `13 × 13` (`78` pares fora da diagonal, todos válidos sobre os `5.000` artigos). A tabela a seguir resume o subconjunto principal de referência (`7 × 7`):

| Configuração | `kmeans_k5` | `kmeans_k8` | `kmeans_k9` | `agg_average_k8` | `agg_complete_k8` | `agg_ward_k8` | `tfidf_10k_k8` | `tfidf_20k_k8` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `kmeans_k5` | 1,0000 | 0,7179 | 0,7046 | 0,0001 | 0,6468 | 0,5908 | 0,4164 | 0,4031 |
| `kmeans_k8` | 0,7179 | 1,0000 | 0,7868 | 0,0004 | 0,6002 | 0,7139 | 0,4420 | 0,4143 |
| `kmeans_k9` | 0,7046 | 0,7868 | 1,0000 | 0,0004 | 0,5431 | 0,6511 | 0,4660 | 0,4686 |
| `agg_average_k8` | 0,0001 | 0,0004 | 0,0004 | 1,0000 | 0,0007 | 0,0003 | 0,0004 | 0,0001 |
| `agg_complete_k8` | 0,6468 | 0,6002 | 0,5431 | 0,0007 | 1,0000 | 0,6082 | 0,4036 | 0,3677 |
| `agg_ward_k8` | 0,5908 | 0,7139 | 0,6511 | 0,0003 | 0,6082 | 1,0000 | 0,4035 | 0,3712 |
| `tfidf_10k_k8` | 0,4164 | 0,4420 | 0,4660 | 0,0004 | 0,4036 | 0,4035 | 1,0000 | 0,6709 |
| `tfidf_20k_k8` | 0,4031 | 0,4143 | 0,4686 | 0,0001 | 0,3677 | 0,3712 | 0,6709 | 1,0000 |

Principais padrões nas partições completas:

- **Sensibilidade a `k` no K-Means BGE:** forte continuidade hierárquica entre `k=7` e `k=8` (`ARI = 0,9009`) e entre `k=7` e `k=9` (`ARI = 0,8813`), indicando refinamento gradual das partições em torno de `7` a `9` grupos.
- **Concordância entre algoritmos no espaço BGE (`k=8`):** `kmeans_k8` e `agglomerative_ward_k8` atingem `ARI = 0,7139` (e `0,6002` frente a `agglomerative_complete_k8`). Esse resultado expressa **concordância estrutural entre algoritmos distintos**, não devendo ser confundido com estabilidade por semente.
- **Degenerescência de `agglomerative_average_k8`:** apresenta `ARI` próximo de zero (`-0,0004` a `0,0013`) contra todas as demais `12` partições completas.
- **Vocabulário TF-IDF (`10k` versus `20k`):** para `k=8`, `tfidf_10k_k8` e `tfidf_20k_k8` atingem `ARI = 0,6709` — valor próximo ao próprio `ARI médio` entre sementes do `tfidf_10k_k8` (`0,6692`), mostrando que dobrar o vocabulário para `20.000` termos altera a partição em magnitude comparável à própria variação de inicialização do K-Means esparso.

### 5.2. Comparações BGE versus TF-IDF

A comparação entre representações exige separar métricas dependentes de geometria de métricas combinatórias sobre rótulos:

1. **Incomparabilidade direta da Silhouette:** embora o K-Means BGE (`0,1189` a `0,1472`) apresente valores numéricos de Silhouette uma ordem de grandeza acima do K-Means TF-IDF (`0,0099` a `0,0156`), as duas representações têm dimensionalidade, esparsidade e concentração angular completamente distintas (`384` dimensões densas contra `10.000` ou `20.000` termos esparsos com menos de `1%` de entradas não nulas). Portanto, a diferença absoluta de Silhouette **não comprova superioridade científica** da representação semântica.
2. **Comparabilidade válida por ARI:** como os `5.000` artigos estão estritamente alinhados por `openalex_id`, o ARI compara apenas os pares de atribuição de documentos, independentemente do espaço vetorial de origem. Excluindo a partição degenerada `agglomerative_average_k8`, os `36` pares `full` entre BGE e TF-IDF situam-se entre `ARI = 0,2790` (`kmeans_k5` vs `tfidf_20k_k13`) e `ARI = 0,4686` (`kmeans_k9` vs `tfidf_20k_k8`), com média de `0,3792` (e `ARI = 0,4420` entre os candidatos principais `kmeans_k8` e `tfidf_10k_k8`).
3. **Interpretação:** existe estrutura macrotemática compartilhada entre vocabulário explícito (TF-IDF) e similaridade semântica densa (BGE), mas mais da metade da organização fina diverge entre as duas representações — motivando a inspeção qualitativa e temática na Issue #09.

### 5.3. ARI sobre interseções parciais (`grouped_intersection`) e pares inválidos

Nas `198` comparações envolvendo ao menos uma configuração HDBSCAN, o ARI é calculado exclusivamente sobre os artigos agrupados simultaneamente por ambas as partições (`(label_a != -1) & (label_b != -1)`), registrando `n_intersection`, `intersection_coverage_pct` e a contagem de clusters presentes na interseção.

| Par comparado | Escopo | Interseção (`n`) | Cobertura da interseção | Clusters na interseção (`A` / `B`) | ARI | Globalmente idênticas? |
|---|---|---:|---:|---|---:|---|
| `hdbscan_eom_mcs30_ms10` vs `hdbscan_eom_mcs50_ms10` | `grouped_intersection` | 1.298 | 25,96% | 4 / 3 | 1,0000 | Não (`1.342` vs `1.298` agrupados) |
| `hdbscan_eom_mcs30_ms10` vs `hdbscan_eom_mcs50_ms20` | `grouped_intersection` | 1.053 | 21,06% | 3 / 3 | 0,9984 | Não |
| `kmeans_k9` vs `hdbscan_leaf_mcs50_ms5` | `grouped_intersection` | 955 | 19,10% | 9 / 9 | 0,8977 | Não |
| `agglomerative_ward_k8` vs `hdbscan_leaf_mcs50_ms5` | `grouped_intersection` | 955 | 19,10% | 7 / 9 | 0,8261 | Não |
| `kmeans_k8` vs `hdbscan_leaf_mcs50_ms5` | `grouped_intersection` | 955 | 19,10% | 8 / 9 | 0,8127 | Não |
| `kmeans_k8` vs `hdbscan_eom_mcs50_ms5` | `grouped_intersection` | 1.335 | 26,70% | 8 / 6 | 0,6443 | Não |
| `agglomerative_average_k8` vs todas as `11` HDBSCAN | `grouped_intersection` | 840–2.840 | 16,80%–56,80% | 1 / 2–18 | Inválido* | Não |
| `hdbscan_eom_mcs20_ms5` / `mcs30_ms5` vs `8` outras HDBSCAN | `grouped_intersection` | 955–1.342 | 19,10%–26,84% | 1 / 3–9 | Inválido* | Não |

*\*`valid=False`, `ari` ausente, `ari_unavailable_reason="menos_de_2_clusters_na_intersecao"`.*

Resultados metodológicos centrais sobre interseções parciais:

1. **ARI elevado em interseção pequena não representa concordância global:** por exemplo, `kmeans_k8` e `hdbscan_leaf_mcs50_ms5` atingem `ARI = 0,8127` (e `kmeans_k9` atinge `0,8977`), mas apenas sobre os `955` artigos mais densos (`19,10%` do corpus). Isso mostra que os núcleos densos extraídos pelo HDBSCAN Leaf alinham-se fortemente às divisões do K-Means nesse subconjunto específico, nada informando sobre os `4.045` artigos restantes (`80,90%`).
2. **`ARI = 1,0` parcial sem identidade global:** `hdbscan_eom_mcs30_ms10` (`1.342` artigos em `4` clusters) e `hdbscan_eom_mcs50_ms10` (`1.298` artigos em `3` clusters) concordam perfeitamente (`ARI = 1,0000`) nos `1.298` artigos em comum (`25,96%`), porque a elevação de `min_cluster_size` de `30` para `50` apenas converteu em ruído um quarto cluster de `44` artigos sem alterar os outros três grupos. Ainda assim, as partições completas sobre os `5.000` artigos são distintas (`identical_partition=False`).
3. **Causa exata dos `27` pares inválidos:**
   - `11` pares ocorrem entre `agglomerative_average_k8` e todas as `11` configurações HDBSCAN: todos os artigos agrupados pelo HDBSCAN caem integralmente dentro do macrocluster dominante de `4.983` artigos do linkage Average, restando apenas `1` cluster de `agglomerative_average_k8` na interseção.
   - `16` pares ocorrem entre `hdbscan_eom_mcs20_ms5` (e sua partição gêmea `hdbscan_eom_mcs30_ms5`) contra `8` outras configurações HDBSCAN mais restritivas: nestas interseções (`955` a `1.342` artigos), o cluster menor de `31` artigos desaparece no ruído da outra configuração e sobra apenas `1` cluster na interseção. Em vez de retornar `ARI = 0,0` artificial, o pipeline marca o par como `valid=False` com `menos_de_2_clusters_na_intersecao`.

## 6. Configurações distintas com partições globalmente idênticas

A verificação por assinatura canônica SHA-256 (invariante à permutação de rótulos de clusters e sensível ao conjunto exato de ruído `-1` nos `5.000` artigos) identificou exatamente **dois pares** de configurações que produziram a mesma partição global, exportados em `identical_partitions.csv`:

| `config_a` | `config_b` | Escopo | Clusters | Agrupados | Ruído | Cobertura | `exact_label_match` | ARI | Assinatura canônica (prefixo) |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| `hdbscan_eom_mcs20_ms5` | `hdbscan_eom_mcs30_ms5` | `grouped_only` | 2 | 2.840 | 2.160 | 56,80% | `True` | 1,0000 | `29206e626c3adddd...` |
| `hdbscan_eom_mcs50_ms10` | `hdbscan_eom_mcs100_ms10` | `grouped_only` | 3 | 1.298 | 3.702 | 25,96% | `True` | 1,0000 | `5282a58f093b47f0...` |

Interpretação:

- Em `min_samples=5`, o menor cluster selecionado por EOM em `min_cluster_size=20` já possuía `31` artigos; elevar `min_cluster_size` para `30` não podou nenhum ramo da árvore condensada do HDBSCAN.
- De forma análoga, em `min_samples=10`, o menor cluster selecionado em `min_cluster_size=50` já continha `135` artigos (`> 100`), tornando `min_cluster_size=100` inócuo na seleção EOM.
- Ambas as configurações de cada par foram preservadas em `evaluation_metrics.csv` e `partition_concordance.csv` para rastreabilidade integral do grid experimental, mas não devem ser contabilizadas como soluções independentes na seleção final de candidatos.

## 7. Artefatos exportados e reprodutibilidade

O pipeline da Issue #08 está organizado em `src/research_atlas/evaluation/`:

- `metrics.py`: validação geométrica, diagnóstico de partições, amostragem segura da Silhouette cosseno e suporte a matrizes esparsas sem densificação;
- `pipeline.py`: carregamento verificado por SHA-256, alinhamento de IDs, avaliação consolidada das `24` configurações e reconciliação com as Issues #06 e #07;
- `stability.py`: consolidação da estabilidade entre sementes, assinatura canônica de partições e cálculo de concordância ARI em escopo completo e em interseção agrupada;
- `export.py`: interface de execução explícita e escrita atômica em diretório temporário com verificação pós-escrita de hashes SHA-256.

Comando de reprodução determinística:

```bash
PYTHONPATH=src uv run python -m research_atlas.evaluation.export
```

Artefatos gerados em `data/processed/evaluation/v1/` (`run_id` determinístico `b11fb931938fc7f4575495ff6523a3786baf3361bebb1288d79c833b3917c4d9`):

| Arquivo | Registros / Dimensão | SHA-256 |
|---|---|---|
| `evaluation_metrics.csv` | `24` linhas × `42` colunas | `a036e430400aeb3296faae07dc9322320594f62c627e97d9fd3273a3a0bef860` |
| `seed_stability.csv` | `7` linhas × `14` colunas | `3cedcbb8a5a3adc044e86ba0565a24e14fb711642fa42cd901cb0bb3733666aa` |
| `partition_concordance.csv` | `276` linhas × `26` colunas | `3537635324d8d2df0686c93ee47117a1edb26a58b030bdaa986c57926e4a3c61` |
| `full_ari_matrix.csv` | Matriz `13 × 13` | `3337f00539487ece8811fc477ae1507337048d1467373c78630ede9d82d891ec` |
| `identical_partitions.csv` | `2` linhas × `15` colunas | `1c2a6931633ad487473ed6c2db715be4ca9a6b67508df17ecab427f9aa682cad` |
| `manifest.json` | Manifesto (`16` insumos + `5` saídas) | `3988a6b121c7804e9ef317aae8e1c529d9a6467b2bee9967cef3db2b8b9f3466` |

Duas execuções consecutivas produzem exatamente os mesmos bytes e os mesmos hashes SHA-256 (nenhum timestamp variável de relógio é gravado no manifesto).

## 8. Limitações metodológicas e resultados inconclusivos

1. **Ausência de estabilidade sob reamostragem:** a avaliação de estabilidade restringiu-se à variação entre sementes do K-Means (`seed_initialization`). Não foram executados experimentos de subamostragem/bootstrap do corpus; portanto, a sensibilidade das partições K-Means, Agglomerative e HDBSCAN à remoção ou substituição de artigos permanece como limitação científica aberta.
2. **Ausência de Davies-Bouldin e Calinski-Harabasz no TF-IDF:** por decisão metodológica de não densificar matrizes esparsas de alta dimensão, a comparação interna do TF-IDF apoiou-se apenas em Silhouette cosseno, distribuição de tamanhos e estabilidade por semente.
3. **Inconclusividade de uma métrica única para escolha de `k`:** no espaço BGE, `k=5` maximiza a estabilidade por semente (`0,9979`) e CH entre os candidatos salvos (`255,21`), `k=8` maximiza a Silhouette (`0,1472`) e `k=9` minimiza DB entre partições equilibradas (`2,9904`). Assim, `kmeans_k8` (junto a `kmeans_k5`, `agglomerative_ward_k8`, `tfidf_10k_k8` e núcleos densos como `hdbscan_eom_mcs50_ms5` / `hdbscan_leaf_mcs50_ms5`) deve ser tratado como **candidato analítico complementar**, não como número verdadeiro demonstrado de comunidades.
4. **Cobertura limitada nos núcleos HDBSCAN:** todas as configurações HDBSCAN não degeneradas excluem entre `73,16%` e `83,20%` dos artigos, impedindo seu uso isolado como mapa global do corpus.
5. **Interdisciplinaridade ainda não validada:** nem métricas geométricas internas nem concordância por ARI comprovam que clusters mistos representem comunidades científicas genuinamente interdisciplinares. Tanto a auditoria humana de `100` artigos do dataset quanto a validação qualitativa das pontes temáticas permanecem pendentes.

## 9. Conclusão e encaminhamento para a Issue #09

A Issue #08 consolidou uma base quantitativa auditável, determinística e livre de comparações metodologicamente inválidas para as `24` configurações experimentais do ResearchAtlas:

- Para **organização temática global**, `kmeans_k8` e `agglomerative_ward_k8` no espaço BGE apresentam equilíbrio de tamanhos e elevada concordância mútua (`ARI = 0,7139`), enquanto `kmeans_k5` oferece referência macro mais estável à inicialização (`ARI médio = 0,9979`) e `tfidf_10k_k8` atua como contraponto lexical (`ARI = 0,4420` frente a `kmeans_k8`).
- Para **inspeção de núcleos temáticos altamente coesos**, configurações como `hdbscan_eom_mcs50_ms5` (`26,70%` de cobertura, `6` clusters) e `hdbscan_leaf_mcs50_ms5` (`19,10%` de cobertura, `9` clusters, fortemente alinhados aos grupos do K-Means em sua interseção) funcionam como lentes locais complementares, desde que reportadas junto à taxa de ruído.
- Configurações degeneradas (`agglomerative_average_k8`, `hdbscan_eom_mcs20_ms5`) e duplicadas (`hdbscan_eom_mcs30_ms5`, `hdbscan_eom_mcs100_ms10`) foram formalmente diagnosticadas.

A próxima etapa científica concentra-se na **Issue #09 — Interpretação e validação interdisciplinar**, dedicada à análise de termos e artigos representativos, cruzamento com áreas de amostragem, identificação de possíveis pontes temáticas e validação humana.
