# Comparação de algoritmos de clusterização semântica — ResearchAtlas

**Issue:** #07 — Comparar K-Means, HDBSCAN e Agglomerative com embeddings  
**Dataset:** ResearchAtlas v1  
**Representação:** BAAI/bge-small-en-v1.5  
**Status:** experimentos concluídos; interpretação científica preliminar

## 1. Objetivo

Comparar três abordagens de clusterização não supervisionada sobre uma representação semântica comum de artigos científicos: K-Means, Agglomerative Clustering e HDBSCAN.

O objetivo é investigar diferenças na formação dos agrupamentos, na seleção de parâmetros e na capacidade de identificar comunidades temáticas, preservando a rastreabilidade dos rótulos e resultados experimentais.

## 2. Dados e representação

Foram utilizados 5.000 artigos científicos do ResearchAtlas v1, distribuídos igualmente entre Medicina, Astronomia, Biologia, Clima e Engenharia.

Cada artigo foi representado por um embedding BGE-small-en-v1.5 de 384 dimensões, com normalização L2.

Os embeddings foram produzidos na Issue #05 e carregados sem necessidade de reprocessamento.

**SHA-256 dos embeddings:**

`13fe24f37622790f6d6630fbdb5e91742060f95a8d7310597d3bd98f2a7f465c`

As áreas científicas não foram utilizadas como rótulos de treinamento.

## 3. K-Means

### 3.1. Configuração

Foram avaliados valores de `k` entre 2 e 15, utilizando:

- `random_state=42`;
- `n_init=10`;
- embeddings normalizados;
- distância cosseno na avaliação por Silhouette;
- amostra de 2.000 artigos para a Silhouette.

### 3.2. Resultados

A maior Silhouette observada foi **0,1472 para k=8**.

O menor Davies-Bouldin foi **2,9904 para k=9**, enquanto o maior Calinski-Harabasz foi **361,87 para k=2**.

As métricas favoreceram configurações diferentes, não permitindo determinar um número ideal de clusters apenas por um indicador.

### 3.3. Estabilidade entre inicializações

Foram comparadas cinco sementes: 11, 21, 42, 73 e 101.

| k | ARI médio | Desvio-padrão |
|---|---:|---:|
| 5 | 0,9979 | 0,0012 |
| 7 | 0,9471 | 0,0594 |
| 8 | 0,8621 | 0,0915 |
| 9 | 0,8926 | 0,0628 |

O valor `k=5` apresentou a maior consistência entre inicializações.

O valor `k=8` foi mantido como candidato para investigação temática por apresentar a maior Silhouette no intervalo testado.

Essas medidas representam estabilidade entre inicializações sobre o mesmo dataset, não estabilidade sob reamostragem.

## 4. Agglomerative Clustering

Foram comparadas três estratégias com oito clusters.

| Linkage | Distância | Silhouette | Maior cluster |
|---|---|---:|---:|
| Average | Cosseno | 0,1082 | 4.983 |
| Complete | Cosseno | 0,1112 | 1.008 |
| Ward | Euclidiana | 0,1142 | 949 |

A configuração Average + cosseno produziu uma partição extremamente concentrada, com 99,66% dos artigos em um único grupo.

Complete + cosseno produziu agrupamentos mais distribuídos, mas ainda apresentou grupos muito pequenos.

Ward + euclidiana apresentou a melhor Silhouette entre as três estratégias e distribuiu os artigos em grupos de 350 a 949 elementos.

O ARI entre Ward e K-Means, ambos com `k=8`, foi 0,7139.

A estratégia Ward foi mantida como principal referência hierárquica, sem pressupor superioridade geral sobre os demais métodos.

## 5. HDBSCAN

O HDBSCAN foi avaliado com distância euclidiana sobre embeddings normalizados.

Foram examinadas duas estratégias de seleção de clusters: EOM e Leaf.

### 5.1. Seleção EOM

Foram testadas nove combinações de `min_cluster_size` e `min_samples`.

| Configuração | Clusters | Cobertura | Ruído |
|---|---:|---:|---:|
| (20, 5) | 2 | 56,80% | 43,20% |
| (20, 10) | 8 | 22,36% | 77,64% |
| (30, 10) | 4 | 26,84% | 73,16% |
| (50, 5) | 6 | 26,70% | 73,30% |
| (50, 10) | 3 | 25,96% | 74,04% |
| (100, 5) | 4 | 25,26% | 74,74% |

A configuração (20, 5) apresentou a maior cobertura, mas concentrou 2.809 dos 2.840 artigos agrupados em um único cluster.

A configuração (50, 5) produziu seis clusters com tamanhos entre 72 e 488 artigos, agrupando 26,70% do dataset.

### 5.2. Seleção Leaf

| Configuração | Clusters | Cobertura | Silhouette dos agrupados |
|---|---:|---:|---:|
| (20, 5) | 18 | 16,80% | 0,2764 |
| (50, 5) | 9 | 19,10% | 0,2471 |

A seleção Leaf produziu comunidades menores e mais específicas, mas deixou a maior parte dos artigos sem atribuição.

As Silhouettes foram calculadas exclusivamente sobre os artigos agrupados. Portanto, não são diretamente comparáveis às Silhouettes das partições completas.

O rótulo `-1` representa ruído segundo o algoritmo, não ausência de relevância científica.

## 6. Comparação metodológica

Os algoritmos apresentaram comportamentos distintos.

**K-Means:** produziu partições completas e permitiu experimentar diferentes quantidades de clusters. Apresentou a maior Silhouette observada entre as partições completas principais.

**Agglomerative Ward:** produziu uma partição completa com distribuição relativamente equilibrada, funcionando como alternativa hierárquica.

**HDBSCAN:** identificou subconjuntos densos e comunidades menores, mas apresentou cobertura reduzida e sensibilidade aos parâmetros.

Os resultados favorecem o uso do K-Means como candidato para organização global dos artigos e do HDBSCAN como método exploratório de núcleos densos. Essas escolhas ainda exigem validação temática.

## 7. Interpretação temática preliminar

Para `k=8`, os clusters do K-Means apresentaram temas preliminares relacionados a:

- Manutenção preditiva industrial;
- Estruturas moleculares e criomicroscopia;
- Classificação e observações astronômicas;
- Controle e otimização de sistemas;
- Genômica e predição molecular;
- Modelagem climática;
- Diagnóstico médico por imagens;
- Aplicações gerais de IA na saúde.

O último grupo apresentou artigos de todas as áreas científicas, mas foi dominado por Medicina.

Uma amostra exploratória de 25 artigos, com cinco representantes por área, foi preparada para avaliação humana independente.

A composição multidisciplinar não foi tratada como comprovação automática de interdisciplinaridade.

A validação científica aprofundada pertence à Issue #09.

## 8. Artefatos e reprodutibilidade

O notebook experimental está disponível em:

`notebooks/02_clusterizacao.ipynb`

O módulo reutilizável de execução dos algoritmos está localizado em:

`src/research_atlas/clustering/algorithms.py`

Os artefatos estão organizados em:

`data/processed/clustering/v1/`

Arquivos:

- `cluster_assignments.parquet`;
- `kmeans_metrics.csv`;
- `kmeans_stability.csv`;
- `agglomerative_metrics.csv`;
- `hdbscan_eom_metrics.csv`;
- `hdbscan_leaf_metrics.csv`;
- `manifest.json`.

O arquivo Parquet contém 5.000 artigos e rótulos de 18 configurações experimentais, preservando o vínculo entre cada artigo e seu embedding original.

O manifesto registra a versão da representação, os parâmetros das execuções e os hashes SHA-256 dos artefatos.

Os testes automatizados verificam a execução dos algoritmos, a integridade dos arquivos, a correspondência dos identificadores e a consistência dos rótulos com as métricas exportadas.

## 9. Limitações

- Os resultados dependem da versão dos embeddings e do dataset.
- A escolha dos hiperparâmetros foi exploratória e não exaustiva.
- O K-Means requer especificar previamente a quantidade de clusters.
- O Agglomerative é sensível ao método de ligação.
- O HDBSCAN apresentou cobertura reduzida nas configurações testadas.
- A Silhouette do HDBSCAN foi calculada apenas sobre os artigos agrupados nos experimentos Leaf.
- O ARI entre inicializações não avalia estabilidade sob perturbações do dataset.
- As métricas internas não comprovam interdisciplinaridade.
- A interpretação temática depende de validação humana.
- O tempo de execução e o consumo de memória não foram medidos de maneira sistemática e comparável.

## 10. Conclusão

Os três algoritmos foram implementados e avaliados sobre a mesma representação semântica dos 5.000 artigos.

O K-Means apresentou resultados adequados para uma organização global exploratória. O Agglomerative Ward ofereceu uma alternativa hierárquica, enquanto o HDBSCAN identificou núcleos densos com cobertura limitada.

Os experimentos foram preservados com rótulos por artigo, parâmetros rastreáveis e testes automatizados.

A avaliação aprofundada das métricas e da estabilidade continuará na Issue #08. A investigação das conexões interdisciplinares e a validação temática serão aprofundadas na Issue #09.