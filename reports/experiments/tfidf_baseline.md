# Baseline TF-IDF + K-Means — ResearchAtlas

**Issue:** #06 — Implementar baseline TF-IDF + K-Means  
**Dataset:** ResearchAtlas v1  
**Status:** experimentos e validação automatizada concluídos

## 1. Objetivo

Implementar um baseline de clusterização textual utilizando TF-IDF e K-Means, permitindo avaliar uma representação lexical tradicional dos artigos científicos e estabelecer uma referência para futuras comparações com embeddings semânticos.

O experimento não utiliza as áreas científicas como rótulos de treinamento.

## 2. Dataset

O experimento utiliza 5.000 artigos científicos publicados entre 2020 e 2025, distribuídos igualmente entre Medicina, Astronomia, Biologia, Clima e Engenharia.

Os textos foram construídos pela concatenação de título e resumo, preservando a ordem original dos registros.

A integridade do dataset foi verificada por SHA-256:

`533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389`

## 3. Representação TF-IDF

A configuração inicial utilizou:

| Parâmetro | Valor |
|---|---|
| lowercase | True |
| stop_words | english |
| ngram_range | (1, 2) |
| min_df | 3 |
| max_df | 0.95 |
| max_features | 20.000 |
| sublinear_tf | True |
| norm | l2 |

A matriz resultante apresentou 5.000 linhas, 20.000 características e 648.816 elementos não nulos, com densidade aproximada de 0,6488%.

A representação foi mantida em formato esparso CSR.

## 4. Experimento inicial

Aplicou-se K-Means com cinco clusters, `random_state=42` e `n_init=10`.

O experimento apresentou Silhouette de 0,0099 utilizando distância cosseno e amostra de 2.000 artigos.

O maior cluster reuniu 1.845 artigos, correspondendo a 36,9% do dataset.

## 5. Sensibilidade ao número de clusters

Foram avaliados valores de `k` entre 2 e 15, mantendo fixa a representação TF-IDF inicial.

A maior Silhouette foi 0,0140 para `k=13`.

Entretanto, a diferença em relação a `k=10`, cuja Silhouette foi 0,0138, foi pequena. Nenhuma configuração apresentou forte separação segundo essa métrica.

A inércia diminuiu conforme o número de clusters aumentou, comportamento esperado do K-Means.

## 6. Sensibilidade aos parâmetros TF-IDF

Com `k=8`, foram comparadas quatro configurações:

| Configuração | Características | Silhouette |
|---|---:|---:|
| Baseline | 20.000 | 0,0124 |
| Apenas unigramas | 13.137 | 0,0137 |
| Vocabulário reduzido | 10.000 | 0,0146 |
| min_df=5 | 20.000 | 0,0125 |

A representação com vocabulário limitado a 10.000 características apresentou a maior Silhouette entre as configurações testadas.

Essa observação não demonstra que reduzir o vocabulário sempre melhora a clusterização.

## 7. Estabilidade entre inicializações

Utilizando a representação TF-IDF de 10.000 características, foram testadas cinco sementes: 11, 21, 42, 73 e 101.

A estabilidade foi quantificada por meio do Adjusted Rand Index (ARI) entre pares de execuções.

| k | Silhouette | ARI médio | ARI mínimo |
|---|---:|---:|---:|
| 5 | 0,0130 | 0,6578 | 0,4742 |
| 8 | 0,0146 | 0,6692 | 0,5701 |
| 13 | 0,0156 | 0,5044 | 0,4440 |

O valor `k=8` foi selecionado como configuração principal por apresentar o maior ARI médio e mínimo entre os candidatos, além de uma distribuição menos concentrada que `k=5`.

A escolha representa um compromisso entre separação, estabilidade e distribuição dos agrupamentos, não a identificação comprovada do número ideal de comunidades científicas.

## 8. Interpretação temática exploratória

Os oito clusters da configuração principal apresentaram os seguintes temas preliminares:

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

Os temas foram propostos a partir dos termos mais relevantes nos centroides e da inspeção dos títulos representativos.

O cluster 5 apresentou composição multidisciplinar, contendo artigos relacionados a aplicações de redes neurais e análise de imagens. Isso sugere similaridade metodológica entre disciplinas, mas não constitui evidência suficiente de interdisciplinaridade científica efetiva.

A validação temática aprofundada pertence à Issue #09.

## 9. Comparação exploratória com embeddings BGE

Na configuração `k=8`, foram observados:

| Representação | Silhouette | ARI médio |
|---|---:|---:|
| TF-IDF 10k | 0,0146 | 0,6692 |
| BGE-small-en-v1.5 | 0,1472 | 0,8621 |

As duas representações foram avaliadas sobre os mesmos 5.000 artigos, mas em espaços vetoriais diferentes.

A maior estabilidade observada para BGE é relevante, embora a comparação das Silhouettes absolutas não seja suficiente para estabelecer superioridade científica.

A comparação aprofundada de métodos e métricas será tratada nas Issues #07 e #08.

## 10. Reprodutibilidade

O experimento utiliza `random_state=42`, `n_init=10` e Silhouette com distância cosseno e `sample_size=2000`.

O pipeline reutilizável está implementado em `src/research_atlas/features/tfidf.py`.

Os resultados são preservados em `data/processed/tfidf_baseline/v1/`, incluindo rótulos por artigo, tabelas de métricas e manifesto com parâmetros e hashes SHA-256.

Os testes automatizados validam o processamento, a execução do K-Means, a avaliação, a integridade dos artefatos e a correspondência dos artigos com o dataset original.

## 11. Limitações

- A Silhouette permaneceu próxima de zero nas configurações avaliadas.
- A escolha de hiperparâmetros foi baseada em uma busca limitada, não exaustiva.
- O ARI entre inicializações não representa estabilidade sob reamostragem.
- Os agrupamentos podem refletir vocabulário compartilhado, não necessariamente comunidades científicas.
- A interpretação temática é exploratória e ainda não foi validada por avaliadores humanos.
- Os resultados dependem da versão do dataset e das bibliotecas utilizadas.
- A comparação entre representações TF-IDF e BGE exige avaliação metodológica adicional.

## 12. Conclusão

O baseline TF-IDF + K-Means foi implementado e avaliado de maneira reproduzível.

A configuração principal selecionada utiliza 10.000 características TF-IDF e K-Means com oito clusters. Embora a separação medida pela Silhouette seja baixa, os agrupamentos apresentam temas preliminares interpretáveis.

Os resultados constituem uma referência para os experimentos posteriores de clusterização semântica do ResearchAtlas.