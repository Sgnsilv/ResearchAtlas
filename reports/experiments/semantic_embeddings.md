# ResearchAtlas — Geração de embeddings semânticos

## 1. Objetivo

Construir representações vetoriais de artigos científicos para utilização nos experimentos de clusterização e investigação de comunidades de pesquisa interdisciplinares.

## 2. Dataset

Foi utilizada a versão de referência v1 do ResearchAtlas, contendo 5.000 artigos científicos distribuídos entre Medicina, Astronomia, Biologia, Clima e Engenharia.

**Arquivo:** `dataset_selected.parquet`

**SHA-256 do dataset:**

`533fe010fee10dcc87e4ec196886e8a8d9ec6e17f0f4f3d507dc0cf278c79389`

A integridade do arquivo foi verificada antes do processamento.

Os textos foram construídos pela concatenação de título e resumo, preservando o conteúdo linguístico original, sem remoção de stopwords ou stemming.

Os 195 artigos sinalizados por possíveis problemas textuais foram mantidos, permitindo análises posteriores de sensibilidade.

## 3. Ambiente de execução

- Plataforma: Google Colab
- Acelerador: NVIDIA Tesla T4
- Framework: PyTorch
- Biblioteca: Sentence Transformers
- Processamento: GPU
- Batch size: 16

## 4. Seleção do modelo

Foram avaliados dois modelos de embeddings:

| Modelo | Dimensões | Limite de tokens | Artigos acima do limite |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 384 | 256 | 3.552 (71,04%) |
| BAAI/bge-small-en-v1.5 | 384 | 512 | 412 (8,24%) |

O BGE-small-en-v1.5 foi selecionado para a representação inicial por permitir processar integralmente uma proporção maior dos textos, mantendo embeddings de 384 dimensões.

Essa escolha considera cobertura textual e custo computacional. Não constitui evidência de superioridade na qualidade dos agrupamentos, que deverá ser verificada experimentalmente.

## 5. Segmentação textual

Os textos que excediam o limite do BGE foram divididos em trechos utilizando os IDs originais do tokenizador.

Parâmetros:

| Parâmetro | Valor |
|---|---:|
| Limite total | 512 tokens |
| Tokens especiais | 2 |
| Conteúdo por trecho | 510 tokens |
| Sobreposição | 64 tokens |

A segmentação foi realizada diretamente sobre os IDs dos tokens, evitando diferenças produzidas por decodificação e retokenização.

### Resultados

| Indicador | Resultado |
|---|---:|
| Artigos processados | 5.000 |
| Artigos com um trecho | 4.588 |
| Artigos com múltiplos trechos | 412 |
| Total de trechos | 5.678 |
| Máximo de trechos por artigo | 12 |
| Maior sequência enviada ao modelo | 512 tokens |

Todos os trechos respeitaram o limite configurado.

## 6. Geração dos embeddings

Os 5.678 trechos foram processados pelo modelo BGE-small-en-v1.5.

Foi utilizada a representação `sentence_embedding` do Sentence Transformers, seguida de normalização L2.

O processamento por IDs foi comparado ao método `encode()` em uma amostra de artigos não segmentados, apresentando similaridade praticamente igual a 1.

### Agregação por artigo

Para artigos com múltiplos trechos, foi calculada uma média ponderada dos embeddings normalizados.

Os pesos consideraram a quantidade de tokens novos em cada trecho, descontando a sobreposição.

Após a agregação, cada vetor foi normalizado novamente com norma L2.

Dessa forma, cada artigo foi representado por exatamente um vetor.

## 7. Resultados

| Indicador | Resultado |
|---|---|
| Quantidade de embeddings | 5.000 |
| Dimensão | 384 |
| Formato | float32 |
| Valores não finitos | Nenhum |
| Normalização L2 | Validada |
| Correspondência com OpenAlex ID | Validada |

O resultado é uma matriz de dimensão **5.000 × 384**.

## 8. Artefatos

Os artefatos foram inicialmente preservados no Google Drive:

`ResearchAtlas/embeddings/bge-small-en-v1.5-v1/`

Arquivos:

- `embeddings.npy`: matriz de embeddings.
- `metadata.parquet`: metadados e correspondência entre artigos e vetores.
- `manifest.json`: identificação do modelo, parâmetros e hashes.

**SHA-256 dos embeddings:**

`13fe24f37622790f6d6630fbdb5e91742060f95a8d7310597d3bd98f2a7f465c`

## 9. Validação

Foram verificados:

- Integridade do dataset de entrada.
- Unicidade dos 5.000 identificadores.
- Limites dos trechos tokenizados.
- Equivalência entre a implementação do notebook e o módulo reutilizável.
- Dimensão dos embeddings.
- Ausência de valores não finitos.
- Normalização dos vetores.
- Correspondência entre índices e metadados.
- Integridade dos artefatos persistidos.

## 10. Limitações

A estratégia de chunking preserva a cobertura textual, mas o processamento independente de trechos não modela diretamente relações contextuais entre segmentos distantes.

A média ponderada é uma estratégia inicial de agregação e poderá perder informações específicas de trechos individuais.

A repetição de tokens na sobreposição também pode influenciar as representações.

Além disso, a qualidade científica dos artigos ainda depende da auditoria de relevância, e alguns resumos podem conter textos inadequados mesmo quando estão estruturalmente válidos.

A qualidade semântica dos embeddings será avaliada posteriormente nos experimentos de clusterização.

## 11. Próximas etapas

- Disponibilizar os artefatos aos demais integrantes do projeto.
- Implementar validação automatizada de carregamento e integridade.
- Comparar representações e estratégias de processamento.
- Executar os algoritmos de clusterização previstos no protocolo.
- Avaliar a estabilidade e a interpretação dos agrupamentos.

**Estado:** geração e validação dos embeddings concluídas; avaliação experimental dos agrupamentos pendente.