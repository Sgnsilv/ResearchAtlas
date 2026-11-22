# Dataset piloto — Coleta, qualidade e análise exploratória

## 1. Objetivo

Construir e avaliar um dataset piloto de artigos científicos sobre aplicações de inteligência artificial em cinco áreas: Medicina, Astronomia, Biologia, Clima e Engenharia.

Esta etapa tem caráter exploratório e busca validar o pipeline de coleta, pré-processamento, análise de qualidade textual e seleção de artigos para auditoria científica.

## 2. Metodologia

### 2.1 Fonte dos dados

Os artigos foram recuperados por meio da API OpenAlex, utilizando os termos:

- `machine learning`
- `deep learning`
- `artificial intelligence`
- `neural networks`

A coleta considerou publicações entre 2020 e 2025, em inglês, classificadas como artigos e com resumos disponíveis na fonte.

Foram recuperados até 20 registros por combinação de área e termo de busca.

### 2.2 Pré-processamento

O pipeline realizou:

1. Reconstrução dos resumos a partir dos índices invertidos da OpenAlex.
2. Normalização textual e dos identificadores.
3. Consolidação de registros duplicados pelo identificador OpenAlex.
4. Preservação das áreas e dos termos utilizados na recuperação.
5. Validação estrutural dos artigos.
6. Sinalização de possíveis problemas na qualidade textual dos resumos.

### 2.3 Critérios de qualidade textual

Foram definidos indicadores exploratórios:

| Indicador | Critério |
|---|---|
| `short_abstract` | Menos de 50 palavras |
| `long_abstract` | Mais de 500 palavras |
| `suspicious_content` | Presença de padrões textuais potencialmente inadequados |

Esses indicadores não provocam exclusão automática. Os registros sinalizados exigem avaliação posterior.

## 3. Resultados

### 3.1 Coleta e validação

| Indicador | Resultado |
|---|---:|
| Registros recuperados | 400 |
| Artigos únicos | 398 |
| Ocorrências duplicadas | 2 |
| Artigos estruturalmente válidos | 398 |
| Artigos estruturalmente inválidos | 0 |

### 3.2 Distribuição por área de coleta

| Área | Artigos |
|---|---:|
| Medicina | 80 |
| Astronomia | 80 |
| Biologia | 80 |
| Clima | 80 |
| Engenharia | 78 |

Nenhum artigo foi recuperado por mais de uma área no dataset piloto.

### 3.3 Distribuição temporal

| Ano | Artigos |
|---|---:|
| 2020 | 97 |
| 2021 | 105 |
| 2022 | 79 |
| 2023 | 57 |
| 2024 | 40 |
| 2025 | 20 |

Observou-se concentração nos anos iniciais. Os artigos de 2020 e 2021 representam aproximadamente 50,8% da amostra.

Essa distribuição pode estar relacionada à estratégia de recuperação dos primeiros resultados por relevância e não deve ser interpretada como tendência temporal da produção científica.

### 3.4 Qualidade dos resumos

O comprimento médio dos resumos foi de aproximadamente 196,61 palavras, com mediana de 194,5 palavras, mínimo de 7 e máximo de 1.223 palavras.

| Indicador | Artigos |
|---|---:|
| Resumos curtos | 11 |
| Conteúdo suspeito | 3 |
| Resumos longos | 1 |
| Artigos com pelo menos um alerta | 13 |

Os indicadores podem ocorrer simultaneamente em um mesmo artigo.

Foram encontrados exemplos de textos bibliográficos, páginas de pré-visualização, resumos aparentemente truncados e declarações administrativas no lugar do conteúdo científico esperado.

## 4. Auditoria científica

Foi gerada uma amostra de 100 artigos, com 20 registros por área e seleção reproduzível, buscando distribuir os artigos entre os anos disponíveis.

A distribuição temporal da amostra foi:

| Ano | Artigos |
|---|---:|
| 2020 | 20 |
| 2021 | 20 |
| 2022 | 17 |
| 2023 | 16 |
| 2024 | 15 |
| 2025 | 12 |

Foram gerados dois arquivos CSV para avaliação independente por duas avaliadoras.

A auditoria deverá considerar:

- Relevância do artigo como aplicação de IA à área científica.
- Adequação do resumo para análise semântica.
- Justificativa para classificações duvidosas.

**Estado:** amostra preparada; avaliações humanas ainda pendentes.

Após a avaliação, poderão ser calculados o percentual de relevância, a concordância observada e o coeficiente Kappa de Cohen, seguidos de resolução das divergências.

## 5. Limitações

O dataset piloto não constitui uma amostra representativa da literatura científica do período.

As principais limitações são:

- Seleção dos primeiros resultados de cada consulta, potencialmente sujeita a viés de relevância.
- Distribuição temporal desigual.
- Ausência de auditoria humana concluída.
- Possibilidade de resumos inadequados mesmo quando há metadados estruturalmente válidos.
- Ausência de artigos recuperados por múltiplas áreas, que não implica ausência de interdisciplinaridade científica.

## 6. Próximas etapas

A coleta definitiva deverá utilizar amostragem reprodutível e estratificada por área e ano, preservando os metadados necessários aos experimentos de aprendizado de máquina.

A avaliação de relevância e a qualidade textual devem ser consideradas antes da geração de embeddings e da execução dos algoritmos de clusterização.

A hipótese de interdisciplinaridade será investigada nos experimentos semânticos, e não inferida exclusivamente das áreas utilizadas na coleta.

## 7. Reprodutibilidade

Com o ambiente do projeto configurado e uma chave válida da OpenAlex, executar:

```bash
PYTHONPATH=src uv run --env-file .env python -m research_atlas.data.pilot
PYTHONPATH=src uv run python -m research_atlas.analysis.eda
PYTHONPATH=src uv run python -m research_atlas.analysis.audit
uv run pytest -v
```

Na execução de referência, os 39 testes automatizados foram aprovados.

Os resultados numéricos deste relatório correspondem à execução piloto analisada e podem variar em novas consultas à OpenAlex.