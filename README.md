 Desempenho vs. Resultado

Análise de dados que compara o desempenho ofensivo/defensivo histórico dos times
com os resultados efetivamente obtidos, usando um modelo de Poisson para estimar
quantos pontos cada time "deveria" ter feito — e identificar quem teve sorte e quem teve azar.

Pergunta central

O time que "merece" vencer, baseado no seu desempenho em campo, é realmente o que mais pontua?
Ou existem times sistematicamente "azarados" (jogam bem, pontuam pouco) e "sortudos"
(jogam mal, pontuam muito)?

Metodologia

1. Coleta de dados históricos de partidas do Brasileirão (dataset público, 2003-2024)
2. Cálculo de índices de ataque e defesa por time, relativos à média da liga
3. Estimativa de gols esperados por partida (ataque de um time x defesa do outro)
4. Conversão de gols esperados em pontos esperados via distribuição de Poisson
5. Índice de Sorte/Azar = Pontos Reais - Pontos Esperados
6. Validação: teste de consistência do índice entre temporadas (2022-2024)

métrica própria de "desempenho esperado" a partir do histórico de gols

Estrutura do projeto

```
analise-brasileirao/
│ - data/
│ - raw/           
│   └── processed/    
│ notebooks/         
│ - src/              
│ - reports/
│   └── figures/       
│ - requirements.txt
└── README.md
```

Fontes de dados

- Brasileirao_Dataset (adaoduque) - github.com/adaoduque/Brasileirao_Dataset
  Dataset público com todas as partidas do Brasileirão Série A desde 2003

## Modelo de risco de rebaixamento na rodada 19

O projeto também inclui um script para estimar quais clubes tinham maior risco de
rebaixamento a partir da fotografia da tabela após 19 rodadas.

Ele reconstrói as temporadas completas do dataset, monta atributos como pontos,
saldo, aproveitamento, distância para o 16º colocado, desempenho recente e
percentual de pontos em casa/fora, e treina uma regressão logística simples com
validação por temporada.

```bash
python src/risco_rebaixamento.py --output-dir output
```

Por padrão, o script usa a última temporada com dados até a rodada 19 como alvo
e as cinco temporadas completas anteriores como treino. No dataset público usado
pelo projeto, a última temporada disponível no momento da implementação é 2024.
Para simular 2026, use a tabela atual salva em
`data/raw/brasileirao_2026_r19_snapshot.csv` como alvo. Essa tabela foi montada
a partir da classificação publicada em 20/07/2026, com clubes ainda entre 18 e
19 jogos disputados. Fonte consultada: TMC / Placar Futebol, tabela do
Brasileirão Série A 2026.

Para escolher outra temporada-alvo:

```bash
python src/risco_rebaixamento.py --target-season 2023 --output-dir output
```

Para projetar o risco de rebaixamento em 2026:

```bash
python src/risco_rebaixamento.py --target-season 2026 --target-table-csv data/raw/brasileirao_2026_r19_snapshot.csv --output-dir output_2026
```

Arquivos gerados:

- `output/base_risco_rebaixamento.csv`: base modelada por clube/temporada;
- `output/validacao_por_temporada.csv`: validação deixando uma temporada fora;
- `output/probabilidades_rebaixamento_<temporada>.csv`: ranking de risco;
- `output/graficos/ranking_risco_rebaixamento_<temporada>.png`;
- `output/graficos/r19_vs_posicao_final_<temporada>.png`.

O modelo não deve ser lido como "estes serão os rebaixados", e sim como uma
estimativa de risco com base nos dados disponíveis até a rodada 19.

Análise e modelagem completas.

Gustavo Cruz | [LinkedIn](https://linkedin.com/in/gustavo-goncalves-cruz)| [GitHub](https://github.com/Guszzs)
