# ⚽ Desempenho vs. Resultado: Quem tem sorte e quem tem azar no Brasileirão?

Análise de dados que compara o desempenho real dos times (medido via xG - Expected Goals)
com os resultados efetivamente obtidos, identificando quais equipes tiveram sorte ou azar
ao longo do campeonato.

## 🎯 Pergunta central

O time que "merece" vencer, baseado no seu desempenho em campo, é realmente o que mais pontua?
Ou existem times sistematicamente "azarados" (jogam bem, pontuam pouco) e "sortudos"
(jogam mal, pontuam muito)?

## 📊 Metodologia

1. Coleta de dados de xG (Expected Goals) do Brasileirão via FootyStats
2. Coleta de resultados reais e pontos via API de dados esportivos
3. Conversão de xG em "Pontos Esperados" usando distribuição de Poisson
4. Cálculo do Índice de Sorte/Azar = Pontos Reais - Pontos Esperados
5. Validação da robustez da métrica em múltiplas temporadas

## 📁 Estrutura do projeto

```
analise-brasileirao/
├── data/
│   ├── raw/           # Dados brutos, como baixados das fontes
│   └── processed/     # Dados limpos e prontos para análise
├── notebooks/         # Jupyter notebooks de exploração e análise
├── src/               # Scripts Python reutilizáveis (coleta, limpeza, cálculos)
├── reports/
│   └── figures/       # Gráficos finais para o post/apresentação
├── requirements.txt
└── README.md
```

## 🛠️ Como rodar

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📈 Fontes de dados

- [FootyStats - xG Brasileirão](https://footystats.org/pt/brazil/serie-a/xg)
- API de dados esportivos (resultados, tabela, mando de campo)

## 📌 Status

🚧 Em desenvolvimento — projeto de 3 semanas, [Semana 1/3]

## 👤 Autor

Seu nome | [LinkedIn](#) | [GitHub](#)
