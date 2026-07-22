from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risco_rebaixamento import (
    RegressaoLogisticaSimples,
    carregar_partidas,
    jogos_em_formato_longo,
    montar_features_temporada,
    temporadas_completas,
)
from simulacao_monte_carlo import (
    INITIAL_ELO,
    ajustar_elo_com_contexto,
    ajustar_elo_com_snapshot,
    calcular_elo_historico,
    carregar_contexto,
    carregar_tabela_atual,
    nome_exibicao,
    normalizar_time,
)


QUALITY_FEATURES = [
    "gols_pro_por_jogo",
    "gols_contra_por_jogo",
    "saldo_por_jogo",
    "aproveitamento_ultimos_5",
    "variacao_posicao_ultimas_5",
    "elo_ajustado",
    "serie_a_last_5",
    "previous_season_score",
    "squad_strength_0_100",
    "recent_points_last5",
    "recent_goal_diff_last5",
]

FEATURE_LABELS = {
    "gols_pro_por_jogo": ("ataque pouco produtivo", "ataque permissivo ao risco"),
    "gols_contra_por_jogo": ("defesa consistente", "defesa vulneravel"),
    "saldo_por_jogo": ("saldo de gols baixo", "saldo de gols instavel"),
    "aproveitamento_ultimos_5": ("forma recente ruim", "forma recente instavel"),
    "variacao_posicao_ultimas_5": ("tendencia recente negativa", "oscilacao recente"),
    "elo_ajustado": ("Elo baixo", "Elo superestimado"),
    "serie_a_last_5": ("pouca experiencia Serie A", "historico recente em risco"),
    "previous_season_score": ("temporada anterior fraca", "temporada anterior enganosa"),
    "squad_strength_0_100": ("elenco fragil", "elenco em alerta"),
    "recent_points_last5": ("poucos pontos recentes", "forma recente enganosa"),
    "recent_goal_diff_last5": ("saldo recente ruim", "saldo recente instavel"),
}


def historico_contextual(jogos: pd.DataFrame, temporadas: list[int], cutoff_round: int) -> pd.DataFrame:
    linhas = []
    final_por_temporada = {
        temporada: montar_features_temporada(jogos, temporada, 38, incluir_rotulo=True)
        for temporada in temporadas
    }

    for temporada in temporadas:
        base = montar_features_temporada(jogos, temporada, cutoff_round, incluir_rotulo=True)
        temporadas_anteriores = [t for t in temporadas if temporada - 5 <= t < temporada]
        presencas = pd.concat(
            [final_por_temporada[t][["time", "posicao_final"]] for t in temporadas_anteriores],
            ignore_index=True,
        ) if temporadas_anteriores else pd.DataFrame(columns=["time", "posicao_final"])
        serie_a_last_5 = presencas.groupby("time").size()
        anterior = final_por_temporada.get(temporada - 1)
        if anterior is not None:
            pos_anterior = anterior.set_index("time")["posicao_final"]
        else:
            pos_anterior = pd.Series(dtype=float)

        base["serie_a_last_5"] = base["time"].map(serie_a_last_5).fillna(0)
        base["previous_season_position"] = base["time"].map(pos_anterior).fillna(22)
        base["previous_season_score"] = (21 - base["previous_season_position"]).clip(lower=-1)
        base["squad_strength_0_100"] = 50 + base["serie_a_last_5"] * 6 + base["previous_season_score"] * 1.4
        base["recent_points_last5"] = base["aproveitamento_ultimos_5"] * 15
        base["recent_goal_diff_last5"] = base["saldo_por_jogo"] * 5
        linhas.append(base)

    return pd.concat(linhas, ignore_index=True)


def adicionar_elo_historico(base: pd.DataFrame, partidas: pd.DataFrame) -> pd.DataFrame:
    ratings = calcular_elo_historico(partidas)
    base = base.copy()
    base["elo_ajustado"] = base["time"].map(lambda t: ratings.get(normalizar_time(t), INITIAL_ELO))
    return base


def preparar_alvo_2026(
    table_csv: Path,
    context_csv: Path,
    partidas: pd.DataFrame,
) -> pd.DataFrame:
    tabela = carregar_tabela_atual(table_csv)
    contexto = carregar_contexto(context_csv)
    ratings = calcular_elo_historico(partidas)
    ratings = ajustar_elo_com_contexto(ratings, contexto)
    ratings = ajustar_elo_com_snapshot(ratings, tabela)

    alvo = tabela.merge(contexto, on="time", how="left")
    alvo["elo_ajustado"] = alvo["time"].map(lambda t: ratings.get(t, INITIAL_ELO))
    alvo["aproveitamento_ultimos_5"] = alvo["recent_points_last5"] / 15
    alvo["variacao_posicao_ultimas_5"] = alvo["recent_goal_diff_last5"]
    alvo["rebaixado"] = np.nan
    return alvo


def contribuições(modelo: RegressaoLogisticaSimples, dados: pd.DataFrame) -> pd.DataFrame:
    if modelo.mean_ is None or modelo.std_ is None or modelo.weights_ is None:
        raise RuntimeError("Modelo sem parametros treinados.")
    x = dados[QUALITY_FEATURES].to_numpy(dtype=float)
    scaled = (x - modelo.mean_) / modelo.std_
    contrib = scaled * modelo.weights_[1:]
    rows = []
    for idx, row in dados.reset_index(drop=True).iterrows():
        pares = sorted(
            zip(QUALITY_FEATURES, contrib[idx]),
            key=lambda item: item[1],
            reverse=True,
        )
        top = []
        for feature, value in pares:
            if value <= 0:
                continue
            feature_idx = QUALITY_FEATURES.index(feature)
            abaixo_da_media = scaled[idx, feature_idx] < 0
            label_baixo, label_alto = FEATURE_LABELS[feature]
            top.append(label_baixo if abaixo_da_media else label_alto)
            if len(top) == 3:
                break
        rows.append(
            {
                "time": row["time"],
                "principais_fatores": ", ".join(top) if top else "risco distribuido entre indicadores",
            }
        )
    return pd.DataFrame(rows)


def salvar_card_qualidade(resultado: pd.DataFrame, path: Path) -> None:
    plot = resultado.sort_values("risco_ml_sem_tabela", ascending=False)
    fig = plt.figure(figsize=(12, 16), facecolor="#F8FAFC")
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.78])
    y = np.arange(len(plot))
    risco = plot["risco_ml_sem_tabela"] * 100
    cores = ["#DC2626" if v >= 50 else "#F97316" if v >= 25 else "#FACC15" if v >= 8 else "#16A34A" for v in risco]
    ax.barh(y, risco, color=cores, height=0.72)
    ax.invert_yaxis()
    ax.set_xlim(-24, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#CBD5E1", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Probabilidade estimada de rebaixamento (%)", color="#334155")

    for i, row in enumerate(plot.itertuples()):
        fatores = ", ".join(str(row.principais_fatores).split(", ")[:2])
        label = f"{row.risco_ml_sem_tabela * 100:4.1f}% | {fatores}"
        ax.text(-21.5, i, f"{i + 1:02d}", va="center", ha="left", fontsize=10, color="#475569", weight="bold")
        ax.text(-17.0, i, nome_exibicao(row.time), va="center", ha="left", fontsize=12, color="#0F172A", weight="bold")
        if row.risco_ml_sem_tabela >= 0.72:
            ax.text(row.risco_ml_sem_tabela * 100 - 1, i, label, va="center", ha="right", fontsize=9.4, color="white", weight="bold")
        else:
            ax.text(row.risco_ml_sem_tabela * 100 + 1, i, label, va="center", ha="left", fontsize=9.4, color="#0F172A")

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")

    fig.text(0.07, 0.955, "Brazilian Relegation Predictor", fontsize=26, weight="bold", color="#0F172A")
    fig.text(0.07, 0.928, "Modelo sem pontos e sem posicao na tabela como variaveis explicativas", fontsize=12, color="#475569")
    fig.text(
        0.07,
        0.895,
        "Features: ataque, defesa, saldo, forma recente, Elo, elenco, historico de Serie A e temporada anterior.",
        fontsize=10.5,
        color="#64748B",
    )
    fig.text(
        0.07,
        0.035,
        "Maiores riscos: " + ", ".join(nome_exibicao(t) for t in plot.head(4)["time"].tolist()),
        fontsize=12,
        weight="bold",
        color="#991B1B",
    )
    fig.text(0.07, 0.018, "Modelo experimental de classificacao: alvo = terminou no Z4.", fontsize=9, color="#64748B")
    fig.savefig(path, dpi=180, format="jpg", pil_kwargs={"quality": 94})
    plt.close(fig)


def executar(output_dir: Path, table_csv: Path, context_csv: Path, cutoff_round: int) -> None:
    partidas = carregar_partidas()
    jogos = jogos_em_formato_longo(partidas)
    completas = temporadas_completas(partidas)
    treino_temporadas = [t for t in completas if t <= 2024][-10:]
    base = historico_contextual(jogos, treino_temporadas, cutoff_round)
    base = adicionar_elo_historico(base, partidas)
    modelo = RegressaoLogisticaSimples(learning_rate=0.04, epochs=14000, l2=0.12)
    modelo.fit(base[QUALITY_FEATURES], base["rebaixado"])

    alvo = preparar_alvo_2026(table_csv, context_csv, partidas)
    alvo["risco_ml_sem_tabela"] = modelo.predict_proba(alvo[QUALITY_FEATURES])[:, 1]
    explicacoes = contribuições(modelo, alvo)
    resultado = alvo.merge(explicacoes, on="time", how="left").sort_values("risco_ml_sem_tabela", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graficos").mkdir(exist_ok=True)
    base.to_csv(output_dir / "base_ml_sem_tabela.csv", index=False)
    resultado.to_csv(output_dir / "risco_ml_sem_tabela_2026.csv", index=False)
    salvar_card_qualidade(resultado, output_dir / "graficos" / "risco_ml_sem_tabela_2026.jpg")

    print("Top 6 risco sem pontos/posicao:")
    print(
        resultado[["time", "risco_ml_sem_tabela", "principais_fatores"]]
        .head(6)
        .to_string(index=False, formatters={"risco_ml_sem_tabela": "{:.1%}".format})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelo de risco sem pontos e sem posicao atual.")
    parser.add_argument("--output-dir", type=Path, default=Path("output_2026"))
    parser.add_argument("--table-csv", type=Path, default=Path("data/raw/brasileirao_2026_r19_snapshot.csv"))
    parser.add_argument("--context-csv", type=Path, default=Path("data/raw/brasileirao_2026_team_context.csv"))
    parser.add_argument("--cutoff-round", type=int, default=19)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    executar(args.output_dir, args.table_csv, args.context_csv, args.cutoff_round)
