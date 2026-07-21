from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_URL = (
    "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/"
    "master/campeonato-brasileiro-full.csv"
)

FEATURES = [
    "posicao_r19",
    "pontos",
    "vitorias",
    "empates",
    "derrotas",
    "gols_pro",
    "gols_contra",
    "saldo_gols",
    "aproveitamento",
    "distancia_para_16",
    "distancia_para_z4",
    "pontos_por_jogo",
    "gols_pro_por_jogo",
    "gols_contra_por_jogo",
    "saldo_por_jogo",
    "aproveitamento_ultimos_5",
    "variacao_posicao_ultimas_5",
    "percentual_pontos_casa",
    "percentual_pontos_fora",
]

SNAPSHOT_FEATURES = [
    "posicao_r19",
    "pontos",
    "vitorias",
    "empates",
    "derrotas",
    "gols_pro",
    "gols_contra",
    "saldo_gols",
    "aproveitamento",
    "distancia_para_16",
    "distancia_para_z4",
    "pontos_por_jogo",
    "gols_pro_por_jogo",
    "gols_contra_por_jogo",
    "saldo_por_jogo",
]


def carregar_partidas(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url).rename(columns={"rodata": "rodada"})
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["data", "rodada", "mandante", "visitante"]).copy()
    df["rodada"] = df["rodada"].astype(int)
    df = df.sort_values("ID").reset_index(drop=True)
    df["temporada"] = identificar_temporadas(df)
    df = df.sort_values(["temporada", "rodada", "data", "ID"]).reset_index(drop=True)
    return df


def identificar_temporadas(df: pd.DataFrame) -> pd.Series:
    temporada_atual = int(df.iloc[0]["data"].year)
    temporadas = []
    rodada_anterior = None

    for _, row in df.iterrows():
        rodada = int(row["rodada"])
        if rodada_anterior is not None and rodada < rodada_anterior:
            temporada_atual = int(row["data"].year)
        temporadas.append(temporada_atual)
        rodada_anterior = rodada

    return pd.Series(temporadas, index=df.index)


def pontos_do_jogo(gols_pro: int, gols_contra: int) -> int:
    if gols_pro > gols_contra:
        return 3
    if gols_pro == gols_contra:
        return 1
    return 0


def jogos_em_formato_longo(partidas: pd.DataFrame) -> pd.DataFrame:
    casa = pd.DataFrame(
        {
            "temporada": partidas["temporada"],
            "rodada": partidas["rodada"],
            "data": partidas["data"],
            "time": partidas["mandante"],
            "adversario": partidas["visitante"],
            "mando": "casa",
            "gols_pro": partidas["mandante_Placar"],
            "gols_contra": partidas["visitante_Placar"],
        }
    )
    fora = pd.DataFrame(
        {
            "temporada": partidas["temporada"],
            "rodada": partidas["rodada"],
            "data": partidas["data"],
            "time": partidas["visitante"],
            "adversario": partidas["mandante"],
            "mando": "fora",
            "gols_pro": partidas["visitante_Placar"],
            "gols_contra": partidas["mandante_Placar"],
        }
    )
    jogos = pd.concat([casa, fora], ignore_index=True)
    jogos["pontos"] = [
        pontos_do_jogo(gp, gc)
        for gp, gc in zip(jogos["gols_pro"], jogos["gols_contra"])
    ]
    jogos["vitoria"] = (jogos["pontos"] == 3).astype(int)
    jogos["empate"] = (jogos["pontos"] == 1).astype(int)
    jogos["derrota"] = (jogos["pontos"] == 0).astype(int)
    return jogos.sort_values(["temporada", "rodada", "data"]).reset_index(drop=True)


def tabela_ate_rodada(jogos: pd.DataFrame, temporada: int, rodada: int) -> pd.DataFrame:
    amostra = jogos[(jogos["temporada"] == temporada) & (jogos["rodada"] <= rodada)]
    tabela = (
        amostra.groupby("time")
        .agg(
            jogos=("time", "size"),
            pontos=("pontos", "sum"),
            vitorias=("vitoria", "sum"),
            empates=("empate", "sum"),
            derrotas=("derrota", "sum"),
            gols_pro=("gols_pro", "sum"),
            gols_contra=("gols_contra", "sum"),
        )
        .reset_index()
    )
    tabela["saldo_gols"] = tabela["gols_pro"] - tabela["gols_contra"]
    tabela = tabela.sort_values(
        ["pontos", "vitorias", "saldo_gols", "gols_pro"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    tabela["posicao"] = np.arange(1, len(tabela) + 1)
    return tabela


def montar_features_temporada(
    jogos: pd.DataFrame,
    temporada: int,
    rodada_corte: int,
    incluir_rotulo: bool = True,
) -> pd.DataFrame:
    tabela_corte = tabela_ate_rodada(jogos, temporada, rodada_corte)
    if tabela_corte.empty or tabela_corte["jogos"].max() == 0:
        raise ValueError(f"Sem jogos suficientes para a temporada {temporada}.")
    if len(tabela_corte) < 17:
        raise ValueError(f"Temporada {temporada} nao tem clubes suficientes na rodada {rodada_corte}.")

    tabela_final = tabela_ate_rodada(jogos, temporada, 38) if incluir_rotulo else None
    tabela_anterior = tabela_ate_rodada(jogos, temporada, max(1, rodada_corte - 5))

    recentes = jogos[
        (jogos["temporada"] == temporada)
        & (jogos["rodada"] > rodada_corte - 5)
        & (jogos["rodada"] <= rodada_corte)
    ]
    pontos_recentes = recentes.groupby("time")["pontos"].sum()

    pontos_16 = tabela_corte.loc[tabela_corte["posicao"] == 16, "pontos"].iloc[0]
    pontos_17 = tabela_corte.loc[tabela_corte["posicao"] == 17, "pontos"].iloc[0]
    posicao_anterior = tabela_anterior.set_index("time")["posicao"]
    posicao_final = tabela_final.set_index("time")["posicao"] if tabela_final is not None else None

    casa_fora = (
        jogos[(jogos["temporada"] == temporada) & (jogos["rodada"] <= rodada_corte)]
        .groupby(["time", "mando"])["pontos"]
        .sum()
        .unstack(fill_value=0)
    )

    df = tabela_corte.copy()
    df["temporada"] = temporada
    df = df.rename(columns={"posicao": "posicao_r19"})
    df["aproveitamento"] = df["pontos"] / (df["jogos"] * 3)
    df["distancia_para_16"] = df["pontos"] - pontos_16
    df["distancia_para_z4"] = df["pontos"] - pontos_17
    df["pontos_por_jogo"] = df["pontos"] / df["jogos"]
    df["gols_pro_por_jogo"] = df["gols_pro"] / df["jogos"]
    df["gols_contra_por_jogo"] = df["gols_contra"] / df["jogos"]
    df["saldo_por_jogo"] = df["saldo_gols"] / df["jogos"]
    df["aproveitamento_ultimos_5"] = (
        df["time"].map(pontos_recentes).fillna(0) / 15
    )
    df["variacao_posicao_ultimas_5"] = (
        df["time"].map(posicao_anterior) - df["posicao_r19"]
    )
    df["percentual_pontos_casa"] = (
        df["time"].map(casa_fora.get("casa", pd.Series(dtype=float))).fillna(0)
        / df["pontos"].replace(0, np.nan)
    ).fillna(0)
    df["percentual_pontos_fora"] = (
        df["time"].map(casa_fora.get("fora", pd.Series(dtype=float))).fillna(0)
        / df["pontos"].replace(0, np.nan)
    ).fillna(0)
    if posicao_final is None:
        df["posicao_final"] = np.nan
        df["rebaixado"] = np.nan
    else:
        df["posicao_final"] = df["time"].map(posicao_final)
        df["rebaixado"] = (df["posicao_final"] >= 17).astype(int)
    return df


def temporadas_completas(partidas: pd.DataFrame) -> list[int]:
    resumo = partidas.groupby("temporada").agg(
        jogos=("ID", "count"), rodadas=("rodada", "max")
    )
    completas = resumo[(resumo["jogos"] >= 380) & (resumo["rodadas"] == 38)]
    return sorted(completas.index.astype(int).tolist())


class RegressaoLogisticaSimples:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 12000, l2: float = 0.05):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RegressaoLogisticaSimples":
        x_arr = x.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        self.mean_ = x_arr.mean(axis=0)
        self.std_ = x_arr.std(axis=0)
        self.std_[self.std_ == 0] = 1
        x_scaled = (x_arr - self.mean_) / self.std_
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]

        positivos = max(y_arr.sum(), 1)
        negativos = max(len(y_arr) - y_arr.sum(), 1)
        pesos_classe = np.where(y_arr == 1, len(y_arr) / (2 * positivos), len(y_arr) / (2 * negativos))

        self.weights_ = np.zeros(x_design.shape[1])
        for _ in range(self.epochs):
            prob = self._sigmoid(x_design @ self.weights_)
            erro = (prob - y_arr) * pesos_classe
            grad = (x_design.T @ erro) / len(y_arr)
            grad[1:] += self.l2 * self.weights_[1:] / len(y_arr)
            self.weights_ -= self.learning_rate * grad
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.weights_ is None:
            raise RuntimeError("Modelo ainda nao foi treinado.")
        x_arr = x.to_numpy(dtype=float)
        x_scaled = (x_arr - self.mean_) / self.std_
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        prob = self._sigmoid(x_design @ self.weights_)
        return np.c_[1 - prob, prob]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def auc_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    y = y_true.to_numpy()
    scores = y_prob.to_numpy()
    positivos = scores[y == 1]
    negativos = scores[y == 0]
    if len(positivos) == 0 or len(negativos) == 0:
        return np.nan
    comparacoes = [(p > n) + 0.5 * (p == n) for p in positivos for n in negativos]
    return float(np.mean(comparacoes))


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    return float(np.mean((y_prob.to_numpy() - y_true.to_numpy()) ** 2))


def treinar_modelo(base: pd.DataFrame, features: list[str] | None = None) -> RegressaoLogisticaSimples:
    features = features or FEATURES
    return RegressaoLogisticaSimples().fit(base[features], base["rebaixado"])


def validar_por_temporada(base: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for temporada in sorted(base["temporada"].unique()):
        treino = base[base["temporada"] != temporada]
        teste = base[base["temporada"] == temporada].copy()
        if treino["rebaixado"].nunique() < 2:
            continue
        modelo = treinar_modelo(treino)
        teste["risco_rebaixamento"] = modelo.predict_proba(teste[FEATURES])[:, 1]
        top4 = teste.nlargest(4, "risco_rebaixamento")
        acertos_top4 = int(top4["rebaixado"].sum())
        try:
            auc = auc_score(teste["rebaixado"], teste["risco_rebaixamento"])
        except Exception:
            auc = np.nan
        linhas.append(
            {
                "temporada_teste": temporada,
                "acertos_top4": acertos_top4,
                "auc": auc,
                "brier": brier_score(teste["rebaixado"], teste["risco_rebaixamento"]),
            }
        )
    return pd.DataFrame(linhas)


def faixa_de_risco(probabilidade: float) -> str:
    if probabilidade >= 0.70:
        return "muito alto"
    if probabilidade >= 0.50:
        return "alto"
    if probabilidade >= 0.30:
        return "moderado"
    return "baixo"


def principais_fatores(row: pd.Series) -> str:
    fatores = []
    if row["pontos_por_jogo"] < 1.0:
        fatores.append("baixa pontuacao")
    if row["saldo_gols"] < -5:
        fatores.append("saldo negativo")
    if row["gols_contra_por_jogo"] > 1.4:
        fatores.append("defesa vulneravel")
    if row.get("aproveitamento_ultimos_5", 1.0) < 0.35:
        fatores.append("momento recente ruim")
    if row["distancia_para_16"] < 0:
        fatores.append("abaixo do 16o colocado")
    return ", ".join(fatores[:3]) or "risco relativo pelo conjunto de indicadores"


def salvar_graficos(resultado: pd.DataFrame, out_dir: Path, temporada: int) -> None:
    fig_dir = out_dir / "graficos"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ranking = resultado.sort_values("risco_rebaixamento", ascending=True)
    cores = ["#C1121F" if p >= 0.5 else "#457B9D" for p in ranking["risco_rebaixamento"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(ranking["time"], ranking["risco_rebaixamento"] * 100, color=cores)
    ax.set_xlabel("Risco estimado de rebaixamento (%)")
    ax.set_title(f"Ranking de risco de rebaixamento - Brasileirao {temporada}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"ranking_risco_rebaixamento_{temporada}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    eixo_y = "posicao_final" if resultado["posicao_final"].notna().any() else "posicao_r19"
    rotulo_y = "Posicao final" if eixo_y == "posicao_final" else "Posicao atual"
    ax.scatter(
        resultado["pontos"],
        resultado[eixo_y],
        s=120,
        c=resultado["risco_rebaixamento"],
        cmap="Reds",
        edgecolors="white",
        linewidths=1,
    )
    for _, row in resultado.iterrows():
        ax.annotate(row["time"], (row["pontos"], row[eixo_y]), fontsize=8)
    ax.axhline(16.5, color="#333333", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Pontos apos 19 rodadas")
    ax.set_ylabel(rotulo_y)
    ax.set_title(f"Rodada 19 vs {rotulo_y.lower()} - Brasileirao {temporada}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"r19_vs_posicao_final_{temporada}.png", dpi=160)
    plt.close(fig)


def carregar_snapshot_tabela(path: Path, temporada: int) -> pd.DataFrame:
    tabela = pd.read_csv(path)
    rename = {
        "posicao": "posicao_r19",
        "pts": "pontos",
        "j": "jogos",
        "v": "vitorias",
        "e": "empates",
        "d": "derrotas",
        "sg": "saldo_gols",
        "gp": "gols_pro",
        "gc": "gols_contra",
    }
    tabela = tabela.rename(columns={k: v for k, v in rename.items() if k in tabela.columns})
    obrigatorias = [
        "time",
        "posicao_r19",
        "pontos",
        "jogos",
        "vitorias",
        "empates",
        "derrotas",
        "saldo_gols",
        "gols_pro",
        "gols_contra",
    ]
    faltantes = [col for col in obrigatorias if col not in tabela.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no snapshot: {faltantes}")

    tabela = tabela.copy()
    tabela["temporada"] = temporada
    tabela["aproveitamento"] = tabela["pontos"] / (tabela["jogos"] * 3)
    pontos_16 = tabela.loc[tabela["posicao_r19"] == 16, "pontos"].iloc[0]
    pontos_17 = tabela.loc[tabela["posicao_r19"] == 17, "pontos"].iloc[0]
    tabela["distancia_para_16"] = tabela["pontos"] - pontos_16
    tabela["distancia_para_z4"] = tabela["pontos"] - pontos_17
    tabela["pontos_por_jogo"] = tabela["pontos"] / tabela["jogos"]
    tabela["gols_pro_por_jogo"] = tabela["gols_pro"] / tabela["jogos"]
    tabela["gols_contra_por_jogo"] = tabela["gols_contra"] / tabela["jogos"]
    tabela["saldo_por_jogo"] = tabela["saldo_gols"] / tabela["jogos"]
    tabela["posicao_final"] = np.nan
    tabela["rebaixado"] = np.nan
    return tabela


def executar(
    target_season: int | None,
    cutoff_round: int,
    training_window: int,
    output_dir: Path,
    target_table_csv: Path | None = None,
) -> None:
    partidas = carregar_partidas()
    jogos = jogos_em_formato_longo(partidas)
    completas = temporadas_completas(partidas)
    if not completas:
        raise RuntimeError("Nenhuma temporada completa encontrada no dataset.")

    temporadas_com_dados = sorted(partidas["temporada"].astype(int).unique().tolist())
    temporadas_com_corte = sorted(
        partidas.groupby("temporada")["rodada"].max().loc[lambda s: s >= cutoff_round].index.astype(int).tolist()
    )
    target = target_season or (2026 if target_table_csv else temporadas_com_corte[-1])
    if not target_table_csv and target not in temporadas_com_corte:
        raise ValueError(
            f"Temporada {target} nao tem dados ate a rodada {cutoff_round}. "
            f"Temporadas disponiveis: {temporadas_com_dados}"
        )

    historicas = [t for t in completas if t < target][-training_window:]
    if len(historicas) < 2:
        raise ValueError("Use pelo menos duas temporadas historicas para treinar.")

    base_treino = pd.concat(
        [montar_features_temporada(jogos, t, cutoff_round, incluir_rotulo=True) for t in historicas],
        ignore_index=True,
    )
    if target_table_csv:
        alvo = carregar_snapshot_tabela(target_table_csv, target)
        features_modelo = SNAPSHOT_FEATURES
    else:
        alvo = montar_features_temporada(
            jogos,
            target,
            cutoff_round,
            incluir_rotulo=target in completas,
        )
        features_modelo = FEATURES

    modelo = treinar_modelo(base_treino, features_modelo)
    alvo["risco_rebaixamento"] = modelo.predict_proba(alvo[features_modelo])[:, 1]
    alvo["faixa"] = alvo["risco_rebaixamento"].map(faixa_de_risco)
    alvo["principais_fatores"] = alvo.apply(principais_fatores, axis=1)
    alvo = alvo.sort_values("risco_rebaixamento", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.concat([base_treino, alvo], ignore_index=True)
    base.to_csv(output_dir / "base_risco_rebaixamento.csv", index=False)
    validar_por_temporada(base_treino).to_csv(
        output_dir / "validacao_por_temporada.csv", index=False
    )
    alvo.to_csv(output_dir / f"probabilidades_rebaixamento_{target}.csv", index=False)
    salvar_graficos(alvo, output_dir, target)

    print(f"Temporada-alvo: {target}")
    print(f"Treino: {historicas}")
    print("\nTop 4 riscos de rebaixamento:")
    print(
        alvo[
            [
                "time",
                "risco_rebaixamento",
                "faixa",
                "pontos",
                "posicao_r19",
                "posicao_final",
                "principais_fatores",
            ]
        ]
        .head(4)
        .to_string(index=False, formatters={"risco_rebaixamento": "{:.1%}".format})
    )
    print(f"\nArquivos salvos em: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima risco de rebaixamento do Brasileirao com a tabela da rodada 19."
    )
    parser.add_argument("--target-season", type=int, default=None)
    parser.add_argument(
        "--target-table-csv",
        type=Path,
        default=None,
        help="CSV com a tabela atual da temporada-alvo, usado para projetar temporadas em andamento.",
    )
    parser.add_argument("--cutoff-round", type=int, default=19)
    parser.add_argument("--training-window", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    executar(
        target_season=args.target_season,
        cutoff_round=args.cutoff_round,
        training_window=args.training_window,
        output_dir=args.output_dir,
        target_table_csv=args.target_table_csv,
    )
