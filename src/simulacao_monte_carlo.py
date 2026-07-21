from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risco_rebaixamento import carregar_partidas


HOME_ADVANTAGE = 60
INITIAL_ELO = 1500
K_FACTOR = 28


def normalizar_time(nome: str) -> str:
    mapa = {
        "Athletico-PR": "Athletico-PR",
        "Atletico-MG": "Atletico-MG",
        "Red Bull Bragantino": "Bragantino",
        "Sao Paulo": "Sao Paulo",
        "Gremio": "Gremio",
        "Vitoria": "Vitoria",
    }
    return mapa.get(nome, nome)


def resultado_real(gols_mandante: int, gols_visitante: int) -> float:
    if gols_mandante > gols_visitante:
        return 1.0
    if gols_mandante == gols_visitante:
        return 0.5
    return 0.0


def atualizar_elo(ratings: dict[str, float], row: pd.Series) -> None:
    mandante = normalizar_time(row["mandante"])
    visitante = normalizar_time(row["visitante"])
    ratings.setdefault(mandante, INITIAL_ELO)
    ratings.setdefault(visitante, INITIAL_ELO)

    elo_m = ratings[mandante]
    elo_v = ratings[visitante]
    esperado_m = 1 / (1 + 10 ** ((elo_v - (elo_m + HOME_ADVANTAGE)) / 400))
    placar_m = resultado_real(int(row["mandante_Placar"]), int(row["visitante_Placar"]))
    margem = abs(int(row["mandante_Placar"]) - int(row["visitante_Placar"]))
    multiplicador = 1 + np.log1p(margem)
    ajuste = K_FACTOR * multiplicador * (placar_m - esperado_m)
    ratings[mandante] = elo_m + ajuste
    ratings[visitante] = elo_v - ajuste


def calcular_elo_historico(partidas: pd.DataFrame, ate_temporada: int = 2024) -> dict[str, float]:
    ratings: dict[str, float] = {}
    historico = partidas[partidas["temporada"] <= ate_temporada].sort_values(
        ["temporada", "rodada", "data", "ID"]
    )
    for _, row in historico.iterrows():
        atualizar_elo(ratings, row)
    return ratings


def ajustar_elo_com_snapshot(ratings: dict[str, float], tabela: pd.DataFrame) -> dict[str, float]:
    ajustados = ratings.copy()
    media_ppj = tabela["pontos_por_jogo"].mean()
    media_saldo = tabela["saldo_por_jogo"].mean()
    for _, row in tabela.iterrows():
        time = normalizar_time(row["time"])
        base = ajustados.get(time, INITIAL_ELO)
        ajuste_ppj = (row["pontos_por_jogo"] - media_ppj) * 120
        ajuste_saldo = (row["saldo_por_jogo"] - media_saldo) * 65
        ajustados[time] = base + ajuste_ppj + ajuste_saldo
    return ajustados


def carregar_tabela_atual(path: Path) -> pd.DataFrame:
    tabela = pd.read_csv(path)
    tabela["time"] = tabela["time"].map(normalizar_time)
    tabela["pontos_por_jogo"] = tabela["pontos"] / tabela["jogos"]
    tabela["saldo_por_jogo"] = tabela["saldo_gols"] / tabela["jogos"]
    return tabela


def probabilidades_partida(elo_mandante: float, elo_visitante: float) -> tuple[float, float, float]:
    diff = (elo_mandante + HOME_ADVANTAGE) - elo_visitante
    forca_m = 1 / (1 + np.exp(-diff / 260))
    p_empate = 0.30 - min(abs(diff), 350) * 0.00028
    p_empate = float(np.clip(p_empate, 0.18, 0.31))
    restante = 1 - p_empate
    p_mandante = restante * forca_m
    p_visitante = restante * (1 - forca_m)
    return float(p_mandante), p_empate, float(p_visitante)


def ordenar_tabela(tabela: pd.DataFrame) -> pd.DataFrame:
    return tabela.sort_values(
        ["pontos", "vitorias", "saldo_gols", "gols_pro"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def executar_simulacoes(
    tabela_path: Path,
    fixtures_path: Path,
    output_dir: Path,
    n_simulations: int,
    seed: int,
) -> None:
    partidas = carregar_partidas()
    tabela = carregar_tabela_atual(tabela_path)
    fixtures = pd.read_csv(fixtures_path)
    ratings = ajustar_elo_com_snapshot(calcular_elo_historico(partidas), tabela)
    rng = np.random.default_rng(seed)

    times = tabela["time"].tolist()
    idx = {time: i for i, time in enumerate(times)}
    n_times = len(times)
    pontos = np.tile(tabela["pontos"].to_numpy(dtype=int), (n_simulations, 1))
    vitorias = np.tile(tabela["vitorias"].to_numpy(dtype=int), (n_simulations, 1))
    gols_pro = np.tile(tabela["gols_pro"].to_numpy(dtype=int), (n_simulations, 1))
    gols_contra = np.tile(tabela["gols_contra"].to_numpy(dtype=int), (n_simulations, 1))

    for _, jogo in fixtures.iterrows():
        mandante = normalizar_time(jogo["mandante"])
        visitante = normalizar_time(jogo["visitante"])
        i_m = idx[mandante]
        i_v = idx[visitante]
        p_m, p_e, _ = probabilidades_partida(
            ratings.get(mandante, INITIAL_ELO),
            ratings.get(visitante, INITIAL_ELO),
        )
        sorteios = rng.random(n_simulations)
        win_m = sorteios < p_m
        draw = (sorteios >= p_m) & (sorteios < p_m + p_e)
        win_v = ~(win_m | draw)

        pontos[win_m, i_m] += 3
        pontos[draw, i_m] += 1
        pontos[draw, i_v] += 1
        pontos[win_v, i_v] += 3

        vitorias[win_m, i_m] += 1
        vitorias[win_v, i_v] += 1

        gols_pro[win_m, i_m] += 1
        gols_contra[win_m, i_v] += 1
        gols_pro[draw, i_m] += 1
        gols_pro[draw, i_v] += 1
        gols_contra[draw, i_m] += 1
        gols_contra[draw, i_v] += 1
        gols_pro[win_v, i_v] += 1
        gols_contra[win_v, i_m] += 1

    saldo = gols_pro - gols_contra
    rebaixamentos = np.zeros(n_times, dtype=int)
    soma_posicao = np.zeros(n_times, dtype=float)
    soma_pontos = pontos.sum(axis=0).astype(float)
    distribuicao = np.zeros((n_times, n_times), dtype=int)

    for sim in range(n_simulations):
        ordem = np.lexsort((-gols_pro[sim], -saldo[sim], -vitorias[sim], -pontos[sim]))
        posicoes = np.empty(n_times, dtype=int)
        posicoes[ordem] = np.arange(1, n_times + 1)
        soma_posicao += posicoes
        rebaixamentos[posicoes >= 17] += 1
        distribuicao[np.arange(n_times), posicoes - 1] += 1

    resultado = pd.DataFrame(
        {
            "time": times,
            "prob_rebaixamento": rebaixamentos / n_simulations,
            "posicao_media": soma_posicao / n_simulations,
            "pontos_medios": soma_pontos / n_simulations,
            "elo_ajustado": [ratings.get(t, INITIAL_ELO) for t in times],
        }
    ).sort_values("prob_rebaixamento", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graficos").mkdir(exist_ok=True)
    resultado.to_csv(output_dir / "monte_carlo_rebaixamento_2026.csv", index=False)
    distribuicao_df = pd.DataFrame(distribuicao, index=times, columns=range(1, 21)).div(n_simulations)
    distribuicao_df.to_csv(output_dir / "distribuicao_posicoes_2026.csv")
    salvar_grafico_rebaixamento(resultado, output_dir / "graficos" / "monte_carlo_rebaixamento_2026.png")
    salvar_grafico_posicoes(distribuicao_df, output_dir / "graficos" / "distribuicao_posicoes_2026.png")

    print(f"Simulacoes: {n_simulations}")
    print("\nTop 6 risco de rebaixamento por Monte Carlo:")
    print(
        resultado.head(6).to_string(
            index=False,
            formatters={
                "prob_rebaixamento": "{:.1%}".format,
                "posicao_media": "{:.1f}".format,
                "pontos_medios": "{:.1f}".format,
                "elo_ajustado": "{:.0f}".format,
            },
        )
    )
    print(f"\nArquivos salvos em: {output_dir}")


def salvar_grafico_rebaixamento(resultado: pd.DataFrame, path: Path) -> None:
    plot = resultado.sort_values("prob_rebaixamento", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot["time"], plot["prob_rebaixamento"] * 100, color="#B91C1C")
    ax.set_xlabel("Probabilidade de terminar no Z4 (%)")
    ax.set_title("Risco de rebaixamento por Monte Carlo - Brasileirao 2026")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def salvar_grafico_posicoes(distribuicao: pd.DataFrame, path: Path) -> None:
    focos = distribuicao.loc[:, 17:20].sum(axis=1).sort_values(ascending=False).head(6).index
    dados = distribuicao.loc[focos, 12:20] * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(dados))
    for pos in dados.columns:
        ax.bar(dados.index, dados[pos], bottom=bottom, label=f"{pos}o")
        bottom += dados[pos].to_numpy()
    ax.set_ylabel("Probabilidade (%)")
    ax.set_title("Distribuicao de posicoes dos clubes com maior risco")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simula o restante do Brasileirao 2026 com Elo e Monte Carlo."
    )
    parser.add_argument("--table-csv", type=Path, default=Path("data/raw/brasileirao_2026_r19_snapshot.csv"))
    parser.add_argument("--fixtures-csv", type=Path, default=Path("data/raw/brasileirao_2026_remaining_fixtures.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_2026"))
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    executar_simulacoes(
        tabela_path=args.table_csv,
        fixtures_path=args.fixtures_csv,
        output_dir=args.output_dir,
        n_simulations=args.simulations,
        seed=args.seed,
    )
