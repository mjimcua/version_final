"""
compare_with_real.py — The numbers of a run of the synthetic that looks like the real extract, next to the
numbers of the real run of 19-oct-2026 (current month 2026-09, six exam months).

    python compare_with_real.py [2026-10]  # runs main.like_real_configuration() and compares, then puts the
                                           # forecast against the future truth the synthetic knows
    from compare_with_real import compare_results; compare_results(results, configuration)

Every indicator is computed from the results of the run (the tables every step returns), never from the
console. The real value is the one read on the screen of the real run; the column `real` says where.
The point is the shape, not the amounts: the synthetic has ~1/20 of the series and of the volume, so
money is compared as shares and rates, never in dollars.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import sys

import numpy as np
import pandas as pd

from block_3_price import renewer_rows, uplift_history
from config import TRUTH_ROLES

# ─── the real run (19-oct-2026, current month 2026-09): indicator → (value, step where it was read) ───
REAL_RUN = {
    "tasa de renovación en unidades, meses cerrados": (0.640, "01 · 13"),
    "tasa de renovación en USD, meses cerrados": (0.729, "01 · 13"),
    "precio medio de una unidad que vence (USD)": (34.7, "01 · 13"),
    "cuota de unidades a 1 año": (0.876, "01 · 14"),
    "tasa a 1 año": (0.65, "01 · 14"),
    "tasa a 2+3 años": (0.54, "01 · 14"),
    "tasa bandas 1+2": (0.62, "01 · 14"),
    "tasa bandas 3+4+5": (0.655, "01 · 14"),
    "cuota de unidades neutral (meses cerrados)": (0.803, "17 · B"),
    "tasa unidades neutral": (0.714, "17 · B"),
    "tasa unidades dormant": (0.445, "17 · B"),
    "tasa unidades softcancel": (0.279, "17 · B"),
    "tasa unidades not_installed": (0.501, "17 · B"),
    "cuota del USD que vence con descuento desconocido": (0.59, "01 · 44"),
    "cuota del dinero por predecir con suelo, sin escalera": (0.82, "10 · 9"),
    "cuota del dinero por predecir con suelo, final": (0.98, "10 · 9"),
    "cuota del dinero con tasa propia, final": (0.49, "10 · 9"),
    "dinero en A_propio + A3_propio_reforzado": (0.80, "11 · 8"),
    "dinero en B_prestado + C_lejano": (0.17, "11 · 8"),
    "φ mediana de los pools": (1.94, "13 · 2"),
    "p-valor del efecto mes de la cartera (alto = sin estacionalidad)": (0.971, "13 · 3"),
    "mejor |err_norm| a corto (selección)": (1.36, "14 · 9"),
    "mejor |err_norm| a medio-largo (selección)": (2.12, "14 · 9"),
    "mejora de la elegida sobre la técnica por defecto a corto (pp)": (-0.11, "14 · 9"),
    "mejora de la elegida sobre la técnica por defecto a medio-largo (pp)": (0.04, "14 · 9"),
    "dentro de la banda a corto": (0.95, "14 · 9"),
    "error del total de los pools juzgados, h = 1": (-0.013, "14 · 9"),
    "error del total de los pools juzgados, h = 6": (-0.030, "14 · 9"),
    "error del total del framework, h = 1 (paso 7.1)": (0.005, "19 · 7"),
    "error del total del framework, h = 6 (paso 7.1)": (0.014, "19 · 7"),
    "WAPE serie a serie framework / serie sola, h = 1": (1.10, "19 · 7  (0.10 / 0.09)"),
    "dentro del intervalo, framework h = 1": (0.95, "19 · 7"),
    "uplift toda la historia": (1.13, "15 · 10"),
    "uplift ventana": (1.16, "15 · 10"),
    "uplift ventana sin subidas": (1.10, "15 · 10"),
    "ratio de realización de la regla del contrato, 0-10 %": (0.98, "15 · 8"),
    "ratio de realización de la regla del contrato, 50-60 %": (0.83, "15 · 8"),
    "sesgo vía contrato en el examen": (0.13, "16 · 7"),
    "sesgo vía estadística en el examen (descuento conocido)": (-0.03, "16 · 7"),
    "subidas en juego (ventana)": (43, "15 · 2"),
    "subidas detectadas en la historia": (77, "22 · 2"),
    "nivel de adquisición Pure-New": (0.97, "7.2"),
    "nivel de adquisición Not-New": (0.54, "7.2"),
    "ajuste del reparto de estados del cliente, año en curso": (-0.018, "7.2"),
    "ajuste del reparto de estados del cliente, año siguiente": (-0.045, "7.2"),
    "tasa USD esperada del primer mes futuro": (0.76, "17 · 12"),
    "renovado del año siguiente / año en curso": (0.935, "17 · by year"),
    "cuota de lo renovado del año siguiente que sale de pipeline modelada": (0.31, "17 · by year"),
    "primer mes futuro: esperado / lo ya renovado que se borró": (0.994, "17 · 12 y 21 · 7"),
}


def share(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def compare_results(results: dict, configuration) -> pd.DataFrame:
    """One row per indicator: the real value, the synthetic one, and where the real one was read."""
    synthetic = {}
    raw = results["fine_table"]
    closed = raw[raw["rol"].isin(TRUTH_ROLES)]
    units, renewed = configuration.pipeline_units_col, configuration.renewed_units_col
    usd, renewed_usd = configuration.pipeline_usd_col, configuration.renewed_usd_col

    # [1] the portfolio: rates, price, term, band, states, discount
    synthetic["tasa de renovación en unidades, meses cerrados"] = share(closed[renewed].sum(), closed[units].sum())
    synthetic["tasa de renovación en USD, meses cerrados"] = share(closed[renewed_usd].sum(), closed[usd].sum())
    synthetic["precio medio de una unidad que vence (USD)"] = share(closed[usd].sum(), closed[units].sum())
    one_year = closed["tr_term"] == "1 year"
    synthetic["cuota de unidades a 1 año"] = share(closed.loc[one_year, units].sum(), closed[units].sum())
    synthetic["tasa a 1 año"] = share(closed.loc[one_year, renewed].sum(), closed.loc[one_year, units].sum())
    synthetic["tasa a 2+3 años"] = share(closed.loc[~one_year, renewed].sum(), closed.loc[~one_year, units].sum())
    for label, bands in (("tasa bandas 1+2", ["1", "2"]), ("tasa bandas 3+4+5", ["3", "4", "5"])):
        rows = closed[closed["tr_band"].astype(str).isin(bands)]
        synthetic[label] = share(rows[renewed].sum(), rows[units].sum())
    states = ["dormant", "softcancel", "not_installed"]
    neutral = (closed[states].sum(axis=1) == 0)
    synthetic["cuota de unidades neutral (meses cerrados)"] = share(closed.loc[neutral, units].sum(), closed[units].sum())
    synthetic["tasa unidades neutral"] = share(closed.loc[neutral, renewed].sum(), closed.loc[neutral, units].sum())
    for state in states:
        only = (closed[state] == 1) & (closed[states].sum(axis=1) == 1)
        synthetic[f"tasa unidades {state}"] = share(closed.loc[only, renewed].sum(), closed.loc[only, units].sum())
    unknown = raw[configuration.discount_value_column].isna()
    synthetic["cuota del USD que vence con descuento desconocido"] = share(raw.loc[unknown, usd].sum(), raw[usd].sum())

    # [2] the ladder and the risk levels
    ladder = results["ladder"]["summary"]
    synthetic["cuota del dinero por predecir con suelo, sin escalera"] = float(ladder["pct_usd_floor"].iloc[0])
    synthetic["cuota del dinero por predecir con suelo, final"] = float(ladder["pct_usd_floor"].iloc[-1])
    synthetic["cuota del dinero con tasa propia, final"] = float(ladder["pct_usd_own_rate"].iloc[-1])
    levels = results["money_by_level"].set_index("nivel_riesgo")["pct_usd"]
    synthetic["dinero en A_propio + A3_propio_reforzado"] = float(levels.reindex(["A_propio", "A2_propio_corto",
                                                                               "A3_propio_reforzado"]).fillna(0).sum())
    synthetic["dinero en B_prestado + C_lejano"] = float(levels.reindex(["B_prestado", "C_lejano"]).fillna(0).sum())

    # [3] the dynamics and the backtest of the rate
    synthetic["φ mediana de los pools"] = float(results["pool_dynamics"]["phi"].median())
    synthetic["p-valor del efecto mes de la cartera (alto = sin estacionalidad)"] = float(results["portfolio_dynamics"]["p_valor_mes"])
    backtest = results["backtest"]
    selection = backtest["predictions"][backtest["predictions"]["proposito"] != "examen"]
    ranking = selection.assign(abs_norm=selection["err_norm"].abs()).groupby(["tramo_h", "tecnica"])["abs_norm"].mean()
    for label, band in (("corto", "corto"), ("medio-largo", "medio_largo")):
        synthetic[f"mejor |err_norm| a {label} (selección)"] = float(ranking.loc[band].min()) if band in ranking.index.get_level_values(0) else np.nan
        pools = backtest["exam_by_pool"][backtest["exam_by_pool"]["tramo_h"] == band]
        synthetic[f"mejora de la elegida sobre la técnica por defecto a {label} (pp)"] = float(pools["mejora_pp"].mean())
    short_pools = backtest["exam_by_pool"][backtest["exam_by_pool"]["tramo_h"] == "corto"]
    synthetic["dentro de la banda a corto"] = float(short_pools["dentro_banda"].mean())
    exam_total = backtest["exam_total"]
    for horizon in (1, 6):
        rows = exam_total[exam_total["h"] == horizon]
        synthetic[f"error del total de los pools juzgados, h = {horizon}"] = share(rows["elegida_renovadas_pred"].sum(),
                                                                                  rows["renovadas_reales"].sum()) - 1
    summary = results["series_exam"]["summary"].set_index(["metodo", "h"])
    synthetic["error del total del framework, h = 1 (paso 7.1)"] = float(summary.loc[("framework", 1), "error_total_medio"])
    synthetic["error del total del framework, h = 6 (paso 7.1)"] = float(summary.loc[("framework", 6), "error_total_medio"])
    synthetic["WAPE serie a serie framework / serie sola, h = 1"] = share(summary.loc[("framework", 1), "wape_series"],
                                                                         summary.loc[("raw", 1), "wape_series"])
    synthetic["dentro del intervalo, framework h = 1"] = float(summary.loc[("framework", 1), "en_intervalo"])

    # [4] the uplift: three ways, the contract rule, the increases
    current = configuration.calendar_boundaries()["current"]
    renewers = renewer_rows(results["fine_table"], configuration)
    window = uplift_history(renewers, current, results["price_increases"], configuration.uplift_window_months, True,
                            configuration)
    synthetic["uplift toda la historia"] = share(renewers["_numerador"].sum(), renewers["_denominador"].sum())
    synthetic["uplift ventana"] = share(window["_numerador"].sum(), window["_denominador"].sum())
    synthetic["uplift ventana sin subidas"] = share(window["_numerador_uplift"].sum(), window["_denominador"].sum())
    contract = results["contract_check"].set_index("tramo")["ratio_realizacion"]
    synthetic["ratio de realización de la regla del contrato, 0-10 %"] = float(contract.iloc[0])
    middle = [label for label in contract.index if "50-60" in str(label)]
    synthetic["ratio de realización de la regla del contrato, 50-60 %"] = float(contract.loc[middle[0]]) if middle else np.nan
    paths = results["uplift_backtest"].set_index(["filas", "via"])["sesgo"]
    synthetic["sesgo vía contrato en el examen"] = float(paths.get(("descuento conocido", "contrato"), np.nan))
    synthetic["sesgo vía estadística en el examen (descuento conocido)"] = float(paths.get(("descuento conocido", "estadistica"), np.nan))
    synthetic["subidas en juego (ventana)"] = len(results["price_increases"])
    synthetic["subidas detectadas en la historia"] = len(results["power_bi"]["price_calendar"])

    # [5] the forecast: acquisitions, states, the months and the years
    for value, label in (("Acquisition_Pure-New", "Pure-New"), ("Acquisition_Not-New", "Not-New")):
        rows = closed[closed["net_new"] == value]
        months = pd.PeriodIndex(rows[configuration.period_col].astype(str), freq="M")
        recent = rows[months >= current - 3][units].sum()
        before = rows[(months >= current - 15) & (months < current - 12)][units].sum()
        synthetic[f"nivel de adquisición {label}"] = share(recent, before)
    maturation = results["forecast"]["maturation"].copy()
    maturation["ano"] = maturation[configuration.period_col].astype(str).str[:4].astype(int)
    by_year = maturation.groupby("ano")[["ajuste_usd", "renovado_usd_base"]].sum()
    current_year = current.year
    synthetic["ajuste del reparto de estados del cliente, año en curso"] = share(by_year.loc[current_year, "ajuste_usd"], by_year.loc[current_year, "renovado_usd_base"])
    synthetic["ajuste del reparto de estados del cliente, año siguiente"] = share(by_year.loc[current_year + 1, "ajuste_usd"], by_year.loc[current_year + 1, "renovado_usd_base"])
    by_month = results["forecast"]["by_month"]
    synthetic["tasa USD esperada del primer mes futuro"] = float(by_month["tasa_usd"].iloc[0])
    years = results["forecast"]["by_year"].set_index("ano")
    synthetic["renovado del año siguiente / año en curso"] = share(years.loc[current_year + 1, "total_usd"], years.loc[current_year, "total_usd"])
    synthetic["cuota de lo renovado del año siguiente que sale de pipeline modelada"] = share(
        years.loc[current_year + 1, ["esperado_proyectada_usd", "esperado_simulada_usd"]].sum(), years.loc[current_year + 1, "total_usd"])
    first_month = str(current)
    wiped = raw[(raw[configuration.period_col].astype(str) == first_month)]["s0_renovados_usd"].sum()
    synthetic["primer mes futuro: esperado / lo ya renovado que se borró"] = share(by_month["esperado_usd"].iloc[0], wiped)

    table = pd.DataFrame([{"indicador": label, "real": real, "sintetico": synthetic.get(label, np.nan), "donde_real": where}
                          for label, (real, where) in REAL_RUN.items()])
    table["sintetico"] = table["sintetico"].astype(float).round(3)
    return table


def compare_with_truth(results: dict, truth: pd.DataFrame) -> pd.DataFrame:
    """The forecast of every future month against what REALLY renews in it (the synthetic knows it): the only exam
    of the whole forecast, the reparto of the states, the uplift and the band included. One row per month up to the
    last month whose whole pipeline is in the extract.
        error_pct              the forecast against the truth
        error_sin_reparto_pct  the same with the states of today (without the reparto of step 7.2)
        banda_cuadratura_pct   the half-width of the band the forecast shows (independent errors)"""
    by_month = results["forecast"]["by_month"].assign(period=lambda frame: frame["period"].astype(str)).set_index("period")
    truth = truth.set_index("period")
    months = [month for month in truth.index if month in by_month.index]
    table = pd.DataFrame({"verdad_usd": truth.loc[months, "renovado_usd"], "forecast_usd": by_month.loc[months, "esperado_usd"]})
    maturation = results["forecast"].get("maturation")
    if maturation is not None and len(maturation):
        base = maturation.assign(period=maturation["period"].astype(str)).set_index("period")["renovado_usd_base"]
        table["sin_reparto_usd"] = base.reindex(months)
    table["error_pct"] = 100 * (table["forecast_usd"] / table["verdad_usd"] - 1)
    if "sin_reparto_usd" in table:
        table["error_sin_reparto_pct"] = 100 * (table["sin_reparto_usd"] / table["verdad_usd"] - 1)
    table["banda_cuadratura_pct"] = 100 * (by_month.loc[months, "banda_cuadratura_alta"] - by_month.loc[months, "esperado_usd"]) / table["forecast_usd"]
    total = table[["verdad_usd", "forecast_usd"] + (["sin_reparto_usd"] if "sin_reparto_usd" in table else [])].sum()
    table.loc["TOTAL"] = list(total) + [100 * (total["forecast_usd"] / total["verdad_usd"] - 1)] + (
        [100 * (total["sin_reparto_usd"] / total["verdad_usd"] - 1)] if "sin_reparto_usd" in table else []) + [np.nan]
    return table.round(2)


if __name__ == "__main__":
    from main import LIKE_REAL_SEED, like_real_configuration, run
    from synthetic_real import build_raw_like_real
    configuration = like_real_configuration(*sys.argv[1:2])
    results = run(configuration)
    _, future_truth = build_raw_like_real(seed=LIKE_REAL_SEED, return_truth=True)
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 70):
        print("\n=== the synthetic next to the real run ===")
        print(compare_results(results, configuration).to_string(index=False))
        print("\n=== the forecast against the future truth of the synthetic ===")
        print(compare_with_truth(results, future_truth).to_string())
