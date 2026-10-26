"""
resumen_negocio.py — The summary for the business: how the forecast was built and how much to trust it, in a
few direct lines.

It reads the results of the run (it computes nothing new) and says, in this order:
  1. LA PIPELINE      what falls due from the current month to the end of the horizon, and how many forecast series
  2. LA PREDICCIÓN    N series × M months = P predictions of the rate; with what support (own, borrowed) and which
                      techniques carry the money
  3. EL RESULTADO     the expected renewals per year with their band; LA MADURACIÓN: what the states still to come take
                      from the forecast with today's states (0 with the identity box)
  4. CUÁNTO FIARSE    the error of the total of a month in the exam, per horizon (and with the states predicted, when
                      the extract dates them); the gain over the baseline in USD and in how many months it holds
  5. EL RUIDO         the binomial noise: how much the rate of a series alone moves by chance, how much is left after
                      the ladder and the credibility, and how close the exam error is to that limit
  6. LA BANDA         what the official band of every year is made of: the rate and the price, the simulated
                      acquisitions, the time series universe — against the share of the money each one is
  7. LO QUE HAY QUE VALIDAR   the hypotheses of the blocks that do not hold or cannot be told on these data
                      (verdicts.py, sff_veredicto_bloques)
  8. OJO              what the numbers do not cover

Used by step 8.6 (on screen at the end of the run, and at the top of informe_sff.md).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (Config, METHOD_BASELINE, METHOD_FRAMEWORK, PIPELINE_ORIGIN_COLUMN, PIPELINE_REAL, SERIES_ID_COLUMN,
                    VERDICT_FAILS, VERDICT_INCONCLUSIVE, VERDICT_WARNING)

OWN_LEVEL_PREFIX = "A"                  # the risk levels of a series predicted with its own support (A, A2, A3)
BORROWED_LEVEL_PREFIX = "B"             # merged with relatives that share every mandatory dim
TECHNIQUES_NAMED = 3                    # the techniques named by their share of the money
QUESTIONS_NAMED = 4                     # the hypotheses named in LO QUE HAY QUE VALIDAR


def money(value: float) -> str:
    """$1.2M, $340k, $5,100: money as the business reads it."""
    if not np.isfinite(value):
        return "—"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if abs(value) >= 1e4:
        return f"${value / 1e3:,.0f}k"
    return f"${value:,.0f}"


def gain_text(gain: float) -> str:
    """The gain over the baseline in words: less error is a gain, more error is not."""
    return f"{money(gain)} menos de error al mes" if gain >= 0 else f"{money(-gain)} MÁS de error al mes que la baseline"


def months_text(horizon: int) -> str:
    return f"{int(horizon)} mes" if int(horizon) == 1 else f"{int(horizon)} meses"


def exam_error_of_months(results: dict, configuration: Config):
    """The typical error of the total of every future month, from the exam: the mean |error| of the monthly total at
    its horizon (h = the month − the last closed one), interpolated between the examined horizons and held beyond the
    last one; with the customer states predicted (honest exam) when the extract dates them. None without an exam.
    It is the official band the forecast already carries (error_examen_*, step 7.2); it is recomputed only when the
    forecast was built without the exam."""
    exam = results.get("series_exam")
    if exam is None:
        return None
    official = results["forecast"]["by_month"]
    if "error_examen_usd" in official.columns:
        return official.assign(error_pct=official["error_examen_pct"], error_usd=official["error_examen_usd"])
    summary = exam["summary"]
    method = "framework_honesto" if (summary["metodo"] == "framework_honesto").any() else METHOD_FRAMEWORK
    rows = summary[summary["metodo"] == method].sort_values("h")
    if rows.empty:
        return None
    by_month = results["forecast"]["by_month"].copy()
    last_closed = configuration.calendar_boundaries()["current"] - 1
    horizons = by_month[configuration.period_col].map(lambda month: (month - last_closed).n).to_numpy(dtype=float)
    by_month["error_pct"] = np.interp(horizons, rows["h"].to_numpy(dtype=float), rows["error_total_medio"].to_numpy(dtype=float))
    by_month["error_usd"] = by_month["error_pct"] * by_month["esperado_usd"]
    return by_month


def business_summary(results: dict, configuration: Config) -> list:
    """The lines of the summary for the business (plain text, one idea per line)."""
    lines = []
    forecast = results.get("forecast")
    if forecast is None:
        return ["(no hay forecast en esta ejecución)"]
    rows = forecast["forecast"]
    period, usd_due = configuration.period_col, configuration.pipeline_usd_col

    # 1 · the pipeline
    months = sorted(rows[period].unique())
    month_count = (months[-1] - months[0]).n + 1
    real = rows[rows[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL]
    extended = rows[rows[PIPELINE_ORIGIN_COLUMN] != PIPELINE_REAL]
    series_with_pipeline = rows.loc[rows[usd_due] > 0, SERIES_ID_COLUMN].nunique()
    lines.append(f"LA PIPELINE · de {months[0]} a {months[-1]} ({month_count} meses) vencen {money(rows[usd_due].sum())}: "
                 f"{money(real[usd_due].sum())} ya están en el extracto y {money(extended[usd_due].sum())} son renovaciones y "
                 f"captación que aún no existen (horizonte extendido). Lo forman {series_with_pipeline:,} forecast series.")

    # 2 · the prediction: series × months, support and techniques
    predictions = rows[rows[usd_due] > 0].drop_duplicates([SERIES_ID_COLUMN, period])
    lines.append(f"LA PREDICCIÓN · cada forecast series se predice mes a mes: {predictions[SERIES_ID_COLUMN].nunique():,} series × "
                 f"hasta {month_count} meses = {len(predictions):,} predicciones de la tasa de renovación. Cada una: lo que vence × "
                 f"tasa × uplift (el precio al renovar), con las unidades que tendrá al vencer: la maduración de los estados del cliente que aún no "
                 f"han llegado (softcancel, dormant…).")
    money_by_level = results.get("money_by_level")
    if money_by_level is not None and len(money_by_level):
        level = money_by_level.set_index("nivel_riesgo")["pct_usd"]
        own = float(level[[name for name in level.index if name.startswith(OWN_LEVEL_PREFIX)]].sum())
        borrowed = float(level[[name for name in level.index if name.startswith(BORROWED_LEVEL_PREFIX)]].sum())
        lines.append(f"  · soporte: el {own:.0%} del dinero lo predicen series con soporte propio; el {borrowed:.0%} toma prestado "
                     f"de series casi iguales (mismas dimensiones mandatory) y el {max(1 - own - borrowed, 0):.0%} de grupos más "
                     f"amplios o sin historia.")
    with_technique = real[real["esperado_usd"] > 0]
    if len(with_technique):
        technique_share = with_technique.groupby("tecnica")["esperado_usd"].sum() / with_technique["esperado_usd"].sum()
        technique_share = technique_share.sort_values(ascending=False)
        default_share = float(technique_share.get(configuration.default_technique, 0.0))
        named = " · ".join(f"{technique} {share:.0%}" for technique, share in technique_share.head(TECHNIQUES_NAMED).items())
        # what the exam says of the techniques chosen (the verdict of block 2), so the line does not promise a gain
        # the exam did not see
        technique_rows = pd.DataFrame()
        rate_verdict = getattr(configuration, "block_verdicts", {}).get("B2")
        if rate_verdict is not None and len(rate_verdict):
            technique_rows = rate_verdict[rate_verdict["pregunta"].str.contains("técnica elegida mejora", regex=False)]
        if len(technique_rows):
            exam_says = " · ".join(f"{row['pregunta'].split('(')[-1].rstrip('?)')} {row['valor']}, {row['veredicto']}"
                                   for _, row in technique_rows.iterrows())
            exam_text = f"; en el examen, frente a la técnica por defecto: {exam_says}"
        else:
            exam_text = ""
        lines.append(f"  · técnicas: {named}. La técnica por defecto ({configuration.default_technique}, la media de los 3 "
                     f"últimos meses) lleva el {default_share:.0%}; las demás ganaron en los meses de selección del backtest"
                     f"{exam_text}.")

    # 3 · the result
    by_year = forecast["by_year"]
    future_years = by_year[by_year["esperado_usd"] > 0]
    monthly_error = exam_error_of_months(results, configuration)
    for _, year in future_years.iterrows():
        time_series_part = sum(float(year.get(column_name, 0.0)) for column_name in
                               ("ingresos_ts_real_usd", "ingresos_ts_usd", "esperado_ts_reentrada_usd"))
        line = (f"EL RESULTADO {int(year['ano'])} · {money(year['total_usd'])} ({money(year['renovado_real_usd'])} ya "
                f"renovados + {money(year['esperado_usd'])} esperados")
        if time_series_part:
            line += (f" + {money(time_series_part)} del universo time_series, retail a suscripción: "
                     f"{money(year.get('ingresos_ts_real_usd', 0.0))} ya convertidos, {money(year.get('ingresos_ts_usd', 0.0))} "
                     f"simulados y {money(year.get('esperado_ts_reentrada_usd', 0.0))} de renovación de su reentrada")
        line += ")"
        if monthly_error is not None:
            of_year = monthly_error[monthly_error[period].map(lambda month: month.year) == int(year["ano"])]
            typical = of_year["error_usd"].sum()
            future_part = year["esperado_usd"] + float(year.get("ingresos_ts_usd", 0.0)) + float(year.get("esperado_ts_reentrada_usd", 0.0))
            line += (f"; error típico ±{money(typical)} ({typical / future_part:.1%} de lo que falta), del examen: el error "
                     f"de cada mes a su horizonte, sumado (meses en la misma dirección): tasa y precio, captación simulada"
                     + (" y la simulación time series" if time_series_part else ""))
        lines.append(line + ".")

    # 3b · the maturation of the states (stage 4.1): what it takes from the forecast with today's states
    maturation = forecast.get("maturation")
    if maturation is None:
        lines.append("LA MADURACIÓN · desactivada (caja identidad): el forecast es el de los estados de hoy; maduración 0.")
    elif len(maturation):
        year_of = maturation[period].map(lambda month: month.year)
        by_year_maturation = maturation.groupby(year_of)[["renovado_usd_base", "ajuste_usd"]].sum()
        parts = " · ".join(f"{int(year)} {money(row['ajuste_usd'])} ({row['ajuste_usd'] / row['renovado_usd_base']:+.1%})"
                           for year, row in by_year_maturation.iterrows() if row["renovado_usd_base"] > 0)
        lines.append(f"LA MADURACIÓN · los estados del cliente que aún no han llegado (una licencia que vence dentro de meses aún puede hacer "
                     f"softcancel o quedarse dormant) cambian el forecast con los estados de hoy en: {parts}.")

    # 4 · how much to trust it: the exam of the total, the gain over the baseline
    exam = results.get("series_exam")
    if exam is not None:
        summary = exam["summary"]
        framework = summary[summary["metodo"] == METHOD_FRAMEWORK].set_index("h")
        honest = summary[summary["metodo"] == "framework_honesto"].set_index("h")
        baseline = summary[summary["metodo"] == METHOD_BASELINE].set_index("h")
        if len(framework):
            errors = honest["error_total_medio"] if len(honest) else framework["error_total_medio"]
            monthly_expected = rows.groupby(period)["esperado_usd"].sum().mean()
            per_horizon = " · ".join(f"a {months_text(horizon)} un {error:.1%} ({money(error * monthly_expected)})"
                                     for horizon, error in errors.sort_index().items())
            lines.append(f"CUÁNTO FIARSE · en el examen (meses ya cerrados, prediciendo con lo que se sabía antes), el total de un "
                         f"mes se erró de media {per_horizon}, sobre los {money(monthly_expected)} que se esperan renovar de media "
                         f"al mes" + (" (con los estados del cliente también predichos)." if len(honest) else "."))
            if len(baseline) and "ganancia_usd_mes" in framework.columns:
                gains = " · ".join(f"a {months_text(horizon)} {gain_text(row['ganancia_usd_mes'])}, mejor en el "
                                   f"{row['meses_mejor_que_baseline']:.0%} de los meses"
                                   for horizon, row in framework.iterrows())
                lines.append(f"  · frente a la baseline (su celda mandatory, últimos {configuration.baseline_months} meses, el cálculo "
                             f"que cualquiera haría): {gains}.")

    # 5 · the binomial noise: before and after the improvements, and the exam against it
    estimate = results.get("series_estimate")
    if estimate is not None and len(estimate):
        weights = estimate["usd_por_predecir"].fillna(0)
        alone = estimate["error_binomial_pp"]
        after = configuration.z * estimate["se_estimacion_pp"]
        within_alone = float(weights[alone <= configuration.promise_pp].sum() / weights.sum()) if weights.sum() else np.nan
        within_after = float(weights[after <= configuration.promise_pp].sum() / weights.sum()) if weights.sum() else np.nan
        mean_alone = float(np.average(alone.fillna(alone.max()), weights=weights + 1e-12))
        mean_after = float(np.average(after.fillna(alone.max()), weights=weights + 1e-12))
        line = (f"EL RUIDO · sola, la tasa de una serie se mueve por azar ±{mean_alone:.1f} pp de media (90 %, ponderado por "
                f"dinero); con la escalera y la credibilidad, ±{mean_after:.1f} pp. Dinero con la tasa conocida a ±"
                f"{configuration.promise_pp:.0f} pp: {within_alone:.0%} sola → {within_after:.0%} tras las mejoras.")
        if exam is not None and len(framework) and "error_vs_ruido" in framework.columns:
            ratio = framework["error_vs_ruido"]
            times = f"{ratio.min():.1f}" if round(ratio.min(), 1) == round(ratio.max(), 1) else f"{ratio.min():.1f}–{ratio.max():.1f}"
            line += f" En el examen, el error de cada serie es {times} veces su ruido inevitable (1 = lo mejor posible)."
        lines.append(line)

    # 6 · what the band is made of, against the share of the money of each part
    if "banda_tasa_y_precio_usd" in by_year.columns:
        for _, year in future_years.iterrows():
            band = float(year.get("error_examen_usd", 0.0))
            if band <= 0:
                continue
            time_series_money = float(year.get("ingresos_ts_usd", 0.0)) + float(year.get("esperado_ts_reentrada_usd", 0.0))
            to_come = float(year["esperado_usd"]) + time_series_money
            lines.append(f"LA BANDA {int(year['ano'])} · ±{money(band)}: tasa y precio {year['banda_tasa_y_precio_usd'] / band:.0%} "
                         f"(tasa {money(year['banda_tasa_usd'])}, precio {money(year['banda_precio_usd'])}), captación simulada "
                         f"{year['banda_captacion_usd'] / band:.0%}, universo time series {year['banda_time_series_usd'] / band:.0%}"
                         + (f" (es el {time_series_money / to_come:.0%} del dinero que falta)" if time_series_money and to_come else "")
                         + ". Es el error típico (≈ 1 error medio), no un rango del 90 %.")

    # 7 · the hypotheses that do not hold or cannot be told on these data
    verdicts = [frame for frame in getattr(configuration, "block_verdicts", {}).values() if len(frame)]
    if verdicts:
        every = pd.concat(verdicts, ignore_index=True)
        questions = every[every["tipo"] == "hipótesis"]
        open_ones = questions[questions["veredicto"].isin([VERDICT_FAILS, VERDICT_INCONCLUSIVE, VERDICT_WARNING])]
        if len(questions):
            named = " · ".join(f"{row['bloque']} {row['pregunta'].rstrip('?').lstrip('¿')} ({row['veredicto']}: {row['valor']})"
                               for _, row in open_ones.head(QUESTIONS_NAMED).iterrows())
            lines.append(f"LO QUE HAY QUE VALIDAR · {len(questions) - len(open_ones)} de {len(questions)} hipótesis del framework se "
                         f"cumplen en estos datos" + (f"; no se cumplen o no son concluyentes: {named}" if named else "")
                         + (f" (y {len(open_ones) - QUESTIONS_NAMED} más)" if len(open_ones) > QUESTIONS_NAMED else "")
                         + ". Todas en sff_veredicto_bloques.")

    # 8 · what the numbers do not cover
    caveats = ["la estructura (escalera, credibilidad, técnica elegida) se estima con todos los meses cerrados: el examen del "
               "framework es algo optimista, sobre todo a 6 meses o más"]
    examined = max(configuration.backtest_horizons)
    last_closed = configuration.calendar_boundaries()["current"] - 1
    if (months[-1] - last_closed).n > examined:
        caveats.append(f"el examen llega a {months_text(examined)}: más lejos, el error típico se toma igual que a {examined}, y "
                       f"seguramente es mayor")
    without_columns = [state for state, sign in configuration.structural_timevarying_dims.items()
                       if sign == "negative" and state not in configuration.maturation_dims]
    if without_columns:
        caveats.append(f"{', '.join(without_columns)} no trae(n) su fecha: su maduración sale de la proporción histórica, no de "
                       f"una curva examinada")
    lines.append("OJO · " + "; ".join(caveats) + ".")
    return lines


def show_business_summary(results: dict, configuration: Config, step_label: str) -> list:
    """The summary on screen, under its header; returns the lines (for the report)."""
    doc = configuration.logger.doc
    lines = business_summary(results, configuration)
    doc(f"[{step_label}] ══════════ RESUMEN PARA NEGOCIO ══════════")
    for line in lines:
        doc(f"[{step_label}] {line}")
    doc(f"[{step_label}] ══════════ FIN DEL RESUMEN PARA NEGOCIO ══════════")
    return lines
