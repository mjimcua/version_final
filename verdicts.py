"""
verdicts.py — THE VERDICT OF EVERY BLOCK: the hypotheses the framework rests on, contrasted with the data of the run.

WHY: the framework is a chain of hypotheses (a series alone is noise; borrowing history lowers the error; the technique
chosen in the backtest beats the default one; the statistical uplift beats the contract rule; the maturation curve
predicts the final share of a state; simulating the acquisition beats "the same as a year ago"; the framework beats the
baseline...). On the synthetic data they are all built in, so the gains say little. On the real data every one of them
must be checked again. This module says, after every block, which hypotheses held, with the number that says so.

WHAT A VERDICT ROW IS (one per question; the table sff_veredicto_bloques, written by step 8.6):
  bloque        B1 … B8
  paso          the step whose result answers it
  tipo          hipótesis (what the framework assumes) · examen (a measured precision) · decisión (what the run chose)
                · dato (a fact about the data the hypotheses rest on) · aviso
  pregunta      the question, in one line
  metrica       how it is measured
  valor         the number of this run
  referencia    what it is compared with (the baseline, the threshold, the alternative)
  veredicto     se cumple · no se cumple · no concluyente · dato · revisar (config.py, VERDICT_*)
  detalle       one line: what it means, what to do

THE PROFILE OF THE PIPELINE (block 1; the table sff_perfil_datos): volume, distribution, concentration, support and
history of the money to predict, by dimension and by month — what has to be seen of the real data before reading any
gain of the framework.

THE INTERFACE:
  rows = verdict_of_block("B3", tables, configuration)    the rows of one block, from the results of the run so far
  record_the_verdict("B3", tables, configuration)         the rows, kept in configuration.block_verdicts and on screen
  verdict_table(configuration)                            every block's rows, in order (sff_veredicto_bloques)
  pipeline_profile(fine_table, series_table, series_rate, configuration)   the profile (sff_perfil_datos)

  The orchestrator (pipeline.py) records the verdict of every block right after it runs; nothing here changes a number
  of the forecast. A verdict that cannot be computed (the block did not run, a result is missing) is left out.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats

from config import (CALENDAR_ROLE_COLUMN, Config, METHOD_BASELINE, METHOD_FRAMEWORK, ROLE_PROJECTION, SERIES_ID_COLUMN,
                    TRUTH_ROLES, UNKNOWN_DISCOUNT_BUCKET, VERDICT_FAILS, VERDICT_HOLDS, VERDICT_INCONCLUSIVE,
                    VERDICT_INFORMATIVE, VERDICT_WARNING, negative_states)

# ─── named constants (the thresholds of the questions are fields of the Config: verdict_*) ───────────────
VERDICT_COLUMNS = ["bloque", "paso", "tipo", "pregunta", "metrica", "valor", "referencia", "veredicto", "detalle"]
BLOCK_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
KIND_HYPOTHESIS, KIND_EXAM, KIND_DECISION, KIND_FACT, KIND_WARNING = "hipótesis", "examen", "decisión", "dato", "aviso"

MONTHS_PER_YEAR = 12
FULL_HISTORY_MONTHS = MONTHS_PER_YEAR  # a year of own history: every season seen once
CONCENTRATION_SHARES = (0.5, 0.8, 0.95)
CONCENTRATION_TOPS = (10, 50, 100, 500, 1000)
HISTORY_BUCKETS = [(0, 0, "0 · sin historia"), (1, 11, "1-11 meses"), (12, 23, "12-23 meses"), (24, 35, "24-35 meses"),
                   (36, 10 ** 6, "≥ 36 meses")]


# ═══════════════════════════════════════════════════════════════════════════════════
# THE ROWS
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_row(block: str, step: str, kind: str, question: str, metric: str, value, reference, verdict: str,
                detail: str = "") -> dict:
    """One row of the verdict (VERDICT_COLUMNS); value and reference as text, already formatted."""
    return {"bloque": block, "paso": step, "tipo": kind, "pregunta": question, "metrica": metric,
            "valor": value if isinstance(value, str) else f"{value}", "referencia": reference if isinstance(reference, str)
            else f"{reference}", "veredicto": verdict, "detalle": detail}


def percent(value: float, decimals: int = 1) -> str:
    """0.123 → '12.3%'; a NaN → '—'."""
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:.{decimals}f}%"


def money(value: float) -> str:
    """$1.2M, $340k, $5,100."""
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if abs(value) >= 1e4:
        return f"${value / 1e3:,.0f}k"
    return f"${value:,.0f}"


def paired_comparison(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """Two methods on the same months: |error| of A against |error| of B, month by month.

    RETURNS: months, wins of A (|a| < |b|), the sign test p-value (two-sided, ties left out) and the paired t-test
    p-value on |b| − |a| (two-sided; NaN with fewer than 3 months or no variation). With 6 exam months a sign test
    can only be significant at 6 of 6 (p = 0.03); 5 of 6 gives p = 0.22: consistent, not proven."""
    a, b = np.abs(np.asarray(errors_a, dtype=float)), np.abs(np.asarray(errors_b, dtype=float))
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    differences = b - a
    wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
    decided = wins + losses
    sign_p = float(stats.binomtest(wins, decided, 0.5).pvalue) if decided else np.nan
    t_p = np.nan
    if len(differences) >= 3 and np.std(differences, ddof=1) > 0:
        t_p = float(stats.ttest_1samp(differences, 0.0).pvalue)
    return {"meses": int(len(a)), "gana": wins, "pierde": losses, "p_signos": sign_p, "p_t": t_p,
            "diferencia_media": float(differences.mean()) if len(differences) else np.nan}


def gain_verdict(comparison: dict, significance: float) -> str:
    """Better and significant (sign or t test below `significance`, the Config's verdict_significance) → se cumple;
    better on average but not significant → no concluyente; not better → no se cumple."""
    if not comparison["meses"]:
        return VERDICT_INCONCLUSIVE
    if comparison["diferencia_media"] <= 0:
        return VERDICT_FAILS
    p_values = [p for p in (comparison["p_signos"], comparison["p_t"]) if np.isfinite(p)]
    return VERDICT_HOLDS if p_values and min(p_values) < significance else VERDICT_INCONCLUSIVE


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PROFILE OF THE PIPELINE (block 1)
# ═══════════════════════════════════════════════════════════════════════════════════

PROFILE_COLUMNS = ["seccion", "dimension", "valor", "orden", "series", "unidades_vencen", "usd_vence",
                   "unidades_renovadas", "usd_renovado", "tasa_unidades", "tasa_usd", "usd_vence_12m_anteriores",
                   "crecimiento_anual", "usd_vence_futuro", "cuota_usd_futuro", "cuota_futuro_con_soporte"]


def pipeline_profile(fine_table: pd.DataFrame, series_table: pd.DataFrame, series_rate: pd.DataFrame,
                     configuration: Config) -> pd.DataFrame:
    """The profile of the pipeline: what has to be seen of the data before reading any gain of the framework.

    SECTIONS (one schema, PROFILE_COLUMNS; the closed figures are the last 12 closed months, the future ones every month
    from the current one on):
      cartera        one row: the whole portfolio
      mes            one row per month: due, renewed and the rate (closed months), due (future months)
      dimension      one row per value of every mandatory dim, extra_renovacion, customer state and discount bucket:
                     its weight in the money to predict, its rate, the growth of its pipeline, how much of its future
                     money sits in series with support of their own
      concentracion  how many series hold 50 / 80 / 95 % of the money to predict; the share of the top 10 … 1000
      soporte        the money to predict by own support of the series (n_propio: units due in a typical closed month)
      historia       the money to predict by months of own history of the series
    """
    period = configuration.period_col
    units_due, usd_due = configuration.pipeline_units_col, configuration.pipeline_usd_col
    units_renewed, usd_renewed = configuration.renewed_units_col, configuration.renewed_usd_col
    current = configuration.calendar_boundaries()["current"]
    last_year = fine_table[(fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)) & (fine_table[period] >= current - MONTHS_PER_YEAR)]
    year_before = fine_table[(fine_table[period] >= current - 2 * MONTHS_PER_YEAR) & (fine_table[period] < current - MONTHS_PER_YEAR)]
    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION]
    future_total = float(future[usd_due].sum())
    supported = set(series_rate.loc[series_rate["n_propio"] >= configuration.support_floor, SERIES_ID_COLUMN])

    def figures(closed: pd.DataFrame, before: pd.DataFrame, ahead: pd.DataFrame) -> dict:
        due_units, due_usd = float(closed[units_due].sum()), float(closed[usd_due].sum())
        renewed_units, renewed_usd = float(closed[units_renewed].sum()), float(closed[usd_renewed].sum())
        before_usd = float(before[usd_due].sum())
        ahead_usd = float(ahead[usd_due].sum())
        ahead_supported = float(ahead.loc[ahead[SERIES_ID_COLUMN].isin(supported), usd_due].sum())
        return {"series": int(pd.concat([closed[SERIES_ID_COLUMN], ahead[SERIES_ID_COLUMN]]).nunique()),
                "unidades_vencen": due_units, "usd_vence": due_usd, "unidades_renovadas": renewed_units,
                "usd_renovado": renewed_usd, "tasa_unidades": renewed_units / due_units if due_units else np.nan,
                "tasa_usd": renewed_usd / due_usd if due_usd else np.nan, "usd_vence_12m_anteriores": before_usd,
                "crecimiento_anual": due_usd / before_usd - 1 if before_usd else np.nan, "usd_vence_futuro": ahead_usd,
                "cuota_usd_futuro": ahead_usd / future_total if future_total else np.nan,
                "cuota_futuro_con_soporte": ahead_supported / ahead_usd if ahead_usd else np.nan}

    rows = [{"seccion": "cartera", "dimension": "TOTAL", "valor": "TOTAL", "orden": 0,
             **figures(last_year, year_before, future)}]

    # every month: what fell due and renewed, or what will fall due
    by_month = fine_table.groupby(period).agg(series=(SERIES_ID_COLUMN, "nunique"), unidades_vencen=(units_due, "sum"),
                                              usd_vence=(usd_due, "sum"), unidades_renovadas=(units_renewed, "sum"),
                                              usd_renovado=(usd_renewed, "sum")).reset_index()
    for order, month_row in enumerate(by_month.itertuples(index=False), 1):
        month = getattr(month_row, period)
        closed = month < current
        rows.append({"seccion": "mes", "dimension": period, "valor": str(month), "orden": order, "series": month_row.series,
                     "unidades_vencen": month_row.unidades_vencen, "usd_vence": month_row.usd_vence,
                     "unidades_renovadas": month_row.unidades_renovadas if closed else np.nan,
                     "usd_renovado": month_row.usd_renovado if closed else np.nan,
                     "tasa_unidades": month_row.unidades_renovadas / month_row.unidades_vencen if closed and month_row.unidades_vencen else np.nan,
                     "tasa_usd": month_row.usd_renovado / month_row.usd_vence if closed and month_row.usd_vence else np.nan,
                     "usd_vence_futuro": month_row.usd_vence if not closed else np.nan,
                     "cuota_usd_futuro": month_row.usd_vence / future_total if not closed and future_total else np.nan})

    # every value of every dimension
    dimensions = list(dict.fromkeys(configuration.business_mandatory_dims + list(configuration.extra_renovacion)
                                    + negative_states(configuration)
                                    + ([configuration.discount_bucket_column] if configuration.discount_value_column else [])))
    for dimension in [column for column in dimensions if column in fine_table.columns]:
        values = pd.Series(fine_table[dimension].astype(str).unique()).sort_values().tolist()
        last_year_value, before_value, future_value = (last_year[dimension].astype(str), year_before[dimension].astype(str),
                                                       future[dimension].astype(str))
        for order, value in enumerate(values, 1):
            rows.append({"seccion": "dimension", "dimension": dimension, "valor": value, "orden": order,
                         **figures(last_year[last_year_value == value], year_before[before_value == value],
                                   future[future_value == value])})

    # how concentrated the money to predict is
    money_of_series = future.groupby(SERIES_ID_COLUMN)[usd_due].sum().sort_values(ascending=False)
    cumulative = money_of_series.cumsum() / future_total if future_total else money_of_series * 0
    for order, share in enumerate(CONCENTRATION_SHARES, 1):
        needed = int((cumulative < share).sum()) + 1 if len(cumulative) else 0
        rows.append({"seccion": "concentracion", "dimension": "series para la cuota", "valor": f"{share:.0%}", "orden": order,
                     "series": needed, "cuota_usd_futuro": share})
    for order, top in enumerate(CONCENTRATION_TOPS, len(CONCENTRATION_SHARES) + 1):
        if top > len(money_of_series):
            continue
        rows.append({"seccion": "concentracion", "dimension": "cuota de las mayores series", "valor": f"top {top}",
                     "orden": order, "series": top, "usd_vence_futuro": float(money_of_series.head(top).sum()),
                     "cuota_usd_futuro": float(cumulative.iloc[top - 1])})

    # the money to predict by own support and by own history of its series
    card = series_table[[SERIES_ID_COLUMN, "usd_por_predecir"]].merge(
        series_rate[[SERIES_ID_COLUMN, "n_propio", "meses_historia"]], on=SERIES_ID_COLUMN, how="left")
    card = card[card["usd_por_predecir"] > 0]
    total_to_predict = float(card["usd_por_predecir"].sum())
    floor, own_floor = configuration.support_floor, configuration.own_rate_floor
    support_buckets = [("0 · sin historia propia", card["n_propio"].fillna(0) <= 0),
                       (f"1 · < {floor:.0f} al mes", (card["n_propio"] > 0) & (card["n_propio"] < floor)),
                       (f"2 · {floor:.0f}-{own_floor:.0f} al mes", (card["n_propio"] >= floor) & (card["n_propio"] < own_floor)),
                       (f"3 · ≥ {own_floor:.0f} al mes", card["n_propio"] >= own_floor)]
    for order, (name, mask) in enumerate(support_buckets, 1):
        rows.append({"seccion": "soporte", "dimension": "n_propio", "valor": name, "orden": order,
                     "series": int(mask.sum()), "usd_vence_futuro": float(card.loc[mask, "usd_por_predecir"].sum()),
                     "cuota_usd_futuro": float(card.loc[mask, "usd_por_predecir"].sum()) / total_to_predict if total_to_predict else np.nan})
    months_of_history = card["meses_historia"].fillna(0)
    for order, (low, high, name) in enumerate(HISTORY_BUCKETS, 1):
        mask = (months_of_history >= low) & (months_of_history <= high)
        rows.append({"seccion": "historia", "dimension": "meses_historia", "valor": name, "orden": order,
                     "series": int(mask.sum()), "usd_vence_futuro": float(card.loc[mask, "usd_por_predecir"].sum()),
                     "cuota_usd_futuro": float(card.loc[mask, "usd_por_predecir"].sum()) / total_to_predict if total_to_predict else np.nan})
    return pd.DataFrame(rows).reindex(columns=PROFILE_COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 1 · PREPARATION: the data the hypotheses rest on
# ═══════════════════════════════════════════════════════════════════════════════════

def checks_of_steps(configuration: Config, prefix: str) -> dict:
    """The checks of the steps of a block (configuration.check_history): how many, warnings, failures, and the
    descriptions of the warnings."""
    rows = [row for row in configuration.check_history if str(row["paso"]).startswith(prefix)]
    return {"pasos": len(rows), "comprobaciones": sum(row["comprobaciones"] for row in rows),
            "avisos": sum(row["avisos"] for row in rows), "fallos": sum(row["fallos"] for row in rows),
            "avisos_detalle": " · ".join(f"{row['paso']}: {row['avisos_detalle']}" for row in rows if row["avisos"])}


def verdict_of_preparation(tables: dict, configuration: Config) -> list:
    """B1: volume, history, concentration, support, stability of the portfolio rate, discount, states, the checks."""
    block = "B1"
    fine = tables["fine_table"]
    series_table = tables.get("series_table", tables.get("series"))
    series_rate = tables["series_rate"]
    period, usd_due = configuration.period_col, configuration.pipeline_usd_col
    units_due, units_renewed = configuration.pipeline_units_col, configuration.renewed_units_col
    current = configuration.calendar_boundaries()["current"]
    rows = []

    # the profile of the pipeline, kept for step 8.6 (sff_perfil_datos) and the report
    profile = pipeline_profile(fine, series_table, series_rate, configuration)
    configuration.data_profile = profile

    # [1] the volume: the money to predict and how the pipeline moves
    future = fine[fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION]
    next_year = future[future[period] < current + MONTHS_PER_YEAR]
    closed_year = fine[(fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)) & (fine[period] >= current - MONTHS_PER_YEAR)]
    growth = float(next_year[usd_due].sum() / closed_year[usd_due].sum() - 1) if closed_year[usd_due].sum() else np.nan
    rows.append(verdict_row(block, "1.6", KIND_FACT, "¿Cuánto dinero hay que predecir y cómo se mueve la pipeline?",
                            "USD que vence desde el mes en curso · pipeline de los próximos 12 meses frente a los 12 cerrados",
                            f"{money(float(future[usd_due].sum()))} · {growth:+.1%}", "—", VERDICT_INFORMATIVE,
                            "un cambio fuerte de volumen cambia la mezcla de la cartera: la sección 'dimension' de "
                            "sff_perfil_datos dice qué partes crecen o caen"))

    # [2] the money to predict has a history of its own
    card = series_table[[SERIES_ID_COLUMN, "usd_por_predecir"]].merge(
        series_rate[[SERIES_ID_COLUMN, "n_propio", "meses_historia"]], on=SERIES_ID_COLUMN, how="left")
    card = card[card["usd_por_predecir"] > 0]
    total = float(card["usd_por_predecir"].sum())
    with_year = float(card.loc[card["meses_historia"].fillna(0) >= FULL_HISTORY_MONTHS, "usd_por_predecir"].sum()) / total if total else np.nan
    rows.append(verdict_row(block, "1.7", KIND_HYPOTHESIS, "¿El dinero a predecir tiene historia propia?",
                            f"cuota del USD por predecir en series con ≥ {FULL_HISTORY_MONTHS} meses de historia propia",
                            percent(with_year), f"≥ {configuration.verdict_history_coverage:.0%}",
                            VERDICT_HOLDS if with_year >= configuration.verdict_history_coverage else VERDICT_FAILS,
                            "la escalera y la credibilidad cubren el resto; si la cuota es baja, el forecast depende de "
                            "series prestadas y de la celda mandatory"))

    # [3] concentration and own support
    ordered = card["usd_por_predecir"].sort_values(ascending=False)
    series_for_80 = int((ordered.cumsum() / total < 0.8).sum()) + 1 if total else 0
    own_floor_share = float(card.loc[card["n_propio"] >= configuration.own_rate_floor, "usd_por_predecir"].sum()) / total if total else np.nan
    floor_share = float(card.loc[card["n_propio"] >= configuration.support_floor, "usd_por_predecir"].sum()) / total if total else np.nan
    rows.append(verdict_row(block, "1.7", KIND_FACT, "¿Cuán concentrado está el dinero a predecir?",
                            "series que hacen el 80 % del USD por predecir, de las series con dinero",
                            f"{series_for_80:,} de {len(card):,}", "—", VERDICT_INFORMATIVE,
                            "donde el dinero se concentra en pocas series grandes, la escalera mueve poco dinero: la "
                            "precisión del total depende de esas series"))
    rows.append(verdict_row(block, "1.7", KIND_FACT, "¿Cuánto dinero tiene soporte propio?",
                            f"cuota del USD por predecir en series con ≥ {configuration.support_floor:.0f} y ≥ "
                            f"{configuration.own_rate_floor:.0f} unidades al mes", f"{percent(floor_share)} · {percent(own_floor_share)}",
                            "—", VERDICT_INFORMATIVE, "lo que queda por debajo es lo que la escalera tiene que mejorar"))

    # [4] the stability of the portfolio rate: its monthly variation against the binomial noise of its volume
    closed = fine[fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    monthly = closed.groupby(period)[[units_due, units_renewed]].sum()
    monthly = monthly[monthly.index >= current - 2 * MONTHS_PER_YEAR]
    if len(monthly) >= 3:
        rate = monthly[units_renewed] / monthly[units_due]
        noise_pp = float(100 * np.sqrt((rate * (1 - rate) / monthly[units_due]).mean()))
        spread_pp = float(100 * rate.std(ddof=1))
        rows.append(verdict_row(block, "1.7", KIND_FACT, "¿Cuánto se mueve la tasa de la cartera mes a mes?",
                                f"desviación de la tasa mensual (unidades) en los últimos {len(monthly)} meses cerrados, "
                                "frente a su ruido binomial", f"±{spread_pp:.1f} pp (de {100 * rate.min():.1f}% a {100 * rate.max():.1f}%)",
                                f"ruido ±{noise_pp:.2f} pp", VERDICT_INFORMATIVE,
                                "una variación muy por encima del ruido en el total es mezcla, tendencia o estación: lo que "
                                "la escalera y las técnicas tienen que explicar; el paso 2.5 lo mide por pool"))

    # [5] the discount and the customer states of the money to predict
    if configuration.discount_value_column and configuration.discount_bucket_column in future.columns:
        unknown = float(future.loc[future[configuration.discount_bucket_column] == UNKNOWN_DISCOUNT_BUCKET, usd_due].sum()
                        / future[usd_due].sum()) if future[usd_due].sum() else np.nan
        rows.append(verdict_row(block, "1.4", KIND_FACT, "¿Cuánto del dinero a predecir NO conoce su descuento?",
                                "cuota del USD por predecir sin descuento (tramo sin_dato)", percent(unknown), "—",
                                VERDICT_WARNING if unknown > 0.5 else VERDICT_INFORMATIVE,
                                "sin descuento la regla del contrato no se puede aplicar: el precio sale del uplift "
                                "estadístico de la celda"))
    states = [state for state in negative_states(configuration) if state in future.columns]
    if states and future[usd_due].sum():
        shares = " · ".join(f"{state} {percent(float(future.loc[future[state].astype(float) == 1, usd_due].sum() / future[usd_due].sum()))}"
                            for state in states)
        rows.append(verdict_row(block, "1.2", KIND_FACT, "¿Cuánto dinero a predecir lleva HOY un estado del cliente?",
                                "cuota del USD por predecir con cada estado negativo, en la foto del extracto", shares, "—",
                                VERDICT_INFORMATIVE, "el bloque 4 predice la cuota al vencer; la de hoy está por debajo en "
                                                     "los meses lejanos"))

    # [6] the time series universe and the checks of the block
    time_series_rows = tables.get("time_series_rows")
    if time_series_rows is not None and len(time_series_rows):
        closed_ts = time_series_rows[(time_series_rows[period] < current) & (time_series_rows[period] >= current - MONTHS_PER_YEAR)]
        rows.append(verdict_row(block, "1.2", KIND_FACT, "¿Cuánto pesa el universo time series (retail a suscripción)?",
                                "filas y revenue de los últimos 12 meses cerrados",
                                f"{len(time_series_rows):,} filas · {money(float(closed_ts[configuration.renewed_usd_col].sum()))}",
                                "—", VERDICT_INFORMATIVE, "se predice aparte (bloque 6), con su propia banda"))
    checks = checks_of_steps(configuration, "1.")
    rows.append(verdict_row(block, "1.1-1.7", KIND_HYPOTHESIS, "¿El extracto cuadra con la Config y es coherente?",
                            "comprobaciones de los pasos 1.1 a 1.7: fallos · avisos",
                            f"{checks['comprobaciones']} comprobaciones · {checks['fallos']} fallos · {checks['avisos']} avisos",
                            "0 fallos", VERDICT_HOLDS if checks["fallos"] == 0 else VERDICT_FAILS,
                            checks["avisos_detalle"] or "sin avisos"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 2 · RATE
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_rate(tables: dict, configuration: Config) -> list:
    """B2: the noise of a series alone and what the ladder recovers, the merges, the signal beyond the noise, the
    technique chosen against the default one in the exam, the band of the technique."""
    block = "B2"
    rows = []
    estimate = tables["series_estimate"]

    # [1] a series alone is mostly noise; the ladder and the credibility know the rate better
    weights = estimate["usd_por_predecir"].fillna(0)
    if weights.sum():
        alone = estimate["error_binomial_pp"]
        after = configuration.z * estimate["se_estimacion_pp"]
        within_alone = float(weights[alone <= configuration.promise_pp].sum() / weights.sum())
        within_after = float(weights[after <= configuration.promise_pp].sum() / weights.sum())
        rows.append(verdict_row(block, "2.3", KIND_HYPOTHESIS, "¿Una serie sola no basta y la escalera conoce mejor su tasa?",
                                f"USD por predecir con la tasa conocida a ±{configuration.promise_pp:.0f} pp: sola → tras "
                                "escalera y credibilidad", f"{within_alone:.0%} → {within_after:.0%}", "más que sola",
                                VERDICT_HOLDS if within_after > within_alone else VERDICT_FAILS,
                                "conocer mejor la tasa media no garantiza predecir mejor cada mes: lo dice el examen 7.1"))
    levels = tables.get("money_by_level")
    if levels is not None and len(levels):
        share = levels.set_index("nivel_riesgo")["pct_usd"]
        own = float(share[[name for name in share.index if name.startswith("A")]].sum())
        borrowed = float(share[[name for name in share.index if name.startswith("B")]].sum())
        rows.append(verdict_row(block, "2.3", KIND_FACT, "¿Cuánto dinero predice cada serie con lo suyo y cuánto con lo prestado?",
                                "cuota del USD por predecir por nivel de riesgo: propio (A) · prestado de casi iguales (B) · resto",
                                f"{own:.0%} · {borrowed:.0%} · {max(1 - own - borrowed, 0):.0%}", "—", VERDICT_INFORMATIVE,
                                "el resto (C, S, D) es lo más incierto: grupos lejanos, signo bajo el suelo o sin historia"))

    # [1b] the credibility k: estimated from the data, or the default of the Config
    borrowing = estimate[estimate["z"].notna() & (estimate["z"] < 1) & (estimate["usd_por_predecir"] > 0)]
    if len(borrowing) and borrowing["usd_por_predecir"].sum():
        default_share = float(borrowing.loc[np.isclose(borrowing["k"], configuration.k_cred), "usd_por_predecir"].sum()
                              / borrowing["usd_por_predecir"].sum())
        rows.append(verdict_row(block, "2.3", KIND_HYPOTHESIS, "¿La k de credibilidad sale de los datos?",
                                f"USD de las series que se apoyan en su referencia (z < 1) con la k por defecto (k_cred = "
                                f"{configuration.k_cred:g})", percent(default_share, 0), "< 50 %",
                                VERDICT_WARNING if default_share >= 0.5 else VERDICT_HOLDS,
                                "una k por defecto es un parámetro, no una medida: probar su sensibilidad en el examen 7.1 "
                                "(k_cred × 0,5 y × 2) antes de fiarse de la mezcla serie-referencia"))

    # [2] merging improves the error of the monthly rate of the series it merges
    merges = tables["ladder"]["merges"] if tables.get("ladder") else None
    if merges is not None and len(merges):
        improves = float(merges["improves"].mean())
        rows.append(verdict_row(block, "2.2", KIND_HYPOTHESIS, "¿Fusionar una serie con sus parientes mejora el error de su tasa mensual?",
                                "fusiones que bajan el error (sesgo² + ruido) de la serie", f"{improves:.0%} de {len(merges):,}",
                                "> 50 %", VERDICT_HOLDS if improves > 0.5 else VERDICT_FAILS,
                                "las que no mejoran son fusiones forzadas por el soporte: su sesgo pesa más que el ruido que quitan"))

    # [3] the rate moves beyond the binomial noise
    dynamics, portfolio = tables.get("pool_dynamics"), tables.get("portfolio_dynamics")
    if dynamics is not None and len(dynamics) and dynamics["usd_por_predecir"].sum():
        money_total = dynamics["usd_por_predecir"].sum()
        beyond = float(dynamics.loc[dynamics["phi"] > configuration.phi_with_engine, "usd_por_predecir"].sum() / money_total)
        trend = float(dynamics.loc[dynamics["tendencia"].ne(0), "usd_por_predecir"].sum() / money_total)
        seasonal = float(dynamics.loc[dynamics["estacional"].eq(1), "usd_por_predecir"].sum() / money_total)
        rows.append(verdict_row(block, "2.5", KIND_HYPOTHESIS, "¿La tasa de los pools se mueve más allá del ruido binomial?",
                                f"USD de pools con φ > {configuration.phi_with_engine} · con tendencia · estacionales",
                                f"{beyond:.0%} · {trend:.0%} · {seasonal:.0%}", "—",
                                VERDICT_HOLDS if beyond > 0.5 else VERDICT_FAILS,
                                "si casi todo es ruido (φ ≈ 1) la media es la mejor técnica y el backtest no debería ganar a "
                                "la técnica por defecto"
                                + (f"; la cartera: efecto mes p = {portfolio['p_valor_mes']:.2f}, tendencia "
                                   f"{portfolio['tendencia_pp_ano']:+.1f} pp/año" if portfolio else "")))

    # [4] the technique chosen in the backtest beats the default one in the exam (it never saw the exam months)
    backtest = tables.get("backtest")
    if backtest is not None and len(backtest["exam_by_pool"]):
        exam = backtest["exam_by_pool"].merge(tables["pool_reference"][["composition_id", "usd_por_predecir"]],
                                              on="composition_id", how="left")
        for band_name, band in exam.groupby("tramo_h", sort=False):
            weights = band["usd_por_predecir"].fillna(0) + 1e-9
            gain = float(np.average(band["mejora_pp"], weights=weights))
            chosen = float(np.average(band["elegida_err_pp_medio"], weights=weights))
            default = float(np.average(band["por_defecto_err_pp_medio"], weights=weights))
            changed = band[band["tecnica_origen"] != "por_defecto"]
            p_value = float(stats.ttest_1samp(changed["mejora_pp"], 0.0).pvalue) if len(changed) >= 3 else np.nan
            # a gain must be significant and above the noise; a loss only significant (a small loss that holds pool after
            # pool says the selection does not help)
            significant = np.isfinite(p_value) and p_value < configuration.verdict_significance
            if significant and gain < 0:
                verdict = VERDICT_FAILS
            elif significant and gain > configuration.verdict_technique_gain_pp:
                verdict = VERDICT_HOLDS
            else:
                verdict = VERDICT_INCONCLUSIVE
            rows.append(verdict_row(block, "2.6", KIND_HYPOTHESIS,
                                    f"¿La técnica elegida mejora a la técnica por defecto en el examen ({band_name})?",
                                    "error medio de la tasa por pool en el examen, ponderado por dinero: elegida vs por defecto",
                                    f"{chosen:.2f} pp vs {default:.2f} pp (mejora {gain:+.2f} pp)",
                                    f"mejora > {configuration.verdict_technique_gain_pp} pp con p < {configuration.verdict_significance} (p = {p_value:.2f})", verdict,
                                    "si no mejora en los datos reales, la selección de técnica es complejidad sin ganancia: "
                                    f"quedarse con {configuration.default_technique} (o la mejor del ranking) en todos los pools"))
        inside = float(exam["dentro_banda"].mean())
        rows.append(verdict_row(block, "2.6", KIND_EXAM, "¿La banda de la técnica cubre lo que promete en el examen?",
                                "errores de la técnica elegida dentro de su banda del 90 %", percent(inside, 0),
                                f"≥ {configuration.min_exam_coverage:.0%} (min_exam_coverage, el check 5 del 2.6)",
                                VERDICT_HOLDS if inside >= configuration.min_exam_coverage else VERDICT_FAILS,
                                "con una banda nominal del 90 %, un umbral del 80 % es holgado: subir min_exam_coverage a 0,85"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 3 · PRICE
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_price(tables: dict, configuration: Config) -> list:
    """B3: the contract rule against the statistical uplift, the price by states, the way of the uplift, the error of
    the price of the monthly total."""
    block = "B3"
    rows = []
    comparison, verdict = tables.get("uplift_backtest"), tables.get("uplift_verdict")
    if comparison is None or verdict is None:
        return rows
    known = comparison[comparison["filas"] == "descuento conocido"].set_index("via")
    if {"contrato", "estadistica"} <= set(known.index):
        rows.append(verdict_row(block, "3.3", KIND_HYPOTHESIS, "¿La regla del contrato 1/(1 − d) predice el precio donde se conoce el descuento?",
                                "WAPE del USD renovado en el examen: contrato vs estadística (sesgo del contrato)",
                                f"{percent(known.loc['contrato', 'wape'])} vs {percent(known.loc['estadistica', 'wape'])} "
                                f"(sesgo {known.loc['contrato', 'sesgo']:+.1%})", "la vía estadística",
                                VERDICT_HOLDS if known.loc["contrato", "wape"] < known.loc["estadistica", "wape"] else VERDICT_FAILS,
                                f"el forecast usa la vía {verdict['via_usada_con_descuento']}; un sesgo positivo del contrato "
                                "dice que los renovadores no llegan al MSRP del vencimiento"))
    check = tables.get("contract_check")
    if check is not None and len(check) and check["usd_renovado"].sum():
        realization = float(np.average(check["ratio_realizacion"], weights=check["usd_renovado"]))
        rows.append(verdict_row(block, "3.1", KIND_FACT, "¿Cuánto de la regla del contrato se realiza en el pasado?",
                                "uplift observado / uplift de la regla, ponderado por el USD renovado", percent(realization, 0),
                                "100 % = la regla es exacta", VERDICT_INFORMATIVE,
                                "por debajo de 100 %: descuentos de retención o un MSRP de referencia distinto del de la tarifa"))
    all_rows = comparison[comparison["filas"] == "todas"].set_index("via")
    if {"estadistica", "estadistica_estado"} <= set(all_rows.index):
        rows.append(verdict_row(block, "3.3", KIND_HYPOTHESIS, "¿El precio por estado del cliente mejora el precio?",
                                "WAPE de toda fila renovadora en el examen: con el factor de su estado vs sin él",
                                f"{percent(all_rows.loc['estadistica_estado', 'wape'], 2)} vs {percent(all_rows.loc['estadistica', 'wape'], 2)}",
                                "menor con el factor", VERDICT_HOLDS if verdict.get("factor_estado") else VERDICT_FAILS,
                                "se usa solo si baja el error" + (" (se usa)" if verdict.get("factor_estado") else " (no se usa)")))
    ways = verdict.get("formas_uplift")
    if ways is not None and len(ways):
        everything = ways[ways["filas"] == "todas"].set_index("forma")
        used = ways.loc[ways["usada"] == "←", "forma"]
        if len(everything) and len(used):
            used_way, best_way = used.iloc[0], everything["wape"].idxmin()
            used_error, best_error = float(everything.loc[used_way, "wape"]), float(everything.loc[best_way, "wape"])
            rows.append(verdict_row(block, "3.3", KIND_HYPOTHESIS, "¿La forma del uplift estadístico de la Config es la que mejor predice?",
                                    "WAPE de toda fila renovadora en el examen, cada forma estimada antes del examen",
                                    f"{used_way} {percent(used_error, 2)}", f"mejor: {best_way} {percent(best_error, 2)}",
                                    VERDICT_HOLDS if used_error <= best_error + 0.002 else VERDICT_WARNING,
                                    "la forma la fija la Config (uplift_window_months, uplift_without_increases); si otra yerra "
                                    "menos en los datos reales, cambiarla o elegirla por examen como la vía"))
    monthly = verdict.get("error_precio_por_mes")
    if monthly is not None and len(monthly):
        bias = float(monthly["error_precio"].mean())
        rows.append(verdict_row(block, "3.3", KIND_EXAM, "¿Cuánto falla el precio del total de un mes?",
                                "error medio |USD predicho / real − 1| del total de cada mes de examen, con las vías que usa el forecast",
                                f"{percent(verdict.get('error_precio_total', np.nan))} (sesgo {bias:+.1%})", "—", VERDICT_INFORMATIVE,
                                "es la parte del precio de la banda oficial (7.2)"))
    increases = tables.get("price_increases")
    if increases is not None and len(increases):
        active = increases[increases["meses_futuros_con_efecto"] > 0]
        rows.append(verdict_row(block, "3.1", KIND_FACT, "¿Hay subidas de precio en curso que el forecast tenga que llevar?",
                                "subidas detectadas · con meses futuros dentro de su ciclo",
                                f"{len(increases)} · {len(active)}", "—", VERDICT_INFORMATIVE,
                                "contrastar con el calendario real de tarifas (sff_price_increase_calendar)"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 4 · CUSTOMER STATES
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_states(tables: dict, configuration: Config) -> list:
    """B4: the curve of every dated state against the states seen with no curve; the states with no date."""
    block = "B4"
    rows = []
    box = tables.get("state_maturation")
    if box is None:
        return rows
    if getattr(box, "identity", False):
        rows.append(verdict_row(block, "4.1", KIND_WARNING, "¿Se predicen los estados al vencer?", "distribute_marks",
                                "no (caja identidad)", "—", VERDICT_WARNING,
                                "el forecast usa los estados de hoy: los meses lejanos quedan optimistas"))
        return rows
    for state, exam in getattr(box, "curve_exams", {}).items():
        worst = float(exam["error_curva_abs_pp"].max())
        seen = float(exam["error_vista_pp"].abs().max())
        rows.append(verdict_row(block, "4.1", KIND_HYPOTHESIS, f"¿La curva de maduración predice la cuota final de {state}?",
                                "error de la cuota del estado en el examen (pp, el peor horizonte): con la curva vs con el estado visto entonces",
                                f"{worst:.2f} pp vs {seen:.2f} pp", "menor con la curva",
                                VERDICT_HOLDS if worst < seen else VERDICT_FAILS,
                                ("NO se usa: toma su cuota histórica (error mayor que con el estado visto, o por encima de "
                                 f"{configuration.maturation_curve_max_error_pp:g} pp); "
                                 if state in configuration.rejected_maturation_dims else "")
                                + "el efecto en dinero lo dice el examen honesto del 7.1 (veredicto del bloque 7)"))
    dated = set(getattr(box, "curve_exams", {}))
    for state in box.states:
        if state not in dated:
            rows.append(verdict_row(block, "4.1", KIND_WARNING, f"¿Se puede examinar la maduración de {state}?",
                                    "columnas de maduración del estado (<estado>_month_k)", "no las trae", "—", VERDICT_WARNING,
                                    "su cuota al vencer sale de la proporción histórica, sin examen: pedir al extracto la "
                                    "fecha del estado"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 5 · ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_future_pipeline(tables: dict, configuration: Config) -> list:
    """B5: the simulated acquisition of every value against "the same as a year ago", and against what was sold."""
    block = "B5"
    rows = []
    future_pipeline = tables.get("future_pipeline")
    if future_pipeline is None:
        return rows
    summary = future_pipeline["exam"]["summary"]
    acquisition = configuration.acquisition_column
    if len(summary) and acquisition in summary.columns:
        for value, exam in summary.groupby(acquisition, sort=False):
            # every (origin, month) examined counts once: the horizons weighted by their months, as step 5.1 says it
            weights = exam["meses"].astype(float) if "meses" in exam else None
            simulated = float(np.average(exam["error_usd"], weights=weights))
            baseline = float(np.average(exam["error_base_usd"], weights=weights))
            bias = float(np.average(exam["sesgo_unidades"], weights=weights)) if "sesgo_unidades" in exam else np.nan
            rows.append(verdict_row(block, "5.1", KIND_HYPOTHESIS,
                                    f"¿Simular la captación {value} bate a 'lo mismo que hace un año'?",
                                    "error medio del USD vendido en el examen (6 orígenes): simulación (nivel) vs nivel 1",
                                    f"{percent(simulated)} vs {percent(baseline)} (sesgo {bias:+.1%})", "menor la simulación",
                                    VERDICT_HOLDS if simulated < baseline else VERDICT_FAILS,
                                    "si el nivel de los últimos meses empeora la simulación, ese valor debería simularse con "
                                    "nivel 1 (lo mismo que hace un año): elegir el nivel por examen, como el bloque 6"))
    comparison = future_pipeline.get("comparison")
    if comparison is not None and len(comparison) and {"usd_simulado", "usd_ano_anterior"} <= set(comparison.columns):
        ratio = float(comparison["usd_simulado"].sum() / comparison["usd_ano_anterior"].sum()) if comparison["usd_ano_anterior"].sum() else np.nan
        rows.append(verdict_row(block, "5.1", KIND_FACT, "¿Cuánta captación se simula frente al año anterior?",
                                "USD simulado de la ventana / USD vendido los mismos meses un año antes",
                                f"{ratio:.2f}× ({money(float(comparison['usd_simulado'].sum()))})", "0,5× a 2×",
                                VERDICT_INFORMATIVE, ""))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 6 · TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_time_series(tables: dict, configuration: Config) -> list:
    """B6: the level rule chosen by the exam and how much the simulation misses."""
    block = "B6"
    rows = []
    universe = tables.get("time_series_universe")
    if universe is None or not len(universe.table):
        return rows
    by_rule = universe.exam.get("by_rule")
    if by_rule is not None and len(by_rule):
        means = by_rule.groupby("candidato", sort=False)["error_medio_regla"].first().sort_values()
        chosen = universe.exam["candidate"]
        others = " · ".join(f"{rule} {percent(error, 0)}" for rule, error in means.head(4).items() if rule != chosen)
        error = float(means.get(chosen, np.nan))
        rows.append(verdict_row(block, "6.1", KIND_EXAM, "¿Qué regla del nivel y qué ventana del valor por unidad simulan mejor el "
                                "universo time series, y cuánto falla?",
                                "error medio del revenue total de un mes en el examen (6 orígenes)", f"{chosen} {percent(error, 0)}",
                                others or "—", VERDICT_HOLDS if error <= configuration.verdict_time_series_error else VERDICT_WARNING,
                                "un error alto hace ancha su banda: el universo pesa poco en dinero y mucho en la banda "
                                "(veredicto del bloque 7)"))
    table = universe.table
    next_year = table[(table["origen"] != "ts_real") & (table[configuration.period_col].map(lambda month: month.year) == universe.next_year)]
    rows.append(verdict_row(block, "6.1", KIND_FACT, "¿Cuánto añade el universo time series al año siguiente?",
                            "revenue simulado + renovación esperada de su reentrada", money(float(next_year["revenue"].sum())),
                            "—", VERDICT_INFORMATIVE, ""))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 7 · EXAM AND ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_exam_and_assembly(tables: dict, configuration: Config) -> list:
    """B7: the framework against the baseline, against the ladder with the default technique and against the series
    alone, with the paired tests; the bias; the states predicted; the maturation; what the band is made of and what it
    covers."""
    block = "B7"
    rows = []
    exam = tables.get("series_exam")
    if exam is not None and len(exam["by_month"]):
        by_month, summary = exam["by_month"], exam["summary"]
        for horizon, months in by_month.groupby("h"):
            framework = months[f"{METHOD_FRAMEWORK}_error_total"].to_numpy()
            for rival, question, consequence in (
                    (METHOD_BASELINE, "¿El framework bate a la baseline en las unidades renovadas del mes?",
                     "la baseline es su celda mandatory con los últimos meses: lo que cualquiera calcularía"),
                    ("por_defecto", "¿La técnica elegida aporta sobre la escalera con la técnica por defecto?",
                     "si no aporta, la selección de técnica sobra (ver el veredicto del bloque 2)"),
                    ("raw", "¿El framework bate a la serie sola en las unidades renovadas del mes?",
                     "la serie sola con su propia historia, sin escalera")):
                column = f"{rival}_error_total"
                if column not in months.columns:
                    continue
                comparison = paired_comparison(framework, months[column].to_numpy())
                rows.append(verdict_row(block, "7.1", KIND_HYPOTHESIS, f"{question} (h = {int(horizon)})",
                                        "error medio |total predicho / real − 1| en los meses de examen; meses ganados; p del "
                                        "test de signos y del t pareado",
                                        f"{percent(np.abs(framework).mean())} vs {percent(np.abs(months[column]).mean())} · gana "
                                        f"{comparison['gana']} de {comparison['meses']}",
                                        f"p signos {comparison['p_signos']:.2f} · p t {comparison['p_t']:.2f}",
                                        gain_verdict(comparison, configuration.verdict_significance), consequence))
            # the same against the baseline in money: the business reads USD (units × the price of the real renewals)
            if {f"{METHOD_FRAMEWORK}_error_usd", f"{METHOD_BASELINE}_error_usd", "renovado_usd_real"} <= set(months.columns):
                real_usd = months["renovado_usd_real"].replace(0, np.nan)
                framework_usd = (months[f"{METHOD_FRAMEWORK}_error_usd"] / real_usd).to_numpy()
                baseline_usd = (months[f"{METHOD_BASELINE}_error_usd"] / real_usd).to_numpy()
                comparison = paired_comparison(framework_usd, baseline_usd)
                gain = float(np.abs(months[f"{METHOD_BASELINE}_error_usd"]).mean() - np.abs(months[f"{METHOD_FRAMEWORK}_error_usd"]).mean())
                rows.append(verdict_row(block, "7.1", KIND_HYPOTHESIS,
                                        f"¿El framework bate a la baseline en el USD renovado del mes? (h = {int(horizon)})",
                                        "error medio |USD predicho / real − 1| en los meses de examen; meses ganados; p del test de "
                                        "signos y del t pareado; ganancia media en USD al mes",
                                        f"{percent(np.nanmean(np.abs(framework_usd)))} vs {percent(np.nanmean(np.abs(baseline_usd)))} · "
                                        f"gana {comparison['gana']} de {comparison['meses']} · {money(gain)} al mes",
                                        f"p signos {comparison['p_signos']:.2f} · p t {comparison['p_t']:.2f}",
                                        gain_verdict(comparison, configuration.verdict_significance),
                                        "en unidades y en USD pueden no coincidir: el precio de cada serie pondera su error"))
            row = summary[(summary["metodo"] == METHOD_FRAMEWORK) & (summary["h"] == horizon)]
            alone = summary[(summary["metodo"] == "raw") & (summary["h"] == horizon)]
            if len(row) and len(alone):
                rows.append(verdict_row(block, "7.1", KIND_HYPOTHESIS, f"¿Serie a serie, el framework yerra menos que la serie sola? (h = {int(horizon)})",
                                        "WAPE de las unidades renovadas, serie a serie (sin compensaciones)",
                                        f"{percent(row['wape_series'].iloc[0])} vs {percent(alone['wape_series'].iloc[0])}",
                                        "menor el framework",
                                        VERDICT_HOLDS if row["wape_series"].iloc[0] < alone["wape_series"].iloc[0] else VERDICT_FAILS,
                                        "si la serie sola gana serie a serie, la escalera solo compensa en el total: revisar el "
                                        "suelo de soporte y la credibilidad"))
                error, bias = float(row["error_total_medio"].iloc[0]), float(row["sesgo_total_medio"].iloc[0])
                rows.append(verdict_row(block, "7.1", KIND_WARNING if abs(bias) > configuration.verdict_bias_share * error else KIND_EXAM,
                                        f"¿El error del total es sesgo (todos los meses en la misma dirección)? (h = {int(horizon)})",
                                        "sesgo medio frente al error medio del total", f"sesgo {bias:+.1%} de un error {percent(error)}",
                                        f"sesgo < {configuration.verdict_bias_share:.0%} del error",
                                        VERDICT_WARNING if abs(bias) > configuration.verdict_bias_share * error else VERDICT_HOLDS,
                                        "un error que es sesgo no se arregla con más técnicas: es un cambio de nivel que la "
                                        "historia no anticipa (mirar la tasa de los meses de examen frente al entrenamiento)"))
            honest = f"{'framework_honesto'}_error_total"
            today = f"{'framework_estados_hoy'}_error_total"
            if honest in months.columns and today in months.columns:
                comparison = paired_comparison(months[honest].to_numpy(), months[today].to_numpy())
                rows.append(verdict_row(block, "7.1", KIND_HYPOTHESIS, f"¿Predecir los estados del cliente mejora el total? (h = {int(horizon)})",
                                        "error del total con los estados predichos (honesto) vs con los estados vistos entonces",
                                        f"{percent(np.abs(months[honest]).mean())} vs {percent(np.abs(months[today]).mean())} · "
                                        f"gana {comparison['gana']} de {comparison['meses']}",
                                        f"p signos {comparison['p_signos']:.2f}", gain_verdict(comparison, configuration.verdict_significance),
                                        "si los estados vistos aciertan más, puede ser que el sesgo de la tasa compense el de los "
                                        "estados: comparar los sesgos de las dos filas en sff_examen_cartera"))
        # did the level of the rate change in the exam months? (a shift the history could not anticipate is bias)
        fine = tables.get("fine_table")
        if fine is not None:
            units_due, units_renewed = configuration.pipeline_units_col, configuration.renewed_units_col
            boundaries = configuration.calendar_boundaries()
            closed = fine[fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
            exam_months = closed[closed[configuration.period_col] >= boundaries["test_start"]]
            training_year = closed[(closed[configuration.period_col] < boundaries["test_start"])
                                   & (closed[configuration.period_col] >= boundaries["test_start"] - MONTHS_PER_YEAR)]
            if exam_months[units_due].sum() and training_year[units_due].sum():
                exam_rate = float(exam_months[units_renewed].sum() / exam_months[units_due].sum())
                training_rate = float(training_year[units_renewed].sum() / training_year[units_due].sum())
                noise = float(np.sqrt(exam_rate * (1 - exam_rate) / exam_months[units_due].sum()) * 100)
                shift = 100 * (exam_rate - training_rate)
                rows.append(verdict_row(block, "7.1", KIND_FACT, "¿Cambió el nivel de la tasa en los meses de examen?",
                                        "tasa (unidades) de los meses de examen frente a los 12 meses anteriores",
                                        f"{100 * exam_rate:.1f}% vs {100 * training_rate:.1f}% ({shift:+.1f} pp)",
                                        f"ruido ±{noise:.2f} pp", VERDICT_WARNING if abs(shift) > 3 * noise else VERDICT_INFORMATIVE,
                                        "un cambio de nivel en el examen se convierte en sesgo de todo método que aprende del "
                                        "pasado: explica el sesgo del total"))

        # what the official band covers: ±1 mean error at each horizon
        method = "framework_honesto" if "framework_honesto_error_total" in by_month.columns else METHOD_FRAMEWORK
        mean_error = by_month.groupby("h")[f"{method}_error_total"].apply(lambda errors: errors.abs().mean())
        inside = float((by_month[f"{method}_error_total"].abs() <= by_month["h"].map(mean_error)).mean())
        rows.append(verdict_row(block, "7.2", KIND_FACT, "¿Qué cubre la banda oficial?",
                                "meses de examen cuyo error del total cae dentro de ±1 error medio de su horizonte",
                                percent(inside, 0), "no es un intervalo del 90 %", VERDICT_INFORMATIVE,
                                "la banda es el error típico (≈ 1 error medio): para un rango del 90 % hay que multiplicarla "
                                "aproximadamente por 2"))

    forecast = tables.get("forecast")
    if forecast is not None:
        by_year = forecast["by_year"]
        maturation = forecast.get("maturation")
        if maturation is not None and len(maturation):
            year_of = maturation[configuration.period_col].map(lambda month: month.year)
            effect = maturation.groupby(year_of)[["renovado_usd_base", "ajuste_usd"]].sum()
            text = " · ".join(f"{year} {money(row['ajuste_usd'])} ({row['ajuste_usd'] / row['renovado_usd_base']:+.1%})"
                              for year, row in effect.iterrows() if row["renovado_usd_base"])
            rows.append(verdict_row(block, "7.2", KIND_FACT, "¿Cuánto cambia el forecast la maduración de los estados?",
                                    "lo esperado con los estados al vencer frente a con los de hoy, por año", text, "—",
                                    VERDICT_INFORMATIVE, "es el mayor ajuste de modelo del forecast: validarlo con datos reales "
                                                         "es prioritario"))
        parts = ["banda_tasa_y_precio_usd", "banda_captacion_usd", "banda_time_series_usd"]
        if set(parts) <= set(by_year.columns):
            for _, year in by_year[by_year["esperado_usd"] > 0].iterrows():
                band = float(year["error_examen_usd"])
                if band <= 0:
                    continue
                shares = {name: float(year[name]) / band for name in parts}
                time_series_money = float(year.get("ingresos_ts_usd", 0.0)) + float(year.get("esperado_ts_reentrada_usd", 0.0))
                future_money = float(year["esperado_usd"]) + time_series_money
                ts_money_share = time_series_money / future_money if future_money else 0.0
                flagged = ts_money_share > 0 and shares["banda_time_series_usd"] > configuration.verdict_band_share_over_money * ts_money_share
                rows.append(verdict_row(block, "7.2", KIND_WARNING if flagged else KIND_FACT,
                                        f"¿De qué está hecha la banda de {int(year['ano'])}?",
                                        "partes de la banda oficial: tasa y precio · captación simulada · time series",
                                        f"±{money(band)}: {shares['banda_tasa_y_precio_usd']:.0%} · {shares['banda_captacion_usd']:.0%} · "
                                        f"{shares['banda_time_series_usd']:.0%}",
                                        f"time series = {ts_money_share:.0%} del dinero que falta",
                                        VERDICT_WARNING if flagged else VERDICT_INFORMATIVE,
                                        "una parte que pesa en la banda mucho más que en el dinero es la que más vale mejorar"
                                        if flagged else ""))
        rows_of_forecast = forecast["forecast"]
        if exam is not None and "confidence" in rows_of_forecast.columns:
            per_series = exam["per_series"][[SERIES_ID_COLUMN, "framework_wape"]]
            labelled = (rows_of_forecast.groupby([SERIES_ID_COLUMN, "confidence"])["esperado_usd"].sum().reset_index()
                        .merge(per_series, on=SERIES_ID_COLUMN, how="inner").dropna(subset=["framework_wape"]))
            labelled = labelled[labelled["esperado_usd"] > 0]
            if len(labelled):
                wape_of = labelled.groupby("confidence").apply(
                    lambda block_rows: float(np.average(block_rows["framework_wape"], weights=block_rows["esperado_usd"])),
                    include_groups=False)
                ordered = [wape_of.get(name, np.nan) for name in ("high", "medium", "low")]
                calibrated = all(np.isfinite(value) for value in ordered) and ordered[0] < ordered[1] < ordered[2]
                rows.append(verdict_row(block, "7.2", KIND_HYPOTHESIS, "¿La confianza de cada fila dice cuánto acertó su serie en el examen?",
                                        "WAPE serie a serie en el examen 7.1, ponderado por el USD esperado, por confianza",
                                        " · ".join(f"{name} {percent(value)}" for name, value in wape_of.items()),
                                        "alta < media < baja", VERDICT_HOLDS if calibrated else VERDICT_FAILS,
                                        "si no está ordenada, la etiqueta no sirve para priorizar: recalibrar sus umbrales "
                                        "(confidence_*_pp)"))
        if "confidence" in rows_of_forecast.columns and rows_of_forecast["esperado_usd"].sum():
            confidence = rows_of_forecast.groupby("confidence")["esperado_usd"].sum() / rows_of_forecast["esperado_usd"].sum()
            rows.append(verdict_row(block, "7.2", KIND_FACT, "¿Con qué confianza se predice el dinero esperado?",
                                    "cuota del USD esperado por confianza de la fila (alta · media · baja)",
                                    " · ".join(f"{name} {share:.0%}" for name, share in confidence.items()), "—",
                                    VERDICT_INFORMATIVE, "baja = banda de tasa ancha o examen con error alto"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 8 · OUTPUT AND CONTROL
# ═══════════════════════════════════════════════════════════════════════════════════

def verdict_of_output(tables: dict, configuration: Config) -> list:
    """B8: the chain reconciles, the relations of the Power BI model hold."""
    block = "B8"
    rows = []
    validation = tables.get("validation")
    if validation is not None and len(validation):
        not_ok = validation[validation["estado"] != "ok"]
        rows.append(verdict_row(block, "8.5", KIND_HYPOTHESIS, "¿La cadena cuadra del extracto al forecast?",
                                "comprobaciones de la validación final", f"{len(validation) - len(not_ok)} de {len(validation)} ok",
                                "todas ok", VERDICT_HOLDS if not_ok.empty else VERDICT_WARNING,
                                " · ".join(not_ok["comprobacion"]) if len(not_ok) else ""))
    power_bi = tables.get("power_bi")
    relations = power_bi.get("relations") if isinstance(power_bi, dict) else None
    if relations is not None and len(relations):
        broken = relations[(relations["existe"] == 0) | (relations["clave_unica"] == 0)]
        orphans = relations[relations["filas_n_sin_clave"] > 0]
        rows.append(verdict_row(block, "8.4", KIND_HYPOTHESIS, "¿El modelo de Power BI puede hacer cada relación?",
                                "relaciones con sus dos tablas y una clave única en el lado 1 · relaciones con filas huérfanas",
                                f"{len(relations) - len(broken)} de {len(relations)} · {len(orphans)} con huérfanas",
                                "todas", VERDICT_HOLDS if broken.empty else VERDICT_FAILS,
                                " · ".join(f"{row['tabla_n']}[{row['columna_n']}]: {row['filas_n_sin_clave']:,}"
                                           for _, row in orphans.iterrows())
                                + (" (esperadas: series sin composición; filas de maduración y celdas nunca vistas, que "
                                   "explica s3_1_origen_uplift)" if len(orphans) else "")))
    rows.append(verdict_row(block, "8.1-8.6", KIND_FACT, "¿Qué tablas deja la ejecución para auditar?",
                            "tablas sff_* escritas y releídas en esta ejecución", f"{len(configuration.written_tables)}", "—",
                            VERDICT_INFORMATIVE, "sff_relaciones_modelo dice cómo se relacionan en Power BI"))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# THE REGISTRY: compute, keep, show, tabulate
# ═══════════════════════════════════════════════════════════════════════════════════

VERDICT_OF_BLOCK = {"B1": verdict_of_preparation, "B2": verdict_of_rate, "B3": verdict_of_price, "B4": verdict_of_states,
                    "B5": verdict_of_future_pipeline, "B6": verdict_of_time_series, "B7": verdict_of_exam_and_assembly,
                    "B8": verdict_of_output}
BLOCK_TITLES = {"B1": "PREPARACIÓN: LOS DATOS", "B2": "TASA", "B3": "PRECIO", "B4": "ESTADOS DEL CLIENTE",
                "B5": "ADQUISICIÓN", "B6": "TIME SERIES", "B7": "EXAMEN Y MONTAJE", "B8": "SALIDA Y CONTROL"}


def verdict_of_block(block: str, tables: dict, configuration: Config) -> pd.DataFrame:
    """The verdict rows of one block from the results of the run so far (an empty table for an unknown block)."""
    function = VERDICT_OF_BLOCK.get(block)
    rows = function(tables, configuration) if function else []
    return pd.DataFrame(rows, columns=VERDICT_COLUMNS)


def record_the_verdict(block: str, tables: dict, configuration: Config) -> pd.DataFrame:
    """The verdict of a block, kept in configuration.block_verdicts and on screen. It never stops the run: a verdict
    that cannot be computed is logged and left out (the forecast does not depend on it)."""
    try:
        verdict = verdict_of_block(block, tables, configuration)
    except Exception as error:                     # the verdict reads; a failure here must not lose the run
        configuration.logger.warning(f"[{block}] the verdict of the block could not be computed: {type(error).__name__}: {error}")
        return pd.DataFrame(columns=VERDICT_COLUMNS)
    configuration.block_verdicts[block] = verdict
    show_the_verdict(block, verdict, configuration)
    return verdict


def show_the_verdict(block: str, verdict: pd.DataFrame, configuration: Config) -> None:
    """On screen: the verdict of the block as a table, the profile of the data after block 1, and one line for every
    question that does not hold."""
    if verdict.empty:
        return
    doc = configuration.logger.doc
    doc(f"[{block}] ══════════ VEREDICTO DEL BLOQUE {block[1:]} · {BLOCK_TITLES.get(block, '')} ══════════")
    if block == "B1" and configuration.data_profile is not None:
        show_the_profile(configuration.data_profile, configuration, block)
    configuration.show_table(verdict[["tipo", "paso", "pregunta", "valor", "referencia", "veredicto"]])
    for _, row in verdict[verdict["veredicto"].isin([VERDICT_FAILS, VERDICT_WARNING, VERDICT_INCONCLUSIVE])].iterrows():
        doc(f"[{block}] {row['veredicto'].upper()} · {row['pregunta']} {row['valor']} → {row['detalle']}")
    doc(f"[{block}] ══════════ FIN DEL VEREDICTO DEL BLOQUE {block[1:]} ══════════")


def show_the_profile(profile: pd.DataFrame, configuration: Config, block: str) -> None:
    """On screen: the portfolio, the concentration, the support and the history of the money to predict, and one line
    per dimension (its values, the largest one and the spread of the rate between its values)."""
    doc = configuration.logger.doc
    doc(f"[{block}] el perfil de la pipeline (sff_perfil_datos; tasa = últimos 12 meses cerrados; futuro = desde el mes en curso):")
    portfolio = profile[profile["seccion"] == "cartera"]
    configuration.show_table(portfolio[["series", "usd_vence", "usd_renovado", "tasa_unidades", "tasa_usd",
                                        "crecimiento_anual", "usd_vence_futuro"]].round(3))
    for section in ("concentracion", "soporte", "historia"):
        rows = profile[profile["seccion"] == section]
        if len(rows):
            configuration.show_table(rows[["seccion", "dimension", "valor", "series", "cuota_usd_futuro"]].round(3))
    dimensions = profile[profile["seccion"] == "dimension"]
    lines = []
    for dimension, rows in dimensions.groupby("dimension", sort=False):
        largest = rows.sort_values("cuota_usd_futuro", ascending=False).iloc[0]
        weighty = rows[rows["cuota_usd_futuro"].fillna(0) >= 0.01]
        spread = (100 * (weighty["tasa_unidades"].max() - weighty["tasa_unidades"].min())) if len(weighty) > 1 else np.nan
        lines.append({"dimension": dimension, "valores": len(rows), "mayor": largest["valor"],
                      "cuota_mayor": round(float(largest["cuota_usd_futuro"]), 3) if pd.notna(largest["cuota_usd_futuro"]) else np.nan,
                      "rango_tasa_pp": round(float(spread), 1) if np.isfinite(spread) else np.nan,
                      "crecimiento_mayor": round(float(largest["crecimiento_anual"]), 3) if pd.notna(largest["crecimiento_anual"]) else np.nan})
    if lines:
        doc(f"[{block}] cada dimensión: sus valores, el de más dinero por predecir, y cuánto separa la tasa entre sus valores "
            f"con ≥ 1 % del dinero (rango_tasa_pp):")
        configuration.show_table(pd.DataFrame(lines))


def verdict_table(configuration: Config) -> pd.DataFrame:
    """Every block's verdict, in the order of the blocks (sff_veredicto_bloques)."""
    frames = [configuration.block_verdicts[block] for block in BLOCK_ORDER if block in configuration.block_verdicts]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=VERDICT_COLUMNS)
