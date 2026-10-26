"""
block_7_exam_and_assembly.py — BLOCK 7 · EXAM AND ASSEMBLY: one forecast in money, with a measured band.

MISSION: put together, row by row, everything blocks 2 to 6 predict, into one forecast in money by month and by year,
with an official band that is an error MEASURED on closed months, and a total that is the same answer whichever way it
is added.

MECHANISM: first the exam: every forecast series is predicted in the exam months with only what was known h months
before, under the conditions of the forecast (the same box of the rate, the customer states predicted too), against
the baseline anyone could compute; its error of the monthly total by horizon is the official band. Then the assembly:
every row of pipeline (the extract's and block 5's) asks block 2 its rate, block 3 its uplift and block 4 its states
at due date, and is valued; the rows of block 6 are added with their own band. Then the total by year and origin,
checked against the forecast.

THE STEPS:
  7.1  exam of every forecast series   baseline, alone, ladder with the default technique and the framework, on the same
                                       rows; with the states predicted (the honest exam): the band of the total
  7.2  forecast                        the assembly: rate × uplift × states of every row, the totals by month and year,
                                       the time series universe added, the official band
  7.3  total                           the total by year and origin, checked equal to the forecast's

READS: blocks 1 to 6.
GIVES: series_exam, forecast (rows, by month, by year, maturation), time_series, forecast_total.

THE INTERFACE:
  block = ExamAndAssemblyBlock(configuration).run(tables)     tables: {name: result} of blocks 1 to 6
  block.forecast, block.series_exam, …    block.results()    block.seconds
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time

import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, DUE_UNITS_AT_DUE_COLUMN,
                    DUE_USD_AT_DUE_COLUMN, GATE_LEVEL, MATURATION_ROW_COLUMN, METHOD_BASELINE, METHOD_DEFAULT,
                    METHOD_FRAMEWORK, PATH_CONTRACT, PATH_CONTRACT_NEUTRAL, PATH_STATISTICAL, PIPELINE_ORIGIN_COLUMN,
                    PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, RATE_COLUMN, RATE_FROM_POOL,
                    ROLE_PROJECTION, ROLE_TEST, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_BUSINESS_SUMMARY,
                    TABLE_FORECAST_MATURATION, TABLE_FORECAST_MONTH, TABLE_FORECAST_TOTAL, TABLE_PORTFOLIO_EXAM,
                    TABLE_PORTFOLIO_EXAM_SUMMARY, TABLE_SERIES_EXAM, TABLE_SERIES_EXAM_DETAIL, TOTAL_ORIGIN_EXPECTED,
                    TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_SIMULATED, TOTAL_ORIGIN_TOTAL,
                    TRUTH_ROLES, TS_REENTRY, UNIT_ID_COLUMN, UNITS_AT_DUE_COLUMN, UNITS_TODAY_COLUMN,
                    UPLIFT_CELL_ID_COLUMN, UPLIFT_ORIGIN_CELL, UPLIFT_ORIGIN_CONTRACT, UPLIFT_ORIGIN_GLOBAL,
                    UPLIFT_ORIGIN_MANDATORY, WITH_DATES_COLUMN, has_no_negative_state, join_columns, state_combination)
from block_2_rate import CHOSEN_MODE, DEFAULT_MODE, RateBox, band_of_horizon
from block_3_price import contract_rule, state_factor_of_rows
from block_4_customer_states import (StateMaturation, honest_total_exam, maturation_by_month, show_maturation_by_month,
                                     show_states_above_final, states_above_final)
from block_5_future_pipeline import FuturePipeline
from block_6_time_series import TimeSeriesUniverse
from prediction import index_history, months_known, predict_composition
from price_increases import increase_factor, price_group_of
from report_queries import exam_by_month_query, exam_summary_query, forecast_total_query
from techniques import CATALOGUE, logit


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 7.1 · EXAM OF EVERY FORECAST SERIES  (until 25-oct-2026: step_19_series_exam.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The exam of every forecast series: how much better the framework predicts than
# the BASELINE (the calculation anyone could make) and than the forecast series alone (raw).
#
# In the exam months, with only what was known h months before (origin T − h), every forecast series with
# something due is predicted by four methods, on the same rows. They are the steps of a cascade: each one adds
# one piece of the framework to the one before.
#
#   baseline     THE reference: the rate of its mandatory cell over the last baseline_months closed months known
#                at the origin (the portfolio's if the cell has none), × what falls due. No ladder, no credibility,
#                no technique, no states. Every gain of the framework is measured against it, in USD.
#   raw          the forecast series alone, with its own history (no ladder): the technique of its composition on
#                its own monthly rates (the default technique if it cannot; its own level if it has fewer than 3
#                months; its mandatory cell if it has none). Not a step of the cascade: what a series says alone
#   por_defecto  the ladder and the credibility, with the DEFAULT technique for every composition: what the
#                support of the relatives is worth, before choosing techniques
#   framework    THE BOX OF THE RATE (step 2.7), the same the forecast uses, learned at the origin: the
#                technique CHOSEN for its composition and band, moved toward its credibility reference by (1 − z) × the
#                difference of levels known at the origin; a series with no composition: its cell with its customer
#                states
# With maturation_dims, the honest exam of the states adds the last step for the total (honesto).
#
# MONEY: every prediction is also in USD, predicted units × the price its real renewals had (renewed USD /
# renewed units of the series and month; without renewals, the AUV due × the uplift of the exam month). The price
# is the same for every method, so the USD differences are the differences of the rate: what predicting the rate
# better is worth in money. The gain of a method = the baseline's mean |error| of the monthly total in USD minus
# its own; and its consistency = the share of exam months × horizons in which it errs less than the baseline.
#
# WHAT IS NOT OUT OF SAMPLE: the structure (the ladder, the credibility k, the gates, the technique chosen per band
# and the error bands) is estimated with every closed month, the exam months included; for h ≥ 6 the choice of the
# technique has seen months after the origin. The framework's exam is therefore somewhat optimistic; the baseline
# and the series alone have no such advantage.
#
# Every prediction also carries the BINOMIAL NOISE of the real rate it tries to hit, √(p(1−p)/units due)
# with p its predicted rate: the error that even a perfect prediction would make (the noise rule: an error
# of the size of the noise is the limit, not a failure). error_over_noise = RMS error / RMS noise: about 1,
# the prediction is at the limit; well above 1, something knowable is missing; well below 1, suspicious.
#
# Every prediction keeps its steps (composition rate, shift, series rate) and, for raw and framework,
# its INTERVAL, built as the forecast builds its band (prediction.py): the error quantiles of the
# technique × the binomial error with the series' own units due. A prediction is IN THE INTERVAL when
# the real rate falls inside it. Predicted units = rate × the series' own units due.
#
# Outputs:
#   sff_series_exam_detail      forecast series × exam month × h × method: every prediction, traced
#   sff_series_exam             forecast series: per method, predictions, in the interval, units
#                               predicted and real, error, coverage (the improvement over the raw)
#   sff_examen_cartera(_resumen) the portfolio: the error of the total per exam month and method
#
# Actions (logged as they are done):
#   1. the exam: the calendar, the forecast series with something due in the exam months
#   2. the four methods for every series, month and horizon (with the interval of raw, por_defecto and framework)
#   3. the errors: per forecast series, and of the total per month
#   4. check the exam                                               checks 1-4
#   5. write the exam                                               checks 5-8
#   6. count the checks; stop if any failed
#   7. show the methods against each other, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. no prediction used a month after its origin
#    2. every method predicted every series of every exam month
#    3. the real renewals of the exam equal the renewals of the exam months (nothing counted twice)
#    4. the summary per series adds up to the detail (counts and units)
#    5-8. tables sff_series_exam_detail, sff_series_exam, sff_examen_cartera, sff_examen_cartera_resumen written

STEP_7_1_LABEL = "7.1"
STEP_7_1_NAME = "EXAM OF EVERY FORECAST SERIES"
STEP_7_1_PURPOSE = ("predict every forecast series in the exam months, with only what was known h months before, four ways "
                "on the same rows: the baseline (mandatory cell, last months), alone with its own history (raw), the "
                "ladder with the default technique, and the framework; every prediction traced, in units and USD, so the "
                "gain over the baseline can be measured in money, month by month, and summed")
STEP_7_1_ACTIONS = ["the exam: the calendar, the forecast series with something due in the exam months",
                "the four methods for every series, month and horizon (with the interval of raw, por_defecto and framework)",
                "the errors: per forecast series, and of the total per month",
                "check the exam (checks 1-4)",
                "write the exam (checks 5-8)",
                "count the checks; stop if any failed",
                "show the methods against each other, as tables"]
STEP_7_1_OUTPUT = ("every prediction of every forecast series, traced · the exam per series and method · the error of the total "
               "per month · tables sff_series_exam_detail, sff_series_exam, sff_examen_cartera, sff_examen_cartera_resumen")

METHOD_RAW = "raw"
METHODS_IN_ORDER = [METHOD_BASELINE, METHOD_RAW, METHOD_DEFAULT, METHOD_FRAMEWORK]      # the order of the cascade (raw apart)


def examine_series(rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                   backtest: dict, configuration: Config, reference_members: pd.DataFrame = None,
                   fine_table: pd.DataFrame = None, rate_box: RateBox = None) -> dict:
    """The exam of every forecast series by the four methods.

    INPUT:   rated_units (step 1.7) · series_estimate (2.3: composition, z, reference) · the composition series
             (2.4) · the backtest (2.6: decision and error bands) · the members of every credibility reference (2.2).
    OUTPUT:  dict(detail, per_series, by_month, summary).
    RULES:   see the module header.
    EDGE CASES: a series with no composition (solo_historia, mixed) is predicted by its mandatory cell in the
             framework; its raw method still uses its own history. The baseline has no interval.
    """
    configuration.log_step_start(STEP_7_1_LABEL, STEP_7_1_NAME, STEP_7_1_PURPOSE, STEP_7_1_ACTIONS, STEP_7_1_OUTPUT)
    check_log = []
    period = configuration.period_col
    due, renewed = configuration.pipeline_units_col, configuration.renewed_units_col

    # [1] the exam
    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & (rated_units[CALENDAR_ROLE_COLUMN] == ROLE_TEST)
                       & (rated_units[due] > 0)].copy()
    closed = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                         & (rated_units[due] > 0)].copy()
    closed["_cell"] = join_columns(closed, configuration.business_mandatory_dims)
    real["_cell"] = join_columns(real, configuration.business_mandatory_dims)
    real = real.merge(series_estimate[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "z", "credibility_ref_id"]],
                      on=SERIES_ID_COLUMN, how="left")
    exam_months = sorted(real[period].unique())
    configuration.log_action(STEP_7_1_LABEL, 1, f"exam months {exam_months[0]}..{exam_months[-1]} · {real[SERIES_ID_COLUMN].nunique():,} "
                                            f"forecast series · {len(real):,} series × month · horizons {configuration.backtest_horizons}")

    # [2] the four methods
    decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    composition_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()
                                    & (pool_series["vencen"] > 0)]
    # THE BOX OF THE RATE (the forecast's): every (month, horizon) below asks it with only what was known at its origin
    if rate_box is None:
        rate_box = RateBox(rated_units, series_estimate, pool_series, backtest, reference_members, configuration)
    # the series alone: its own history, indexed ONCE
    series_index = series_history_index(closed, period, due, renewed)
    rows, latest_used = [], []
    for target_month in exam_months:
        for horizon in configuration.backtest_horizons:
            origin = target_month - horizon
            known = closed[closed[period] <= origin]
            latest_used.append((known[period].max(), composition_truth.loc[composition_truth[period] <= origin, period].max(), origin))
            month_rows = real[real[period] == target_month].copy()
            month_rows["h"], month_rows["origin"] = horizon, origin
            month_rows["tramo_h"] = band_of_horizon(int(horizon), configuration.horizon_bands)
            rows.append(predict_month(month_rows, known, rate_box, series_index, decision, origin, horizon, configuration))
    detail = pd.concat(rows, ignore_index=True)
    methods = [method for method in METHODS_IN_ORDER if method in set(detail["method"])]
    configuration.log_action(STEP_7_1_LABEL, 2, f"{len(detail):,} predictions · methods {methods}")

    # [3] the errors
    per_series = errors_per_series(detail, configuration)
    by_month, summary = errors_of_the_total(detail, methods, configuration)
    honest = None
    if configuration.maturation_dims_in_use and fine_table is not None:
        honest = honest_total_exam(detail[detail["method"] == METHOD_FRAMEWORK], fine_table, configuration)
        by_month, summary = add_the_honest_exam(by_month, summary, honest)
    configuration.log_action(STEP_7_1_LABEL, 3, "per method, all series: " + " · ".join(
        f"{method} WAPE {row['wape']:.1%}" + (f", {row['in_interval']:.0%} in the interval" if pd.notna(row["in_interval"]) else "")
        for method, row in overall_by_method(detail, configuration).iterrows()))

    # [4] the checks
    configuration.log_action(STEP_7_1_LABEL, 4, "checking the exam")
    peeked = [origin for used, composition_used, origin in latest_used
              if (pd.notna(used) and used > origin) or (pd.notna(composition_used) and composition_used > origin)]
    configuration.log_check(STEP_7_1_LABEL, check_log, "no prediction used a month after its origin", not peeked,
                            failure_detail=f"{len(peeked)} exams used later months", context=f"{len(latest_used)} month × horizon exams")
    expected_rows = len(real) * len(configuration.backtest_horizons)
    per_method = detail.groupby("method").size()
    configuration.log_check(STEP_7_1_LABEL, check_log, "every method predicted every series of every exam month",
                            bool((per_method == expected_rows).all()) and detail["pred_rate"].notna().all(),
                            failure_detail=f"rows per method {per_method.to_dict()} for {expected_rows:,}",
                            context=f"{len(methods)} methods × {expected_rows:,}")
    real_total = real[renewed].sum() * len(configuration.backtest_horizons)
    adds_up = all(abs(detail.loc[detail["method"] == method, "real_units"].sum() - real_total) <= configuration.units_tolerance for method in methods)
    configuration.log_check(STEP_7_1_LABEL, check_log, "the real renewals of the exam equal the renewals of the exam months (nothing counted twice)",
                            adds_up, failure_detail="a method lost or duplicated renewals",
                            context=f"{real[renewed].sum():,.0f} renewed units in the exam months")
    framework = detail[detail["method"] == METHOD_FRAMEWORK]
    consistent = (int(per_series["framework_predictions"].sum()) == len(framework)
                  and abs(per_series["framework_real_units"].sum() - framework["real_units"].sum()) <= configuration.units_tolerance)
    configuration.log_check(STEP_7_1_LABEL, check_log, "the summary per series adds up to the detail (counts and units)", consistent,
                            failure_detail="sff_series_exam does not add up to sff_series_exam_detail")

    # [5] the tables
    configuration.log_action(STEP_7_1_LABEL, 5, "writing the exam")
    configuration.write_table(STEP_7_1_LABEL, check_log, detail, TABLE_SERIES_EXAM_DETAIL)
    configuration.write_table(STEP_7_1_LABEL, check_log, per_series, TABLE_SERIES_EXAM)
    configuration.write_table(STEP_7_1_LABEL, check_log, by_month, TABLE_PORTFOLIO_EXAM)
    configuration.write_table(STEP_7_1_LABEL, check_log, summary, TABLE_PORTFOLIO_EXAM_SUMMARY)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_7_1_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_7_1_LABEL, STEP_7_1_NAME, check_log)

    # [7] the methods against each other
    configuration.log_action(STEP_7_1_LABEL, 7, "the methods on the same rows (error_total: of the sum of the portfolio; wape_series: "
                                            "series by series, no compensation):")
    configuration.show_table(summary)
    show_the_cascade(summary, configuration)
    if honest is not None:
        show_the_honest_exam(honest, configuration)
    configuration.show_query(STEP_7_1_LABEL, "the exam per method and horizon", exam_summary_query(configuration))
    configuration.show_query(STEP_7_1_LABEL, "the total of every exam month per method (sff_examen_cartera)",
                             exam_by_month_query(configuration, methods))
    return dict(detail=detail, per_series=per_series, by_month=by_month, summary=summary)


def show_the_cascade(summary: pd.DataFrame, configuration: Config) -> None:
    """The cascade on screen: from the baseline, what every piece of the framework is worth, per horizon: the mean
    |error| of the monthly total in USD, the gain over the baseline and in how many exam months it erred less."""
    doc = configuration.logger.doc
    cascade = summary[summary["metodo"].isin([METHOD_BASELINE, METHOD_DEFAULT, METHOD_FRAMEWORK])]
    if cascade.empty or METHOD_BASELINE not in set(cascade["metodo"]):
        return
    doc(f"[{STEP_7_1_LABEL}] ══════════ LA CASCADA: DE LA BASELINE AL FRAMEWORK ══════════")
    doc(f"[{STEP_7_1_LABEL}] baseline = su celda mandatory, últimos {configuration.baseline_months} meses · por_defecto = + escalera "
        f"y credibilidad con la técnica por defecto ({configuration.default_technique}) · framework = + la técnica elegida. "
        f"error_usd_medio: |error| medio del total de un mes en USD; ganancia_usd_mes: lo que se yerra menos que la baseline; "
        f"meses_mejor_que_baseline: en cuántos meses de examen yerra menos")
    shown = cascade[["h", "metodo", "error_total_medio", "error_usd_medio", "ganancia_usd_mes", "meses_mejor_que_baseline"]].copy()
    configuration.show_table(shown.round({"error_total_medio": 4, "error_usd_medio": 0, "ganancia_usd_mes": 0,
                                          "meses_mejor_que_baseline": 2}))
    for horizon, block in cascade.groupby("h"):
        rows = block.set_index("metodo")
        if METHOD_FRAMEWORK not in rows.index:
            continue
        framework, baseline = rows.loc[METHOD_FRAMEWORK], rows.loc[METHOD_BASELINE]
        consistent = framework["meses_mejor_que_baseline"]
        doc(f"[{STEP_7_1_LABEL}] VALORACIÓN h = {horizon}: la baseline yerra el total de un mes en ${baseline['error_usd_medio']:,.0f} de media "
            f"({baseline['error_total_medio']:.1%}); el framework en ${framework['error_usd_medio']:,.0f} ({framework['error_total_medio']:.1%}): "
            + (f"${framework['ganancia_usd_mes']:,.0f} menos al mes" if framework['ganancia_usd_mes'] >= 0
               else f"${-framework['ganancia_usd_mes']:,.0f} MÁS al mes que la baseline")
            + f", mejor en el {consistent:.0%} de los meses de examen"
            + (" (consistente)" if consistent >= 0.75 else " (NO consistente: gana algunos meses y pierde otros)" if consistent >= 0.5
               else " (la baseline gana más meses)"))
    doc(f"[{STEP_7_1_LABEL}] ══════════ FIN DE LA CASCADA ══════════")


def add_the_honest_exam(by_month: pd.DataFrame, summary: pd.DataFrame, honest: pd.DataFrame) -> tuple:
    """The honest exam next to the others: its predictions and errors in the table by month, and two rows of the
    summary (framework_estados_hoy: the states seen at the origin and no curve; framework_honesto: with the curve)."""
    columns = ["mes", "h", "estados_hoy_pred", "estados_hoy_error_total", "honesto_pred", "honesto_error_total"]
    by_month = by_month.merge(honest[columns].rename(columns={
        "estados_hoy_pred": "framework_estados_hoy_pred", "estados_hoy_error_total": "framework_estados_hoy_error_total",
        "honesto_pred": "framework_honesto_pred", "honesto_error_total": "framework_honesto_error_total"}), on=["mes", "h"], how="left")
    rows = []
    for method, column in (("framework_estados_hoy", "estados_hoy_error_total"), ("framework_honesto", "honesto_error_total")):
        for horizon, errors in honest.groupby("h")[column]:
            rows.append({"metodo": method, "h": horizon, "error_total_medio": errors.abs().mean(),
                         "sesgo_total_medio": errors.mean()})
    summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)
    return by_month, summary


def show_the_honest_exam(honest: pd.DataFrame, configuration: Config) -> None:
    """The honest exam on screen: the error of the total with the final states (the ceiling), with the states seen at
    the origin, and with those plus the maturation curve, per horizon and month; with a VALORACIÓN line."""
    doc = configuration.logger.doc
    doc(f"[{STEP_7_1_LABEL}] ══════════ EXAMEN HONESTO DE LOS ESTADOS DEL CLIENTE ({', '.join(configuration.maturation_dims_in_use)}) ══════════")
    doc(f"[{STEP_7_1_LABEL}] the total of every exam month predicted three ways with the same rates of the framework: with the "
        f"FINAL states (techo: what predicting the states perfectly would give, the exam of always), with the states as they "
        f"were h months before (estados_hoy) and with those plus the maturation curve known at the origin (honesto):")
    without_columns = [state for state, sign in configuration.structural_timevarying_dims.items()
                       if sign == "negative" and state not in configuration.maturation_dims_in_use]
    if without_columns:
        doc(f"[{STEP_7_1_LABEL}] OJO: {', '.join(without_columns)} no trae(n) columnas de maduración, o su curva no pasó su examen "
            f"(4.1): aquí siguen con su estado FINAL. El examen es honesto solo para "
            f"{', '.join(configuration.maturation_dims_in_use)}; para los demás estados "
            f"sigue siendo el techo")
    error_columns = ["techo_estados_finales_error_total", "estados_hoy_error_total", "honesto_error_total"]
    shown = honest[["mes", "h", "renovadas_reales"] + error_columns].copy()
    shown[error_columns] = shown[error_columns].round(4)
    configuration.show_table(shown)
    per_horizon = honest.groupby("h")[error_columns].agg(lambda errors: errors.abs().mean())
    bias = honest.groupby("h")[error_columns].mean().add_suffix("_sesgo")
    configuration.show_table(per_horizon.join(bias).round(4).reset_index())
    for horizon, row in per_horizon.iterrows():
        doc(f"[{STEP_7_1_LABEL}] VALORACIÓN h = {horizon}: the error of the total is {row['honesto_error_total']:.1%} honest, against "
            f"{row['techo_estados_finales_error_total']:.1%} with the final states (the ceiling) and {row['estados_hoy_error_total']:.1%} "
            f"with the states seen and no curve: predicting the states is worth "
            f"{row['honesto_error_total'] - row['techo_estados_finales_error_total']:+.1%} of the total at this horizon")
    doc(f"[{STEP_7_1_LABEL}] ══════════ FIN DEL EXAMEN HONESTO ══════════")


def series_history_index(closed: pd.DataFrame, period: str, due: str, renewed: str) -> dict:
    """The closed months of every forecast series (index_history), with units due and renewed and the
    logit of its own rate per month. The monthly rate of a small series is often 0 or 1 (2 of 2 renewed): its logit
    would be ±14 and one month would pull any average of logits far away, which is not the series' fault but the
    method's. The rate is taken with the continuity correction (renewed + ½) / (due + 1), so a series alone is judged
    by what its months say, not by the infinity of its 0 % and 100 %."""
    series_index = index_history(closed, SERIES_ID_COLUMN, period, [renewed, due])
    for entry in series_index.values():
        entry["renewed"], entry["due"] = entry[renewed], entry[due]
        entry["rate_logit"] = logit((entry["renewed"] + 0.5) / (entry["due"] + 1.0))
        entry["rate_logit"].flags.writeable = False
    return series_index

def predict_month(month_rows: pd.DataFrame, known: pd.DataFrame, rate_box: RateBox, series_index: dict,
                  decision: pd.Series, origin, horizon: int, configuration: Config) -> pd.DataFrame:
    """The predictions of one exam month and horizon, one row per series and method. The framework and por_defecto
    come from THE BOX OF THE RATE at the origin; the series alone from its own history (series_history_index); the
    baseline from its mandatory cell. Each one uses only the months up to the origin."""
    period, due, renewed = configuration.period_col, configuration.pipeline_units_col, configuration.renewed_units_col
    band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
    global_level = known[renewed].sum() / known[due].sum()
    # a series alone with no month of its own: the level of its cell (the box's fallback)
    cell_with_states, _, box_global_level = rate_box.fallback_levels(origin)
    month_rows = month_rows.assign(cell_level=rate_box.cell_key(month_rows).map(cell_with_states).fillna(box_global_level))

    # the framework: THE BOX OF THE RATE learned at the origin (the same box the forecast uses); and the same box with
    # the default technique for every composition (por_defecto: the ladder without the technique choice)
    framework, with_default = month_rows.copy(), month_rows.copy()
    for frame, method, mode in ((framework, METHOD_FRAMEWORK, CHOSEN_MODE), (with_default, METHOD_DEFAULT, DEFAULT_MODE)):
        rates = rate_box.predict(month_rows, origin, mode, min_months=configuration.exam_min_months_to_predict)
        frame["method"] = method
        frame["technique"] = rates["tecnica"]
        frame["composition_rate"] = rates["tasa_pool_h"]
        frame["credibility_shift_logit"] = rates["credibility_shift_logit"]
        frame["pred_rate"] = rates["tasa"]

    # raw: the forecast series alone, with its own months
    own_rate, own_technique = {}, {}
    composition_of = month_rows.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN].to_dict()
    for series_id in sorted(set(month_rows[SERIES_ID_COLUMN])):
        entry = series_index.get(series_id)
        months_at_origin = months_known(entry, origin) if entry is not None else 0
        if months_at_origin == 0:
            continue                                  # no own month yet: the cell level, below (fillna)
        composition_id = composition_of.get(series_id)
        if months_at_origin < configuration.exam_min_months_to_predict:
            own_rate[series_id] = entry["renewed"][:months_at_origin].sum() / entry["due"][:months_at_origin].sum()
            own_technique[series_id] = "own_level"
            continue
        technique = decision.get((composition_id, band_name), configuration.default_technique)
        if months_at_origin < int(CATALOGUE.get(technique, (None, None, None, 1))[3]):
            technique = configuration.default_technique
        technique, rate = predict_composition(entry["rate_logit"][:months_at_origin], entry["month"][:months_at_origin],
                                              technique, horizon, configuration.default_technique)
        own_rate[series_id], own_technique[series_id] = rate, technique
    raw = month_rows.copy()
    raw["method"] = METHOD_RAW
    raw["technique"] = raw[SERIES_ID_COLUMN].map(own_technique).fillna("mandatory_cell")
    raw["composition_rate"] = np.nan
    raw["credibility_shift_logit"] = 0.0
    raw["pred_rate"] = raw[SERIES_ID_COLUMN].map(own_rate).fillna(raw["cell_level"])

    # the interval of raw, por_defecto and framework, as the forecast builds its band (the box's)
    for frame in (raw, with_default, framework):
        frame["band_low"], frame["band_high"] = rate_box.band(
            pd.DataFrame({"tecnica": frame["technique"], "h": frame["h"], "tasa": frame["pred_rate"]}), frame[due])

    # the baseline: the mandatory cell over the last baseline_months closed months known at the origin (no interval)
    window = known[known[period] > origin - configuration.baseline_months]
    window_rate = window[renewed].sum() / window[due].sum() if window[due].sum() > 0 else global_level
    cell_window_rate = window.groupby("_cell")[renewed].sum() / window.groupby("_cell")[due].sum().where(lambda value: value > 0)
    baseline = month_rows.copy()
    baseline["method"], baseline["technique"] = METHOD_BASELINE, f"last {configuration.baseline_months} months of its cell"
    baseline["pred_rate"] = baseline["_cell"].map(cell_window_rate).fillna(window_rate).to_numpy(dtype=float)
    baseline["composition_rate"], baseline["credibility_shift_logit"] = np.nan, 0.0
    baseline["band_low"], baseline["band_high"] = np.nan, np.nan

    predictions = pd.concat([baseline, raw, with_default, framework], ignore_index=True)
    predictions["due_units"] = predictions[due]
    predictions["real_units"] = predictions[renewed]
    predictions["real_rate"] = predictions["real_units"] / predictions["due_units"]
    predictions["pred_units"] = predictions["pred_rate"] * predictions["due_units"]
    predictions["err_units"] = predictions["pred_units"] - predictions["real_units"]
    predictions["err_pp"] = 100 * (predictions["pred_rate"] - predictions["real_rate"])
    # the money: the same price for every method (the one its real renewals had), so USD differences are rate differences
    usd_due, usd_renewed = configuration.pipeline_usd_col, configuration.renewed_usd_col
    renewed_usd = predictions[usd_renewed].fillna(0.0)
    renewers_due_value = (predictions["real_units"] * predictions[usd_due] / predictions["due_units"]).sum()
    month_uplift = renewed_usd.sum() / renewers_due_value if renewers_due_value > 0 else 1.0
    price = np.where(predictions["real_units"] > 0, renewed_usd / predictions["real_units"].where(predictions["real_units"] > 0),
                     predictions[usd_due] / predictions["due_units"] * month_uplift)
    predictions["real_usd"] = renewed_usd
    predictions["pred_usd"] = predictions["pred_units"] * price
    predictions["err_usd"] = predictions["pred_usd"] - predictions["real_usd"]
    # the noise of the rate it tries to hit: what even a perfect prediction would miss by
    predictions["noise_pp"] = 100 * np.sqrt(np.clip(predictions["pred_rate"] * (1 - predictions["pred_rate"]), 0, None)
                                            / predictions["due_units"].clip(lower=1))
    has_interval = predictions["band_low"].notna()
    predictions["in_band"] = np.where(has_interval, ((predictions["real_rate"] >= predictions["band_low"] - 1e-12)
                                                     & (predictions["real_rate"] <= predictions["band_high"] + 1e-12)).astype(float), np.nan)
    return predictions[[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, period, "h", "tramo_h", "origin", "method", "technique",
                        "composition_rate", "credibility_shift_logit", "pred_rate", "band_low", "band_high", "real_rate",
                        "due_units", "pred_units", "real_units", "err_units", "err_pp", "noise_pp", "in_band",
                        "pred_usd", "real_usd", "err_usd"]]


def errors_per_series(detail: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per forecast series and method (raw, framework): predictions, in the interval, units, error, coverage;
    side by side, so the improvement over the raw is one subtraction."""
    frames = []
    for method in (METHOD_RAW, METHOD_FRAMEWORK):
        block = detail[detail["method"] == method].assign(abs_err_units=lambda frame: frame["err_units"].abs(),
                                                          abs_err_pp=lambda frame: frame["err_pp"].abs())
        grouped = block.groupby(SERIES_ID_COLUMN)
        summary = pd.DataFrame({"predictions": grouped.size(), "in_band": grouped["in_band"].sum(),
                                "pred_units": grouped["pred_units"].sum(), "real_units": grouped["real_units"].sum(),
                                "abs_err_units": grouped["abs_err_units"].sum(), "mae_pp": grouped["abs_err_pp"].mean(),
                                "bias_pp": grouped["err_pp"].mean(),
                                "rmse_pp": np.sqrt(block.assign(_e2=block["err_pp"] ** 2).groupby(SERIES_ID_COLUMN)["_e2"].mean()),
                                "noise_pp": np.sqrt(block.assign(_n2=block["noise_pp"] ** 2).groupby(SERIES_ID_COLUMN)["_n2"].mean())})
        summary["error_over_noise"] = summary["rmse_pp"] / summary["noise_pp"].where(summary["noise_pp"] > 0)
        summary["wape"] = summary["abs_err_units"] / summary["real_units"].where(summary["real_units"] > 0)
        summary["coverage"] = summary["in_band"] / summary["predictions"]
        frames.append(summary.add_prefix(f"{method}_"))
    per_series = pd.concat(frames, axis=1).reset_index()
    per_series["improvement_mae_pp"] = per_series[f"{METHOD_RAW}_mae_pp"] - per_series[f"{METHOD_FRAMEWORK}_mae_pp"]
    return per_series


def overall_by_method(detail: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per method, over every series: WAPE and the share of predictions in their interval."""
    grouped = detail.assign(abs_err_units=detail["err_units"].abs()).groupby("method")
    return pd.DataFrame({"wape": grouped["abs_err_units"].sum() / grouped["real_units"].sum(),
                         "in_interval": grouped["in_band"].mean()})


def errors_of_the_total(detail: pd.DataFrame, methods: list, configuration: Config) -> tuple:
    """By month and horizon: real, predicted and error of the total per method, in units and USD; the summary per
    method and horizon, with the gain over the baseline in USD and its consistency."""
    period = configuration.period_col
    month_rows, summary_rows = [], []
    for (target_month, horizon), block in detail.groupby([period, "h"]):
        first = block[block["method"] == methods[0]]
        real_units, real_usd = first["real_units"].sum(), first["real_usd"].sum()
        row = {"mes": target_month, "h": horizon, "renovadas_reales": real_units, "renovado_usd_real": real_usd}
        for method in methods:
            rows = block[block["method"] == method]
            row[f"{method}_pred"] = rows["pred_units"].sum()
            row[f"{method}_error_total"] = (row[f"{method}_pred"] - real_units) / real_units
            row[f"{method}_pred_usd"] = rows["pred_usd"].sum()
            row[f"{method}_error_usd"] = row[f"{method}_pred_usd"] - real_usd
        month_rows.append(row)
    by_month = pd.DataFrame(month_rows)
    for horizon in sorted(detail["h"].unique()):
        block = detail[detail["h"] == horizon]
        months = by_month[by_month["h"] == horizon]
        baseline_abs_usd = months[f"{METHOD_BASELINE}_error_usd"].abs() if METHOD_BASELINE in methods else None
        for method in methods:
            month_errors = months[f"{method}_error_total"]
            abs_usd = months[f"{method}_error_usd"].abs()
            rows = block[block["method"] == method]
            summary_rows.append({"metodo": method, "h": horizon, "error_total_medio": float(month_errors.abs().mean()),
                                 "sesgo_total_medio": float(month_errors.mean()),
                                 "error_usd_medio": float(abs_usd.mean()),
                                 "ganancia_usd_mes": float(baseline_abs_usd.mean() - abs_usd.mean()) if baseline_abs_usd is not None else np.nan,
                                 "meses_mejor_que_baseline": float((abs_usd < baseline_abs_usd).mean())
                                                             if baseline_abs_usd is not None and method != METHOD_BASELINE else np.nan,
                                 "wape_series": float(rows["err_units"].abs().sum() / rows["real_units"].sum()),
                                 "error_vs_ruido": float(np.sqrt((rows["err_pp"] ** 2).mean()) / np.sqrt((rows["noise_pp"] ** 2).mean()))
                                                   if (rows["noise_pp"] > 0).any() else np.nan,
                                 "en_intervalo": float(rows["in_band"].mean()) if rows["in_band"].notna().any() else np.nan})
    order = {method: position for position, method in enumerate(methods)}
    summary = pd.DataFrame(summary_rows)
    summary["_orden"] = summary["metodo"].map(order)
    return by_month, summary.sort_values(["h", "_orden"]).drop(columns="_orden").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 7.2 · FORECAST  (until 25-oct-2026: step_20_forecast.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE FORECAST IN MONEY: the assembly. Every future row takes its rate, its uplift and its
# customer states from the boxes of the stages before, and is valued: units × rate, USD × rate × uplift; then the totals
# by month and by year, with the official band of the exam.
#
# WHAT THIS STEP DOES NOT DO (each lives in its own stage, with its own exam):
#   · the rate of a series at a horizon          → the box of the rate (step 2.7), examined by step 7.1
#   · the uplift of a cell and of a state        → step 3.1, examined by step 3.3
#   · the customer states at due date            → the box of the customer states (step 4.1), examined by step 7.1
#   · the rows the extract cannot have yet       → the box of the future pipeline (step 5.1), with its exam
#   · the error of a total                       → the exam of every forecast series (step 7.1)
#
# THE FUTURE ROWS: the fine rows of the extract from the current month on (pipeline 'real') and the rows of the
# SIMULATION WINDOW (current month … simulation_end) that the box of step 5.1 gives: proyectada (the 1-year licences
# due in the window renew as this forecast expects, due 12 months later) and simulada (the acquisitions of the window).
# THE TIME_SERIES UNIVERSE (the box of step 6.1, by region) is not predicted here: its rows come simulated, with their
# band. The totals ADD them: its revenue (rest of this year and whole of next year), its re-entry as pipeline of next
# year and its expected renewals, and its error in the band (see THE TOTALS).
#
# For every future row:
#   · h = months from the last closed month to the row's month (1 = the first month to predict)
#   · THE RATE: the box of step 2.7 (the technique chosen for its composition and horizon, moved toward its credibility
#     reference; without composition, its mandatory cell with its customer states, then its combination of states, then
#     the portfolio). Its band: the error quantiles of its technique at its horizon × the binomial error with the units
#     due of its forecast unit; clipped to [0, 1].
#   · THE UPLIFT of its cell: the CONTRACT rule 1 / (1 − discount) where the discount is known and the uplift backtest
#     (step 3.3) chose it; the STATISTICAL uplift of its cell otherwise (with its bootstrap band), without the price
#     increases (step 3.1): a row whose month is still inside the cycle of an increase of its price group takes it again
#     (factor_subida); and × the price factor of its customer states when step 3.3 found it predicts better
#     (factor_estado).
#   · EXPECTED: renewed units = units × rate; renewed USD = USD × rate × uplift; the band of the USD from the bands of
#     the rate and the uplift.
# THE CUSTOMER STATES AT DUE DATE (the box of step 4.1). The future rows carry TODAY's states: a licence due in 10 months
# has not had the time to stop using the product or to be softcancelled, and a projected or simulated row is born
# neutral. The box says, per group (the series without its states), month and combination of states, how many units
# there will be when they fall due. This step only VALUES what the box moves; the data does not change:
#   · the forecast with TODAY's states is kept (esperado_*_hoy: units due of the row × its rate × its uplift)
#   · a neutral row keeps (1 − l) of its units and USD due (vencen_*_al_vencer), l = the share of the neutral units of
#     its group and month the box moves; what leaves, leaves at the row's average price (10 units of $10 = $100)
#   · what arrives at a series with a state becomes a PREDICTION row (origen_fila maduracion, fila_maduracion = 1): its
#     data columns at 0, its units and USD at due date = what it receives; ONE row per destination series, month and
#     origin of the pipeline (aggregate_the_prediction_rows: exact, the uplift USD-weighted); it is predicted like any
#     future row, with the rate and the uplift of ITS series; its discount is unknown (statistical uplift)
#   · the forecast (esperado_*) = units and USD at due date × rate × uplift; maduracion_* = at due date − today
#   The extract rows are matured first: their expected renewals at due date are the pipeline of the projected rows;
#   then the projected and simulated rows (no maturation columns: the box gives them the historical final share).
#   The identity box leaves every row as it is: the forecast is the one with today's states.
# THE TOTALS by month and by year. THE OFFICIAL BAND is the exam's (error_examen_*, banda_examen_*): the error the exam
# of step 7.1 measured on the total of a month at the same horizon (with the states predicted when the extract dates
# them); by year, the errors of its months added (the months of a year err together). The LINEAR band (Σ of the row
# bands: every row wrong in the same direction) and the QUADRATURE one (√Σ of the row half-widths²: rows independent)
# stay as references: they are not the error of a total.
#   The time_series universe in the totals (0 without time_series rows):
#     by month  ingresos_ts_usd (revenue of ts_proyectado and ts_simulado) · vence_ts_reentrada_usd (its pipeline) ·
#               esperado_ts_reentrada (its expected renewals) · error_ts_usd (the half band its exam gives) ·
#               usd_vence_total = usd_vence + vence_ts_reentrada_usd · total_usd = esperado_usd + ingresos_ts_usd +
#               esperado_ts_reentrada. esperado_usd stays the renewals of the normal universe (check 5).
#     by year   ingresos_ts_real_usd (the closed months of the universe) · ingresos_ts_usd · vence_ts_reentrada_usd ·
#               esperado_ts_reentrada_usd · total_usd = renovado_real_usd + esperado_usd + the three ts revenues.
#     the band  the official one has every error measured on closed months (add_the_exam_band): the rate (7.1), the
#               price (3.3), the volume of the simulated acquisitions (5.1) and the time_series universe (6.1);
#               the linear band adds the ts bands; the quadrature one adds error_ts_usd as one more term. Every band
#               is centred on total_usd.
#
# Actions (logged as they are done):
#   1. the future rows and their horizon
#   2. the rate of every future row from the box of the rate, with its technique
#   3. where the rate of every future row comes from (pool, cell or global) and its band
#   4. the uplift of every future row (contract or statistical) and its band
#   5. the expected renewals with today's states and their band, row by row
#   6. the customer states of the extract rows at due date (the box of step 4.1), valued
#   7. the rows of the window (the box of step 5.1), predicted and matured; the confidence of every row
#   8. the totals by month and by year, with the band of the exam
#   9. check the forecast                                              checks 1-7
#  10. write the months, the summary and the maturation                checks 8-10
#  11. count the checks; stop if any failed
#  12. show the forecast by month, the answers by year and the maturation, as tables; the detector of
#      states above their final share (ESTADOS DEL CLIENTE POR ENCIMA DE SU PROPORCIÓN FINAL)
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every future row has a rate, an uplift and an expected value
#    2. every rate and every band is inside [0, 1] and the band contains the rate
#    3. no closed row is forecast; every future row is
#    4. the pipeline of the future rows is conserved (Σ USD due)
#    5. the totals are the sum of the rows
#    6. the maturation moves units, it does not create them (Σ units at due date = Σ units due; a neutral row keeps
#       between 0 and all of its units)
#    7. no negative state is TODAY above its final share by mark_excess_warning_pp or more in any future
#       month (WARN: the extract sets the state too early, e.g. a fresh cohort set to dormant; the detector says
#       which months and what it costs the forecast)
#    8-10. tables sff_forecast_mes, sff_resumen_negocio, sff_forecast_maduracion written and read back
#
# Output: (the forecast row by row, by month, by year) · tables sff_forecast_mes, sff_resumen_negocio,
# sff_forecast_maduracion. The forecast row by row reaches sff_nucleo (step 8.3).

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_7_2_LABEL = "7.2"
STEP_7_2_NAME = "FORECAST"
STEP_7_2_PURPOSE = ("assemble the forecast in money: every future row (of the extract and of the future pipeline) takes its "
                "rate (box of step 2.7), its uplift (step 3.1) and its customer states at due date (box of step 4.1), and "
                "is valued; the totals by month and by year with the official band of the exam (step 7.1)")
STEP_7_2_ACTIONS = ["the future rows of the extract",
                "the rate of every future row from the box of the rate, with its technique",
                "where the rate of every future row comes from (pool, cell or global) and its band",
                "the uplift of every future row (contract or statistical) and its band",
                "the expected renewals with today's states and their band, row by row",
                "the customer states of the extract rows at due date (the box of step 4.1), valued",
                "the rows of the window (the box of step 5.1): predicted and matured; the confidence of every row",
                "the totals by month and by year, by origin of the pipeline, with the band of the exam",
                "check the forecast (checks 1-7)",
                "write the months, the summary and the maturation (checks 8-10)",
                "count the checks; stop if any failed",
                "show the forecast by month, the answers by year and the maturation, as tables"]
STEP_7_2_OUTPUT = "every future row with rate, uplift, expected USD and bands (in memory) · totals by month and year · three tables"



def assemble_forecast(fine_table: pd.DataFrame, forecast_units: pd.DataFrame, series_estimate: pd.DataFrame,
                      pool_series: pd.DataFrame, pool_reference: pd.DataFrame, backtest: dict, uplift_cells: pd.DataFrame,
                      uplift_verdict: dict, rated_units: pd.DataFrame, price_increases: pd.DataFrame,
                      configuration: Config, uplift_state_factors: pd.DataFrame = None,
                      state_maturation: StateMaturation = None, rate_box: RateBox = None,
                      series_exam: dict = None, future_pipeline: FuturePipeline = None,
                      time_series_universe: TimeSeriesUniverse = None, acquisition_exam: dict = None) -> dict:
    """The forecast row by row (extract and future pipeline), by month and by year; checked and written."""
    configuration.log_step_start(STEP_7_2_LABEL, STEP_7_2_NAME, STEP_7_2_PURPOSE, STEP_7_2_ACTIONS, STEP_7_2_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    boundaries = configuration.calendar_boundaries()
    last_closed = boundaries["current"] - 1
    box = state_maturation if state_maturation is not None else StateMaturation(fine_table, last_closed, configuration)
    rate_box = rate_box if rate_box is not None else RateBox(rated_units, series_estimate, pool_series, backtest, None, configuration)
    future_pipeline = future_pipeline if future_pipeline is not None else FuturePipeline(fine_table, configuration)
    context = dict(series_estimate=series_estimate, pool_series=pool_series, pool_reference=pool_reference, backtest=backtest,
                   uplift_cells=uplift_cells, uplift_verdict=uplift_verdict, price_increases=price_increases,
                   uplift_state_factors=uplift_state_factors, rated_units=rated_units, rate_box=rate_box,
                   forecast_units=forecast_units, last_closed=last_closed)

    # [1] the future rows of the extract
    future = fine_table[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION].copy()
    future["_fila"] = future.index
    future[PIPELINE_ORIGIN_COLUMN] = PIPELINE_REAL
    configuration.log_action(STEP_7_2_LABEL, 1, f"{len(future):,} future rows in the extract, {future[period_column].min()}.."
                                            f"{future[period_column].max()}; ${future[configuration.pipeline_usd_col].sum():,.0f} due; "
                                            f"last closed month {last_closed}")

    # [2]-[5] rate, uplift and expected renewals of the extract's future rows, with TODAY's states
    future = predict_rows(future, context, configuration)
    configuration.log_action(STEP_7_2_LABEL, 2, f"rates of the estimation ids at h = 1..{future['h'].max()} (technique of each id and band)")
    configuration.log_action(STEP_7_2_LABEL, 3, f"rate origins: {future['origen_tasa'].value_counts().to_dict()}")
    with_increase = future[future["factor_subida"] != 1.0]
    configuration.log_action(STEP_7_2_LABEL, 4, f"uplift paths: {future['via_uplift'].value_counts().to_dict()} · "
                                            f"{len(with_increase):,} rows still inside the cycle of a price increase "
                                            f"(${with_increase[configuration.pipeline_usd_col].sum():,.0f} due)"
                                            + (f" · price factor of the states applied ({future['factor_estado'].ne(1.0).sum():,} rows)"
                                               if uplift_verdict.get("factor_estado") else ""))
    configuration.log_action(STEP_7_2_LABEL, 5, f"expected ${future['esperado_usd'].sum():,.0f} renewed of "
                                            f"${future[configuration.pipeline_usd_col].sum():,.0f} due in the extract, with today's states")

    # [6] the maturation of the states of the extract rows: the box moves the units, this step values them. First the
    #     extract: its expected renewals at due date are the pipeline of the projected rows
    future, extract_table = mature_and_value(future, box, context, configuration, with_dates=True)
    arrived = future[MATURATION_ROW_COLUMN] == 1
    configuration.log_action(STEP_7_2_LABEL, 6, f"maturation of the extract ({box.describe()}): "
                                            f"{future['maduracion_unidades'].clip(lower=0).sum():,.0f} units arrive at a series with a "
                                            f"state in {int(arrived.sum()):,} prediction rows · "
                                            f"{future['maduracion_delta_usd'].sum():+,.0f} USD in the forecast")

    # [7] the simulation window: the box of the future pipeline gives its rows (the renewals of the extract's 1-year
    #     licences due in the window, the simulated acquisitions); predicted with neutral states and then matured by the
    #     same box of the states (they have no maturation columns: the historical final share)
    extension = future_pipeline.new_rows(future)
    extension_table = pd.DataFrame()
    if len(extension):
        extension = predict_rows(extension, context, configuration)
        extension["_fila"] = np.nan
        extension, extension_table = mature_and_value(extension, box, context, configuration, with_dates=False)
        future = pd.concat([future, extension], ignore_index=True)
    window = configuration.simulation_window
    created = extension[extension[MATURATION_ROW_COLUMN] == 0] if len(extension) else extension
    configuration.log_action(STEP_7_2_LABEL, 7, f"simulation window {window[0] if window else '—'}..{window[-1] if window else '—'} → "
                                            f"due {window[0] + 12 if window else '—'}..{window[-1] + 12 if window else '—'}: "
                                            f"{created[PIPELINE_ORIGIN_COLUMN].value_counts().to_dict() if len(created) else {}} rows · "
                                            f"${created[configuration.pipeline_usd_col].sum() if len(created) else 0:,.0f} due · "
                                            f"${extension['esperado_usd'].sum() if len(extension) else 0:,.0f} expected (matured)")

    # the confidence of every row: how sure the forecast is of its rate
    future["confidence"] = confidence_of_rows(future, context, configuration)
    configuration.log_action(STEP_7_2_LABEL, 7, "confidence (USD expected): " + " · ".join(
        f"{label} {share:.0%}" for label, share in
        (future.groupby("confidence")["esperado_usd"].sum() / max(future["esperado_usd"].sum(), 1e-9)).items()))

    # the maturation by month, over the whole future (extract and extension), and the detector
    box_tables = pd.concat([table for table in (extract_table, extension_table) if len(table)], ignore_index=True) \
        if len(extract_table) or len(extension_table) else extract_table
    maturation = maturation_by_month(future, box_tables, box, configuration)
    states_detector = states_above_final(maturation, box, configuration) if maturation is not None else None

    # [8] the totals
    by_month, by_year = totals(future, fine_table, configuration, series_exam, time_series_universe,
                               price_error=uplift_verdict.get("error_precio_total", 0.0), acquisition_exam=acquisition_exam)
    configuration.log_action(STEP_7_2_LABEL, 8, f"{len(by_month)} months · {len(by_year)} years"
                             + (f" · time_series universe added: ${by_month['ingresos_ts_usd'].sum():,.0f} revenue, "
                                f"${by_month['esperado_ts_reentrada'].sum():,.0f} expected of its re-entry"
                                if by_month["ingresos_ts_usd"].sum() or by_month["esperado_ts_reentrada"].sum() else ""))

    # [9] the checks
    configuration.log_action(STEP_7_2_LABEL, 9, "checking the forecast")
    check_forecast(future, fine_table, by_month, configuration, check_log)
    units_today = future[configuration.pipeline_units_col].sum()
    units_at_due = future[DUE_UNITS_AT_DUE_COLUMN].sum()
    usd_difference = future[DUE_USD_AT_DUE_COLUMN].sum() - future[configuration.pipeline_usd_col].sum()
    neutral_kept = future.loc[future[MATURATION_ROW_COLUMN] == 0, DUE_UNITS_AT_DUE_COLUMN] / future.loc[
        future[MATURATION_ROW_COLUMN] == 0, configuration.pipeline_units_col].where(lambda units: units > 0)
    configuration.log_check(STEP_7_2_LABEL, check_log,
                            "the maturation moves units, it does not create them (Σ units and USD at due date = Σ due; "
                            "a row keeps between 0 and all of its units)",
                            abs(units_at_due - units_today) <= 1e-6 * max(units_today, 1.0)
                            and abs(usd_difference) <= configuration.money_tolerance
                            and bool(neutral_kept.dropna().between(-1e-9, 1 + 1e-9).all()),
                            failure_detail=f"units today {units_today:,.2f} vs at due date {units_at_due:,.2f} · USD "
                                           f"difference ${usd_difference:,.2f} · rows outside [0, 1]: "
                                           f"{int((~neutral_kept.dropna().between(-1e-9, 1 + 1e-9)).sum())}",
                            context=f"{future['maduracion_unidades'].clip(lower=0).sum():,.0f} units move")
    if states_detector is not None:
        flagged = states_detector[states_detector["exceso_pp"] >= configuration.mark_excess_warning_pp]
        configuration.log_check(STEP_7_2_LABEL, check_log,
                                f"no negative state is TODAY above its final share by {configuration.mark_excess_warning_pp:.1f} pp or "
                                f"more in any future month (a state only grows: one above its final share is in the state too early "
                                f"in the extract, e.g. a fresh cohort set to dormant)",
                                flagged.empty,
                                failure_detail=" · ".join(
                                    f"{state}: " + ", ".join(f"{row[configuration.period_col]} {row['exceso_pp']:+.1f} pp"
                                                            for _, row in rows_of_state.iterrows())
                                    + f" (≈ ${rows_of_state['impacto_usd_aprox'].sum():,.0f} the forecast lacks)"
                                    for state, rows_of_state in flagged.groupby("estado", sort=False))
                                + " · see ESTADOS DEL CLIENTE POR ENCIMA DE SU PROPORCIÓN FINAL",
                                blocking=False, context=f"{len(states_detector):,} month × state pairs looked at")

    # [08] the tables (the forecast row by row is not a table: sff_nucleo carries every future row with its forecast)
    configuration.log_action(STEP_7_2_LABEL, 10, "writing the months, the summary and the maturation")
    configuration.write_table(STEP_7_2_LABEL, check_log, by_month, TABLE_FORECAST_MONTH)
    configuration.write_table(STEP_7_2_LABEL, check_log, by_year, TABLE_BUSINESS_SUMMARY)
    if maturation is not None:
        configuration.write_table(STEP_7_2_LABEL, check_log, maturation, TABLE_FORECAST_MATURATION)

    # [09] the count of the checks; stop if anything failed
    configuration.log_action(STEP_7_2_LABEL, 11, "counting the checks")
    configuration.log_check_summary(STEP_7_2_LABEL, STEP_7_2_NAME, check_log)

    # [10] the forecast by month and the answers by year; the maturation next to it (label 4.1)
    configuration.log_action(STEP_7_2_LABEL, 12, "the forecast by month (banda_examen: the official band, the error of the exam "
                                             "at the same horizon; banda lineal and cuadratura: references, rows all wrong "
                                             "in the same direction or independent):")
    configuration.show_table(by_month)
    configuration.logger.doc(f"[{STEP_7_2_LABEL}] by year: renewed in the closed months + expected in the future ones (by origin "
                             f"of the pipeline):")
    configuration.show_table(by_year)
    if maturation is not None:
        show_maturation_by_month(maturation, box, configuration, STEP_7_2_LABEL)
        show_states_above_final(states_detector, configuration, STEP_7_2_LABEL)
    result = dict(forecast=future, by_month=by_month, by_year=by_year)
    if maturation is not None:
        result["maturation"] = maturation
    return result


# ─── the maturation of the states, valued (actions 6 and 7) ───────────────────────

def units_seen_today(rows: pd.DataFrame, configuration: Config, with_dates: bool) -> pd.DataFrame:
    """The input of the box: every future row with its units due today and whether it carries the maturation
    columns (the extract's rows do; the projected and simulated ones do not)."""
    timing_columns = [column for column in configuration.maturation_columns if column in rows.columns]
    seen = rows[list(dict.fromkeys(configuration.rate_series_columns + [configuration.period_col] + timing_columns))].copy()
    seen[UNITS_TODAY_COLUMN] = rows[configuration.pipeline_units_col].to_numpy(dtype=float)
    seen[WITH_DATES_COLUMN] = with_dates
    return seen


def mature_and_value(rows: pd.DataFrame, box: StateMaturation, context: dict, configuration: Config,
                     with_dates: bool) -> tuple:
    """The rows (predicted with today's states) after the maturation of the box: the neutral rows keep what does not
    leave, the units that arrive at a series with a state become prediction rows, everything valued. Returns (rows with
    the prediction rows appended, the table of the box)."""
    table = box.mature(units_seen_today(rows, configuration, with_dates))
    rows, arriving = apply_the_maturation(rows, table, box, configuration)
    if len(arriving):
        arriving = predict_rows(arriving, context, configuration)
        arriving = aggregate_the_prediction_rows(arriving, context, configuration)
        rows = pd.concat([rows, arriving], ignore_index=True)
    return expected_renewals(rows, configuration), table


MIXED_VALUE = "varias"      # an aggregated prediction row whose fine rows had different values in a column


def aggregate_the_prediction_rows(arriving: pd.DataFrame, context: dict, configuration: Config) -> pd.DataFrame:
    """The prediction rows of the customer states, ONE per destination series, month and origin of the pipeline.

    WHY:     they are born one per series, month, uplift cell and origin (160.535 in the like-real synthetic for
             66.336 extract rows): the core would grow by more than double with rows that hold no data.
    RULES:   the rate depends only on the series and the horizon, so it is the same in every fine row of a series and
             month: the aggregation is EXACT if the uplift is the USD-weighted mean of the fine rows (Σ USD × rate ×
             uplift = rate × Σ USD × the weighted uplift). Units and USD at due date are summed; every other number is
             the USD-weighted mean (identical when the fine rows agree); a text column keeps its value when the fine
             rows agree and takes 'varias' when they do not (the uplift cell, the discount bucket…). The band of the
             rate is computed again with the units of the aggregated row (a series without forecast unit takes its
             own units: the sum, not each fine piece)."""
    period = configuration.period_col
    keys = [SERIES_ID_COLUMN, period, PIPELINE_ORIGIN_COLUMN]
    weight = arriving[DUE_USD_AT_DUE_COLUMN].astype(float)
    weight = weight.where(weight > 0, arriving[DUE_UNITS_AT_DUE_COLUMN].astype(float)).clip(lower=1e-12)
    work = arriving.assign(_peso=weight)
    grouped = work.groupby(keys, sort=False, dropna=False)
    summed = [DUE_UNITS_AT_DUE_COLUMN, DUE_USD_AT_DUE_COLUMN, configuration.pipeline_units_col, configuration.pipeline_usd_col]
    other = [column for column in arriving.columns if column not in keys and column not in summed]
    numeric = [column for column in other if pd.api.types.is_numeric_dtype(arriving[column])
               and not pd.api.types.is_bool_dtype(arriving[column])]
    text = [column for column in other if column not in numeric]

    result = grouped[summed].sum()
    if numeric:
        # the fine rows agree (the rate, h, z…): that value exactly; they differ (the uplift…): the USD-weighted mean
        # of the rows that have one (a NaN — the discount of a prediction row — does not count; all NaN → NaN)
        present = work[numeric].notna()
        weights_present = present.mul(work["_peso"], axis=0)
        weighted = (work[numeric].fillna(0.0).mul(work["_peso"], axis=0)
                    .assign(**{column: work[column] for column in keys}).groupby(keys, sort=False, dropna=False)[numeric].sum())
        weight_of = weights_present.assign(**{column: work[column] for column in keys}) \
            .groupby(keys, sort=False, dropna=False)[numeric].sum()
        weighted_mean = weighted / weight_of.where(weight_of > 0)
        lowest, highest = grouped[numeric].min(), grouped[numeric].max()
        agree = (lowest == highest) | (lowest.isna() & highest.isna())
        result = result.join(lowest.where(agree, weighted_mean))
    if text:
        distinct = grouped[text].nunique(dropna=False)
        first = grouped[text].first()
        result = result.join(first.mask(distinct > 1, MIXED_VALUE))
    result = result.reset_index()[arriving.columns]
    result[MATURATION_ROW_COLUMN] = 1
    # the band of the rate with the units of the aggregated row
    unit_units = result[UNIT_ID_COLUMN].map(context["forecast_units"].set_index(UNIT_ID_COLUMN)[configuration.pipeline_units_col])
    unit_units = unit_units.fillna(result[DUE_UNITS_AT_DUE_COLUMN])
    result["tasa_baja"], result["tasa_alta"] = context["rate_box"].band(result, unit_units)
    return result


def apply_the_maturation(rows: pd.DataFrame, table: pd.DataFrame, box: StateMaturation, configuration: Config) -> tuple:
    """The units the box moves, row by row. Returns (rows with vencen_*_al_vencer, the new prediction rows).

    RULES:   per group and month, l = the share of its neutral units the box moves: a neutral row keeps (1 − l) of its
             units and USD due. What leaves, leaves at the row's AVERAGE price (l × units, l × USD), and goes to every
             combination with a state in proportion to what the box gives it. The prediction rows are one per destination
             series, month, uplift cell and origin of the pipeline: the dims of the neutral rows they come from with the
             states of their combination, their data columns at 0, their units and USD at due date = what they receive;
             their discount is unknown (the statistical uplift of their cell). The rows with a state of today do not move."""
    period = configuration.period_col
    units_column, usd_column = configuration.pipeline_units_col, configuration.pipeline_usd_col
    rows = rows.copy()
    rows[DUE_UNITS_AT_DUE_COLUMN] = rows[units_column].astype(float)
    rows[DUE_USD_AT_DUE_COLUMN] = rows[usd_column].astype(float)
    rows[MATURATION_ROW_COLUMN] = 0
    if table.empty or box.identity:
        return rows, rows.iloc[0:0]
    neutral_of_pair = table[table["combinacion"] == "neutral"].set_index(["_grupo", period])
    leaving_share = (1.0 - neutral_of_pair[UNITS_AT_DUE_COLUMN] / neutral_of_pair[UNITS_TODAY_COLUMN].where(lambda units: units > 0))
    leaving_share = leaving_share.clip(lower=0.0, upper=1.0).fillna(0.0)
    leaving_share = leaving_share[leaving_share > 0]
    if leaving_share.empty:
        return rows, rows.iloc[0:0]

    # the neutral rows keep what does not leave
    rows["_grupo"] = box.group_key(rows)
    is_neutral = state_combination(rows, box.states) == "neutral"
    share = pd.Series(list(zip(rows["_grupo"], rows[period])), index=rows.index).map(leaving_share).fillna(0.0).where(is_neutral, 0.0)
    rows[DUE_UNITS_AT_DUE_COLUMN] = rows[units_column] * (1.0 - share)
    rows[DUE_USD_AT_DUE_COLUMN] = rows[usd_column] * (1.0 - share)

    # what leaves, at the average price of its row, by group, month, uplift cell and origin of the pipeline
    leaving = rows[share > 0].assign(_sale_unidades=rows[units_column] * share, _sale_usd=rows[usd_column] * share)
    carried = list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion
                                 + ([configuration.discount_bucket_column] if configuration.discount_value_column else [])
                                 + ([configuration.term_column] if configuration.term_column else [])
                                 + ([configuration.acquisition_column] if configuration.acquisition_column else [])))
    carried = [column for column in carried if column in leaving.columns]
    keys = ["_grupo", period, UPLIFT_CELL_ID_COLUMN, PIPELINE_ORIGIN_COLUMN]
    leaving = (leaving.groupby(keys, dropna=False)
               .agg(**{column: (column, "first") for column in carried if column not in keys},
                    _sale_unidades=("_sale_unidades", "sum"), _sale_usd=("_sale_usd", "sum"))
               .reset_index())
    # where it goes: every combination with a state, in proportion to what the box gives it
    arriving_units = table[table["combinacion"] != "neutral"].assign(
        _recibe=lambda frame: (frame[UNITS_AT_DUE_COLUMN] - frame[UNITS_TODAY_COLUMN]).clip(lower=0.0))
    arriving_units = arriving_units[arriving_units["_recibe"] > 0]
    arriving_units["_fraccion"] = arriving_units["_recibe"] / arriving_units.groupby(["_grupo", period])["_recibe"].transform("sum")
    destinations = arriving_units[["_grupo", period, "combinacion", "_fraccion"] + box.states]
    new_rows = leaving.drop(columns=[state for state in box.states if state in leaving.columns]).merge(
        destinations, on=["_grupo", period], how="inner")
    if new_rows.empty:
        return rows.drop(columns=["_grupo"]), rows.iloc[0:0].drop(columns=["_grupo"])
    new_rows[units_column] = 0.0
    new_rows[usd_column] = 0.0
    new_rows[DUE_UNITS_AT_DUE_COLUMN] = new_rows["_sale_unidades"] * new_rows["_fraccion"]
    new_rows[DUE_USD_AT_DUE_COLUMN] = new_rows["_sale_usd"] * new_rows["_fraccion"]
    new_rows[MATURATION_ROW_COLUMN] = 1
    if configuration.discount_value_column:
        new_rows[configuration.discount_value_column] = np.nan
    new_rows[SERIES_ID_COLUMN] = join_columns(new_rows, configuration.rate_series_columns)
    new_rows[UNIT_ID_COLUMN] = new_rows[SERIES_ID_COLUMN] + "|" + new_rows[period].astype(str)
    new_rows[CALENDAR_ROLE_COLUMN] = ROLE_PROJECTION
    new_rows["_fila"] = np.nan
    new_rows = new_rows.drop(columns=["_grupo", "combinacion", "_fraccion", "_sale_unidades", "_sale_usd"])
    return rows.drop(columns=["_grupo"]), new_rows


def expected_renewals(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The expected renewals of every row, two ways: with TODAY's states (esperado_*_hoy: its units and USD due) and
    the forecast (esperado_*: its units and USD at due date, after the maturation); the band of the forecast; and the
    maturation of the row (maduracion_*: at due date − today, of the units due and of the expected renewals)."""
    rows = rows.copy()
    units_column, usd_column = configuration.pipeline_units_col, configuration.pipeline_usd_col
    if DUE_UNITS_AT_DUE_COLUMN not in rows.columns:
        rows[DUE_UNITS_AT_DUE_COLUMN] = rows[units_column].astype(float)
        rows[DUE_USD_AT_DUE_COLUMN] = rows[usd_column].astype(float)
    if MATURATION_ROW_COLUMN not in rows.columns:
        rows[MATURATION_ROW_COLUMN] = 0
    rows["esperado_unidades_hoy"] = rows[units_column] * rows["tasa"]
    rows["esperado_usd_hoy"] = rows[usd_column] * rows["tasa"] * rows["uplift"]
    rows["esperado_unidades"] = rows[DUE_UNITS_AT_DUE_COLUMN] * rows["tasa"]
    rows["esperado_usd"] = rows[DUE_USD_AT_DUE_COLUMN] * rows["tasa"] * rows["uplift"]
    rows["esperado_usd_bajo"] = rows[DUE_USD_AT_DUE_COLUMN] * rows["tasa_baja"] * rows["uplift_bajo"]
    rows["esperado_usd_alto"] = rows[DUE_USD_AT_DUE_COLUMN] * rows["tasa_alta"] * rows["uplift_alto"]
    rows["maduracion_unidades"] = rows[DUE_UNITS_AT_DUE_COLUMN] - rows[units_column]
    rows["maduracion_usd"] = rows[DUE_USD_AT_DUE_COLUMN] - rows[usd_column]
    rows["maduracion_delta_unidades"] = rows["esperado_unidades"] - rows["esperado_unidades_hoy"]
    rows["maduracion_delta_usd"] = rows["esperado_usd"] - rows["esperado_usd_hoy"]
    return rows


CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW = "high", "medium", "low"


def confidence_of_rows(rows: pd.DataFrame, context: dict, configuration: Config) -> np.ndarray:
    """The confidence of every future row:
      high    the half-width of its rate band ≤ confidence_high_band_pp (5), its composition judged by
              the backtest, and the exam error of the chosen technique for its horizon band ≤
              confidence_high_exam_pp (5)
      medium  its composition judged and the half-width of its band ≤ confidence_medium_band_pp (10)
      low     the rest: a wider band, a composition below the support floor (not judged), or a rate
              from the mandatory cell or the global rate (no composition)"""
    band_half_width_pp = 100 * (rows["tasa_alta"] - rows["tasa_baja"]) / 2
    judged_ids = set(context["pool_reference"].loc[context["pool_reference"]["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN])
    judged = rows[COMPOSITION_ID_COLUMN].isin(judged_ids) & (rows["origen_tasa"] == RATE_FROM_POOL)
    exam_of_key = context["backtest"]["exam_by_pool"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["elegida_err_pp_medio"].to_dict()
    horizon_band = rows["h"].map(lambda horizon: band_of_horizon(int(horizon), configuration.horizon_bands))
    exam_error = pd.Series([exam_of_key.get(pair, np.nan) for pair in zip(rows[COMPOSITION_ID_COLUMN], horizon_band)],
                           index=rows.index, dtype=float)
    high = (judged & (band_half_width_pp <= configuration.confidence_high_band_pp)
            & (exam_error <= configuration.confidence_high_exam_pp))
    medium = judged & (band_half_width_pp <= configuration.confidence_medium_band_pp)
    return np.select([high, medium], [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM], default=CONFIDENCE_LOW)


def predict_rows(rows: pd.DataFrame, context: dict, configuration: Config) -> pd.DataFrame:
    """Rate, uplift and expected renewals (with bands) of future rows, of the extract or extended."""
    rows = rows.copy()
    rows["h"] = rows[configuration.period_col].map(lambda month: (month - context["last_closed"]).n)
    rows = rate_of_rows(rows, context["rate_box"], context["last_closed"], context["forecast_units"], configuration)
    rows = uplift_of_rows(rows, context["uplift_cells"], context["uplift_verdict"], configuration,
                          price_increases=context.get("price_increases"), state_factors=context.get("uplift_state_factors"))
    return expected_renewals(rows, configuration)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RATE
# ═══════════════════════════════════════════════════════════════════════════════════

def rate_of_rows(future: pd.DataFrame, rate_box: RateBox, last_closed, forecast_units: pd.DataFrame,
                 configuration: Config) -> pd.DataFrame:
    """The rate of every future row from THE BOX OF THE RATE (stage 2.7), learned with every closed month: its
    composition's technique moved toward its credibility reference, or the level of its cell with its customer
    states, or the portfolio's; and its band, with the units due of its forecast unit."""
    # the same rule as the exam (step 7.1): a composition with fewer than exam_min_months_to_predict closed months does
    # not predict (its rows take their cell with their states): what was examined is what is forecast
    rates = rate_box.predict(future, last_closed, min_months=configuration.exam_min_months_to_predict)
    future = future.drop(columns=[column for column in rates.columns if column in future.columns]).join(rates)
    # the band: quantiles of the normalised error × binomial error with the unit's units due (prediction.py)
    unit_units = future[UNIT_ID_COLUMN].map(forecast_units.set_index(UNIT_ID_COLUMN)[configuration.pipeline_units_col])
    own_units = future[DUE_UNITS_AT_DUE_COLUMN] if DUE_UNITS_AT_DUE_COLUMN in future.columns else future[configuration.pipeline_units_col]
    unit_units = unit_units.fillna(own_units)       # an extended row or a prediction row of the states: its own units
    future["tasa_baja"], future["tasa_alta"] = rate_box.band(future, unit_units)
    return future


def uplift_of_rows(future: pd.DataFrame, uplift_cells: pd.DataFrame, uplift_verdict: dict, configuration: Config,
                   price_increases: pd.DataFrame = None, state_factors: pd.DataFrame = None) -> pd.DataFrame:
    """Contract where the discount is known and the backtest chose it (on every row with a known discount, or only on
    the neutral ones: PATH_CONTRACT_NEUTRAL); statistical otherwise.

    The statistical uplift of the cell is the one without price increases (step 3.1); a row whose month is
    still inside the cycle of an increase of its price group takes it again: uplift and band × factor_subida
    (1 elsewhere, and for the contract rule, which is the discount, not the tariff).
    When step 3.3 found that the price by states predicts better (uplift_verdict["factor_estado"]), the statistical
    uplift of a row is also × the factor of its combination of negative states (step 3.1: a softcancel renews with a
    retention offer, below its cell): factor_estado (1 elsewhere, and for the contract rule)."""
    cells = uplift_cells.set_index(UPLIFT_CELL_ID_COLUMN)
    statistical = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift"])
    # where the uplift of every row comes from: its cell (its row of sff_uplift_celda), else its mandatory cell, else
    # the global uplift; the contract rule below overwrites it with "contrato"
    uplift_origin = pd.Series(np.where(statistical.notna(), UPLIFT_ORIGIN_CELL, UPLIFT_ORIGIN_GLOBAL),
                              index=future.index, dtype=object)
    # a cell never seen (an extended row with a new discount bucket): its mandatory cell, then the global uplift
    if statistical.isna().any():
        uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
        weights = uplift_cells["renovadores"].clip(lower=1e-9)
        by_mandatory = (uplift_cells.assign(_w=weights, _wu=weights * uplift_cells["uplift"])
                        .groupby("uplift_celda_mandatory_id")[["_w", "_wu"]].sum())
        mandatory_uplift = by_mandatory["_wu"] / by_mandatory["_w"]
        global_uplift = float((weights * uplift_cells["uplift"]).sum() / weights.sum())
        of_mandatory = join_columns(future, list(uplift_mandatory)).map(mandatory_uplift)
        uplift_origin[statistical.isna() & of_mandatory.notna()] = UPLIFT_ORIGIN_MANDATORY
        statistical = statistical.fillna(of_mandatory.fillna(global_uplift))
    future = future.drop(columns=[column for column in ("uplift", "uplift_bajo", "uplift_alto", "via_uplift", "origen_uplift")
                                  if column in future.columns])
    future["uplift"] = statistical
    future["uplift_bajo"] = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift_bajo"]).fillna(statistical)
    future["uplift_alto"] = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift_alto"]).fillna(statistical)
    future["via_uplift"] = PATH_STATISTICAL
    future["origen_uplift"] = uplift_origin
    # the price by states, when step 3.3 found it predicts better
    future["factor_estado"] = 1.0
    if uplift_verdict.get("factor_estado"):
        future["factor_estado"] = state_factor_of_rows(future, state_factors, configuration)
        for uplift_column in ("uplift", "uplift_bajo", "uplift_alto"):
            future[uplift_column] = future[uplift_column] * future["factor_estado"]
    # the price increases still in their cycle, applied again (the uplift of the cell is the one without them)
    future["factor_subida"] = 1.0
    if configuration.uplift_without_increases and price_increases is not None and len(price_increases):
        group_dims = [column_name for column_name in configuration.price_group_dims if column_name in future.columns]
        factor = increase_factor(price_group_of(future, group_dims), future[configuration.period_col], price_increases)
        future["factor_subida"] = factor
        for uplift_column in ("uplift", "uplift_bajo", "uplift_alto"):
            future[uplift_column] = future[uplift_column] * factor
    discount = configuration.discount_value_column
    path_with_discount = uplift_verdict.get("via_usada_con_descuento")
    if discount and path_with_discount in (PATH_CONTRACT, PATH_CONTRACT_NEUTRAL):
        known = future[discount].notna()
        if path_with_discount == PATH_CONTRACT_NEUTRAL:
            # only the neutral rows of today: the units that will still take a state leave them in the distribution
            # of the states (action 7), at the USD rate of their combination
            known &= has_no_negative_state(future, configuration)
        contract = contract_rule(future[discount], configuration)
        future.loc[known, "uplift"] = contract
        future.loc[known, "uplift_bajo"] = contract
        future.loc[known, "uplift_alto"] = contract
        future.loc[known, "via_uplift"] = PATH_CONTRACT
        future.loc[known, "origen_uplift"] = UPLIFT_ORIGIN_CONTRACT
        future.loc[known, "factor_subida"] = 1.0
        future.loc[known, "factor_estado"] = 1.0
    future["uplift_bajo"] = np.minimum(future["uplift_bajo"], future["uplift"])
    future["uplift_alto"] = np.maximum(future["uplift_alto"], future["uplift"])
    return future


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TOTALS AND THE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════

def totals(future: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config, series_exam: dict = None,
           time_series_universe: TimeSeriesUniverse = None, price_error: float = 0.0,
           acquisition_exam: dict = None) -> tuple:
    """By month (expected, split by origin of the pipeline, the time_series universe, the band of the exam and the
    linear and quadrature references) and by year (real renewals and revenue of the closed months + expected by origin
    + the time_series universe, with the band of the exam)."""
    period_column = configuration.period_col
    future = future.assign(_media_baja=(future["esperado_usd"] - future["esperado_usd_bajo"]) ** 2,
                           _media_alta=(future["esperado_usd_alto"] - future["esperado_usd"]) ** 2)
    grouped = future.groupby(period_column)
    by_month = pd.DataFrame({"usd_vence": grouped[configuration.pipeline_usd_col].sum(),
                             "esperado_usd": grouped["esperado_usd"].sum(),
                             "banda_lineal_baja": grouped["esperado_usd_bajo"].sum(),
                             "banda_lineal_alta": grouped["esperado_usd_alto"].sum(),
                             "_cuadratura_baja": grouped["_media_baja"].sum(), "_cuadratura_alta": grouped["_media_alta"].sum()})
    for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        by_month[f"esperado_{origin}"] = future[future[PIPELINE_ORIGIN_COLUMN] == origin].groupby(period_column)["esperado_usd"].sum()
    # the error of the volume of the simulated acquisitions, from their exam (step 5.1), row by row
    by_month["error_captacion_usd"] = (future["esperado_usd"] * acquisition_error_of_rows(future, acquisition_exam, configuration)
                                       ).groupby(future[period_column]).sum()

    # the time_series universe: its revenue, its re-entry (pipeline and expected) and its half band, per month
    universe_by_month = (time_series_universe.by_month() if time_series_universe is not None
                         else pd.DataFrame(columns=[period_column] + TIME_SERIES_MONTH_COLUMNS))
    by_month = by_month.join(universe_by_month.set_index(period_column)[TIME_SERIES_MONTH_COLUMNS], how="outer")
    by_month = by_month.fillna(0.0).astype(float).reset_index().rename(columns={"index": period_column})
    by_month["usd_vence_total"] = by_month["usd_vence"] + by_month["vence_ts_reentrada_usd"]
    by_month["total_usd"] = by_month["esperado_usd"] + by_month["ingresos_ts_usd"] + by_month["esperado_ts_reentrada"]
    ts_part = by_month["ingresos_ts_usd"] + by_month["esperado_ts_reentrada"]
    by_month["banda_lineal_baja"] = by_month["banda_lineal_baja"] + ts_part - by_month["error_ts_usd"]
    by_month["banda_lineal_alta"] = by_month["banda_lineal_alta"] + ts_part + by_month["error_ts_usd"]
    by_month["banda_cuadratura_baja"] = by_month["total_usd"] - np.sqrt(by_month["_cuadratura_baja"] + by_month["error_ts_usd"] ** 2)
    by_month["banda_cuadratura_alta"] = by_month["total_usd"] + np.sqrt(by_month["_cuadratura_alta"] + by_month["error_ts_usd"] ** 2)
    by_month["tasa_usd"] = by_month["esperado_usd"] / by_month["usd_vence"].where(by_month["usd_vence"] != 0)
    month_order = ([period_column, "usd_vence", "esperado_usd"]
                   + [f"esperado_{origin}" for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED)]
                   + ["error_captacion_usd"] + TIME_SERIES_MONTH_COLUMNS
                   + ["usd_vence_total", "total_usd", "banda_lineal_baja", "banda_lineal_alta",
                                                  "banda_cuadratura_baja", "banda_cuadratura_alta", "tasa_usd"])
    by_month = by_month[month_order]

    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    closed_by_year = closed.groupby(closed[period_column].map(lambda month: month.year))[configuration.renewed_usd_col].sum()
    ts_real_by_year = (time_series_universe.real_revenue_by_year() if time_series_universe is not None
                       else pd.Series(dtype=float))
    month_year = by_month[period_column].map(lambda month: month.year)
    year_of = future[period_column].map(lambda month: month.year)
    future_by_year = future.groupby(year_of)
    years = sorted(set(closed_by_year.index) | set(ts_real_by_year.index) | set(month_year))
    by_year = pd.DataFrame({"ano": years})
    by_year["renovado_real_usd"] = by_year["ano"].map(closed_by_year).fillna(0.0)
    for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        by_origin = future[future[PIPELINE_ORIGIN_COLUMN] == origin].groupby(year_of[future[PIPELINE_ORIGIN_COLUMN] == origin])["esperado_usd"].sum()
        by_year[f"esperado_{origin}_usd"] = by_year["ano"].map(by_origin).fillna(0.0)
    by_year["esperado_usd"] = by_year["ano"].map(future_by_year["esperado_usd"].sum()).fillna(0.0)
    months_of_year = by_month.groupby(month_year)
    by_year["ingresos_ts_real_usd"] = by_year["ano"].map(ts_real_by_year).fillna(0.0)
    by_year["ingresos_ts_usd"] = by_year["ano"].map(months_of_year["ingresos_ts_usd"].sum()).fillna(0.0)
    by_year["vence_ts_reentrada_usd"] = by_year["ano"].map(months_of_year["vence_ts_reentrada_usd"].sum()).fillna(0.0)
    by_year["esperado_ts_reentrada_usd"] = by_year["ano"].map(months_of_year["esperado_ts_reentrada"].sum()).fillna(0.0)
    by_year["total_usd"] = (by_year["renovado_real_usd"] + by_year["esperado_usd"] + by_year["ingresos_ts_real_usd"]
                            + by_year["ingresos_ts_usd"] + by_year["esperado_ts_reentrada_usd"])
    error_ts_of_year = by_year["ano"].map(months_of_year["error_ts_usd"].sum()).fillna(0.0)
    by_year["banda_cuadratura_usd"] = np.sqrt(by_year["ano"].map(future_by_year["_media_alta"].sum()).fillna(0.0)
                                              + error_ts_of_year ** 2)
    real_of_year = by_year["renovado_real_usd"] + by_year["ingresos_ts_real_usd"]
    by_year["banda_lineal_baja"] = real_of_year + by_year["ano"].map(months_of_year["banda_lineal_baja"].sum()).fillna(0.0)
    by_year["banda_lineal_alta"] = real_of_year + by_year["ano"].map(months_of_year["banda_lineal_alta"].sum()).fillna(0.0)
    if series_exam is not None:
        by_month, by_year = add_the_exam_band(by_month, by_year, series_exam, configuration, price_error)
    return by_month, by_year


def acquisition_error_of_rows(future: pd.DataFrame, acquisition_exam: dict, configuration: Config) -> np.ndarray:
    """The error of the volume of every simulated acquisition row (0 for the other rows): the error of the USD sold that
    the exam of the acquisitions (step 5.1) measured for its acquisition value at its horizon of sale (h = the month it
    is sold − the current month: 0 for the current month), interpolated and held beyond the last one examined; the mean
    over the acquisition values for a value the exam did not see."""
    error = np.zeros(len(future))
    summary = acquisition_exam["summary"] if acquisition_exam is not None else pd.DataFrame()
    simulated = (future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_SIMULATED).to_numpy()
    if summary.empty or not simulated.any():
        return error
    acquisition = configuration.acquisition_column
    current = configuration.calendar_boundaries()["current"]
    sale_horizon = np.array([(month - configuration.one_year_term_months - current).n
                             for month in future.loc[simulated, configuration.period_col]], dtype=float)
    every_value = summary.groupby("h")["error_usd"].mean()
    values_of_rows = future.loc[simulated, acquisition].to_numpy() if acquisition in future.columns else [None] * int(simulated.sum())
    row_errors = np.empty(len(sale_horizon))
    for position, (value, horizon) in enumerate(zip(values_of_rows, sale_horizon)):
        of_value = summary[summary[acquisition] == value].set_index("h")["error_usd"].sort_index()
        curve = of_value if len(of_value) else every_value
        row_errors[position] = np.interp(horizon, curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float))
    error[simulated] = row_errors
    return error


# the columns the time_series universe adds to every month (TimeSeriesUniverse.by_month)
MONTHS_PER_YEAR = 12
TIME_SERIES_MONTH_COLUMNS = ["ingresos_ts_usd", "vence_ts_reentrada_usd", "esperado_ts_reentrada", "error_ts_usd"]


def exam_error_by_horizon(series_exam: dict) -> pd.Series:
    """The typical error of the total of a month in the exam (step 7.1), per horizon: the mean |error| of the monthly
    total, with the customer states predicted too (framework_honesto) when the extract dates them."""
    summary = series_exam["summary"]
    method = "framework_honesto" if (summary["metodo"] == "framework_honesto").any() else METHOD_FRAMEWORK
    return summary[summary["metodo"] == method].set_index("h")["error_total_medio"].sort_index()


def add_the_exam_band(by_month: pd.DataFrame, by_year: pd.DataFrame, series_exam: dict, configuration: Config,
                      price_error: float = 0.0) -> tuple:
    """THE OFFICIAL BAND OF THE TOTAL: every error that was MEASURED on closed months, at the horizon of every month.

    RULES (h = the month − the last closed month; every error interpolated between the horizons examined and held
    beyond the last one: a lower bound there):
      · the rate (exam of step 7.1, error_examen_pct): × the expected renewals of the extract and of the simulated
        acquisitions; a projected renewal (a renewal of a renewal expected in the window) carries the error of the
        month its volume was predicted in too: √(error(h)² + error(h − 12)²)
      · the price (exam of the uplift, step 3.3, error_precio_pct): the error of the renewed USD of the total of a
        month with the paths the forecast uses, × every expected renewal
      · the rate and the price of the same rows are different errors: in quadrature
      · the volume of the simulated acquisitions (exam of step 5.1, error_captacion_usd) and the time_series universe
        (its exam, step 6.1, error_ts_usd): other parts of the total, added
      error_examen_usd = √(rate² + price²) + acquisitions + time_series. By year: the errors of its months ADDED (the
      exam shows the months of a year err together). The linear band (every row in the same direction) and the
      quadrature one (rows independent) stay as references: they are not the error of a total."""
    errors = exam_error_by_horizon(series_exam)
    if errors.empty:
        return by_month, by_year
    period = configuration.period_col
    last_closed = configuration.calendar_boundaries()["current"] - 1
    horizons = by_month[period].map(lambda month: (month - last_closed).n).to_numpy(dtype=float)
    examined_h, examined_error = errors.index.to_numpy(dtype=float), errors.to_numpy(dtype=float)
    by_month["error_examen_pct"] = np.interp(horizons, examined_h, examined_error)
    volume_error = np.where(horizons > MONTHS_PER_YEAR, np.interp(horizons - MONTHS_PER_YEAR, examined_h, examined_error), 0.0)
    projected_error = np.sqrt(by_month["error_examen_pct"] ** 2 + volume_error ** 2)
    by_month["error_tasa_usd"] = (by_month["error_examen_pct"] * (by_month["esperado_usd"] - by_month[f"esperado_{PIPELINE_PROJECTED}"])
                                  + projected_error * by_month[f"esperado_{PIPELINE_PROJECTED}"])
    by_month["error_precio_pct"] = float(price_error)
    by_month["error_precio_usd"] = by_month["error_precio_pct"] * by_month["esperado_usd"]
    by_month["error_examen_usd"] = (np.sqrt(by_month["error_tasa_usd"] ** 2 + by_month["error_precio_usd"] ** 2)
                                    + by_month["error_captacion_usd"] + by_month["error_ts_usd"])
    by_month["banda_examen_baja"] = by_month["total_usd"] - by_month["error_examen_usd"]
    by_month["banda_examen_alta"] = by_month["total_usd"] + by_month["error_examen_usd"]
    year_of_month = by_month[period].map(lambda month: month.year)
    error_of_year = by_month.groupby(year_of_month)["error_examen_usd"].sum()
    by_year["error_examen_usd"] = by_year["ano"].map(error_of_year).fillna(0.0)
    # what the band of every year is made of (its months added, as the band): the rate and the price together (in
    # quadrature, month by month), the simulated acquisitions and the time series universe; they add up to the band
    rate_and_price = np.sqrt(by_month["error_tasa_usd"] ** 2 + by_month["error_precio_usd"] ** 2)
    parts = pd.DataFrame({"banda_tasa_usd": by_month["error_tasa_usd"], "banda_precio_usd": by_month["error_precio_usd"],
                          "banda_tasa_y_precio_usd": rate_and_price, "banda_captacion_usd": by_month["error_captacion_usd"],
                          "banda_time_series_usd": by_month["error_ts_usd"]}).groupby(year_of_month).sum()
    for column_name in parts.columns:
        by_year[column_name] = by_year["ano"].map(parts[column_name]).fillna(0.0)
    by_year["banda_examen_baja"] = by_year["total_usd"] - by_year["error_examen_usd"]
    by_year["banda_examen_alta"] = by_year["total_usd"] + by_year["error_examen_usd"]
    return by_month, by_year


def check_forecast(future: pd.DataFrame, fine_table: pd.DataFrame, by_month: pd.DataFrame,
                   configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    missing = future[["tasa", "uplift", "esperado_usd"]].isna().any(axis=1)
    configuration.log_check(STEP_7_2_LABEL, check_log, "every future row has a rate, an uplift and an expected value",
                            not missing.any(), failure_detail=f"{int(missing.sum()):,} rows incomplete",
                            context=f"{len(future):,} rows")
    bad_band = ~(future["tasa"].between(0, 1) & (future["tasa_baja"] <= future["tasa"] + 1e-12)
                 & (future["tasa"] <= future["tasa_alta"] + 1e-12) & future["tasa_baja"].between(0, 1) & future["tasa_alta"].between(0, 1))
    configuration.log_check(STEP_7_2_LABEL, check_log, "every rate and band is inside [0, 1] and the band contains the rate",
                            not bad_band.any(), failure_detail=f"{int(bad_band.sum()):,} rows with a bad rate or band",
                            examples=future.loc[bad_band, ["tasa", "tasa_baja", "tasa_alta"]])
    future_roles = fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    configuration.log_check(STEP_7_2_LABEL, check_log, "no closed row is forecast; every future row of the extract is",
                            set(future.loc[(future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL) & (future[MATURATION_ROW_COLUMN] == 0), "_fila"])
                            == set(fine_table.index[future_roles]),
                            failure_detail="the forecast rows are not the future rows")
    real_rows = (future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL) & (future[MATURATION_ROW_COLUMN] == 0)
    due_difference = future.loc[real_rows, configuration.pipeline_usd_col].sum() - fine_table.loc[future_roles, configuration.pipeline_usd_col].sum()
    configuration.log_check(STEP_7_2_LABEL, check_log, "the pipeline of the extract's future rows is conserved (Σ USD due)",
                            abs(due_difference) <= configuration.money_tolerance, failure_detail=f"difference ${due_difference:,.2f}",
                            context=f"${future.loc[real_rows, configuration.pipeline_usd_col].sum():,.0f} due in the extract")
    total_difference = by_month["esperado_usd"].sum() - future["esperado_usd"].sum()
    configuration.log_check(STEP_7_2_LABEL, check_log, "the totals are the sum of the rows", abs(total_difference) <= configuration.money_tolerance,
                            failure_detail=f"difference ${total_difference:,.2f}", context=f"${future['esperado_usd'].sum():,.0f} expected")


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 7.3 · TOTAL  (until 25-oct-2026: step_21_time_series.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE TOTAL of the forecast, by year and origin, with the time_series universe in it.
#
# The time_series universe (retail to subscription) is BUILT in step 6.1 (its box, TimeSeriesUniverse in
# time_series_universe.py, by region, with its exam) and ADDED by the forecast to its months and its band (step 7.2).
# This step gives the answer of every year separable by origin: what was renewed, what is expected and what the
# universe brings, and checks that it is the same answer as the forecast's.
#
# THE ORIGINS (units and USD due, the pipeline; and renewed, renewals or revenue):
#   pipeline_renovado_real   renewals booked in the closed months (normal universe)
#   pipeline_real_esperado   expected renewals of the extract's future pipeline
#   pipeline_proyectada      expected renewals of the extended re-entries
#   pipeline_simulada        expected renewals of the simulated acquisition
#   ts_real                  revenue of the time_series universe in the closed months of this year
#   ts_proyectado            its simulated revenue of the rest of this year
#   ts_simulado              its simulated revenue of the whole of next year
#   ts_reentrada             the projected months of this year falling due next year: pipeline (40 % off) and
#                            expected renewals (renewed at 100 %, at the region's rate)
#   TOTAL                    every year: the sum of its origins
#   Total this year = pipeline + ts_real + ts_proyectado · Total next year = forecast + ts_simulado + ts_reentrada
#
# Actions (logged as they are done):
#   1. the rows of the time_series universe (from the box of step 6.1) by origin and year
#   2. the total by year and origin
#   3. check the total                                                   checks 1-2
#   4. write the total (the time_series rows reach sff_nucleo)           check 3
#   5. count the checks; stop if any failed
#   6. show the total by year × origin
#
# Checks (logged as they are made, numbered, at the level of their status):
#   1. the origins of every year add up to its total
#   2. the TOTAL of every year = total_usd of the forecast by year (sff_resumen_negocio): the same answer
#   3. table sff_forecast_total written and read back
#
# Output: the time_series rows (region × month, in memory: they reach sff_nucleo) and the total (year × origin) ·
# table sff_forecast_total.

STEP_7_3_LABEL = "7.3"
STEP_7_3_NAME = "TOTAL"
STEP_7_3_PURPOSE = ("give the total of every year separable by origin, with the time_series universe (built in step 6.1, "
                "added to the months of the forecast in step 7.2) in it, and check it is the forecast's answer")
STEP_7_3_ACTIONS = ["the rows of the time_series universe by origin and year",
                "the total by year and origin",
                "check the total (checks 1-2)",
                "write the total (check 3; the time_series rows reach sff_nucleo)",
                "count the checks; stop if any failed",
                "show the total by year × origin"]
STEP_7_3_OUTPUT = "region × month rows of the time_series universe · year × origin total · table sff_forecast_total"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP
# ═══════════════════════════════════════════════════════════════════════════════════

def build_time_series_and_total(time_series_universe: TimeSeriesUniverse, fine_table: pd.DataFrame, forecast: dict,
                                configuration: Config) -> tuple:
    """The total by year and origin, with the time_series universe.

    INPUT:   the time_series universe (box of step 6.1) · the fine table (the booked renewals) · the forecast of step 7.2
             (its rows and its answer by year) · the Config.
    OUTPUT:  (time_series table: region × month; total table: year × origin).
    EDGE CASES: no time_series row → only the total of the normal universe."""
    configuration.log_step_start(STEP_7_3_LABEL, STEP_7_3_NAME, STEP_7_3_PURPOSE, STEP_7_3_ACTIONS, STEP_7_3_OUTPUT)
    check_log = []
    period = configuration.period_col
    current = configuration.calendar_boundaries()["current"]
    time_series_table = time_series_universe.table if time_series_universe is not None else pd.DataFrame(
        columns=[period, "origen", "unidades", "valor", "revenue", "unidades_renovadas"])

    # [1] the rows of the universe
    if len(time_series_table):
        by_origin = (time_series_table.assign(ano=time_series_table[period].map(lambda month: month.year))
                     .groupby(["origen", "ano"])["revenue"].agg(["size", "sum"]))
        configuration.log_action(STEP_7_3_LABEL, 1, " · ".join(f"{origin} {year}: {int(row['size']):,} rows ${row['sum']:,.0f}"
                                                           for (origin, year), row in by_origin.iterrows()))
    else:
        configuration.log_action(STEP_7_3_LABEL, 1, "no time_series row: the total is the one of the normal universe")

    # [2] the total by year and origin
    total = forecast_total(fine_table, forecast["forecast"], time_series_table, current, configuration)
    configuration.log_action(STEP_7_3_LABEL, 2, "total by year and origin: " + " · ".join(
        f"{int(row['ano'])} ${row['usd_renovado']:,.0f}" for _, row in total[total["origen"] == TOTAL_ORIGIN_TOTAL].iterrows()))

    # [3] the checks
    configuration.log_action(STEP_7_3_LABEL, 3, "checking the total")
    check_the_total(total, forecast["by_year"], current, configuration, check_log)

    # [4] the table
    configuration.log_action(STEP_7_3_LABEL, 4, "writing the total")
    configuration.write_table(STEP_7_3_LABEL, check_log, total, TABLE_FORECAST_TOTAL)

    # [5] the count of the checks; stop if anything failed
    configuration.log_action(STEP_7_3_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_7_3_LABEL, STEP_7_3_NAME, check_log)

    # [6] the total
    configuration.log_action(STEP_7_3_LABEL, 6, "the total by year × origin (usd_vence: pipeline; usd_renovado: renewals / revenue):")
    configuration.show_table(total)
    configuration.show_query(STEP_7_3_LABEL, "the total by year and origin (sff_forecast_total)", forecast_total_query(configuration))
    return time_series_table, total


def forecast_total(fine_table: pd.DataFrame, forecast_rows: pd.DataFrame, time_series_table: pd.DataFrame, current,
                   configuration: Config) -> pd.DataFrame:
    """By year × origin: units and USD due (pipeline) and renewed (renewals or revenue); plus TOTAL per year."""
    period = configuration.period_col
    year_of = lambda frame: frame[period].map(lambda month: month.year)
    rows = []
    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    for year, block in closed.groupby(year_of(closed)):
        rows.append({"ano": year, "origen": TOTAL_ORIGIN_RENEWED, "unidades_vencen": block[configuration.pipeline_units_col].sum(),
                     "usd_vence": block[configuration.pipeline_usd_col].sum(),
                     "unidades_renovadas": block[configuration.renewed_units_col].sum(),
                     "usd_renovado": block[configuration.renewed_usd_col].sum()})
    origin_of = {PIPELINE_REAL: TOTAL_ORIGIN_EXPECTED, PIPELINE_PROJECTED: TOTAL_ORIGIN_PROJECTED,
                 PIPELINE_SIMULATED: TOTAL_ORIGIN_SIMULATED}
    for (year, origin), block in forecast_rows.groupby([year_of(forecast_rows), PIPELINE_ORIGIN_COLUMN]):
        rows.append({"ano": year, "origen": origin_of.get(origin, origin), "unidades_vencen": block[configuration.pipeline_units_col].sum(),
                     "usd_vence": block[configuration.pipeline_usd_col].sum(), "unidades_renovadas": block["esperado_unidades"].sum(),
                     "usd_renovado": block["esperado_usd"].sum()})
    for (year, origin), block in time_series_table.groupby([year_of(time_series_table), "origen"]):
        is_reentry = origin == TS_REENTRY
        rows.append({"ano": year, "origen": origin,
                     "unidades_vencen": block["unidades"].sum() if is_reentry else 0.0,
                     "usd_vence": block["valor"].sum() if is_reentry else 0.0,
                     "unidades_renovadas": block["unidades_renovadas"].sum() if is_reentry else block["unidades"].sum(),
                     "usd_renovado": block["revenue"].sum()})
    by_origin = pd.DataFrame(rows)
    # the years answered: the current one and the ones after it
    by_origin = by_origin[by_origin["ano"] >= current.year]
    totals = by_origin.groupby("ano")[["unidades_vencen", "usd_vence", "unidades_renovadas", "usd_renovado"]].sum().reset_index()
    totals["origen"] = TOTAL_ORIGIN_TOTAL
    return pd.concat([by_origin, totals], ignore_index=True).sort_values(["ano", "origen"]).reset_index(drop=True)


def check_the_total(total: pd.DataFrame, by_year: pd.DataFrame, current, configuration: Config, check_log: list) -> None:
    """Checks 1 and 2."""
    # [1] the origins add up to the total of every year
    parts = total[total["origen"] != TOTAL_ORIGIN_TOTAL].groupby("ano")["usd_renovado"].sum()
    totals = total[total["origen"] == TOTAL_ORIGIN_TOTAL].set_index("ano")["usd_renovado"]
    differences = (parts - totals).abs()
    configuration.log_check(STEP_7_3_LABEL, check_log, "the origins of every year add up to its total",
                            bool((differences <= configuration.money_tolerance).all()),
                            failure_detail=f"differences: {differences.to_dict()}", context=f"{len(totals)} years")
    # [2] the same answer as the forecast by year
    answered = by_year[by_year["ano"] >= current.year].set_index("ano")["total_usd"]
    compared = pd.concat([totals.rename("total"), answered.rename("forecast")], axis=1).fillna(0.0)
    differences = (compared["total"] - compared["forecast"]).abs()
    configuration.log_check(STEP_7_3_LABEL, check_log,
                            "the TOTAL of every year = total_usd of the forecast by year (the same answer)",
                            bool((differences <= configuration.money_tolerance).all()),
                            failure_detail=" · ".join(f"{year}: total ${row['total']:,.2f} vs forecast ${row['forecast']:,.2f}"
                                                      for year, row in compared[differences > configuration.money_tolerance].iterrows()),
                            context=" · ".join(f"{year} ${value:,.0f}" for year, value in totals.items()))


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 7 · THE BLOCK: run the steps in order, keep the results
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B7"
BLOCK_NAME = "EXAM AND ASSEMBLY"
BLOCK_MISSION = ("one forecast in money from everything blocks 2 to 6 predict, with an official band measured on "
                 "closed months and a total that is the same answer whichever way it is added")
BLOCK_READS = ("fine_table", "forecast_units", "rated_units", "series_estimate", "pool_series", "pool_reference",
               "backtest", "ladder", "rate_box", "uplift_cells", "uplift_state_factors", "uplift_verdict",
               "price_increases", "state_maturation", "future_pipeline", "time_series_universe")
BLOCK_GIVES = ("series_exam", "forecast", "time_series", "forecast_total")


class ExamAndAssemblyBlock:
    """BLOCK 7 · EXAM AND ASSEMBLY (see the header of this module): its results are attributes, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.series_exam = None
        self.forecast = None
        self.time_series = None
        self.forecast_total = None
        self.seconds = {}

    def run(self, tables: dict) -> "ExamAndAssemblyBlock":
        """Every step of the block, in order. tables: the results of blocks 1 to 6, by name (BLOCK_READS)."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 7, BLOCK_NAME, BLOCK_MISSION)

        started = time.time()
        self.series_exam = examine_series(tables["rated_units"], tables["series_estimate"], tables["pool_series"],
                                          tables["backtest"], configuration,
                                          reference_members=tables["ladder"]["reference_members"],
                                          fine_table=tables["fine_table"], rate_box=tables["rate_box"])
        self.seconds["7.1 series_exam"] = time.time() - started

        started = time.time()
        self.forecast = assemble_forecast(tables["fine_table"], tables["forecast_units"], tables["series_estimate"],
                                          tables["pool_series"], tables["pool_reference"], tables["backtest"],
                                          tables["uplift_cells"], tables["uplift_verdict"], tables["rated_units"],
                                          tables["price_increases"], configuration,
                                          uplift_state_factors=tables["uplift_state_factors"],
                                          state_maturation=tables["state_maturation"], rate_box=tables["rate_box"],
                                          series_exam=self.series_exam,
                                          future_pipeline=tables["future_pipeline"]["box"],
                                          time_series_universe=tables["time_series_universe"],
                                          acquisition_exam=tables["future_pipeline"]["exam"])
        self.seconds["7.2 forecast"] = time.time() - started

        started = time.time()
        self.time_series, self.forecast_total = build_time_series_and_total(tables["time_series_universe"],
                                                                            tables["fine_table"], self.forecast,
                                                                            configuration)
        self.seconds["7.3 total"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
