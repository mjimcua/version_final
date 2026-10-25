"""
step_19_series_exam.py — The exam of every forecast series: how much better the framework predicts than
the BASELINE (the calculation anyone could make) and than the forecast series alone (raw).

In the exam months, with only what was known h months before (origin T − h), every forecast series with
something due is predicted by four methods, on the same rows. They are the steps of a cascade: each one adds
one piece of the framework to the one before.

  baseline     THE reference: the rate of its mandatory cell over the last baseline_months closed months known
               at the origin (the portfolio's if the cell has none), × what falls due. No ladder, no credibility,
               no technique, no states. Every gain of the framework is measured against it, in USD.
  raw          the forecast series alone, with its own history (no ladder): the technique of its composition on
               its own monthly rates (the default technique if it cannot; its own level if it has fewer than 3
               months; its mandatory cell if it has none). Not a step of the cascade: what a series says alone
  por_defecto  the ladder and the credibility, with the DEFAULT technique for every composition: what the
               support of the relatives is worth, before choosing techniques
  framework    THE BOX OF THE RATE (step_13_rate_box.py), the same the forecast uses, learned at the origin: the
               technique CHOSEN for its composition and band, moved toward its credibility reference by (1 − z) × the
               difference of levels known at the origin; a series with no composition: its cell with its customer
               states
With maturation_dims, the honest exam of the states adds the last step for the total (honesto).

MONEY: every prediction is also in USD, predicted units × the price its real renewals had (renewed USD /
renewed units of the series and month; without renewals, the AUV due × the uplift of the exam month). The price
is the same for every method, so the USD differences are the differences of the rate: what predicting the rate
better is worth in money. The gain of a method = the baseline's mean |error| of the monthly total in USD minus
its own; and its consistency = the share of exam months × horizons in which it errs less than the baseline.

WHAT IS NOT OUT OF SAMPLE: the structure (the ladder, the credibility k, the gates, the technique chosen per band
and the error bands) is estimated with every closed month, the exam months included; for h ≥ 6 the choice of the
technique has seen months after the origin. The framework's exam is therefore somewhat optimistic; the baseline
and the series alone have no such advantage.

Every prediction also carries the BINOMIAL NOISE of the real rate it tries to hit, √(p(1−p)/units due)
with p its predicted rate: the error that even a perfect prediction would make (the noise rule: an error
of the size of the noise is the limit, not a failure). error_over_noise = RMS error / RMS noise: about 1,
the prediction is at the limit; well above 1, something knowable is missing; well below 1, suspicious.

Every prediction keeps its steps (composition rate, shift, series rate) and, for raw and framework,
its INTERVAL, built as the forecast builds its band (prediction.py): the error quantiles of the
technique × the binomial error with the series' own units due. A prediction is IN THE INTERVAL when
the real rate falls inside it. Predicted units = rate × the series' own units due.

Outputs:
  sff_series_exam_detail      forecast series × exam month × h × method: every prediction, traced
  sff_series_exam             forecast series: per method, predictions, in the interval, units
                              predicted and real, error, coverage (the improvement over the raw)
  sff_examen_cartera(_resumen) the portfolio: the error of the total per exam month and method

Actions (logged as they are done):
  1. the exam: the calendar, the forecast series with something due in the exam months
  2. the four methods for every series, month and horizon (with the interval of raw, por_defecto and framework)
  3. the errors: per forecast series, and of the total per month
  4. check the exam                                               checks 1-4
  5. write the exam                                               checks 5-8
  6. count the checks; stop if any failed
  7. show the methods against each other, as tables

Checks (logged as they are made, numbered, at the level of their status):
   1. no prediction used a month after its origin
   2. every method predicted every series of every exam month
   3. the real renewals of the exam equal the renewals of the exam months (nothing counted twice)
   4. the summary per series adds up to the detail (counts and units)
   5-8. tables sff_series_exam_detail, sff_series_exam, sff_examen_cartera, sff_examen_cartera_resumen written
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, METHOD_BASELINE, METHOD_DEFAULT,
                    METHOD_FRAMEWORK, RATE_COLUMN, ROLE_TEST, SERIES_ID_COLUMN, SYNTHETIC_COLUMN, TABLE_PORTFOLIO_EXAM,
                    TABLE_PORTFOLIO_EXAM_SUMMARY, TABLE_SERIES_EXAM, TABLE_SERIES_EXAM_DETAIL, TRUTH_ROLES,
                    join_columns)
from prediction import index_history, months_known, predict_composition
from step_13_rate_box import CHOSEN_MODE, DEFAULT_MODE, RateBox
from report_queries import exam_by_month_query, exam_summary_query
from step_12_backtest import band_of_horizon
from techniques import CATALOGUE, logit
from step_17_customer_states import honest_total_exam


STEP_LABEL = "19"
STEP_NAME = "EXAM OF EVERY FORECAST SERIES"
STEP_PURPOSE = ("predict every forecast series in the exam months, with only what was known h months before, four ways "
                "on the same rows: the baseline (mandatory cell, last months), alone with its own history (raw), the "
                "ladder with the default technique, and the framework; every prediction traced, in units and USD, so the "
                "gain over the baseline can be measured in money, month by month, and summed")
STEP_ACTIONS = ["the exam: the calendar, the forecast series with something due in the exam months",
                "the four methods for every series, month and horizon (with the interval of raw, por_defecto and framework)",
                "the errors: per forecast series, and of the total per month",
                "check the exam (checks 1-4)",
                "write the exam (checks 5-8)",
                "count the checks; stop if any failed",
                "show the methods against each other, as tables"]
STEP_OUTPUT = ("every prediction of every forecast series, traced · the exam per series and method · the error of the total "
               "per month · tables sff_series_exam_detail, sff_series_exam, sff_examen_cartera, sff_examen_cartera_resumen")

METHOD_RAW = "raw"
METHODS_IN_ORDER = [METHOD_BASELINE, METHOD_RAW, METHOD_DEFAULT, METHOD_FRAMEWORK]      # the order of the cascade (raw apart)


def examine_series(rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                   backtest: dict, configuration: Config, reference_members: pd.DataFrame = None,
                   fine_table: pd.DataFrame = None, rate_box: RateBox = None) -> dict:
    """The exam of every forecast series by the four methods.

    INPUT:   rated_units (step 06) · series_estimate (11: composition, z, reference) · the composition series
             (12) · the backtest (14: decision and error bands) · the members of every credibility reference (10).
    OUTPUT:  dict(detail, per_series, by_month, summary).
    RULES:   see the module header.
    EDGE CASES: a series with no composition (solo_historia, mixed) is predicted by its mandatory cell in the
             framework; its raw method still uses its own history. The baseline has no interval.
    """
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
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
    configuration.log_action(STEP_LABEL, 1, f"exam months {exam_months[0]}..{exam_months[-1]} · {real[SERIES_ID_COLUMN].nunique():,} "
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
    configuration.log_action(STEP_LABEL, 2, f"{len(detail):,} predictions · methods {methods}")

    # [3] the errors
    per_series = errors_per_series(detail, configuration)
    by_month, summary = errors_of_the_total(detail, methods, configuration)
    honest = None
    if configuration.maturation_dims and fine_table is not None:
        honest = honest_total_exam(detail[detail["method"] == METHOD_FRAMEWORK], fine_table, configuration)
        by_month, summary = add_the_honest_exam(by_month, summary, honest)
    configuration.log_action(STEP_LABEL, 3, "per method, all series: " + " · ".join(
        f"{method} WAPE {row['wape']:.1%}" + (f", {row['in_interval']:.0%} in the interval" if pd.notna(row["in_interval"]) else "")
        for method, row in overall_by_method(detail, configuration).iterrows()))

    # [4] the checks
    configuration.log_action(STEP_LABEL, 4, "checking the exam")
    peeked = [origin for used, composition_used, origin in latest_used
              if (pd.notna(used) and used > origin) or (pd.notna(composition_used) and composition_used > origin)]
    configuration.log_check(STEP_LABEL, check_log, "no prediction used a month after its origin", not peeked,
                            failure_detail=f"{len(peeked)} exams used later months", context=f"{len(latest_used)} month × horizon exams")
    expected_rows = len(real) * len(configuration.backtest_horizons)
    per_method = detail.groupby("method").size()
    configuration.log_check(STEP_LABEL, check_log, "every method predicted every series of every exam month",
                            bool((per_method == expected_rows).all()) and detail["pred_rate"].notna().all(),
                            failure_detail=f"rows per method {per_method.to_dict()} for {expected_rows:,}",
                            context=f"{len(methods)} methods × {expected_rows:,}")
    real_total = real[renewed].sum() * len(configuration.backtest_horizons)
    adds_up = all(abs(detail.loc[detail["method"] == method, "real_units"].sum() - real_total) <= configuration.units_tolerance for method in methods)
    configuration.log_check(STEP_LABEL, check_log, "the real renewals of the exam equal the renewals of the exam months (nothing counted twice)",
                            adds_up, failure_detail="a method lost or duplicated renewals",
                            context=f"{real[renewed].sum():,.0f} renewed units in the exam months")
    framework = detail[detail["method"] == METHOD_FRAMEWORK]
    consistent = (int(per_series["framework_predictions"].sum()) == len(framework)
                  and abs(per_series["framework_real_units"].sum() - framework["real_units"].sum()) <= configuration.units_tolerance)
    configuration.log_check(STEP_LABEL, check_log, "the summary per series adds up to the detail (counts and units)", consistent,
                            failure_detail="sff_series_exam does not add up to sff_series_exam_detail")

    # [5] the tables
    configuration.log_action(STEP_LABEL, 5, "writing the exam")
    configuration.write_table(STEP_LABEL, check_log, detail, TABLE_SERIES_EXAM_DETAIL)
    configuration.write_table(STEP_LABEL, check_log, per_series, TABLE_SERIES_EXAM)
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_PORTFOLIO_EXAM)
    configuration.write_table(STEP_LABEL, check_log, summary, TABLE_PORTFOLIO_EXAM_SUMMARY)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [7] the methods against each other
    configuration.log_action(STEP_LABEL, 7, "the methods on the same rows (error_total: of the sum of the portfolio; wape_series: "
                                            "series by series, no compensation):")
    configuration.show_table(summary)
    show_the_cascade(summary, configuration)
    if honest is not None:
        show_the_honest_exam(honest, configuration)
    configuration.show_query(STEP_LABEL, "the exam per method and horizon", exam_summary_query(configuration))
    configuration.show_query(STEP_LABEL, "the total of every exam month per method (sff_examen_cartera)",
                             exam_by_month_query(configuration, methods))
    return dict(detail=detail, per_series=per_series, by_month=by_month, summary=summary)


def show_the_cascade(summary: pd.DataFrame, configuration: Config) -> None:
    """The cascade on screen: from the baseline, what every piece of the framework is worth, per horizon: the mean
    |error| of the monthly total in USD, the gain over the baseline and in how many exam months it erred less."""
    doc = configuration.logger.doc
    cascade = summary[summary["metodo"].isin([METHOD_BASELINE, METHOD_DEFAULT, METHOD_FRAMEWORK])]
    if cascade.empty or METHOD_BASELINE not in set(cascade["metodo"]):
        return
    doc(f"[{STEP_LABEL}] ══════════ LA CASCADA: DE LA BASELINE AL FRAMEWORK ══════════")
    doc(f"[{STEP_LABEL}] baseline = su celda mandatory, últimos {configuration.baseline_months} meses · por_defecto = + escalera "
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
        doc(f"[{STEP_LABEL}] VALORACIÓN h = {horizon}: la baseline yerra el total de un mes en ${baseline['error_usd_medio']:,.0f} de media "
            f"({baseline['error_total_medio']:.1%}); el framework en ${framework['error_usd_medio']:,.0f} ({framework['error_total_medio']:.1%}): "
            + (f"${framework['ganancia_usd_mes']:,.0f} menos al mes" if framework['ganancia_usd_mes'] >= 0
               else f"${-framework['ganancia_usd_mes']:,.0f} MÁS al mes que la baseline")
            + f", mejor en el {consistent:.0%} de los meses de examen"
            + (" (consistente)" if consistent >= 0.75 else " (NO consistente: gana algunos meses y pierde otros)" if consistent >= 0.5
               else " (la baseline gana más meses)"))
    doc(f"[{STEP_LABEL}] ══════════ FIN DE LA CASCADA ══════════")


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
    doc(f"[{STEP_LABEL}] ══════════ EXAMEN HONESTO DE LOS ESTADOS DEL CLIENTE ({', '.join(configuration.maturation_dims)}) ══════════")
    doc(f"[{STEP_LABEL}] the total of every exam month predicted three ways with the same rates of the framework: with the "
        f"FINAL states (techo: what predicting the states perfectly would give, the exam of always), with the states as they "
        f"were h months before (estados_hoy) and with those plus the maturation curve known at the origin (honesto):")
    without_columns = [state for state, sign in configuration.structural_timevarying_dims.items()
                       if sign == "negative" and state not in configuration.maturation_dims]
    if without_columns:
        doc(f"[{STEP_LABEL}] OJO: {', '.join(without_columns)} no trae(n) columnas de maduración: aquí siguen con su estado "
            f"FINAL. El examen es honesto solo para {', '.join(configuration.maturation_dims)}; para los demás estados "
            f"sigue siendo el techo")
    error_columns = ["techo_estados_finales_error_total", "estados_hoy_error_total", "honesto_error_total"]
    shown = honest[["mes", "h", "renovadas_reales"] + error_columns].copy()
    shown[error_columns] = shown[error_columns].round(4)
    configuration.show_table(shown)
    per_horizon = honest.groupby("h")[error_columns].agg(lambda errors: errors.abs().mean())
    bias = honest.groupby("h")[error_columns].mean().add_suffix("_sesgo")
    configuration.show_table(per_horizon.join(bias).round(4).reset_index())
    for horizon, row in per_horizon.iterrows():
        doc(f"[{STEP_LABEL}] VALORACIÓN h = {horizon}: the error of the total is {row['honesto_error_total']:.1%} honest, against "
            f"{row['techo_estados_finales_error_total']:.1%} with the final states (the ceiling) and {row['estados_hoy_error_total']:.1%} "
            f"with the states seen and no curve: predicting the states is worth "
            f"{row['honesto_error_total'] - row['techo_estados_finales_error_total']:+.1%} of the total at this horizon")
    doc(f"[{STEP_LABEL}] ══════════ FIN DEL EXAMEN HONESTO ══════════")


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
