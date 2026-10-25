"""
step_20_forecast.py — THE FORECAST IN MONEY: the assembly. Every future row takes its rate, its uplift and its
customer states from the boxes of the stages before, and is valued: units × rate, USD × rate × uplift; then the totals
by month and by year, with the official band of the exam.

WHAT THIS STEP DOES NOT DO (each lives in its own stage, with its own exam):
  · the rate of a series at a horizon          → the box of the rate (step 13), examined by step 19
  · the uplift of a cell and of a state        → step 14, examined by step 16
  · the customer states at due date            → the box of the customer states (step 17), examined by step 19
  · the rows the extract cannot have yet       → the box of the future pipeline (step 18), with its exam
  · the error of a total                       → the exam of every forecast series (step 19)

THE FUTURE ROWS: the fine rows of the extract from the current month on (pipeline 'real') and the rows of the
SIMULATION WINDOW (current month … simulation_end) that the box of step 18 gives: proyectada (the 1-year licences
due in the window renew as this forecast expects, due 12 months later) and simulada (the acquisitions of the window).
The time_series universe re-enters in step 21.

For every future row:
  · h = months from the last closed month to the row's month (1 = the first month to predict)
  · THE RATE: the box of step 13 (the technique chosen for its composition and horizon, moved toward its credibility
    reference; without composition, its mandatory cell with its customer states, then its combination of states, then
    the portfolio). Its band: the error quantiles of its technique at its horizon × the binomial error with the units
    due of its forecast unit; clipped to [0, 1].
  · THE UPLIFT of its cell: the CONTRACT rule 1 / (1 − discount) where the discount is known and the uplift backtest
    (step 16) chose it; the STATISTICAL uplift of its cell otherwise (with its bootstrap band), without the price
    increases (step 14): a row whose month is still inside the cycle of an increase of its price group takes it again
    (factor_subida); and × the price factor of its customer states when step 16 found it predicts better
    (factor_estado).
  · EXPECTED: renewed units = units × rate; renewed USD = USD × rate × uplift; the band of the USD from the bands of
    the rate and the uplift.
THE CUSTOMER STATES AT DUE DATE (the box of step 17). The future rows carry TODAY's states: a licence due in 10 months
has not had the time to stop using the product or to be softcancelled, and a projected or simulated row is born
neutral. The box says, per group (the series without its states), month and combination of states, how many units
there will be when they fall due. This step only VALUES what the box moves; the data does not change:
  · the forecast with TODAY's states is kept (esperado_*_hoy: units due of the row × its rate × its uplift)
  · a neutral row keeps (1 − l) of its units and USD due (vencen_*_al_vencer), l = the share of the neutral units of
    its group and month the box moves; what leaves, leaves at the row's average price (10 units of $10 = $100)
  · what arrives at a series with a state becomes a PREDICTION row (origen_fila maduracion, fila_maduracion = 1): its
    data columns at 0, its units and USD at due date = what it receives; ONE row per destination series, month and
    origin of the pipeline (aggregate_the_prediction_rows: exact, the uplift USD-weighted); it is predicted like any
    future row, with the rate and the uplift of ITS series; its discount is unknown (statistical uplift)
  · the forecast (esperado_*) = units and USD at due date × rate × uplift; maduracion_* = at due date − today
  The extract rows are matured first: their expected renewals at due date are the pipeline of the projected rows;
  then the projected and simulated rows (no maturation columns: the box gives them the historical final share).
  The identity box leaves every row as it is: the forecast is the one with today's states.
THE TOTALS by month and by year. THE OFFICIAL BAND is the exam's (error_examen_*, banda_examen_*): the error the exam
of step 19 measured on the total of a month at the same horizon (with the states predicted when the extract dates
them); by year, the errors of its months added (the months of a year err together). The LINEAR band (Σ of the row
bands: every row wrong in the same direction) and the QUADRATURE one (√Σ of the row half-widths²: rows independent)
stay as references: they are not the error of a total.

Actions (logged as they are done):
  1. the future rows and their horizon
  2. the rate of every future row from the box of the rate, with its technique
  3. where the rate of every future row comes from (pool, cell or global) and its band
  4. the uplift of every future row (contract or statistical) and its band
  5. the expected renewals with today's states and their band, row by row
  6. the customer states of the extract rows at due date (the box of step 17), valued
  7. the rows of the window (the box of step 18), predicted and matured; the confidence of every row
  8. the totals by month and by year, with the band of the exam
  9. check the forecast                                              checks 1-7
 10. write the months, the summary and the maturation                checks 8-10
 11. count the checks; stop if any failed
 12. show the forecast by month, the answers by year and the maturation, as tables; the detector of
     states above their final share (ESTADOS DEL CLIENTE POR ENCIMA DE SU PROPORCIÓN FINAL)

Checks (logged as they are made, numbered, at the level of their status):
   1. every future row has a rate, an uplift and an expected value
   2. every rate and every band is inside [0, 1] and the band contains the rate
   3. no closed row is forecast; every future row is
   4. the pipeline of the future rows is conserved (Σ USD due)
   5. the totals are the sum of the rows
   6. the maturation moves units, it does not create them (Σ units at due date = Σ units due; a neutral row keeps
      between 0 and all of its units)
   7. no negative state is TODAY above its final share by mark_excess_warning_pp or more in any future
      month (WARN: the extract sets the state too early, e.g. a fresh cohort set to dormant; the detector says
      which months and what it costs the forecast)
   8-10. tables sff_forecast_mes, sff_resumen_negocio, sff_forecast_maduracion written and read back

Output: (the forecast row by row, by month, by year) · tables sff_forecast_mes, sff_resumen_negocio,
sff_forecast_maduracion. The forecast row by row reaches sff_nucleo (step 24).
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, DUE_UNITS_AT_DUE_COLUMN, DUE_USD_AT_DUE_COLUMN,
                    GATE_LEVEL, MATURATION_ROW_COLUMN, METHOD_FRAMEWORK, PATH_CONTRACT, PATH_CONTRACT_NEUTRAL,
                    PATH_STATISTICAL, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED,
                    RATE_FROM_POOL, ROLE_PROJECTION, SERIES_ID_COLUMN, TABLE_BUSINESS_SUMMARY,
                    TABLE_FORECAST_MATURATION, TABLE_FORECAST_MONTH, TRUTH_ROLES, UNITS_AT_DUE_COLUMN,
                    UNITS_TODAY_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN, WITH_DATES_COLUMN, has_no_negative_state,
                    join_columns, state_combination)
from step_12_backtest import band_of_horizon
from step_14_uplift import contract_rule, state_factor_of_rows
from step_13_rate_box import RateBox
from step_18_future_pipeline import FuturePipeline
from step_17_customer_states import (StateMaturation, states_above_final, maturation_by_month, show_states_above_final,
                                show_maturation_by_month)
from price_increases import increase_factor, price_group_of


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_LABEL = "20"
STEP_NAME = "FORECAST"
STEP_PURPOSE = ("assemble the forecast in money: every future row (of the extract and of the future pipeline) takes its "
                "rate (box of step 13), its uplift (step 14) and its customer states at due date (box of step 17), and "
                "is valued; the totals by month and by year with the official band of the exam (step 19)")
STEP_ACTIONS = ["the future rows of the extract",
                "the rate of every future row from the box of the rate, with its technique",
                "where the rate of every future row comes from (pool, cell or global) and its band",
                "the uplift of every future row (contract or statistical) and its band",
                "the expected renewals with today's states and their band, row by row",
                "the customer states of the extract rows at due date (the box of step 17), valued",
                "the rows of the window (the box of step 18): predicted and matured; the confidence of every row",
                "the totals by month and by year, by origin of the pipeline, with the band of the exam",
                "check the forecast (checks 1-7)",
                "write the months, the summary and the maturation (checks 8-10)",
                "count the checks; stop if any failed",
                "show the forecast by month, the answers by year and the maturation, as tables"]
STEP_OUTPUT = "every future row with rate, uplift, expected USD and bands (in memory) · totals by month and year · three tables"



def assemble_forecast(fine_table: pd.DataFrame, forecast_units: pd.DataFrame, series_estimate: pd.DataFrame,
                      pool_series: pd.DataFrame, pool_reference: pd.DataFrame, backtest: dict, uplift_cells: pd.DataFrame,
                      uplift_verdict: dict, rated_units: pd.DataFrame, price_increases: pd.DataFrame,
                      configuration: Config, uplift_state_factors: pd.DataFrame = None,
                      state_maturation: StateMaturation = None, rate_box: RateBox = None,
                      series_exam: dict = None, future_pipeline: FuturePipeline = None) -> dict:
    """The forecast row by row (extract and future pipeline), by month and by year; checked and written."""
    configuration.log_step_start(STEP_LABEL, STEP_NAME, STEP_PURPOSE, STEP_ACTIONS, STEP_OUTPUT)
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
    configuration.log_action(STEP_LABEL, 1, f"{len(future):,} future rows in the extract, {future[period_column].min()}.."
                                            f"{future[period_column].max()}; ${future[configuration.pipeline_usd_col].sum():,.0f} due; "
                                            f"last closed month {last_closed}")

    # [2]-[5] rate, uplift and expected renewals of the extract's future rows, with TODAY's states
    future = predict_rows(future, context, configuration)
    configuration.log_action(STEP_LABEL, 2, f"rates of the estimation ids at h = 1..{future['h'].max()} (technique of each id and band)")
    configuration.log_action(STEP_LABEL, 3, f"rate origins: {future['origen_tasa'].value_counts().to_dict()}")
    with_increase = future[future["factor_subida"] != 1.0]
    configuration.log_action(STEP_LABEL, 4, f"uplift paths: {future['via_uplift'].value_counts().to_dict()} · "
                                            f"{len(with_increase):,} rows still inside the cycle of a price increase "
                                            f"(${with_increase[configuration.pipeline_usd_col].sum():,.0f} due)"
                                            + (f" · price factor of the states applied ({future['factor_estado'].ne(1.0).sum():,} rows)"
                                               if uplift_verdict.get("factor_estado") else ""))
    configuration.log_action(STEP_LABEL, 5, f"expected ${future['esperado_usd'].sum():,.0f} renewed of "
                                            f"${future[configuration.pipeline_usd_col].sum():,.0f} due in the extract, with today's states")

    # [6] the maturation of the states of the extract rows: the box moves the units, this step values them. First the
    #     extract: its expected renewals at due date are the pipeline of the projected rows
    future, extract_table = mature_and_value(future, box, context, configuration, with_dates=True)
    arrived = future[MATURATION_ROW_COLUMN] == 1
    configuration.log_action(STEP_LABEL, 6, f"maturation of the extract ({box.describe()}): "
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
    configuration.log_action(STEP_LABEL, 7, f"simulation window {window[0] if window else '—'}..{window[-1] if window else '—'} → "
                                            f"due {window[0] + 12 if window else '—'}..{window[-1] + 12 if window else '—'}: "
                                            f"{created[PIPELINE_ORIGIN_COLUMN].value_counts().to_dict() if len(created) else {}} rows · "
                                            f"${created[configuration.pipeline_usd_col].sum() if len(created) else 0:,.0f} due · "
                                            f"${extension['esperado_usd'].sum() if len(extension) else 0:,.0f} expected (matured)")

    # the confidence of every row: how sure the forecast is of its rate
    future["confidence"] = confidence_of_rows(future, context, configuration)
    configuration.log_action(STEP_LABEL, 7, "confidence (USD expected): " + " · ".join(
        f"{label} {share:.0%}" for label, share in
        (future.groupby("confidence")["esperado_usd"].sum() / max(future["esperado_usd"].sum(), 1e-9)).items()))

    # the maturation by month, over the whole future (extract and extension), and the detector
    box_tables = pd.concat([table for table in (extract_table, extension_table) if len(table)], ignore_index=True) \
        if len(extract_table) or len(extension_table) else extract_table
    maturation = maturation_by_month(future, box_tables, box, configuration)
    states_detector = states_above_final(maturation, box, configuration) if maturation is not None else None

    # [8] the totals
    by_month, by_year = totals(future, fine_table, configuration, series_exam)
    configuration.log_action(STEP_LABEL, 8, f"{len(by_month)} months · {len(by_year)} years")

    # [9] the checks
    configuration.log_action(STEP_LABEL, 9, "checking the forecast")
    check_forecast(future, fine_table, by_month, configuration, check_log)
    units_today = future[configuration.pipeline_units_col].sum()
    units_at_due = future[DUE_UNITS_AT_DUE_COLUMN].sum()
    usd_difference = future[DUE_USD_AT_DUE_COLUMN].sum() - future[configuration.pipeline_usd_col].sum()
    neutral_kept = future.loc[future[MATURATION_ROW_COLUMN] == 0, DUE_UNITS_AT_DUE_COLUMN] / future.loc[
        future[MATURATION_ROW_COLUMN] == 0, configuration.pipeline_units_col].where(lambda units: units > 0)
    configuration.log_check(STEP_LABEL, check_log,
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
        configuration.log_check(STEP_LABEL, check_log,
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
    configuration.log_action(STEP_LABEL, 10, "writing the months, the summary and the maturation")
    configuration.write_table(STEP_LABEL, check_log, by_month, TABLE_FORECAST_MONTH)
    configuration.write_table(STEP_LABEL, check_log, by_year, TABLE_BUSINESS_SUMMARY)
    if maturation is not None:
        configuration.write_table(STEP_LABEL, check_log, maturation, TABLE_FORECAST_MATURATION)

    # [09] the count of the checks; stop if anything failed
    configuration.log_action(STEP_LABEL, 11, "counting the checks")
    configuration.log_check_summary(STEP_LABEL, STEP_NAME, check_log)

    # [10] the forecast by month and the answers by year; the maturation next to it (label 17)
    configuration.log_action(STEP_LABEL, 12, "the forecast by month (banda_examen: the official band, the error of the exam "
                                             "at the same horizon; banda lineal and cuadratura: references, rows all wrong "
                                             "in the same direction or independent):")
    configuration.show_table(by_month)
    configuration.logger.doc(f"[{STEP_LABEL}] by year: renewed in the closed months + expected in the future ones (by origin "
                             f"of the pipeline):")
    configuration.show_table(by_year)
    if maturation is not None:
        show_maturation_by_month(maturation, box, configuration)
        show_states_above_final(states_detector, configuration)
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
    """The rate of every future row from THE BOX OF THE RATE (stage 13), learned with every closed month: its
    composition's technique moved toward its credibility reference, or the level of its cell with its customer
    states, or the portfolio's; and its band, with the units due of its forecast unit."""
    rates = rate_box.predict(future, last_closed)
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

    The statistical uplift of the cell is the one without price increases (step 14); a row whose month is
    still inside the cycle of an increase of its price group takes it again: uplift and band × factor_subida
    (1 elsewhere, and for the contract rule, which is the discount, not the tariff).
    When step 16 found that the price by states predicts better (uplift_verdict["factor_estado"]), the statistical
    uplift of a row is also × the factor of its combination of negative states (step 14: a softcancel renews with a
    retention offer, below its cell): factor_estado (1 elsewhere, and for the contract rule)."""
    cells = uplift_cells.set_index(UPLIFT_CELL_ID_COLUMN)
    statistical = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift"])
    # a cell never seen (an extended row with a new discount bucket): its mandatory cell, then the global uplift
    if statistical.isna().any():
        uplift_mandatory = configuration.uplift_mandatory_dims or configuration.business_mandatory_dims
        weights = uplift_cells["renovadores"].clip(lower=1e-9)
        by_mandatory = (uplift_cells.assign(_w=weights, _wu=weights * uplift_cells["uplift"])
                        .groupby("uplift_celda_mandatory_id")[["_w", "_wu"]].sum())
        mandatory_uplift = by_mandatory["_wu"] / by_mandatory["_w"]
        global_uplift = float((weights * uplift_cells["uplift"]).sum() / weights.sum())
        fallback = join_columns(future, list(uplift_mandatory)).map(mandatory_uplift).fillna(global_uplift)
        statistical = statistical.fillna(fallback)
    future = future.drop(columns=[column for column in ("uplift", "uplift_bajo", "uplift_alto", "via_uplift") if column in future.columns])
    future["uplift"] = statistical
    future["uplift_bajo"] = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift_bajo"]).fillna(statistical)
    future["uplift_alto"] = future[UPLIFT_CELL_ID_COLUMN].map(cells["uplift_alto"]).fillna(statistical)
    future["via_uplift"] = PATH_STATISTICAL
    # the price by states, when step 16 found it predicts better
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
        future.loc[known, "factor_subida"] = 1.0
        future.loc[known, "factor_estado"] = 1.0
    future["uplift_bajo"] = np.minimum(future["uplift_bajo"], future["uplift"])
    future["uplift_alto"] = np.maximum(future["uplift_alto"], future["uplift"])
    return future


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TOTALS AND THE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════

def totals(future: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config, series_exam: dict = None) -> tuple:
    """By month (expected, split by origin of the pipeline, the band of the exam and the linear and quadrature
    references) and by year (real renewals of the closed months + expected by origin, with the band of the exam)."""
    period_column = configuration.period_col
    future = future.assign(_media_baja=(future["esperado_usd"] - future["esperado_usd_bajo"]) ** 2,
                           _media_alta=(future["esperado_usd_alto"] - future["esperado_usd"]) ** 2)
    grouped = future.groupby(period_column)
    by_month = pd.DataFrame({"usd_vence": grouped[configuration.pipeline_usd_col].sum(),
                             "esperado_usd": grouped["esperado_usd"].sum(),
                             "banda_lineal_baja": grouped["esperado_usd_bajo"].sum(),
                             "banda_lineal_alta": grouped["esperado_usd_alto"].sum(),
                             "banda_cuadratura_baja": grouped["esperado_usd"].sum() - np.sqrt(grouped["_media_baja"].sum()),
                             "banda_cuadratura_alta": grouped["esperado_usd"].sum() + np.sqrt(grouped["_media_alta"].sum())})
    for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        by_month[f"esperado_{origin}"] = future[future[PIPELINE_ORIGIN_COLUMN] == origin].groupby(period_column)["esperado_usd"].sum()
    by_month = by_month.fillna(0.0).reset_index()
    by_month["tasa_usd"] = by_month["esperado_usd"] / by_month["usd_vence"]

    closed = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    closed_by_year = closed.groupby(closed[period_column].map(lambda month: month.year))[configuration.renewed_usd_col].sum()
    year_of = future[period_column].map(lambda month: month.year)
    future_by_year = future.groupby(year_of)
    years = sorted(set(closed_by_year.index) | set(future_by_year.groups))
    by_year = pd.DataFrame({"ano": years})
    by_year["renovado_real_usd"] = by_year["ano"].map(closed_by_year).fillna(0.0)
    for origin in (PIPELINE_REAL, PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        by_origin = future[future[PIPELINE_ORIGIN_COLUMN] == origin].groupby(year_of[future[PIPELINE_ORIGIN_COLUMN] == origin])["esperado_usd"].sum()
        by_year[f"esperado_{origin}_usd"] = by_year["ano"].map(by_origin).fillna(0.0)
    by_year["esperado_usd"] = by_year["ano"].map(future_by_year["esperado_usd"].sum()).fillna(0.0)
    by_year["total_usd"] = by_year["renovado_real_usd"] + by_year["esperado_usd"]
    by_year["banda_cuadratura_usd"] = by_year["ano"].map(np.sqrt(future_by_year["_media_alta"].sum())).fillna(0.0)
    by_year["banda_lineal_baja"] = by_year["renovado_real_usd"] + by_year["ano"].map(future_by_year["esperado_usd_bajo"].sum()).fillna(0.0)
    by_year["banda_lineal_alta"] = by_year["renovado_real_usd"] + by_year["ano"].map(future_by_year["esperado_usd_alto"].sum()).fillna(0.0)
    if series_exam is not None:
        by_month, by_year = add_the_exam_band(by_month, by_year, series_exam, configuration)
    return by_month, by_year


def exam_error_by_horizon(series_exam: dict) -> pd.Series:
    """The typical error of the total of a month in the exam (step 19), per horizon: the mean |error| of the monthly
    total, with the customer states predicted too (framework_honesto) when the extract dates them."""
    summary = series_exam["summary"]
    method = "framework_honesto" if (summary["metodo"] == "framework_honesto").any() else METHOD_FRAMEWORK
    return summary[summary["metodo"] == method].set_index("h")["error_total_medio"].sort_index()


def add_the_exam_band(by_month: pd.DataFrame, by_year: pd.DataFrame, series_exam: dict, configuration: Config) -> tuple:
    """THE OFFICIAL BAND OF THE TOTAL: the error the exam measured on the total of a month at the same horizon.

    RULES:   h = the month − the last closed month; its error = the exam's at that horizon, interpolated between the
             examined horizons and held beyond the last one (a lower bound there: further is likely worse). By month:
             expected ± error × expected. By year: the errors of its months ADDED (in the same direction): the
             exam shows the months of a year err together, not independently. The linear band (every row in the same
             direction) and the quadrature one (rows independent) stay as references: they are not the error of a
             total."""
    errors = exam_error_by_horizon(series_exam)
    if errors.empty:
        return by_month, by_year
    period = configuration.period_col
    last_closed = configuration.calendar_boundaries()["current"] - 1
    horizons = by_month[period].map(lambda month: (month - last_closed).n).to_numpy(dtype=float)
    by_month["error_examen_pct"] = np.interp(horizons, errors.index.to_numpy(dtype=float), errors.to_numpy(dtype=float))
    by_month["error_examen_usd"] = by_month["error_examen_pct"] * by_month["esperado_usd"]
    by_month["banda_examen_baja"] = by_month["esperado_usd"] - by_month["error_examen_usd"]
    by_month["banda_examen_alta"] = by_month["esperado_usd"] + by_month["error_examen_usd"]
    error_of_year = by_month.groupby(by_month[period].map(lambda month: month.year))["error_examen_usd"].sum()
    by_year["error_examen_usd"] = by_year["ano"].map(error_of_year).fillna(0.0)
    by_year["banda_examen_baja"] = by_year["total_usd"] - by_year["error_examen_usd"]
    by_year["banda_examen_alta"] = by_year["total_usd"] + by_year["error_examen_usd"]
    return by_month, by_year


def check_forecast(future: pd.DataFrame, fine_table: pd.DataFrame, by_month: pd.DataFrame,
                   configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    missing = future[["tasa", "uplift", "esperado_usd"]].isna().any(axis=1)
    configuration.log_check(STEP_LABEL, check_log, "every future row has a rate, an uplift and an expected value",
                            not missing.any(), failure_detail=f"{int(missing.sum()):,} rows incomplete",
                            context=f"{len(future):,} rows")
    bad_band = ~(future["tasa"].between(0, 1) & (future["tasa_baja"] <= future["tasa"] + 1e-12)
                 & (future["tasa"] <= future["tasa_alta"] + 1e-12) & future["tasa_baja"].between(0, 1) & future["tasa_alta"].between(0, 1))
    configuration.log_check(STEP_LABEL, check_log, "every rate and band is inside [0, 1] and the band contains the rate",
                            not bad_band.any(), failure_detail=f"{int(bad_band.sum()):,} rows with a bad rate or band",
                            examples=future.loc[bad_band, ["tasa", "tasa_baja", "tasa_alta"]])
    future_roles = fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    configuration.log_check(STEP_LABEL, check_log, "no closed row is forecast; every future row of the extract is",
                            set(future.loc[(future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL) & (future[MATURATION_ROW_COLUMN] == 0), "_fila"])
                            == set(fine_table.index[future_roles]),
                            failure_detail="the forecast rows are not the future rows")
    real_rows = (future[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL) & (future[MATURATION_ROW_COLUMN] == 0)
    due_difference = future.loc[real_rows, configuration.pipeline_usd_col].sum() - fine_table.loc[future_roles, configuration.pipeline_usd_col].sum()
    configuration.log_check(STEP_LABEL, check_log, "the pipeline of the extract's future rows is conserved (Σ USD due)",
                            abs(due_difference) <= configuration.money_tolerance, failure_detail=f"difference ${due_difference:,.2f}",
                            context=f"${future.loc[real_rows, configuration.pipeline_usd_col].sum():,.0f} due in the extract")
    total_difference = by_month["esperado_usd"].sum() - future["esperado_usd"].sum()
    configuration.log_check(STEP_LABEL, check_log, "the totals are the sum of the rows", abs(total_difference) <= configuration.money_tolerance,
                            failure_detail=f"difference ${total_difference:,.2f}", context=f"${future['esperado_usd'].sum():,.0f} expected")
