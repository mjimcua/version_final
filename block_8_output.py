"""
block_8_output.py — BLOCK 8 · OUTPUT AND CONTROL: write, explain and validate, without changing any number.

MISSION: leave the forecast where business and the analyst read it (the core and its satellites in SQL, Power BI,
the report) so that every number can be followed back to the step and the row it comes from.

MECHANISM: one wide table (the core) with every row of the run and every decision as a column, prefixed by the step
that made it (s<block>_<step>_), whose sums ARE the business answers; satellites for the drill-down related by ids;
the auxiliary tables of Power BI; a final validation of the whole chain; the report. Every output is checked against
the step it comes from.

THE STEPS:
  8.1  new rows       the rows the framework adds or wipes, month by month
  8.2  audit          the satellites of the core: ids of every stage, techniques, credibility references, dynamics
  8.3  core           sff_nucleo, its legend and the forecast series dimension
  8.4  Power BI       the auxiliary tables: sources, price monitor, calendar of increases, parameters
  8.5  validation     the chain agrees from the raw to the forecast
  8.6  report         the report, the summary for business and the charts

READS: every result of blocks 1 to 7.
GIVES: new_rows, audit, core, core_legend, forecast_series, power_bi, validation, report.

THE INTERFACE:
  block = OutputBlock(configuration).run(tables)     tables: {name: result} of blocks 1 to 7
  block.core, block.report, …    block.results()    block.seconds
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os
import re
import time

import numpy as np
import pandas as pd

from config import (TABLE_COMPOSITION_MEMBERS, TABLE_LADDER_MERGES, TABLE_LADDER_STEPS, TABLE_MODEL_RELATIONS,
                    TABLE_POOL_SERIES, TABLE_SERIES_EXAM, TABLE_SERIES_EXAM_DETAIL, TABLE_UPLIFT_CELLS)
from config import (CALENDAR_ROLE_COLUMN, COMPOSITION_ID_COLUMN, Config, COVERAGE_COLUMN, CURRENT_MONTH_COLUMN,
                    DUE_UNITS_AT_DUE_COLUMN, DUE_USD_AT_DUE_COLUMN, FINE_ROWS_COLUMN, GATE_LEVEL,
                    MATURATION_ROW_COLUMN, METHOD_BASELINE, METHOD_FRAMEWORK, PIPELINE_ORIGIN_COLUMN,
                    PIPELINE_PROJECTED, PIPELINE_REAL, PIPELINE_SIMULATED, PURPOSE_EXAM, PURPOSE_SELECTION,
                    RATE_COLUMN, RATE_FROM_POOL, REPORT_FILE_NAME, ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN,
                    ROLES_IN_ORDER, ROUTE_COLUMN, ROW_FROM_GAP, ROW_FROM_MATURATION, ROW_FROM_RAW, ROW_ORIGIN_COLUMN,
                    S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN, S0_RENEWED_USD_COLUMN,
                    SERIES_ID_COLUMN, SIGN_COLUMN, SYNTHETIC_COLUMN, TABLE_COMPOSITION, TABLE_COMPOSITION_FORECAST_ALL,
                    TABLE_COMPOSITION_TECHNIQUES, TABLE_CORE, TABLE_CORE_LEGEND, TABLE_CREDIBILITY,
                    TABLE_CREDIBILITY_MEMBERS, TABLE_FORECAST_SERIES, TABLE_SERIES_DYNAMICS, TABLE_VALIDATION,
                    TECHNIQUE_COMPOSITION_BELOW_FLOOR, TECHNIQUE_NOT_ENOUGH_HISTORY, TECHNIQUE_TESTED,
                    TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_SIMULATED,
                    TOTAL_ORIGIN_TOTAL, TRUTH_ROLES, TRUTH_ROLES as TRUTH_ROLES_OF_CORE, TS_PROJECTED, TS_REAL,
                    TABLE_BLOCK_VERDICTS, TABLE_DATA_PROFILE, VERDICT_FAILS, VERDICT_HOLDS, VERDICT_INCONCLUSIVE,
                    VERDICT_WARNING,
                    TS_REENTRY, TS_SIMULATED, UNIT_ID_COLUMN, UNKNOWN_DISCOUNT_BUCKET, UPLIFT_CELL_ID_COLUMN,
                    parameter_table)
from block_2_rate import band_of_horizon, credibility_k_detail, dynamics_of_one_series
from graficos import CHART_FOLDER, draw_charts
from price_increases import increase_calendar, price_monitor
from report_queries import new_rows_query, year_by_region_columns, year_by_region_query
from resumen_negocio import business_summary, gain_text, show_business_summary
from verdicts import record_the_verdict, verdict_table
from techniques import CATALOGUE, inverse_logit, predict_logit
from time_series_charts import HISTORY_CHART, PREDICTION_CHART


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 8.1 · THE ROWS THE FRAMEWORK ADDS OR WIPES  (until 25-oct-2026: step_22_new_rows.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The rows the framework adds or wipes, month by month, in one format.
#
# The forecast does not use the extract as it comes. Five kinds of rows are added or wiped, each for a
# reason, and this step puts them side by side so they can be explained:
#
#   hueco                        step 1.7 · a forecast unit with nothing due, added inside the history of
#                                an estimable series so its monthly series has no holes (the techniques
#                                read consecutive months: a trend, a season, a moving average). Every
#                                measure is 0: it adds a month, not money.
#   resultado_adelantado_borrado step 1.3 · a renewal already booked from the current month on: the month
#                                has not finished (or has not started); the forecast predicts it instead.
#   pipeline_parcial_borrada     step 1.3 · the PARTIAL pipeline of the 1-year licences due 12 months after
#                                the current month or later: it is generated by the sales and renewals
#                                from the current month on, which have only started (the current month is
#                                half way, the next ones have not begun). Left as it is, the forecast of
#                                those months would run on a pipeline half built; step 7.2 rebuilds it whole.
#   proyectada                   step 7.2 · the expected renewal of a 1-year licence due in the simulation
#                                window, due 12 months later: what replaces the wiped pipeline.
#   simulada                     step 7.2 · an acquisition simulated in the simulation window, due 12 months
#                                later: also replaces wiped pipeline.
#   maduracion                   stage 4.1, valued in step 7.2 · the units of a neutral series that will carry a
#                                state when they fall due, as a prediction row of their series with a state: no data
#                                (s00/s02 at 0), units and USD at due date; they leave a neutral row at its price.
#
# One row per month × origin: rows, series, units and USD (due, or wiped), and USD expected to renew for
# the rows the forecast predicts. It is a REPORT, not a table of the run: everything in it is a query over
# sff_nucleo, and the step prints that query (report_queries.py) instead of writing a second copy.
#
# Actions (logged as they are done):
#   1. the gaps of step 1.7
#   2. what step 1.3 wiped (from the s0_ columns of the fine table)
#   3. what step 7.2 created
#   4. the table: month × origin; the totals per origin, side by side
#   5. check that the raw = what stays + what was wiped                    check 1
#   6. count the checks; stop if any failed
#   7. show the totals per origin and the months with activity, as tables, and the SQL that reproduces them
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. what the raw had due = what stays due + the partial pipeline wiped (units and USD)

STEP_8_1_LABEL = "8.1"
STEP_8_1_NAME = "THE ROWS THE FRAMEWORK ADDS OR WIPES"
STEP_8_1_PURPOSE = ("put side by side, month by month and in one format, every row the framework adds or wipes: the gaps "
                "that make the monthly series continuous, what the calendar wipes because it is not known yet, and what "
                "the forecast creates to replace it")
STEP_8_1_ACTIONS = ["the gaps of step 1.7",
                "what step 1.3 wiped (from the s0_ columns of the fine table)",
                "what step 7.2 created",
                "the table: month × origin; the totals per origin, side by side",
                "check that the raw = what stays + what was wiped (check 1)",
                "count the checks; stop if any failed",
                "show the totals per origin and the months with activity, as tables, and the SQL that reproduces them"]
STEP_8_1_OUTPUT = "one row per month × origin (rows, series, units, USD, USD expected): a report, with its SQL over sff_nucleo"

ORIGIN_GAP = "hueco"
ORIGIN_WIPED_RESULT = "resultado_adelantado_borrado"
ORIGIN_WIPED_PIPELINE = "pipeline_parcial_borrada"
ORIGINS_IN_ORDER = [ORIGIN_GAP, ORIGIN_WIPED_RESULT, ORIGIN_WIPED_PIPELINE, PIPELINE_PROJECTED, PIPELINE_SIMULATED, ROW_FROM_MATURATION]
WHAT_IT_IS = {ORIGIN_GAP: "added (step 1.7): a month with nothing due inside the history of a series, so it has no holes",
              ORIGIN_WIPED_RESULT: "wiped (step 1.3): a renewal booked from the current month on; the forecast predicts it",
              ORIGIN_WIPED_PIPELINE: "wiped (step 1.3): partial pipeline of 1-year licences, half built by sales and renewals that have only started",
              PIPELINE_PROJECTED: "created (step 7.2): expected renewal of a 1-year licence due in the window, 12 months later",
              PIPELINE_SIMULATED: "created (step 7.2): acquisition simulated in the window, due 12 months later",
              ROW_FROM_MATURATION: "predicted (stage 4.1, valued in step 7.2): units of a neutral series that will carry a state "
                                   "when they fall due, in their series with a state; no data: units and USD at due date"}


def count_the_new_rows(fine_table: pd.DataFrame, rated_units: pd.DataFrame, forecast_rows: pd.DataFrame,
                       configuration: Config) -> pd.DataFrame:
    """The rows added or wiped, one row per month × origin.

    INPUT:   the fine table (step 1.4, with the s0_ columns of step 1.3) · the forecast units with the gaps (step 1.7)
             · the future rows of the forecast (step 7.2).
    OUTPUT:  month × origin: rows, series, units, USD, USD expected (a report: its SQL over sff_nucleo is printed).
    RULES:   counts what the steps did; computes nothing new.
    EDGE CASES: an origin with no rows does not appear; a run without a simulation window has no proyectada / simulada.
    """
    configuration.log_step_start(STEP_8_1_LABEL, STEP_8_1_NAME, STEP_8_1_PURPOSE, STEP_8_1_ACTIONS, STEP_8_1_OUTPUT)
    check_log = []
    period, due_units, due_usd = configuration.period_col, configuration.pipeline_units_col, configuration.pipeline_usd_col
    parts = []

    # [1] the gaps
    gaps = rated_units[rated_units[SYNTHETIC_COLUMN] == 1]
    parts.append(summarise(gaps, ORIGIN_GAP, period, units=0.0, usd=0.0))
    configuration.log_action(STEP_8_1_LABEL, 1, f"{len(gaps):,} gap rows in {gaps[SERIES_ID_COLUMN].nunique():,} series "
                                            f"(every measure 0: a month, not money)")

    # [2] what the calendar wiped
    wiped_result = fine_table[fine_table[S0_RENEWED_UNITS_COLUMN].fillna(0) > fine_table[configuration.renewed_units_col].fillna(0)]
    parts.append(summarise(wiped_result, ORIGIN_WIPED_RESULT, period,
                           units=wiped_result[S0_RENEWED_UNITS_COLUMN].fillna(0) - wiped_result[configuration.renewed_units_col].fillna(0),
                           usd=wiped_result[S0_RENEWED_USD_COLUMN].fillna(0) - wiped_result[configuration.renewed_usd_col].fillna(0)))
    wiped_pipeline = fine_table[fine_table[S0_PIPELINE_UNITS_COLUMN] > fine_table[due_units] + configuration.units_tolerance]
    parts.append(summarise(wiped_pipeline, ORIGIN_WIPED_PIPELINE, period,
                           units=wiped_pipeline[S0_PIPELINE_UNITS_COLUMN] - wiped_pipeline[due_units],
                           usd=wiped_pipeline[S0_PIPELINE_USD_COLUMN] - wiped_pipeline[due_usd]))
    configuration.log_action(STEP_8_1_LABEL, 2, f"wiped by the calendar: {len(wiped_result):,} rows of early results · "
                                            f"{len(wiped_pipeline):,} rows of pipeline not known yet")

    # [3] what the forecast created: the extended horizon, and the prediction rows of the maturation (their units are
    #     the ones at due date: they have no data)
    is_maturation = (forecast_rows[MATURATION_ROW_COLUMN] == 1) if MATURATION_ROW_COLUMN in forecast_rows else pd.Series(False, index=forecast_rows.index)
    counts = {}
    for origin in (PIPELINE_PROJECTED, PIPELINE_SIMULATED):
        created = (forecast_rows[(forecast_rows[PIPELINE_ORIGIN_COLUMN] == origin) & ~is_maturation]
                   if PIPELINE_ORIGIN_COLUMN in forecast_rows else forecast_rows.iloc[0:0])
        counts[origin] = len(created)
        parts.append(summarise(created, origin, period, units=created[due_units], usd=created[due_usd],
                               expected=created["esperado_usd"] if "esperado_usd" in created else None))
    matured = forecast_rows[is_maturation]
    counts[ROW_FROM_MATURATION] = len(matured)
    if len(matured):
        parts.append(summarise(matured, ROW_FROM_MATURATION, period, units=matured[DUE_UNITS_AT_DUE_COLUMN],
                               usd=matured[DUE_USD_AT_DUE_COLUMN], expected=matured["esperado_usd"]))
    configuration.log_action(STEP_8_1_LABEL, 3, "created by the forecast: " + " · ".join(f"{origin} {count:,} rows"
                                                                                     for origin, count in counts.items()))

    # [4] the table
    table = pd.concat([part for part in parts if len(part)], ignore_index=True)
    table["origen"] = pd.Categorical(table["origen"], ORIGINS_IN_ORDER, ordered=True)
    table = table.sort_values(["mes", "origen"]).reset_index(drop=True)
    table["origen"] = table["origen"].astype(str)
    roles = configuration.role_of_months(table["mes"])
    table.insert(1, CALENDAR_ROLE_COLUMN, roles)
    totals = totals_per_origin(table)
    configuration.log_action(STEP_8_1_LABEL, 4, f"{len(table):,} month × origin rows")

    # [5] the reconciliation with the raw
    configuration.log_action(STEP_8_1_LABEL, 5, "checking the reconciliation with the raw")
    raw_units, raw_usd = float(fine_table[S0_PIPELINE_UNITS_COLUMN].sum()), float(fine_table[S0_PIPELINE_USD_COLUMN].sum())
    kept_units, kept_usd = float(fine_table[due_units].sum()), float(fine_table[due_usd].sum())
    wiped = table[table["origen"] == ORIGIN_WIPED_PIPELINE]
    reconciles = (abs(raw_units - kept_units - wiped["unidades"].sum()) <= configuration.units_tolerance * max(raw_units, 1)
                  and abs(raw_usd - kept_usd - wiped["usd"].sum()) <= configuration.units_tolerance * max(raw_usd, 1))
    configuration.log_check(STEP_8_1_LABEL, check_log, "what the raw had due = what stays due + the partial pipeline wiped (units and USD)",
                            reconciles, failure_detail="the wiped pipeline does not explain the difference with the raw",
                            context=f"raw {raw_units:,.0f} units · ${raw_usd:,.0f} = kept {kept_units:,.0f} · ${kept_usd:,.0f} "
                                    f"+ wiped {wiped['unidades'].sum():,.0f} · ${wiped['usd'].sum():,.0f}")

    # [6] the count of the checks
    configuration.log_action(STEP_8_1_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_8_1_LABEL, STEP_8_1_NAME, check_log)

    # [7] the tables and their SQL
    configuration.log_action(STEP_8_1_LABEL, 7, "per origin (what each one is, and how much):")
    configuration.show_table(totals)
    configuration.logger.doc(f"[{STEP_8_1_LABEL}] the months with wiped or created pipeline (the gaps are spread over the history):")
    configuration.show_table(table[table["origen"] != ORIGIN_GAP].reset_index(drop=True))
    configuration.show_query(STEP_8_1_LABEL, "the rows the framework adds or wipes, month × origin", new_rows_query(configuration))
    return table


def summarise(rows: pd.DataFrame, origin: str, period_column: str, units, usd, expected=None) -> pd.DataFrame:
    """One row per month: rows, series, units, USD and USD expected of one origin."""
    if not len(rows):
        return pd.DataFrame()
    frame = pd.DataFrame({"mes": rows[period_column].to_numpy(), SERIES_ID_COLUMN: rows[SERIES_ID_COLUMN].to_numpy(),
                          "unidades": np.broadcast_to(np.asarray(units, dtype=float), len(rows)),
                          "usd": np.broadcast_to(np.asarray(usd, dtype=float), len(rows)),
                          "esperado_usd": (np.asarray(expected, dtype=float) if expected is not None else np.full(len(rows), np.nan))})
    grouped = frame.groupby("mes")
    summary = pd.DataFrame({"filas": grouped.size(), "series": grouped[SERIES_ID_COLUMN].nunique(),
                            "unidades": grouped["unidades"].sum(), "usd": grouped["usd"].sum(),
                            "esperado_usd": grouped["esperado_usd"].sum(min_count=1)}).reset_index()
    summary.insert(1, "origen", origin)
    return summary


def totals_per_origin(table: pd.DataFrame) -> pd.DataFrame:
    """One row per origin: what it is, its months, rows, units, USD and USD expected."""
    rows = []
    for origin in ORIGINS_IN_ORDER:
        block = table[table["origen"] == origin]
        if not len(block):
            continue
        rows.append({"origen": origin, "que_es": WHAT_IT_IS[origin], "meses": f"{block['mes'].min()}..{block['mes'].max()}",
                     "filas": int(block["filas"].sum()), "unidades": float(block["unidades"].sum()), "usd": float(block["usd"].sum()),
                     "esperado_usd": float(block["esperado_usd"].sum()) if block["esperado_usd"].notna().any() else np.nan})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 8.2 · AUDIT TABLES  (until 25-oct-2026: step_23_audit.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The audit tables: the satellites of the core (sff_nucleo) for the drill-down.
#
# The core keeps, for every forecast series, what the report needs. When a number is in doubt,
# these tables explain the whole way it came from, joined to the core by its keys:
#
#   table                          one row per                                  key in the core
#   sff_composition                id of any stage (0 raw … 3 composition)      s2_2_stage0_id … s2_2_stage3_id
#   sff_composition_techniques     composition × horizon band × technique       s2_2_stage3_id
#   sff_credibility                credibility reference                        s2_3_credibility_ref_id
#   sff_credibility_members        reference × forecast series in its rate      s2_3_credibility_ref_id
#   sff_series_dynamics            forecast series                              s1_4_fs_id
#   sff_composition_forecast_all   composition × future horizon × technique     s2_2_stage3_id
#
# EVERY TEST HAS ITS INTERVAL, built as the forecast builds its band: the error quantiles of the
# technique in the backtest (step 2.6) × the binomial error with the series' own units due. A test
# is IN THE INTERVAL when the real rate of the series falls inside it. The exam of every forecast
# series (its chosen technique, every exam month and horizon) is summarised in sff_series_exam and
# in the core: how much it erred and how many predictions were in the interval.
#
# The 4 stages only give the forecast better conditions to predict: the backtest, the exam and the
# forecast are referred back to every forecast series (its own units due, its own renewals).
#
# NOTHING IS COUNTED TWICE: every table is checked against the table it comes from (the checks
# below). A forecast series is in one id per stage, in one composition, and has one reference; a
# reference may hold series of many compositions, so its members are not summed across references.
#
# Actions (logged as they are done):
#   1. sff_composition: every id of every stage                         checks 1-2
#   2. sff_composition_techniques: every technique of every composition check 3
#   3. sff_credibility and its members                                  check 4
#   4. sff_series_dynamics                                              check 5
#   5. sff_composition_forecast_all                                     check 6
#   6. write the audit tables                                           checks 7-12
#   7. count the checks; stop if any failed
#
# The backtest of every technique referred back to every forecast series is NOT here: step 7.1 examines
# every forecast series (sff_series_exam, sff_series_exam_detail) and the backtest of every composition is
# sff_backtest_predicciones (step 2.6).
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every stage is a partition: the units due of its ids add up to the raw's
#    2. every stage holds every estimable series once
#    3. every composition and band has exactly one chosen technique, the one of step 2.6
#    4. the support and the rate of every reference, recomputed from its members, are the ones used
#    5. one dynamics row per estimable forecast series
#    6. the chosen technique gives the rate step 7.2 used (rows with no credibility shift)
#    7-12. the six tables written and read back
#
# Output: dict of the audit tables · the tables listed above.

STEP_8_2_LABEL = "8.2"
STEP_8_2_NAME = "AUDIT TABLES"
STEP_8_2_PURPOSE = ("build the satellites of the core for the drill-down: every id of every stage, every technique of every "
                "composition, every credibility reference and its members, the dynamics of every forecast series, the "
                "future rate of every technique; each one checked against the table it comes from, so nothing is counted twice")
STEP_8_2_ACTIONS = ["sff_composition: every id of every stage (checks 1-2)",
                "sff_composition_techniques: every technique of every composition (check 3)",
                "sff_credibility and its members (check 4)",
                "sff_series_dynamics (check 5)",
                "sff_composition_forecast_all (check 6)",
                "write the audit tables (checks 7-12)",
                "count the checks; stop if any failed"]
STEP_8_2_OUTPUT = "six audit tables joined to the core by fs_id, the stage ids and the credibility reference"

PERCENTAGE_POINTS = 100
STAGES = (0, 1, 2, 3)


def build_audit_tables(ladder: dict, series_rate: pd.DataFrame, series_estimate: pd.DataFrame,
                       rated_units: pd.DataFrame, pool_series: pd.DataFrame, pool_reference: pd.DataFrame,
                       pool_dynamics: pd.DataFrame, backtest: dict, forecast_rows: pd.DataFrame,
                       configuration: Config, rate_box=None) -> dict:
    """The six audit tables, checked against their sources and written.

    INPUT:   the ladder (step 2.2), series_rate (1.7), series_estimate (2.3), rated_units (1.7), the
             composition series and reference (2.4), their dynamics (2.5), the backtest (2.6) and the
             forecast rows (7.2).
    OUTPUT:  dict(composition, composition_techniques, credibility, credibility_members,
             series_dynamics, composition_forecast_all).
    RULES:   see the module header.
    EDGE CASES: a composition below the support floor is not judged by the backtest: its
             techniques have status composition_below_floor and its series have no backtest rows.
    """
    configuration.log_step_start(STEP_8_2_LABEL, STEP_8_2_NAME, STEP_8_2_PURPOSE, STEP_8_2_ACTIONS, STEP_8_2_OUTPUT)
    check_log = []
    history = real_history(rated_units, configuration)
    groups = ladder["groups"]

    # [1] every id of every stage
    composition = composition_table(ladder["stages"], groups, series_estimate, pool_reference, backtest, history,
                                    forecast_rows, configuration)
    configuration.log_action(STEP_8_2_LABEL, 1, f"{len(composition):,} ids over the 4 stages · "
                                            f"{int(composition['is_composition'].sum()):,} compositions")
    check_stages(ladder["stages"], history, configuration, check_log)

    # [2] every technique of every composition
    techniques = composition_techniques(pool_series, pool_reference, backtest, configuration)
    configuration.log_action(STEP_8_2_LABEL, 2, f"{len(techniques):,} composition × band × technique rows · status "
                                            f"{techniques['status'].value_counts().to_dict()}")
    check_chosen_once(techniques, backtest["decision"], [COMPOSITION_ID_COLUMN, "tramo_h"],
                      "every composition and band has exactly one chosen technique, the one of step 2.6",
                      configuration, check_log)

    # [3] the credibility references and their members
    credibility, members = credibility_tables(groups, ladder["reference_members"], series_rate, configuration)
    configuration.log_action(STEP_8_2_LABEL, 3, f"{len(credibility):,} references · {len(members):,} reference × series rows "
                                            f"({members['role'].value_counts().to_dict() if len(members) else {}})")
    check_references(credibility, members, history, configuration, check_log)

    # [4] the dynamics of every forecast series
    series_dynamics = series_dynamics_table(groups, series_rate, history, pool_dynamics, configuration)
    configuration.log_action(STEP_8_2_LABEL, 4, f"{len(series_dynamics):,} forecast series · measurable "
                                            f"{series_dynamics['measurable'].value_counts().to_dict()} · "
                                            f"{int(series_dynamics['differs_from_composition'].sum()):,} differ from their composition")
    configuration.log_check(STEP_8_2_LABEL, check_log, "one dynamics row per estimable forecast series",
                            len(series_dynamics) == len(groups) and series_dynamics[SERIES_ID_COLUMN].is_unique,
                            failure_detail=f"{len(series_dynamics):,} rows for {len(groups):,} series")

    # [5] the future rate of every technique
    if rate_box is None:
        from step_13_rate_box import RateBox
        rate_box = RateBox(rated_units, series_estimate, pool_series, backtest, ladder.get("reference_members"), configuration)
    forecast_all = composition_forecast_all(rate_box, backtest["decision"], forecast_rows, configuration)
    configuration.log_action(STEP_8_2_LABEL, 5, f"{len(forecast_all):,} composition × horizon × technique rates")
    check_forecast_all(forecast_all, forecast_rows, series_estimate, configuration, check_log)

    # [6] the tables
    configuration.log_action(STEP_8_2_LABEL, 6, "writing the audit tables")
    tables = dict(composition=composition, composition_techniques=techniques, credibility=credibility,
                  credibility_members=members, series_dynamics=series_dynamics, composition_forecast_all=forecast_all)
    for name, table_name in (("composition", TABLE_COMPOSITION), ("composition_techniques", TABLE_COMPOSITION_TECHNIQUES),
                             ("credibility", TABLE_CREDIBILITY), ("credibility_members", TABLE_CREDIBILITY_MEMBERS),
                             ("series_dynamics", TABLE_SERIES_DYNAMICS),
                             ("composition_forecast_all", TABLE_COMPOSITION_FORECAST_ALL)):
        configuration.write_table(STEP_8_2_LABEL, check_log, tables[name], table_name)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_8_2_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_8_2_LABEL, STEP_8_2_NAME, check_log)
    return tables


def real_history(rated_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The real closed months of every forecast series with something due: its units due and renewed."""
    real = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                       & (rated_units[configuration.pipeline_units_col] > 0)]
    return real[[SERIES_ID_COLUMN, configuration.period_col, CALENDAR_ROLE_COLUMN,
                 configuration.pipeline_units_col, configuration.renewed_units_col]].copy()


def monthly_support(history: pd.DataFrame, members: pd.DataFrame, key: str, configuration: Config) -> pd.DataFrame:
    """Per key: support (median of the monthly units due summed over its members), units due, renewed, rate."""
    joined = history.merge(members, on=SERIES_ID_COLUMN)
    monthly = joined.groupby([key, configuration.period_col])[[configuration.pipeline_units_col,
                                                               configuration.renewed_units_col]].sum().reset_index()
    grouped = monthly.groupby(key)
    result = pd.DataFrame({"support": grouped[configuration.pipeline_units_col].median(),
                           "units_due": grouped[configuration.pipeline_units_col].sum(),
                           "units_renewed": grouped[configuration.renewed_units_col].sum()})
    result["rate"] = result["units_renewed"] / result["units_due"].where(result["units_due"] > 0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# 1 · THE COMPOSITIONS (every id of every stage)
# ═══════════════════════════════════════════════════════════════════════════════════

def composition_table(stages: pd.DataFrame, groups: pd.DataFrame, series_estimate: pd.DataFrame,
                      pool_reference: pd.DataFrame, backtest: dict, history: pd.DataFrame, forecast_rows: pd.DataFrame,
                      configuration: Config) -> pd.DataFrame:
    """One row per id of any stage: the stages it appears in, its series, support, rate; and, for a
    composition (stage 3), how it is predicted."""
    long = pd.concat([pd.DataFrame({SERIES_ID_COLUMN: stages[SERIES_ID_COLUMN], "stage": stage,
                                    "id": stages[f"stage{stage}_id"]}) for stage in STAGES], ignore_index=True)
    # an id that repeats in several stages has the same series in all of them: one membership per id
    membership = long.drop_duplicates(["id", SERIES_ID_COLUMN])[["id", SERIES_ID_COLUMN]]
    support = monthly_support(history, membership, "id", configuration)
    table = (long.groupby("id").agg(first_stage=("stage", "min"), last_stage=("stage", "max"))
             .join(membership.groupby("id").size().rename("series")).join(support).reset_index())
    table["is_composition"] = (table["last_stage"] == 3).astype(int)

    # the compositions: how they are predicted
    per_composition = (series_estimate.dropna(subset=["credibility_ref_id"]).drop_duplicates(COMPOSITION_ID_COLUMN)
                       .set_index(COMPOSITION_ID_COLUMN)[["credibility_ref_id", "ref_support", "ref_rate", "k", "z"]])
    estimated = series_estimate.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["tasa_estimada"]
    table = table.join(per_composition, on="id").join(estimated.rename("estimated_rate"), on="id")
    table = table.join(pool_reference.set_index(COMPOSITION_ID_COLUMN)[["gate", "usd_por_predecir"]], on="id")
    decision = backtest["decision"].pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h", values="tecnica").add_prefix("technique_")
    exam = backtest["exam_by_pool"].pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h", values="elegida_err_pp_medio").add_prefix("exam_err_pp_")
    table = table.join(decision, on="id").join(exam, on="id")
    if forecast_rows is not None and len(forecast_rows):
        expected = forecast_rows.groupby([COMPOSITION_ID_COLUMN, "confidence"])["esperado_usd"].sum().unstack(fill_value=0.0)
        expected = expected.add_prefix("expected_usd_")
        table = table.join(expected, on="id")
    return table.rename(columns={"id": COMPOSITION_ID_COLUMN}).sort_values(["first_stage", COMPOSITION_ID_COLUMN]).reset_index(drop=True)


def check_stages(stages: pd.DataFrame, history: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 and 2: every stage is a partition of the raw."""
    raw_due = float(history[history[SERIES_ID_COLUMN].isin(stages[SERIES_ID_COLUMN])][configuration.pipeline_units_col].sum())
    due_by_stage = {}
    for stage in STAGES:
        members = stages[[SERIES_ID_COLUMN, f"stage{stage}_id"]].rename(columns={f"stage{stage}_id": "id"})
        due_by_stage[stage] = float(monthly_support(history, members, "id", configuration)["units_due"].sum())
    configuration.log_check(STEP_8_2_LABEL, check_log, "every stage is a partition: the units due of its ids add up to the raw's",
                            all(abs(due - raw_due) <= configuration.units_tolerance for due in due_by_stage.values()),
                            failure_detail=f"units due by stage {due_by_stage} vs {raw_due:,.0f}",
                            context=f"{raw_due:,.0f} units due in every stage")
    configuration.log_check(STEP_8_2_LABEL, check_log, "every stage holds every estimable series once",
                            stages[SERIES_ID_COLUMN].is_unique and stages[[f"stage{stage}_id" for stage in STAGES]].notna().all().all(),
                            failure_detail="a series without an id in some stage, or twice",
                            context=f"{len(stages):,} series")


# ═══════════════════════════════════════════════════════════════════════════════════
# 2 · THE TECHNIQUES OF EVERY COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════════

def composition_techniques(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, backtest: dict,
                           configuration: Config) -> pd.DataFrame:
    """Composition × band × technique: status and why, months, errors in selection and exam, rank, chosen."""
    truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()]
    months_available = truth.groupby(COMPOSITION_ID_COLUMN).size()
    gate = pool_reference.set_index(COMPOSITION_ID_COLUMN)["gate"]
    predictions = backtest["predictions"].assign(abs_err_pp=lambda frame: frame["err_pp"].abs(),
                                                 abs_err_norm=lambda frame: frame["err_norm"].abs())
    by_purpose = (predictions.groupby([COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "proposito"])
                  [["abs_err_pp", "abs_err_norm"]].mean().unstack("proposito"))
    by_purpose.columns = [f"{purpose}_{measure}" for measure, purpose in by_purpose.columns]
    chosen = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    # tested = it predicted in the selection months of that band (the history of the exam months does not count:
    # the choice is made before them)
    selected = set(map(tuple, predictions.loc[predictions["proposito"] == PURPOSE_SELECTION,
                                              [COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"]].drop_duplicates().to_numpy()))

    rows = []
    for composition_id in pool_reference[COMPOSITION_ID_COLUMN]:
        for band_name in configuration.horizon_bands:
            for technique_id, technique in CATALOGUE.items():
                required = int(technique[3])
                available = int(months_available.get(composition_id, 0))
                if gate.get(composition_id) != GATE_LEVEL:
                    status = TECHNIQUE_COMPOSITION_BELOW_FLOOR
                elif available < required or (composition_id, band_name, technique_id) not in selected:
                    status = TECHNIQUE_NOT_ENOUGH_HISTORY
                else:
                    status = TECHNIQUE_TESTED
                rows.append({COMPOSITION_ID_COLUMN: composition_id, "tramo_h": band_name, "tecnica": technique_id,
                             "status": status, "months_available": available, "months_required": required,
                             "is_chosen": int(chosen.get((composition_id, band_name)) == technique_id)})
    table = pd.DataFrame(rows).join(by_purpose, on=[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
    table = table.rename(columns={f"{PURPOSE_SELECTION}_abs_err_norm": "selection_err_norm",
                                  f"{PURPOSE_SELECTION}_abs_err_pp": "selection_err_pp",
                                  f"{PURPOSE_EXAM}_abs_err_norm": "exam_err_norm",
                                  f"{PURPOSE_EXAM}_abs_err_pp": "exam_err_pp"})
    tested = table["status"] == TECHNIQUE_TESTED
    table["rank"] = np.nan
    table.loc[tested, "rank"] = (table[tested].groupby([COMPOSITION_ID_COLUMN, "tramo_h"])["selection_err_norm"]
                                 .rank(method="min"))
    return table


def check_chosen_once(table: pd.DataFrame, decision: pd.DataFrame, keys: list, name: str, configuration: Config,
                      check_log: list) -> None:
    """Exactly one chosen technique per key, and it is the technique of step 2.6's decision."""
    chosen = table[table["is_chosen"] == 1]
    per_key = chosen.groupby(keys).size()
    matches = chosen.merge(decision[keys + ["tecnica"]], on=keys, suffixes=("", "_decision"))
    configuration.log_check(STEP_8_2_LABEL, check_log, name,
                            bool((per_key == 1).all()) and len(per_key) == len(decision)
                            and bool((matches["tecnica"] == matches["tecnica_decision"]).all()),
                            failure_detail=f"{int((per_key != 1).sum())} keys without exactly one chosen · "
                                           f"{len(per_key)} keys with a choice for {len(decision)} decisions",
                            context=f"{len(decision):,} composition × band")


# ═══════════════════════════════════════════════════════════════════════════════════
# 3 · THE CREDIBILITY
# ═══════════════════════════════════════════════════════════════════════════════════

def credibility_tables(groups: pd.DataFrame, reference_members: pd.DataFrame, series_rate: pd.DataFrame,
                       configuration: Config) -> tuple:
    """One row per reference (support, rate, k and its parts), and one per reference × forecast series."""
    references = (groups.dropna(subset=["credibility_ref_id"])
                  .drop_duplicates("credibility_ref_id")[["credibility_ref_id", "credibility_ref_step", "ref_series",
                                                          "ref_support", "ref_rate"]])
    k_detail = credibility_k_detail(groups, configuration)
    credibility = references.merge(k_detail, on="credibility_ref_id", how="left")
    credibility["compositions_using"] = credibility["compositions_using"].fillna(0).astype(int)
    credibility["k"] = credibility["k"].fillna(configuration.k_cred)
    credibility["k_source"] = credibility["k_source"].fillna("default")

    composition_of = groups.set_index(SERIES_ID_COLUMN)[COMPOSITION_ID_COLUMN]
    reference_of_composition = groups.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["credibility_ref_id"]
    members = reference_members.merge(series_rate[[SERIES_ID_COLUMN, "n_propio", "tasa_propia"]], on=SERIES_ID_COLUMN, how="left")
    members[COMPOSITION_ID_COLUMN] = members[SERIES_ID_COLUMN].map(composition_of)
    borrows = members[COMPOSITION_ID_COLUMN].map(reference_of_composition) == members["credibility_ref_id"]
    members["role"] = np.where(borrows, "borrower", "lender")
    members = members.rename(columns={"n_propio": "support", "tasa_propia": "own_rate"})
    return credibility, members


def check_references(credibility: pd.DataFrame, members: pd.DataFrame, history: pd.DataFrame, configuration: Config,
                     check_log: list) -> None:
    """Check 4: every reference recomputed from its members."""
    if credibility.empty:
        configuration.log_not_evaluated(STEP_8_2_LABEL, check_log, "every reference recomputed from its members", "no reference")
        return
    recomputed = monthly_support(history, members[["credibility_ref_id", SERIES_ID_COLUMN]], "credibility_ref_id", configuration)
    compared = credibility.join(recomputed, on="credibility_ref_id")
    compared["members"] = compared["credibility_ref_id"].map(members.groupby("credibility_ref_id").size())
    wrong = compared[((compared["support"] - compared["ref_support"]).abs() > configuration.units_tolerance)
                     | ((compared["rate"] - compared["ref_rate"]).abs() > configuration.rate_tolerance)
                     | (compared["members"] != compared["ref_series"])]
    configuration.log_check(STEP_8_2_LABEL, check_log, "the support, rate and series of every reference, recomputed from its members, "
                            "are the ones used", wrong.empty, failure_detail=f"{len(wrong)} references differ",
                            context=f"{len(credibility):,} references",
                            examples=wrong[["credibility_ref_id", "ref_support", "support", "ref_rate", "rate"]])


# ═══════════════════════════════════════════════════════════════════════════════════
# 4 · THE DYNAMICS OF EVERY FORECAST SERIES
# ═══════════════════════════════════════════════════════════════════════════════════

def series_dynamics_table(groups: pd.DataFrame, series_rate: pd.DataFrame, history: pd.DataFrame,
                          pool_dynamics: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """φ, trend and seasonality of every estimable forecast series on its own months, whether they can
    be trusted, and how they compare with its composition's."""
    support = series_rate.set_index(SERIES_ID_COLUMN)["n_propio"]
    monthly = history.rename(columns={configuration.pipeline_units_col: "vencen", configuration.renewed_units_col: "renovadas"})
    composition_dynamics = (pool_dynamics.set_index(COMPOSITION_ID_COLUMN)[["tendencia", "estacional"]]
                            if pool_dynamics is not None and len(pool_dynamics) else pd.DataFrame(columns=["tendencia", "estacional"]))
    # one sort of the whole history, then the blocks arrive already in order (no sort per series)
    monthly = monthly.sort_values([SERIES_ID_COLUMN, configuration.period_col])
    months_of_series = dict(iter(monthly.groupby(SERIES_ID_COLUMN, sort=False)))
    no_months = monthly.iloc[0:0]
    rows = []
    for series_id, composition_id in zip(groups[SERIES_ID_COLUMN], groups[COMPOSITION_ID_COLUMN]):
        months = months_of_series.get(series_id, no_months)
        own_support = float(support.get(series_id, 0.0))
        row = {SERIES_ID_COLUMN: series_id, COMPOSITION_ID_COLUMN: composition_id, "months": len(months),
               "own_support": own_support}
        if len(months) < configuration.audit_trend_min_months:
            row["measurable"] = "short_history"
        else:
            row["measurable"] = "yes" if own_support >= configuration.support_floor else "low_support"
            measured = dynamics_of_one_series(months, configuration.period_col, configuration)
            row.update(phi=measured["phi"], trend=measured["tendencia"], trend_pp_year=measured["tendencia_pp_ano"],
                       trend_p_value=measured["p_valor_tendencia"])
            if len(months) >= configuration.audit_seasonality_min_months:
                row.update(seasonal=int(measured["estacional"]), seasonal_p_value=measured["p_valor_mes"],
                           amplitude_pp=measured["amplitud_pp"], high_months=measured["meses_alto"],
                           low_months=measured["meses_bajo"])
        if composition_id in composition_dynamics.index:
            row["composition_trend"] = composition_dynamics.loc[composition_id, "tendencia"]
            row["composition_seasonal"] = int(composition_dynamics.loc[composition_id, "estacional"])
        rows.append(row)
    table = pd.DataFrame(rows)
    for column_name in ("phi", "trend", "trend_pp_year", "trend_p_value", "seasonal", "seasonal_p_value", "amplitude_pp",
                        "high_months", "low_months", "composition_trend", "composition_seasonal"):
        if column_name not in table.columns:
            table[column_name] = np.nan
    # a series that behaves differently from its composition (only when both are measured and trusted)
    trusted = (table["measurable"] == "yes") & table["composition_trend"].notna()
    different_trend = table["trend"].fillna(0).ne(0) & table["trend"].ne(table["composition_trend"])
    different_season = table["seasonal"].notna() & table["seasonal"].ne(table["composition_seasonal"])
    table["differs_from_composition"] = (trusted & (different_trend | different_season)).astype(int)
    return table


# ═══════════════════════════════════════════════════════════════════════════════════
# 5 · THE BACKTEST REFERRED BACK TO EVERY FORECAST SERIES
# ═══════════════════════════════════════════════════════════════════════════════════

def composition_forecast_all(rate_box, decision: pd.DataFrame, forecast_rows: pd.DataFrame,
                             configuration: Config) -> pd.DataFrame:
    """Composition × future horizon × technique: the rate each technique gives with the whole history
    (before the credibility shift), and which one was chosen for that horizon's band."""
    horizons = sorted(int(horizon) for horizon in forecast_rows["h"].dropna().unique()) if forecast_rows is not None else []
    chosen = decision.set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"]
    last_closed = configuration.calendar_boundaries()["current"] - 1
    rows = []
    # the history of every composition as THE BOX OF THE RATE knows it at the last closed month
    for composition_id in sorted(rate_box.composition_index):
        history_logit, months, _, _ = rate_box.composition_history(composition_id, last_closed)
        months_available = len(months)
        for horizon in horizons:
            band_name = band_of_horizon(horizon, configuration.horizon_bands)
            for technique_id, technique in CATALOGUE.items():
                enough = months_available >= int(technique[3])
                value = predict_logit(technique_id, history_logit, months, horizon) if enough else np.nan
                rows.append({COMPOSITION_ID_COLUMN: composition_id, "h": horizon, "tramo_h": band_name,
                             "tecnica": technique_id, "months_available": months_available,
                             "rate": float(inverse_logit(value)) if np.isfinite(value) else np.nan,
                             "is_chosen": int(chosen.get((composition_id, band_name)) == technique_id)})
    return pd.DataFrame(rows)


def check_forecast_all(forecast_all: pd.DataFrame, forecast_rows: pd.DataFrame, series_estimate: pd.DataFrame,
                       configuration: Config, check_log: list) -> None:
    """Check 6: the chosen technique gives the rate step 7.2 used, on the rows with no credibility shift."""
    z_of = series_estimate.drop_duplicates(COMPOSITION_ID_COLUMN).set_index(COMPOSITION_ID_COLUMN)["z"]
    rows = forecast_rows[(forecast_rows["origen_tasa"] == RATE_FROM_POOL)]
    rows = rows[rows[COMPOSITION_ID_COLUMN].map(z_of).fillna(1) >= 1]
    chosen = forecast_all[forecast_all["is_chosen"] == 1].set_index([COMPOSITION_ID_COLUMN, "h"])["rate"].to_dict()
    expected = pd.Series([chosen.get(pair, np.nan) for pair in zip(rows[COMPOSITION_ID_COLUMN], rows["h"].astype(int))],
                         index=rows.index, dtype=float)
    differs = (expected - rows[RATE_COLUMN]).abs() > configuration.rate_tolerance
    configuration.log_check(STEP_8_2_LABEL, check_log, "the chosen technique gives the rate step 7.2 used (rows with no credibility shift)",
                            not bool(differs.any()), failure_detail=f"{int(differs.sum())} future rows differ",
                            context=f"{len(rows):,} future rows compared")


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 8.3 · CORE TABLE  (until 25-oct-2026: step_24_core.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The core table: the whole story of every row in ONE wide table.
#
# Decided on 27-sep-2026: the prediction lives in one wide table at the fine grain, history
# and future, WITHOUT hash keys (in Power BI they cost dictionary and space and say nothing
# to a reader). The columns go from left to right in the order of the steps, each with the
# prefix of the step that creates it (s1_1_, s1_3_, s1_4_…), so a row reads as its own story.
# The dimensions keep their names: they are the slicers.
#
# It can be built after ANY step from 03 on: every block of columns is added only when the
# step that produces it has run (its argument is not None), and the legend describes the
# columns present. main.py builds it last, with everything. It grows with every step of the
# framework: a new step adds its block here (today steps 1.1-7.1 and the forecast). The values of steps 2.5 and
# 2.6 belong to the ESTIMATION ID of the series (the pool it takes its rate from); a series
# whose pool was not measured (no support, short history) has them empty.
#
# Three levels live in the same table, and the legend says which is which:
#   · fila    a value of the row: it adds up (SUM)
#   · unidad  a value of the forecast unit (fu_id), repeated in its fine rows: do NOT add it up
#   · serie   a value of the rate series (fs_id), repeated in all its rows: do NOT add it up;
#             with one series selected, MAX (or MIN) returns it
# A RATE is never stored per row: a rate belongs to an aggregation and is the ratio of sums
# of the rows selected (in Power BI too). The estimates of the ladder are stored, as
# attributes of the series.
#
# GAPS are rows: a month with no expirations inside the history of a series (step 1.7) gets
# a row with every measure at 0 (origen_fila = hueco), so the months of a series are
# complete and the sums still reconcile with the extract.
#
# Actions (logged as they are done):
#   1. the rows of the extract (the fine table) with their raw and step-02 measures and their ids
#   2. the gap rows of step 1.7
#   3. the values of the forecast unit (step 1.5), if it has run
#   4. the forecast rows (step 7.2), the final block, and the forecast series dimension (one row per series)
#   5. the legend: every column, its step, its level and how to aggregate it
#   6. check the core against the extract, the legend and the total      checks 1-7
#   7. write the core, the dimension and the legend                      checks 8-10
#   8. count the checks; stop if any failed
#   9. show one series as it looks in the core, and how to read it in Power BI
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the forecast series dimension has one row per forecast series of the core
#    2. the exam sums of the dimension equal step 7.1's (each series once)
#    3. every original record and every synthetic row, once
#    4. the money reconciles with the extract (units and USD due, renewed as the raw had it)
#    5. every row has the values of every block present
#    6. the legend describes every column of the core
#    7. the core summed by year and origin = sff_forecast_total
#    8-10. tables sff_nucleo, sff_forecast_series and sff_nucleo_leyenda written and read back
#
# Output: the core table, the forecast series dimension and the legend · tables sff_nucleo, sff_forecast_series,
# sff_nucleo_leyenda.

STEP_COLUMN_PATTERN = re.compile(r"^s\d+_")      # a column of a step block of the core (s1_4_fs_id, s7_2_tasa…)

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_8_3_LABEL = "8.3"
STEP_8_3_NAME = "CORE TABLE"
STEP_8_3_PURPOSE = ("put the whole story of every row in one wide table, left to right in the order of the steps, "
                "with the gaps as explicit rows, so that selecting one forecast series shows everything the "
                "framework knows about it; a legend says how every column is aggregated")
STEP_8_3_ACTIONS = ["the rows of the extract (the fine table) with their raw and step-02 measures and their ids",
                "the gap rows of step 1.7",
                "the values of the forecast unit (steps 1.5, 2.1), if they have run",
                "the forecast rows (step 7.2), the final block, and the forecast series dimension (one row per series)",
                "the legend: every column, its step, its level and how to aggregate it",
                "check the core against the extract and the legend (checks 1-4)",
                "write the core and its legend (checks 5-6)",
                "count the checks; stop if any failed",
                "show one series as it looks in the core, and how to read it in Power BI"]
STEP_8_3_OUTPUT = "one row per fine row of the extract plus one per gap · tables sff_nucleo, sff_nucleo_leyenda"

# ─── named constants ─────────────────────────────────────────────────────────────
LEVEL_ROW, LEVEL_UNIT, LEVEL_SERIES = "fila", "unidad", "serie"
AGGREGATE_SUM = "SUM"
AGGREGATE_ATTRIBUTE = "no sumar: atributo repetido; con una serie o unidad seleccionada, MAX"
AGGREGATE_SLICER = "segmentar / filtrar"

# The measure columns of the core: (core name, step, description). The raw measures are as
# the extract had them; the step-02 renewals are the ones the framework learns from.
RAW_MEASURES = [("s1_1_vencen_unidades", "1.1", "units falling due, as in the extract"),
                ("s1_1_vencen_usd", "1.1", "USD falling due, as in the extract"),
                ("s1_1_renovadas_unidades", "1.1", "renewed units as the extract had them (early results of the future included)"),
                ("s1_1_renovado_usd", "1.1", "renewed USD as the extract had them (early results of the future included)")]
CALENDAR_MEASURES = [("s1_3_vencen_unidades", "1.3", "units due after the calendar: the 1-year licences sold or renewed from the current month on are 0 (not known yet, projected)"),
                     ("s1_3_vencen_usd", "1.3", "USD due after the calendar: the 1-year licences sold or renewed from the current month on are 0 (not known yet, projected)"),
                     ("s1_3_renovadas_unidades", "1.3", "renewed units after the calendar: null → 0 in closed months, wiped from the current month on"),
                     ("s1_3_renovado_usd", "1.3", "renewed USD after the calendar: null → 0 in closed months, wiped from the current month on")]

# steps 3.1-7.2, per FUTURE ROW (a value of the row: the money columns add up)
ROW_VALUES_STEP_7_2 = [(PIPELINE_ORIGIN_COLUMN, "s7_2_origen_pipeline", "7.2", "real (extract) · proyectada · simulada (a row of the "
                                                                             "maturation: the origin of the units it receives)"),
                      ("_vencen_unidades", "s7_2_vencen_unidades", "7.2", "units due of the future row TODAY: the extract's or the extended "
                                                                         "horizon's; 0 on a row of the maturation (SUM)"),
                      ("_vencen_usd", "s7_2_vencen_usd", "7.2", "USD due of the future row TODAY: the extract's or the extended horizon's; "
                                                              "0 on a row of the maturation (SUM)"),
                      (DUE_UNITS_AT_DUE_COLUMN, "s7_2_vencen_unidades_al_vencer", "7.2",
                       "units the row will have WHEN IT FALLS DUE, after the maturation of the states (stage 4.1): a neutral row "
                       "keeps what does not take a state, a row of the maturation has what it receives (SUM)"),
                      (DUE_USD_AT_DUE_COLUMN, "s7_2_vencen_usd_al_vencer", "7.2",
                       "USD due the row will have when it falls due (what leaves a neutral row leaves at its average price) (SUM)"),
                      ("maduracion_unidades", "s7_2_maduracion_unidades", "4.1",
                       "units at due date − units today: negative on a neutral row (they take a state), positive on a row of "
                       "the maturation (they arrive at its series with a state) (SUM)"),
                      ("h", "s7_2_h", "7.2", "months from the last closed month"),
                      ("origen_tasa", "s7_2_origen_tasa", "7.2", "pool (technique of its estimation id) · celda_mandatory · global"),
                      ("tecnica", "s7_2_tecnica", "7.2", "the technique that predicted the rate of this row"),
                      ("tasa", "s7_2_tasa", "7.2", "the predicted rate of the row (units)"),
                      ("tasa_baja", "s7_2_tasa_baja", "7.2", "low end of the rate band"),
                      ("tasa_alta", "s7_2_tasa_alta", "7.2", "high end of the rate band"),
                      ("confidence", "s7_2_confidence", "7.2", "high · medium · low: how sure the forecast is of the rate of the row"),
                      ("via_uplift", "s3_1_via_uplift", "3.1", "contrato (1/(1 − d)) · estadistica (its cell)"),
                      ("origen_uplift", "s3_1_origen_uplift", "7.2",
                       "where the uplift of the row comes from: celda (its row of sff_uplift_celda) · celda_mandatory or "
                       "global (a cell never seen, an extended row: blank in the relation with sff_uplift_celda) · contrato"),
                      ("uplift", "s3_1_uplift", "3.1", "the uplift applied to the row"),
                      ("esperado_unidades", "s7_2_esperado_unidades", "7.2", "expected renewed units (SUM)"),
                      ("esperado_usd", "s7_2_esperado_usd", "7.2", "expected renewed USD (SUM)"),
                      ("esperado_usd_bajo", "s7_2_esperado_usd_bajo", "7.2", "low end of the row's USD band (SUM = the worst case of a total)"),
                      ("esperado_usd_alto", "s7_2_esperado_usd_alto", "7.2", "high end of the row's USD band (SUM = the worst case of a total)"),
                      ("esperado_unidades_hoy", "s7_2_esperado_unidades_hoy", "7.2",
                       "expected renewed units with TODAY's states: units due today × rate (SUM)"),
                      ("esperado_usd_hoy", "s7_2_esperado_usd_hoy", "7.2",
                       "expected USD with TODAY's states: USD due today × rate × uplift (SUM)"),
                      ("maduracion_delta_unidades", "s7_2_maduracion_delta_unidades", "4.1",
                       "the maturation of the renewed units: already IN the forecast (expected = today's + this) (SUM)"),
                      ("maduracion_delta_usd", "s7_2_maduracion_delta_usd", "4.1",
                       "the maturation of the renewed USD: already IN the forecast (expected = today's + this) (SUM)")]

# The blocks of values of the forecast unit, of the series and of its estimation id:
# (source column, core name, step, description). A block enters when its step has run.
UNIT_VALUES_STEP_1_5 = [(FINE_ROWS_COLUMN, "s1_5_filas_finas", "1.5", "fine rows added into the unit")]
SERIES_VALUES_STEP_1_6 = [(ROUTE_COLUMN, "s1_6_ruta", "1.6", "predecible · solo_historia · solo_futuro"),
                         (COVERAGE_COLUMN, "s1_6_cobertura", "1.6", "the roles the series has, in time order"),
                         ("usd_por_predecir", "s1_6_usd_por_predecir", "1.6", "USD due in the months to predict, whole series")]
SERIES_VALUES_STEP_1_7 = [("n_propio", "s1_7_n_propio", "1.7", "units due in a typical closed month of the series"),
                         ("tasa_propia", "s1_7_tasa_propia", "1.7", "own rate of the series: Σ renewed / Σ due over its closed months"),
                         ("error_binomial_pp", "s1_7_error_binomial_pp", "1.7", "Wilson half-width of one month at n_propio (pp)"),
                         (SIGN_COLUMN, "s1_7_signo", "1.7", "neutro · negativo · positivo · mixto"),
                         ("bajo_suelo", "s1_7_bajo_suelo", "1.7", "1 when n_propio is below support_floor")]
# step 2.2, per forecast series: the 4 stages of the ladder (grouping by any stage id adds up to the raw)
SERIES_VALUES_STEP_2_2 = [("stage0_id", "s2_2_stage0_id", "2.2", "stage 0 · raw: the forecast series itself"),
                         ("stage0_support", "s2_2_stage0_support", "2.2", "stage 0 · its own support (units due in a typical month)"),
                         ("stage1_id", "s2_2_stage1_id", "2.2", "stage 1 · sign: its group after the signs are merged (repeats stage 0 if nothing changed)"),
                         ("stage1_support", "s2_2_stage1_support", "2.2", "stage 1 · the support of that group"),
                         ("stage2_id", "s2_2_stage2_id", "2.2", "stage 2 · extras: its group after the extras are annulled"),
                         ("stage2_support", "s2_2_stage2_support", "2.2", "stage 2 · the support of that group"),
                         ("stage3_id", "s2_2_stage3_id", "2.2", "stage 3 · collapse: its COMPOSITION, the group that is predicted"),
                         ("stage3_support", "s2_2_stage3_support", "2.2", "stage 3 · the support of its composition")]
# step 2.3, per forecast series: the rate of its composition and the credibility (stage 4)
SERIES_VALUES_STEP_2_3 = [("group_series", "s2_3_composition_series", "2.3", "forecast series in its composition"),
                         ("group_rate", "s2_3_composition_rate", "2.3", "rate of its composition"),
                         ("credibility_ref_id", "s2_3_credibility_ref_id", "2.3", "the wider group its rate is blended with (below own_rate_floor)"),
                         ("ref_support", "s2_3_ref_support", "2.3", "support of the credibility reference"),
                         ("ref_rate", "s2_3_ref_rate", "2.3", "rate of the credibility reference"),
                         ("credibility_effect_pp", "s2_3_credibility_effect_pp", "2.3", "how much the credibility moves the rate of its composition (pp); 0 when it predicts alone"),
                         ("alcanzo_suelo", "s2_3_alcanzo_suelo", "2.3", "1 when its final group reaches the support floor"),
                         ("k", "s2_3_k", "2.3", "Bühlmann k of the reference"),
                         ("z", "s2_3_z", "2.3", "credibility of the group's own rate: n / (n + k); 1 without a reference"),
                         ("tasa_estimada", "s2_3_tasa_estimada", "2.3", "diagnostic: z·group + (1 − z)·reference over the whole history (the forecast moves the technique's prediction by the same credibility, step 7.2)"),
                         ("se_estimacion_pp", "s2_3_se_estimacion_pp", "2.3", "error of the estimate (pp)"),
                         ("se_prediccion_pp", "s2_3_se_prediccion_pp", "2.3", "error of next month's prediction (pp): never below the series' own noise"),
                         ("nivel_riesgo", "s2_3_nivel_riesgo", "2.3", "how the rate was obtained: A_propio … N_sin_impacto")]
# per forecast series, from the audit step: its own dynamics (to cross the precision with it)
SERIES_VALUES_DYNAMICS = [("phi", "s2_5_series_phi", "2.5", "volatility of its own rate: observed variation / binomial noise (≈ 1: only chance)"),
                          ("trend", "s2_5_series_trend", "2.5", "+1 / −1 significant trend of its own rate, 0 none"),
                          ("trend_pp_year", "s2_5_series_trend_pp_year", "2.5", "slope of its own rate, pp per year"),
                          ("seasonal", "s2_5_series_seasonal", "2.5", "1 when its own rate has a month effect (≥ 24 months)"),
                          ("amplitude_pp", "s2_5_series_amplitude_pp", "2.5", "size of its month effect (pp)"),
                          ("measurable", "s2_5_series_measurable", "2.5", "yes · low_support (< 30 a month: not conclusive) · short_history"),
                          ("differs_from_composition", "s2_5_series_differs", "2.5", "1 when its trend or season differs from its composition's")]
# per forecast series: the exam of its chosen technique (every exam month and horizon), as ratios repeated on its rows
SERIES_VALUES_EXAM = [("framework_mae_pp", "s7_1_exam_mae_pp", "7.1", "mean absolute error of its rate in the exam, framework (pp)"),
                      ("framework_bias_pp", "s7_1_exam_bias_pp", "7.1", "mean error of its rate in the exam, framework (pp): + over-predicts"),
                      ("framework_wape", "s7_1_exam_wape", "7.1", "Σ |predicted − real| / Σ real renewed units in the exam, framework"),
                      ("framework_coverage", "s7_1_exam_coverage", "7.1", "share of its framework predictions whose real rate fell inside the interval"),
                      ("raw_mae_pp", "s7_1_raw_mae_pp", "7.1", "the same error predicting the series alone with its own history (raw)"),
                      ("raw_coverage", "s7_1_raw_coverage", "7.1", "share of its raw predictions inside their interval"),
                      ("improvement_mae_pp", "s7_1_improvement_mae_pp", "7.1", "raw error − framework error (pp): + the framework predicts it better"),
                      ("framework_rmse_pp", "s7_1_exam_rmse_pp", "7.1", "root mean square error of its rate in the exam, framework (pp)"),
                      ("framework_noise_pp", "s7_1_exam_noise_pp", "7.1", "binomial noise of its monthly rate: the error of a perfect prediction (pp)"),
                      ("framework_error_over_noise", "s7_1_exam_error_over_noise", "7.1",
                       "framework error / noise: ≈ 1 at the limit · well above 1, something knowable is missing"),
                      ("raw_error_over_noise", "s7_1_raw_error_over_noise", "7.1", "the same for the series alone (raw)")]
# … and as counts and units: in the dimension a SUM over any filter counts every forecast series once
SERIES_SUMMABLE_EXAM = [("framework_predictions", "s7_1_exam_predictions", "7.1", "exam predictions (SUM)"),
                        ("framework_in_band", "s7_1_exam_in_band", "7.1", "framework predictions inside their interval (SUM)"),
                        ("framework_pred_units", "s7_1_exam_pred_units", "7.1", "renewed units predicted by the framework (SUM)"),
                        ("framework_real_units", "s7_1_exam_real_units", "7.1", "renewed units real in the exam (SUM)"),
                        ("framework_abs_err_units", "s7_1_exam_abs_err_units", "7.1", "Σ |predicted − real| units, framework (SUM)"),
                        ("raw_in_band", "s7_1_raw_in_band", "7.1", "raw predictions inside their interval (SUM)"),
                        ("raw_abs_err_units", "s7_1_raw_abs_err_units", "7.1", "Σ |predicted − real| units, raw (SUM)"),
                        ("raw_real_units", "s7_1_raw_real_units", "7.1", "renewed units real in the exam, raw rows (SUM)"),
                        ("raw_predictions", "s7_1_raw_predictions", "7.1", "raw exam predictions (SUM)")]
# step 2.5, per estimation id (the dynamics of the rate the series takes)
ESTIMATION_VALUES_STEP_2_5 = [("phi", "s2_5_phi", "2.5", "φ of the estimation id: observed variation of its rate / binomial noise (≈ 1: nothing to model)"),
                             ("tendencia", "s2_5_tendencia", "2.5", "+1 / −1 significant trend of the rate, 0 none"),
                             ("tendencia_pp_ano", "s2_5_tendencia_pp_ano", "2.5", "slope of the rate, pp per year"),
                             ("estacional", "s2_5_estacional", "2.5", "1 when the month of the year is significant (on the rate without trend)"),
                             ("amplitud_pp", "s2_5_amplitud_pp", "2.5", "highest month minus lowest month (pp)"),
                             ("meses_alto", "s2_5_meses_alto", "2.5", "calendar months above the others beyond twice their error"),
                             ("meses_bajo", "s2_5_meses_bajo", "2.5", "calendar months below the others beyond twice their error")]
# step 2.6, per estimation id and horizon band (pivoted: one column per band)
ESTIMATION_VALUES_STEP_2_6 = [("tecnica", "s2_6_tecnica", "2.6", "the technique chosen for the estimation id in this horizon band"),
                             ("tecnica_origen", "s2_6_origen", "2.6", "campeon (beat the default technique) · por_defecto · sin_soporte (not judged)"),
                             ("elegida_err_pp_medio", "s2_6_examen_err_pp", "2.6", "mean |error| of the chosen technique in the exam months (pp of rate)"),
                             ("por_defecto_err_pp_medio", "s2_6_examen_por_defecto_err_pp", "2.6", "the same for the default technique: the one to beat"),
                             ("dentro_banda", "s2_6_examen_dentro_banda", "2.6", "share of the exam errors inside the band (nominal 90 %)")]


def build_core_table(fine_table: pd.DataFrame, configuration: Config, forecast_units: pd.DataFrame = None,
                     rated_units: pd.DataFrame = None,
                     series_table: pd.DataFrame = None, series_rate: pd.DataFrame = None,
                     series_estimate: pd.DataFrame = None, pool_dynamics: pd.DataFrame = None,
                     technique_decision: pd.DataFrame = None, exam_by_pool: pd.DataFrame = None,
                     forecast: pd.DataFrame = None, time_series_rows: pd.DataFrame = None,
                     time_series_table: pd.DataFrame = None, forecast_total: pd.DataFrame = None,
                     ladder_stages: pd.DataFrame = None, series_dynamics: pd.DataFrame = None,
                     series_exam: pd.DataFrame = None) -> tuple:
    """The core table with every block whose step has run, and its legend; checked and written.

    INPUT:   the fine table and the results of every step that has run (None = not run).
    OUTPUT:  (core, legend). The core has EVERY row: the original records of the pipeline and of the
             time_series universe, and the synthetic ones (gaps, extended horizon, ts_proyectado,
             ts_simulado, ts_reentrada); every decision as a column; and the FINAL block (forecast_*) that answers the
             business questions by summing it.
    RULES:   the forecast_* columns are the same for every row: pipeline due (forecast_to_renew_*) and renewed or
             revenue (forecast_renewed_*), real or expected, with its origin. A total of any year and origin
             is a SUM of the core; sff_forecast_total must equal it (check 5).
    EDGE CASES: a step not run leaves its block out; no time_series universe → no ts rows."""
    configuration.log_step_start(STEP_8_3_LABEL, STEP_8_3_NAME, STEP_8_3_PURPOSE, STEP_8_3_ACTIONS, STEP_8_3_OUTPUT)
    check_log = []
    dimension_columns = core_dimension_columns(configuration)
    blocks_present = []

    # [1] the rows of the extract
    raw_rows = rows_of_the_extract(fine_table, dimension_columns, configuration)
    configuration.log_action(STEP_8_3_LABEL, 1, f"{len(raw_rows):,} rows of the extract with {len(dimension_columns)} "
                                            f"dimension columns, their measures and their ids")

    # [2] the gap rows
    gap_rows = rows_of_the_gaps(rated_units, raw_rows.columns, configuration) if rated_units is not None else raw_rows.iloc[0:0]
    core = pd.concat([raw_rows, gap_rows], ignore_index=True) if len(gap_rows) else raw_rows
    extension_rows = rows_of_the_extension(forecast, core.columns, configuration) if forecast is not None else core.iloc[0:0]
    if len(extension_rows):
        core = pd.concat([core, extension_rows], ignore_index=True)
    time_series_core_rows = rows_of_the_time_series(time_series_rows, time_series_table, core.columns, configuration)
    if len(time_series_core_rows):
        core = pd.concat([core, time_series_core_rows], ignore_index=True)
    configuration.log_action(STEP_8_3_LABEL, 2, f"{len(gap_rows):,} gap rows · {len(extension_rows):,} extended-horizon rows · "
                                            f"{len(time_series_core_rows):,} time_series rows "
                                            f"({time_series_core_rows[ROW_ORIGIN_COLUMN].value_counts().to_dict() if len(time_series_core_rows) else {}})")

    # [3] the values of the forecast unit
    for block_frame, block_specs in ((forecast_units, UNIT_VALUES_STEP_1_5),):
        if block_frame is not None:
            core = add_block(core, block_frame, UNIT_ID_COLUMN, "s1_4_fu_id", block_specs)
            blocks_present.append(block_specs)
    configuration.log_action(STEP_8_3_LABEL, 3, f"unit blocks added: {[specs[0][2] for specs in blocks_present] or 'none'}")

    # [4] the forecast series dimension: one row per forecast series with everything of its level
    #     (route, own rate, the 4 stages, its composition and credibility, technique, dynamics, exam);
    #     the core keeps only what belongs to a row (or a unit) and joins it by s1_4_fs_id
    if forecast is not None:
        forecast_values = forecast.assign(_clave=forecast_keys(forecast), _vencen_unidades=forecast[configuration.pipeline_units_col],
                                          _vencen_usd=forecast[configuration.pipeline_usd_col])
        forecast_values = forecast_values[["_clave"] + [source for source, _, _, _ in ROW_VALUES_STEP_7_2]]
        forecast_values = forecast_values.rename(columns={source: name for source, name, _, _ in ROW_VALUES_STEP_7_2})
        core = core.merge(forecast_values, on="_clave", how="left")
        blocks_present.append(ROW_VALUES_STEP_7_2)
    core = core.drop(columns=["_fila", "_clave"], errors="ignore")
    core = add_final_block(core, configuration)
    blocks_present.append(FINAL_VALUES)
    if "s1_5_filas_finas" in core.columns:
        core.loc[core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP, "s1_5_filas_finas"] = 0     # a gap adds no fine row
    core = core.sort_values(["s1_4_fs_id", configuration.period_col, ROW_ORIGIN_COLUMN]).reset_index(drop=True)
    dimension = build_series_dimension(core, series_table, series_rate, series_estimate, ladder_stages, pool_dynamics,
                                       technique_decision, exam_by_pool, series_dynamics, series_exam, configuration)
    configuration.log_action(STEP_8_3_LABEL, 4, f"the core has {len(core):,} rows × {len(core.columns)} columns · the forecast series "
                                            f"dimension {len(dimension):,} rows × {len(dimension.columns)} columns, joined by s1_4_fs_id")

    # [5] the legend
    legend = core_legend(core, dimension_columns, blocks_present, configuration)
    legend = pd.concat([legend.assign(tabla="sff_" + TABLE_CORE), dimension_legend(dimension)], ignore_index=True)
    configuration.log_action(STEP_8_3_LABEL, 5, f"legend of {len(legend)} columns: {legend['nivel'].value_counts().to_dict()}")

    # [6] the checks
    configuration.log_action(STEP_8_3_LABEL, 6, "checking the core against the extract and the legend")
    check_core(core, fine_table, gap_rows, legend[legend["tabla"] == "sff_" + TABLE_CORE], blocks_present, forecast_total,
               configuration, check_log)
    check_dimension(core, dimension, series_exam, configuration, check_log)

    # [7] the tables, written
    configuration.log_action(STEP_8_3_LABEL, 7, "writing the core and its legend")
    configuration.write_table(STEP_8_3_LABEL, check_log, core, TABLE_CORE)
    configuration.write_table(STEP_8_3_LABEL, check_log, dimension, TABLE_FORECAST_SERIES)
    configuration.write_table(STEP_8_3_LABEL, check_log, legend, TABLE_CORE_LEGEND)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_8_3_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_8_3_LABEL, STEP_8_3_NAME, check_log)

    # [9] one series as it looks in the core, how to read it in Power BI, and the questions answered from it
    log_core_report(core, dimension, configuration)
    log_core_answers(core, configuration)
    return core, legend, dimension


DIMENSION_SUMMARY_VALUES = [("s7_2_esperado_usd_total", "7.2", "USD expected to renew in the months to predict (sum of its rows)"),
                            ("s7_2_pct_usd_high", "7.2", "share of its expected USD with confidence high"),
                            ("s7_2_pct_usd_medium", "7.2", "share of its expected USD with confidence medium"),
                            ("s7_2_pct_usd_low", "7.2", "share of its expected USD with confidence low"),
                            ("s7_2_confidence", "7.2", "the confidence of most of its expected USD")]


def build_series_dimension(core: pd.DataFrame, series_table, series_rate, series_estimate, ladder_stages, pool_dynamics,
                           technique_decision, exam_by_pool, series_dynamics, series_exam, configuration: Config) -> pd.DataFrame:
    """sff_forecast_series: one row per forecast series of the core, with every value of its level."""
    dimension = pd.DataFrame({"s1_4_fs_id": sorted(core["s1_4_fs_id"].dropna().unique())})
    ladder_stages = stages_of_every_series(ladder_stages, series_rate)
    for block_frame, block_specs in ((series_table, SERIES_VALUES_STEP_1_6), (series_rate, SERIES_VALUES_STEP_1_7),
                                     (ladder_stages, SERIES_VALUES_STEP_2_2), (series_estimate, SERIES_VALUES_STEP_2_3),
                                     (series_dynamics, SERIES_VALUES_DYNAMICS),
                                     (series_exam, SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM)):
        if block_frame is not None:
            dimension = add_block(dimension, block_frame, SERIES_ID_COLUMN, "s1_4_fs_id", block_specs)
    if pool_dynamics is not None and len(pool_dynamics) and "s2_2_stage3_id" in dimension.columns:
        dynamics_values = pool_dynamics[[COMPOSITION_ID_COLUMN] + [source for source, _, _, _ in ESTIMATION_VALUES_STEP_2_5]]
        dynamics_values = dynamics_values.rename(columns={source: name for source, name, _, _ in ESTIMATION_VALUES_STEP_2_5})
        dimension = dimension.merge(dynamics_values, left_on="s2_2_stage3_id", right_on=COMPOSITION_ID_COLUMN,
                                    how="left").drop(columns=COMPOSITION_ID_COLUMN)
    if technique_decision is not None and "s2_2_stage3_id" in dimension.columns:
        dimension, _ = add_backtest_block(dimension, technique_decision, exam_by_pool, configuration)
    # the money to predict and its confidence, from the rows of the core
    if "s7_2_confidence" in core.columns:
        future = core[core["s7_2_esperado_usd"].notna()]
        by_label = future.pivot_table(index="s1_4_fs_id", columns="s7_2_confidence", values="s7_2_esperado_usd", aggfunc="sum").fillna(0.0)
        total = by_label.sum(axis=1)
        summary = pd.DataFrame({"s7_2_esperado_usd_total": total})
        for label in ("high", "medium", "low"):
            summary[f"s7_2_pct_usd_{label}"] = (by_label[label] / total.where(total > 0)) if label in by_label.columns else 0.0
        summary["s7_2_confidence"] = by_label.idxmax(axis=1)
        dimension = dimension.merge(summary, left_on="s1_4_fs_id", right_index=True, how="left")
    return dimension


def dimension_legend(dimension: pd.DataFrame) -> pd.DataFrame:
    """The legend of sff_forecast_series: every column with its step and how to aggregate it."""
    described = {name: (step, description) for _, name, step, description in
                 SERIES_VALUES_STEP_1_6 + SERIES_VALUES_STEP_1_7 + SERIES_VALUES_STEP_2_2 + SERIES_VALUES_STEP_2_3
                 + SERIES_VALUES_DYNAMICS + SERIES_VALUES_EXAM + SERIES_SUMMABLE_EXAM + ESTIMATION_VALUES_STEP_2_5}
    described.update({name: (step, description) for name, step, description in DIMENSION_SUMMARY_VALUES})
    summable = {name for _, name, _, _ in SERIES_SUMMABLE_EXAM} | {"s1_6_usd_por_predecir", "s7_2_esperado_usd_total"}
    rows = [("s1_4_fs_id", "1.4", LEVEL_SERIES, "clave: relación 1 → n con sff_nucleo[s1_4_fs_id]", "the forecast series")]
    for column_name in dimension.columns[1:]:
        base = next((name for _, name, _, _ in ESTIMATION_VALUES_STEP_2_6 if column_name.startswith(name + "_")), None)
        step, description = described.get(column_name, ("2.6", next((d for _, n, _, d in ESTIMATION_VALUES_STEP_2_6 if n == base), ""))
                                          if base else ("", ""))
        rows.append((column_name, step, LEVEL_SERIES, AGGREGATE_SUM if column_name in summable else AGGREGATE_SLICER, description))
    legend = pd.DataFrame(rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
    return legend.assign(tabla="sff_" + TABLE_FORECAST_SERIES)


def check_dimension(core: pd.DataFrame, dimension: pd.DataFrame, series_exam, configuration: Config, check_log: list) -> None:
    """Check 6 and 7: one row per forecast series of the core; its exam sums equal step 7.1's."""
    configuration.log_check(STEP_8_3_LABEL, check_log, "the forecast series dimension has one row per forecast series of the core",
                            dimension["s1_4_fs_id"].is_unique and set(dimension["s1_4_fs_id"]) == set(core["s1_4_fs_id"].dropna()),
                            failure_detail="the dimension and the core do not have the same forecast series",
                            context=f"{len(dimension):,} forecast series")
    if series_exam is None or "s7_1_exam_real_units" not in dimension.columns:
        configuration.log_not_evaluated(STEP_8_3_LABEL, check_log, "the exam sums of the dimension equal step 7.1's", "no exam")
        return
    same = (abs(dimension["s7_1_exam_real_units"].sum() - series_exam["framework_real_units"].sum()) < 1e-6
            and int(dimension["s7_1_exam_predictions"].sum()) == int(series_exam["framework_predictions"].sum()))
    configuration.log_check(STEP_8_3_LABEL, check_log, "the exam sums of the dimension equal step 7.1's (each series once)", same,
                            failure_detail="the dimension lost or duplicated exam predictions",
                            context=f"{int(dimension['s7_1_exam_predictions'].sum()):,} exam predictions")


def add_block(core: pd.DataFrame, block_frame: pd.DataFrame, key_column: str, core_key: str, block_specs: list) -> pd.DataFrame:
    """The values of one block, joined to every row of the core by its unit or its series."""
    source_columns = [source for source, _, _, _ in block_specs]
    values = block_frame[[key_column] + source_columns].drop_duplicates(key_column)
    values = values.rename(columns={source: name for source, name, _, _ in block_specs})
    return core.merge(values, left_on=core_key, right_on=key_column, how="left").drop(columns=key_column)


def add_backtest_block(core: pd.DataFrame, technique_decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                       configuration: Config) -> tuple:
    """The chosen technique and the exam error of the row's estimation id, one column per horizon band."""
    per_band = technique_decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]]
    if exam_by_pool is not None:
        per_band = per_band.merge(exam_by_pool[[COMPOSITION_ID_COLUMN, "tramo_h", "elegida_err_pp_medio", "por_defecto_err_pp_medio",
                                                "dentro_banda"]], on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
    wide = per_band.pivot(index=COMPOSITION_ID_COLUMN, columns="tramo_h")
    specs = []
    renamed = {}
    for source, name, step, description in ESTIMATION_VALUES_STEP_2_6:
        for band_name in configuration.horizon_bands:
            if (source, band_name) in wide.columns:
                renamed[(source, band_name)] = f"{name}_{band_name}"
                specs.append((source, f"{name}_{band_name}", step, f"{description} (band {band_name})"))
    wide = wide[list(renamed)]
    wide.columns = [renamed[column] for column in wide.columns]
    core = core.merge(wide, left_on="s2_2_stage3_id", right_index=True, how="left")
    return core, specs


# ─── the final block: the same columns for every row, the ones the business questions sum ───
FINAL_UNIVERSE_COLUMN = "forecast_universe"
FINAL_ORIGIN_COLUMN = "forecast_pipeline_source"
FINAL_STATE_COLUMN = "forecast_status"
FINAL_YEAR_COLUMN = "forecast_year"
FINAL_UNIVERSE_PIPELINE, FINAL_UNIVERSE_TIME_SERIES = "pipeline", "time_series"
STATE_REAL, STATE_EXPECTED = "actual", "forecast"
TS_WITHOUT_RESULT = "ts_sin_resultado"       # an original time_series row of a month not closed yet (its result is projected)
FINAL_VALUES = [
    ("forecast_universe", "forecast_universe", "FIN", "pipeline (renewals) · time_series (retail to subscription)"),
    ("forecast_pipeline_source", "forecast_pipeline_source", "FIN", "pipeline_renovado_real · pipeline_real_esperado · pipeline_proyectada · pipeline_simulada · "
                                        "ts_real · ts_proyectado · ts_simulado · ts_reentrada · hueco · ts_sin_resultado"),
    ("forecast_status", "forecast_status", "FIN", "actual (already happened) · forecast (predicted)"),
    ("forecast_year", "forecast_year", "FIN", "the year of the month"),
    ("forecast_to_renew_units", "forecast_to_renew_units", "FIN", "PIPELINE: units falling due (extract, extended horizon, ts re-entry) · SUM"),
    ("forecast_to_renew_USD", "forecast_to_renew_USD", "FIN", "PIPELINE: USD falling due (acquisition and ts re-entry: at their discount) · SUM"),
    ("forecast_renewed_units", "forecast_renewed_units", "FIN", "RENEWED units: real in closed months, expected in the future; ts: converted units · SUM"),
    ("forecast_renewed_USD", "forecast_renewed_USD", "FIN", "RENEWED USD (or time_series revenue): real in closed months, expected in the future · SUM"),
    ("forecast_renewed_USD_low", "forecast_renewed_USD_low", "FIN", "low end of the row's band (SUM = the worst case of a total; real rows: the real value)"),
    ("forecast_renewed_USD_high", "forecast_renewed_USD_high", "FIN", "high end of the row's band (SUM = the worst case of a total; real rows: the real value)"),
    ("forecast_to_renew_units_at_due", "forecast_to_renew_units_at_due", "FIN",
     "PIPELINE WHEN IT FALLS DUE: the units of every series after the maturation of the states (stage 4.1); the same total "
     "as forecast_to_renew_units, moved from the neutral series to the ones with a state · SUM"),
    ("forecast_to_renew_USD_at_due", "forecast_to_renew_USD_at_due", "FIN",
     "PIPELINE WHEN IT FALLS DUE, in USD (what moves, moves at the average price of its neutral row) · SUM"),
    ("forecast_maturation_units", "forecast_maturation_units", "FIN",
     "units at due date − units today: − on a neutral series (they take a state), + on a in the state one · SUM"),
    ("forecast_renewed_units_maturation_adj", "forecast_renewed_units_maturation_adj", "FIN",
     "the part of forecast_renewed_units due to the states still to come (already IN it; 0 where nothing moved) · SUM"),
    ("forecast_renewed_USD_maturation_adj", "forecast_renewed_USD_maturation_adj", "FIN",
     "the part of forecast_renewed_USD due to the states still to come (already IN it): forecast − this = the "
     "forecast with today's states · SUM")]
FINAL_SUMMABLE = {"forecast_to_renew_units", "forecast_to_renew_USD", "forecast_renewed_units", "forecast_renewed_USD", "forecast_renewed_USD_low",
                  "forecast_renewed_USD_high", "forecast_to_renew_units_at_due", "forecast_to_renew_USD_at_due",
                  "forecast_maturation_units", "forecast_renewed_units_maturation_adj", "forecast_renewed_USD_maturation_adj"}


def rows_of_the_time_series(time_series_rows, time_series_table, core_columns, configuration: Config) -> pd.DataFrame:
    """The time_series universe in the core: its ORIGINAL records (ts_real in the closed months,
    ts_sin_resultado in the months not closed yet) and its SYNTHETIC rows (ts_proyectado, ts_simulado, ts_reentrada),
    with the band of its exam. Helper columns _ts_* carry their final values until the final block reads them."""
    frames = []
    helper = ["_ts_vence_unidades", "_ts_vence_usd", "_ts_renovadas", "_ts_usd", "_ts_usd_bajo", "_ts_usd_alto"]
    current = configuration.calendar_boundaries()["current"]
    if time_series_rows is not None and len(time_series_rows):
        original = pd.DataFrame({configuration.period_col: time_series_rows[configuration.period_col].to_numpy()})
        for column_name in core_columns:
            if column_name in time_series_rows.columns and column_name not in original.columns:
                original[column_name] = time_series_rows[column_name].to_numpy()
        closed = original[configuration.period_col] < current
        original[ROW_ORIGIN_COLUMN] = np.where(closed, TS_REAL, TS_WITHOUT_RESULT)
        original["s1_1_vencen_unidades"] = time_series_rows[configuration.pipeline_units_col].fillna(0).to_numpy()
        original["s1_1_vencen_usd"] = time_series_rows[configuration.pipeline_usd_col].fillna(0).to_numpy()
        original["s1_1_renovadas_unidades"] = time_series_rows[configuration.renewed_units_col].fillna(0).to_numpy()
        original["s1_1_renovado_usd"] = time_series_rows[configuration.renewed_usd_col].fillna(0).to_numpy()
        original["s1_3_rol"] = configuration.role_of_months(original[configuration.period_col])
        original["_ts_vence_unidades"], original["_ts_vence_usd"] = 0.0, 0.0
        original["_ts_renovadas"] = np.where(closed, original["s1_1_renovadas_unidades"], 0.0)
        original["_ts_usd"] = np.where(closed, original["s1_1_renovado_usd"], 0.0)
        original["_ts_usd_bajo"], original["_ts_usd_alto"] = original["_ts_usd"], original["_ts_usd"]
        frames.append(original)
    if time_series_table is not None and len(time_series_table):
        synthetic = time_series_table[time_series_table["origen"].isin([TS_PROJECTED, TS_SIMULATED, TS_REENTRY])]
        rows = pd.DataFrame({ROW_ORIGIN_COLUMN: synthetic["origen"].to_numpy(),
                             configuration.period_col: synthetic[configuration.period_col].to_numpy()})
        for column_name in core_columns:
            if column_name in synthetic.columns and column_name not in rows.columns:
                rows[column_name] = synthetic[column_name].to_numpy()
        for measure_name, _, _ in RAW_MEASURES:
            rows[measure_name] = 0.0
        rows["s1_3_rol"] = configuration.role_of_months(rows[configuration.period_col])
        reentry = (synthetic["origen"] == TS_REENTRY).to_numpy()
        rows["_ts_vence_unidades"] = np.where(reentry, synthetic["unidades"].to_numpy(), 0.0)
        rows["_ts_vence_usd"] = np.where(reentry, synthetic["valor"].to_numpy(), 0.0)      # the pipeline: value at 40 % off
        rows["_ts_renovadas"] = np.where(reentry, synthetic.get("unidades_renovadas", pd.Series(0.0, index=synthetic.index)).fillna(0).to_numpy(),
                                         synthetic["unidades"].to_numpy())
        rows["_ts_usd"] = synthetic["revenue"].to_numpy()
        rows["_ts_usd_bajo"] = synthetic.get("revenue_bajo", synthetic["revenue"]).to_numpy()
        rows["_ts_usd_alto"] = synthetic.get("revenue_alto", synthetic["revenue"]).to_numpy()
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=list(core_columns))
    return pd.concat(frames, ignore_index=True).reindex(columns=list(core_columns) + helper)


def add_final_block(core: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The final block: universe, origin, state, year, pipeline due and renewed (or revenue), row by row."""
    origin = core[ROW_ORIGIN_COLUMN]
    closed = core["s1_3_rol"].isin(TRUTH_ROLES_OF_CORE)
    is_time_series = origin.astype(str).str.startswith("ts_")
    zeros = pd.Series(0.0, index=core.index)
    column = lambda name: core[name].fillna(0.0) if name in core.columns else zeros

    raw_closed, raw_future = (origin == ROW_FROM_RAW) & closed, (origin == ROW_FROM_RAW) & ~closed
    extended = origin.isin([PIPELINE_PROJECTED, PIPELINE_SIMULATED])
    # a row of the maturation: a prediction (no data); its money belongs to the origin of the units it receives
    matured = origin == ROW_FROM_MATURATION
    pipeline_of_row = core["s7_2_origen_pipeline"] if "s7_2_origen_pipeline" in core.columns else pd.Series(None, index=core.index)
    core[FINAL_UNIVERSE_COLUMN] = np.where(is_time_series, FINAL_UNIVERSE_TIME_SERIES, FINAL_UNIVERSE_PIPELINE)
    core[FINAL_ORIGIN_COLUMN] = np.select(
        [raw_closed, raw_future, origin == PIPELINE_PROJECTED, origin == PIPELINE_SIMULATED, origin == ROW_FROM_GAP,
         matured & (pipeline_of_row == PIPELINE_REAL), matured & (pipeline_of_row == PIPELINE_PROJECTED),
         matured & (pipeline_of_row == PIPELINE_SIMULATED)],
        [TOTAL_ORIGIN_RENEWED, TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_SIMULATED, ROW_FROM_GAP,
         TOTAL_ORIGIN_EXPECTED, TOTAL_ORIGIN_PROJECTED, TOTAL_ORIGIN_SIMULATED],
        default=origin.astype(str))
    extended = extended | matured
    core[FINAL_STATE_COLUMN] = np.where(raw_closed | (origin == ROW_FROM_GAP) | origin.isin([TS_REAL, TS_WITHOUT_RESULT]),
                                        STATE_REAL, STATE_EXPECTED)
    core[FINAL_YEAR_COLUMN] = core[configuration.period_col].map(lambda month: month.year)

    extract_rows = origin.isin([ROW_FROM_RAW, ROW_FROM_GAP])            # the pipeline after the calendar (s02): not what is not known yet
    core["forecast_to_renew_units"] = np.select([extract_rows, extended, is_time_series],
                                           [column("s1_3_vencen_unidades"), column("s7_2_vencen_unidades"), column("_ts_vence_unidades")], 0.0)
    core["forecast_to_renew_USD"] = np.select([extract_rows, extended, is_time_series],
                                      [column("s1_3_vencen_usd"), column("s7_2_vencen_usd"), column("_ts_vence_usd")], 0.0)
    core["forecast_renewed_units"] = np.select([raw_closed, raw_future | extended, is_time_series],
                                               [column("s1_3_renovadas_unidades"), column("s7_2_esperado_unidades"), column("_ts_renovadas")], 0.0)
    core["forecast_renewed_USD"] = np.select([raw_closed, raw_future | extended, is_time_series],
                                         [column("s1_3_renovado_usd"), column("s7_2_esperado_usd"), column("_ts_usd")], 0.0)
    with_band = raw_future | extended
    # the band: the forecast's for its rows; the exam of the time_series simulation for its rows (none on ts_real)
    core["forecast_renewed_USD_low"] = np.select([with_band, is_time_series], [column("s7_2_esperado_usd_bajo"), column("_ts_usd_bajo")],
                                                 core["forecast_renewed_USD"])
    core["forecast_renewed_USD_high"] = np.select([with_band, is_time_series], [column("s7_2_esperado_usd_alto"), column("_ts_usd_alto")],
                                                  core["forecast_renewed_USD"])
    # the pipeline when it falls due and the maturation: next to the forecast (closed rows: the pipeline as it was)
    core["forecast_to_renew_units_at_due"] = np.where(raw_future | extended, column("s7_2_vencen_unidades_al_vencer"),
                                                      core["forecast_to_renew_units"])
    core["forecast_to_renew_USD_at_due"] = np.where(raw_future | extended, column("s7_2_vencen_usd_al_vencer"),
                                                    core["forecast_to_renew_USD"])
    core["forecast_maturation_units"] = np.where(raw_future | extended, column("s7_2_maduracion_unidades"), 0.0)
    core["forecast_renewed_units_maturation_adj"] = np.where(raw_future | extended, column("s7_2_maduracion_delta_unidades"), 0.0)
    core["forecast_renewed_USD_maturation_adj"] = np.where(raw_future | extended, column("s7_2_maduracion_delta_usd"), 0.0)
    # the closed-month measures: observed, not predicted: only in the closed months of the extract
    for measure in configuration.closed_month_measures:
        core[measure.core_column] = np.where(raw_closed, column(measure.s1_3_column), np.nan)
    return core.drop(columns=[name for name in core.columns if name.startswith("_ts_")])


def stages_of_every_series(ladder_stages, series_rate):
    """The 4 stages of every forecast series: the estimable ones as the ladder left them; the others
    (solo_historia, solo_futuro: they do not climb) are themselves in every stage, with their own support."""
    if ladder_stages is None or series_rate is None:
        return ladder_stages
    missing = series_rate[~series_rate[SERIES_ID_COLUMN].isin(ladder_stages[SERIES_ID_COLUMN])]
    if missing.empty:
        return ladder_stages
    themselves = pd.DataFrame({SERIES_ID_COLUMN: missing[SERIES_ID_COLUMN].to_numpy()})
    for stage in range(4):
        themselves[f"stage{stage}_id"] = missing[SERIES_ID_COLUMN].to_numpy()
        themselves[f"stage{stage}_support"] = missing["n_propio"].to_numpy()
    return pd.concat([ladder_stages, themselves], ignore_index=True)


def core_dimension_columns(configuration: Config) -> list:
    """The dimension columns of the core, as the extract names them (they are the slicers):
    the rate series columns, the revaluation extras, the exact discount and its bucket."""
    discount = [configuration.discount_value_column, configuration.discount_bucket_column] if configuration.discount_value_column else []
    return list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion + discount))


def rows_of_the_extract(fine_table: pd.DataFrame, dimension_columns: list, configuration: Config) -> pd.DataFrame:
    """One row per fine row: its month, its dimensions, its measures (raw and after the
    calendar), its role and its ids in text (no hash keys)."""
    raw_rows = pd.DataFrame({ROW_ORIGIN_COLUMN: ROW_FROM_RAW, configuration.period_col: fine_table[configuration.period_col]})
    for column_name in dimension_columns:
        raw_rows[column_name] = fine_table[column_name]
    raw_rows["s1_1_vencen_unidades"] = fine_table[S0_PIPELINE_UNITS_COLUMN]
    raw_rows["s1_1_vencen_usd"] = fine_table[S0_PIPELINE_USD_COLUMN]
    raw_rows["s1_3_vencen_unidades"] = fine_table[configuration.pipeline_units_col]
    raw_rows["s1_3_vencen_usd"] = fine_table[configuration.pipeline_usd_col]
    raw_rows["s1_1_renovadas_unidades"] = fine_table[S0_RENEWED_UNITS_COLUMN]
    raw_rows["s1_1_renovado_usd"] = fine_table[S0_RENEWED_USD_COLUMN]
    raw_rows["s1_3_rol"] = fine_table[CALENDAR_ROLE_COLUMN]
    raw_rows["s1_3_es_mes_en_curso"] = fine_table[CURRENT_MONTH_COLUMN]
    raw_rows["s1_3_renovadas_unidades"] = fine_table[configuration.renewed_units_col]
    raw_rows["s1_3_renovado_usd"] = fine_table[configuration.renewed_usd_col]
    for measure in configuration.closed_month_measures:
        raw_rows[measure.s1_3_column] = fine_table[measure.raw_column]
    raw_rows["s1_4_fs_id"] = fine_table[SERIES_ID_COLUMN]
    raw_rows["s1_4_fu_id"] = fine_table[UNIT_ID_COLUMN]
    raw_rows["s1_4_uplift_cell_id"] = fine_table[UPLIFT_CELL_ID_COLUMN]
    raw_rows["_fila"] = fine_table.index      # the position of the fine row: the forecast joins on it, then it is dropped
    raw_rows["_clave"] = [f"f{position}" for position in fine_table.index]
    return raw_rows.reset_index(drop=True)


def forecast_keys(forecast: pd.DataFrame) -> list:
    """The join key of every forecast row: 'f<position>' for a row of the extract, 'e<n>' for an extended row."""
    return [f"f{int(position)}" if pd.notna(position) else f"e{number}"
            for number, position in enumerate(forecast["_fila"])]


def rows_of_the_extension(forecast: pd.DataFrame, core_columns, configuration: Config) -> pd.DataFrame:
    """One row per future row that is not in the extract: the extended horizon (proyectada, simulada) and the
    prediction rows of the maturation (maduracion): its month, dims and ids; the measures of the extract at 0 (it is
    not in the extract: its pipeline is in s7_2_vencen_*, at due date in s7_2_vencen_*_al_vencer)."""
    keys = forecast_keys(forecast)
    extension = forecast[forecast["_fila"].isna()]
    if extension.empty:
        return pd.DataFrame(columns=core_columns)
    is_maturation = (extension[MATURATION_ROW_COLUMN] == 1).to_numpy() if MATURATION_ROW_COLUMN in extension.columns \
        else np.zeros(len(extension), dtype=bool)
    rows = pd.DataFrame({ROW_ORIGIN_COLUMN: np.where(is_maturation, ROW_FROM_MATURATION, extension[PIPELINE_ORIGIN_COLUMN].to_numpy()),
                         configuration.period_col: extension[configuration.period_col].to_numpy()})
    for column_name in core_columns:
        # the dimensions of the extract (softcancel, sku… too); the step blocks (s1_1_, s1_4_, s7_2_…) are filled below
        if column_name in extension.columns and column_name not in rows.columns and not STEP_COLUMN_PATTERN.match(column_name):
            rows[column_name] = extension[column_name].to_numpy()
    for measure_name, _, _ in RAW_MEASURES:
        rows[measure_name] = 0.0
    rows["s1_3_rol"] = extension[CALENDAR_ROLE_COLUMN].to_numpy()
    rows["s1_3_es_mes_en_curso"] = 0
    rows["s1_4_fs_id"] = extension[SERIES_ID_COLUMN].to_numpy()
    rows["s1_4_fu_id"] = extension[UNIT_ID_COLUMN].to_numpy()
    rows["s1_4_uplift_cell_id"] = extension[UPLIFT_CELL_ID_COLUMN].to_numpy()
    rows["_clave"] = [key for key, position in zip(keys, forecast["_fila"]) if pd.isna(position)]
    return rows.reindex(columns=list(core_columns))


def rows_of_the_gaps(rated_units: pd.DataFrame, core_columns, configuration: Config) -> pd.DataFrame:
    """One row per gap of step 1.7: its month, its series columns, every measure at 0, its
    role; the revaluation columns are empty (a gap has no contract)."""
    gaps = rated_units[rated_units[SYNTHETIC_COLUMN] == 1]
    if gaps.empty:
        return pd.DataFrame(columns=core_columns)
    gap_rows = pd.DataFrame({ROW_ORIGIN_COLUMN: ROW_FROM_GAP, configuration.period_col: gaps[configuration.period_col].to_numpy()})
    for column_name in configuration.rate_series_columns:
        gap_rows[column_name] = gaps[column_name].to_numpy()
    for measure_name, _, _ in RAW_MEASURES + CALENDAR_MEASURES:
        gap_rows[measure_name] = 0.0
    gap_rows["s1_3_rol"] = gaps[CALENDAR_ROLE_COLUMN].to_numpy()
    gap_rows["s1_3_es_mes_en_curso"] = 0
    gap_rows["s1_4_fs_id"] = gaps[SERIES_ID_COLUMN].to_numpy()
    gap_rows["s1_4_fu_id"] = gaps[UNIT_ID_COLUMN].to_numpy()
    return gap_rows.reindex(columns=core_columns)


def core_legend(core: pd.DataFrame, dimension_columns: list, blocks_present: list, configuration: Config) -> pd.DataFrame:
    """Every column of the core with its step, its level and how to aggregate it."""
    legend_rows = [(ROW_ORIGIN_COLUMN, "8.3", LEVEL_ROW, AGGREGATE_SLICER, "raw = a row of the extract · hueco = a month with nothing "
                    "due inside a history · proyectada / simulada = a row of the extended horizon (renewal falling due again / "
                    "last year's acquisition) · maduracion = a prediction row of the stage 4.1: the units of a neutral series "
                    "that will carry a state when they fall due, in their series with a state (no data: its s00/s02 measures are 0) · "
                    "ts_real / ts_sin_resultado = an original row of the time_series universe · "
                    "ts_proyectado / ts_reentrada = its synthetic rows"),
                   (configuration.period_col, "1.1", LEVEL_ROW, AGGREGATE_SLICER, "the month")]
    for column_name in dimension_columns:
        description = ("the exact discount (share)" if column_name == configuration.discount_value_column
                       else "the discount bucket, derived in step 1.4" if column_name == configuration.discount_bucket_column
                       else "a dimension of the extract")
        legend_rows.append((column_name, "1.4" if column_name == configuration.discount_bucket_column else "1.1",
                            LEVEL_ROW, AGGREGATE_SLICER, description))
    for name, step, description in RAW_MEASURES:
        legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
    legend_rows += [("s1_3_rol", "1.3", LEVEL_ROW, AGGREGATE_SLICER, "entrenamiento · examen · proyeccion"),
                    ("s1_3_es_mes_en_curso", "1.3", LEVEL_ROW, AGGREGATE_SLICER, "1 in the current month")]
    for name, step, description in CALENDAR_MEASURES:
        legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM, description))
    legend_rows += [("s1_4_fs_id", "1.4", LEVEL_ROW, AGGREGATE_SLICER, "the rate series: select one to see its story"),
                    ("s1_4_fu_id", "1.4", LEVEL_ROW, AGGREGATE_SLICER, "the forecast unit: the series in its month"),
                    ("s1_4_uplift_cell_id", "1.4", LEVEL_ROW, AGGREGATE_SLICER, "the price context (uplift cell)")]
    unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_1_5}
    row_names = {name for _, name, _, _ in ROW_VALUES_STEP_7_2}
    summable = {"s7_2_esperado_unidades", "s7_2_esperado_usd", "s7_2_esperado_usd_bajo", "s7_2_esperado_usd_alto",
                "s7_2_vencen_unidades", "s7_2_vencen_usd", "s7_2_vencen_unidades_al_vencer", "s7_2_vencen_usd_al_vencer",
                "s7_2_maduracion_unidades", "s7_2_maduracion_delta_unidades", "s7_2_esperado_unidades_hoy", "s7_2_esperado_usd_hoy",
                "s7_2_maduracion_delta_usd"}
    for block_specs in blocks_present:
        for _, name, step, description in block_specs:
            if name in {spec_name for _, spec_name, _, _ in FINAL_VALUES}:
                legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM if name in FINAL_SUMMABLE else AGGREGATE_SLICER,
                                    description))
                continue
            if name in row_names:
                legend_rows.append((name, step, LEVEL_ROW, AGGREGATE_SUM if name in summable else AGGREGATE_SLICER
                                    if name in ("s7_2_origen_tasa", "s7_2_tecnica", "s3_1_via_uplift", "s7_2_h")
                                    else "no sumar: valor de la fila (una tasa o un uplift)", description))
                continue
            if name in {spec_name for _, spec_name, _, _ in SERIES_SUMMABLE_EXAM}:
                legend_rows.append((name, step, LEVEL_SERIES, AGGREGATE_SUM, description))
                continue
            level = LEVEL_UNIT if name in unit_names else LEVEL_SERIES
            legend_rows.append((name, step, level, AGGREGATE_ATTRIBUTE, description))
    for measure in configuration.closed_month_measures:
        legend_rows.append((measure.s1_3_column, "1.3", LEVEL_ROW, AGGREGATE_SUM,
                            f"{measure.config_field} ({measure.raw_column}) after the calendar (closed months; wiped "
                            f"from the current month on)"))
        legend_rows.append((measure.core_column, "FIN", LEVEL_ROW, AGGREGATE_SUM,
                            f"{measure.config_field} ({measure.raw_column}): observed in the closed months of the "
                            f"extract, empty elsewhere (it is only known once the month closes) · SUM"))
    legend = pd.DataFrame(legend_rows, columns=["columna", "paso", "nivel", "como_agregar", "descripcion"])
    return legend[legend["columna"].isin(core.columns)].drop_duplicates("columna").reset_index(drop=True)


def check_core(core: pd.DataFrame, fine_table: pd.DataFrame, gap_rows: pd.DataFrame, legend: pd.DataFrame,
               blocks_present: list, forecast_total, configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    # [1] rows: the extract, the gaps, the extended horizon and the time_series universe
    origin_counts = core[ROW_ORIGIN_COLUMN].value_counts().to_dict()
    configuration.log_check(STEP_8_3_LABEL, check_log, "every original record and every synthetic row, once",
                            int((core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW).sum()) == len(fine_table)
                            and int((core[ROW_ORIGIN_COLUMN] == ROW_FROM_GAP).sum()) == len(gap_rows),
                            failure_detail=f"rows by origin {origin_counts} for {len(fine_table):,} fine rows and {len(gap_rows):,} gaps",
                            context=" · ".join(f"{origin} {count:,}" for origin, count in origin_counts.items()))

    # [2] the money of the pipeline reconciles with the extract (the time_series rows are apart)
    pipeline_rows = core[FINAL_UNIVERSE_COLUMN] == FINAL_UNIVERSE_PIPELINE
    reconciliation = {"s1_1_vencen_unidades": fine_table[S0_PIPELINE_UNITS_COLUMN].sum(),
                      "s1_1_vencen_usd": fine_table[S0_PIPELINE_USD_COLUMN].sum(),
                      "s1_3_vencen_unidades": fine_table[configuration.pipeline_units_col].sum(),
                      "s1_3_vencen_usd": fine_table[configuration.pipeline_usd_col].sum(),
                      "s1_1_renovadas_unidades": fine_table[S0_RENEWED_UNITS_COLUMN].sum(),
                      "s1_1_renovado_usd": fine_table[S0_RENEWED_USD_COLUMN].sum()}
    differences = {name: float(core.loc[pipeline_rows, name].sum() - total) for name, total in reconciliation.items()
                   if abs(core.loc[pipeline_rows, name].sum() - total) > configuration.money_tolerance}
    configuration.log_check(STEP_8_3_LABEL, check_log, "the money reconciles with the extract", not differences,
                            failure_detail=f"differences core − extract: {differences}",
                            context=f"${reconciliation['s1_1_vencen_usd']:,.0f} due · ${reconciliation['s1_1_renovado_usd']:,.0f} renewed")

    # [3] every row has the values of every block present (the first column of each block);
    #     a gap is not a unit of step 1.5, so the unit blocks are checked on the rows of the extract
    unit_names = {name for _, name, _, _ in UNIT_VALUES_STEP_1_5}
    first_columns = [block_specs[0][1] for block_specs in blocks_present if block_specs and block_specs[0][1] in core.columns
                     and not block_specs[0][1].startswith(("s2_5_", "s2_6_", "s7_2_", "s7_1_", "forecast_"))]
    series_columns = [name for name in first_columns if name not in unit_names]
    unit_columns = [name for name in first_columns if name in unit_names]
    raw_rows = core[ROW_ORIGIN_COLUMN] == ROW_FROM_RAW
    history_rows = core[ROW_ORIGIN_COLUMN].isin([ROW_FROM_RAW, ROW_FROM_GAP])       # an extended row may be a new series
    without_values = (int(core.loc[history_rows, series_columns].isna().any(axis=1).sum()) if series_columns else 0) + \
                     (int(core.loc[raw_rows, unit_columns].isna().any(axis=1).sum()) if unit_columns else 0)
    configuration.log_check(STEP_8_3_LABEL, check_log, "every row has the values of every block present", without_values == 0,
                            failure_detail=f"{without_values:,} rows without some block's values",
                            context=f"{len(blocks_present)} blocks")

    # [4] the legend covers the core
    undescribed = [column_name for column_name in core.columns if column_name not in set(legend["columna"])]
    configuration.log_check(STEP_8_3_LABEL, check_log, "the legend describes every column of the core", not undescribed,
                            failure_detail=f"columns without legend: {undescribed}",
                            context=f"{len(core.columns)} columns")

    # [5] the core answers the totals: summed by year and origin it equals sff_forecast_total
    if forecast_total is None:
        configuration.log_not_evaluated(STEP_8_3_LABEL, check_log, "the core summed by year and origin = sff_forecast_total",
                                        "step 7.3 has not run")
        return
    parts = forecast_total[forecast_total["origen"] != TOTAL_ORIGIN_TOTAL].set_index(["ano", "origen"])
    summed = core.groupby([FINAL_YEAR_COLUMN, FINAL_ORIGIN_COLUMN])[["forecast_to_renew_USD", "forecast_renewed_USD"]].sum()
    summed.index = summed.index.set_names(["ano", "origen"])
    compared = parts.join(summed, how="left").fillna(0.0)
    mismatched = compared[((compared["usd_vence"] - compared["forecast_to_renew_USD"]).abs() > configuration.money_tolerance)
                          | ((compared["usd_renovado"] - compared["forecast_renewed_USD"]).abs() > configuration.money_tolerance)]
    configuration.log_check(STEP_8_3_LABEL, check_log, "the core summed by year and origin = sff_forecast_total (the total comes from the core)",
                            mismatched.empty, failure_detail=f"{len(mismatched)} year × origin differ",
                            context=f"{len(compared)} year × origin compared",
                            examples=mismatched.reset_index()[["ano", "origen", "usd_vence", "forecast_to_renew_USD", "usd_renovado",
                                                               "forecast_renewed_USD"]])


def log_core_answers(core: pd.DataFrame, configuration: Config) -> None:
    """The business questions answered by SUMMING the core: renewed and pipeline by year, state and origin."""
    current_year = configuration.calendar_boundaries()["current"].year
    answers = (core[core[FINAL_YEAR_COLUMN] >= current_year]
               .groupby([FINAL_YEAR_COLUMN, FINAL_STATE_COLUMN, FINAL_ORIGIN_COLUMN])[["forecast_to_renew_USD", "forecast_renewed_USD"]].sum()
               .reset_index())
    answers = answers[(answers["forecast_to_renew_USD"] != 0) | (answers["forecast_renewed_USD"] != 0)]
    configuration.logger.doc(f"[{STEP_8_3_LABEL}] the questions answered by SUMMING the core (forecast_renewed_USD: renewed or revenue, "
                             f"real or expected · forecast_to_renew_USD: pipeline), by year, state and origin:")
    configuration.show_table(answers)


def log_core_report(core: pd.DataFrame, dimension: pd.DataFrame, configuration: Config) -> None:
    """Action 9: one series as it looks in the two tables, and how to read them in Power BI."""
    largest_series = core.groupby("s1_4_fs_id")["s1_1_vencen_usd"].sum().idxmax()
    configuration.log_action(STEP_8_3_LABEL, 9, f"the series with the most money, '{largest_series}': its row of sff_forecast_series "
                                            f"and its last {configuration.core_months_shown} closed months in sff_nucleo:")
    shown = ["s1_4_fs_id", "s1_6_ruta", "s1_7_n_propio", "s1_7_tasa_propia", "s2_2_stage3_id", "s2_2_stage3_support", "s2_3_tasa_estimada",
             "s2_3_nivel_riesgo", "s7_1_exam_mae_pp", "s7_1_raw_mae_pp", "s7_1_exam_coverage", "s7_2_confidence"]
    configuration.show_table(dimension[dimension["s1_4_fs_id"] == largest_series][[column for column in shown if column in dimension.columns]])
    series_rows = core[(core["s1_4_fs_id"] == largest_series) & core["s1_3_rol"].isin([ROLE_TRAIN, ROLE_TEST])]
    shown_columns = [configuration.period_col, ROW_ORIGIN_COLUMN, "s1_3_rol", "s1_1_vencen_unidades", "s1_3_renovadas_unidades"]
    configuration.show_table(series_rows.tail(configuration.core_months_shown)[[column for column in shown_columns if column in core.columns]])
    configuration.logger.doc(f"[{STEP_8_3_LABEL}] in Power BI: relate sff_forecast_series[s1_4_fs_id] 1 → n sff_nucleo[s1_4_fs_id]; a "
                             f"slicer on the dimension selects forecast series (by id, confidence, risk level, stage, exam…) and "
                             f"filters their rows. A rate is a measure (a ratio of sums), never a column. Measures to create:")
    configuration.show_table(pd.DataFrame([
        ("Tasa renovación (meses cerrados)",
         "DIVIDE(CALCULATE(SUM(sff_nucleo[s1_3_renovadas_unidades]), sff_nucleo[s1_3_rol] IN {\"entrenamiento\", \"examen\"}), "
         "CALCULATE(SUM(sff_nucleo[s1_1_vencen_unidades]), sff_nucleo[s1_3_rol] IN {\"entrenamiento\", \"examen\"}))"),
        ("Vence USD", "SUM(sff_nucleo[s1_1_vencen_usd])"),
        ("Renovado (actual + forecast)", "SUM(sff_nucleo[forecast_renewed_USD])  — segmentar por forecast_year, forecast_status, forecast_pipeline_source"),
        ("Pipeline", "SUM(sff_nucleo[forecast_to_renew_USD])  — segmentar por forecast_year, forecast_pipeline_source"),
        ("Examen: dentro del intervalo", "DIVIDE(SUM(sff_forecast_series[s7_1_exam_in_band]), SUM(sff_forecast_series[s7_1_exam_predictions]))"),
        ("Examen: WAPE framework / raw", "DIVIDE(SUM(sff_forecast_series[s7_1_exam_abs_err_units]), SUM(sff_forecast_series[s7_1_exam_real_units]))"
                                        "  ·  the same with s7_1_raw_abs_err_units")], columns=["medida", "DAX"]))


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 8.4 · ADAPTATION TO POWER BI  (until 25-oct-2026: step_25_power_bi.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The adaptation to Power BI: the auxiliary tables the report needs, so that
# Power BI only relates and sums.
#
# The core (sff_nucleo) and the forecast series dimension (sff_forecast_series) carry the numbers. What
# Power BI still needs is the context to show them to business: a name for every code, an order, a block.
# That context is built here, once, in Python, instead of as calculated tables or DAX inside the report.
#
# Tables written (one per auxiliary table; this step will grow with the report):
#
#   sff_price_increase_calendar    one row per increase detected: price group, start month, size of the
#                                  step, uplift before and during the cycle, the renewed USD it touched
#   sff_price_increase_monitor     one row per PRICE GROUP (price_group_dims: a tariff changes by region and
#                                  product) and closed pipeline month: the uplift of everyone, the uplift
#                                  of the ISOLATED renewals (the detector: the normal renewal process, no
#                                  retention discounts in it), the step against the average of the previous 12
#                                  months, and the increase cycle (a price increase affects the renewals
#                                  of ONE cycle: who renews pays the new price, but a year later everyone
#                                  already bought at it, so the step lasts configuration.price_cycle_months and then fades)
#   sff_parametros                 one row per field of the Config: its value in this run, its default,
#                                  whether it was changed, the steps whose code reads it and the steps
#                                  its comment documents, and the comment (what it decides)
#   sff_relaciones_modelo          one row per relation of the Power BI model (MODEL_RELATIONS): the table and
#                                  key of the 1 side, the table and column of the n side, what it is for, and
#                                  what the data of this run says of it: rows on each side, whether the key is
#                                  unique and how many rows of the n side find no key (they would go blank)
#   sff_forecast_pipeline_source   one row per value of forecast_pipeline_source (the origin of every
#                                  row of the core): its business label, the block it belongs to and its
#                                  order in the report. Relation in Power BI:
#                                      sff_forecast_pipeline_source[forecast_pipeline_source]
#                                          1 → n  sff_nucleo[forecast_pipeline_source]
#
# Actions (logged as they are done):
#   1. the table of pipeline sources: code, label, block, order
#   2. the price increase monitor and its calendar: the uplift of the closed months and its steps
#   3. the parameters of this run: every field of the Config, its value and the steps that read it
#   4. check the tables against the core                                   checks 1-4
#   5. write them                                                          checks 5-8
#   6. the relations of the Power BI model, checked on the tables this run wrote      checks 9-11
#   7. count the checks; stop if any failed
#   8. show the tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every forecast_pipeline_source of the core has its row (no row of the core left without a label)
#    2. one row per source (the key is unique: the relation in Power BI is 1 → n)
#    3. every closed pipeline month with renewals has its uplift in the monitor
#    4. an increase is only flagged against a reference of enough previous months
#    5-8. tables sff_forecast_pipeline_source, sff_price_increase_monitor, sff_price_increase_calendar and
#         sff_parametros written and read back
#    9. every relation of the model joins columns that exist in tables this run wrote
#   10. the key of the 1 side of every relation is unique (else Power BI cannot make it 1 → n)
#   11. every row of the n side finds its key on the 1 side (WARN: a row without it goes blank in Power BI)
#   12. table sff_relaciones_modelo written and read back

STEP_8_4_LABEL = "8.4"
STEP_8_4_NAME = "ADAPTATION TO POWER BI"
STEP_8_4_PURPOSE = ("build the auxiliary tables of the Power BI report (names, orders and blocks of the codes of the core), "
                "so that Power BI only relates and sums: no logic inside the report")
STEP_8_4_ACTIONS = ["the table of pipeline sources: code, label, block, order",
                "the price increase monitor: the uplift of the closed months and its steps",
                "the parameters of this run: every field of the Config, its value and the steps that read it",
                "check the tables against the core (checks 1-4)",
                "write them (checks 5-8)",
                "the relations of the Power BI model, checked on the tables this run wrote (checks 9-12)",
                "count the checks; stop if any failed",
                "show the tables"]
STEP_8_4_OUTPUT = ("tables sff_forecast_pipeline_source (one row per forecast_pipeline_source), sff_price_increase_monitor "
               "(price group × closed pipeline month), sff_price_increase_calendar (one row per increase), "
               "sff_parametros (one row per field of the Config) and sff_relaciones_modelo (one row per relation)")

PIPELINE_SOURCE_TABLE = "forecast_pipeline_source"
PRICE_MONITOR_TABLE = "price_increase_monitor"
PARAMETERS_TABLE = "parametros"
PRICE_CALENDAR_TABLE = "price_increase_calendar"

# the detector of price increases, over the uplift of the isolated renewals (no retention event):
                               # (one noisy month is not an increase; the price of it: a step is confirmed 2 months late)

# the blocks of the report: where each source sits when the pipeline of a year is broken down
BLOCK_EXTRACT_PIPELINE = "Extract pipeline"        # due dates already in the extract
BLOCK_EXTENDED_HORIZON = "Extended horizon"        # pipeline built by the framework (simulation window, +12 months)
BLOCK_TIME_SERIES = "Time series"                  # the retail to subscription universe
BLOCK_CONTROL = "Control"                          # rows with no money: they keep the sums equal to the reporting

# one row per source: (code in the core, business label, block, order in the report)
PIPELINE_SOURCES = [
    (TOTAL_ORIGIN_RENEWED,   "Pipeline already due (actual renewal)",    BLOCK_EXTRACT_PIPELINE, 1),
    (TOTAL_ORIGIN_EXPECTED,  "Known pipeline still to fall due",         BLOCK_EXTRACT_PIPELINE, 2),
    (TOTAL_ORIGIN_PROJECTED, "Re-renewal of 1-year licences",            BLOCK_EXTENDED_HORIZON, 3),
    (TOTAL_ORIGIN_SIMULATED, "Simulated acquisition",                    BLOCK_EXTENDED_HORIZON, 4),
    (TS_REAL,                "Retail to subscription (actual)",          BLOCK_TIME_SERIES,      5),
    (TS_PROJECTED,           "Retail to subscription (rest of the year)", BLOCK_TIME_SERIES,     6),
    (TS_SIMULATED,           "Retail to subscription (next year)",       BLOCK_TIME_SERIES,      7),
    (TS_REENTRY,             "Retail to subscription (re-entry)",        BLOCK_TIME_SERIES,      8),
    (ROW_FROM_GAP,           "Months with nothing due",                  BLOCK_CONTROL,          9),
    (TS_WITHOUT_RESULT,      "Time series month not closed yet",         BLOCK_CONTROL,          10),
]


def build_power_bi_tables(core_table: pd.DataFrame, configuration: Config) -> dict:
    """The auxiliary tables of the Power BI report.

    INPUT:   the core (step 8.3), with its column forecast_pipeline_source.
    OUTPUT:  {"pipeline_source": the table of pipeline sources}.
    RULES:   every code of the core has its row; the table lists every source the framework can produce,
             also those absent in this run (a run without acquisition has no simulated acquisition, and the
             report keeps the same rows).
    EDGE CASES: a code in the core that the table does not know stops the step (check 1): a new origin
             was added to the core and this table was not updated.
    """
    configuration.log_step_start(STEP_8_4_LABEL, STEP_8_4_NAME, STEP_8_4_PURPOSE, STEP_8_4_ACTIONS, STEP_8_4_OUTPUT)
    check_log = []

    # [1] the table of pipeline sources
    pipeline_source_table = build_pipeline_source_table()
    configuration.log_action(STEP_8_4_LABEL, 1, f"{len(pipeline_source_table)} pipeline sources in "
                                            f"{pipeline_source_table['source_block'].nunique()} blocks")

    # [2] the price increase monitor, per price group, and the calendar of the increases it detects
    price_monitor = price_increase_monitor(core_table, configuration)
    calendar = increase_calendar(price_monitor, configuration)
    groups = price_monitor["price_group"].nunique()
    configuration.log_action(STEP_8_4_LABEL, 2, f"{groups:,} price groups ({', '.join(configuration.price_group_dims) or 'the portfolio'}) "
                                            f"× {price_monitor['period'].nunique()} closed months · detector on "
                                            f"{price_monitor.attrs['detector_series']} · {len(calendar)} increases detected in "
                                            f"{calendar['price_group'].nunique() if len(calendar) else 0} groups")

    # [3] the parameters of this run
    parameters = parameter_table(configuration)
    changed = parameters[parameters["cambiado"] == 1]
    configuration.log_action(STEP_8_4_LABEL, 3, f"{len(parameters)} parameters in the Config · {len(changed)} with a value of "
                                            f"this run (not its default) · "
                                            f"{int((parameters['pasos'] == '').sum())} read by no step")

    # [4] the checks against the core
    configuration.log_action(STEP_8_4_LABEL, 4, "checking the tables against the core")
    sources_in_core = set(core_table[FINAL_ORIGIN_COLUMN].dropna().unique())
    sources_in_table = set(pipeline_source_table[FINAL_ORIGIN_COLUMN])
    sources_without_row = sorted(sources_in_core - sources_in_table)
    configuration.log_check(STEP_8_4_LABEL, check_log,
                            "every forecast_pipeline_source of the core has its row",
                            not sources_without_row,
                            failure_detail=f"sources of the core without a row: {sources_without_row}",
                            context=f"{len(sources_in_core)} sources in the core, "
                                    f"{len(sources_in_table - sources_in_core)} with no rows in this run")

    repeated_sources = pipeline_source_table[pipeline_source_table[FINAL_ORIGIN_COLUMN].duplicated()][FINAL_ORIGIN_COLUMN].tolist()
    configuration.log_check(STEP_8_4_LABEL, check_log,
                            "one row per source (the relation in Power BI is 1 → n)",
                            not repeated_sources,
                            failure_detail=f"repeated sources: {repeated_sources}")

    with_renewals = price_monitor[price_monitor["renewed_units"] > 0]
    configuration.log_check(STEP_8_4_LABEL, check_log,
                            "every closed pipeline month with renewals has its uplift in the monitor",
                            with_renewals["uplift"].notna().all(),
                            failure_detail=f"{int(with_renewals['uplift'].isna().sum())} months with renewals and no uplift",
                            context=f"{len(price_monitor):,} months")
    flags_without_reference = price_monitor[(price_monitor["price_increase_flag"] == 1)
                                            & price_monitor["uplift_previous_12"].isna()]
    configuration.log_check(STEP_8_4_LABEL, check_log,
                            f"an increase is only flagged against a reference of ≥ {configuration.price_min_reference_months} previous months",
                            flags_without_reference.empty,
                            failure_detail=f"{len(flags_without_reference)} months flagged without a reference")

    # [5] the writes
    configuration.log_action(STEP_8_4_LABEL, 5, "writing the tables")
    configuration.write_table(STEP_8_4_LABEL, check_log, pipeline_source_table, PIPELINE_SOURCE_TABLE)
    configuration.write_table(STEP_8_4_LABEL, check_log, price_monitor.drop(columns=["renewed_units"]), PRICE_MONITOR_TABLE)
    configuration.write_table(STEP_8_4_LABEL, check_log, calendar, PRICE_CALENDAR_TABLE)
    configuration.write_table(STEP_8_4_LABEL, check_log, parameters, PARAMETERS_TABLE)

    # [6] the relations of the model of Power BI, on the tables this run wrote
    relations = relations_of_the_model(configuration)
    configuration.log_action(STEP_8_4_LABEL, 6, f"{len(relations)} relations of the Power BI model, checked on the data")
    check_the_relations(relations, configuration, check_log)
    configuration.write_table(STEP_8_4_LABEL, check_log, relations, TABLE_MODEL_RELATIONS)

    # [7] the count
    configuration.log_action(STEP_8_4_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_8_4_LABEL, STEP_8_4_NAME, check_log)

    # [8] the tables on screen
    configuration.show_table(relations)
    configuration.log_action(STEP_8_4_LABEL, 8, "the pipeline sources, as Power BI will show them, the calendar of the price "
                                            "increases, and the parameters this run changed (all of them in sff_parametros):")
    configuration.show_table(pipeline_source_table)
    configuration.logger.doc(f"[{STEP_8_4_LABEL}] ══════════ CALENDARIO DE SUBIDAS DETECTADAS ══════════")
    configuration.logger.doc(f"[{STEP_8_4_LABEL}] por grupo de precio ({', '.join(configuration.price_group_dims) or 'cartera'}): un "
                             f"escalón ≥ {configuration.price_step_threshold:.0%} frente a sus 12 meses previos, mantenido "
                             f"{configuration.price_persistence_months} meses, en meses con ≥ "
                             f"{configuration.price_group_min_renewed_units:,.0f} renovaciones; las 30 de más valor:")
    if len(calendar):
        shown_calendar = calendar.head(30).copy()
        for column_name in ["escalon", "uplift_antes", "uplift_ciclo"]:
            shown_calendar[column_name] = shown_calendar[column_name].round(3)
        shown_calendar["usd_renovado_ciclo"] = shown_calendar["usd_renovado_ciclo"].round(0)
        configuration.show_table(shown_calendar.drop(columns=["price_group"]))
    else:
        configuration.logger.doc(f"[{STEP_8_4_LABEL}] no increase detected in any group")
    configuration.logger.doc(f"[{STEP_8_4_LABEL}] ══════════ FIN DEL CALENDARIO DE SUBIDAS ══════════")
    changed = parameters[parameters["cambiado"] == 1]
    configuration.logger.doc(f"[{STEP_8_4_LABEL}] {len(changed)} of {len(parameters)} parameters with a value of this run "
                             f"(the rest keep their default; every one, with the steps that read it, in "
                             f"{configuration.table_prefix}parametros):")
    configuration.show_table(changed[["parametro", "valor", "por_defecto", "pasos"]])

    return {"pipeline_source": pipeline_source_table, "price_monitor": price_monitor.drop(columns=["renewed_units"]),
            "price_calendar": calendar, "parameters": parameters, "relations": relations}


# THE MODEL OF POWER BI: every relation, 1 side → n side. The dimension of the forecast series filters the core and
# the satellites of a series; a composition and a credibility reference filter theirs; the uplift cell, the core.
# (1 table, its key, n table, its column, what the relation is for)
MODEL_RELATIONS = [
    (TABLE_FORECAST_SERIES, "s1_4_fs_id", TABLE_CORE, "s1_4_fs_id", "the forecast series filters its rows"),
    (PIPELINE_SOURCE_TABLE, FINAL_ORIGIN_COLUMN, TABLE_CORE, FINAL_ORIGIN_COLUMN, "the business label of every origin"),
    (TABLE_UPLIFT_CELLS, UPLIFT_CELL_ID_COLUMN, TABLE_CORE, "s1_4_uplift_cell_id", "the uplift of the cell of every row"),
    (TABLE_FORECAST_SERIES, "s1_4_fs_id", TABLE_SERIES_EXAM, SERIES_ID_COLUMN, "the exam of the series, summed"),
    (TABLE_FORECAST_SERIES, "s1_4_fs_id", TABLE_SERIES_EXAM_DETAIL, SERIES_ID_COLUMN, "every prediction of its exam"),
    (TABLE_FORECAST_SERIES, "s1_4_fs_id", TABLE_LADDER_STEPS, SERIES_ID_COLUMN, "its group in every pass of the ladder"),
    (TABLE_FORECAST_SERIES, "s1_4_fs_id", TABLE_LADDER_MERGES, SERIES_ID_COLUMN, "every merge of the series"),
    (TABLE_FORECAST_SERIES, "s1_4_fs_id", TABLE_SERIES_DYNAMICS, SERIES_ID_COLUMN, "the dynamics of the series"),
    (TABLE_COMPOSITION, COMPOSITION_ID_COLUMN, TABLE_FORECAST_SERIES, "s2_2_stage3_id", "the composition of every series"),
    (TABLE_COMPOSITION, COMPOSITION_ID_COLUMN, TABLE_COMPOSITION_MEMBERS, COMPOSITION_ID_COLUMN,
     "the series in the rate of the composition (uses / lends)"),
    (TABLE_COMPOSITION, COMPOSITION_ID_COLUMN, TABLE_POOL_SERIES, COMPOSITION_ID_COLUMN, "the monthly series of the composition"),
    (TABLE_COMPOSITION, COMPOSITION_ID_COLUMN, TABLE_COMPOSITION_TECHNIQUES, COMPOSITION_ID_COLUMN,
     "every technique of its backtest"),
    (TABLE_COMPOSITION, COMPOSITION_ID_COLUMN, TABLE_COMPOSITION_FORECAST_ALL, COMPOSITION_ID_COLUMN,
     "the future rate of every technique"),
    (TABLE_CREDIBILITY, "credibility_ref_id", TABLE_FORECAST_SERIES, "s2_3_credibility_ref_id",
     "the credibility reference of every series"),
    (TABLE_CREDIBILITY, "credibility_ref_id", TABLE_CREDIBILITY_MEMBERS, "credibility_ref_id",
     "the series in the rate of the reference"),
]


def relations_of_the_model(configuration: Config) -> pd.DataFrame:
    """Every relation of MODEL_RELATIONS with what the tables this run wrote say of it: the rows of each side, whether
    the key is unique on the 1 side, and the rows of the n side whose value finds no key (null values are not
    counted: Power BI leaves them out of the relation)."""
    written = configuration.written_tables
    rows = []
    for one_table, one_key, many_table, many_column, purpose in MODEL_RELATIONS:
        one, many = written.get(one_table), written.get(many_table)
        row = {"tabla_1": configuration.table_prefix + one_table, "clave_1": one_key,
               "tabla_n": configuration.table_prefix + many_table, "columna_n": many_column, "para_que": purpose,
               "existe": int(one is not None and many is not None and one_key in one.columns
                             and many_column in many.columns)}
        if row["existe"]:
            keys = one[one_key].dropna().astype(str)
            values = many[many_column].dropna().astype(str)
            row.update(filas_1=len(one), filas_n=len(many), clave_unica=int(not keys.duplicated().any()),
                       filas_n_sin_clave=int((~values.isin(set(keys))).sum()))
        else:
            row.update(filas_1=0, filas_n=0, clave_unica=0, filas_n_sin_clave=0)
        rows.append(row)
    return pd.DataFrame(rows)


def check_the_relations(relations: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 9 to 11: the columns exist, the key of the 1 side is unique, the n side finds its keys."""
    describe = lambda frame: "; ".join(f"{row['tabla_1']}[{row['clave_1']}] → {row['tabla_n']}[{row['columna_n']}]"
                                       for _, row in frame.iterrows())
    missing = relations[relations["existe"] == 0]
    configuration.log_check(STEP_8_4_LABEL, check_log, "every relation of the model joins columns of tables this run wrote",
                            missing.empty, blocking=False, failure_detail=f"{len(missing)} relations without their tables "
                                                                           f"or columns: {describe(missing)}",
                            context=f"{len(relations)} relations")
    present = relations[relations["existe"] == 1]
    repeated = present[present["clave_unica"] == 0]
    configuration.log_check(STEP_8_4_LABEL, check_log, "the key of the 1 side of every relation is unique (1 → n)",
                            repeated.empty, failure_detail=f"repeated keys: {describe(repeated)}")
    orphans = present[present["filas_n_sin_clave"] > 0]
    configuration.log_check(STEP_8_4_LABEL, check_log,
                            "every row of the n side finds its key on the 1 side (it would go blank in Power BI)",
                            orphans.empty, blocking=False,
                            failure_detail=" · ".join(f"{row['tabla_n']}[{row['columna_n']}]: {row['filas_n_sin_clave']:,} rows"
                                                      for _, row in orphans.iterrows())
                            + " (expected: a series that does not climb the ladder has no composition; an extended row "
                              "with a cell never seen takes the uplift of its mandatory cell, s3_1_origen_uplift)")



def price_increase_monitor(core_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per price group and closed pipeline month of the core: the uplifts and the detection of the
    price increases (price_increases.price_monitor, the same that steps 3.1 and 3.3 read from the fine table)."""
    closed = core_table[(core_table["forecast_status"] == "actual") & (core_table["forecast_universe"] == "pipeline")
                        & (core_table["forecast_to_renew_units"] > 0)]
    return price_monitor(closed, configuration)


def build_pipeline_source_table() -> pd.DataFrame:
    """One row per pipeline source: code, business label, block and order."""
    pipeline_source_table = pd.DataFrame(PIPELINE_SOURCES,
                                         columns=[FINAL_ORIGIN_COLUMN, "source_label", "source_block", "source_order"])
    return pipeline_source_table


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 8.5 · VALIDATION  (until 25-oct-2026: step_26_validation.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The final validation: does everything agree, from the raw to the forecast?
#
# Each step checked its own work. This step checks the chain ACROSS steps, once everything
# has run, and leaves the result in one table:
#   · the money: Σ USD due of the extract = Σ of the fine table = Σ of the forecast units;
#     the future USD due of the extract = Σ USD due of the forecast
#   · every series with money to predict has a rate and a risk level
#   · every future row takes its rate from its series' estimation id when it has one
#   · the forecast is coherent with the past: the expected rate of the future (USD) is within
#     ±15 pp of the last closed year's (warning: a larger jump must be explained)
#   · the precision measured in the exam of the portfolio (step 7.1, series by series): the
#     error of the total renewals is within ±10 % in every exam month (warning: the forecast is
#     less reliable than promised)
#
# Actions (logged as they are done):
#   1. the money along the chain
#   2. the coverage of the series and the forecast rows
#   3. the coherence of the forecast with the past and with the exam
#   4. write the validation                                            check 7
#   5. count the checks; stop if any failed
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. Σ USD due: extract = fine table = forecast units
#    2. the future USD due of the extract = Σ USD due of the forecast
#    3. every series with money to predict has a rate and a risk level
#    4. every future row of a series with an estimation id (with exam_min_months_to_predict closed months) takes its
#       rate from the pool
#    5. the expected rate of the future is within ±15 pp of the last closed year's   (warning only)
#    6. the error of the total renewals in the exam is within ±10 %                   (warning only)
#    7. table sff_validacion written and read back
#
# Output: the validation (one row per check) · table sff_validacion.

STEP_8_5_LABEL = "8.5"
STEP_8_5_NAME = "VALIDATION"
STEP_8_5_PURPOSE = ("check, once everything has run, that the chain agrees from the raw to the forecast: the money is "
                "conserved, every series and future row is covered, and the forecast is coherent with the past and "
                "with the precision measured in the exam")
STEP_8_5_ACTIONS = ["the money along the chain",
                "the coverage of the series and the forecast rows",
                "the coherence of the forecast with the past and with the exam",
                "write the validation (check 7)",
                "count the checks; stop if any failed"]
STEP_8_5_OUTPUT = "one row per check · table sff_validacion"



def validate_chain(extract_summary: dict, results: dict, configuration: Config) -> pd.DataFrame:
    """The checks across steps; the table of the validation."""
    configuration.log_step_start(STEP_8_5_LABEL, STEP_8_5_NAME, STEP_8_5_PURPOSE, STEP_8_5_ACTIONS, STEP_8_5_OUTPUT)
    check_log = []
    fine, units, forecast = results["fine_table"], results["forecast_units"], results["forecast"]["forecast"]
    due = configuration.pipeline_usd_col

    # [1] the money
    configuration.log_action(STEP_8_5_LABEL, 1, "the money along the chain")
    time_series_due = results["time_series_rows"][due].fillna(0).sum() if results.get("time_series_rows") is not None else 0.0
    wiped = fine[S0_PIPELINE_USD_COLUMN].sum() - fine[due].sum()               # step 1.3: the pipeline not known yet
    totals = {"extracto": extract_summary["usd_vence"] - time_series_due - wiped, "tabla_fina": fine[due].sum(), "forecast_units": units[due].sum()}
    configuration.log_check(STEP_8_5_LABEL, check_log, "Σ USD due: extract (without time_series and the pipeline not known yet) = fine table = forecast units",
                            max(totals.values()) - min(totals.values()) <= configuration.money_tolerance,
                            failure_detail=f"totals differ: {totals}", context=f"${totals['extracto']:,.0f}")
    future_due = fine.loc[fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION, due].sum()
    configuration.log_check(STEP_8_5_LABEL, check_log, "the future USD due of the extract = Σ USD due of the forecast's extract rows",
                            abs(future_due - forecast.loc[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, due].sum()) <= configuration.money_tolerance,
                            failure_detail=f"${future_due:,.0f} vs ${forecast.loc[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL, due].sum():,.0f}",
                            context=f"${future_due:,.0f} (the extended horizon adds its own pipeline)")

    # [2] coverage
    configuration.log_action(STEP_8_5_LABEL, 2, "the coverage of the series and the forecast rows")
    estimate = results["series_estimate"]
    with_money = estimate[estimate["usd_por_predecir"] > 0]
    uncovered = with_money[with_money["nivel_riesgo"].isna()]
    forecast_series = set(forecast[SERIES_ID_COLUMN])
    configuration.log_check(STEP_8_5_LABEL, check_log, "every series with money to predict has a rate and a risk level",
                            uncovered.empty and set(with_money[SERIES_ID_COLUMN]) <= forecast_series,
                            failure_detail=f"{len(uncovered)} series without level", context=f"{len(with_money):,} series")
    pooled_ids = set(results["pool_reference"][COMPOSITION_ID_COLUMN])
    # a composition with fewer closed months than exam_min_months_to_predict does not predict, in the exam and in the
    # forecast (step 7.2): its rows take their cell
    rate_box = results.get("rate_box")
    if rate_box is not None:
        pooled_ids = {composition_id for composition_id in pooled_ids
                      if len(rate_box.composition_index.get(composition_id, {}).get("month", []))
                      >= configuration.exam_min_months_to_predict}
    should_pool = forecast[COMPOSITION_ID_COLUMN].isin(pooled_ids)
    configuration.log_check(STEP_8_5_LABEL, check_log, "every future row of a series with an estimation id with enough closed "
                            "months takes its rate from the pool",
                            bool((forecast.loc[should_pool, "origen_tasa"] == RATE_FROM_POOL).all()),
                            failure_detail="rows with a pool that took another rate",
                            context=f"{int(should_pool.sum()):,} of {len(forecast):,} rows from a pool")

    # [3] coherence
    configuration.log_action(STEP_8_5_LABEL, 3, "the coherence of the forecast with the past and with the exam")
    closed = fine[fine[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    last_year = closed[configuration.period_col].max().year
    last_year_rows = closed[closed[configuration.period_col].map(lambda month: month.year) == last_year]
    past_rate = last_year_rows[configuration.renewed_usd_col].sum() / last_year_rows[due].sum()
    extract_rows = forecast[forecast[PIPELINE_ORIGIN_COLUMN] == PIPELINE_REAL]
    future_rate = extract_rows["esperado_usd"].sum() / extract_rows[due].sum()
    configuration.log_check(STEP_8_5_LABEL, check_log, f"the expected rate of the future is within ±{configuration.max_rate_jump_pp:.0f} pp of {last_year}'s",
                            abs(future_rate - past_rate) * 100 <= configuration.max_rate_jump_pp,
                            failure_detail=f"future {future_rate:.1%} vs {last_year} {past_rate:.1%}: explain the jump",
                            context=f"future {future_rate:.1%} (USD, uplift included) vs {last_year} {past_rate:.1%}", blocking=False)
    exam = results.get("portfolio_exam")
    exam_error = (exam["framework_error_total"].abs().max() if exam is not None
                  else results["backtest"]["exam_total"]["elegida_error_pct"].abs().max())
    configuration.log_check(STEP_8_5_LABEL, check_log, f"the error of the total renewals in the exam is within ±{configuration.max_exam_total_error:.0%}",
                            exam_error <= configuration.max_exam_total_error,
                            failure_detail=f"the worst month misses by {exam_error:.1%}", context=f"worst month {exam_error:.1%}",
                            blocking=False)

    # [4] the table
    configuration.log_action(STEP_8_5_LABEL, 4, "writing the validation")
    validation = pd.DataFrame(check_log, columns=["estado", "comprobacion", "detalle"])
    configuration.write_table(STEP_8_5_LABEL, check_log, validation, TABLE_VALIDATION)

    # [5] the count of the checks
    configuration.log_action(STEP_8_5_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_8_5_LABEL, STEP_8_5_NAME, check_log)
    return validation


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 8.6 · REPORT  (until 25-oct-2026: step_27_report.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The report: what the data are, which hypotheses of the framework hold on them, and how well it predicts. One
# chapter per block, in the order of the diagram, after the verdict:
#   0. The verdict: every hypothesis of every block, its number and whether it holds (verdicts.py); what the official
#      band is made of.
#   1. B1 · The data: the extract, what was corrected, and the profile of the pipeline (volume, concentration,
#      support, history, every dimension).
#   2. B2 · The rate: the binomial support before and after the ladder, the dynamics, the backtest and the exam.
#   3. B3 · The price: the uplift, the contract rule, the paths and the ways in the exam, the error of the price.
#   4. B4 · The customer states: the exam of the curve, the honest exam, the maturation by year.
#   5. B5 · The future pipeline: the simulated acquisitions against the year before, and their exam.
#   6. B6 · The time series universe: the exam of every level rule, the universe by year and origin.
#   7. B7 · The forecast: by month and by year, the official band and what it is made of, the total by origin.
#   8. B8 · Control: the rows added and wiped, the validation of the chain, the relations of the Power BI model, and
#      what to validate with real data.
#
# Actions (logged as they are done):
#   1. the verdict of block 8 (the last one), the chapters, the summary for the business and the charts
#   2. check the report                                                 check 1
#   3. write the report file
#   4. write the verdicts of every block and the profile of the pipeline  checks 2-3
#   5. count the checks; stop if any failed
#   6. show the headline numbers, as a table, and the summary for the business (resumen_negocio.py)
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the report has its nine chapters
#    2. sff_veredicto_bloques written and read back
#    3. sff_perfil_datos written and read back
#
# Output: the text of the report · file informe_sff.md · tables sff_veredicto_bloques, sff_perfil_datos.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_8_6_LABEL = "8.6"
STEP_8_6_NAME = "REPORT"
STEP_8_6_PURPOSE = ("tell, with the numbers of this run, what the data are, which hypotheses of the framework hold on "
                    "them (the verdict of every block) and how well it predicts, one chapter per block; and leave the "
                    "verdicts and the profile of the pipeline as tables")
STEP_8_6_ACTIONS = ["the verdict of block 8, the nine chapters of the report, the summary for the business and the charts",
                    "check the report (check 1)",
                    "write the report file",
                    "write the verdicts of every block and the profile of the pipeline (checks 2-3)",
                    "the year by region, read from sff_nucleo with its SQL: renewed so far and expected to December "
                    "(check 4)",
                    "count the checks; stop if any failed",
                    "show the headline numbers, as a table, and the summary for the business",
                    "show the final table: the year by region"]
STEP_8_6_OUTPUT = ("the text of the report · file informe_sff.md · tables sff_veredicto_bloques (block × question) and "
                   "sff_perfil_datos (the profile of the pipeline)")

# ─── named constants ─────────────────────────────────────────────────────────────
CHAPTER_COUNT = 9


def markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    """A pandas table as a markdown table (no extra library)."""
    if frame is None or frame.empty:
        return "_(no rows)_\n"
    def cell(value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:,.{decimals}f}"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)
    percent_columns = {column for column in frame.columns if str(column).startswith("pct") or str(column).endswith("_pct")
                       or str(column) == "dentro_banda"}
    frame = frame.copy()
    for column in percent_columns:
        frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    separator = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator] + rows) + "\n"


def build_report(extract_summary: dict, results: dict, configuration: Config) -> str:
    """The markdown report, one chapter per block after the verdict; checked and written, with the tables of the
    verdicts and of the profile of the pipeline. Returns its text."""
    configuration.log_step_start(STEP_8_6_LABEL, STEP_8_6_NAME, STEP_8_6_PURPOSE, STEP_8_6_ACTIONS, STEP_8_6_OUTPUT)
    check_log = []

    # [1] the verdict of block 8 (block 1 to 7 were recorded after each one ran), the chapters, the summary, the charts
    record_the_verdict("B8", results, configuration)
    headline, chapters = [], []
    chapters.append(chapter_verdict(results, configuration))
    chapters.append(chapter_data(extract_summary, results, configuration))
    chapters.append(chapter_rate(results, configuration, headline))
    chapters.append(chapter_price(results, configuration))
    chapters.append(chapter_states(results, configuration))
    chapters.append(chapter_future_pipeline(results, configuration))
    chapters.append(chapter_time_series(results, configuration))
    chapters.append(chapter_forecast(results, configuration, headline))
    chapters.append(chapter_control(results, configuration))
    summary_lines = business_summary(results, configuration)
    charts = draw_charts(results, configuration)
    time_series = results.get("time_series")
    if time_series is not None and len(time_series):        # drawn by steps 1.2 and 6.1, with this run's universe
        for name in (HISTORY_CHART, PREDICTION_CHART):
            path = os.path.join(configuration.output_folder or ".", CHART_FOLDER, f"{name}.png")
            if os.path.exists(path):
                charts[name] = path
    configuration.log_action(STEP_8_6_LABEL, 1, f"{len(charts)} charts drawn in {os.path.join(configuration.output_folder or '.', CHART_FOLDER)}: "
                                            + ", ".join(charts))
    report_text = report_cover(extract_summary, results, configuration, headline, summary_lines, charts) + "\n".join(chapters)
    configuration.log_action(STEP_8_6_LABEL, 1, f"{len(chapters)} chapters written ({len(report_text):,} characters)")

    # [2] the check
    configuration.log_action(STEP_8_6_LABEL, 2, "checking the report")
    configuration.log_check(STEP_8_6_LABEL, check_log, f"the report has its {CHAPTER_COUNT} chapters",
                            report_text.count("\n## ") == CHAPTER_COUNT,
                            failure_detail=f"{report_text.count(chr(10) + '## ')} chapters found")

    # [3] the file
    configuration.log_action(STEP_8_6_LABEL, 3, "writing the report file")
    os.makedirs(configuration.output_folder, exist_ok=True)
    report_path = os.path.join(configuration.output_folder, REPORT_FILE_NAME)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
    configuration.logger.doc(f"[{STEP_8_6_LABEL}] report written: {report_path}")

    # [4] the verdicts and the profile, as tables (Power BI reads them next to the rest)
    configuration.log_action(STEP_8_6_LABEL, 4, "writing the verdicts of every block and the profile of the pipeline")
    configuration.write_table(STEP_8_6_LABEL, check_log, verdict_table(configuration), TABLE_BLOCK_VERDICTS)
    if configuration.data_profile is not None:
        configuration.write_table(STEP_8_6_LABEL, check_log, configuration.data_profile, TABLE_DATA_PROFILE)
    else:
        configuration.log_not_evaluated(STEP_8_6_LABEL, check_log, f"table {configuration.table_prefix}{TABLE_DATA_PROFILE} "
                                        "written and read back", "block 1 left no profile")

    # [5] the year by region, from the core as SQL reads it (shown at the end, action 8)
    configuration.log_action(STEP_8_6_LABEL, 5, "the year by region, read from the core with its SQL")
    year_by_region = read_the_year_by_region(results, configuration, check_log)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_8_6_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_8_6_LABEL, STEP_8_6_NAME, check_log)

    # [7] the headline numbers, and the summary for the business
    configuration.log_action(STEP_8_6_LABEL, 7, "the headline numbers of this run:")
    configuration.show_table(pd.DataFrame(headline, columns=["indicador", "valor"]))
    show_business_summary(results, configuration, STEP_8_6_LABEL)

    # [8] the final table
    show_the_year_by_region(year_by_region, configuration)
    return report_text


# ─── the final table: the year by region (actions 5 and 8) ───────────────────────
def read_the_year_by_region(results: dict, configuration: Config, check_log: list):
    """The USD renewed in the current year by region (report_queries.year_by_region_query), read from sff_nucleo
    as written, so the table on screen is exactly what the SQL gives in SQL Server or Power BI. Check 4: its TOTAL
    is the TOTAL of the year in sff_forecast_total (step 7.3). None without a SQL engine (the tables are CSV)."""
    if configuration.sql_engine is None:
        configuration.log_not_evaluated(STEP_8_6_LABEL, check_log, "the year by region adds up to the total of the year",
                                        "no SQL engine: the core is a CSV, the query cannot run")
        return None
    sql = year_by_region_query(configuration)
    year_by_region = pd.read_sql(sql, configuration.sql_engine)
    year = configuration.calendar_boundaries()["current"].year
    total_row = year_by_region[year_by_region["orden_total"] == 1]
    query_total = float(total_row["usd_total_ano"].iloc[0]) if len(total_row) else 0.0
    forecast_total = results.get("forecast_total")
    year_total = float(forecast_total.loc[(forecast_total["ano"] == year) & (forecast_total["origen"] == TOTAL_ORIGIN_TOTAL),
                                          "usd_renovado"].sum()) if forecast_total is not None else np.nan
    configuration.log_check(STEP_8_6_LABEL, check_log,
                            f"the year by region adds up to the total of {year} in sff_forecast_total (step 7.3)",
                            np.isfinite(year_total) and abs(query_total - year_total) <= max(configuration.money_tolerance,
                                                                               configuration.money_relative_tolerance * abs(year_total)),
                            failure_detail=f"${query_total:,.0f} by region vs ${year_total:,.0f} in sff_forecast_total",
                            context=f"${query_total:,.0f} renewed in {year}")
    year_by_region.attrs["sql"] = sql
    return year_by_region


def show_the_year_by_region(year_by_region, configuration: Config) -> None:
    """Action 8: the final table on screen, the money in whole dollars, and the SQL that gives it."""
    if year_by_region is None:
        return
    year = configuration.calendar_boundaries()["current"].year
    last_closed = configuration.calendar_boundaries()["current"] - 1
    columns = year_by_region_columns(configuration)
    heading = (f"THE YEAR {year} BY {' AND '.join(columns)}: renewed USD, real up to {last_closed} and expected from "
               f"{last_closed + 1} to {year}-12 (each region, its TOTAL, and the TOTAL of the year)")
    configuration.log_action(STEP_8_6_LABEL, 8, heading)
    if not configuration.console_shows_the_documentation:          # a quiet console still gets the final table
        print(f"[{STEP_8_6_LABEL}] {heading}")
    shown = year_by_region.drop(columns=["orden_total", "orden_region"]).copy()
    for column_name in ["usd_real_hasta_hoy", "usd_esperado_hasta_fin_de_ano", "usd_total_ano"]:
        shown[column_name] = shown[column_name].map(lambda value: f"{value:,.0f}")
    configuration.show_table(shown, always=True)
    configuration.show_query(STEP_8_6_LABEL, f"the year {year} by region", year_by_region.attrs["sql"])


def as_section(chapter_text: str) -> str:
    """A chapter of the earlier report as a section of a block's chapter: its '## n · title' becomes '### title'."""
    return re.sub(r"^## (\d+ · )?", "### ", chapter_text, count=1)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHAPTERS
# ═══════════════════════════════════════════════════════════════════════════════════

def report_cover(extract_summary: dict, results: dict, configuration: Config, headline: list, summary_lines: list = (),
                 charts: dict = None) -> str:
    series = results["series"]
    headline.insert(0, ("filas del extracto", f"{extract_summary['filas']:,}"))
    headline.insert(1, ("forecast series", f"{len(series):,}"))
    headline.insert(2, ("USD por predecir", f"${series['usd_por_predecir'].sum():,.0f}"))
    lines = ["# SFF · Informe de la ejecución",
             f"Mes en curso: **{configuration.calendar_boundaries()['current']}** · {configuration.calendar_description()}",
             "", "### Resumen para negocio", ""] + [f"- {line.strip().lstrip('·').strip()}" for line in summary_lines] + [
             "", "### Cifras principales", "", markdown_table(pd.DataFrame(headline, columns=["indicador", "valor"])), ""]
    if charts:
        lines += ["### Gráficos", ""] + [f"![{name}]({CHART_FOLDER}/{os.path.basename(path)})" for name, path in charts.items()] + [""]
    return "\n".join(lines)


def chapter_raw(extract_summary: dict, results: dict, configuration: Config) -> str:
    fine = results["fine_table"]
    boundaries = configuration.calendar_boundaries()
    closed = fine[configuration.period_col] < boundaries["current"]
    null_read_as_zero = int((closed & fine[S0_RENEWED_UNITS_COLUMN].isna()).sum())
    projection = fine[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    early = fine[projection & fine[S0_RENEWED_UNITS_COLUMN].fillna(0).ne(0)]
    per_role = (fine.groupby(CALENDAR_ROLE_COLUMN).agg(meses=(configuration.period_col, "nunique"), filas=(configuration.period_col, "size"),
                                                       usd_vence=(configuration.pipeline_usd_col, "sum"))
                .reindex(ROLES_IN_ORDER).dropna(how="all").reset_index())
    checks = pd.DataFrame(configuration.check_history)
    lines = ["## 1 · El raw y cómo lo mejoramos", "",
             f"El extracto tiene **{extract_summary['filas']:,} filas × {extract_summary['columnas']} columnas**, de "
             f"{fine[configuration.period_col].min()} a {fine[configuration.period_col].max()}. "
             f"Cada columna tiene un rol declarado en la Config y el calendario se genera desde el mes en curso.", "",
             markdown_table(per_role, 0),
             "**Comprobaciones de cada paso** (un fallo detiene la ejecución; un aviso se registra y sigue):", "",
             markdown_table(checks[["paso", "nombre", "comprobaciones", "ok", "avisos", "fallos", "avisos_detalle"]], 0) if len(checks) else "",
             "**Lo que se corrigió o completó en el raw:**", "",
             f"- Renovaciones nulas en meses cerrados leídas como 0 (nadie renovó): **{null_read_as_zero:,} filas**.",
             f"- Resultados adelantados borrados desde el mes en curso (el futuro no ha empezado): **{len(early):,} filas**, "
             f"{early[S0_RENEWED_UNITS_COLUMN].sum():,.0f} unidades y ${early[S0_RENEWED_USD_COLUMN].sum():,.0f}. "
             f"El raw original queda en las columnas s0_.",
             ]
    wiped_pipeline = fine[S0_PIPELINE_USD_COLUMN] - fine[configuration.pipeline_usd_col]
    lines.append(f"- Pipeline de licencias de 1 año vendidas o renovadas desde el mes en curso (vence desde "
                 f"{boundaries['current'] + 12}): **aún no se conoce**, se borra y se proyecta: **{int((wiped_pipeline != 0).sum()):,} "
                 f"filas**, ${wiped_pipeline.sum():,.0f} (el raw la conserva en s0_vencen_*).")
    if configuration.discount_value_column:
        unknown = (fine[configuration.discount_bucket_column] == UNKNOWN_DISCOUNT_BUCKET).mean()
        lines.append(f"- Descuento: tramo derivado del descuento exacto con los cortes {configuration.discount_bucket_edges} %; "
                     f"**{unknown:.1%}** de las filas sin dato (tramo `{UNKNOWN_DISCOUNT_BUCKET}`).")
    lines.append(f"- Grano: {len(fine):,} filas = {len(fine):,} filas finas distintas (unidad + valores de revalorización + "
                 f"descuento exacto); el dinero se conserva al agregar a {len(results['forecast_units']):,} forecast units.")
    return "\n".join(lines) + "\n"


def chapter_new_rows(results: dict, configuration: Config) -> str:
    """Chapter 2: every row the framework adds or wipes, in one format (step 8.1), and the series with most gaps."""
    lines = ["## 2 · Las filas que añade (y borra) el framework", "",
             "El forecast no usa el extracto tal cual. Añade o borra filas de cinco tipos, cada uno por un motivo:", "",
             "- **hueco** (paso 1.7): una forecast unit sin nada que vencer, añadida DENTRO de la historia de una serie "
             "estimable para que su serie mensual no tenga agujeros (las técnicas leen meses consecutivos: una tendencia, una "
             "estación, una media móvil). Todas sus medidas son 0 y su tasa queda **nula, nunca 0 %**: añade un mes, no dinero.",
             "- **resultado_adelantado_borrado** (paso 1.3): una renovación ya registrada desde el mes en curso; el mes no "
             "ha terminado, y el forecast la predice.",
             "- **pipeline_parcial_borrada** (paso 1.3): la pipeline de las licencias de 1 año que vencen 12 meses después "
             "del mes en curso o más tarde. La generan las ventas y renovaciones desde el mes en curso, que solo han empezado "
             "(el mes va por la mitad y los siguientes no han empezado): está a medio crear. Si se dejara, el forecast de "
             "esos meses se calcularía sobre una pipeline a medias; el paso 7.2 la reconstruye entera.",
             "- **proyectada** y **simulada** (paso 7.2): las renovaciones esperadas de las licencias de 1 año que vencen en la "
             "ventana de simulación, y la captación simulada; vencen 12 meses después y sustituyen a la pipeline borrada.", ""]
    new_rows = results.get("new_rows")
    if new_rows is not None and len(new_rows):
        totals = totals_of_new_rows(new_rows)
        lines += ["**Por origen:**", "", markdown_table(totals, 0), ""]
        future = new_rows[new_rows["origen"] != "hueco"]
        if len(future):
            lines += ["**Mes a mes, lo borrado y lo creado** (las unidades y el dinero que vencen; `esperado_usd`: lo que el "
                      "forecast espera renovar de las filas creadas):", "", markdown_table(future, 0), ""]
    rate_summary = results["series_rate"]
    with_gaps = rate_summary[rate_summary["huecos"] > 0].sort_values("huecos", ascending=False)
    history_months = rate_summary["meses_historia"].sum()
    lines += [f"**Los huecos:** {int(with_gaps['huecos'].sum()):,} en {len(with_gaps):,} series, el "
              f"{with_gaps['huecos'].sum() / history_months if history_months else 0:.1%} de los meses de historia. "
              f"Las series con más huecos:", "",
              markdown_table(with_gaps.head(configuration.report_top_rows)[[SERIES_ID_COLUMN, "meses_historia", "huecos", "n_propio", "usd_por_predecir"]], 0)]
    return "\n".join(lines) + "\n"


def totals_of_new_rows(new_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per origin: months, rows, units, USD and USD expected (the same format as step 8.1)."""
    rows = []
    for origin, block in new_rows.groupby("origen", sort=False):
        rows.append({"origen": origin, "meses": f"{block['mes'].min()}..{block['mes'].max()}", "filas": int(block["filas"].sum()),
                     "unidades": float(block["unidades"].sum()), "usd": float(block["usd"].sum()),
                     "esperado_usd": float(block["esperado_usd"].sum()) if block["esperado_usd"].notna().any() else np.nan})
    return pd.DataFrame(rows)


def exam_error_against_noise(results: dict):
    """The exam error against the binomial noise, by size of the series (contracts due in the month) and for
    the total of the portfolio: what share of the error is noise no prediction can remove."""
    series_exam = results.get("series_exam")
    if not series_exam:
        return None, None
    detail = series_exam["detail"]
    detail = detail[detail["method"].isin(["raw", "framework"])].copy()
    detail["tamano"] = pd.cut(detail["due_units"], [0, 30, 271, np.inf], right=False,
                              labels=["< 30 al mes", "30-270 al mes", "≥ 271 al mes"])
    rows = []
    for (size, method), block in detail.groupby(["tamano", "method"], observed=True):
        rmse, noise = float(np.sqrt((block["err_pp"] ** 2).mean())), float(np.sqrt((block["noise_pp"] ** 2).mean()))
        rows.append({"tamano": str(size), "metodo": method, "predicciones": len(block), "error_pp": rmse, "ruido_pp": noise,
                     "error_vs_ruido": rmse / noise if noise > 0 else np.nan,
                     "parte_del_error_que_es_ruido": min(1.0, noise ** 2 / rmse ** 2) if rmse > 0 else np.nan})
    by_size = pd.DataFrame(rows)
    # the total of the portfolio, every exam month and horizon: its error and its noise, in pp of the rate
    total_rows = []
    for (month, horizon, method), block in detail.groupby(["period", "h", "method"]):
        due = block["due_units"].sum()
        total_rows.append({"metodo": method, "error_pp": 100 * (block["pred_units"].sum() - block["real_units"].sum()) / due,
                           "ruido_pp": 100 * np.sqrt((block["pred_rate"] * (1 - block["pred_rate"]) * block["due_units"]).sum()) / due})
    total = pd.DataFrame(total_rows).groupby("metodo").agg(error_pp=("error_pp", lambda values: float(np.sqrt((values ** 2).mean()))),
                                                           ruido_pp=("ruido_pp", lambda values: float(np.sqrt((values ** 2).mean()))))
    total["error_vs_ruido"] = total["error_pp"] / total["ruido_pp"]
    return by_size, total.reset_index()


def chapter_support(results: dict, configuration: Config, headline: list) -> str:
    card = results["series_estimate"]
    total_usd = card["usd_por_predecir"].sum()
    with_history = card["meses_historia"] > 0

    def bucket(n):
        if n <= 0:
            return "0 · sin historia propia"
        if n < configuration.support_floor:
            return f"1 · < {configuration.support_floor:.0f} (no fiable solo)"
        if n < configuration.own_rate_floor:
            return f"2 · {configuration.support_floor:.0f}-{configuration.own_rate_floor:.0f} (evidencia, sin precisión)"
        return f"3 · ≥ {configuration.own_rate_floor:.0f} (precisa sola: ±5 pp)"

    before = card.assign(soporte=card["n_propio"].map(bucket)).groupby("soporte").agg(
        series=(SERIES_ID_COLUMN, "size"), usd_por_predecir=("usd_por_predecir", "sum")).reset_index()
    before["pct_usd"] = before["usd_por_predecir"] / total_usd if total_usd else 0.0

    before_error = card["error_binomial_pp"].where(with_history)
    after_error = configuration.z * card["se_estimacion_pp"]
    within_before = card.loc[before_error <= configuration.promise_pp, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0
    within_after = card.loc[after_error <= configuration.promise_pp, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0
    headline.append(("USD con la tasa conocida a ±5 pp: antes → después de la escalera", f"{within_before:.0%} → {within_after:.0%}"))
    comparison = pd.DataFrame([
        {"medida": "error de la tasa (90 %), ponderado por USD", "antes_pp": np.average(before_error.fillna(50), weights=card["usd_por_predecir"] + 1e-9),
         "despues_pp": np.average(after_error.fillna(50), weights=card["usd_por_predecir"] + 1e-9)},
        {"medida": f"% del USD con la tasa conocida a ±{configuration.promise_pp:.0f} pp", "antes_pp": within_before * 100, "despues_pp": within_after * 100}])
    levels = results["money_by_level"]
    lines = ["## 3 · El soporte binomial, antes y después de la escalera", "",
             "La tasa de un mes es k renovaciones de n contratos: aunque nada cambie, oscila por azar "
             "(error binomial √(p(1−p)/n)). Con **30** contratos al mes una serie tiene evidencia para prestar; con **271** "
             "su tasa se conoce a ±5 pp y puede ir sola. **Antes**: cada serie con su propio soporte. **Después**: la "
             "escalera junta las series pasada a pasada (signo, extras y dimensiones mandatory en el orden de colapso) hasta "
             "que cada grupo llega a 30; el grupo presta su tasa a sus series y, por debajo de 271, la mezcla con la de una "
             "referencia más amplia por credibilidad (etapa 4). Ver `DOC_escalera.md` y `DOC_modelo_datos.md`.", "",
             "**Antes · el dinero por soporte propio (el dial):**", "", markdown_table(before),
             "**Antes y después · el error con el que se CONOCE la tasa de cada serie** (antes: su error binomial con su "
             "propio soporte; después: el error de la estimación de la escalera). La predicción de un mes concreto conserva "
             "además el ruido de su propio tamaño, que ninguna escalera elimina: está en el nivel de riesgo.", "",
             markdown_table(comparison, 1),
             "**Etapa a etapa · cómo mejora el soporte** (0 raw · 1 signo · 2 extras · 3 colapso = la composición con la que "
             "se predice; agrupando por el id de cada etapa, las unidades que vencen suman lo mismo):", "",
             markdown_table(results["ladder"]["stage_summary"], 2) if results.get("ladder") else "",
             "**Pasada a pasada · el detalle dentro de cada etapa** (cada pasada es un reparto: las unidades que vencen suman lo "
             "mismo en todas; los grupos son menos y más grandes; pct_usd_floor / pct_usd_own_rate: dinero por predecir en "
             "grupos que llegan a 30 / a 271):", "",
             markdown_table(results["ladder"]["summary"], 2) if results.get("ladder") else "",
             "**Después · el dinero por nivel de riesgo** (error_pp: error de predicción del mes siguiente, ponderado por dinero):", "",
             markdown_table(levels)]
    return "\n".join(lines) + "\n"


def chapter_dynamics(results: dict, configuration: Config) -> str:
    portfolio = results.get("portfolio_dynamics")
    profile = results.get("portfolio_profile")
    dynamics = results.get("pool_dynamics")
    if portfolio is None:
        return "## 4 · La dinámica de la tasa\n\n_(paso 2.5 no ejecutado)_\n"
    verdict = "**hay efecto mes** más allá del ruido" if portfolio["estacional"] else "**no hay efecto mes** más allá del ruido"
    lines = ["## 4 · La dinámica de la tasa: ¿hay algo más que ruido?", "",
             "φ compara lo que varía la tasa mes a mes con lo que variaría solo por muestreo: φ ≈ 1, nada que modelar "
             "(la media es la mejor técnica); φ > 1, algo la mueve (tendencia, estación, cambio de nivel o de mezcla). La "
             "estacionalidad se prueba sobre la tasa sin su tendencia (prueba F del mes del año, 5 %). Es descriptivo: no "
             "restringe ninguna técnica; el backtest decide.", "",
             f"**Cartera completa:** {verdict} (p = {portfolio['p_valor_mes']:.3f}, amplitud {portfolio['amplitud_pp']:.1f} pp, "
             f"consistencia entre mitades {portfolio['consistencia']:.2f}); tendencia {portfolio['tendencia_pp_ano']:+.1f} pp/año "
             f"(p = {portfolio['p_valor_tendencia']:.3f}); φ {portfolio['phi']:.1f}. Un φ de cartera alto con tendencia suele "
             f"ser cambio de mezcla, no comportamiento.", "",
             markdown_table(profile)]
    if dynamics is not None and len(dynamics):
        lines += ["**Los pools con soporte, uno a uno:**", "",
                  markdown_table(dynamics.sort_values("usd_por_predecir", ascending=False).head(configuration.report_top_rows)
                                 [[COMPOSITION_ID_COLUMN, "meses", "phi", "tendencia_pp_ano", "estacional", "amplitud_pp",
                                   "meses_alto", "meses_bajo", "usd_por_predecir"]])]
    return "\n".join(lines) + "\n"


def chapter_precision(results: dict, configuration: Config, headline: list) -> str:
    backtest = results.get("backtest")
    if backtest is None:
        return "## 5 · Qué tal se predice la tasa\n\n_(paso 2.6 no ejecutado)_\n"
    predictions, decision, exam_by_pool = backtest["predictions"], backtest["decision"], backtest["exam_by_pool"]
    boundaries = configuration.calendar_boundaries()
    reference = results["pool_reference"]
    judged_share = (reference.loc[reference["gate"] == GATE_LEVEL, "usd_por_predecir"].sum()
                    / max(reference["usd_por_predecir"].sum(), 1))

    exam_by_band = exam_by_pool.merge(reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
    band_rows = []
    for band_name, rows in exam_by_band.groupby("tramo_h"):
        weights = rows["usd_por_predecir"] + 1e-9
        band_rows.append({"tramo": band_name, "pools": len(rows),
                          "error_elegida_pp": np.average(rows["elegida_err_pp_medio"], weights=weights),
                          "error_por_defecto_pp": np.average(rows["por_defecto_err_pp_medio"], weights=weights),
                          "sesgo_elegida_pp": np.average(rows["elegida_sesgo_pp"], weights=weights),
                          "dentro_banda": rows["dentro_banda"].mean()})
    by_band = pd.DataFrame(band_rows)

    card = results["series_estimate"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN, "nivel_riesgo", "usd_por_predecir"]]
    short = exam_by_pool[exam_by_pool["tramo_h"] == list(configuration.horizon_bands)[0]]
    by_level = card.merge(short[[COMPOSITION_ID_COLUMN, "elegida_err_pp_medio", "por_defecto_err_pp_medio"]], on=COMPOSITION_ID_COLUMN, how="left")
    by_level = (by_level.dropna(subset=["elegida_err_pp_medio"]).groupby("nivel_riesgo")
                .apply(lambda rows: pd.Series({"series": int(len(rows)), "usd_por_predecir": rows["usd_por_predecir"].sum(),
                                               "error_elegida_pp": np.average(rows["elegida_err_pp_medio"], weights=rows["usd_por_predecir"] + 1e-9),
                                               "error_por_defecto_pp": np.average(rows["por_defecto_err_pp_medio"], weights=rows["usd_por_predecir"] + 1e-9)}),
                       include_groups=False).reset_index())
    if len(by_level):
        by_level["series"] = by_level["series"].astype(int)

    portfolio_summary = results.get("portfolio_exam_summary")
    if portfolio_summary is not None:
        for horizon, block in portfolio_summary.groupby("h"):
            framework = block[block["metodo"] == METHOD_FRAMEWORK]
            baseline = block[block["metodo"] == METHOD_BASELINE]
            if framework.empty or baseline.empty:
                continue
            framework, baseline = framework.iloc[0], baseline.iloc[0]
            alone = block[block["metodo"] == "raw"]
            headline.append((f"error del TOTAL en el examen, h = {horizon}: framework vs baseline vs serie sola",
                             f"{framework['error_total_medio']:.1%} vs {baseline['error_total_medio']:.1%}"
                             + (f" vs {alone['error_total_medio'].iloc[0]:.1%}" if len(alone) else "")
                             + (f" · {gain_text(framework['ganancia_usd_mes'])}, mejor en el "
                                f"{framework['meses_mejor_que_baseline']:.0%} de los meses" if pd.notna(framework.get("ganancia_usd_mes")) else "")))
    if len(by_band):
        first = by_band.iloc[0]
        headline.append((f"error medio de la tasa por pool en el examen ({first['tramo']})",
                         f"{first['error_elegida_pp']:.1f} pp (técnica por defecto {first['error_por_defecto_pp']:.1f} pp)"))
    choices = (decision.groupby(["tramo_h", "tecnica_origen"]).size().rename("ids").reset_index())
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
    ranking = (selection.assign(abs_norm=selection["err_norm"].abs()).groupby(["tramo_h", "tecnica"])["abs_norm"].mean()
               .rename("err_norm_medio").reset_index().sort_values(["tramo_h", "err_norm_medio"]))
    lines = ["## 5 · Qué tal se predice la tasa de renovación", "",
             "**Cómo se mide.** Cada pool con soporte se predice en meses que ya ocurrieron, sin mirar el futuro: para el mes T "
             "a horizonte h, cada técnica solo ve hasta T − h. Los meses de **selección** (los "
             f"{configuration.backtest_selection_months} anteriores al examen) eligen la técnica; los meses de **examen** "
             f"({boundaries['test_start']} a {boundaries['current'] - 1}) la miden sin que la haya visto. El error se "
             "compara con el ruido binomial del mes (err_norm ≈ 1: tan cerca como permite el azar). Una técnica sustituye a la "
             f"técnica por defecto ({configuration.default_technique}) solo si le gana por un margen en los mismos meses. Compiten todas las técnicas que "
             "la historia permite, también las de series temporales.", "",
             f"Los pools juzgados cubren el **{judged_share:.0%}** del dinero por predecir; el resto toma la técnica por defecto.", "",
             "**Ranking en los meses de selección** (error normalizado medio):", "", markdown_table(ranking),
             "**Elecciones:**", "", markdown_table(choices, 0),
             "**Precisión en el examen por tramo** (error medio de la tasa en pp, ponderado por dinero; dentro_banda: "
             "proporción de errores dentro de la banda del 90 %):", "", markdown_table(by_band),
             "**Precisión en el examen por nivel de riesgo** (tramo corto):", "", markdown_table(by_level),
             "**La cartera en el examen, serie a serie: la cascada desde la baseline** (paso 7.1; cada serie predicha con solo "
             "lo que se sabía h meses antes. baseline: la tasa de su celda mandatory en los últimos "
             f"{configuration.baseline_months} meses × lo que vence, el cálculo que cualquiera podría hacer; raw: la serie con su "
             "propia historia, sin escalera; por_defecto: escalera y credibilidad con la técnica por defecto; framework: además la "
             "técnica elegida. error_total: de la suma de la cartera; error_usd_medio y ganancia_usd_mes: el mismo error en USD "
             "y lo que se yerra menos que la baseline; meses_mejor_que_baseline: consistencia; wape_series: serie a serie, sin "
             "compensaciones. La estructura (escalera, k, técnica elegida) se estima con todos los meses cerrados: el examen del "
             "framework es algo optimista):", "",
             markdown_table(results["portfolio_exam_summary"], 3) if results.get("portfolio_exam_summary") is not None else "",
             markdown_table(results["portfolio_exam"], 3) if results.get("portfolio_exam") is not None else ""]
    by_size, total = exam_error_against_noise(results)
    if by_size is not None and len(by_size):
        lines += ["", "**El error frente al ruido** (la regla del ruido: una diferencia menor que el ruido no es una diferencia; "
                  "un error del tamaño del ruido no es un fallo, es el límite). `ruido_pp`: lo que se equivocaría una predicción "
                  "perfecta, √(p(1−p)/n). `error_vs_ruido` ≈ 1: al límite; claramente mayor que 1: falta algo que se podía saber. "
                  "`parte_del_error_que_es_ruido`: la parte del error que ninguna predicción puede quitar. Por tamaño de la serie:", "",
                  markdown_table(by_size.assign(parte_del_error_que_es_ruido=by_size["parte_del_error_que_es_ruido"].map("{:.0%}".format)), 2),
                  "", "Y el **total de la cartera** en cada mes de examen, con el mismo cálculo: al juntar todo el volumen, el ruido baja "
                  "con la raíz del tamaño, y lo que queda por encima del ruido es error del modelo:", "",
                  markdown_table(total, 3)]
    precision_by_type = exam_precision_by_series_type(results)
    if precision_by_type is not None:
        overall = precision_by_type.iloc[0]
        headline.append(("examen serie a serie: dentro del intervalo · WAPE, framework frente a la serie sola (raw)",
                         f"{overall['en_intervalo']:.0%} vs {overall['en_intervalo_raw']:.0%} · {overall['wape']:.1%} vs {overall['wape_raw']:.1%}"))
        lines += ["**Cuánto acertamos, forecast serie a forecast serie, y cuánto mejora frente a la serie sola** (paso 7.1: en cada "
                  "mes de examen y horizonte, el framework —su composición con credibilidad— y la serie sola con su propia historia "
                  "(raw), sobre las mismas filas y su propia pipeline; cada predicción con su intervalo, construido como la banda "
                  "del forecast). Cruzado por el tipo de serie: volatilidad (φ de su propia tasa), tendencia y estacionalidad:", "",
                  markdown_table(precision_by_type.assign(
                      predicciones=precision_by_type["predicciones"].astype(int),
                      en_intervalo=precision_by_type["en_intervalo"].map("{:.0%}".format),
                      en_intervalo_raw=precision_by_type["en_intervalo_raw"].map("{:.0%}".format),
                      wape=precision_by_type["wape"].map("{:.1%}".format),
                      wape_raw=precision_by_type["wape_raw"].map("{:.1%}".format),
                      sesgo=precision_by_type["sesgo"].map("{:+.1%}".format)), 0)]
    return "\n".join(lines) + "\n"


def exam_precision_by_series_type(results: dict):
    """The exam of every forecast series summed by type of series (all, by volatility, trend and seasonality):
    the framework against the series alone (raw), counts and units added, then the ratios."""
    audit, series_exam = results.get("audit"), results.get("series_exam")
    if not audit or not series_exam:
        return None
    exam = series_exam["per_series"].merge(audit["series_dynamics"], on=SERIES_ID_COLUMN, how="left")
    exam = exam[exam["framework_predictions"] > 0]
    if exam.empty:
        return None
    measured = exam["measurable"] == "yes"
    segments = [("todas las series examinadas", exam.index == exam.index),
                ("volatilidad baja (φ ≤ 1,5)", exam["phi"] <= 1.5),
                ("volatilidad alta (φ > 1,5)", exam["phi"] > 1.5),
                ("con tendencia (medible)", measured & exam["trend"].fillna(0).ne(0)),
                ("sin tendencia (medible)", measured & exam["trend"].fillna(0).eq(0)),
                ("estacional (medible)", measured & exam["seasonal"].eq(1)),
                ("no estacional (medible)", measured & exam["seasonal"].eq(0)),
                ("sin dinámica medible / sin composición", ~measured)]
    rows = []
    for name, mask in segments:
        block = exam[mask]
        if block.empty:
            continue
        rows.append({"segmento": name, "series": len(block), "predicciones": block["framework_predictions"].sum(),
                     "en_intervalo": block["framework_in_band"].sum() / block["framework_predictions"].sum(),
                     "en_intervalo_raw": block["raw_in_band"].sum() / block["raw_predictions"].sum(),
                     "wape": block["framework_abs_err_units"].sum() / block["framework_real_units"].sum(),
                     "wape_raw": block["raw_abs_err_units"].sum() / block["raw_real_units"].sum(),
                     "sesgo": block["framework_pred_units"].sum() / block["framework_real_units"].sum() - 1})
    return pd.DataFrame(rows)


def chapter_uplift(results: dict, configuration: Config) -> str:
    cells, check = results.get("uplift_cells"), results.get("contract_check")
    comparison, verdict = results.get("uplift_backtest"), results.get("uplift_verdict")
    if cells is None:
        return "## 6 · La revalorización\n\n_(pasos 3.1-3.3 no ejecutados)_\n"
    by_origin = cells.groupby("uplift_origen").agg(celdas=(UPLIFT_CELL_ID_COLUMN, "size"), uplift_medio=("uplift", "mean"),
                                                   renovadores=("renovadores", "sum")).reset_index()
    lines = ["## 6 · La revalorización: a qué precio se renueva", "",
             "El uplift es lo que paga quien renueva respecto a lo que vencía (1,00 = mismo precio). Dos vías: la "
             "**estadística** (el uplift observado en las renovaciones pasadas de su celda de uplift: dimensiones mandatory, "
             "extras de revalorización y tramo de descuento; con menos de "
             f"{configuration.uplift_floor:.0f} renovadores toma el de su padre) y la del **contrato** (quien pagó con "
             "descuento d renueva al MSRP del vencimiento: 1 / (1 − d)), sobre todas las filas con descuento conocido o "
             "solo sobre las neutras (sin estado: contrato_neutras); el examen elige.", "",
             "**Las celdas por origen de su uplift:**", "", markdown_table(by_origin),
             "**La regla de contrato frente a las renovaciones pasadas con descuento conocido** (ratio_realizacion 1 = exacta):", "",
             markdown_table(check, 3),
             f"**El backtest del uplift** (meses de examen; precio de las renovaciones reales predicho por cada vía, con el "
             f"uplift estadístico estimado ANTES del examen): la vía usada donde hay descuento es la "
             f"**{verdict['via_usada_con_descuento']}**.", "", markdown_table(comparison, 3)]
    return "\n".join(lines) + "\n"


VERDICT_MARKS = {VERDICT_HOLDS: "✔", VERDICT_FAILS: "✘", VERDICT_INCONCLUSIVE: "≈", VERDICT_WARNING: "⚠"}


def chapter_verdict(results: dict, configuration: Config) -> str:
    """Chapter 0: every hypothesis of every block with its number and verdict; what the official band is made of."""
    verdicts = verdict_table(configuration)
    lines = ["## 0 · El veredicto: qué hipótesis se cumplen con estos datos", "",
             "Cada bloque descansa en hipótesis (una serie sola es ruido; la técnica elegida mejora a la por defecto; la "
             "vía estadística predice el precio mejor que el contrato; la curva predice la cuota final de un estado; simular "
             "la captación bate a 'lo mismo que hace un año'; el framework bate a la baseline…). En el sintético se cumplen "
             "por construcción; con datos reales hay que comprobarlas una a una. ✔ se cumple · ✘ no se cumple · ≈ no "
             "concluyente (pocos meses, una diferencia dentro del ruido) · ⚠ revisar. El detalle, en `sff_veredicto_bloques`.", ""]
    if len(verdicts):
        counts = verdicts["veredicto"].value_counts()
        lines += [f"**{len(verdicts)} preguntas** · " + " · ".join(f"{VERDICT_MARKS.get(name, '')} {name}: {count}".strip()
                                                                for name, count in counts.items()), ""]
        shown = verdicts.assign(v=verdicts["veredicto"].map(lambda name: VERDICT_MARKS.get(name, "·")))
        lines += [markdown_table(shown[["bloque", "paso", "v", "pregunta", "valor", "referencia"]]), ""]
        attention = verdicts[verdicts["veredicto"].isin([VERDICT_FAILS, VERDICT_WARNING, VERDICT_INCONCLUSIVE])]
        if len(attention):
            lines += ["**Lo que no se cumple o hay que mirar:**", ""]
            lines += [f"- **{row['bloque']} · {row['pregunta']}** {row['valor']} ({row['veredicto']}): {row['detalle']}"
                      for _, row in attention.iterrows()]
            lines.append("")
    forecast = results.get("forecast")
    if forecast is not None and "banda_tasa_y_precio_usd" in forecast["by_year"].columns:
        by_year = forecast["by_year"]
        band = by_year[by_year["esperado_usd"] > 0][["ano", "total_usd", "error_examen_usd", "banda_tasa_usd", "banda_precio_usd",
                                                      "banda_tasa_y_precio_usd", "banda_captacion_usd", "banda_time_series_usd"]]
        lines += ["**De qué está hecha la banda oficial** (paso 7.2; error típico del total del año, cada parte medida en "
                  "meses cerrados: tasa y precio en cuadratura mes a mes, más la captación simulada y el universo time "
                  "series; las tres últimas columnas suman la banda):", "", markdown_table(band, 0)]
    return "\n".join(lines) + "\n"


def chapter_data(extract_summary: dict, results: dict, configuration: Config) -> str:
    """Chapter 1 · B1: the extract and what was corrected (the earlier chapter 1), and the profile of the pipeline."""
    lines = ["## 1 · B1 · Los datos: el extracto, lo que se corrigió y el perfil de la pipeline", "",
             as_section(chapter_raw(extract_summary, results, configuration))]
    profile = configuration.data_profile
    if profile is not None and len(profile):
        view = lambda section, columns: profile[profile["seccion"] == section][columns]
        lines += ["### El perfil de la pipeline (`sff_perfil_datos`)", "",
                  "Lo que hay que ver de los datos antes de leer cualquier ganancia del framework: cuánto dinero hay, dónde "
                  "está, cuánta historia y soporte propio tiene, y cómo se reparte. Tasa: últimos 12 meses cerrados; futuro: "
                  "desde el mes en curso; crecimiento_anual: la pipeline de los últimos 12 meses cerrados frente a los 12 "
                  "anteriores; cuota_futuro_con_soporte: la parte de su dinero futuro en series con soporte propio.", "",
                  "**La cartera:**", "", markdown_table(view("cartera", ["series", "usd_vence", "usd_renovado", "tasa_unidades",
                                                                          "tasa_usd", "crecimiento_anual", "usd_vence_futuro"]), 3),
                  "**La concentración del dinero a predecir:**", "", markdown_table(view("concentracion", ["dimension", "valor", "series", "cuota_usd_futuro"]), 3),
                  "**El soporte propio y la historia de las series con dinero a predecir:**", "",
                  markdown_table(view("soporte", ["valor", "series", "usd_vence_futuro", "cuota_usd_futuro"]), 3),
                  markdown_table(view("historia", ["valor", "series", "usd_vence_futuro", "cuota_usd_futuro"]), 3),
                  "**Cada dimensión, valor a valor** (las de más dinero por predecir primero dentro de cada dimensión):", "",
                  markdown_table(view("dimension", ["dimension", "valor", "series", "usd_vence", "tasa_unidades", "tasa_usd",
                                                    "crecimiento_anual", "cuota_usd_futuro", "cuota_futuro_con_soporte"])
                                 .sort_values(["dimension", "cuota_usd_futuro"], ascending=[True, False]), 3),
                  "**Mes a mes:**", "", markdown_table(view("mes", ["valor", "series", "usd_vence", "usd_renovado", "tasa_unidades",
                                                                     "tasa_usd"]), 3)]
    return "\n".join(lines) + "\n"


def chapter_rate(results: dict, configuration: Config, headline: list) -> str:
    """Chapter 2 · B2: the support before and after the ladder, the dynamics, how well the rate is predicted."""
    return "\n".join(["## 2 · B2 · La tasa de renovación", "",
                      as_section(chapter_support(results, configuration, headline)),
                      as_section(chapter_dynamics(results, configuration)),
                      as_section(chapter_precision(results, configuration, headline))]) + "\n"


def chapter_price(results: dict, configuration: Config) -> str:
    """Chapter 3 · B3: the uplift and the contract rule, the paths, the ways of the statistical uplift, the error of the
    price of the monthly total."""
    lines = ["## 3 · B3 · El precio: a qué precio se renueva", "", as_section(chapter_uplift(results, configuration))]
    verdict = results.get("uplift_verdict") or {}
    ways = verdict.get("formas_uplift")
    if ways is not None and len(ways):
        lines += ["**Las formas del uplift estadístico en el mismo examen** (estimadas antes del examen; ← la de la Config, "
                  "la que usa el forecast):", "", markdown_table(ways, 4)]
    monthly = verdict.get("error_precio_por_mes")
    if monthly is not None and len(monthly):
        lines += [f"**El error del precio del total de cada mes de examen** (la parte del precio de la banda oficial: "
                  f"{verdict.get('error_precio_total', 0):.1%} de media):", "", markdown_table(monthly, 4)]
    return "\n".join(lines) + "\n"


def chapter_states(results: dict, configuration: Config) -> str:
    """Chapter 4 · B4: the exam of the maturation curve, the honest exam of the total, the maturation by year."""
    lines = ["## 4 · B4 · Los estados del cliente al vencer", "",
             "El extracto trae los estados de hoy; una licencia que vence dentro de meses aún puede hacer softcancel o quedarse "
             "dormant. La caja del paso 4.1 aprende la curva de cada estado fechado y mueve unidades a los estados que "
             "tendrán al vencer (sin crearlas). Tres pruebas: la curva frente a la cuota final, el total del mes con los "
             "estados predichos (examen honesto, paso 7.1) y lo que cambia el forecast."]
    box = results.get("state_maturation")
    for state, exam in (getattr(box, "curve_exams", {}) or {}).items():
        lines += ["", f"**La curva de {state} en el examen** (su cuota de las unidades que vencen predicha con lo visto h meses "
                      "antes, frente a la final; cuota_vista_entonces: los estados vistos entonces, sin curva):", "",
                  markdown_table(exam, 4)]
    undated = [state for state in getattr(box, "states", []) if state not in (getattr(box, "curve_exams", {}) or {})]
    if undated:
        lines += ["", f"**Sin fecha:** {', '.join(undated)}: su cuota al vencer sale de la proporción histórica, sin examen."]
    exam = results.get("series_exam")
    if exam is not None and "framework_honesto_error_total" in exam["by_month"].columns:
        honest = exam["by_month"][["mes", "h", "renovadas_reales", "framework_error_total", "framework_estados_hoy_error_total",
                                   "framework_honesto_error_total"]]
        lines += ["", "**El examen honesto** (paso 7.1: el total de cada mes de examen con las tasas del framework y los "
                      "estados finales (el techo), los vistos h meses antes, o esos más la curva):", "", markdown_table(honest, 4)]
    forecast = results.get("forecast")
    maturation = forecast.get("maturation") if forecast is not None else None
    if maturation is not None and len(maturation):
        year_of = maturation[configuration.period_col].map(lambda month: month.year)
        by_year = maturation.groupby(year_of)[["renovado_usd_base", "ajuste_usd", "renovado_usd_final"]].sum().reset_index()
        by_year["ajuste_pct"] = by_year["ajuste_usd"] / by_year["renovado_usd_base"]
        lines += ["", "**Lo que cambia el forecast** (con los estados de hoy → con los estados al vencer):", "",
                  markdown_table(by_year.rename(columns={configuration.period_col: "ano"}), 0)]
    return "\n".join(lines) + "\n"


def chapter_future_pipeline(results: dict, configuration: Config) -> str:
    """Chapter 5 · B5: the simulated acquisitions against the year before, and their exam."""
    lines = ["## 5 · B5 · La pipeline futura: renovaciones proyectadas y captación simulada", "",
             "**La ventana de simulación** (del mes en curso, incluido, a diciembre): lo que ocurre en ella vence doce meses "
             "después. Las licencias de 1 año que vencen en la ventana renuevan como prevé el forecast y su renovación vence "
             "el mismo mes del año siguiente (**proyectada**). La captación de la ventana se simula por cada valor de "
             "captación: lo vendido el mismo mes del año anterior × el nivel, al valor medio de 12 meses "
             f"(descuento {configuration.acquisition_discount:.0%}; **simulada**). La pipeline de las licencias de 1 año "
             "vendidas o renovadas desde el mes en curso aún no se conoce: el paso 1.3 la borra y se proyecta."]
    future_pipeline = results.get("future_pipeline")
    if future_pipeline is not None:
        comparison = future_pipeline.get("comparison")
        if comparison is not None and len(comparison):
            lines += ["", "**Lo simulado frente al año anterior:**", "", markdown_table(comparison, 3)]
        if len(future_pipeline["exam"]["summary"]):
            exam = future_pipeline["exam"]["summary"].copy()
            for column_name in ("error_unidades", "error_base_unidades", "error_usd", "error_base_usd", "sesgo_unidades"):
                if column_name in exam.columns:
                    exam[column_name] = 100 * exam[column_name]
            lines += ["", "**El examen de la captación simulada:** en cada uno de los últimos "
                          f"{configuration.acquisition_exam_origins} meses cerrados se simula la captación como si fuese el mes "
                          "en curso, con los meses anteriores solamente, y se compara con lo vendido. h = meses tras el origen; "
                          "errores en % de lo vendido; base = lo mismo que un año antes (nivel 1):", "", markdown_table(exam, 1)]
    return "\n".join(lines) + "\n"


def chapter_time_series(results: dict, configuration: Config) -> str:
    """Chapter 6 · B6: the exam of every level rule and the universe by year and origin."""
    lines = ["## 6 · B6 · El universo time series (retail a suscripción)", ""]
    universe = results.get("time_series_universe")
    if universe is None or not len(universe.table):
        return "\n".join(lines + ["_(sin filas time series en esta ejecución)_"]) + "\n"
    by_rule = universe.exam["by_rule"]
    if len(by_rule):
        shown = (by_rule.assign(error_pct=100 * by_rule["error_usd"])
                 .pivot_table(index="candidato", columns="h", values="error_pct", sort=False).round(1))
        shown.columns = [f"h{int(h)}" for h in shown.columns]
        shown["media"] = (100 * by_rule.groupby("candidato", sort=False)["error_medio_regla"].first()).round(1)
        shown = shown.sort_values("media")
        lines += ["El universo solo trae su resultado (lo que convirtió): se predice con su propia serie, el mismo mes del año "
                  "anterior × el nivel de la región, valorado al valor por unidad de sus últimos meses. **La regla del nivel "
                  "y la ventana del valor por unidad las elige el examen**: en cada uno de los últimos "
                  f"{configuration.ts_exam_origins} meses cerrados se simula con los meses anteriores y se compara el revenue "
                  f"total del mes con lo que convirtió; simula el candidato de menor error medio, **{universe.exam['candidate']}**, y su error "
                  "a cada horizonte es la banda de sus filas (error en %; detalle en `sff_ts_examen`):", "",
                  markdown_table(shown.reset_index(), 1)]
    time_series = results.get("time_series")
    if time_series is not None and len(time_series):
        by_origin = (time_series.assign(ano=time_series[configuration.period_col].map(lambda month: month.year))
                     .groupby(["ano", "origen"])[["unidades", "valor", "revenue", "revenue_bajo", "revenue_alto"]].sum()
                     .reset_index())
        lines += ["", "**El universo por año y origen** (el detalle región × mes está en sff_nucleo):", "",
                  markdown_table(by_origin, 0)]
    return "\n".join(lines) + "\n"


def chapter_forecast(results: dict, configuration: Config, headline: list) -> str:
    """Chapter 7 · B7: the forecast by month and by year, the official band and what it is made of, where the rate of
    the future rows comes from, the total by year and origin."""
    forecast = results.get("forecast")
    if forecast is None:
        return "## 7 · B7 · El forecast en dinero\n\n_(paso 7.2 no ejecutado)_\n"
    rows, by_month, by_year = forecast["forecast"], forecast["by_month"], forecast["by_year"]
    exam_band = "error_examen_usd" in by_year.columns
    for _, year_row in by_year[by_year["esperado_usd"] > 0].iterrows():
        if exam_band:
            headline.append((f"TOTAL {int(year_row['ano'])}: renovado real + esperado + time_series (± error del examen)",
                             f"${year_row['total_usd']:,.0f} ± ${year_row['error_examen_usd']:,.0f}"))
        else:
            headline.append((f"TOTAL {int(year_row['ano'])}: renovado real + esperado + time_series (± cuadratura, sin examen)",
                             f"${year_row['total_usd']:,.0f} ± ${year_row['banda_cuadratura_usd']:,.0f}"))
    origins = rows.groupby("origen_tasa").agg(filas=("esperado_usd", "size"), usd_vence=(configuration.pipeline_usd_col, "sum"),
                                              esperado_usd=("esperado_usd", "sum")).reset_index()
    lines = ["## 7 · B7 · El forecast en dinero", "",
             "Cada fila futura: **USD que vence × tasa de su serie × uplift de su celda**, con las unidades que tendrá al "
             "vencer. La tasa la predice la técnica elegida para la composición de la serie a su horizonte; una serie que toma "
             "prestado conserva su diferencia de nivel con el pool en proporción a su credibilidad"
             + (" (activado)" if configuration.apply_credibility_shift else " (desactivado)") + ". **La banda oficial es la "
             "de los exámenes** (error_examen_usd): el error de la tasa (examen 7.1; una renovación proyectada suma el del mes "
             "en que se predijo su volumen), el del precio del total (3.3; con la tasa, en cuadratura), el del volumen de la "
             "captación simulada (5.1) y el del universo time series (6.1); por año, los errores de sus meses sumados. Es el "
             "error típico, no un intervalo del 90 %. Las bandas **lineal** y **cuadratura** quedan como referencia.", "",
             "**Por mes:**", "", markdown_table(by_month, 0),
             "**Por año** (lo renovado en los meses cerrados + lo esperado en los futuros; banda_*: de qué está hecha la banda):", "",
             markdown_table(by_year, 0),
             "**De dónde sale la tasa de las filas futuras:**", "", markdown_table(origins, 0)]
    total = results.get("forecast_total")
    if total is not None and len(total):
        for _, row in total[total["origen"] == TOTAL_ORIGIN_TOTAL].iterrows():
            headline.append((f"pipeline {int(row['ano'])}: vence, con la reentrada time_series", f"${row['usd_vence']:,.0f}"))
        lines += ["**El total del forecast por año y origen** (paso 7.3; es la SUMA de `sff_nucleo` por `forecast_year` y "
                  "`forecast_pipeline_source`, comprobado en el núcleo: en Power BI, SUM(forecast_renewed_USD) y "
                  "SUM(forecast_to_renew_USD)). Total de este año = renovaciones de la pipeline + ts_real + ts_proyectado · "
                  "Total del año siguiente = forecast extendido + ts_simulado + ts_reentrada. usd_vence: pipeline; "
                  "usd_renovado: renovaciones o revenue:", "", markdown_table(total, 0)]
    return "\n".join(lines) + "\n"


def chapter_control(results: dict, configuration: Config) -> str:
    """Chapter 8 · B8: the rows added and wiped (8.1), the validation of the chain (8.5), the relations of the Power BI
    model (8.4), where every series is, and what to validate with real data."""
    lines = ["## 8 · B8 · Control: filas añadidas, validación, Power BI y lo que hay que validar con datos reales", "",
             as_section(chapter_new_rows(results, configuration))]
    validation = results.get("validation")
    if validation is not None:
        lines += ["### La validación final de la cadena (paso 8.5)", "", markdown_table(validation)]
    power_bi = results.get("power_bi")
    relations = power_bi.get("relations") if isinstance(power_bi, dict) else None
    if relations is not None and len(relations):
        lines += ["### Las relaciones del modelo de Power BI (paso 8.4, `sff_relaciones_modelo`)", "",
                  "Cada relación comprobada sobre las tablas de esta ejecución: la clave del lado 1 es única y cuántas filas "
                  "del lado n no la encuentran (quedan en blanco en Power BI).", "", markdown_table(relations, 0)]
    series = results.get("series")
    count = f" ({len(series):,} series)" if series is not None else ""
    lines += ["### Dónde está cada serie", "",
              f"`sff_forecast_series` tiene una fila por forecast series{count} con todo lo que el framework sabe de ella "
              "(ruta, soporte, etapas, composición, credibilidad, tasa estimada, nivel de riesgo, dinámica, técnica de cada "
              "tramo y su examen); `sff_nucleo`, lo mismo fila a fila con los meses (en Power BI: unir por `s1_4_fs_id`). "
              "`sff_veredicto_bloques` y `sff_perfil_datos` dan el veredicto y el perfil de esta ejecución.", "",
              "### Lo que hay que validar con datos reales", "",
              "En el sintético las hipótesis se cumplen por construcción. Con datos reales, en este orden: (1) el perfil de "
              "la pipeline (capítulo 1): concentración, soporte propio, historia, descuento conocido; (2) la escalera: si la "
              "serie sola gana serie a serie y en el total, el suelo de soporte y la credibilidad están mal calibrados; (3) la "
              "técnica elegida frente a la técnica por defecto: si no mejora en el examen, simplificar; (4) el sesgo del total "
              "en el examen: un error que es sesgo es un cambio de nivel, no falta de técnica; (5) la maduración de los estados, "
              "el ajuste de modelo más grande del forecast; (6) la captación simulada por valor frente a 'lo mismo que hace un "
              "año'; (7) la regla del nivel del universo time series y su peso en la banda. Cada punto es una fila de "
              "`sff_veredicto_bloques` (capítulo 0)."]
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 8 · THE BLOCK: run the steps in order, keep the results
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B8"
BLOCK_NAME = "OUTPUT AND CONTROL"
BLOCK_MISSION = ("write, explain and validate the forecast without changing any number, so every number can be "
                 "followed back to the step and the row it comes from")
BLOCK_READS = ("extract_summary", "fine_table", "forecast_units", "rated_units", "series_table", "series_rate",
               "time_series_rows", "ladder", "series_estimate", "money_by_level", "pool_series", "pool_reference",
               "pool_dynamics", "portfolio_profile", "portfolio_dynamics", "backtest", "rate_box", "contract_check",
               "uplift_cells", "uplift_backtest", "uplift_verdict", "price_increases", "state_maturation",
               "future_pipeline", "time_series_universe", "series_exam", "forecast", "time_series", "forecast_total")
BLOCK_GIVES = ("new_rows", "audit", "core", "core_legend", "forecast_series", "power_bi", "validation", "report")


def results_of_the_run(tables: dict) -> dict:
    """The tables of a run under the names the report, the validation and the tests read."""
    renamed = {"series_table": "series"}
    results = {renamed.get(name, name): value for name, value in tables.items()}
    if "series_exam" in tables:
        results["portfolio_exam"] = tables["series_exam"]["by_month"]
        results["portfolio_exam_summary"] = tables["series_exam"]["summary"]
    return results


class OutputBlock:
    """BLOCK 8 · OUTPUT AND CONTROL (see the header of this module): its results are attributes, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.new_rows = None
        self.audit = None
        self.core = None
        self.core_legend = None
        self.forecast_series = None
        self.power_bi = None
        self.validation = None
        self.report = None
        self.seconds = {}

    def run(self, tables: dict) -> "OutputBlock":
        """Every step of the block, in order. tables: every result of blocks 1 to 7, by name; the validation and the
        report read them all, with what this block adds."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 8, BLOCK_NAME, BLOCK_MISSION)
        tables = dict(tables)

        started = time.time()
        self.new_rows = count_the_new_rows(tables["fine_table"], tables["rated_units"], tables["forecast"]["forecast"],
                                           configuration)
        self.seconds["8.1 new_rows"] = time.time() - started

        started = time.time()
        self.audit = build_audit_tables(tables["ladder"], tables["series_rate"], tables["series_estimate"],
                                        tables["rated_units"], tables["pool_series"], tables["pool_reference"],
                                        tables["pool_dynamics"], tables["backtest"], tables["forecast"]["forecast"],
                                        configuration, rate_box=tables["rate_box"])
        self.seconds["8.2 audit"] = time.time() - started

        started = time.time()
        self.core, self.core_legend, self.forecast_series = build_core_table(
            tables["fine_table"], configuration, forecast_units=tables["forecast_units"],
            rated_units=tables["rated_units"], series_table=tables["series_table"],
            series_rate=tables["series_rate"], series_estimate=tables["series_estimate"],
            pool_dynamics=tables["pool_dynamics"], technique_decision=tables["backtest"]["decision"],
            exam_by_pool=tables["backtest"]["exam_by_pool"], forecast=tables["forecast"]["forecast"],
            time_series_rows=tables["time_series_rows"], time_series_table=tables["time_series"],
            forecast_total=tables["forecast_total"], ladder_stages=tables["ladder"]["stages"],
            series_dynamics=self.audit["series_dynamics"], series_exam=tables["series_exam"]["per_series"])
        self.seconds["8.3 core"] = time.time() - started

        started = time.time()
        self.power_bi = build_power_bi_tables(self.core, configuration)
        self.seconds["8.4 power_bi"] = time.time() - started

        tables.update(new_rows=self.new_rows, audit=self.audit, core=self.core, core_legend=self.core_legend,
                      forecast_series=self.forecast_series, power_bi=self.power_bi)
        started = time.time()
        self.validation = validate_chain(tables["extract_summary"], results_of_the_run(tables), configuration)
        self.seconds["8.5 validation"] = time.time() - started

        tables.update(validation=self.validation)
        started = time.time()
        self.report = build_report(tables["extract_summary"], results_of_the_run(tables), configuration)
        self.seconds["8.6 report"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
