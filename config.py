"""
config.py — The configuration of SFF (mirror version, grown step by step).

What it holds (grows with the steps that need it):
  · THE COLUMN CONTRACT: which role every column of the raw plays. A column has one
    role (the two extra groups may share a column); the raw may not carry a column
    with no role, nor miss a declared one.
  · THE CALENDAR: the current month and the number of exam months. The role of every
    month is generated from them; the raw's own role columns are never read.

  · THE LOG: level, colours and an optional file. The logger is configured once, when
    the Config is built, and every step logs through it:
      configuration.log_step_start(...)   title, purpose, actions and output     (blue)
      configuration.log_action(...)       one action of the step, as it is done  (blue)
      configuration.log_check(...)        one numbered check: ok / WARN / FAIL   (green / yellow / red)
      configuration.log_check_summary()   the count of the checks; stops if any FAILED
      configuration.logger.doc(...)       any explanation or summary line        (blue)
      configuration.show_table(...)       concrete rows, as a pandas table (never logged)
  · THE IDS: how a row names its rate series, its forecast unit and its uplift cell
    (joined text ids; the core and the BI join on them).
  · THE DISCOUNT: ONE column of the extract (the exact discount, a share: 0.25 = 25 %)
    serves both sides. As it is, it feeds the contract rule of the uplift
    (1 / (1 − discount)) and is part of the identity of a fine row. Cut into buckets
    (discount_bucket_edges), it is a dimension of the uplift cell. A null discount is
    unknown, not 0 %: its bucket is "sin_dato" and it takes the statistical uplift.
  · THE STATISTICAL PARAMETERS: the z of the bands and the support floor.
  · THE TABLES: where they are written (a SQL engine, or CSV files when there is none).
    Every table is read back after writing and its row count checked.

Every parameter of every step is here, tagged with the steps that read it ([..]); a test checks the tags.

Usage: subclass Config in main.py, override `read_raw()` and declare the columns there.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import datetime
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from logging_helpers import DOC_LEVEL, LoggerManager


# ═══════════════════════════════════════════════════════════════════════════════
# THE VOCABULARY: the labels the framework persists. Column names, the values inside them and the logical names of the
# tables (table_prefix is added when written). They are in Spanish because they are the contract with the tables, the
# report and Power BI; a label changes here and nowhere else. Not fields of the Config: changing one between runs would
# break the tables and the BI that read them.
# ═══════════════════════════════════════════════════════════════════════════════
# ─── the role of every month (generated from the calendar of the configuration) ──
ROLE_TRAIN = "entrenamiento"          # closed months the forecast learns from
ROLE_TEST = "examen"                  # closed months that only evaluate
ROLE_PROJECTION = "proyeccion"        # the current month and the future

# ─── the columns step 02 adds to the raw ─────────────────────────────────────────
CALENDAR_ROLE_COLUMN = "rol"                          # the role of the row's month (the three above)
CURRENT_MONTH_COLUMN = "es_mes_en_curso"              # 1 in the current month, 0 elsewhere
S0_RENEWED_UNITS_COLUMN = "s0_renovados_unidades"     # the raw's renewed units, before step 02 touches them
S0_RENEWED_USD_COLUMN = "s0_renovados_usd"            # the raw's renewed USD, before step 02 touches them
S0_PIPELINE_UNITS_COLUMN = "s0_vencen_unidades"   # the pipeline as the raw had it (step 02 wipes what is not known yet)
S0_PIPELINE_USD_COLUMN = "s0_vencen_usd"

# ─── the ids and keys step 03 adds (kept in English: the BI joins on these names) ─
SERIES_ID_COLUMN = "fs_id"                   # the rate series: mandatory | timevarying | extra_renovacion
UNIT_ID_COLUMN = "fu_id"                     # the forecast unit: fs_id | month (where the RATE is predicted)
UPLIFT_CELL_ID_COLUMN = "uplift_cell_id"     # the price context: uplift mandatory | extra_revalorizacion

# The three roles in time order (coverage patterns and tables follow it).
ROLES_IN_ORDER = [ROLE_TRAIN, ROLE_TEST, ROLE_PROJECTION]

# ─── the columns steps 04 and 05 add ─────────────────────────────────────────────
FINE_ROWS_COLUMN = "n_filas_finas"           # step 04: fine rows added into a forecast unit
COVERAGE_COLUMN = "cobertura"                # step 05: the roles a series has, in time order, joined with "+"
ROUTE_COLUMN = "ruta"                        # step 05: how the series will be treated

# ─── the route of a series (from its coverage) ───────────────────────────────────
ROUTE_PREDICTABLE = "predecible"             # closed months AND something to predict: its rate is estimated
ROUTE_HISTORY_ONLY = "solo_historia"         # nothing to predict: kept, it lends history to its relatives
ROUTE_FUTURE_ONLY = "solo_futuro"            # nothing to learn from: predicted from its relatives

# ─── more tables ─────────────────────────────────────────────────────────────────

# The months that are truth: closed and used to learn and to evaluate.
TRUTH_ROLES = (ROLE_TRAIN, ROLE_TEST)

# ─── the sign of a series (from its ACTIVE timevarying flags) ─────────────────────
SIGN_NEUTRAL = "neutro"                      # no flag active
SIGN_NEGATIVE = "negativo"                   # only flags that rotate towards churn
SIGN_POSITIVE = "positivo"                   # only flags that rotate towards renewal
SIGN_MIXED = "mixto"                         # both kinds active: never pooled with the others

# ─── the columns step 06 adds ───────────────────────────────────────────────────
RATE_COLUMN = "tasa"                         # step 06: renewed / due units of a unit; null = no information
SYNTHETIC_COLUMN = "sintetica"               # step 06: 1 on a gap row (a month with no unit inside the history)
SIGN_COLUMN = "signo"                        # step 06: the sign of the series

# ─── the tables of step 07 ───────────────────────────────────────────────────────
TABLE_DIMENSIONS = "decision_eta2"           # step 07: how much each dimension separates the rate; collapse order

# ─── block 1 (preparation), when it is persisted to be resumed: what the steps after it read ──
TABLE_BLOCK_1_TIME_SERIES_ROWS = "bloque1_filas_time_series"   # step 01: the time_series universe
TABLE_BLOCK_1_FINE_TABLE = "bloque1_tabla_fina"               # step 03: every row with its ids (index = the row id)
TABLE_BLOCK_1_FORECAST_UNITS = "bloque1_forecast_units"       # step 04: one row per series and month
TABLE_BLOCK_1_SERIES = "bloque1_series"                       # step 05: one row per series with its route
TABLE_BLOCK_1_RATED_UNITS = "bloque1_units_con_tasa"          # step 06: the units with gaps and the rate of every unit
TABLE_BLOCK_1_SERIES_RATE = "bloque1_series_tasa"             # step 06: support, own rate, error and sign of every series
TABLE_BLOCK_1_MANIFEST = "bloque1_manifiesto"                 # every column of those tables and its type
TABLE_BLOCK_1_FINGERPRINT = "bloque1_huella"                  # the extract, current month and Config fields it was built from

# ─── the ladder (steps 08 and 09) ────────────────────────────────────────────────
SIGN_TOKEN = "SIG="                          # inside a relative's pattern: the timevarying block summarised as its sign
WILDCARD = "*"                               # inside a relative's pattern: a dimension collapsed or annulled
COMPOSITION_ID_COLUMN = "composition_id"     # the composition of a forecast series: its group after the 3 merging stages of
                                             # the ladder (step 08); it is what is predicted, and it lends its rate

# The risk level of a series: how its rate is estimated, from best to worst.
LEVEL_OWN = "A_propio"                        # its own support is precise (≥ own_rate_floor) and it has a full year
LEVEL_OWN_SHORT = "A2_propio_corto"           # precise, but less than a year of history
LEVEL_OWN_REINFORCED = "A3_propio_reforzado"  # evidence but not precision: its rate blended with its first pool
LEVEL_BORROWED = "B_prestado"                 # below the floor; a close relative (same mandatory dims)
LEVEL_FAR = "C_lejano"                        # below the floor; the mandatory cell or a collapsed dimension
LEVEL_SIGNED_UNDER_FLOOR = "S_signo_bajo_suelo"   # signed, and no relative of its sign reaches the floor
LEVEL_MIXED = "M_signo_mixto"                 # flags of both signs: never pooled
LEVEL_NO_HISTORY = "D_sin_historia"           # nothing to learn from (solo_futuro)
LEVEL_NO_IMPACT = "N_sin_impacto"             # nothing to predict (solo_historia)

# ─── the tables of steps 08 and 09 ───────────────────────────────────────────────
TABLE_LADDER_STEPS = "ladder_steps"           # step 08: every series × pass: its group id and the group's support
TABLE_LADDER_SUMMARY = "ladder_summary"       # step 08: every pass: groups, support, money in groups that reach the floor
TABLE_RISK_LEVELS = "niveles_riesgo"          # step 09: money by risk level

# ─── the core table (built last, grows with every step) ──────────────────────────
TABLE_CORE = "nucleo"                         # ONE wide table at the fine grain: history and gaps (the future later)
TABLE_CORE_LEGEND = "nucleo_leyenda"          # every column of the core: its step, its level and how to aggregate it
ROW_ORIGIN_COLUMN = "origen_fila"             # where the row comes from
ROW_FROM_RAW = "raw"                          # a row of the extract
ROW_FROM_GAP = "hueco"                        # a month with no expirations inside the history of a series (step 06)
ROW_FROM_MATURATION = "maduracion"            # units that will carry a state when they fall due (stage 17): a prediction row

# ─── steps 10 and 12: the pool series and the backtest ───────────────────────────
GATE_SUPPORT = "soporte"                      # a pool below the support floor: not judged (it takes the default technique)
GATE_LEVEL = "nivel"                          # a pool with support: judged by the backtest
PURPOSE_SELECTION = "seleccion"               # a target month used to CHOOSE the technique
PURPOSE_EXAM = "examen"                       # a target month used only to MEASURE the chosen technique
CHAMPION_ORIGIN = "campeon"                   # the technique beat the default technique by the margin
DEFAULT_ORIGIN = "por_defecto"             # nobody beat it: the default technique stays

TABLE_POOL_SERIES = "pool_serie"              # step 10: the monthly series of every estimation id
TABLE_POOL_REFERENCE = "pool_referencia"      # step 10: one row per estimation id: months, support, rate, gate
TABLE_BACKTEST_PREDICTIONS = "backtest_predicciones"   # step 12: one row per id × target × horizon × technique
TABLE_TECHNIQUE_DECISION = "decision_tecnica"          # step 12: the chosen technique per id and horizon band
TABLE_ERROR_BANDS = "decision_bandas"                  # step 12: error quantiles per technique and horizon
TABLE_EXAM_BY_POOL = "backtest_examen"                 # step 12: the chosen technique vs the default technique in the exam, per id
TABLE_EXAM_TOTAL = "backtest_examen_total"             # step 12: the error of the TOTAL renewals in every exam month

# ─── the report ──────────────────────────────────────────────────────────────────
REPORT_FILE_NAME = "informe_sff.md"          # the report, in the output folder

# ─── steps 14-26: uplift, forecast, validation ───────────────────────────────────
UPLIFT_OWN, UPLIFT_PARENT, UPLIFT_CELL, UPLIFT_GLOBAL = "propia", "padre", "celda", "global"   # where a cell's uplift comes from
PATH_STATISTICAL = "estadistica"             # uplift observed in the past renewals of the cell
PATH_CONTRACT = "contrato"                   # uplift of the contract: 1 / (1 − discount)
PATH_CONTRACT_NEUTRAL = "contrato_neutras"   # the contract on the rows without a negative state, statistical on the rest
PATH_STATISTICAL_STATES = "estadistica_estado"   # the statistical uplift × the price factor of the row's states (step 14)
RATE_FROM_POOL = "pool"                      # the rate comes from the technique chosen for the series' estimation id
RATE_FROM_CELL = "celda_mandatory"           # no pool: the rate of its mandatory cell
RATE_FROM_GLOBAL = "global"                  # nothing else: the rate of the whole portfolio

TABLE_UPLIFT_CELLS = "uplift_celda"          # step 14: the uplift of every uplift cell
TABLE_CONTRACT_CHECK = "uplift_contrato_check"   # step 14: the contract rule against the past renewals
TABLE_UPLIFT_HOMOGENEITY = "uplift_homogeneidad"   # step 14: the renewers' due price against their cell's (exact base only)
TABLE_UPLIFT_BACKTEST = "backtest_uplift"    # step 16: statistical vs contract in the exam months
TABLE_FORECAST_MONTH = "forecast_mes"        # step 20: the total by month with its bands
TABLE_BUSINESS_SUMMARY = "resumen_negocio"   # step 20: the answers by year
TABLE_FORECAST_MATURATION = "forecast_maduracion"   # step 20: the maturation of the state, month by month
TABLE_VALIDATION = "validacion"              # step 26: the final checks across steps

# ─── step 19: the exam of the portfolio (framework vs the baseline) ──────────────
TABLE_PORTFOLIO_EXAM = "examen_cartera"               # per exam month and horizon: real vs every method, units and USD
TABLE_PORTFOLIO_EXAM_SUMMARY = "examen_cartera_resumen"   # per method and horizon: the mean error of the total and by series
METHOD_FRAMEWORK = "framework"
METHOD_BASELINE = "baseline"                 # step 19: the mandatory cell over the last baseline_months: the reference of every gain
METHOD_DEFAULT = "por_defecto"               # step 19: the framework with the default technique everywhere (no technique choice)

# ─── the extended horizon (step 20) ──────────────────────────────────────────────
PIPELINE_ORIGIN_COLUMN = "origen_pipeline"   # where the pipeline of a future row comes from
PIPELINE_REAL = "real"                       # in the extract (known)
PIPELINE_PROJECTED = "proyectada"            # an expected renewal of the simulation window, due 12 months later
PIPELINE_SIMULATED = "simulada"              # an acquisition simulated in the simulation window, due 12 months later

# ─── stage 17: the maturation of the states ───────────────────────────────────────
UNITS_TODAY_COLUMN = "unidades_hoy"                      # the box's input: the units due as seen today
WITH_DATES_COLUMN = "con_fechas"                         # the box's input: the row carries the maturation columns
UNITS_AT_DUE_COLUMN = "unidades_al_vencer"               # the box's output: the units of the combination when it falls due
DUE_UNITS_AT_DUE_COLUMN = "vencen_unidades_al_vencer"    # a future row: the units it will have when it falls due
DUE_USD_AT_DUE_COLUMN = "vencen_usd_al_vencer"           # a future row: the USD due it will have when it falls due
MATURATION_ROW_COLUMN = "fila_maduracion"                # 1 on a prediction row created by the maturation (no data)

# ─── step 21: the time_series universe (retail to subscription) and the forecast total ──
TS_REAL = "ts_real"                          # revenue of the time_series universe in the closed months of the year
TS_PROJECTED = "ts_proyectado"               # simulated revenue from the current month to ts_projection_end
TS_REENTRY = "ts_reentrada"                  # the projected months falling due again a year later, as pipeline
TOTAL_ORIGIN_RENEWED = "pipeline_renovado_real"      # renewals already booked in the closed months (normal universe)
TOTAL_ORIGIN_EXPECTED = "pipeline_real_esperado"     # expected renewals of the extract's future pipeline
TOTAL_ORIGIN_PROJECTED = "pipeline_proyectada"       # expected renewals of the extended re-entries
TOTAL_ORIGIN_SIMULATED = "pipeline_simulada"         # expected renewals of the simulated acquisition
TOTAL_ORIGIN_TOTAL = "TOTAL"
# the tables earlier versions wrote and this one does not (20-oct-2026, DOC_modelo_datos.md "Lo que NO se escribe"):
# if they are still in the database, Power BI would read stale data; the run warns and prints how to drop them
RETIRED_TABLES = ("fact_fine", "fact_fu", "calendario", "lookup_fs", "lookup_fu", "lookup_uplift_cell", "series",
                  "fu_soporte", "fu_huecos", "series_tasa", "decision_eta2_pares", "ladder_groups", "series_estimacion",
                  "composition_dynamics", "estacionalidad_cartera", "dim_tecnica", "forecast", "time_series",
                  "series_backtest", "series_technique_summary", "ficha_serie", "dimension_levels", "dimension_level_values")
TABLE_FORECAST_TOTAL = "forecast_total"      # step 21: year × origin and the total of every year

# ─── step 23: the audit tables (satellites of the core, for the drill-down) ───────
TABLE_COMPOSITION = "composition"                        # every id of every stage: who forms it, support, rate, prediction
TABLE_COMPOSITION_TECHNIQUES = "composition_techniques"  # composition × band × technique: status, errors, rank, chosen
TABLE_CREDIBILITY = "credibility"                        # every credibility reference: support, rate, k and its parts
TABLE_CREDIBILITY_MEMBERS = "credibility_members"        # reference × forecast series: what goes into its rate
TABLE_SERIES_DYNAMICS = "series_dynamics"                # every forecast series: φ, trend, seasonality vs its composition
TABLE_COMPOSITION_FORECAST_ALL = "composition_forecast_all"   # composition × future horizon × technique: the rate of each
TABLE_FORECAST_SERIES = "forecast_series"                # the core's dimension: one row per forecast series (joined by s03_fs_id)
TABLE_LADDER_MERGES = "ladder_merges"                    # step 08: every merge of every series, and whether it improves
TABLE_COMPOSITION_MEMBERS = "composition_members"        # step 08: every series in the rate of every composition (uses / lends)
TABLE_SERIES_EXAM = "series_exam"                        # step 19: every forecast series: raw and framework in the exam
TABLE_SERIES_EXAM_DETAIL = "series_exam_detail"          # step 19: series × exam month × h × method, every prediction traced
TECHNIQUE_TESTED = "tested"
TECHNIQUE_NOT_ENOUGH_HISTORY = "not_enough_history"
TECHNIQUE_COMPOSITION_BELOW_FLOOR = "composition_below_floor"


# ─── named constants ─────────────────────────────────────────────────────────────
# The roles a column of the raw can play.
COLUMN_ROLE_PERIOD = "period"
COLUMN_ROLE_MEASURE = "measure"
COLUMN_ROLE_TIME_SERIES_FLAG = "time_series_flag"
COLUMN_ROLE_MANDATORY = "mandatory"
COLUMN_ROLE_TIMEVARYING = "timevarying"
COLUMN_ROLE_EXTRA_RENOVACION = "extra_renovacion"
COLUMN_ROLE_EXTRA_REVALORIZACION = "extra_revalorizacion"
COLUMN_ROLE_BOTH_EXTRAS = "extra_renovacion+extra_revalorizacion"
COLUMN_ROLE_FORMULA_INPUT = "formula_input"
COLUMN_ROLE_IGNORE = "ignore"
COLUMN_ROLE_MATURATION = "maturation"
MATURATION_COLUMN_INFIX = "_month_"     # <state>_month_<k>: the units of the row whose state was set k months before due

# The name of the framework's logger (a named logger: other libraries keep their own).
LOGGER_NAME = "sff"

# How the ids are built: fields joined with "|"; a unit and its revaluation values with "||".
ID_FIELD_SEPARATOR = "|"
COMBINED_ID_SEPARATOR = "||"
NO_REVALUATION_VALUES = "na"      # the revaluation part of a fine row when no extra_revalorizacion is declared
NULL_ID_TEXT = "null"             # how a null value is written inside an id (e.g. an unknown exact discount)

# The status of a check, and the level it is logged at.
STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED = "ok", "WARN", "FAIL", "--"
LOG_LEVEL_BY_STATUS = {STATUS_OK: logging.INFO, STATUS_WARNING: logging.WARNING,
                       STATUS_FAILED: logging.ERROR, STATUS_NOT_EVALUATED: logging.WARNING}

# The values that mean "this flag is on" in a timevarying column or the time_series flag.
ACTIVE_FLAG_VALUES = (1, True, "1")

# A timevarying column rotates towards churn (negative) or towards renewal (positive).
VALID_TIMEVARYING_SIGNS = ("negative", "positive")

# The bucket of a row whose exact discount is null (unknown, not 0 %).
UNKNOWN_DISCOUNT_BUCKET = "sin_dato"

# The three accepted spellings of a month.
MONTH_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}$")              # 2026-09
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")          # 2026-09-01
DAY_FIRST_DATE_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")    # 01/09/2026


def parse_month(month_value) -> pd.Period:
    """A month as a monthly Period, from "2026-09", "2026-09-01", "01/09/2026" (day
    first) or a date-like object. Anything else raises ValueError: an ambiguous date
    never becomes a silent wrong month."""
    if isinstance(month_value, (pd.Period, pd.Timestamp, datetime.date)):
        return pd.Period(month_value, freq="M")

    month_text = str(month_value).strip()
    if MONTH_ONLY_PATTERN.match(month_text):
        return pd.Period(month_text, freq="M")
    if ISO_DATE_PATTERN.match(month_text):
        return pd.Period(month_text[:7], freq="M")
    if DAY_FIRST_DATE_PATTERN.match(month_text):
        day_text, month_number_text, year_text = month_text.split("/")
        return pd.Period(f"{year_text}-{month_number_text}", freq="M")

    raise ValueError(f"'{month_value}' is not a month: use 'YYYY-MM', 'YYYY-MM-DD' or 'DD/MM/YYYY'")


def discount_bucket_labels(discount_values: pd.Series, bucket_edges_pct: list) -> pd.Series:
    """The bucket of every exact discount, labelled like the extract's discount_interval:
    "07 _ 60-70%" for a discount of 0.65. Buckets are [low, high) in percent, the last one
    closed ([90, 100]); the number is the bucket's position, from 01. A null discount is
    unknown: "sin_dato". A value outside the edges is left null (step 01 already stops a
    discount outside [0, 1])."""
    percent_values = (pd.to_numeric(discount_values, errors="coerce") * 100).round(6)
    labels = [f"{position:02d} _ {low:g}-{high:g}%"
              for position, (low, high) in enumerate(zip(bucket_edges_pct[:-1], bucket_edges_pct[1:]), 1)]
    bucket_codes = pd.cut(percent_values, bins=bucket_edges_pct, labels=labels, right=False, include_lowest=True)
    buckets = bucket_codes.astype(object)
    at_the_top = percent_values == bucket_edges_pct[-1]           # 100 % closes the last bucket
    buckets[at_the_top] = labels[-1]
    buckets[percent_values.isna()] = UNKNOWN_DISCOUNT_BUCKET
    return buckets


def is_one_year(rows: pd.DataFrame, configuration) -> pd.Series:
    """The rows of a 1-year licence (term_column = one_year_term_value); every row when no term_column is
    configured. term_column is a mandatory dim (checked when the Config is built): it is in every step."""
    if not configuration.term_column:
        return pd.Series(True, index=rows.index)
    return rows[configuration.term_column].astype(str) == str(configuration.one_year_term_value)


def has_no_negative_state(rows: pd.DataFrame, configuration) -> pd.Series:
    """The rows without any negative timevarying state on (no softcancel, no dormant, no not_installed…): the
    neutral renewals, the ones the contract rule speaks of."""
    neutral = pd.Series(True, index=rows.index)
    for column_name, sign in configuration.structural_timevarying_dims.items():
        if sign == "negative" and column_name in rows.columns:
            neutral &= ~rows[column_name].isin(ACTIVE_FLAG_VALUES)
    return neutral


def negative_states(configuration) -> list:
    """The timevarying states that lower the rate (structural_timevarying_dims with sign "negative")."""
    return [column_name for column_name, sign in configuration.structural_timevarying_dims.items() if sign == "negative"]


def state_combination(frame: pd.DataFrame, states: list) -> pd.Series:
    """The combination of negative states of every row: "softcancel+dormant", … or "neutral" when none is on."""
    labels = pd.Series("", index=frame.index)
    for state in states:
        active = frame[state].isin(ACTIVE_FLAG_VALUES) if state in frame.columns else pd.Series(False, index=frame.index)
        labels = labels.where(~active, labels + np.where(labels == "", "", "+") + state)
    return labels.where(labels != "", "neutral")


ROLE_PURPOSES = [
    (COLUMN_ROLE_PERIOD, "the month of the row"),
    (COLUMN_ROLE_TIME_SERIES_FLAG, "1 = the time_series universe (retail to subscription), projected apart (step 21)"),
    (COLUMN_ROLE_MEASURE, "summed: units and USD due, renewed, reacquired"),
    (COLUMN_ROLE_MANDATORY, "open the forecast series and the uplift cell; collapsed in stage 3 of the ladder"),
    (COLUMN_ROLE_TIMEVARYING, "signals of the customer, summarised by their sign in stage 1"),
    (COLUMN_ROLE_EXTRA_RENOVACION, "open the forecast series; annulled in stage 2"),
    (COLUMN_ROLE_EXTRA_REVALORIZACION, "open the uplift cell (the price), not the rate"),
    (COLUMN_ROLE_BOTH_EXTRAS, "both extras"),
    (COLUMN_ROLE_FORMULA_INPUT, "inputs of a formula (exact discount, SKU), not a dimension"),
    (COLUMN_ROLE_MATURATION, "when a state was set: <state>_month_<k> = its units in the state k months before due (steps 20, 19)"),
    (COLUMN_ROLE_IGNORE, "read and not used")]
LEVELED_ROLE = "niveles generados (01)"
GENERATED_LEVEL_SUFFIX = "_level_1"           # the coarse level the library generates for a leveled dim
LEVEL_TYPES = ("ordinal", "nominal")          # ordinal: only neighbouring values merge · nominal: any two


def roles_overview(columns, configuration) -> pd.DataFrame:
    """Every role of the Config, in a fixed order, with its columns (0 when the role is empty), what it is for,
    and the dimensions that get generated levels (level_1 grouped by the library, level_2 the raw value)."""
    columns = list(columns)
    present = set(columns)
    roles = configuration.column_roles()                 # in the order the Config declares them
    rows = []
    for role, purpose in ROLE_PURPOSES:
        role_columns = [column for column, column_role in roles.items() if column_role == role and column in present]
        if role == COLUMN_ROLE_BOTH_EXTRAS and not role_columns:
            continue
        rows.append({"rol": role, "columnas": len(role_columns), "nombres": ", ".join(role_columns), "para_que": purpose})
    leveled = [f"{column_name} → {column_name}{GENERATED_LEVEL_SUFFIX} ({level_type})"
               + (" · generado" if f"{column_name}{GENERATED_LEVEL_SUFFIX}" in present else "")
               for column_name, level_type in configuration.leveled_dims.items()]
    if leveled:
        rows.append({"rol": LEVELED_ROLE, "columnas": len(leveled), "nombres": "; ".join(leveled),
                     "para_que": "the column keeps its raw value (fine level); _level_1 groups its values by their standardised "
                                 "rate (JSON in levels_path); the ladder collapses the fine level first"})
    return pd.DataFrame(rows)


# ─── the closed-month measures: their names in the core are the framework's, not the extract's ──
# (the extract may call them anything; the core and Power BI always see these)
CORE_RENEWED_PIPELINE_USD = "forecast_to_renew_USD_renewed"
CORE_ISOLATED_PIPELINE_UNITS = "forecast_isolated_to_renew_units"
CORE_ISOLATED_RENEWED_UNITS = "forecast_isolated_renewed_units"
CORE_ISOLATED_RENEWED_USD = "forecast_isolated_renewed_USD"
CORE_ISOLATED_RENEWED_PIPELINE_USD = "forecast_isolated_to_renew_USD_renewed"
CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR = "forecast_isolated_renewed_USD_sq_over_tr"

# the bands of the renewal ratio of an isolated licence (renewed USD / what it was worth): the Config field
# that names the column, its short name (core column suffix), and its [low, high) limits. The limits are
# the ones the extract uses to fill the columns: if they change there, they change here
ISOLATED_RATIO_BANDS = [
    ("isolated_tr_usd_renewed_lt095_col", "lt095", 0.00, 0.95),
    ("isolated_tr_usd_renewed_095_100_col", "095_100", 0.95, 1.00),
    ("isolated_tr_usd_renewed_100_105_col", "100_105", 1.00, 1.05),
    ("isolated_tr_usd_renewed_105_110_col", "105_110", 1.05, 1.10),
    ("isolated_tr_usd_renewed_110_120_col", "110_120", 1.10, 1.20),
    ("isolated_tr_usd_renewed_ge120_col", "ge120", 1.20, float("inf")),
]
CORE_ISOLATED_BAND_PREFIX = "forecast_isolated_to_renew_USD_renewed_"     # + the band's short name


def weighted_ratio_spread(base: pd.Series, renewed: pd.Series, second_moment: pd.Series) -> pd.Series:
    """The weighted standard deviation of the renewal ratio of a group of licences, from three sums:
    Σ what they were worth (the weights), Σ what they renewed for, and Σ renewed² / what they were worth.
    mean = renewed / base; variance = second_moment / base − mean² (clipped at 0 against rounding)."""
    mean = renewed / base.where(base > 0)
    variance = (second_moment / base.where(base > 0) - mean ** 2).clip(lower=0)
    return np.sqrt(variance)


@dataclass(frozen=True)
class ClosedMonthMeasure:
    """One declared closed-month measure: the Config field that names it, its column in the extract, the
    measure it is a part of (step 01 checks it is not above it; None when it is not a part of anything,
    like the second moment), and its two columns in the core."""
    config_field: str
    raw_column: str
    part_of: Optional[str]
    s02_column: str
    core_column: str


def region_columns_of(configuration) -> list:
    """The region levels of the time_series universe, coarse to fine (the first mandatory dim if none is
    declared). Used by step 01 (it separates the universe) and step 21 (it projects it)."""
    return list(configuration.ts_region_columns) or [configuration.business_mandatory_dims[0]]


def join_columns(frame: pd.DataFrame, columns: list) -> pd.Series:
    """The "|"-joined id of every row from several columns, in the given order (the order
    is part of the id); a null value is written "null". Built on arrays, not Series, so a
    duplicated index cannot misalign it."""
    joined_ids = id_text(frame[columns[0]])
    for column_name in columns[1:]:
        joined_ids = joined_ids + ID_FIELD_SEPARATOR + id_text(frame[column_name])
    return pd.Series(joined_ids, index=frame.index)


def id_text(values: pd.Series) -> np.ndarray:
    """The values of one column as the text they take inside an id; null → "null"."""
    texts = values.astype(str).to_numpy(dtype=object)
    texts[values.isna().to_numpy()] = NULL_ID_TEXT
    return texts


def running_in_notebook() -> bool:
    """True inside a Jupyter notebook (where display() renders an HTML table)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:

    # ─── where the raw comes from ───────────────────────────────────────────────────
    raw_extract_sql: Optional[str] = None # [03] the SQL of the extract (or a table name): read_raw may use it, and the
                                          # diagnostics write their reproduction queries on top of it

    # ─── the columns of the raw, by role ────────────────────────────────────────────
    period_col: str = "period"                              # [CFG ORQ 00 01 02 03 04 05 06 08 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27] the month of the row
    pipeline_units_col: str = "total_tr_units"              # [CFG 01 02 03 04 05 06 08 10 11 13 14 15 17 18 19 20 21 22 23 24 27] subscriptions falling due
    pipeline_usd_col: str = "total_tr_usd"                  # [CFG 01 02 03 04 05 B1 14 15 17 18 19 20 21 22 24 26 27] their value at due date
    renewed_units_col: str = "total_renewed_units"          # [CFG 01 02 04 06 08 10 11 13 14 15 17 19 21 22 23 24] subscriptions renewed
    renewed_usd_col: str = "total_renewed_usd"              # [CFG 01 02 14 15 17 19 20 21 22 24 26 27] their value at renewal
    extra_measure_cols: list = field(default_factory=list)  # [CFG 03] other measures (reacquisitions, AUVs): declared, not used yet
    renewed_pipeline_usd_col: Optional[str] = None  # [CFG 01 14 15] the USD that was FALLING DUE of the contracts that RENEWED
                                                    # (built per licence, then summed): the exact base of the uplift.
                                                    # Only known once the month closes, so it gets the closed-month
                                                    # treatment below. None = not in the extract: the uplift falls back
                                                    # to renewed units × the row's average due price (an approximation
                                                    # that is exact only if renewers were worth the row's average)
    # the ISOLATED renewals: the contracts that went through the normal renewal process, isolated from any
    # retention event (in Kamelot: those that never had a softcancel, before, during or after the renewal).
    # The reference to read the revaluation without retention discounts (a price increase shows there). Each
    # one is a part of its counterpart above; all optional; like renewed_pipeline_usd_col, only known once the
    # month closes (null in a closed month = 0, wiped from the current month on, in the core only closed)
    isolated_pipeline_units_col: Optional[str] = None        # [CFG 15] units due of the isolated contracts
    isolated_renewed_units_col: Optional[str] = None         # [CFG 15] their renewed units
    isolated_renewed_usd_col: Optional[str] = None           # [CFG 01 14 15] their renewed USD
    isolated_renewed_pipeline_usd_col: Optional[str] = None  # [CFG 01 14 15] what the isolated renewers were worth before renewing
    # the DISPERSION of the isolated renewals, built per licence with ratio = renewed USD / what it was worth
    # and summed like every measure (so it adds up to any level: series, cell, month, year):
    isolated_renewed_usd_sq_over_tr_col: Optional[str] = None  # [CFG 01 15] Σ renewed USD² / what it was worth: with the isolated
                                                               # base and renewed USD, the exact weighted std deviation
    isolated_tr_usd_renewed_lt095_col: Optional[str] = None    # [CFG] what the isolated renewers were worth, ratio < 0.95
    isolated_tr_usd_renewed_095_100_col: Optional[str] = None  # [CFG] ... ratio in [0.95, 1.00)
    isolated_tr_usd_renewed_100_105_col: Optional[str] = None  # [CFG] ... ratio in [1.00, 1.05)
    isolated_tr_usd_renewed_105_110_col: Optional[str] = None  # [CFG] ... ratio in [1.05, 1.10)
    isolated_tr_usd_renewed_110_120_col: Optional[str] = None  # [CFG] ... ratio in [1.10, 1.20)
    isolated_tr_usd_renewed_ge120_col: Optional[str] = None    # [CFG] ... ratio ≥ 1.20 (the six add up to the isolated base)
    flag_time_series_col: str = "flag_time_series"          # [CFG 01] states the rows of the time_series universe

    business_mandatory_dims: list = field(default_factory=list)       # [CFG 01 07 08 13 14 17 19 20] open the series and the uplift cell
    leveled_dims: dict = field(default_factory=dict)  # [CFG 01] mandatory dims that get a generated coarse level (step 01):
                                                      # {column: "ordinal" | "nominal"}. The column keeps its raw value
                                                      # (the fine level); <column>_level_1 groups its values by their
                                                      # standardised renewal rate and is added to the mandatory dims
                                                      # right after it, when the Config is built
    levels_path: Optional[str] = None                 # [01] the JSON of the generated groups (None: <output_folder>/sff_levels.json);
                                                      # a later run reuses it; delete it to regenerate
    level_merge_max_pp: Optional[float] = None        # [01] two neighbouring values merge while their standardised rates differ
                                                      # by at most this (pp). None: the binomial noise of a series at the
                                                      # support floor, 100·√(p(1−p)/support_floor), p = the training rate
    structural_timevarying_dims: dict = field(default_factory=dict)   # [CFG 01 06 08 17 18 19 27] column → "negative" | "positive"
    distribute_marks: bool = True   # [17] the box of stage 17 matures the negative states still to come (the forecast
                                    # carries them). False = the identity box: the forecast with today's states
    maturation_dims: list = field(default_factory=list)  # [CFG 01 17 19 27] the states whose extract says WHEN they were set:
                                                      # every one carries <state>_month_0 … <state>_month_<maturation_months>
                                                      # (the units of the row in the state k months before due; 0 = the due month,
                                                      # the last one = that many months or more). Step 00 fails without them.
                                                      # Their states still to come follow the maturation curve (step 20) and
                                                      # the honest exam sees them as they were h months before (step 19)
    maturation_months: int = 12                       # [CFG 17] the last k of the maturation columns
    extra_renovacion: list = field(default_factory=list)              # [CFG 01 07 08] enter the rate series only
    extra_revalorizacion: list = field(default_factory=list)          # [CFG 01 18 20 24] enter the uplift cell only

    uplift_mandatory_dims: Optional[list] = None      # [CFG 14 20] mandatory dims of the uplift cell; None or [] = all of them
    # the discount: one column, both sides (see the module header). None = no discount in the extract
    discount_value_column: Optional[str] = "discount"          # [CFG 01 03 14 15 16 18 20 24 27] the exact discount, a share (0.25 = 25 %)
    discount_bucket_edges: list = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # [CFG 03 18 27] in %
    discount_bucket_column: str = "tramo_descuento"            # [CFG 03 B1 14 15 18 20 24 27] the derived bucket (a dimension of the uplift cell)
    sku_column: Optional[str] = None              # [CFG 15] the SKU (formula input for the price scenarios)
    ignore_cols: list = field(default_factory=list)   # [CFG] read by nobody; may be absent from the raw

    # ─── the calendar (the role of every month is generated from it) ────────────────
    current_month: Optional[str] = None   # [CFG 01] first month of the future; no default on purpose. If the last months
                                          # are not mature yet (late renewals still arriving), set current_month to the
                                          # first immature one: it becomes proyeccion, is predicted, and its partial
                                          # renewals are wiped in step 02. "Closed" always means: before current_month
    test_months: int = 3                  # [CFG] closed months before it that only evaluate

    # ─── statistical parameters ────────────────────────────────────────────────────
    z: float = 1.645                      # [06 27] 90 % two-sided: every band and every binomial error uses it
    support_floor: float = 30.0           # [01 06 08 09 10 11 23 27] contracts in a typical month below which a series' own rate
                                          # moves ±15 pp by chance (90 %, p = 0.5): it will borrow support

    # ─── the ladder (steps 08 and 09) ───
    own_rate_floor: float = 271.0         # [08 09 27] contracts in a typical month to predict ALONE: ±5 pp at 90 % (p = 0.5).
                                          # 30 says who may speak; 271 who may speak alone
    collapse_passes: int = 2              # [08] stage 3 merges at most this many mandatory dims (collapse order); the passes
                                          # after them are only searched for a credibility reference (stage 4)
    k_cred: float = 60.0                  # [09 23] Bühlmann k when a relative has too few siblings to estimate it: a series
                                          # with n = 30 keeps 33 % of its own rate ("twice the floor to be believed half")
    own_level_min_history_months: int = 12    # [09] a full year to be level A (a seasonal series has seen every season)
    signed_ladder_max_loss: float = 0.05  # [08] a signed series may collapse mandatory dims (sign kept) while the cumulative
                                          # R² lost (step 07) stays ≤ this; 0 = its ladder ends at the cell × sign
    # ─── the dynamics of the rate (step 11): descriptive, it does NOT restrict any technique ───
    dynamics_min_months: int = 24         # [11] months a pool needs to measure its month effect (two of each month)
    dynamics_significance: float = 0.05   # [11] a month effect or a trend is declared when its p-value is below this
    # ─── the uplift (steps 14 and 16) ───
    uplift_floor: float = 30.0            # [14 27] renewers a cell needs to use its own uplift; below, its parent's (tramo kept)
    uplift_cap: float = 3.0               # [14] an uplift is clipped to [0.01, 3]: a renewer never pays 3× what fell due
    uplift_bootstrap_samples: int = 200   # [14] resamples of the renewer rows for the uplift band (p5-p95)
    random_seed: int = 42                 # [14] the bootstrap is reproducible
    # ─── the future pipeline (step 18) and the forecast (step 20) ───
    # the SIMULATION WINDOW: from the current month (included) to simulation_end. What happens in it
    # (renewals of 1-year licences, acquisitions, time_series conversions) falls due 12 months later
    simulation_end: Optional[str] = None          # [CFG] the last month simulated; None = December of the current month's year
    term_column: Optional[str] = None             # [CFG 02 18 20] the column with the term of the licence; None = every licence is of 1 year
    one_year_term_value: str = "1 year"           # [CFG 18] the value of term_column that means a 1-year licence (only those re-enter)
    renewal_reentry_discount: float = 0.0         # [CFG 18] a renewal falls due again at the price it renewed: no discount
    acquisition_column: Optional[str] = None      # [15 18 20] the column that says a row is an acquisition; None = no acquisition simulated
    acquisition_values: list = field(default_factory=list)      # [CFG 15 18] its values that are acquisition: each one is simulated apart
    renewed_acquisition_value: Any = None   # [CFG 18] the value of acquisition_column once an acquisition has renewed: the
                                            # renewal it creates (a proyectada row of step 18) is no longer an
                                            # acquisition. None = the proyectada row keeps the value of the row due
    dims_after_renewal: dict = field(default_factory=dict)   # [CFG 18] other dims a renewal changes: column → the value of the
                                                             # renewal it creates (prev_OperationGroup → the renewal
                                                             # operation). The timevarying states of a renewal are always
                                                             # neutral: nothing is known yet of its dormant, softcancel…
                                                                # (they have different proportions)
    acquisition_discount: float = 0.4             # [CFG 18 27] an acquisition is sold with this discount (it renews without it)
    acquisition_level_window_months: int = 3      # [18] acquisition level: last 3 months sold vs the same months a year before
    acquisition_auv_window_months: int = 12       # [18] acquisition value per unit: Σ value / Σ units of the last 12 closed months
    acquisition_level_dims: list = field(default_factory=list)   # [18] the level of the acquisitions is measured per these dims
                                                  # (e.g. ["tr_regional_level_1"]); empty = per acquisition value over the
                                                  # whole portfolio. Never per fine group: a ratio of small numbers explodes
    acquisition_level_min_units: float = 1000.0   # [18] a level key needs this many units sold a year before; below, the level
                                                  # of its acquisition value over the portfolio
    acquisition_ratio_low: float = 0.5            # [18] simulated acquisitions over the same month a year before outside
    acquisition_ratio_high: float = 2.0           # [18] [low, high]: WARN (the simulation is not the shape × a plausible level)
    acquisition_exam_origins: int = 6             # [18 27] the exam of the acquisitions: the last 6 closed months are origins; at
                                                  # each one the box is rebuilt with the months before it only
    # the confidence of a future row (step 20): how sure the forecast is of its rate
    confidence_high_band_pp: float = 5.0      # [20] high: its rate band within ±5 pp (the precision of own_rate_floor)…
    confidence_high_exam_pp: float = 5.0      # [20] …its composition judged by the backtest, with an exam error ≤ 5 pp
    confidence_medium_band_pp: float = 10.0   # [20] medium: judged, and a band within ±10 pp; low: the rest (or not judged)
    rate_fallback_with_states: bool = False   # [13] a series without composition takes the level of its mandatory cell AND its
                                              # customer states (True) or of its mandatory cell (False). The exam of 25-oct
                                              # (like-real synthetic): with the states, the error of the total at 6 months
                                              # rises from 1.38 % to 1.53 % (honest: 1.53 % → 1.65 %)
    apply_credibility_shift: bool = True  # [13 27] a group below own_rate_floor moves its predicted rate toward its credibility
                                          # reference by (1 − z) of their difference of level (logit scale); False = the
                                          # group's prediction as is

    # ─── the baseline (step 19): the calculation anyone could make, THE reference of every gain ───
    baseline_months: int = 3              # [19 27] the rate of its mandatory cell over the last N closed months known at the
                                          # origin (the portfolio's when the cell has nothing), × what falls due
                                          # (every mandatory dim) or columns joined with "+" (e.g. "region+product")
    # ─── the time_series universe: retail to subscription (step 21) ───
    ts_region_columns: list = field(default_factory=list)   # [CFG] the region levels of a time_series row, coarse to fine;
                                              # [] = the first mandatory dim. The projection is made at the FINEST level and
                                              # the renewal rate climbs fine → coarse → global; the same columns must be in
                                              # the normal pipeline (its rate is read there)
    ts_level_window_months: int = 3           # [21] the level: the last 3 closed months of this year vs the same months of last
                                              # year (recent enough to follow the campaigns, long enough not to be one month)
    ts_auv_window_months: int = 12            # [21] the value per unit: Σ value / Σ units of the last 12 closed months (a year)
    ts_rate_window_months: int = 12           # [21] the renewal rate of the region in the normal pipeline: last 12 closed months
    ts_acquisition_discount: float = 0.4      # [CFG 21] retail-to-subscription offers: acquired at 40 % off, renewed at 100 %
    # ─── the backtest of the rate (step 12) ───
    default_technique: str = "T3_ma3"     # [12 13 19 27] the technique to beat: the mean of the last 3 monthly rates (in logit)
                                          # of the composition; another one is chosen only if it beats it in the selection months
    backtest_selection_months: int = 6    # [12 27] closed months BEFORE the exam used as targets to CHOOSE the technique
    backtest_horizons: list = field(default_factory=lambda: [1, 6])   # [12 17 19 27] months ahead judged: next month, and six months
    horizon_bands: dict = field(default_factory=lambda: {"corto": [1, 1], "medio_largo": [2, 6]})   # [12 13 19 20 23 24 27] a horizon between
                                          # two judged ones takes the band of the judged one above it (the wider)
    default_margin_by_band: dict = field(default_factory=lambda: {"corto": 0.10, "medio_largo": 0.0})  # [12] how much a
                                          # technique must beat the default technique (in |error| / binomial error) to replace it
    backtest_min_history_months: int = 12 # [12] months a technique must see before predicting (every technique can compete)
    backtest_min_predictions: int = 3     # [12] predictions a technique needs in a band to be chosen
    band_low_quantile: float = 0.05       # [12] the error band: 5 % and 95 % quantiles of the normalised error (90 %)
    band_high_quantile: float = 0.95   # [12]

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE PARAMETERS OF THE STEPS: every number a step decides with, here and nowhere else.
    # Each one says the step it belongs to; the run prints and writes the whole table
    # (sff_parametros, step 25), with the steps that read each parameter.
    # ═══════════════════════════════════════════════════════════════════════════════
    # Not here, on purpose: DEFINITIONS, not choices (changing them would make the code wrong, not tuned):
    # 100 percentage points, 12 months a year, the largest binomial variance (0.25), the stage numbers of the
    # ladder, the windows a technique is named after (T3_ma3 = 3 months) and their minimum history, the bands of the isolated ratio (ISOLATED_RATIO_BANDS: they are
    # the extract's columns).
    # ─── checks of several steps ───
    money_tolerance: float = 0.01                 # [01 02 04 08 20 21 24 26] two money figures "match" within this (USD)
    example_rows_shown: int = 3                   # [CFG 03] example rows shown under a failed check (every step's checks)
    measure_relative_tolerance: float = 1e-6      # [01] relative margin of the checks between a measure and its whole
    money_relative_tolerance: float = 1e-9        # [04] relative margin of the money conserved from rows to units
    units_tolerance: float = 1e-6                 # [01 19 22 23] two unit counts "match" within this
    rate_tolerance: float = 1e-9                  # [13 23] two rates "match" within this
    support_tolerance: float = 1e-9               # [08 10] two supports "match" within this
    rounding_tolerance: float = 1e-9              # [09] margin for rounding when comparing z and supports
    # ─── step 02 · the calendar ───
    min_training_months: int = 12                 # [02] fewer training months: the calendar is not valid
    one_year_term_months: int = 12                # [02 18] months between the sale and the due date of a 1-year licence
    # ─── step 06 · the rate series ───
    no_evidence_rate: float = 0.5                 # [06] the rate the Wilson error is computed at for a series with no own rate
    largest_series_shown: int = 5                 # [06 10] the biggest series listed on screen
    # ─── step 07 · the dimensions ───
    dimension_min_series: int = 3                 # [07] fewer series in the base: every figure is 0 (no evidence, declared)
    dimension_figure_decimals: int = 4            # [07] decimals of the explained-variance figures
    # ─── step 09 · the ladder ───
    min_siblings_for_k: int = 3                   # [09] fewer siblings: the k of the Config (k_cred) instead of the measured one
    homogeneous_pool_factor: float = 10.0         # [09] a reference with no between variance gets k = this × k_cred
    level_clip: float = 1e-6                      # [13 19] a level of exactly 0 or 1 is clipped by this before its logit
    min_between_variance: float = 1e-6            # [09] below this between variance a reference is homogeneous
    # ─── step 11 · the dynamics ───
    month_significance_sigmas: float = 2.0        # [11] a month is 'high' or 'low' beyond this many errors
    phi_with_engine: float = 1.5                  # [11] summary: φ above this = clearly more than noise
    dynamics_min_shared_months: int = 3           # [11] fewer calendar months in both halves: no stability verdict
    # ─── the techniques (steps 12, 13, 19 and 23) ───
    logit_clip: float = 0.001                     # [12 13 19 23] every rate is clipped to [this, 1 − this] before its logit:
                                                  # the floor and the ceiling of any predicted rate
    ewma_halflife_months: float = 3.0             # [12 13 19 23] T4_ewma: the weights halve every this many months
    ses_alpha: float = 0.3                        # [12 13 19 23] T9_ses (and Theta): the weight of the newest month
    holt_alpha: float = 0.3                       # [12 13 19 23] T10_holt_damped: level
    holt_beta: float = 0.1                        # [12 13 19 23] T10_holt_damped: trend
    holt_damping: float = 0.9                     # [12 13 19 23] φ of every damped trend (Holt, linear, Theta, Holt-Winters)
    theta_weight: float = 0.5                     # [12 13 19 23] T12_theta: share of the damped linear trend
    temporal_credibility_k: float = 6.0           # [12 13 19 23] T14_temporal_cred: k of z = n / (n + k)
    recent_window_months: int = 6                 # [12 13 19 23] T14_temporal_cred: the recent window
    holt_winters_alpha: float = 0.3               # [12 13 19 23] T11_holt_winters: level
    holt_winters_beta: float = 0.05               # [12 13 19 23] T11_holt_winters: trend
    holt_winters_gamma: float = 0.2               # [12 13 19 23] T11_holt_winters: season
    # ─── step 12 · the backtest ───
    min_exam_coverage: float = 0.80               # [12] the exam must cover at least this share of what falls due
    # ─── steps 14, 16 and 20 · the uplift ───
    min_uplift: float = 0.01                      # [14] the lowest uplift accepted
    uplift_band_low: float = 0.05                 # [14] the uplift band: this percentile of the bootstrap...
    uplift_band_high: float = 0.95                # [14] ...to this one
    contract_discount_cap: float = 0.99           # [14] the discount is capped here before 1/(1−d) (in contract_rule, which 16 and 17 call)
    # ─── step 15 · the revaluation report (the diagnostics of the price) ───
    revaluation_window_months: int = 12           # [15] the distribution is measured over the last N closed months
    revaluation_monthly_months: int = 24          # [15] the monthly path shows the last N closed months
    revaluation_units_floor: float = 271.0        # [15] a series enters when it averages this many units due a month...
    revaluation_top_series: int = 50              # [15] ...or is among this many biggest by USD due
    revaluation_uplift_buckets: list = field(default_factory=lambda: [0.0, 0.80, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05,   # [15]
                                                                      1.10, 1.20, 1.50, float("inf")])   # histogram
    histogram_width: int = 40                     # [15] characters of the longest histogram bar
    # ─── steps 14, 15, 16, 20 and 25 · the price increases (price_increases.py) ───
    price_step_threshold: float = 0.03            # [14 15 16 20 25] a month this far above the average of its previous 12...
    price_persistence_months: int = 3             # [14 15 16 20 25] ...this many months in a row is an increase (not noise)
    price_cycle_months: int = 12                  # [14 15 16 20 25] the effect of an increase lasts one renewal cycle
    price_min_reference_months: int = 6           # [14 15 16 20 25] fewer previous months: no reference yet, no flag
    price_group_dims: list = field(default_factory=list)   # [14 15 16 20 25] the dims a tariff changes by (region levels,
                                                           # product): every group is watched on its own. Empty = the portfolio
    price_group_min_renewed_units: float = 100.0  # [14 15 16 20 25] a group's month with fewer renewals is left out of the detection
    # ─── steps 14, 16 and 20 · the window of the statistical uplift ───
    uplift_window_months: int = 12                # [14 15 16] the statistical uplift is estimated over the last N closed months
                                                  # (12 = one price cycle: every yearly increase enters once); 0 = all the history
    uplift_without_increases: bool = True         # [14 15 16 20] the months of the window inside the cycle of a detected price
                                                  # increase are deflated by it (the uplift of the cell is the one without
                                                  # increases) and the increase is applied again to the future months still
                                                  # inside its cycle; after the cycle its effect is lost
    price_increases_shown: int = 15               # [15] increases (and unconfirmed steps) with a VALORACIÓN line on screen
    price_series_shown: int = 5                   # [15] forecast series with the most money in a group with an increase,
                                                  # shown month by month so the step can be seen
    price_series_months_shown: int = 36           # [15] the closed months shown for each of them
    # ─── step 17 · the maturation of the customer states ───
    maturation_min_units: float = 100.0           # [17] a cell (or its calendar month) needs this many units to measure itself
    maturation_history_months: int = 12           # [17 27] the final states are measured over the last N closed months
    mark_excess_warning_pp: float = 2.0           # [17 20] a future month whose state (dormant…) is TODAY this many points above
                                                  # its final share in the history: WARN (a state only grows, so the extract
                                                  # shows too much in the state, e.g. a fresh cohort set to dormant before its first use)
    # ─── step 26 · the validation ───
    max_rate_jump_pp: float = 15.0                # [26] a future year whose rate moves more than this from the last closed year is flagged
    max_exam_total_error: float = 0.10            # [26] the exam's total error above this share is flagged
    # ─── step 19 · the series exam ───
    exam_min_months_to_predict: int = 3           # [13 19] fewer own months: the raw method takes the series' own level
    # ─── step 21 · the time series ───
    top_regions_shown: int = 10                   # [21] regions listed on screen
    # ─── step 23 · the audit ───
    audit_trend_min_months: int = 12              # [23] a trend is measured with at least this many months
    audit_seasonality_min_months: int = 24        # [23] a month effect needs at least this many months
    # ─── step 27 · the report, and step 24 · the core ───
    promise_pp: float = 5.0                       # [27] the promise to the business: a rate known within ± this (90 %)
    report_top_rows: int = 10                     # [27] rows of the report's top tables
    core_months_shown: int = 6                    # [24] months shown on screen by the core

    # ─── where the tables are written ───────────────────────────────────────────────
    sql_engine: Optional[object] = None   # [CFG ORQ B1] a SQLAlchemy engine; None → CSV files in output_folder
    sql_schema: Optional[str] = None      # [CFG ORQ B1] "dbo" in SQL Server; None in SQLite
    table_prefix: str = "sff_"            # [CFG ORQ B1] every table the framework writes starts with it
    output_folder: str = "salida"         # [CFG ORQ 01 B1 27] where the CSV tables go when there is no engine

    # ─── the log (configured once, when the Config is built) ────────────────────────
    log_level: str = "INFO"               # [CFG] DEBUG · INFO · WARNING · ERROR
    log_colors: bool = True               # [CFG] colours on the console (off in tests)
    log_file: Optional[str] = None        # [CFG] also write the log to this file, without colours

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE RAW SOURCE
    # ═══════════════════════════════════════════════════════════════════════════════

    def read_raw(self) -> pd.DataFrame:
        """The raw extract. Overridden by the Config of main.py (SQL, CSV, synthetic)."""
        raise NotImplementedError("override read_raw() in your Config subclass (main.py)")

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE COLUMN CONTRACT
    # ═══════════════════════════════════════════════════════════════════════════════

    def __post_init__(self) -> None:
        """The logger is configured, and a misdeclared Config stops here, before any data is read."""
        # [0] the log: one configuration for the whole run; the count of the checks of every
        #     step is kept for the report
        LoggerManager(self.log_level, use_colors=self.log_colors, log_file=self.log_file).get_logger_configured(LOGGER_NAME)
        self.check_history = []

        # [1] every timevarying column carries a valid sign
        invalid_signs = {column_name: sign for column_name, sign in self.structural_timevarying_dims.items()
                         if sign not in VALID_TIMEVARYING_SIGNS}
        if invalid_signs:
            raise ValueError(f"timevarying signs must be one of {VALID_TIMEVARYING_SIGNS}: {invalid_signs}")

        # [2] the states with maturation columns are declared timevarying dims; the leveled dims: mandatory, with a
        #     valid type; their generated level is a mandatory dim from here on (step 01 creates it before any step
        #     reads the dims)
        not_timevarying = [state for state in self.maturation_dims if state not in self.structural_timevarying_dims]
        if not_timevarying:
            raise ValueError(f"maturation_dims must be structural_timevarying_dims: {not_timevarying}")
        if int(self.maturation_months) < 1:
            raise ValueError(f"maturation_months must be at least 1 (found {self.maturation_months})")
        not_mandatory = [column_name for column_name in self.leveled_dims if column_name not in self.business_mandatory_dims]
        if not_mandatory:
            raise ValueError(f"leveled_dims must be mandatory dims: {not_mandatory}")
        invalid_types = {column_name: level_type for column_name, level_type in self.leveled_dims.items()
                         if level_type not in LEVEL_TYPES}
        if invalid_types:
            raise ValueError(f"leveled_dims types must be one of {LEVEL_TYPES}: {invalid_types}")
        expanded = []
        for column_name in self.business_mandatory_dims:
            if column_name in self.generated_columns:
                continue                                   # re-inserted right after its column
            expanded.append(column_name)
            if column_name in self.leveled_dims:
                expanded.append(f"{column_name}{GENERATED_LEVEL_SUFFIX}")
        self.business_mandatory_dims = expanded

        # [3] the term of a licence is a declared mandatory dim (it exists in every step)
        if self.term_column and self.term_column not in self.business_mandatory_dims:
            raise ValueError(f"term_column '{self.term_column}' must be one of the mandatory dims")

        # [4] no column declared with two roles (built and checked by column_roles), and
        #     the uplift cell takes its mandatory dims from the declared ones
        self.column_roles()
        unknown_uplift_dims = [column_name for column_name in (self.uplift_mandatory_dims or [])
                               if column_name not in self.business_mandatory_dims]
        if unknown_uplift_dims:
            raise ValueError(f"uplift_mandatory_dims must be mandatory dims: {unknown_uplift_dims}")
        if self.renewed_acquisition_value is not None and self.renewed_acquisition_value in self.acquisition_values:
            raise ValueError(f"renewed_acquisition_value ({self.renewed_acquisition_value}) cannot be an acquisition value: "
                             f"it is what an acquisition becomes once it renews")
        unknown_after_renewal = [column_name for column_name in self.dims_after_renewal
                                 if column_name not in self.rate_series_columns + list(self.extra_revalorizacion)]
        if unknown_after_renewal:
            raise ValueError(f"dims_after_renewal must be dims of the rate series or the uplift: {unknown_after_renewal}")
        # the dispersion of the isolated renewals is only readable against their base and renewed USD
        dispersion_declared = [field_name for field_name in ["isolated_renewed_usd_sq_over_tr_col"]
                               + [band[0] for band in ISOLATED_RATIO_BANDS] if getattr(self, field_name)]
        if dispersion_declared and not (self.isolated_renewed_pipeline_usd_col and self.isolated_renewed_usd_col):
            raise ValueError(f"{dispersion_declared} need isolated_renewed_pipeline_usd_col and isolated_renewed_usd_col "
                             f"(the base and the renewed USD the dispersion is measured against)")

        # [5] the calendar parameters are well formed
        if self.current_month is not None:
            parse_month(self.current_month)
        edges = list(self.discount_bucket_edges)
        if edges != sorted(set(edges)) or edges[0] != 0 or edges[-1] != 100:
            raise ValueError(f"discount_bucket_edges must go up strictly from 0 to 100 (in %): {edges}")
        for discount_name in ("acquisition_discount", "renewal_reentry_discount"):
            if not (0 <= float(getattr(self, discount_name)) < 1):
                raise ValueError(f"{discount_name} must be in [0, 1) (found {getattr(self, discount_name)})")
        if not (0 <= float(self.ts_acquisition_discount) < 1):
            raise ValueError(f"ts_acquisition_discount must be in [0, 1) (found {self.ts_acquisition_discount})")
        if int(self.test_months) < 1:
            raise ValueError(f"test_months must be at least 1 (found {self.test_months}): without an exam nothing is evaluated")

    @property
    def logger(self) -> logging.Logger:
        """The framework's logger: every step logs through it."""
        return logging.getLogger(LOGGER_NAME)

    def log_step_start(self, step_label: str, step_name: str, step_purpose: str,
                       step_actions: list, step_output: str) -> None:
        """The opening of a step, in blue: its title, its purpose, the actions it will
        take (numbered, as log_action will report them) and what it returns or writes."""
        self.logger.log(DOC_LEVEL, f"[{step_label}] ═══ STEP {step_label} · {step_name} ═══")
        self.logger.log(DOC_LEVEL, f"[{step_label}] purpose: {step_purpose}")
        self.logger.log(DOC_LEVEL, f"[{step_label}] actions:")
        for action_number, action_text in enumerate(step_actions, 1):
            self.logger.log(DOC_LEVEL, f"[{step_label}]    {action_number}. {action_text}")
        self.logger.log(DOC_LEVEL, f"[{step_label}] output: {step_output}")

    def log_action(self, step_label: str, action_number: int, action_text: str) -> None:
        """One action of a step, when it is done, with its result (blue)."""
        self.logger.log(DOC_LEVEL, f"[{step_label}] ▸ {action_number}. {action_text}")

    def log_check(self, step_label: str, check_log: list, description: str, passed: bool,
                  failure_detail: str = "", context: str = "", blocking: bool = True,
                  examples: Optional[pd.DataFrame] = None) -> bool:
        """One check of a step, numbered, logged at the level of its status (ok green, WARN
        yellow, FAIL red); its status is kept in `check_log` for the summary. `context` is
        shown when it passes, `failure_detail` when it fails; `examples` (a few concrete
        rows) are shown under the line as a pandas table."""
        if passed:
            status, detail = STATUS_OK, context
        else:
            status, detail = (STATUS_FAILED if blocking else STATUS_WARNING), failure_detail
        check_log.append((status, description, detail))

        has_examples = examples is not None and len(examples) > 0
        detail_text = f"   {detail}" if detail else ""
        examples_text = f"   · {min(len(examples), self.example_rows_shown)} example rows below" if has_examples else ""
        self.logger.log(LOG_LEVEL_BY_STATUS[status],
                        f"[{step_label}]  {len(check_log):>2}. {status:<4}  {description}{detail_text}{examples_text}")
        if has_examples:
            self.show_table(examples.head(self.example_rows_shown))
        return passed

    def log_not_evaluated(self, step_label: str, check_log: list, description: str, reason: str) -> None:
        """A check that cannot be made because an earlier one failed (yellow)."""
        check_log.append((STATUS_NOT_EVALUATED, description, reason))
        self.logger.warning(f"[{step_label}]  {len(check_log):>2}. {STATUS_NOT_EVALUATED:<4}  {description}   {reason}")

    def log_check_summary(self, step_label: str, step_name: str, check_log: list) -> None:
        """The count of the checks of a step, at the level of the worst one; stops with a
        ValueError naming the failed checks if any FAILED."""
        counts = {status: sum(1 for logged in check_log if logged[0] == status)
                  for status in (STATUS_OK, STATUS_WARNING, STATUS_FAILED, STATUS_NOT_EVALUATED)}
        self.check_history.append({"paso": step_label, "nombre": step_name, "comprobaciones": len(check_log),
                                   "ok": counts[STATUS_OK], "avisos": counts[STATUS_WARNING],
                                   "fallos": counts[STATUS_FAILED], "no_evaluadas": counts[STATUS_NOT_EVALUATED],
                                   "avisos_detalle": " · ".join(description for status, description, _ in check_log
                                                                if status in (STATUS_WARNING, STATUS_FAILED))})
        if counts[STATUS_FAILED]:
            summary_level = logging.ERROR
        elif counts[STATUS_WARNING] or counts[STATUS_NOT_EVALUATED]:
            summary_level = logging.WARNING
        else:
            summary_level = logging.INFO
        self.logger.log(summary_level, f"[{step_label}] {len(check_log)} checks: {counts[STATUS_OK]} ok · "
                                       f"{counts[STATUS_WARNING]} warnings · {counts[STATUS_FAILED]} failed · "
                                       f"{counts[STATUS_NOT_EVALUATED]} not evaluated")

        failed_checks = [logged for logged in check_log if logged[0] == STATUS_FAILED]
        if failed_checks:
            numbered_failures = "\n".join(f"  {number}. {description} — {detail}"
                                          for number, (status, description, detail) in enumerate(failed_checks, 1))
            raise ValueError(f"step {step_label} · {step_name}: {len(failed_checks)} checks failed\n{numbered_failures}")

    def write_table(self, step_label: str, check_log: list, frame: pd.DataFrame, table_name: str) -> None:
        """Write a table, read it back and log a check on its row count.

        The physical name is table_prefix + table_name. With an engine the table is
        replaced in SQL (schema sql_schema); without one it is a CSV in output_folder.
        Monthly Periods are written as text ("2026-09"). The check fails if the rows read
        back are not the rows written. If a blocking check of the step has already failed,
        nothing is written and the write is logged as not evaluated.
        """
        physical_name = f"{self.table_prefix}{table_name}"

        # a table built from data that already failed a blocking check is not written:
        # it would replace a good table with a wrong one, and it costs time for nothing
        if any(logged[0] == STATUS_FAILED for logged in check_log):
            self.log_not_evaluated(step_label, check_log, f"table {physical_name} written and read back",
                                   "not written: an earlier check of this step failed")
            return

        persisted_frame = frame.copy()
        for column_name in persisted_frame.columns:
            if isinstance(persisted_frame[column_name].dtype, pd.PeriodDtype):
                persisted_frame[column_name] = persisted_frame[column_name].astype(str)

        write_start_time = time.time()
        if self.sql_engine is not None:
            # chunks of 10,000 rows; with an engine created with fast_executemany=True (see main.py),
            # pyodbc sends each chunk in one round trip instead of one INSERT per row
            persisted_frame.to_sql(physical_name, self.sql_engine, schema=self.sql_schema,
                                   if_exists="replace", index=False, chunksize=10_000)
            qualified_name = f"{self.sql_schema}.{physical_name}" if self.sql_schema else physical_name
            rows_read_back = int(pd.read_sql(f"SELECT COUNT(*) AS row_count FROM {qualified_name}",
                                             self.sql_engine)["row_count"].iloc[0])
            destination = f"SQL {qualified_name}"
        else:
            os.makedirs(self.output_folder, exist_ok=True)
            destination = os.path.join(self.output_folder, f"{physical_name}.csv")
            persisted_frame.to_csv(destination, index=False)
            rows_read_back = len(pd.read_csv(destination))
        elapsed_seconds = time.time() - write_start_time

        self.log_check(step_label, check_log, f"table {physical_name} written and read back",
                       rows_read_back == len(persisted_frame),
                       failure_detail=f"{len(persisted_frame):,} rows written but {rows_read_back:,} read back from {destination}",
                       context=f"{rows_read_back:,} rows × {len(persisted_frame.columns)} columns → "
                               f"{destination} ({elapsed_seconds:.1f}s)")

    def show_query(self, step_label: str, what: str, sql: str) -> None:
        """The SQL that reproduces an aggregated report from the tables the run writes (report_queries.py)."""
        self.logger.doc(f"[{step_label}] {what}: the SQL that reproduces it from the tables of the run:\n{sql}")

    def show_table(self, table: pd.DataFrame) -> None:
        """Concrete rows (examples) as a pandas table, never through the logger: rendered
        with display() in a notebook, printed aligned (to_string) in a terminal. The log
        line that introduces the table says how many rows follow."""
        sys.stdout.flush()      # the log lines already written come before the table
        shown_table = table.reset_index(drop=True)
        if running_in_notebook():
            from IPython.display import display
            display(shown_table)
        else:
            print(shown_table.to_string())

    @property
    def generated_columns(self) -> list:
        """The columns the library adds to the raw (step 01): the coarse level of every leveled dim."""
        return [f"{column_name}{GENERATED_LEVEL_SUFFIX}" for column_name in self.leveled_dims]

    @property
    def extract_mandatory_dims(self) -> list:
        """The mandatory dims the extract brings (the steps that validate the extract, 00 and 01, read these;
        from step 01 on, business_mandatory_dims, with the generated levels)."""
        return [column_name for column_name in self.business_mandatory_dims if column_name not in self.generated_columns]

    @property
    def closed_month_measures(self) -> list:
        """Every DECLARED measure that is only known once the month closes (null closed = 0, wiped from the
        current month on, carried to the core only in the closed months), each with the measure it is a part
        of and its columns in the core. Undeclared ones (None) are left out."""
        exact_base_or_due = self.renewed_pipeline_usd_col or self.pipeline_usd_col
        candidates = [
            ("renewed_pipeline_usd_col", self.renewed_pipeline_usd_col, self.pipeline_usd_col,
             "s02_renewed_pipeline_usd", CORE_RENEWED_PIPELINE_USD),
            ("isolated_pipeline_units_col", self.isolated_pipeline_units_col, self.pipeline_units_col,
             "s02_isolated_pipeline_units", CORE_ISOLATED_PIPELINE_UNITS),
            ("isolated_renewed_units_col", self.isolated_renewed_units_col, self.renewed_units_col,
             "s02_isolated_renewed_units", CORE_ISOLATED_RENEWED_UNITS),
            ("isolated_renewed_usd_col", self.isolated_renewed_usd_col, self.renewed_usd_col,
             "s02_isolated_renewed_usd", CORE_ISOLATED_RENEWED_USD),
            ("isolated_renewed_pipeline_usd_col", self.isolated_renewed_pipeline_usd_col, exact_base_or_due,
             "s02_isolated_renewed_pipeline_usd", CORE_ISOLATED_RENEWED_PIPELINE_USD),
            ("isolated_renewed_usd_sq_over_tr_col", self.isolated_renewed_usd_sq_over_tr_col, None,
             "s02_isolated_renewed_usd_sq_over_tr", CORE_ISOLATED_RENEWED_USD_SQ_OVER_TR),
        ]
        for field_name, short_name, _, _ in ISOLATED_RATIO_BANDS:
            candidates.append((field_name, getattr(self, field_name), self.isolated_renewed_pipeline_usd_col,
                               f"s02_isolated_band_{short_name}", CORE_ISOLATED_BAND_PREFIX + short_name))
        return [ClosedMonthMeasure(config_field, raw_column, part_of, s02_column, core_column)
                for config_field, raw_column, part_of, s02_column, core_column in candidates if raw_column]

    @property
    def isolated_ratio_bands(self) -> list:
        """The DECLARED bands of the isolated renewal ratio: (extract column, core column, low, high, label)."""
        declared = []
        for field_name, short_name, low, high in ISOLATED_RATIO_BANDS:
            if getattr(self, field_name):
                label = f"< {high:.2f}" if low == 0 else f"≥ {low:.2f}" if high == float("inf") else f"{low:.2f} – {high:.2f}"
                declared.append((getattr(self, field_name), CORE_ISOLATED_BAND_PREFIX + short_name, low, high, label))
        return declared

    @property
    def closed_month_measure_cols(self) -> list:
        """The extract columns of the declared closed-month measures."""
        return [measure.raw_column for measure in self.closed_month_measures]

    @property
    def rate_series_columns(self) -> list:
        """The columns that define ONE renewal-rate series: mandatory + timevarying +
        extra_renovacion, in this order (the field order of fs_id and fu_id)."""
        return (self.business_mandatory_dims + list(self.structural_timevarying_dims)
                + self.extra_renovacion)

    @property
    def dimension_columns(self) -> list:
        """Every dimension, each once, in the order of the taxonomy: mandatory, timevarying,
        extra_renovacion, extra_revalorizacion."""
        return list(dict.fromkeys(self.business_mandatory_dims + list(self.structural_timevarying_dims)
                                  + self.extra_renovacion + self.extra_revalorizacion))

    @property
    def simulation_window(self) -> list:
        """The months simulated: from the current month (included) to simulation_end (December by default)."""
        current = self.calendar_boundaries()["current"]
        end = parse_month(self.simulation_end) if self.simulation_end else pd.Period(f"{current.year}-12", freq="M")
        return list(pd.period_range(current, end, freq="M")) if end >= current else []

    @property
    def uplift_cell_columns(self) -> list:
        """The columns of ONE uplift cell (the price context): the uplift mandatory dims
        (all mandatory unless uplift_mandatory_dims says fewer) + extra_revalorizacion +
        the discount bucket derived from the exact discount (when there is one)."""
        uplift_mandatory = self.uplift_mandatory_dims or self.business_mandatory_dims      # None or [] = all, as steps 14 and 20
        bucket = [self.discount_bucket_column] if self.discount_value_column else []
        return list(uplift_mandatory) + self.extra_revalorizacion + bucket

    @property
    def fine_row_columns(self) -> list:
        """What makes a fine row unique inside its forecast unit: the extra_revalorizacion
        values and the exact discount (the extract comes one row per exact discount)."""
        exact_discount = [self.discount_value_column] if self.discount_value_column else []
        return self.extra_revalorizacion + exact_discount

    def maturation_columns_of(self, state: str) -> list:
        """The maturation columns of one state, k = 0 … maturation_months (the last one: that many months or more)."""
        return [f"{state}{MATURATION_COLUMN_INFIX}{months_before}" for months_before in range(int(self.maturation_months) + 1)]

    @property
    def maturation_columns(self) -> list:
        """Every maturation column the Config declares, state by state."""
        return [column for state in self.maturation_dims for column in self.maturation_columns_of(state)]

    @property
    def core_measures(self) -> list:
        """The four measures every step computes with."""
        return [self.pipeline_units_col, self.pipeline_usd_col, self.renewed_units_col, self.renewed_usd_col]

    def column_roles(self) -> dict:
        """Every declared column with its role: {column: role}.

        A column may have ONE role. The only exception: the two extra groups may share a
        column (its role is then "extra_renovacion+extra_revalorizacion"). Any other
        column declared twice raises ValueError naming every conflict at once.
        """
        # [1] every (column, role) pair the Config declares
        declared_pairs = [(self.period_col, COLUMN_ROLE_PERIOD),
                          (self.flag_time_series_col, COLUMN_ROLE_TIME_SERIES_FLAG)]
        for measure_column in self.core_measures + list(self.extra_measure_cols) + self.closed_month_measure_cols:
            declared_pairs.append((measure_column, COLUMN_ROLE_MEASURE))
        for mandatory_column in self.business_mandatory_dims:
            declared_pairs.append((mandatory_column, COLUMN_ROLE_MANDATORY))
        for timevarying_column in self.structural_timevarying_dims:
            declared_pairs.append((timevarying_column, COLUMN_ROLE_TIMEVARYING))
        for extra_column in self.extra_renovacion:
            declared_pairs.append((extra_column, COLUMN_ROLE_EXTRA_RENOVACION))
        for extra_column in self.extra_revalorizacion:
            declared_pairs.append((extra_column, COLUMN_ROLE_EXTRA_REVALORIZACION))
        for formula_input_column in (self.discount_value_column, self.sku_column):
            if formula_input_column:
                declared_pairs.append((formula_input_column, COLUMN_ROLE_FORMULA_INPUT))
        for maturation_column in self.maturation_columns:
            declared_pairs.append((maturation_column, COLUMN_ROLE_MATURATION))
        for ignored_column in self.ignore_cols:
            declared_pairs.append((ignored_column, COLUMN_ROLE_IGNORE))

        # [2] group the roles of each column
        roles_by_column = {}
        for column_name, role in declared_pairs:
            roles_by_column.setdefault(column_name, [])
            if role not in roles_by_column[column_name]:
                roles_by_column[column_name].append(role)

        # [3] one role per column, except a column shared by the two extra groups
        both_extras = {COLUMN_ROLE_EXTRA_RENOVACION, COLUMN_ROLE_EXTRA_REVALORIZACION}
        column_roles, conflicts = {}, {}
        for column_name, roles in roles_by_column.items():
            if len(roles) == 1:
                column_roles[column_name] = roles[0]
            elif set(roles) == both_extras:
                column_roles[column_name] = COLUMN_ROLE_BOTH_EXTRAS
            else:
                conflicts[column_name] = roles
        if conflicts:
            raise ValueError(f"columns declared with more than one role: {conflicts}")
        return column_roles

    # ═══════════════════════════════════════════════════════════════════════════════
    # THE CALENDAR
    # ═══════════════════════════════════════════════════════════════════════════════

    def calendar_boundaries(self) -> dict:
        """The two months that cut the calendar: current (the first month of projection; every
        month before it is closed) and test_start (the first exam month: current − test_months).
        Raises ValueError when current_month is not declared."""
        if self.current_month is None:
            raise ValueError("current_month is not declared: set it in your Config "
                             "(e.g. current_month=\"2026-09\" or \"01/09/2026\")")
        current = parse_month(self.current_month)
        test_start = current - int(self.test_months)
        return dict(current=current, test_start=test_start)

    def role_of_months(self, periods) -> np.ndarray:
        """The role of every month: ≥ current → proyeccion · ≥ test_start → examen ·
        earlier → entrenamiento."""
        boundaries = self.calendar_boundaries()
        month_values = pd.Series(periods).reset_index(drop=True)
        # np.select takes the FIRST condition that holds, so the order is the rule
        conditions = [month_values >= boundaries["current"],
                      month_values >= boundaries["test_start"]]
        return np.select(conditions, [ROLE_PROJECTION, ROLE_TEST], default=ROLE_TRAIN)

    def calendar_description(self) -> str:
        """The calendar in one line, for the console."""
        boundaries = self.calendar_boundaries()
        current, test_start = boundaries["current"], boundaries["test_start"]
        return (f"{ROLE_TRAIN} ≤ {test_start - 1} · {ROLE_TEST} {test_start}..{current - 1} · "
                f"{ROLE_PROJECTION} ≥ {current}")


# ═══════════════════════════════════════════════════════════════════════════════
# THE PARAMETER TABLE: every field of the Config, its value, and the steps that read it
# ═══════════════════════════════════════════════════════════════════════════════
MODULE_LABELS = {"pipeline.py": "ORQ", "block_1_preparation.py": "B1"}   # every other stage is a step_NN_ module: its number
# the modules that hold several steps, one section each ("# STEP NN · …"; "# BLOCK 1 · …" is the block's own code)
SECTIONED_MODULES = {"block_1_preparation.py"}
TECHNIQUE_USERS = ["12", "13", "19", "23"]          # the steps that call the techniques
PREDICTION_USERS = ["13", "19"]                      # the steps that call the prediction (the credibility shift)
# the shared modules that read the Config themselves (configuration.<field>), and the steps that call them
SHARED_MODULE_USERS = {"price_increases.py": ["14", "15", "16", "20", "25"], "maturation.py": ["17"],
                       "resumen_negocio.py": ["27"], "graficos.py": ["27"]}
LABEL_ORDER = ["CFG", "ORQ"] + [f"{number:02d}" for number in range(7)] + ["B1"] + [f"{number:02d}" for number in range(7, 28)]


def label_of_module(file_name: str) -> Optional[str]:
    """The step label of a module of the framework: step_20_forecast.py → "17"; None for a module that is no step."""
    if file_name in MODULE_LABELS:
        return MODULE_LABELS[file_name]
    match = re.match(r"step_(\d\d)_", file_name)
    return match.group(1) if match else None


def field_comments() -> dict:
    """The comment written next to every field of the Config (its first line), from this file's source."""
    comments, inside_config = {}, False
    with open(os.path.abspath(__file__), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("class Config"):
                inside_config = True
                continue
            if inside_config and line.startswith("    def "):
                break
            match = re.match(r"^    (\w+): [^#]*#\s*(.*)$", line.rstrip("\n"))
            if inside_config and match and match.group(1) not in comments:
                comments[match.group(1)] = match.group(2).strip()
    return comments


def sections_of_module(source: str, module_label: str) -> list:
    """(label, code) of every section of a module that holds several steps: a "# STEP NN · …" line opens the section of
    step NN; a "# BLOCK 1 · …" line, or the code before the first step, belongs to the module itself."""
    sections, label, start = [], module_label, 0
    for match in re.finditer(r"^# (?:STEP (\d\d)|BLOCK \d+) · .*$", source, flags=re.M):
        sections.append((label, source[start:match.start()]))
        label, start = (match.group(1) or module_label), match.start()
    sections.append((label, source[start:]))
    return sections


def steps_reading_each_field(field_names: list) -> dict:
    """For every field, the steps whose code reads it: configuration.<field> in a step module, through the
    techniques or the prediction (parameters["<field>"]), or the Config itself (self.<field>, label CFG)."""
    folder = os.path.dirname(os.path.abspath(__file__))
    sources = {}
    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith(".py") and (label_of_module(file_name) or file_name in SHARED_MODULE_USERS
                                          or file_name in ("techniques.py", "prediction.py", "config.py")):
            with open(os.path.join(folder, file_name), encoding="utf-8") as handle:
                sources[file_name] = handle.read()
    readers = {}
    for field_name in field_names:
        steps = set()
        attribute = re.compile(rf"configuration\.{field_name}\b")
        for file_name, source in sources.items():
            if file_name in SECTIONED_MODULES:
                steps.update(label for label, chunk in sections_of_module(source, label_of_module(file_name))
                             if attribute.search(chunk))
                continue
            label = label_of_module(file_name)
            if label and attribute.search(source):
                steps.add(label)
            if file_name in SHARED_MODULE_USERS and attribute.search(source):
                steps.update(SHARED_MODULE_USERS[file_name])
            if f'parameters["{field_name}"]' in source:
                steps.update(TECHNIQUE_USERS if file_name == "techniques.py" else PREDICTION_USERS)
        config_source = sources.get("config.py", "")
        # the Config reads it itself (self.<field>), in a helper of this module (configuration.<field>), or by name
        # (getattr over a list of field names, like the isolated ratio bands)
        if (re.search(rf"self\.{field_name}\b", config_source) or attribute.search(config_source)
                or f'"{field_name}"' in config_source):
            steps.add("CFG")
        readers[field_name] = sorted(steps, key=lambda label: LABEL_ORDER.index(label) if label in LABEL_ORDER else 99)
    return readers


def parameter_table(configuration) -> pd.DataFrame:
    """Every field of the Config: its value in this run, its default, whether it was changed, the steps that
    read it (from the code) and the steps its comment documents ([..]), and the comment itself."""
    from dataclasses import MISSING, fields as dataclass_fields

    def as_text(value) -> str:
        if value is MISSING:
            return "(required)"
        text = repr(value) if not hasattr(value, "connect") else "(engine)"
        return text if len(text) <= 160 else text[:157] + "…"

    config_fields = dataclass_fields(type(configuration))
    comments = field_comments()
    readers = steps_reading_each_field([config_field.name for config_field in config_fields])
    rows = []
    for config_field in config_fields:
        value = getattr(configuration, config_field.name)
        default = (config_field.default if config_field.default is not MISSING
                   else config_field.default_factory() if config_field.default_factory is not MISSING else MISSING)
        comment = comments.get(config_field.name, "")
        documented = re.match(r"\[([^\]]+)\]\s*", comment)
        try:
            changed = default is MISSING or bool(value != default)
        except (TypeError, ValueError):
            changed = value is not default
        rows.append({"parametro": config_field.name,
                     "valor": as_text(value),
                     "por_defecto": as_text(default),
                     "cambiado": int(changed),
                     "pasos": " ".join(readers[config_field.name]),
                     "pasos_documentados": documented.group(1) if documented else "",
                     "descripcion": comment[documented.end():] if documented else comment})
    return pd.DataFrame(rows)
