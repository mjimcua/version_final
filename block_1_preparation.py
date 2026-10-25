"""
block_1_preparation.py — BLOCK 1 · PREPARATION: from the raw extract to the tables every prediction starts from.

WHAT THE BLOCK DOES: it checks that the extract fits the Config and that its values can be computed with, gives
every month its role, names every row by its contexts (series, unit, price cell), adds the rows of every forecast
unit, decides the route of every series and prepares its renewal rate. It predicts nothing.

THE STEPS (each one keeps its number, its actions and its checks, as before 25-oct-2026, when they were one file each):
  00  validate the raw         the extract fits the Config: columns with a role, months that hold the calendar
  01  values and levels        the two universes, the values, the generated <column>_level_1
  02  apply the calendar       the role of every month; the future must look like it has not started
  03  fine table               every row with its ids: fs_id, fu_id, uplift_cell_id, the discount bucket
  04  forecast units           the fine rows of every series in its month, added up
  05  series and routes        the months of every series and how it will be treated
  06  rate series              gaps, the rate of every unit, the support, own rate, error and sign of every series

THE INTERFACE:
  block = PreparationBlock(configuration).run()        every step, in order; the results stay in memory
  block.persist_block()                                 optional: the results to SQL (or CSV without an engine)
  block = PreparationBlock(configuration).resume_block_from_sql()   optional: the results back from SQL
  Orchestrator(configuration).run(preparation=block)   the rest of the pipeline, from step 07, with this block

  The results are attributes of the block (in memory unless persisted):
    raw_extract        the extract as read
    validated_extract  step 00: the same rows, period as a monthly Period
    leveled_raw        step 01: the renewal universe with its generated levels
    time_series_rows   step 01: the time_series universe
    calendared_raw     step 02: the role of every month, the s0_ columns
    fine_table         step 03: every row with its ids
    forecast_units     step 04: one row per series and month
    series_table       step 05: one row per series with its route
    rated_units        step 06: the units with the gap rows and the rate of every unit
    series_rate        step 06: one row per series with its support, own rate, error and sign

PERSIST AND RESUME:
  persist_block writes what the steps after the block read (time_series_rows, fine_table, forecast_units,
  series_table, rated_units, series_rate) and two small tables: the manifest (every column and its type, to give
  every column back its type when it is read) and the fingerprint (what the block was built from: the rows and the
  USD due of the extract, the current month, and the value of every field of the Config that the block reads).
  resume_block_from_sql reads them back, every column with its type and every table with its index, and reads the
  extract again (steps 26 and 27 need it). It stops if the extract, the current month or a field of the Config the
  block reads has changed since it was persisted: the block would no longer be the one of this run.
  validated_extract and leveled_raw are not persisted: no step after the block reads them. calendared_raw is the fine
  table without the four columns of step 03.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import (ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, COLUMN_ROLE_IGNORE, COMBINED_ID_SEPARATOR,
                    COVERAGE_COLUMN, CURRENT_MONTH_COLUMN, Config, FINE_ROWS_COLUMN, GENERATED_LEVEL_SUFFIX,
                    ID_FIELD_SEPARATOR, ISOLATED_RATIO_BANDS, NO_REVALUATION_VALUES, RATE_COLUMN, ROLES_IN_ORDER,
                    ROLE_PROJECTION, ROLE_TEST, ROLE_TRAIN, ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY,
                    ROUTE_PREDICTABLE, S0_PIPELINE_UNITS_COLUMN, S0_PIPELINE_USD_COLUMN, S0_RENEWED_UNITS_COLUMN,
                    S0_RENEWED_USD_COLUMN, SERIES_ID_COLUMN, SIGN_COLUMN, SIGN_MIXED, SIGN_NEGATIVE, SIGN_NEUTRAL,
                    SIGN_POSITIVE, STATUS_FAILED, SYNTHETIC_COLUMN, TABLE_BLOCK_1_FINE_TABLE, TABLE_BLOCK_1_FINGERPRINT,
                    TABLE_BLOCK_1_FORECAST_UNITS, TABLE_BLOCK_1_MANIFEST, TABLE_BLOCK_1_RATED_UNITS,
                    TABLE_BLOCK_1_SERIES, TABLE_BLOCK_1_SERIES_RATE, TABLE_BLOCK_1_TIME_SERIES_ROWS, TRUTH_ROLES,
                    UNIT_ID_COLUMN, UNKNOWN_DISCOUNT_BUCKET, UPLIFT_CELL_ID_COLUMN, discount_bucket_labels, is_one_year,
                    join_columns, parse_month, region_columns_of, roles_overview, steps_reading_each_field)
from report_queries import calendar_query, roles_query


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 00 · VALIDATE RAW  (until 25-oct-2026: step_00_validate_raw.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The first look at the raw: does it fit the configuration?
#
# STRUCTURE, not content: the columns and the months. The values inside the measures
# and the dimensions are step 01.
#
# Actions (logged as they are done):
#   1. read the roles the Config declares
#   2. compare the columns of the raw with them                       checks 1-4
#   3. read the months of the raw and fit the calendar to them        checks 5-10
#   4. count the checks; stop if any failed
#   5. convert the period to a monthly Period
#   6. show the columns by role and the calendar, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the raw has rows
#    2. no column name appears twice
#    3. every column of the raw has a role in the Config
#    4. every declared column is in the raw (the ignored ones may be absent)
#    5. every row has a month
#    6. every value of the period column is a month
#    7. current_month is declared
#    8. the current month is inside the months of the raw
#    9. the calendar leaves months to train
#   10. every month between the first and the last has rows            (warning only)
#
# Output: a copy of the raw, same rows and columns, with the period as a monthly Period.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_00_LABEL = "00"
STEP_00_NAME = "VALIDATE RAW"
STEP_00_PURPOSE = ("check that the raw fits the configuration before reading any value: every column has a role "
                "and is there, and the months of the raw can hold the calendar (current month, exam, training)")
STEP_00_ACTIONS = ["read the roles the Config declares",
                "compare the columns of the raw with them (checks 1-4)",
                "read the months of the raw and fit the calendar to them (checks 5-10)",
                "count the checks; stop if any failed",
                "convert the period to a monthly Period",
                "show the columns by role and the calendar, as tables"]
STEP_00_OUTPUT = "the raw, same rows and columns, with the period as a monthly Period (nothing is written)"


def validate_raw(raw: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Check the columns and the months of the raw against the Config, logging every action and check."""
    configuration.log_step_start(STEP_00_LABEL, STEP_00_NAME, STEP_00_PURPOSE, STEP_00_ACTIONS, STEP_00_OUTPUT)
    check_log = []
    period_column = configuration.period_col

    # [1] the roles the Config declares
    column_roles = configuration.column_roles()
    configuration.log_action(STEP_00_LABEL, 1, f"the Config declares {len(column_roles)} columns in "
                                            f"{len(set(column_roles.values()))} roles")

    # [2] the columns of the raw against the declared ones
    configuration.log_action(STEP_00_LABEL, 2, f"the raw has {len(raw.columns)} columns and {len(raw):,} rows")
    configuration.log_check(STEP_00_LABEL, check_log, "the raw has rows", not raw.empty,
                            failure_detail="the raw has no rows", context=f"{len(raw):,} rows")

    duplicated_columns = sorted(set(raw.columns[raw.columns.duplicated()]))     # a SQL join can repeat a name
    configuration.log_check(STEP_00_LABEL, check_log, "no column name appears twice", not duplicated_columns,
                            failure_detail=f"duplicated column names in the raw: {duplicated_columns}")

    columns_without_role = [column_name for column_name in raw.columns if column_name not in column_roles]
    configuration.log_check(STEP_00_LABEL, check_log, "every column of the raw has a role in the Config",
                            not columns_without_role,
                            failure_detail=f"columns with no role in the Config: {columns_without_role} "
                                           f"(declare their role or add them to ignore_cols)",
                            context=f"{len(raw.columns)} columns")

    missing_columns = [column_name for column_name, role in column_roles.items()
                       if role != COLUMN_ROLE_IGNORE and column_name not in raw.columns
                       and column_name not in configuration.generated_columns]       # made in step 01
    configuration.log_check(STEP_00_LABEL, check_log, "every declared column is in the raw (ignored ones may be absent; "
                            "the generated levels are made in step 01)",
                            not missing_columns,
                            failure_detail=f"columns declared in the Config but missing from the raw: {missing_columns}")

    # [3] the months, and the calendar on them (every distinct value read once)
    raw_months, months_by_value = [], {}
    if period_column in raw.columns and period_column not in duplicated_columns:
        raw_months, months_by_value = check_months(raw, configuration, check_log)
    else:
        configuration.log_action(STEP_00_LABEL, 3, f"no usable '{period_column}' column: the months cannot be read")
        configuration.log_not_evaluated(STEP_00_LABEL, check_log, "the months and the calendar",
                                        f"no usable '{period_column}' column")

    # [4] the count of the checks; stop if anything failed
    configuration.log_action(STEP_00_LABEL, 4, "counting the checks")
    configuration.log_check_summary(STEP_00_LABEL, STEP_00_NAME, check_log)

    # [5] the period as a monthly Period: the month of every distinct value, already read in [3], mapped onto the rows
    validated = raw.copy()
    validated[period_column] = pd.PeriodIndex(validated[period_column].map(months_by_value), freq="M")
    configuration.log_action(STEP_00_LABEL, 5, f"'{period_column}' converted to a monthly Period")

    # [6] what the raw is
    log_raw_report(validated, configuration, column_roles, raw_months)
    return validated


def check_months(raw: pd.DataFrame, configuration: Config, check_log: list) -> tuple:
    """Checks 5 to 10: the months can be read and the calendar fits them. Returns the sorted months and
    the month of every distinct value of the period column (read once each)."""
    period_column = configuration.period_col
    period_values = raw[period_column]

    # [5] no row without a month
    rows_without_month = int(period_values.isna().sum())
    configuration.log_check(STEP_00_LABEL, check_log, "every row has a month", rows_without_month == 0,
                            failure_detail=f"{rows_without_month:,} rows with no {period_column}")

    # [6] every distinct value is a month (read once per distinct value)
    months_by_value, unparseable_values = {}, []
    for distinct_value in period_values.dropna().unique():
        try:
            months_by_value[distinct_value] = parse_month(distinct_value)
        except ValueError:
            unparseable_values.append(distinct_value)
    raw_months = sorted(set(months_by_value.values()))
    months_text = f"{len(raw_months)} months, {raw_months[0]}..{raw_months[-1]}" if raw_months else "no month"
    configuration.log_action(STEP_00_LABEL, 3, f"the months of the raw: {months_text}")
    configuration.log_check(STEP_00_LABEL, check_log, f"every value of {period_column} is a month", not unparseable_values,
                            failure_detail=f"values that are not a month: {unparseable_values[:10]}",
                            context=months_text)
    if not raw_months:
        configuration.log_not_evaluated(STEP_00_LABEL, check_log, "the calendar against the months of the raw",
                                        "no month could be read")
        return raw_months, months_by_value

    # [7] the current month is declared (the calendar has no default date)
    try:
        boundaries = configuration.calendar_boundaries()
    except ValueError as error:
        configuration.log_check(STEP_00_LABEL, check_log, "current_month is declared", False, failure_detail=str(error))
        configuration.log_not_evaluated(STEP_00_LABEL, check_log, "the calendar against the months of the raw",
                                        "no current month")
        return raw_months, months_by_value
    configuration.log_check(STEP_00_LABEL, check_log, "current_month is declared", True, context=str(boundaries["current"]))

    # [8] the current month is inside the raw
    first_month, last_month = raw_months[0], raw_months[-1]
    configuration.log_check(STEP_00_LABEL, check_log, "the current month is inside the months of the raw",
                            first_month <= boundaries["current"] <= last_month,
                            failure_detail=f"current_month {boundaries['current']} is outside the months of the raw "
                                           f"({first_month}..{last_month}): the extract must carry the pipeline "
                                           f"from the current month on",
                            context=f"{boundaries['current']} in {first_month}..{last_month}")

    # [9] the exam leaves months to train
    configuration.log_check(STEP_00_LABEL, check_log, "the calendar leaves months to train",
                            boundaries["test_start"] > first_month,
                            failure_detail=f"the exam starts at {boundaries['test_start']} and the raw starts at "
                                           f"{first_month}: no month to train (lower test_months)",
                            context=f"training {first_month}..{boundaries['test_start'] - 1}")

    # [08] a month with no row at all is worth a look, not a stop
    expected_months = pd.period_range(first_month, last_month, freq="M")
    months_without_rows = [str(month) for month in expected_months if month not in set(raw_months)]
    configuration.log_check(STEP_00_LABEL, check_log, "every month between the first and the last has rows",
                            not months_without_rows,
                            failure_detail=f"months with no row at all: {months_without_rows}", blocking=False)
    return raw_months, months_by_value


def log_raw_report(validated: pd.DataFrame, configuration: Config, column_roles: dict, raw_months: list) -> None:
    """Action 6: the columns by role and the calendar, rows per role, as tables."""
    configuration.log_action(STEP_00_LABEL, 6, "the columns of the raw, by role (every role of the Config, also the empty "
                                            "ones, and the dimensions that get a generated level in step 01):")
    configuration.show_table(roles_overview(validated.columns, configuration))

    configuration.logger.doc(f"[{STEP_00_LABEL}] the calendar of the Config on these months: "
                             f"{configuration.calendar_description()}")
    rows_per_role = pd.Series(configuration.role_of_months(validated[configuration.period_col])).value_counts()
    months_per_role = pd.Series(configuration.role_of_months(pd.Series(raw_months))).value_counts()
    configuration.show_table(pd.DataFrame([{CALENDAR_ROLE_COLUMN: role, "meses": int(months_per_role.get(role, 0)), "filas": int(row_count)}
                                           for role, row_count in rows_per_role.items()]))


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 01 · VALUES AND DIMENSION LEVELS  (until 25-oct-2026: step_01_values_and_levels.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The content of the raw: the two universes, the values, and the
# dimensions with a generated level.
#
# CONTENT, not structure: step 00 already checked that the columns and the months fit the
# configuration. This step reads what is inside, in three parts, as one step:
#
#   THE UNIVERSES   the rows flagged time_series (retail to subscription) carry only the region and
#                   the result; the other dimensions and the pipeline are empty. They are separated
#                   first, so every check below reads the renewal universe only. Step 21 receives
#                   them at the end.
#
#   THE VALUES      money without nulls or negatives, renewals coherent with what fell due,
#                   dimensions never empty, flags 0 / 1, discount as a share. Which months each
#                   check looks at:
#                     · the pipeline (units and USD falling due): every month, the future included
#                     · the renewals: only the months BEFORE the current one. From the current month
#                       on they are early results that step 02 wipes, so they are not checked.
#
#   THE LEVELS      some mandatory dimensions come with many values (a band, a term). The Config
#                   declares them in leveled_dims ({column: "ordinal" | "nominal"}); the column keeps
#                   its raw value (the fine level) and this step adds ONE column, <column>_level_1,
#                   with its values grouped by their renewal rate. The Config already counts
#                   <column>_level_1 as a mandatory dim, right after its column, from the moment it
#                   is built; this step only fills it, before the calendar, so every later step sees
#                   the same columns. The ladder collapses the fine level first, then the coarse one
#                   (step 07).
#
# HOW THE GROUPS ARE MADE (only the TRAINING months, taken from the calendar of the Config)
#   1. the STANDARDISED rate of every value: renewed / expected, where expected = the rate of its cell
#      (every other mandatory dim of the extract) × its units due, times the global rate. A value is
#      compared with the others inside the same cell, not in bulk.
#   2. values whose monthly support (median of the monthly units due) is below support_floor go to a
#      RESIDUAL group: too little to tell their rate.
#   3. the others are ordered (ordinal: by their value, numbers as numbers; nominal: by their rate) and
#      the two NEIGHBOURS with the closest rates are merged, again and again, while they differ by at most
#      the MERGE THRESHOLD.
#   4. the rate of every value and of every group, year by year, goes with them as evidence.
#
# THE MERGE THRESHOLD (the same principle as the ladder: merge while the bias it adds is smaller than the
# noise it removes). The generated level is only used by the series that climb the ladder, the ones below
# the support floor; their monthly rate carries at least the binomial noise of a series AT the floor,
# √(p(1−p)/support_floor). Two values whose rates differ by less than that cannot be told apart by any
# series that will use the merge: merging them adds less bias than the noise it removes. With p = 0.64 and
# a floor of 30, the threshold is 8.8 pp. level_merge_max_pp fixes another value (5 pp = the noise of a
# series of about 90 contracts a month). A dimension that ends in a single group does not separate the
# renewal rate: its level_1 is constant and its pass of the ladder merges nothing (step 07).
#
# THE JSON: the first run writes the groups to levels_path; every later run that finds the file USES it, so
# the ids of the forecast series do not move from one month to the next. A dimension missing from the file
# (or with another type, another merge threshold or another support floor) is generated and added; and every
# dimension is generated again when the file was trained for a LATER current month (a run of an earlier month
# must not see its exam and its future through the groups). To regenerate everything, delete the file. A value the file does not know goes to the residual group, with a warning.
#
# Actions (logged as they are done):
#    1. check the time_series flag and separate the two universes
#    2. scope the renewal checks: renewals are checked only in months before the current one
#    3. check the money: pipeline, every month; renewals, closed months
#    4. check the dimensions and the timevarying flags
#    5. check the exact discount, if declared
#    6. look for what is plausible but worth a look (warnings)
#    7. the dimensions with a generated level and the file: use it, or generate
#    8. generate the groups of every dimension that needs them (training months)
#    9. fill <column>_level_1
#   10. check the values the file knows
#   11. the levels as tables (on screen in action 14)
#   12. count the checks; stop if any failed
#   13. report the money of the closed months
#   14. show the criterion, the groups (rate per year), the values and the merge decisions, and the
#       columns by role, as tables
#
# Checks (logged as they are made, numbered in order, at the level of their status):
#   universes                      the time_series flag is 0 / 1
#                                  every time_series row has every region level
#                                  every closed time_series row has its units and value     (warning only)
#   pipeline, every month          no nulls · no negatives (units and USD)
#   renewals, closed months        no negatives (units and USD)
#                                  how "no renewal" arrives: null or 0 (warning if null,
#                                    with example rows shown as a table; step 02 reads a null as 0)
#                                  units and USD are null together
#                                  no renewal for free (units > 0, USD ≤ 0)
#                                  no USD without units (USD > 0, units = 0)
#   dimensions                     no null or empty value
#   flags                          each timevarying column is 0 / 1
#   exact discount (if declared)   between 0 and 1, or null (unknown)
#   warnings only                  no row renews more units than fall due (rate > 100 %)
#                                  no row falls due with units but USD = 0
#                                  no row repeated in every column (a double load)
#   levels (if declared)           no value of the extract is unknown to the file           (warning only)
#
# Output: (the raw of the renewal universe with <column>_level_1 for every leveled column, the time_series
# rows). Without leveled_dims the renewal raw comes back with its rows and columns unchanged.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_01_LABEL = "01"
STEP_01_NAME = "VALUES AND DIMENSION LEVELS"
STEP_01_PURPOSE = ("separate the time_series universe; check that the values inside the raw can be computed with "
                "(money without nulls or negatives, renewals coherent with what fell due, dimensions never empty, "
                "flags 0 / 1, discount as a share); and give every dimension declared in leveled_dims its coarse "
                "level, <column>_level_1: its values grouped by their standardised renewal rate in the training "
                "months, persisted in a JSON that later runs reuse so the forecast series keep their ids")
STEP_01_ACTIONS = ["check the time_series flag and separate the two universes",
                "scope the renewal checks: renewals are checked only in months before the current one "
                "(from the current month on they are partial and step 02 wipes them)",
                "check the money: pipeline every month, renewals in closed months",
                "check the dimensions and the timevarying flags",
                "check the exact discount, if declared",
                "look for what is plausible but worth a look: warnings",
                "the dimensions with a generated level and the file: use it, or generate",
                "generate the groups of every dimension that needs them (training months)",
                "fill <column>_level_1",
                "check the values the file knows",
                "the levels as tables (on screen in action 14)",
                "count the checks; stop if any failed",
                "report the money of the closed months",
                "show the criterion, the groups (rate per year), the values and the merge decisions, and the "
                "columns by role, as tables"]
STEP_01_OUTPUT = "the renewal raw with <column>_level_1 · the time_series rows · the JSON of the groups"

# ─── named constants ─────────────────────────────────────────────────────────────
# The only values a dichotomous column (timevarying, time_series flag) may carry.
DICHOTOMOUS_VALUES = {0, 1, True, False, "0", "1"}
RESIDUAL_GROUP = "residual"


def validate_values_and_build_levels(validated: pd.DataFrame, configuration: Config) -> tuple:
    """The content of the raw: the two universes, the values and the generated levels.

    INPUT:   the raw validated by step 00 · the Config (flags, measures, dimensions, discount, leveled_dims,
             levels_path, level_merge_max_pp, support_floor, the calendar).
    OUTPUT:  (the renewal raw with <column>_level_1 for every leveled column, the time_series rows).
    RULES:   see the module header.
    EDGE CASES: no time_series row → an empty frame, and step 21 only builds the total. No leveled_dims →
             actions 8-11 and 14 have nothing to do, nothing is written. A dimension with a single value →
             one group. All values below the floor → a single residual group.
    """
    configuration.log_step_start(STEP_01_LABEL, STEP_01_NAME, STEP_01_PURPOSE, STEP_01_ACTIONS, STEP_01_OUTPUT)
    check_log = []

    # [1] the two universes
    renewal_raw, time_series_rows = separate_the_universes(validated, configuration, check_log)

    # [2-6] the values of the renewal universe
    closed_rows = check_the_values(renewal_raw, configuration, check_log)

    # [7-11] the dimensions with a generated level
    leveled_raw, levels_evidence = build_the_levels(renewal_raw, configuration, check_log)

    # [10] the count of the checks; stop if anything failed
    configuration.log_action(STEP_01_LABEL, 12, "counting the checks")
    configuration.log_check_summary(STEP_01_LABEL, STEP_01_NAME, check_log)

    # [11] the money of the closed months, and the rate it implies (a null renewal counts as 0)
    report_the_closed_months(closed_rows, configuration)

    # [12] the evidence of the levels and the columns by role
    if levels_evidence is not None:
        show_the_evidence(*levels_evidence, configuration)
        configuration.logger.doc(f"[{STEP_01_LABEL}] the columns by role, with the generated levels:")
        configuration.show_table(roles_overview(leveled_raw.columns, configuration))
    else:
        configuration.log_action(STEP_01_LABEL, 14, "no dimension declared in leveled_dims: no evidence to show")
    return leveled_raw, time_series_rows


# ═══════════════════════════════════════════════════════════════════════════════════
# PART 1 · THE UNIVERSES
# ═══════════════════════════════════════════════════════════════════════════════════

def separate_the_universes(validated: pd.DataFrame, configuration: Config, check_log: list) -> tuple:
    """Action 1: the time_series flag is checked first (a wrong value would send a row to the wrong
    universe), then the flagged rows are taken out. They carry only the region and the result; the rest of
    their dimensions and their pipeline may be empty."""
    flag_column = configuration.flag_time_series_col
    flag_values = validated[flag_column]
    unexpected_values = [value for value in flag_values.dropna().unique() if value not in DICHOTOMOUS_VALUES]
    has_nulls = bool(flag_values.isna().any())
    configuration.log_check(STEP_01_LABEL, check_log, f"{flag_column} is 0 / 1",
                            not unexpected_values and not has_nulls,
                            failure_detail=f"{flag_column} must be 0 / 1: found {unexpected_values[:10]}"
                                           f"{' and nulls' if has_nulls else ''}",
                            context=f"{flag_values.isin(ACTIVE_FLAG_VALUES).mean():.0%} of rows at 1")

    flagged = flag_values.isin(ACTIVE_FLAG_VALUES)
    time_series_rows = validated[flagged].copy()
    renewal_raw = validated[~flagged].copy()
    region_columns = region_columns_of(configuration)
    configuration.log_action(STEP_01_LABEL, 1, f"{int(flagged.sum()):,} time_series rows (retail to subscription) out of "
                                            f"{len(validated):,}: they go to step 21 · {len(renewal_raw):,} rows of the "
                                            f"renewal universe go on · region levels {region_columns}")

    missing_region = int(time_series_rows[region_columns].isna().any(axis=1).sum()) if len(time_series_rows) else 0
    finest_regions = time_series_rows[region_columns].drop_duplicates().shape[0] if len(time_series_rows) else 0
    configuration.log_check(STEP_01_LABEL, check_log, "every time_series row has every region level", missing_region == 0,
                            failure_detail=f"{missing_region:,} time_series rows without some region level",
                            context=f"{finest_regions} finest regions")

    if len(time_series_rows):
        closed = time_series_rows[configuration.period_col] < configuration.calendar_boundaries()["current"]
        result_columns = [configuration.renewed_units_col, configuration.renewed_usd_col]
        missing_closed = int(time_series_rows.loc[closed, result_columns].isna().any(axis=1).sum())
    else:
        missing_closed = 0
    configuration.log_check(STEP_01_LABEL, check_log, "every closed time_series row has its units and value",
                            missing_closed == 0,
                            failure_detail=f"{missing_closed:,} closed rows without units or value (read as 0)",
                            blocking=False)
    return renewal_raw, time_series_rows


# ═══════════════════════════════════════════════════════════════════════════════════
# PART 2 · THE VALUES
# ═══════════════════════════════════════════════════════════════════════════════════

def check_the_values(renewal_raw: pd.DataFrame, configuration: Config, check_log: list) -> pd.DataFrame:
    """Actions 2-6: the money, the dimensions, the timevarying flags, the discount and the warnings of
    the renewal universe. Returns the rows of the closed months (action 13 reports their money)."""

    # [2] the scope of the renewal checks: the months before the current one. From the current
    #     month on the renewals are partial (the month is still running) and step 02 wipes them,
    #     so only the pipeline is checked there
    current_month = configuration.calendar_boundaries()["current"]
    closed_rows = renewal_raw[renewal_raw[configuration.period_col] < current_month]
    renewed_units = closed_rows[configuration.renewed_units_col]
    renewed_usd = closed_rows[configuration.renewed_usd_col]
    configuration.log_action(STEP_01_LABEL, 2, f"renewals checked in {len(closed_rows):,} rows before {current_month} · "
                                            f"{len(renewal_raw) - len(closed_rows):,} rows from {current_month} on: "
                                            f"pipeline only, renewals wiped in step 02")

    # [3] the money
    configuration.log_action(STEP_01_LABEL, 3, "checking the money")
    for pipeline_column in (configuration.pipeline_units_col, configuration.pipeline_usd_col):
        check_no_nulls(configuration, check_log, renewal_raw, pipeline_column, "every month")
        check_no_negatives(configuration, check_log, renewal_raw, pipeline_column, "every month")
    for renewed_column in (configuration.renewed_units_col, configuration.renewed_usd_col):
        check_no_negatives(configuration, check_log, closed_rows, renewed_column, "closed months")
    for measure in configuration.closed_month_measures:
        closed_month_column = measure.raw_column
        check_no_negatives(configuration, check_log, closed_rows, closed_month_column, "closed months")
        # a part cannot be more than its whole: the measure it is a part of, as the Config declares it
        total_column = measure.part_of
        if total_column is None:
            continue                                  # not a part of anything (the second moment): checked below
        # a null whole is not compared here: it is the null renewal, already warned (and read as 0 by step 02)
        above_total = closed_rows[closed_rows[closed_month_column] > closed_rows[total_column] + configuration.money_tolerance]
        configuration.log_check(STEP_01_LABEL, check_log, f"{closed_month_column} is not above {total_column}, closed months",
                                above_total.empty,
                                failure_detail=f"{len(above_total):,} closed rows with {closed_month_column} above "
                                               f"{total_column} (a part of the renewed above the renewed)",
                                blocking=False, examples=example_rows(above_total, configuration))

    # the dispersion of the isolated renewals: coherent with its base and renewed USD
    base_column, renewed_column = configuration.isolated_renewed_pipeline_usd_col, configuration.isolated_renewed_usd_col
    if configuration.isolated_renewed_usd_sq_over_tr_col:
        moment_column = configuration.isolated_renewed_usd_sq_over_tr_col
        # Σ renewed² / base ≥ (Σ renewed)² / Σ base always holds (a variance cannot be negative): below it, the
        # column was not built as Σ renewed_usd² / due_usd licence by licence
        with_base = closed_rows[closed_rows[base_column] > 0]
        negative_variance = with_base[with_base[moment_column] * with_base[base_column]
                                      < with_base[renewed_column] ** 2 * (1 - configuration.measure_relative_tolerance)
                                      - configuration.money_tolerance]
        configuration.log_check(STEP_01_LABEL, check_log, f"{moment_column} gives a variance ≥ 0, closed months",
                                negative_variance.empty,
                                failure_detail=f"{len(negative_variance):,} closed rows where Σ renewed² / base is below "
                                               f"(Σ renewed)² / Σ base: not built licence by licence as renewed² / due",
                                blocking=False, examples=example_rows(negative_variance, configuration))
    band_columns = [band[0] for band in configuration.isolated_ratio_bands]
    if len(band_columns) == len(ISOLATED_RATIO_BANDS):
        # the six bands cover every ratio: together they are the isolated base
        band_total = closed_rows[band_columns].fillna(0).sum(axis=1)
        off_base = closed_rows[(band_total - closed_rows[base_column].fillna(0)).abs()
                               > configuration.money_tolerance
                               + configuration.measure_relative_tolerance * closed_rows[base_column].fillna(0).abs()]
        configuration.log_check(STEP_01_LABEL, check_log, f"the {len(band_columns)} ratio bands add up to {base_column}, closed months",
                                off_base.empty,
                                failure_detail=f"{len(off_base):,} closed rows whose bands do not add up to the isolated "
                                               f"base (a licence in no band, or in two)",
                                blocking=False, examples=example_rows(off_base, configuration))

    # the isolated measures, read as a whole: are they coherent, and what do they say (before anything else runs)
    if base_column and renewed_column:
        isolated_coherence_summary(closed_rows, configuration, check_log)

    # how "nobody renewed" arrives: null (a LEFT JOIN with no match) or 0. A null is not an
    # error but it is not a number either: step 02 reads it as 0.
    no_result_rows = closed_rows[renewed_units.isna() | (renewed_units == 0)]
    null_rows = int(renewed_units.isna().sum())
    zero_rows = int((renewed_units == 0).sum())
    convention_text = f"{null_rows:,} rows null · {zero_rows:,} rows 0 (of {len(closed_rows):,} closed rows)"
    configuration.log_check(STEP_01_LABEL, check_log, "rows with no renewal arrive as 0, not null, closed months",
                            null_rows == 0,
                            failure_detail=f"{convention_text}: a null means no renewal (read as 0 in step 02)",
                            context=convention_text, blocking=False,
                            examples=example_rows(no_result_rows.sort_values(configuration.renewed_units_col,
                                                                             na_position="first"), configuration))

    null_mismatch = closed_rows[renewed_units.isna() != renewed_usd.isna()]     # one null without the other: broken
    configuration.log_check(STEP_01_LABEL, check_log, "renewed units and USD are null together, closed months",
                            null_mismatch.empty,
                            failure_detail=f"{len(null_mismatch):,} closed rows have one of the two renewal columns null",
                            examples=example_rows(null_mismatch, configuration))

    free_renewals = closed_rows[(renewed_units > 0) & (renewed_usd <= 0)]
    configuration.log_check(STEP_01_LABEL, check_log, "no renewal for free (units > 0 with USD ≤ 0), closed months",
                            free_renewals.empty,
                            failure_detail=f"{len(free_renewals):,} closed rows renew units with USD ≤ 0 "
                                           f"(renewing for free does not exist)",
                            examples=example_rows(free_renewals, configuration))
    usd_without_units = closed_rows[(renewed_usd > 0) & (renewed_units == 0)]
    configuration.log_check(STEP_01_LABEL, check_log, "no renewed USD without renewed units, closed months",
                            usd_without_units.empty,
                            failure_detail=f"{len(usd_without_units):,} closed rows have renewed USD with 0 renewed units",
                            examples=example_rows(usd_without_units, configuration))

    if configuration.renewed_pipeline_usd_col:
        base_column = configuration.renewed_pipeline_usd_col
        base_values = closed_rows[base_column]
        value_without_renewers = closed_rows[(renewed_units == 0) & (base_values > 0)]      # a null renewal: warned above
        configuration.log_check(STEP_01_LABEL, check_log, f"{base_column} is 0 where nothing renewed, closed months",
                                value_without_renewers.empty,
                                failure_detail=f"{len(value_without_renewers):,} closed rows carry a renewers' due value "
                                               f"with 0 renewed units",
                                blocking=False, examples=example_rows(value_without_renewers, configuration))
        renewers_without_value = closed_rows[(renewed_units > 0) & (base_values.fillna(0) == 0)]
        configuration.log_check(STEP_01_LABEL, check_log, f"{base_column} is above 0 where something renewed, closed months",
                                renewers_without_value.empty,
                                failure_detail=f"{len(renewers_without_value):,} closed rows renew units with no renewers' "
                                               f"due value (their uplift cannot be computed: they are left out of it)",
                                blocking=False, examples=example_rows(renewers_without_value, configuration))

    # [4] the dimensions and the timevarying flags
    configuration.log_action(STEP_01_LABEL, 4, f"checking {len(dimension_columns(configuration))} dimensions and "
                                            f"{len(configuration.structural_timevarying_dims)} timevarying flags")
    empty_by_dimension = {}
    for dimension_column in dimension_columns(configuration):
        values = renewal_raw[dimension_column]
        empty_rows = int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())
        if empty_rows:
            empty_by_dimension[dimension_column] = empty_rows
    configuration.log_check(STEP_01_LABEL, check_log, "no dimension has a null or empty value", not empty_by_dimension,
                            failure_detail=f"empty values in dimensions (rows per column): {empty_by_dimension}",
                            context=f"{len(dimension_columns(configuration))} dimensions")

    # the share of rows AND the share of the USD due: a flag on many small rows weighs little in money
    usd_due = renewal_raw[configuration.pipeline_usd_col]
    total_usd_due = float(usd_due.sum())

    def share_of_money(rows_mask) -> str:
        return f"{usd_due[rows_mask].sum() / total_usd_due:.0%} of the USD due" if total_usd_due else "no USD due"

    for timevarying_column in configuration.structural_timevarying_dims:
        values = renewal_raw[timevarying_column]
        unexpected_values = [value for value in values.dropna().unique() if value not in DICHOTOMOUS_VALUES]
        has_nulls = bool(values.isna().any())
        configuration.log_check(STEP_01_LABEL, check_log, f"{timevarying_column} is 0 / 1",
                                not unexpected_values and not has_nulls,
                                failure_detail=f"{timevarying_column} must be 0 / 1: found {unexpected_values[:10]}"
                                               f"{' and nulls' if has_nulls else ''}",
                                context=f"{values.isin(ACTIVE_FLAG_VALUES).mean():.0%} of rows at 1 · "
                                        f"{share_of_money(values.isin(ACTIVE_FLAG_VALUES))}")

    # the maturation columns: WHEN every state was set (k months before due), on the rows that carry the state
    for state in configuration.maturation_dims:
        check_the_maturation_columns(renewal_raw, state, configuration, check_log)

    # [5] the exact discount: a share between 0 and 1, or null (unknown)
    if configuration.discount_value_column:
        discount_values = renewal_raw[configuration.discount_value_column]
        out_of_range = int(((discount_values < 0) | (discount_values > 1)).sum())
        configuration.log_action(STEP_01_LABEL, 5, f"checking {configuration.discount_value_column}")
        configuration.log_check(STEP_01_LABEL, check_log,
                                f"{configuration.discount_value_column} is between 0 and 1 (or null = unknown)",
                                out_of_range == 0,
                                failure_detail=f"{out_of_range:,} rows of {configuration.discount_value_column} "
                                               f"outside [0, 1] (the discount is a share: 0.25 = 25 %)",
                                context=f"{discount_values.isna().mean():.1%} of rows unknown · "
                                        f"{share_of_money(discount_values.isna())}")
    else:
        configuration.log_action(STEP_01_LABEL, 5, "no exact discount declared: nothing to check")

    # [6] warnings: plausible in a real extract, worth a look
    configuration.log_action(STEP_01_LABEL, 6, "looking for what is plausible but worth a look")
    above_pipeline = closed_rows[renewed_units > closed_rows[configuration.pipeline_units_col]]
    excess_units = float((above_pipeline[configuration.renewed_units_col]
                          - above_pipeline[configuration.pipeline_units_col]).sum())
    configuration.log_check(STEP_01_LABEL, check_log, "no row renews more units than fall due, closed months",
                            above_pipeline.empty,
                            failure_detail=f"{len(above_pipeline):,} closed rows renew more units than fall due "
                                           f"(+{excess_units:,.0f} units: a rate above 100 %)",
                            blocking=False, examples=example_rows(above_pipeline, configuration))
    units_without_value = int(((renewal_raw[configuration.pipeline_units_col] > 0)
                               & (renewal_raw[configuration.pipeline_usd_col] == 0)).sum())
    configuration.log_check(STEP_01_LABEL, check_log, "no row falls due with units but USD = 0", units_without_value == 0,
                            failure_detail=f"{units_without_value:,} rows fall due with units but USD = 0",
                            blocking=False)
    repeated_rows = int(renewal_raw.duplicated().sum())
    configuration.log_check(STEP_01_LABEL, check_log, "no row repeated in every column", repeated_rows == 0,
                            failure_detail=f"{repeated_rows:,} rows repeated in every column (a possible double load)",
                            blocking=False)
    return closed_rows


def isolated_coherence_summary(closed_rows: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """The isolated measures of the extract as a whole, in the closed months: one check (the six bands add up
    to the isolated base IN TOTAL), and on screen the figures of an aggregate check of the extract (the rows
    where a part is above its whole, where the bands do not add up, where the variance is negative), the
    mean ratio, the share of the value in every band, and a verdict: coherent or not. Built so the extract
    can be judged in the first minute of the run, without a query on the licences.

    A licence-level error (a ratio outside the isolated renewals, or computed with other amounts than the
    base and the renewed USD) always shows here: its value lands in a band that the base does not have.
    """
    doc = configuration.logger.doc
    base_column, renewed_column = configuration.isolated_renewed_pipeline_usd_col, configuration.isolated_renewed_usd_col
    tolerance = configuration.money_tolerance
    base = closed_rows[base_column].fillna(0.0)
    renewed = closed_rows[renewed_column].fillna(0.0)
    findings = {}
    if configuration.renewed_pipeline_usd_col:
        findings["rows where the isolated base is above every renewer's base"] = int(
            (base > closed_rows[configuration.renewed_pipeline_usd_col].fillna(0.0) + tolerance).sum())
    findings["rows where the isolated renewed USD is above every renewer's"] = int(
        (renewed > closed_rows[configuration.renewed_usd_col].fillna(0.0) + tolerance).sum())
    moment_column = configuration.isolated_renewed_usd_sq_over_tr_col
    if moment_column:
        moment = closed_rows[moment_column].fillna(0.0)
        findings["rows with a negative variance (moment × base < renewed²)"] = int(
            ((base > 0) & (moment * base < renewed ** 2 * (1 - configuration.measure_relative_tolerance) - tolerance)).sum())
    bands = configuration.isolated_ratio_bands
    complete_bands = len(bands) == len(ISOLATED_RATIO_BANDS)
    if complete_bands:
        band_total_by_row = closed_rows[[band[0] for band in bands]].fillna(0.0).sum(axis=1)
        findings["rows whose six bands do not add up to the isolated base"] = int(
            ((band_total_by_row - base).abs() > tolerance + configuration.measure_relative_tolerance * base.abs()).sum())

    base_total, renewed_total = float(base.sum()), float(renewed.sum())
    doc(f"[{STEP_01_LABEL}] ══════════ ISOLATED: COHERENCIA DEL EXTRACTO ══════════")
    if base_total > 0:
        doc(f"[{STEP_01_LABEL}] closed months · isolated base ${base_total:,.0f} · isolated renewed ${renewed_total:,.0f} · "
            f"mean ratio {renewed_total / base_total:.3f}")
    else:
        doc(f"[{STEP_01_LABEL}] no isolated base in the closed months")
    for description, rows in findings.items():
        doc(f"[{STEP_01_LABEL}]   {rows:>10,}  {description}  {'(must be 0)' if rows else '✓'}")
    if complete_bands and base_total > 0:
        band_values = [float(closed_rows[column].fillna(0.0).sum()) for column, _, _, _, _ in bands]
        bands_over_base = sum(band_values) / base_total
        configuration.log_check(STEP_01_LABEL, check_log, "the six ratio bands add up to the isolated base in total, closed months",
                                abs(bands_over_base - 1) <= configuration.measure_relative_tolerance * 10,
                                failure_detail=f"bands / base = {bands_over_base:.4f}: the bands count licences the base "
                                               f"does not (a ratio outside the isolated renewals, or other amounts)",
                                blocking=False, context=f"bands / base = {bands_over_base:.4f}")
        shares = (" · ".join(f"{label} {value / sum(band_values):.1%}" for (_, _, _, _, label), value in zip(bands, band_values))
                  if sum(band_values) > 0 else "—")
        doc(f"[{STEP_01_LABEL}]   bands / base = {bands_over_base:.4f} (must be 1.0000) · the value by ratio: {shares}")
    coherent = all(rows == 0 for rows in findings.values())
    verdict = ("the isolated measures are COHERENT: the dispersion of step 14 (LICENCE BY LICENCE) and of step 25 can be read"
               if coherent else
               "the isolated measures are NOT coherent: the histogram, the std deviation and the shares near 1 of steps "
               "15 and 22 are NOT reliable; the mean ratio is")
    doc(f"[{STEP_01_LABEL}] VALORACIÓN: {verdict}")
    doc(f"[{STEP_01_LABEL}] ══════════ FIN DE LA COHERENCIA ISOLATED ══════════")


def report_the_closed_months(closed_rows: pd.DataFrame, configuration: Config) -> None:
    """Action 13: the money of the closed months, and the rate it implies (a null renewal counts as 0)."""
    pipeline_units = closed_rows[configuration.pipeline_units_col].sum()
    pipeline_usd = closed_rows[configuration.pipeline_usd_col].sum()
    renewed_units = closed_rows[configuration.renewed_units_col].sum()
    renewed_usd = closed_rows[configuration.renewed_usd_col].sum()
    renewal_rate_text = f"{renewed_units / pipeline_units:.1%}" if pipeline_units else "no units due"
    configuration.log_action(STEP_01_LABEL, 13, f"closed months: {pipeline_units:,.0f} units due · "
                                             f"{renewed_units:,.0f} renewed ({renewal_rate_text}) · "
                                             f"${pipeline_usd:,.0f} due · ${renewed_usd:,.0f} renewed")


def check_the_maturation_columns(renewal_raw: pd.DataFrame, state: str, configuration: Config, check_log: list) -> None:
    """The maturation columns of one state: <state>_month_0 … <state>_month_<maturation_months>, the units of the row
    whose state was set k months before its due month (the last column: that many months or more).

    RULES:   numeric and never negative (blocking) · on a row with the state, they add up to its units due · on a row
             without it, null or 0 · in a future month T, no unit can have been in the state after the current month:
             the columns with k < T − current month are 0 (a state dated in the future is a leak: the extract was
             built with information it could not have had). The last three only warn.
    """
    columns = configuration.maturation_columns_of(state)
    units_column = configuration.pipeline_units_col
    not_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(renewal_raw[column])]
    configuration.log_check(STEP_01_LABEL, check_log, f"the maturation columns of {state} are numeric", not not_numeric,
                            failure_detail=f"not numeric: {not_numeric}", context=f"{columns[0]} … {columns[-1]}")
    if not_numeric:
        return
    timing = renewal_raw[columns].astype(float)
    negative_rows = int((timing < 0).any(axis=1).sum())
    configuration.log_check(STEP_01_LABEL, check_log, f"the maturation columns of {state} are never negative", negative_rows == 0,
                            failure_detail=f"{negative_rows:,} rows with a negative maturation value")
    in_state = renewal_raw[state].isin(ACTIVE_FLAG_VALUES)
    dated_units = timing.fillna(0.0).sum(axis=1)
    differs = in_state & ((dated_units - renewal_raw[units_column]).abs() > configuration.units_tolerance)
    units_in_state = float(renewal_raw.loc[in_state, units_column].sum())
    configuration.log_check(STEP_01_LABEL, check_log, f"on the rows with {state} = 1 the maturation columns add up to {units_column}",
                            not differs.any(),
                            failure_detail=f"{int(differs.sum()):,} rows of {int(in_state.sum()):,} where they do not "
                                           f"({abs(float((dated_units - renewal_raw[units_column])[differs].sum())):,.0f} units of difference)",
                            context=f"{int(in_state.sum()):,} rows · {units_in_state:,.0f} units with the state, every one dated",
                            blocking=False, examples=example_rows(renewal_raw[differs], configuration))
    filled_without_state = ~in_state & (timing.fillna(0.0) != 0).any(axis=1)
    configuration.log_check(STEP_01_LABEL, check_log, f"on the rows with {state} = 0 the maturation columns are null or 0",
                            not filled_without_state.any(),
                            failure_detail=f"{int(filled_without_state.sum()):,} rows without the state carry a date of it",
                            blocking=False, examples=example_rows(renewal_raw[filled_without_state], configuration))
    current_month = configuration.calendar_boundaries()["current"]
    months_after_current = renewal_raw[configuration.period_col].map(lambda month: (month - current_month).n).to_numpy()
    impossible = np.zeros(len(renewal_raw), dtype=bool)
    for months_before, column in enumerate(columns[:-1]):        # the last one is "that many months or more": always possible
        impossible |= (months_before < months_after_current) & (timing[column].fillna(0.0).to_numpy() > 0)
    configuration.log_check(STEP_01_LABEL, check_log, f"no unit of a future month has {state} dated after the current month",
                            not impossible.any(),
                            failure_detail=f"{int(impossible.sum()):,} future rows with {state} set after {current_month} "
                                           f"(a leak: the extract knows something that has not happened yet)",
                            context=f"future month T: only k ≥ T − {current_month} can carry units",
                            blocking=False, examples=example_rows(renewal_raw[impossible], configuration))


def check_no_nulls(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    null_rows = int(rows[column_name].isna().sum())
    configuration.log_check(STEP_01_LABEL, check_log, f"{column_name} has no nulls, {scope}", null_rows == 0,
                            failure_detail=f"{null_rows:,} nulls in {column_name} ({scope})")


def check_no_negatives(configuration: Config, check_log: list, rows: pd.DataFrame, column_name: str, scope: str) -> None:
    negative_rows = int((rows[column_name] < 0).sum())
    configuration.log_check(STEP_01_LABEL, check_log, f"{column_name} has no negatives, {scope}", negative_rows == 0,
                            failure_detail=f"{negative_rows:,} negative values in {column_name} ({scope})")


def example_rows(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The columns that identify a row and its money, to show a few rows under a check."""
    return rows[[configuration.period_col] + configuration.extract_mandatory_dims + configuration.core_measures]


def dimension_columns(configuration: Config) -> list:
    """Every dimension column, each once: mandatory, timevarying and both extra groups."""
    all_dimensions = (configuration.extract_mandatory_dims + list(configuration.structural_timevarying_dims)
                      + configuration.extra_renovacion + configuration.extra_revalorizacion)
    return list(dict.fromkeys(all_dimensions))


# ═══════════════════════════════════════════════════════════════════════════════════
# PART 3 · THE DIMENSIONS WITH A GENERATED LEVEL
# ═══════════════════════════════════════════════════════════════════════════════════

def build_the_levels(renewal_raw: pd.DataFrame, configuration: Config, check_log: list) -> tuple:
    """Actions 7-11: the generated coarse level of every dimension in leveled_dims. Returns (the raw with
    <column>_level_1, the evidence to show in action 14 — None when no dimension is leveled)."""
    if not configuration.leveled_dims:
        configuration.log_action(STEP_01_LABEL, 7, "no dimension declared in leveled_dims: nothing to generate")
        return renewal_raw, None

    # the groups are made from the values: if a check of the values already failed, they would be made
    # from wrong data (the step stops in action 12 anyway)
    if any(logged_check[0] == STATUS_FAILED for logged_check in check_log):
        configuration.log_action(STEP_01_LABEL, 7, "the levels are not built: a check of the values failed")
        return renewal_raw, None
    raw = renewal_raw.copy()

    # [7] the declared dimensions and the file
    path = configuration.levels_path or os.path.join(configuration.output_folder or ".", "sff_levels.json")
    stored = read_levels_file(path)
    # a dimension is generated again when the file does not have it, its type changed, its merge criterion changed
    # (the fixed threshold or the support floor it is derived from), or the file was trained with months that are not
    # training months for this run (a later current month: they would leak the exam and the future into the groups)
    trained_later = str(stored.get("current_month", "")) > str(configuration.current_month)
    to_generate = [column_name for column_name, level_type in configuration.leveled_dims.items()
                   if trained_later or column_name not in stored.get("dims", {})
                   or stored["dims"][column_name]["type"] != level_type
                   or stored["dims"][column_name].get("criterion", {}).get("fixed_pp") != configuration.level_merge_max_pp
                   or stored["dims"][column_name].get("criterion", {}).get("support_floor") != configuration.support_floor]
    configuration.log_action(STEP_01_LABEL, 7, f"{len(configuration.leveled_dims)} dimensions with a generated level "
                                            f"{list(configuration.leveled_dims)} · file {path}: "
                                            + (f"found ({stored.get('created', '?')}), " if stored.get("dims") else "not found, ")
                                            + (f"generating {to_generate}" if to_generate else "every dimension in it, reused"))

    # [8] generate what the file does not have
    if to_generate:
        stored.setdefault("dims", {})
        for column_name in to_generate:
            stored["dims"][column_name] = generate_levels(raw, column_name, configuration.leveled_dims[column_name], configuration)
        stored["created"] = stored.get("created") or datetime.now().isoformat(timespec="seconds")
        stored["updated"] = datetime.now().isoformat(timespec="seconds")
        stored["current_month"] = str(configuration.current_month)
        write_levels_file(path, stored)
        configuration.log_action(STEP_01_LABEL, 8, f"groups generated from the training months and written to {path}")
    else:
        configuration.log_action(STEP_01_LABEL, 8, "nothing to generate")

    # [9] fill the coarse level
    unknown_values = {}
    for column_name in configuration.leveled_dims:
        mapping = stored["dims"][column_name]["mapping"]
        values = raw[column_name].astype(str)
        unknown = sorted(set(values.unique()) - set(mapping))
        if unknown:
            unknown_values[column_name] = unknown
        raw[f"{column_name}{GENERATED_LEVEL_SUFFIX}"] = values.map(mapping).fillna(RESIDUAL_GROUP)
    configuration.log_action(STEP_01_LABEL, 9, "filled: " + ", ".join(configuration.generated_columns))

    # [08] the check
    configuration.log_action(STEP_01_LABEL, 10, "checking the values the file knows")
    configuration.log_check(STEP_01_LABEL, check_log, "no value of the extract is unknown to the file", not unknown_values,
                            failure_detail="values sent to the residual: " + "; ".join(f"{column_name}: {values[:10]}"
                                                                                     for column_name, values in unknown_values.items()),
                            blocking=False)

    # [09] the levels as tables (shown in action 14; the file is the record, they are not written to SQL)
    configuration.log_action(STEP_01_LABEL, 11, "the levels as tables")
    levels_table, values_table, decisions_table = levels_as_tables(stored, configuration)
    return raw, (stored, levels_table, values_table, decisions_table)


def natural_key(value: str) -> list:
    """'2 devices' before '10 devices': numbers compared as numbers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def merge_threshold_pp(global_rate: float, configuration: Config) -> tuple:
    """(threshold in pp, how it was set): the binomial noise of a series at the support floor, or the fixed value."""
    if configuration.level_merge_max_pp is not None:
        return float(configuration.level_merge_max_pp), f"fixed in the Config (level_merge_max_pp = {configuration.level_merge_max_pp})"
    noise = 100 * np.sqrt(global_rate * (1 - global_rate) / configuration.support_floor)
    return float(noise), (f"the binomial noise of a series at the support floor: 100·√(p(1−p)/{configuration.support_floor:.0f}) "
                          f"with p = {global_rate:.2f}")


def generate_levels(raw: pd.DataFrame, column_name: str, level_type: str, configuration: Config) -> dict:
    """The groups of one dimension, from the training months: standardised rate per value, residual for the
    rare ones, neighbours merged while their rates differ by at most the merge threshold. Keeps the evidence:
    every value, every group year by year, every merge decision."""
    due, renewed, period = configuration.pipeline_units_col, configuration.renewed_units_col, configuration.period_col
    training = configuration.role_of_months(raw[period]) == ROLE_TRAIN
    train = raw[training & (raw[due].to_numpy() > 0)].copy()
    train["_value"] = train[column_name].astype(str)
    other_dims = [other for other in configuration.business_mandatory_dims
                  if other != column_name and other not in configuration.generated_columns]
    train["_cell"] = join_columns(train, other_dims) if other_dims else "all"
    cell_rate = train.groupby("_cell")[renewed].sum() / train.groupby("_cell")[due].sum()
    train["_expected"] = train["_cell"].map(cell_rate) * train[due]
    train["_year"] = train[period].map(lambda month: month.year)
    global_rate = float(train[renewed].sum() / train[due].sum())
    threshold_pp, threshold_rule = merge_threshold_pp(global_rate, configuration)

    per_value = train.groupby("_value").agg(observed=(renewed, "sum"), expected=("_expected", "sum"), units_due=(due, "sum"))
    per_value["monthly_support"] = train.groupby(["_value", period])[due].sum().groupby(level=0).median()
    rare = per_value["monthly_support"] < configuration.support_floor

    def standardised(values: list, rows: pd.DataFrame = None) -> float:
        if rows is None:
            block = per_value.loc[values]
            return global_rate * block["observed"].sum() / max(block["expected"].sum(), 1e-9)
        expected = rows["_expected"].sum()
        return float(global_rate * rows[renewed].sum() / expected) if expected > 0 else np.nan

    def by_year(values: list) -> dict:
        block = train[train["_value"].isin(values)]
        return {str(year): round(standardised(values, rows), 4) for year, rows in block.groupby("_year")}

    groups = [[value] for value in per_value.index[~rare]]
    if level_type == "ordinal":
        groups.sort(key=lambda group: natural_key(group[0]))
    else:
        groups.sort(key=lambda group: standardised(group))

    def label(group: list) -> str:
        ordered = sorted(group, key=natural_key)
        return " + ".join(ordered) if len(ordered) <= 3 else f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} values)"

    # merge the closest neighbours while they differ by at most the threshold; every decision is kept
    decisions = []
    while len(groups) > 1:
        differences = [abs(standardised(groups[index]) - standardised(groups[index + 1])) for index in range(len(groups) - 1)]
        closest = int(np.argmin(differences))
        if 100 * differences[closest] > threshold_pp:
            break
        decisions.append({"left": label(groups[closest]), "right": label(groups[closest + 1]),
                          "difference_pp": round(100 * differences[closest], 2), "decision": "merged"})
        groups[closest:closest + 2] = [groups[closest] + groups[closest + 1]]
    for index in range(len(groups) - 1):                       # the neighbours that stay apart, and by how much
        decisions.append({"left": label(groups[index]), "right": label(groups[index + 1]),
                          "difference_pp": round(100 * abs(standardised(groups[index]) - standardised(groups[index + 1])), 2),
                          "decision": "kept apart"})

    mapping, group_evidence, value_evidence = {}, [], []
    final_groups = [(label(group), group) for group in groups]
    if rare.any():
        final_groups.append((RESIDUAL_GROUP, list(per_value.index[rare])))
    for group_label, values in final_groups:
        for value in values:
            mapping[value] = group_label
            value_rate = standardised([value])
            support = float(per_value.loc[value, "monthly_support"])
            value_evidence.append({"value": value, "group": group_label, "units_due": float(per_value.loc[value, "units_due"]),
                                   "monthly_support": support, "rate_std": round(value_rate, 4),
                                   "noise_pp": round(100 * np.sqrt(max(value_rate * (1 - value_rate), 0) / max(support, 1)), 2),
                                   "rate_by_year": by_year([value])})
        block = train[train["_value"].isin(values)]
        group_evidence.append({"group": group_label, "values": sorted(values, key=natural_key),
                               "units_due": float(per_value.loc[values, "units_due"].sum()),
                               "monthly_support": float(block.groupby(period)[due].sum().median()),
                               "rate_std": round(standardised(values), 4), "rate_by_year": by_year(values)})
    value_evidence.sort(key=lambda row: natural_key(row["value"]))
    return {"type": level_type,
            "criterion": {"threshold_pp": round(threshold_pp, 2), "rule": threshold_rule, "fixed_pp": configuration.level_merge_max_pp,
                          "global_rate": round(global_rate, 4), "support_floor": configuration.support_floor},
            "training_months": [str(train[period].min()), str(train[period].max())], "mapping": mapping,
            "groups": group_evidence, "values": value_evidence, "decisions": decisions}


def read_levels_file(path: str) -> dict:
    """The stored groups, or an empty dict when there is no file yet."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_levels_file(path: str, content: dict) -> None:
    """The groups, as indented JSON (readable and editable by hand)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)


def levels_as_tables(stored: dict, configuration: Config) -> tuple:
    """The evidence of the JSON as tables: groups, values and merge decisions (one row each), with the rate
    of every year as a column of its own, readable on screen and in Power BI."""
    group_rows, value_rows, decision_rows = [], [], []
    for column_name in configuration.leveled_dims:
        stored_dimension = stored["dims"][column_name]
        threshold = stored_dimension.get("criterion", {}).get("threshold_pp")
        for group in stored_dimension["groups"]:
            group_rows.append({"dimension": column_name, "group": group["group"], "values": ", ".join(group["values"]),
                               "n_values": len(group["values"]), "units_due": group["units_due"],
                               "monthly_support": group["monthly_support"], "rate_std": group["rate_std"],
                               **{f"rate_{year}": rate for year, rate in group["rate_by_year"].items()}})
        for value in stored_dimension.get("values", []):
            value_rows.append({"dimension": column_name, "value": value["value"], "group": value["group"],
                               "units_due": value["units_due"], "monthly_support": value["monthly_support"],
                               "rate_std": value["rate_std"], "noise_pp": value["noise_pp"],
                               **{f"rate_{year}": rate for year, rate in value["rate_by_year"].items()}})
        for decision in stored_dimension.get("decisions", []):
            decision_rows.append({"dimension": column_name, **decision, "threshold_pp": threshold,
                                  "within_5_pp": "yes" if decision["difference_pp"] <= 5 else "no"})
    return pd.DataFrame(group_rows), pd.DataFrame(value_rows), pd.DataFrame(decision_rows)


def show_the_evidence(stored: dict, levels_table: pd.DataFrame, values_table: pd.DataFrame,
                      decisions_table: pd.DataFrame, configuration: Config) -> None:
    """Action 14: the criterion, then the groups year by year, the values and the merge decisions."""
    for column_name in configuration.leveled_dims:
        criterion = stored["dims"][column_name].get("criterion", {})
        groups = [group for group in stored["dims"][column_name]["groups"] if group["group"] != RESIDUAL_GROUP]
        configuration.log_action(STEP_01_LABEL, 14, f"{column_name} ({stored['dims'][column_name]['type']}): neighbours merge while their "
                                                f"standardised rates differ by ≤ {criterion.get('threshold_pp', '?')} pp — "
                                                f"{criterion.get('rule', '?')} · {len(groups)} groups"
                                                + (" · ONE group: the dimension does not separate the renewal rate beyond that "
                                                   "noise; its level_1 is constant and its pass of the ladder merges nothing (step 07)"
                                                   if len(groups) == 1 else ""))
    rate = next(iter(stored["dims"].values())).get("criterion", {}).get("global_rate", 0.64)
    noise_rows = [{"contracts_a_month": support, "binomial_noise_pp": round(100 * np.sqrt(rate * (1 - rate) / support), 1)}
                  for support in (10, 30, 50, 100, 271, 1000)]
    configuration.logger.doc(f"[{STEP_01_LABEL}] what a difference in pp means: the noise of the monthly rate of a series of n contracts "
                             f"(p = {rate:.2f}); a merge helps the series whose noise is larger than the difference it adds:")
    configuration.show_table(pd.DataFrame(noise_rows))
    configuration.logger.doc(f"[{STEP_01_LABEL}] the groups (rate_std: standardised by cell; rate_<year>: the same, year by year — "
                             f"a group is stable when its years stay close):")
    configuration.show_table(levels_table)
    configuration.logger.doc(f"[{STEP_01_LABEL}] every value (noise_pp: the binomial noise of its own monthly rate; two values whose "
                             f"rates differ by less than their noise cannot be told apart month by month):")
    configuration.show_table(values_table)
    configuration.logger.doc(f"[{STEP_01_LABEL}] the merge decisions, in order (within_5_pp: what the old fixed 5 pp would have said):")
    configuration.show_table(decisions_table)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 02 · APPLY CALENDAR  (until 25-oct-2026: step_02_apply_calendar.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The calendar of the Config, applied to every row of the raw.
#
# Steps 00 and 01 only looked at the raw. This is the first step that changes it:
#   · every row gets its role in time, generated from the calendar of the Config
#     (entrenamiento · examen · proyeccion) and the state of the
#     current month. The raw's own role columns, if any, are ignored.
#   · the raw's renewals are kept, untouched, in two s0_ columns: what the raw said
#     before any change (the core table reconciles against them)
#   · closed months: a null renewal means nobody renewed (a LEFT JOIN with no match),
#     so it becomes 0, counted
#   · the current month and later: the renewals already booked are early results of a
#     month still running, so they are wiped. The future must look like it has not
#     started; the pipeline (what falls due) is kept, it is known.
# The extra measures (reacquisitions, AUVs) are not touched: reacquisitions are not
# forecast (a future extension), and no step uses the AUVs.
#
# Actions (logged as they are done):
#   1. keep the raw's renewals and pipeline in the s0_ columns
#   2. give every row its role and the current-month state
#   3. closed months: read a null renewal as 0
#   4. from the current month on: wipe the renewals already booked; and wipe the pipeline of the
#      1-year licences sold or renewed from the current month on (due 12 months later or more):
#      that event has not happened, or only in part, so its pipeline is not known yet and step 20
#      projects it (the raw keeps it in s0_vencen_*)
#   5. check the calendar and the money                               checks 1-9
#   6. count the checks; stop if any failed
#   7. show the calendar per role, as a table, and the SQL of the calendar per month
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every row has one of the three roles
#    2. training has at least a year of months                      (warning only)
#    3. every exam month has rows                                   (warning only)
#    4. the current month has pipeline                              (warning only)
#    5. the pipeline is unchanged except the rows not known yet (units and USD)
#    6. no 1-year row created from the current month on keeps its pipeline
#    7. the renewals of the closed months are unchanged, a null read as 0
#    8. no row from the current month on keeps a renewal
#    9. the s0_ columns hold the raw's renewals and pipeline, untouched
#
# Output: a copy of the raw with six new columns (rol, es_mes_en_curso, s0_renovados_*, s0_vencen_*).
# The calendar per month is not a table: its SQL over sff_nucleo is printed (report_queries.calendar_query).

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_02_LABEL = "02"
STEP_02_NAME = "APPLY CALENDAR"
STEP_02_PURPOSE = ("give every row its role in time from the calendar of the Config, read a null renewal of a "
                "closed month as 0, and wipe the renewals already booked from the current month on: "
                "the future must look like it has not started")
STEP_02_ACTIONS = ["keep the raw's renewals and pipeline in the s0_ columns",
                "give every row its role and the current-month state",
                "closed months: read a null renewal as 0",
                "from the current month on: wipe the renewals already booked, and the pipeline of 1-year licences "
                "sold or renewed from the current month on (due 12 months later): it is not known yet, it will be projected",
                "check the calendar and the money (checks 1-9)",
                "count the checks; stop if any failed",
                "show the calendar per role, as a table"]
STEP_02_OUTPUT = ("the raw with six new columns (rol, es_mes_en_curso, s0_renovados_unidades, s0_renovados_usd, "
               "s0_vencen_unidades, s0_vencen_usd)")

# ─── named constants ─────────────────────────────────────────────────────────────
# Below a year of training months no seasonal pattern can be learned.
# Money must be conserved to the cent when nothing is supposed to change it.


def apply_calendar(validated: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Give every row its role and current-month state, read null closed renewals as 0,
    wipe the early results of the future; log every check; stop if any fails."""
    configuration.log_step_start(STEP_02_LABEL, STEP_02_NAME, STEP_02_PURPOSE, STEP_02_ACTIONS, STEP_02_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    renewed_columns = [configuration.renewed_units_col, configuration.renewed_usd_col]
    boundaries = configuration.calendar_boundaries()
    calendared = validated.copy()

    # [1] what the raw said, before any change
    calendared[S0_RENEWED_UNITS_COLUMN] = calendared[configuration.renewed_units_col]
    calendared[S0_RENEWED_USD_COLUMN] = calendared[configuration.renewed_usd_col]
    calendared[S0_PIPELINE_UNITS_COLUMN] = calendared[configuration.pipeline_units_col]
    calendared[S0_PIPELINE_USD_COLUMN] = calendared[configuration.pipeline_usd_col]
    configuration.log_action(STEP_02_LABEL, 1, f"renewals and pipeline kept as the raw had them in the s0_ columns")

    # [2] the role of every row and the state of the current month, from the calendar
    calendared[CALENDAR_ROLE_COLUMN] = configuration.role_of_months(calendared[period_column])
    calendared[CURRENT_MONTH_COLUMN] = (calendared[period_column] == boundaries["current"]).astype(int).to_numpy()
    closed_rows = calendared[period_column] < boundaries["current"]
    projection_rows = calendared[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    configuration.log_action(STEP_02_LABEL, 2, f"roles given: {configuration.calendar_description()}")

    # [3] closed months: a null renewal means nobody renewed
    null_closed_rows = closed_rows & calendared[configuration.renewed_units_col].isna()
    null_renewals_read_as_zero = int(null_closed_rows.sum())
    calendared.loc[null_closed_rows, renewed_columns] = 0.0
    for closed_month_column in configuration.closed_month_measure_cols:      # each on its own: it may be null alone
        calendared.loc[closed_rows, closed_month_column] = calendared.loc[closed_rows, closed_month_column].fillna(0.0)
    configuration.log_action(STEP_02_LABEL, 3, f"{null_renewals_read_as_zero:,} null renewals of closed months read as 0")

    # [4] the current month and later: wipe the renewals already booked
    early_results = calendared.loc[projection_rows, renewed_columns].fillna(0)
    rows_with_early_results = int((early_results != 0).any(axis=1).sum())
    wiped_units = float(early_results[configuration.renewed_units_col].sum())
    wiped_usd = float(early_results[configuration.renewed_usd_col].sum())
    calendared.loc[projection_rows, renewed_columns] = np.nan
    for closed_month_column in configuration.closed_month_measure_cols:      # only known at close: partial too, wiped too
        calendared.loc[projection_rows, closed_month_column] = np.nan
    configuration.log_action(STEP_02_LABEL, 4, f"{rows_with_early_results:,} rows from {boundaries['current']} on had "
                                            f"renewals already booked: wiped {wiped_units:,.0f} units · ${wiped_usd:,.0f}")

    # [4b] the pipeline that the current month and later will create: a 1-year licence due in
    #      month m was sold or renewed in m − 12; if m − 12 is the current month or later, that
    #      event has not happened (or only in part): its pipeline is wiped and step 20 projects it
    not_known_yet = ((calendared[period_column] >= boundaries["current"] + configuration.one_year_term_months)
                     & is_one_year(calendared, configuration))
    wiped_pipeline_units = float(calendared.loc[not_known_yet, configuration.pipeline_units_col].sum())
    wiped_pipeline_usd = float(calendared.loc[not_known_yet, configuration.pipeline_usd_col].sum())
    calendared.loc[not_known_yet, [configuration.pipeline_units_col, configuration.pipeline_usd_col]] = 0.0
    wiped_per_term = (calendared.loc[not_known_yet, configuration.term_column].astype(str).value_counts().to_dict()
                      if configuration.term_column else "no term_column: every licence is treated as 1-year")
    configuration.log_action(STEP_02_LABEL, "4b", f"{int(not_known_yet.sum()):,} rows of 1-year licences due from "
                                            f"{boundaries['current'] + configuration.one_year_term_months} on (sold or renewed from {boundaries['current']} on): "
                                            f"pipeline wiped {wiped_pipeline_units:,.0f} units · ${wiped_pipeline_usd:,.0f} "
                                            f"(it will be projected) · rows wiped per term: {wiped_per_term}")

    # [5] the checks
    configuration.log_action(STEP_02_LABEL, 5, "checking the calendar and the money")
    months_per_role = months_by_role(calendared, configuration)
    check_roles_and_calendar(calendared, configuration, check_log, months_per_role, boundaries)
    check_money_conserved_by_the_calendar(validated, calendared, configuration, check_log, closed_rows, projection_rows,
                          null_renewals_read_as_zero, rows_with_early_results, wiped_units, wiped_usd, not_known_yet)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_02_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_02_LABEL, STEP_02_NAME, check_log)

    # [7] what the calendar looks like on the raw
    log_calendar_report(calendared, configuration, months_per_role)
    return calendared


def months_by_role(calendared: pd.DataFrame, configuration: Config) -> dict:
    """The sorted months of the raw that fall in each role."""
    months_per_role = {}
    for role in ROLES_IN_ORDER:
        role_months = calendared.loc[calendared[CALENDAR_ROLE_COLUMN] == role, configuration.period_col].unique()
        months_per_role[role] = sorted(role_months)
    return months_per_role


def check_roles_and_calendar(calendared: pd.DataFrame, configuration: Config, check_log: list,
                             months_per_role: dict, boundaries: dict) -> None:
    """Checks 1 to 4: roles assigned, enough training, exam with data, pipeline this month."""
    period_column = configuration.period_col

    # [1] every row has one of the three roles
    unknown_roles = sorted(set(calendared[CALENDAR_ROLE_COLUMN]) - set(ROLES_IN_ORDER))
    rows_per_role = calendared[CALENDAR_ROLE_COLUMN].value_counts()
    configuration.log_check(STEP_02_LABEL, check_log, "every row has one of the three roles", not unknown_roles,
                            failure_detail=f"unknown roles: {unknown_roles}",
                            context=" · ".join(f"{role} {int(rows_per_role.get(role, 0)):,}" for role in ROLES_IN_ORDER))

    # [2] a year of training at least
    training_months = len(months_per_role[ROLE_TRAIN])
    configuration.log_check(STEP_02_LABEL, check_log, f"training has at least {configuration.min_training_months} months",
                            training_months >= configuration.min_training_months,
                            failure_detail=f"only {training_months} training months: no seasonal pattern can be learned",
                            context=f"{training_months} months", blocking=False)

    # [3] every exam month has rows (a month with no rows is a month the exam cannot score)
    exam_months = pd.period_range(boundaries["test_start"], boundaries["current"] - 1, freq="M")
    exam_months_without_rows = [str(month) for month in exam_months if month not in set(months_per_role[ROLE_TEST])]
    configuration.log_check(STEP_02_LABEL, check_log, "every exam month has rows", not exam_months_without_rows,
                            failure_detail=f"exam months with no row: {exam_months_without_rows}",
                            context=f"{len(exam_months)} exam months", blocking=False)

    # [4] the current month has pipeline: without it there is nothing to forecast this month
    current_month_rows = calendared[CURRENT_MONTH_COLUMN] == 1
    current_pipeline_units = float(calendared.loc[current_month_rows, configuration.pipeline_units_col].sum())
    configuration.log_check(STEP_02_LABEL, check_log, f"the current month ({boundaries['current']}) has pipeline",
                            current_pipeline_units > 0,
                            failure_detail=f"no units fall due in {boundaries['current']}",
                            context=f"{current_pipeline_units:,.0f} units due", blocking=False)


def check_money_conserved_by_the_calendar(validated: pd.DataFrame, calendared: pd.DataFrame, configuration: Config,
                          check_log: list, closed_rows: pd.Series, projection_rows: pd.Series,
                          null_renewals_read_as_zero: int, rows_with_early_results: int,
                          wiped_units: float, wiped_usd: float, not_known_yet: pd.Series) -> None:
    """Checks 5 to 9: the step only changed what it was meant to change."""
    renewed_columns = [configuration.renewed_units_col, configuration.renewed_usd_col]

    # [5] the pipeline is unchanged, except the rows whose event has not happened yet
    pipeline_columns = [configuration.pipeline_units_col, configuration.pipeline_usd_col]
    pipeline_differences = {column_name: float(calendared.loc[~not_known_yet, column_name].sum()
                                               - validated.loc[~not_known_yet, column_name].sum())
                            for column_name in pipeline_columns}
    configuration.log_check(STEP_02_LABEL, check_log, "the pipeline is unchanged except the rows not known yet (units and USD)",
                            all(abs(difference) <= configuration.money_tolerance for difference in pipeline_differences.values()),
                            failure_detail=f"the pipeline changed: {pipeline_differences}")
    pipeline_left = float(calendared.loc[not_known_yet, pipeline_columns].abs().sum().sum())
    configuration.log_check(STEP_02_LABEL, check_log, "no 1-year row created from the current month on keeps its pipeline",
                            pipeline_left == 0, failure_detail=f"{pipeline_left:,.0f} left in rows not known yet",
                            context=f"{int(not_known_yet.sum()):,} rows wiped: "
                                    f"${validated.loc[not_known_yet, configuration.pipeline_usd_col].sum():,.0f} "
                                    f"(kept in {S0_PIPELINE_USD_COLUMN})")

    # [6] the renewals of the closed months are unchanged (a null counts as 0 before and after)
    closed_differences = {column_name: float(calendared.loc[closed_rows, column_name].sum()
                                             - validated.loc[closed_rows, column_name].fillna(0).sum())
                          for column_name in renewed_columns}
    configuration.log_check(STEP_02_LABEL, check_log, "the renewals of the closed months are unchanged, a null read as 0",
                            all(abs(difference) <= configuration.money_tolerance for difference in closed_differences.values())
                            and not calendared.loc[closed_rows, renewed_columns].isna().any().any(),
                            failure_detail=f"closed renewals changed: {closed_differences}",
                            context=f"{null_renewals_read_as_zero:,} null renewals read as 0")

    # [7] nothing booked survives from the current month on
    renewals_left = int(calendared.loc[projection_rows, renewed_columns].notna().any(axis=1).sum())
    configuration.log_check(STEP_02_LABEL, check_log, "no row from the current month on keeps a renewal", renewals_left == 0,
                            failure_detail=f"{renewals_left:,} projection rows still carry a renewal",
                            context=f"{rows_with_early_results:,} rows with early results wiped: "
                                    f"{wiped_units:,.0f} units · ${wiped_usd:,.0f}")

    # [8] the s0_ columns are the raw's renewals, untouched
    s0_intact = (calendared[S0_RENEWED_UNITS_COLUMN].equals(validated[configuration.renewed_units_col])
                 and calendared[S0_RENEWED_USD_COLUMN].equals(validated[configuration.renewed_usd_col])
                 and calendared[S0_PIPELINE_UNITS_COLUMN].equals(validated[configuration.pipeline_units_col])
                 and calendared[S0_PIPELINE_USD_COLUMN].equals(validated[configuration.pipeline_usd_col]))
    configuration.log_check(STEP_02_LABEL, check_log, "the s0_ columns hold the raw's renewals and pipeline, untouched", s0_intact,
                            failure_detail="the s0_ columns differ from the raw")


def log_calendar_report(calendared: pd.DataFrame, configuration: Config, months_per_role: dict) -> None:
    """Action 7: per role, its months, rows and money, as a table; and the SQL of the calendar per month."""
    configuration.log_action(STEP_02_LABEL, 7, "per role: months, rows, what fell due and what was renewed "
                                            "(the projection shows no renewal: it was wiped)")
    summary_rows = []
    for role in ROLES_IN_ORDER:
        role_rows = calendared[calendared[CALENDAR_ROLE_COLUMN] == role]
        role_months = months_per_role[role]
        renewed_units = role_rows[configuration.renewed_units_col].sum(min_count=1)
        pipeline_units = role_rows[configuration.pipeline_units_col].sum()
        summary_rows.append({
            CALENDAR_ROLE_COLUMN: role,
            "meses": len(role_months),
            "desde": str(role_months[0]) if role_months else "",
            "hasta": str(role_months[-1]) if role_months else "",
            "filas": len(role_rows),
            "unidades_vencen": pipeline_units,
            "unidades_renovadas": renewed_units,
            "tasa_unidades": renewed_units / pipeline_units if pipeline_units > 0 else np.nan,
        })
    configuration.show_table(pd.DataFrame(summary_rows))
    configuration.show_query(STEP_02_LABEL, "the calendar per role", roles_query(configuration))
    configuration.show_query(STEP_02_LABEL, "the calendar per month", calendar_query(configuration))


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 03 · FINE TABLE  (until 25-oct-2026: step_03_fine_table.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The fine table: every row of the raw, with the ids that name it.
#
# The forecast of a future row is: what falls due × renewal rate × uplift. Each row is
# therefore named by the two contexts it takes its numbers from:
#   · RATE SERIES (fs_id): its mandatory, timevarying and extra_renovacion values. Every
#     distinct combination is one series with its own renewal rate over time.
#   · FORECAST UNIT (fu_id): its series in its month (fs_id | month). The rate is
#     predicted per unit.
#   · UPLIFT CELL (uplift_cell_id): its uplift mandatory dims, its extra_revalorizacion
#     values and its discount bucket (region, product, 60-70 %...). The price at which it
#     renews is estimated per cell, over every month of the cell.
# The discount bucket is DERIVED here from the exact discount with the edges of the Config
# (a null discount goes to "sin_dato"). A fine row is its unit, its extra_revalorizacion
# values and its exact discount (the extract comes one row per exact discount): the grain is
# checked on them. Nothing is aggregated here and no measure changes; step 04 adds the fine
# rows of each unit. The ids are text: the core and the BI join on them (no hash keys).
#
# Actions (logged as they are done):
#   1. build fs_id from the rate series columns
#   2. build fu_id: fs_id | month
#   3. derive the discount bucket from the exact discount
#   4. build uplift_cell_id from the uplift cell columns
#   5. check the bucket, the grain and that nothing else changed            checks 1-3
#   6. keep the fine table in memory (sff_nucleo writes every fine row)
#   7. count the checks; stop if any failed
#   8. show the rows per discount bucket and a few rows with their ids, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every exact discount falls in one bucket (a null one in "sin_dato")
#    2. one row per forecast unit, revaluation values and exact discount (the grain of the raw)
#    3. same rows, same order, no column of step 02 changed
#
# Output: the raw of step 02 with the discount bucket and three id columns (fs_id, fu_id, uplift_cell_id);
# not written to SQL (sff_nucleo carries every fine row).

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_03_LABEL = "03"
STEP_03_NAME = "FINE TABLE"
STEP_03_PURPOSE = ("name every row of the raw by the contexts it takes its numbers from: its rate series (fs_id), its "
                "forecast unit (series in its month, fu_id) and its uplift cell (price context, uplift_cell_id); "
                "nothing is aggregated")
STEP_03_ACTIONS = ["build fs_id from the rate series columns",
                "build fu_id: fs_id | month",
                "derive the discount bucket from the exact discount",
                "build uplift_cell_id from the uplift cell columns",
                "check the bucket, the grain and that nothing else changed (checks 1-3)",
                "keep the fine table in memory (sff_nucleo writes every fine row)",
                "count the checks; stop if any failed",
                "show the rows per discount bucket and a few rows with their ids, as tables"]
STEP_03_OUTPUT = ("the raw with the discount bucket and three id columns (one row per raw row; in memory, "
               "sff_nucleo writes it)")

# ─── named constants ─────────────────────────────────────────────────────────────
# The id columns this step adds, and the temporary name of a fine row inside the grain check.
ADDED_ID_COLUMNS = (SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN)
FINE_ROW_GROUP = "_fila"


def build_fine_table(calendared: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Add the ids that name every row; check the grain."""
    configuration.log_step_start(STEP_03_LABEL, STEP_03_NAME, STEP_03_PURPOSE, STEP_03_ACTIONS, STEP_03_OUTPUT)
    check_log = []
    fine_table = calendared.copy()

    # [1] the rate series of every row
    fine_table[SERIES_ID_COLUMN] = join_columns(fine_table, configuration.rate_series_columns)
    configuration.log_action(STEP_03_LABEL, 1, f"fs_id from {len(configuration.rate_series_columns)} columns "
                                            f"({', '.join(configuration.rate_series_columns)}): "
                                            f"{fine_table[SERIES_ID_COLUMN].nunique():,} series")

    # [2] the forecast unit: the series in its month
    fine_table[UNIT_ID_COLUMN] = (fine_table[SERIES_ID_COLUMN] + ID_FIELD_SEPARATOR
                                  + fine_table[configuration.period_col].astype(str))
    configuration.log_action(STEP_03_LABEL, 2, f"fu_id = fs_id | month: {fine_table[UNIT_ID_COLUMN].nunique():,} forecast units")

    # [3] the discount bucket, from the exact discount (the Config's edges; null → sin_dato)
    if configuration.discount_value_column:
        fine_table[configuration.discount_bucket_column] = discount_bucket_labels(
            fine_table[configuration.discount_value_column], configuration.discount_bucket_edges)
        unknown_share = (fine_table[configuration.discount_bucket_column] == UNKNOWN_DISCOUNT_BUCKET).mean()
        configuration.log_action(STEP_03_LABEL, 3, f"{configuration.discount_bucket_column} from "
                                                f"{configuration.discount_value_column}: "
                                                f"{fine_table[configuration.discount_bucket_column].nunique()} buckets; "
                                                f"{unknown_share:.1%} of the rows '{UNKNOWN_DISCOUNT_BUCKET}'")
    else:
        configuration.log_action(STEP_03_LABEL, 3, "no exact discount declared: no bucket")

    # [4] the uplift cell: the price context of the row
    fine_table[UPLIFT_CELL_ID_COLUMN] = join_columns(fine_table, configuration.uplift_cell_columns)
    configuration.log_action(STEP_03_LABEL, 4, f"uplift_cell_id from {', '.join(configuration.uplift_cell_columns)}: "
                                            f"{fine_table[UPLIFT_CELL_ID_COLUMN].nunique():,} uplift cells")

    # [5] the checks
    configuration.log_action(STEP_03_LABEL, 5, "checking the bucket, the grain and that nothing else changed")
    check_discount_bucket(fine_table, configuration, check_log)
    check_grain(fine_table, configuration, check_log)
    new_columns = [column_name for column_name in fine_table.columns if column_name not in calendared.columns]
    expected_new_columns = len(ADDED_ID_COLUMNS) + (1 if configuration.discount_value_column else 0)
    configuration.log_check(STEP_03_LABEL, check_log, "same rows, same order, no column of step 02 changed",
                            fine_table[list(calendared.columns)].equals(calendared)
                            and len(new_columns) == expected_new_columns,
                            failure_detail="the fine table changed the rows or the columns it received",
                            context=f"{len(fine_table):,} rows · {len(new_columns)} columns added")

    # [6] the fine table is NOT written to SQL: sff_nucleo carries every fine row with its ids (step 24), and writing
    #     the 1M rows here cost 3.5 minutes of the real run for a table nobody read
    configuration.log_action(STEP_03_LABEL, 6, "the fine table stays in memory (every fine row is written in sff_nucleo, step 24)")

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_03_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_03_LABEL, STEP_03_NAME, check_log)

    # [8] the rows per discount bucket, and a few rows with their ids
    if configuration.discount_value_column:
        configuration.log_action(STEP_03_LABEL, 8, "rows and money due per discount bucket:")
        bucket_column = configuration.discount_bucket_column
        configuration.show_table(fine_table.groupby(bucket_column, dropna=False)
                                 .agg(filas=(bucket_column, "size"),
                                      descuento_min=(configuration.discount_value_column, "min"),
                                      descuento_max=(configuration.discount_value_column, "max"),
                                      usd_vence=(configuration.pipeline_usd_col, "sum"))
                                 .reset_index())
    configuration.logger.doc(f"[{STEP_03_LABEL}] a few rows with their ids:")
    configuration.show_table(fine_table[[configuration.period_col, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN]
                                        + configuration.fine_row_columns].head(configuration.example_rows_shown))
    return fine_table


def fine_row_values(fine_table: pd.DataFrame, configuration: Config) -> pd.Series:
    """The extra_revalorizacion values and the exact discount of every row, joined ("na"
    when there is none): with the unit, they are what makes a fine row unique."""
    if configuration.fine_row_columns:
        return join_columns(fine_table, configuration.fine_row_columns)
    return pd.Series(NO_REVALUATION_VALUES, index=fine_table.index)


def check_discount_bucket(fine_table: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 1: every exact discount has a bucket; only a null discount is 'sin_dato'."""
    if not configuration.discount_value_column:
        configuration.log_not_evaluated(STEP_03_LABEL, check_log, "every exact discount falls in one bucket",
                                        "no exact discount declared")
        return
    buckets = fine_table[configuration.discount_bucket_column]
    discount_values = fine_table[configuration.discount_value_column]
    without_bucket = fine_table[buckets.isna()]
    wrongly_unknown = int(((buckets == UNKNOWN_DISCOUNT_BUCKET) & discount_values.notna()).sum())
    configuration.log_check(STEP_03_LABEL, check_log, f"every exact discount falls in one bucket "
                                                   f"(a null one in '{UNKNOWN_DISCOUNT_BUCKET}')",
                            without_bucket.empty and wrongly_unknown == 0,
                            failure_detail=f"{len(without_bucket):,} rows without a bucket · {wrongly_unknown:,} known "
                                           f"discounts labelled '{UNKNOWN_DISCOUNT_BUCKET}'",
                            context=f"edges {configuration.discount_bucket_edges} %",
                            examples=without_bucket[[configuration.period_col, UNIT_ID_COLUMN,
                                                     configuration.discount_value_column]])


def check_grain(fine_table: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 2: no two raw rows are the same unit with the same revaluation values. If they
    were, the raw would not be at the grain the Config declares and every later sum would
    mix them. When it fails, the check says WHY: which columns of the raw take different
    values inside the repeated rows (a column missing from the taxonomy), or that only the
    money differs (the extract is below the declared grain and must be aggregated). The
    first repeated group is shown whole, with those columns."""
    grain_columns = [UNIT_ID_COLUMN] + configuration.fine_row_columns
    repeated_mask = fine_table.duplicated(grain_columns, keep=False)
    repeated_rows_count = int(fine_table.duplicated(grain_columns).sum())
    check_description = "one row per forecast unit, revaluation values and exact discount (the grain of the raw)"
    if repeated_rows_count == 0:
        configuration.log_check(STEP_03_LABEL, check_log, check_description, True,
                                context=f"{len(fine_table):,} rows = {len(fine_table):,} distinct fine rows")
        return

    # which columns separate the repeated rows, and how many repeated groups each one explains (a repeated group:
    # the rows of the same fine row, named by its unit, its revaluation values and its exact discount)
    repeated_rows = fine_table[repeated_mask].copy()
    repeated_rows[FINE_ROW_GROUP] = (repeated_rows[UNIT_ID_COLUMN] + COMBINED_ID_SEPARATOR
                                     + fine_row_values(repeated_rows, configuration))
    repeated_groups = int(repeated_rows[FINE_ROW_GROUP].nunique())
    splitting_columns = columns_that_split_the_grain(repeated_rows, grain_columns, configuration)
    groups_explained = groups_explained_by(repeated_rows, list(splitting_columns))
    if splitting_columns:
        cause_text = ("columns of the raw that differ inside the repeated rows (repeated groups where they differ): "
                      + ", ".join(f"{column_name} ({group_count:,})" for column_name, group_count in splitting_columns.items())
                      + f"; together they explain {groups_explained:,} of the {repeated_groups:,} repeated groups"
                      + (f", the other {repeated_groups - groups_explained:,} differ only in the money"
                         if groups_explained < repeated_groups else "")
                      + " → declare them as dimensions, or aggregate the extract without them")
    else:
        cause_text = (f"no column but the money differs inside the {repeated_groups:,} repeated groups: the extract is "
                      f"below the declared grain (e.g. one row per transaction) → aggregate it to the declared dimensions")

    first_row_group = repeated_rows.iloc[0][FINE_ROW_GROUP]
    first_group = repeated_rows[repeated_rows[FINE_ROW_GROUP] == first_row_group]
    shown_columns = ([configuration.period_col, UNIT_ID_COLUMN] + configuration.fine_row_columns
                     + [column_name for column_name in splitting_columns if column_name not in configuration.fine_row_columns]
                     + configuration.core_measures)
    configuration.log_check(STEP_03_LABEL, check_log, check_description, False,
                            failure_detail=f"{repeated_rows_count:,} rows repeat a fine row already seen "
                                           f"({repeated_mask.sum():,} rows in {repeated_groups:,} groups). {cause_text}",
                            examples=first_group[shown_columns])

    # the same problem, in SQL, to reproduce it on the extract
    configuration.logger.doc(f"[{STEP_03_LABEL}] to reproduce it in SQL (the declared grain, the groups with more than "
                             f"one row, and what differs inside them):")
    print(grain_reproduction_sql(configuration, list(splitting_columns)))


def groups_explained_by(repeated_rows: pd.DataFrame, splitting_columns: list) -> int:
    """How many repeated groups have at least one of the splitting columns taking more than one value."""
    if not splitting_columns:
        return 0
    varies_somewhere = (repeated_rows.groupby(FINE_ROW_GROUP)[splitting_columns].nunique(dropna=False) > 1).any(axis=1)
    return int(varies_somewhere.sum())


def grain_reproduction_sql(configuration: Config, splitting_columns: list) -> str:
    """A SQL query, built from the Config, that finds the groups of the declared grain with
    more than one row on the extract, and says how many of them each splitting column explains.
    The extract is taken from configuration.raw_extract_sql (a placeholder when not declared)."""
    grain_columns = ([configuration.period_col] + configuration.dimension_columns
                     + [column_name for column_name in configuration.fine_row_columns
                        if column_name not in configuration.dimension_columns])
    source = configuration.raw_extract_sql or "<la query o la tabla de tu extracto>"
    select_grain = ",\n           ".join(grain_columns)
    splitting_counts = "".join(f"\n           COUNT(DISTINCT {column_name}) AS distintos_{column_name},"
                               f"\n           MIN({column_name}) AS min_{column_name}, MAX({column_name}) AS max_{column_name},"
                               for column_name in splitting_columns)
    explained_sums = "".join(f",\n       SUM(CASE WHEN distintos_{column_name} > 1 THEN 1 ELSE 0 END) AS grupos_que_separa_{column_name}"
                             for column_name in splitting_columns)
    return (f"WITH raw AS (\n    {source}\n),\n"
            f"grupos AS (\n    SELECT {select_grain},\n           COUNT(*) AS filas,{splitting_counts}"
            f"\n           SUM({configuration.pipeline_units_col}) AS unidades_vencen"
            f"\n    FROM raw\n    GROUP BY {select_grain}\n    HAVING COUNT(*) > 1\n)\n"
            f"SELECT COUNT(*) AS grupos_repetidos,\n       SUM(filas) AS filas_implicadas{explained_sums}\nFROM grupos;\n"
            f"-- to see the groups: SELECT * FROM grupos ORDER BY filas DESC;\n"
            f"-- COUNT(DISTINCT) ignores NULL: a group with a value and a NULL is not counted as split")


def columns_that_split_the_grain(repeated_rows: pd.DataFrame, grain_columns: list, configuration: Config) -> dict:
    """The raw columns (not measures, not the ids added here) that take more than one value
    inside at least one repeated fine row, with how many groups: they are what makes the
    raw finer than the declared dimensions. Sorted by groups, most first."""
    measure_columns = (set(configuration.core_measures) | set(configuration.extra_measure_cols)
                       | set(configuration.closed_month_measure_cols) | set(configuration.maturation_columns))
    added_columns = set(ADDED_ID_COLUMNS) | {FINE_ROW_GROUP, configuration.discount_bucket_column}
    candidate_columns = [column_name for column_name in repeated_rows.columns
                         if column_name not in measure_columns and column_name not in added_columns
                         and column_name not in grain_columns and not column_name.startswith("s0_")]
    grouped = repeated_rows.groupby(FINE_ROW_GROUP, sort=False)
    splitting_columns = {}
    for column_name in candidate_columns:
        groups_where_it_varies = int((grouped[column_name].nunique(dropna=False) > 1).sum())
        if groups_where_it_varies:
            splitting_columns[column_name] = groups_where_it_varies
    return dict(sorted(splitting_columns.items(), key=lambda item: -item[1]))


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 04 · FORECAST UNITS  (until 25-oct-2026: step_04_forecast_units.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The forecast units: one row per rate series in its month.
#
# The fine table has one row per unit AND revaluation values. The renewal rate does not
# depend on them (they only change the price), so the rate is
# studied and forecast per FORECAST UNIT: the fine rows of a unit are added up here.
# Measures are SUMMED, never averaged: every rate later is Σ renewed / Σ due, computed from
# these sums. A unit whose renewals are all null (the projection) keeps them null.
#
# Actions (logged as they are done):
#   1. group the fine rows by forecast unit
#   2. take the description of each unit (series, month, role) from its first row
#   3. add up the four measures and count the fine rows of each unit
#   4. check the units and that the money is conserved                checks 1-3
#   5. count the checks; stop if any failed
#   6. show the units per role and the fine rows per unit, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the four measures are conserved: Σ fine = Σ units
#    2. renewals are numbers in closed months and null from the current month on
#    3. every unit has something falling due in the extract (the ones the calendar wiped on purpose are
#       counted apart)                                                  (warning only)
#
# Output: the forecast units (one row per fu_id: ids, series columns, month, role, current-month state,
# number of fine rows, the four measures summed), in memory. The time_series rows were separated in
# step 01: every unit here is of the renewal universe.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_04_LABEL = "04"
STEP_04_NAME = "FORECAST UNITS"
STEP_04_PURPOSE = ("add up the fine rows of every forecast unit (a rate series in its month): the renewal rate is "
                "studied per unit, because the revaluation values change the price, not the rate")
STEP_04_ACTIONS = ["group the fine rows by forecast unit",
                "take the description of each unit (series, month, role) from its first row",
                "add up the four measures and count the fine rows of each unit",
                "check the units and that the money is conserved (checks 1-3)",
                "count the checks; stop if any failed",
                "show the units per role and the fine rows per unit, as tables"]
STEP_04_OUTPUT = "one row per forecast unit with its four measures summed (in memory)"

# ─── named constants ─────────────────────────────────────────────────────────────
# Money is conserved when the difference is below a cent or a billionth of the total
# (float sums over a million rows differ in the last digits).


def build_forecast_units(fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Add up the fine rows of every forecast unit; check conservation."""
    configuration.log_step_start(STEP_04_LABEL, STEP_04_NAME, STEP_04_PURPOSE, STEP_04_ACTIONS, STEP_04_OUTPUT)
    check_log = []
    measures = configuration.core_measures

    # [1] the fine rows of each unit
    grouped_by_unit = fine_table.groupby(UNIT_ID_COLUMN, sort=False)
    configuration.log_action(STEP_04_LABEL, 1, f"{len(fine_table):,} fine rows grouped into "
                                            f"{grouped_by_unit.ngroups:,} forecast units")

    # [2] the description of each unit: constant inside it (they come from the series and the month)
    description_columns = ([SERIES_ID_COLUMN] + configuration.rate_series_columns
                           + [configuration.period_col, CALENDAR_ROLE_COLUMN, CURRENT_MONTH_COLUMN])
    forecast_units = grouped_by_unit[description_columns].first()
    configuration.log_action(STEP_04_LABEL, 2, f"each unit described by {len(description_columns)} columns "
                                            f"(series, month, role)")

    # [3] the measures, summed (a unit whose values are all null stays null), and the fine rows
    forecast_units[measures] = grouped_by_unit[measures].sum(min_count=1)
    forecast_units[FINE_ROWS_COLUMN] = grouped_by_unit.size()
    forecast_units = forecast_units.reset_index()
    forecast_units = forecast_units[[UNIT_ID_COLUMN] + description_columns + [FINE_ROWS_COLUMN] + measures]
    configuration.log_action(STEP_04_LABEL, 3, f"measures summed; {forecast_units[FINE_ROWS_COLUMN].mean():.2f} "
                                            f"fine rows per unit on average")

    # [4] the checks
    configuration.log_action(STEP_04_LABEL, 4, "checking the units and that the money is conserved")
    check_money_conserved_into_units(fine_table, forecast_units, configuration, check_log)
    check_renewals_by_month(forecast_units, configuration, check_log)
    # a unit at 0 units due: either the calendar wiped it on purpose (step 02: the pipeline of a 1-year
    # licence sold or renewed from the current month on; step 20 projects it), or the extract brings it at 0
    units_without_pipeline = forecast_units[forecast_units[configuration.pipeline_units_col] == 0]
    if S0_PIPELINE_UNITS_COLUMN in fine_table.columns:
        due_in_extract = fine_table.groupby(UNIT_ID_COLUMN)[S0_PIPELINE_UNITS_COLUMN].sum()
        wiped_on_purpose = units_without_pipeline[UNIT_ID_COLUMN].map(due_in_extract).fillna(0) > 0
    else:
        wiped_on_purpose = pd.Series(False, index=units_without_pipeline.index)
    zero_in_extract = units_without_pipeline[~wiped_on_purpose]
    configuration.log_action(STEP_04_LABEL, 4, f"units at 0 units due: {int(wiped_on_purpose.sum()):,} wiped on purpose by the calendar "
                                            f"(step 02, projected in step 20) · {len(zero_in_extract):,} at 0 in the extract")
    configuration.log_check(STEP_04_LABEL, check_log, "every unit has something falling due in the extract", zero_in_extract.empty,
                            failure_detail=f"{len(zero_in_extract):,} units at 0 units due in the extract itself: their rate is undefined",
                            blocking=False,
                            examples=zero_in_extract[[UNIT_ID_COLUMN, configuration.period_col] + measures])

    # [5] the count of the checks; stop if anything failed
    configuration.log_action(STEP_04_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_04_LABEL, STEP_04_NAME, check_log)

    # [6] what the units look like
    log_units_report(forecast_units, configuration)
    return forecast_units


def check_money_conserved_into_units(fine_table: pd.DataFrame, forecast_units: pd.DataFrame, configuration: Config,
                          check_log: list) -> None:
    """Check 1: not a cent appears or disappears when the fine rows are added up."""
    differences = {}
    for measure_column in configuration.core_measures:
        fine_total = float(fine_table[measure_column].sum())
        units_total = float(forecast_units[measure_column].sum())
        tolerance = max(configuration.money_tolerance, configuration.money_relative_tolerance * abs(fine_total))
        if abs(units_total - fine_total) > tolerance:
            differences[measure_column] = units_total - fine_total
    pipeline_usd = float(forecast_units[configuration.pipeline_usd_col].sum())
    configuration.log_check(STEP_04_LABEL, check_log, "the four measures are conserved: Σ fine = Σ units", not differences,
                            failure_detail=f"measures that changed when summed (units − fine): {differences}",
                            context=f"${pipeline_usd:,.0f} due in both tables")


def check_renewals_by_month(forecast_units: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Check 2: after step 02, a closed unit has a renewal (0 or more) and a projection unit
    has none (null). A null closed unit or a projection unit with a number means the
    fine table lost the calendar on the way."""
    renewed_units = forecast_units[configuration.renewed_units_col]
    projection_units = forecast_units[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    null_closed_units = int((~projection_units & renewed_units.isna()).sum())
    booked_projection_units = int((projection_units & renewed_units.notna()).sum())
    configuration.log_check(STEP_04_LABEL, check_log,
                            "renewals are numbers in closed months and null from the current month on",
                            null_closed_units == 0 and booked_projection_units == 0,
                            failure_detail=f"{null_closed_units:,} closed units with a null renewal · "
                                           f"{booked_projection_units:,} projection units with a renewal",
                            context=f"{int((~projection_units).sum()):,} closed units · "
                                    f"{int(projection_units.sum()):,} projection units")


def log_units_report(forecast_units: pd.DataFrame, configuration: Config) -> None:
    """Action 6: units per role and fine rows per unit, as tables."""
    configuration.log_action(STEP_04_LABEL, 6, "units per role (series = distinct rate series with a unit in that role):")
    per_role_rows = []
    for role in ROLES_IN_ORDER:
        role_units = forecast_units[forecast_units[CALENDAR_ROLE_COLUMN] == role]
        per_role_rows.append({CALENDAR_ROLE_COLUMN: role, "unidades": len(role_units), "series": role_units[SERIES_ID_COLUMN].nunique(),
                              "unidades_vencen": role_units[configuration.pipeline_units_col].sum(),
                              "usd_vence": role_units[configuration.pipeline_usd_col].sum()})
    configuration.show_table(pd.DataFrame(per_role_rows))

    configuration.logger.doc(f"[{STEP_04_LABEL}] fine rows per unit (how many fine rows were added into one unit):")
    fine_rows_count = (forecast_units[FINE_ROWS_COLUMN].value_counts().sort_index()
                          .rename_axis(FINE_ROWS_COLUMN).reset_index(name="unidades"))
    configuration.show_table(fine_rows_count)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 05 · SERIES AND ROUTES  (until 25-oct-2026: step_05_series_routes.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The series: which months each one has, and how it will be treated.
#
# A rate series is not forecast the same way: it depends on the months it has:
#   · COVERAGE: the roles of its months, in time order (e.g. entrenamiento+examen+proyeccion)
#   · ROUTE, from the coverage:
#       predecible     closed months AND something to predict: its rate is estimated
#       solo_historia  nothing to predict: kept, it lends its history to its relatives
#       solo_futuro    nothing to learn from: predicted from its relatives
# Nothing is filtered: every series is labelled, none is dropped. Every series is of the renewal
# universe: the time_series rows were separated in step 01.
#
# Actions (logged as they are done):
#   1. group the forecast units by series
#   2. count the months of each series in each role; first and last month
#   3. coverage and route of every series
#   4. the money each series has to predict (due from the current month on)
#   5. check the series                                               checks 1-2
#   6. count the checks; stop if any failed
#   7. show the series and the money per route, as a table
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every forecast unit belongs to a series of the table, and every series has a route
#    2. every series with something to predict has pipeline to predict (warning only)
#
# Output: one row per series (id, series columns, months per role, first and last month, coverage,
# route, units and USD to predict), in memory.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_05_LABEL = "05"
STEP_05_NAME = "SERIES AND ROUTES"
STEP_05_PURPOSE = ("describe every rate series by the months it has (coverage) and decide how it will be treated "
                "(route): estimated, used only as history, or predicted from its relatives; nothing is dropped")
STEP_05_ACTIONS = ["group the forecast units by series",
                "count the months of each series in each role; first and last month",
                "coverage and route of every series",
                "the money each series has to predict (due from the current month on)",
                "check the series (checks 1-2)",
                "count the checks; stop if any failed",
                "show the series and the money per route, as a table"]
STEP_05_OUTPUT = "one row per series with its coverage and route (in memory)"

# ─── named constants ─────────────────────────────────────────────────────────────
COVERAGE_SEPARATOR = "+"
ROUTES_IN_ORDER = [ROUTE_PREDICTABLE, ROUTE_HISTORY_ONLY, ROUTE_FUTURE_ONLY]
MONTH_COUNT_COLUMN = {ROLE_TRAIN: "meses_entrenamiento", ROLE_TEST: "meses_examen", ROLE_PROJECTION: "meses_proyeccion"}


def route_from_coverage(roles_present: set) -> str:
    """The route of a series from the roles it has: nothing to predict → solo_historia;
    something to predict and a closed month (training or exam) → predecible; something
    to predict and no closed month → solo_futuro."""
    has_something_to_predict = ROLE_PROJECTION in roles_present
    has_closed_months = ROLE_TRAIN in roles_present or ROLE_TEST in roles_present
    if not has_something_to_predict:
        return ROUTE_HISTORY_ONLY
    if has_closed_months:
        return ROUTE_PREDICTABLE
    return ROUTE_FUTURE_ONLY


def build_series_routes(forecast_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per series: its months per role, coverage and route; checked."""
    configuration.log_step_start(STEP_05_LABEL, STEP_05_NAME, STEP_05_PURPOSE, STEP_05_ACTIONS, STEP_05_OUTPUT)
    check_log = []
    period_column = configuration.period_col

    # [1] the units of each series
    grouped_by_series = forecast_units.groupby(SERIES_ID_COLUMN, sort=True)
    series_table = grouped_by_series[configuration.rate_series_columns].first()
    configuration.log_action(STEP_05_LABEL, 1, f"{len(forecast_units):,} forecast units grouped into "
                                            f"{len(series_table):,} series")

    # [2] months per role, first and last month
    months_per_role = (forecast_units.groupby([SERIES_ID_COLUMN, CALENDAR_ROLE_COLUMN]).size()
                       .unstack(fill_value=0).reindex(columns=ROLES_IN_ORDER, fill_value=0))
    for role in ROLES_IN_ORDER:
        series_table[MONTH_COUNT_COLUMN[role]] = months_per_role[role]
    series_table["primer_mes"] = grouped_by_series[period_column].min()
    series_table["ultimo_mes"] = grouped_by_series[period_column].max()
    series_table["meses"] = grouped_by_series.size()
    configuration.log_action(STEP_05_LABEL, 2, f"months counted per role; a series has "
                                            f"{series_table['meses'].median():.0f} months (median)")

    # [3] coverage and route
    roles_of_series = months_per_role.apply(lambda month_counts: {role for role in ROLES_IN_ORDER if month_counts[role] > 0},
                                            axis=1)
    series_table[COVERAGE_COLUMN] = roles_of_series.map(
        lambda roles: COVERAGE_SEPARATOR.join(role for role in ROLES_IN_ORDER if role in roles))
    series_table[ROUTE_COLUMN] = roles_of_series.map(route_from_coverage)
    configuration.log_action(STEP_05_LABEL, 3, f"routes: {series_table[ROUTE_COLUMN].value_counts().to_dict()}")

    # [4] what each series has to predict: what falls due from the current month on
    to_predict = forecast_units[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION
    predicted_units = forecast_units[to_predict].groupby(SERIES_ID_COLUMN)
    series_table["unidades_por_predecir"] = predicted_units[configuration.pipeline_units_col].sum()
    series_table["usd_por_predecir"] = predicted_units[configuration.pipeline_usd_col].sum()
    series_table[["unidades_por_predecir", "usd_por_predecir"]] = series_table[["unidades_por_predecir", "usd_por_predecir"]].fillna(0)
    series_table = series_table.reset_index()
    configuration.log_action(STEP_05_LABEL, 4, f"${series_table['usd_por_predecir'].sum():,.0f} to predict across "
                                            f"{int((series_table['usd_por_predecir'] > 0).sum()):,} series")

    # [5] the checks
    configuration.log_action(STEP_05_LABEL, 5, "checking the series")
    check_series(forecast_units, series_table, configuration, check_log)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_05_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_05_LABEL, STEP_05_NAME, check_log)

    # [7] series and money per route
    log_routes_report(series_table, configuration)
    return series_table


def check_series(forecast_units: pd.DataFrame, series_table: pd.DataFrame, configuration: Config,
                 check_log: list) -> None:
    """Checks 1 and 2."""
    # [1] every unit has its series in the table, and every series a known route
    units_without_series = int((~forecast_units[SERIES_ID_COLUMN].isin(set(series_table[SERIES_ID_COLUMN]))).sum())
    unknown_routes = sorted(set(series_table[ROUTE_COLUMN]) - set(ROUTES_IN_ORDER))
    configuration.log_check(STEP_05_LABEL, check_log,
                            "every forecast unit belongs to a series of the table, and every series has a route",
                            units_without_series == 0 and not unknown_routes,
                            failure_detail=f"{units_without_series:,} units without series · unknown routes {unknown_routes}",
                            context=f"{len(series_table):,} series · {len(forecast_units):,} units")

    # [2] a series with months to predict has something due in them
    predictable_routes = series_table[ROUTE_COLUMN].isin([ROUTE_PREDICTABLE, ROUTE_FUTURE_ONLY])
    empty_future = series_table[predictable_routes & (series_table["unidades_por_predecir"] == 0)]
    configuration.log_check(STEP_05_LABEL, check_log, "every series with something to predict has pipeline to predict",
                            empty_future.empty,
                            failure_detail=f"{len(empty_future):,} series have future months with 0 units due",
                            blocking=False,
                            examples=empty_future[[SERIES_ID_COLUMN, COVERAGE_COLUMN, "meses_proyeccion",
                                                   "unidades_por_predecir"]])


def log_routes_report(series_table: pd.DataFrame, configuration: Config) -> None:
    """Action 7: series and money to predict per route, as a table."""
    configuration.log_action(STEP_05_LABEL, 7, "series and money to predict per route:")
    total_usd = series_table["usd_por_predecir"].sum()
    route_rows = []
    for route in ROUTES_IN_ORDER:
        route_series = series_table[series_table[ROUTE_COLUMN] == route]
        route_usd = route_series["usd_por_predecir"].sum()
        route_rows.append({ROUTE_COLUMN: route, "series": len(route_series), "usd_por_predecir": route_usd,
                           "pct_usd": route_usd / total_usd if total_usd else 0.0})
    configuration.show_table(pd.DataFrame(route_rows))


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 06 · RATE SERIES  (until 25-oct-2026: step_06_rate_series.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The rate series: the renewal rate of every unit, and of every series.
#
# A rate series is the monthly sequence of forecast units of one fs_id. Before estimating
# anything later, this step leaves three things ready:
#   · GAPS: a month with no unit INSIDE the history of a series (between its first and
#     last closed month) gets a gap row, marked sintetica = 1, with nothing due. A month
#     with no expirations says nothing about the rate: its rate is null, never 0 %.
#     Only series that will be estimated get gaps (route predecible).
#   · THE RATE OF EVERY UNIT: renewed / due units, only in closed truth months
#     (entrenamiento, examen) of real units with something due. Everywhere else it is
#     null: no information, never 0 %.
#   · THE SUMMARY OF EVERY SERIES: its support (n_propio: the median units due in its
#     real closed months with something due — the size of a typical month, the one we
#     will predict), its own rate (Σ renewed / Σ due over its closed months), the Wilson
#     half-width of that rate at n_propio and z (the binomial error of one month), and its
#     sign (from its ACTIVE timevarying flags: neutro, negativo, positivo or mixto).
#     A series below the support floor will borrow support from its relatives later.
#
# Actions (logged as they are done):
#   1. find the series that get gaps and their history span
#   2. add the gap rows
#   3. the rate of every unit
#   4. the summary of every series: support, own rate, binomial error
#   5. the sign of every series
#   6. check the gaps, the rates and the summary                      checks 1-4
#   7. count the checks; stop if any failed
#   8. show the series by support and the largest ones, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. gap rows only inside the history of estimable series, never repeating a unit
#    2. a rate only in real closed months with something due
#    3. every rate between 0 and 1                                    (warning only)
#    4. every series has its summary row
#
# Output: (the forecast units with the gap rows and the columns tasa and sintetica, the series summary),
# in memory: the gap rows reach sff_nucleo (step 24) and the summary the forecast series dimension.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_06_LABEL = "06"
STEP_06_NAME = "RATE SERIES"
STEP_06_PURPOSE = ("prepare the renewal rate: fill the gaps inside the history of every estimable series (rate "
                "undefined, never 0 %), compute the rate of every unit only where it is truth, and summarise every "
                "series by its support, its own rate, its binomial error and its sign")
STEP_06_ACTIONS = ["find the series that get gaps and their history span",
                "add the gap rows",
                "the rate of every unit",
                "the summary of every series: support, own rate, binomial error",
                "the sign of every series",
                "check the gaps, the rates and the summary (checks 1-4)",
                "count the checks; stop if any failed",
                "show the series by support and the largest ones, as tables"]
STEP_06_OUTPUT = ("the forecast units with gap rows and the rate of every unit · one row per series with its support, "
               "own rate, error and sign (in memory)")

# ─── named constants ─────────────────────────────────────────────────────────────
PERCENTAGE_POINTS = 100


def wilson_half_width_pp(rates: np.ndarray, supports: np.ndarray, z: float) -> np.ndarray:
    """Half the Wilson interval, in pp, for arrays of rates and supports. Wilson is honest for
    small n and rates near 0 or 1 (the Wald p ± z·se crosses 0 and 1 and is too narrow there).
    A support of 0 gives the widest interval, [0, 1]: 50 pp."""
    rates = np.asarray(rates, dtype=float)
    supports = np.asarray(supports, dtype=float)
    safe_supports = np.where(supports > 0, supports, 1.0)
    z_squared = z * z
    denominator = 1 + z_squared / safe_supports
    centre = (rates + z_squared / (2 * safe_supports)) / denominator
    half_width = z * np.sqrt(rates * (1 - rates) / safe_supports + z_squared / (4 * safe_supports ** 2)) / denominator
    low = np.clip(centre - half_width, 0.0, 1.0)
    high = np.clip(centre + half_width, 0.0, 1.0)
    return np.where(supports > 0, PERCENTAGE_POINTS * (high - low) / 2, PERCENTAGE_POINTS / 2)


def build_rate_series(forecast_units: pd.DataFrame, series_table: pd.DataFrame, configuration: Config) -> tuple:
    """Gaps, the rate of every unit and the summary of every series; checked."""
    configuration.log_step_start(STEP_06_LABEL, STEP_06_NAME, STEP_06_PURPOSE, STEP_06_ACTIONS, STEP_06_OUTPUT)
    check_log = []

    # [1] the series that get gaps: the estimated ones; their closed span
    estimable_series = series_table.loc[series_table[ROUTE_COLUMN] == ROUTE_PREDICTABLE, SERIES_ID_COLUMN]
    truth_units = forecast_units[forecast_units[SERIES_ID_COLUMN].isin(set(estimable_series))
                                 & forecast_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    configuration.log_action(STEP_06_LABEL, 1, f"{len(estimable_series):,} estimable series (predecible) with "
                                            f"{len(truth_units):,} units in closed months")

    # [2] the gap rows
    gap_rows = build_gap_rows(truth_units, configuration)
    rated_units = forecast_units.copy()
    rated_units[SYNTHETIC_COLUMN] = 0
    if len(gap_rows):
        rated_units = pd.concat([rated_units, gap_rows[rated_units.columns]], ignore_index=True)
    configuration.log_action(STEP_06_LABEL, 2, f"{len(gap_rows):,} gap rows added in "
                                            f"{gap_rows[SERIES_ID_COLUMN].nunique() if len(gap_rows) else 0:,} series")

    # [3] the rate of every unit: only where it is truth
    pipeline_units = rated_units[configuration.pipeline_units_col]
    rate_is_truth = ((rated_units[SYNTHETIC_COLUMN] == 0) & (pipeline_units > 0)
                     & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES))
    rated_units[RATE_COLUMN] = np.where(rate_is_truth,
                                        rated_units[configuration.renewed_units_col] / pipeline_units.where(pipeline_units > 0),
                                        np.nan)
    configuration.log_action(STEP_06_LABEL, 3, f"rate computed in {int(rate_is_truth.sum()):,} units; null in the other "
                                            f"{int((~rate_is_truth).sum()):,} (future, gaps, nothing due)")

    # [4] the summary of every series
    series_rate = summarise_series(rated_units, series_table, configuration)
    with_history = series_rate["meses_historia"] > 0
    below_floor = with_history & (series_rate["bajo_suelo"] == 1)
    configuration.log_action(STEP_06_LABEL, 4, f"{len(series_rate):,} series summarised: {int(below_floor.sum()):,} with own "
                                            f"history below the support floor ({configuration.support_floor:.0f}) and "
                                            f"{int((~with_history).sum()):,} with no own history carry "
                                            f"${series_rate.loc[below_floor | ~with_history, 'usd_por_predecir'].sum():,.0f} "
                                            f"of ${series_rate['usd_por_predecir'].sum():,.0f} to predict: they will borrow")

    # [5] the sign of every series
    series_rate[SIGN_COLUMN] = sign_of_series(rated_units, series_rate[SERIES_ID_COLUMN], configuration)
    configuration.log_action(STEP_06_LABEL, 5, f"signs: {series_rate[SIGN_COLUMN].value_counts().to_dict()}")

    # [6] the checks
    configuration.log_action(STEP_06_LABEL, 6, "checking the gaps, the rates and the summary")
    check_rate_series(rated_units, gap_rows, truth_units, series_rate, series_table, configuration, check_log)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_06_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_06_LABEL, STEP_06_NAME, check_log)

    # [8] the series by support, and the largest ones
    log_rate_series_report(series_rate, configuration)
    return rated_units, series_rate


def build_gap_rows(truth_units: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per month missing between the first and the last closed month of a series.
    The row copies the description of the series, has nothing due, and takes the role of
    its month in the calendar. Built as one frame (months × series minus the existing), not
    row by row: with thousands of series the row-by-row version takes seconds per step."""
    period_column = configuration.period_col
    if truth_units.empty:
        return pd.DataFrame()

    # [1] every month of every series' span, minus the months it already has
    span = truth_units.groupby(SERIES_ID_COLUMN)[period_column].agg(["min", "max"])
    all_months = pd.concat([pd.DataFrame({SERIES_ID_COLUMN: series_id,
                                          period_column: pd.period_range(first_month, last_month, freq="M")})
                            for series_id, (first_month, last_month) in span.iterrows()], ignore_index=True)
    existing = truth_units[[SERIES_ID_COLUMN, period_column]].drop_duplicates()
    missing = all_months.merge(existing, on=[SERIES_ID_COLUMN, period_column], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if missing.empty:
        return pd.DataFrame()

    # [2] the gap rows: the series' description, nothing due, a new unit id
    template = (truth_units.sort_values(period_column).drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN))
    gap_rows = template.loc[missing[SERIES_ID_COLUMN]].reset_index()
    gap_rows[period_column] = missing[period_column].to_numpy()
    for measure_column in configuration.core_measures:
        gap_rows[measure_column] = 0.0
    gap_rows[CURRENT_MONTH_COLUMN] = 0
    gap_rows[FINE_ROWS_COLUMN] = 0
    gap_rows[SYNTHETIC_COLUMN] = 1
    gap_rows[UNIT_ID_COLUMN] = gap_rows[SERIES_ID_COLUMN] + ID_FIELD_SEPARATOR + gap_rows[period_column].astype(str)

    # [3] the role of its month in the calendar (a gap of an exam month is examen, not the role of the previous month)
    gap_rows[CALENDAR_ROLE_COLUMN] = configuration.role_of_months(gap_rows[period_column])
    return gap_rows


def summarise_series(rated_units: pd.DataFrame, series_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per series of the series table: support, history, own rate, binomial error."""
    labels = series_table[[SERIES_ID_COLUMN, ROUTE_COLUMN, "usd_por_predecir"]]
    history = rated_units[rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)]
    real_months_with_pipeline = history[(history[SYNTHETIC_COLUMN] == 0)
                                        & (history[configuration.pipeline_units_col] > 0)]

    grouped_history = history.groupby(SERIES_ID_COLUMN)
    summary = pd.DataFrame({
        "n_propio": real_months_with_pipeline.groupby(SERIES_ID_COLUMN)[configuration.pipeline_units_col].median(),
        "meses_historia": grouped_history.size(),
        "huecos": grouped_history[SYNTHETIC_COLUMN].sum(),
        "renovadas": grouped_history[configuration.renewed_units_col].sum(),
        "vencen": grouped_history[configuration.pipeline_units_col].sum()})
    summary = labels.merge(summary, left_on=SERIES_ID_COLUMN, right_index=True, how="left")
    summary[["n_propio", "renovadas", "vencen"]] = summary[["n_propio", "renovadas", "vencen"]].fillna(0.0)
    summary[["meses_historia", "huecos"]] = summary[["meses_historia", "huecos"]].fillna(0).astype(int)

    summary["tasa_propia"] = summary["renovadas"] / summary["vencen"].where(summary["vencen"] > 0)
    summary["error_binomial_pp"] = wilson_half_width_pp(summary["tasa_propia"].fillna(configuration.no_evidence_rate).clip(0, 1),
                                                        summary["n_propio"], configuration.z)
    summary["bajo_suelo"] = (summary["n_propio"] < configuration.support_floor).astype(int)
    return summary


def sign_of_series(rated_units: pd.DataFrame, series_ids: pd.Series, configuration: Config) -> np.ndarray:
    """The sign of every series from its ACTIVE timevarying flags (constant inside a series:
    they are part of fs_id): neutro (none active), negativo, positivo, mixto (both kinds)."""
    one_row_per_series = rated_units.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN).loc[series_ids]
    negative_columns = [column_name for column_name, sign in configuration.structural_timevarying_dims.items() if sign == "negative"]
    positive_columns = [column_name for column_name, sign in configuration.structural_timevarying_dims.items() if sign == "positive"]
    negative_active = (one_row_per_series[negative_columns].isin(ACTIVE_FLAG_VALUES).any(axis=1).to_numpy()
                       if negative_columns else np.zeros(len(one_row_per_series), dtype=bool))
    positive_active = (one_row_per_series[positive_columns].isin(ACTIVE_FLAG_VALUES).any(axis=1).to_numpy()
                       if positive_columns else np.zeros(len(one_row_per_series), dtype=bool))
    return np.select([negative_active & positive_active, negative_active, positive_active],
                     [SIGN_MIXED, SIGN_NEGATIVE, SIGN_POSITIVE], default=SIGN_NEUTRAL)


def check_rate_series(rated_units: pd.DataFrame, gap_rows: pd.DataFrame, truth_units: pd.DataFrame,
                      series_rate: pd.DataFrame, series_table: pd.DataFrame, configuration: Config,
                      check_log: list) -> None:
    """Checks 1 to 4."""
    period_column = configuration.period_col

    # [1] gap rows inside the closed span of an estimable series, and never an existing unit
    gaps_outside_span = 0
    if len(gap_rows):
        span = truth_units.groupby(SERIES_ID_COLUMN)[period_column].agg(["min", "max"])
        gap_span = span.loc[gap_rows[SERIES_ID_COLUMN]]
        inside = ((gap_rows[period_column].to_numpy() > gap_span["min"].to_numpy())
                  & (gap_rows[period_column].to_numpy() < gap_span["max"].to_numpy()))
        gaps_outside_span = int((~inside).sum())
    repeated_units = int(rated_units[UNIT_ID_COLUMN].duplicated().sum())
    configuration.log_check(STEP_06_LABEL, check_log,
                            "gap rows only inside the history of estimable series, never repeating a unit",
                            gaps_outside_span == 0 and repeated_units == 0,
                            failure_detail=f"{gaps_outside_span:,} gap rows outside the span · {repeated_units:,} repeated units",
                            context=f"{len(gap_rows):,} gap rows")

    # [2] the rate exists only where it is truth
    rate_where_not_truth = int((rated_units[RATE_COLUMN].notna()
                                & ((rated_units[SYNTHETIC_COLUMN] == 1)
                                   | ~rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                                   | (rated_units[configuration.pipeline_units_col] <= 0))).sum())
    configuration.log_check(STEP_06_LABEL, check_log, "a rate only in real closed months with something due",
                            rate_where_not_truth == 0,
                            failure_detail=f"{rate_where_not_truth:,} units have a rate where there is no truth")

    # [3] rates are proportions (a rate above 1 comes from renewing more than fell due: step 01 warned)
    out_of_range = rated_units[(rated_units[RATE_COLUMN] < 0) | (rated_units[RATE_COLUMN] > 1)]
    configuration.log_check(STEP_06_LABEL, check_log, "every rate between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range):,} units with a rate outside [0, 1]", blocking=False,
                            examples=out_of_range[[UNIT_ID_COLUMN, configuration.pipeline_units_col,
                                                   configuration.renewed_units_col, RATE_COLUMN]])

    # [4] every series of the series table has its summary
    missing_series = len(set(series_table[SERIES_ID_COLUMN]) - set(series_rate[SERIES_ID_COLUMN]))
    configuration.log_check(STEP_06_LABEL, check_log, "every series has its summary row", missing_series == 0,
                            failure_detail=f"{missing_series:,} series without a summary",
                            context=f"{len(series_rate):,} series")


def log_rate_series_report(series_rate: pd.DataFrame, configuration: Config) -> None:
    """Action 8: series and money to predict by support, and the largest series."""
    configuration.log_action(STEP_06_LABEL, 8, f"series by support (n_propio: units due in a typical closed month; "
                                            f"the floor is {configuration.support_floor:.0f}):")
    with_history = series_rate["meses_historia"] > 0
    groups = {"sin historia propia (solo_futuro)": ~with_history,
              f"bajo el suelo (< {configuration.support_floor:.0f})": with_history & (series_rate["bajo_suelo"] == 1),
              f"sobre el suelo (≥ {configuration.support_floor:.0f})": with_history & (series_rate["bajo_suelo"] == 0)}
    total_usd = series_rate["usd_por_predecir"].sum()
    configuration.show_table(pd.DataFrame([{"soporte": group_name, "series": int(mask.sum()),
                                            "error_pp_mediano": series_rate.loc[mask, "error_binomial_pp"].median(),
                                            "usd_por_predecir": series_rate.loc[mask, "usd_por_predecir"].sum(),
                                            "pct_usd": series_rate.loc[mask, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0}
                                           for group_name, mask in groups.items()]))

    configuration.logger.doc(f"[{STEP_06_LABEL}] the {configuration.largest_series_shown} series with the most to predict "
                             f"(own rate ± its binomial error of one month):")
    configuration.show_table(series_rate.nlargest(configuration.largest_series_shown, "usd_por_predecir")[
        [SERIES_ID_COLUMN, ROUTE_COLUMN, SIGN_COLUMN, "n_propio", "meses_historia", "huecos", "tasa_propia",
         "error_binomial_pp", "usd_por_predecir"]])


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 1 · THE BLOCK: run the steps, keep the results, persist them, resume them
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B1"

# what the steps after the block read, and the table each one is persisted in
PERSISTED_RESULTS = {"time_series_rows": TABLE_BLOCK_1_TIME_SERIES_ROWS, "fine_table": TABLE_BLOCK_1_FINE_TABLE,
                     "forecast_units": TABLE_BLOCK_1_FORECAST_UNITS, "series_table": TABLE_BLOCK_1_SERIES,
                     "rated_units": TABLE_BLOCK_1_RATED_UNITS, "series_rate": TABLE_BLOCK_1_SERIES_RATE}
# the columns step 03 adds to the calendared raw (the fine table without them IS the calendared raw)
STEP_03_ADDED_COLUMNS = (SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN)
INDEX_COLUMN = "__indice__"          # the index of a table, persisted as a column (the fine table's index is the row id)
INDEX_POSITION = -1
# fields of the Config the block reads that do not change its results (where it writes, what it shows)
FIELDS_THAT_DO_NOT_CHANGE_THE_BLOCK = {"sql_engine", "sql_schema", "table_prefix", "output_folder", "log_level",
                                       "log_colors", "log_file", "example_rows_shown", "largest_series_shown"}


class PreparationBlock:
    """BLOCK 1 · PREPARATION (see the header of this module): its results are attributes, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.raw_extract = None
        self.validated_extract = None
        self.leveled_raw = None
        self.time_series_rows = None
        self.calendared_raw = None
        self.fine_table = None
        self.forecast_units = None
        self.series_table = None
        self.rated_units = None
        self.series_rate = None
        self.origin = None                # "run" (computed in this session) or "sql" (resumed)
        self.seconds = {}                 # how long every step took

    # ─── run the steps ───────────────────────────────────────────────────────────
    def run(self, raw_extract: pd.DataFrame = None) -> "PreparationBlock":
        """Every step of the block, in order. The raw is read with the Config's read_raw unless it is given."""
        configuration = self.configuration
        if raw_extract is None:
            raw_extract = self.read_the_raw_extract()
        self.raw_extract = raw_extract

        started = time.time()
        self.validated_extract = validate_raw(self.raw_extract, configuration)
        self.seconds["00 validate_raw"] = time.time() - started

        started = time.time()
        self.leveled_raw, self.time_series_rows = validate_values_and_build_levels(self.validated_extract, configuration)
        self.seconds["01 values_and_levels"] = time.time() - started

        started = time.time()
        self.calendared_raw = apply_calendar(self.leveled_raw, configuration)
        self.seconds["02 calendar"] = time.time() - started

        started = time.time()
        self.fine_table = build_fine_table(self.calendared_raw, configuration)
        self.seconds["03 fine_table"] = time.time() - started

        started = time.time()
        self.forecast_units = build_forecast_units(self.fine_table, configuration)
        self.seconds["04 forecast_units"] = time.time() - started

        started = time.time()
        self.series_table = build_series_routes(self.forecast_units, configuration)
        self.seconds["05 series_routes"] = time.time() - started

        started = time.time()
        self.rated_units, self.series_rate = build_rate_series(self.forecast_units, self.series_table, configuration)
        self.seconds["06 rate_series"] = time.time() - started

        self.origin = "run"
        return self

    def read_the_raw_extract(self) -> pd.DataFrame:
        """The extract, read with the Config's read_raw (SQL, CSV or synthetic), timed."""
        configuration = self.configuration
        configuration.logger.doc("[main] ═══ SFF · reading the raw ═══")
        started = time.time()
        raw_extract = configuration.read_raw()
        self.seconds["-- read_raw"] = time.time() - started
        configuration.logger.doc(f"[main] raw read: {len(raw_extract):,} rows × {len(raw_extract.columns)} columns "
                                 f"({self.seconds['-- read_raw']:.1f}s)")
        return raw_extract

    def results(self) -> dict:
        """The results under the names the pipeline gives them (what is not there is left out)."""
        named = {"raw_extract": self.raw_extract, "validated_extract": self.validated_extract,
                 "leveled_raw": self.leveled_raw, "time_series_rows": self.time_series_rows,
                 "calendared_raw": self.calendared_raw, "fine_table": self.fine_table,
                 "forecast_units": self.forecast_units, "series_table": self.series_table,
                 "rated_units": self.rated_units, "series_rate": self.series_rate}
        return {name: value for name, value in named.items() if value is not None}

    # ─── persist ─────────────────────────────────────────────────────────────────
    def persist_block(self) -> None:
        """The results the steps after the block read, to SQL (to CSV in output_folder without an engine), with
        their manifest (every column and its type) and the fingerprint of what the block was built from."""
        configuration = self.configuration
        if self.fine_table is None:
            raise ValueError("block 1 has no results to persist: run() it or resume it first")
        started = time.time()
        manifest_rows = []
        for attribute, table_name in PERSISTED_RESULTS.items():
            frame = getattr(self, attribute)
            sql_frame, columns_of_table = frame_for_sql(frame, table_name)
            write_plain_table(sql_frame, table_name, configuration)
            manifest_rows += columns_of_table
            configuration.logger.doc(f"[{BLOCK_LABEL}] persisted {attribute}: {len(frame):,} rows × "
                                     f"{len(frame.columns)} columns → {configuration.table_prefix}{table_name}")
        write_plain_table(pd.DataFrame(manifest_rows, columns=["tabla", "posicion", "columna", "dtype", "tipo"]),
                          TABLE_BLOCK_1_MANIFEST, configuration)
        fingerprint = self.fingerprint()
        write_plain_table(pd.DataFrame(sorted(fingerprint.items()), columns=["clave", "valor"]),
                          TABLE_BLOCK_1_FINGERPRINT, configuration)
        configuration.logger.doc(f"[{BLOCK_LABEL}] block 1 persisted in {time.time() - started:.1f}s: "
                                 f"{len(PERSISTED_RESULTS)} tables, their manifest and the fingerprint "
                                 f"({fingerprint['extract_rows']} rows of the extract, current month "
                                 f"{fingerprint['current_month']})")

    def fingerprint(self) -> dict:
        """What the block was built from: the extract (rows and USD due), the current month and every field of the
        Config the block reads that can change its results."""
        configuration = self.configuration
        fingerprint = {"extract_rows": str(len(self.raw_extract)),
                       "extract_usd_due": f"{float(self.raw_extract[configuration.pipeline_usd_col].sum()):.6f}",
                       "current_month": str(configuration.calendar_boundaries()["current"])}
        for field_name in fields_the_block_reads(configuration):
            fingerprint[f"config.{field_name}"] = repr(getattr(configuration, field_name))
        return fingerprint

    # ─── resume ──────────────────────────────────────────────────────────────────
    def resume_block_from_sql(self) -> "PreparationBlock":
        """The results persisted by persist_block, read back with their types and indexes; the extract read again.
        Stops if the extract, the current month or a field of the Config the block reads changed since then."""
        configuration = self.configuration
        started = time.time()
        manifest = read_plain_table(TABLE_BLOCK_1_MANIFEST, configuration)
        manifest["posicion"] = pd.to_numeric(manifest["posicion"]).astype(int)     # a CSV gives it back as text
        fingerprint_table = read_plain_table(TABLE_BLOCK_1_FINGERPRINT, configuration)
        persisted_fingerprint = dict(zip(fingerprint_table["clave"], fingerprint_table["valor"].astype(str)))
        self.raw_extract = self.read_the_raw_extract()
        current_fingerprint = self.fingerprint()
        changed = {key: (persisted_fingerprint.get(key, "(absent)"), value) for key, value in current_fingerprint.items()
                   if persisted_fingerprint.get(key) != value}
        changed.update({key: (value, "(absent)") for key, value in persisted_fingerprint.items()
                        if key not in current_fingerprint})
        if changed:
            detail = "\n".join(f"  {key}: persisted {before} · now {now}" for key, (before, now) in changed.items())
            raise ValueError(f"block 1 cannot be resumed: what it was built from has changed since it was persisted "
                             f"(run it again with PreparationBlock(configuration).run()):\n{detail}")
        for attribute, table_name in PERSISTED_RESULTS.items():
            sql_frame = read_plain_table(table_name, configuration)
            setattr(self, attribute, frame_from_sql(sql_frame, manifest[manifest["tabla"] == table_name]))
        self.calendared_raw = self.fine_table.drop(columns=[configuration.discount_bucket_column, *STEP_03_ADDED_COLUMNS],
                                                   errors="ignore")
        self.validated_extract = None
        self.leveled_raw = None
        self.origin = "sql"
        self.seconds["B1 resume_block_from_sql"] = time.time() - started - self.seconds.get("-- read_raw", 0.0)
        configuration.logger.doc(f"[{BLOCK_LABEL}] block 1 resumed from {configuration.table_prefix}{TABLE_BLOCK_1_FINE_TABLE} "
                                 f"and the other {len(PERSISTED_RESULTS) - 1} tables in {time.time() - started:.1f}s "
                                 f"(the extract, the current month and the Config are the ones it was built with)")
        return self


def fields_the_block_reads(configuration: Config) -> list:
    """The fields of the Config the code of the block reads (steps 00 to 06), except the ones that only say where to
    write or what to show."""
    from dataclasses import fields as dataclass_fields
    names = [config_field.name for config_field in dataclass_fields(type(configuration))]
    readers = steps_reading_each_field(names)
    block_labels = {"00", "01", "02", "03", "04", "05", "06", BLOCK_LABEL}
    return sorted(name for name, labels in readers.items()
                  if block_labels & set(labels) and name not in FIELDS_THAT_DO_NOT_CHANGE_THE_BLOCK)


# ─── a table to SQL and back, with its types ─────────────────────────────────────

def frame_for_sql(frame: pd.DataFrame, table_name: str) -> tuple:
    """(the table as SQL can hold it, the manifest rows that give every column its type back).

    RULES:   the index becomes the column __indice__ (the fine table's index is the row id the forecast keeps). A
             monthly Period is written as text; a bool as 0 / 1. An object column keeps its values when they are all of
             one kind (text, numbers or bools); a column that mixes kinds is written as JSON, so every value comes back
             with its own type (a flag written 0 in some rows and "0" in others stays so)."""
    sql_frame = frame.reset_index(drop=False)
    sql_frame = sql_frame.rename(columns={sql_frame.columns[0]: INDEX_COLUMN})
    index_kind = "range" if isinstance(frame.index, pd.RangeIndex) else str(frame.index.dtype)
    manifest_rows = [(table_name, INDEX_POSITION, INDEX_COLUMN, index_kind, frame.index.name or "")]
    for position, column_name in enumerate(frame.columns):
        values = frame[column_name]
        dtype_text, kind = str(values.dtype), ""
        if isinstance(values.dtype, pd.PeriodDtype):
            sql_frame[column_name] = values.astype(str).where(values.notna(), None).to_numpy()
        elif dtype_text in ("bool", "boolean"):
            sql_frame[column_name] = values.astype("Int64").to_numpy()
        elif dtype_text == "category":
            sql_frame[column_name] = values.astype(object).to_numpy()
            kind = "str"
        elif dtype_text == "object":
            kind = kind_of_object_column(values)
            if kind == "json":
                sql_frame[column_name] = values.map(lambda value: None if is_missing(value) else
                                                    json.dumps(value.item() if isinstance(value, np.generic) else value)).to_numpy()
            elif kind == "bool":
                sql_frame[column_name] = values.map(lambda value: None if is_missing(value) else int(bool(value))).to_numpy()
        manifest_rows.append((table_name, position, str(column_name), dtype_text, kind))
    return sql_frame, manifest_rows


def kind_of_object_column(values: pd.Series) -> str:
    """The kind of the values of an object column, ignoring the missing ones: str, int, float, bool or json (mixed)."""
    kinds = set()
    for value in values.dropna().unique() if len(values) else []:
        if isinstance(value, (bool, np.bool_)):
            kinds.add("bool")
        elif isinstance(value, (int, np.integer)):
            kinds.add("int")
        elif isinstance(value, (float, np.floating)):
            kinds.add("float")
        elif isinstance(value, str):
            kinds.add("str")
        else:
            kinds.add("json")
    if not kinds:
        return "str"
    return kinds.pop() if len(kinds) == 1 else "json"


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def frame_from_sql(sql_frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """The table as it was before frame_for_sql: its columns in their order, every one with its type, and its index."""
    manifest = manifest.sort_values("posicion")
    frame = pd.DataFrame(index=range(len(sql_frame)))
    for _, column in manifest[manifest["posicion"] != INDEX_POSITION].iterrows():
        frame[column["columna"]] = restore_column(sql_frame[column["columna"]], column["dtype"], column["tipo"])
    index_row = manifest[manifest["posicion"] == INDEX_POSITION].iloc[0]
    index_name = None if is_missing(index_row["tipo"]) or index_row["tipo"] == "" else index_row["tipo"]
    if index_row["dtype"] == "range":
        # a RangeIndex comes back as one (start, step from its first two values)
        positions = pd.to_numeric(sql_frame[INDEX_COLUMN]).astype("int64").to_numpy()
        step = int(positions[1] - positions[0]) if len(positions) > 1 else 1
        start = int(positions[0]) if len(positions) else 0
        frame.index = pd.RangeIndex(start=start, stop=start + step * len(positions), step=step, name=index_name)
    else:
        index_values = restore_column(sql_frame[INDEX_COLUMN], index_row["dtype"], "")
        frame.index = pd.Index(index_values.to_numpy(), name=index_name)
    return frame


def restore_column(values: pd.Series, dtype_text: str, kind) -> pd.Series:
    """One column back to its type (see frame_for_sql)."""
    kind = "" if is_missing(kind) else str(kind)
    values = values.reset_index(drop=True)
    if dtype_text.startswith("period"):
        return pd.Series(pd.PeriodIndex(values.where(values.notna(), None).astype(object), freq="M"))
    if dtype_text == "bool":
        return pd.to_numeric(values).astype(int).astype(bool)
    if dtype_text == "boolean":
        return pd.to_numeric(values).astype("Int64").astype("boolean")
    if dtype_text == "category":
        return values.astype(object).where(values.notna(), None).astype("category")
    if dtype_text.startswith(("int", "uint", "float", "Int", "UInt", "Float")):
        return pd.to_numeric(values).astype(dtype_text)
    if dtype_text.startswith("datetime"):
        return pd.to_datetime(values)
    if dtype_text in ("str", "string"):
        # the text type of pandas (the default for text from pandas 3 on): the same type back
        return values.where(values.notna(), None).astype(object).astype(pd.api.types.pandas_dtype(dtype_text))
    # object
    if kind == "json":
        return values.map(lambda value: None if is_missing(value) else json.loads(value)).astype(object)
    if kind == "bool":
        return values.map(lambda value: None if is_missing(value) else bool(int(float(value)))).astype(object)
    if kind == "int":
        return values.map(lambda value: None if is_missing(value) else int(float(value))).astype(object)
    if kind == "float":
        return values.map(lambda value: None if is_missing(value) else float(value)).astype(object)
    return values.map(lambda value: None if is_missing(value) else str(value)).astype(object)


def write_plain_table(frame: pd.DataFrame, table_name: str, configuration: Config) -> None:
    """A table to SQL (replaced) or, without an engine, to a CSV in output_folder; no check is logged."""
    physical_name = f"{configuration.table_prefix}{table_name}"
    if configuration.sql_engine is not None:
        frame.to_sql(physical_name, configuration.sql_engine, schema=configuration.sql_schema, if_exists="replace",
                     index=False, chunksize=10_000)
    else:
        os.makedirs(configuration.output_folder, exist_ok=True)
        frame.to_csv(os.path.join(configuration.output_folder, f"{physical_name}.csv"), index=False)


def read_plain_table(table_name: str, configuration: Config) -> pd.DataFrame:
    """A table written by write_plain_table, as it is in SQL (or in its CSV): every value as stored."""
    physical_name = f"{configuration.table_prefix}{table_name}"
    if configuration.sql_engine is not None:
        qualified_name = f"{configuration.sql_schema}.{physical_name}" if configuration.sql_schema else physical_name
        return pd.read_sql(f"SELECT * FROM {qualified_name}", configuration.sql_engine)
    path = os.path.join(configuration.output_folder, f"{physical_name}.csv")
    if not os.path.exists(path):
        raise ValueError(f"block 1 cannot be resumed: {path} does not exist (persist_block() writes it)")
    return pd.read_csv(path, dtype=object, keep_default_na=False, na_values=[""])
