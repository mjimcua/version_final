"""
block_2_rate.py — BLOCK 2 · RATE: the renewal rate of any row of pipeline, at any horizon, with its band.

MISSION: give the best prediction of the renewal rate (renewed / due units) that the history allows, for every row
of pipeline the forecast will value (the extract's and the future pipeline's), at every horizon, with an error band
measured, not assumed.

MECHANISM: a forecast series alone is mostly binomial noise. The block borrows history only where it reduces the
error: the ladder merges a series with the series that differ from it in the dimension that separates the rate the
least, until the group has support (2.1-2.3); every final group becomes one monthly series (2.4), described (2.5,
it decides nothing); a backtest chooses, for every group and horizon, the technique that predicts it best in months
it never saw, and the width of its band (2.6); everything goes into one box that is asked the rate of any row (2.7).
The exam of the whole forecast with this box, against the baseline, is step 7.1.

THE STEPS (each one keeps its actions, its checks and its text on screen):
  2.1  dimensions       how much every dimension separates the rate: the order in which the ladder collapses them
  2.2  ladder groups    merge the series pass by pass until every group has the support floor; a merge only stays
                        if it lowers the error of the series' monthly rate
  2.3  credibility      the rate of every series: its own or its group's, weighed by credibility; its two errors
  2.4  pool series      the monthly series of the rate of every final group (composition)
  2.5  dynamics         φ, trend, seasonality of every composition: descriptive, for the audit, the core and the report
  2.6  backtest         the technique of every composition and horizon band, chosen before the exam, and its band
  2.7  rate box         the interface: RateBox.predict(rows, as_of, mode) → rate and band of every row

READS (block 1): rated_units, series_rate, series_table.
GIVES: rate_box (what blocks 7 and 8 ask) · dimension_decision, ladder, series_estimate, money_by_level,
       pool_series, pool_reference, pool_dynamics, portfolio_profile, portfolio_dynamics, backtest (the audit, the
       core and the report read them).

THE INTERFACE:
  block = RateBlock(configuration).run(rated_units, series_rate, series_table)
  block.rate_box, block.ladder, …       the results, in memory
  block.results()                       {name: result} under the names of the pipeline
  block.seconds                         how long every step took
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time

import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, CHAMPION_ORIGIN, COMPOSITION_ID_COLUMN, Config, DEFAULT_ORIGIN, GATE_LEVEL,
                    GATE_SUPPORT, ID_FIELD_SEPARATOR, LEVEL_BORROWED, LEVEL_FAR, LEVEL_MIXED, LEVEL_NO_HISTORY,
                    LEVEL_NO_IMPACT, LEVEL_OWN, LEVEL_OWN_REINFORCED, LEVEL_OWN_SHORT, LEVEL_SIGNED_UNDER_FLOOR,
                    PURPOSE_EXAM, PURPOSE_SELECTION, RATE_COLUMN, RATE_FROM_CELL, RATE_FROM_GLOBAL, RATE_FROM_POOL,
                    ROUTE_COLUMN, ROUTE_FUTURE_ONLY, ROUTE_HISTORY_ONLY, ROUTE_PREDICTABLE, SERIES_ID_COLUMN,
                    SIGN_COLUMN, SIGN_MIXED, SIGN_NEUTRAL, SIGN_TOKEN, SYNTHETIC_COLUMN, TABLE_BACKTEST_PREDICTIONS,
                    TABLE_COMPOSITION_MEMBERS, TABLE_DIMENSIONS, TABLE_ERROR_BANDS, TABLE_EXAM_BY_POOL,
                    TABLE_EXAM_TOTAL, TABLE_LADDER_MERGES, TABLE_LADDER_STEPS, TABLE_LADDER_SUMMARY,
                    TABLE_POOL_REFERENCE, TABLE_POOL_SERIES, TABLE_RISK_LEVELS, TABLE_TECHNIQUE_DECISION, TRUTH_ROLES,
                    WILDCARD, id_text, join_columns, negative_states, state_combination)
from prediction import band_quantiles, index_history, months_known, predict_composition, rate_band, shifted_rate
from scipy import stats
from techniques import CATALOGUE, eligible_techniques, inverse_logit, logit, predict_logit


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.1 · DIMENSIONS  (until 25-oct-2026: step_07_dimensions.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# Which dimensions separate the renewal rate, and in which order the
# ladder will let the mandatory ones go.
#
# The ladder of step 2.2 lends support to a small series by collapsing it with its relatives:
# it drops mandatory dimensions one by one until there is enough evidence. WHICH one it drops
# first matters: dropping a dimension that separates the rate a lot mixes series that renew
# differently. This step measures it, on the series whose own rate is informative:
# route predecible, own rate known, sign neutro (a series with an active signal would
# contaminate the rates: the signals are treated apart), weighted by n_propio.
#
# For every mandatory and extra_renovacion dimension:
#   · eta2_individual     η²: the share of the variance of the rate between its values, alone
#   · contribucion_unica  the R² lost when it is removed from the model with all dimensions
#   · omega2              ω²: η² corrected for the number of values (a dimension with many
#                         values separates by chance too)
#   · anulable            1 for extra_renovacion: the ladder annuls them before any mandatory
# For the mandatory dimensions, the COLLAPSE ORDER: at every round, among the ones that can
# go (a *_level_N only once its *_level_(N+1) has gone), the one that loses the least R²
# given the ones that remain. That sequential loss is the question the ladder asks at each
# rung (the unique contribution is not: in a nested hierarchy every coarser level has
# unique ≈ 0 while the finer one is there).
#
# Actions (logged as they are done):
#   1. the base: the series whose own rate is informative, with their dimension values
#   2. η², unique contribution and ω² of every dimension
#   3. the collapse order of the mandatory dimensions
#   4. check the figures and the order                                 checks 1-4
#   5. write the decision table                                        check 5
#   6. count the checks; stop if any failed
#   7. show the dimensions in collapse order, as a table
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the base has at least 3 series (else every figure is 0)       (warning only)
#    2. every figure is between 0 and 1
#    3. every mandatory dimension with variation has one collapse position, 1 to K (one without variation: 0, it collapses last and its pass changes no group)
#    4. a *_level_N never collapses before its *_level_(N+1)
#    5. table sff_decision_eta2 written and read back
#
# Output: the decision per dimension · table sff_decision_eta2. The ladder (step 2.2) reads the collapse order.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_2_1_LABEL = "2.1"
STEP_2_1_NAME = "DIMENSIONS"
STEP_2_1_PURPOSE = ("measure how much every dimension separates the renewal rate (η², unique contribution, ω²) and "
                "decide the order in which the ladder will collapse the mandatory ones: first the one whose loss "
                "costs the least R² given the ones that remain")
STEP_2_1_ACTIONS = ["the base: the series whose own rate is informative, with their dimension values",
                "η², unique contribution and ω² of every dimension",
                "the collapse order of the mandatory dimensions",
                "check the figures and the order (checks 1-4)",
                "write the decision table (check 5)",
                "count the checks; stop if any failed",
                "show the dimensions in collapse order, as a table"]
STEP_2_1_OUTPUT = ("one row per dimension (η², unique contribution, ω², collapse order, sequential loss) · "
               "table sff_decision_eta2")

# ─── named constants ─────────────────────────────────────────────────────────────
VALUE_COLUMN = "tasa_propia"
WEIGHT_COLUMN = "n_propio"
LEVEL_MARK = "_level_"          # the naming of a hierarchy: product_level_1 ⊂ product_level_2


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STATISTICS (weighted by the support of each series)
# ═══════════════════════════════════════════════════════════════════════════════════

def weighted_eta2(frame: pd.DataFrame, group_column: str) -> float:
    """η²: weighted sum of squares BETWEEN the values of the column / total. 0 when the
    rate does not vary at all."""
    weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    values = frame[VALUE_COLUMN].to_numpy(dtype=float)
    grand_mean = np.average(values, weights=weights)
    total = float(np.sum(weights * (values - grand_mean) ** 2))
    if total <= 0:
        return 0.0
    grouped = frame.assign(_weighted_value=weights * values).groupby(group_column, observed=True)
    group_weights = grouped[WEIGHT_COLUMN].sum().to_numpy(dtype=float)
    group_means = grouped["_weighted_value"].sum().to_numpy(dtype=float) / group_weights
    between = float(np.sum(group_weights * (group_means - grand_mean) ** 2))
    return between / total


def weighted_omega2(frame: pd.DataFrame, group_column: str) -> float:
    """ω² = (SSB − (k − 1) · MSW) / (SST + MSW): η² corrected for the number of values k.
    0 when there are not more series than values."""
    weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    values = frame[VALUE_COLUMN].to_numpy(dtype=float)
    grand_mean = np.average(values, weights=weights)
    total = float(np.sum(weights * (values - grand_mean) ** 2))
    series_count = len(frame)
    value_count = frame[group_column].nunique()
    if total <= 0 or series_count <= value_count:
        return 0.0
    between = weighted_eta2(frame, group_column) * total
    mean_square_within = (total - between) / (series_count - value_count)
    return float(max(0.0, (between - (value_count - 1) * mean_square_within) / (total + mean_square_within)))


def weighted_r2(frame: pd.DataFrame, dimensions: list, min_series: int) -> float:
    """R² of a weighted least-squares fit of the rate on the dimensions as categories
    (additive model). A design with redundant columns is solved by lstsq."""
    if not dimensions or len(frame) < min_series:
        return 0.0
    design = pd.get_dummies(frame[dimensions].astype(str), drop_first=True).astype(float)
    design.insert(0, "intercept", 1.0)
    root_weights = np.sqrt(frame[WEIGHT_COLUMN].to_numpy(dtype=float))
    values = frame[VALUE_COLUMN].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design.to_numpy() * root_weights[:, None], values * root_weights, rcond=None)[0]
    fitted = design.to_numpy() @ coefficients
    grand_mean = np.average(values, weights=root_weights ** 2)
    total = float(np.sum(root_weights ** 2 * (values - grand_mean) ** 2))
    residual = float(np.sum(root_weights ** 2 * (values - fitted) ** 2))
    return float(max(0.0, 1 - residual / total)) if total > 0 else 0.0


def family_and_level(dimension_name: str, dimensions=()) -> tuple:
    """("product", 2) for "product_level_2". A dimension X without a level of its own is the FINEST level
    of its family when the dimensions include an X_level_N (a leveled dim: its raw value above the level
    the library generated): (X, highest N + 1). Otherwise (name, 1), a dimension outside a hierarchy."""
    if LEVEL_MARK in dimension_name:
        family, _, level_text = dimension_name.rpartition(LEVEL_MARK)
        if level_text.isdigit():
            return family, int(level_text)
    coarser = [family_and_level(other)[1] for other in dimensions
               if other != dimension_name and family_and_level(other)[0] == dimension_name]
    return dimension_name, (max(coarser) + 1 if coarser else 1)


def sequential_collapse_order(base: pd.DataFrame, mandatory_dims: list, configuration: Config) -> list:
    """The collapse order of the mandatory dims, greedy and sequential: at every round, among
    the droppable dims (a *_level_N only when no *_level_(N+1) remains), drop the one whose
    removal from the dims that REMAIN loses the least R². Returns [(dim, loss), ...] from
    the first to collapse to the last; ties go to the name, so the order is reproducible."""
    remaining = list(mandatory_dims)
    collapse_order = []
    current_r2 = weighted_r2(base, remaining, configuration.dimension_min_series)
    while remaining:
        droppable = [dimension for dimension in remaining
                     if not any(family_and_level(other, remaining)
                                == (family_and_level(dimension, remaining)[0], family_and_level(dimension, remaining)[1] + 1)
                                for other in remaining)]
        losses = {}
        for dimension in droppable:
            r2_without = weighted_r2(base, [other for other in remaining if other != dimension], configuration.dimension_min_series)
            losses[dimension] = max(0.0, current_r2 - r2_without)
        next_dimension = min(droppable, key=lambda dimension: (losses[dimension], dimension))
        collapse_order.append((next_dimension, round(losses[next_dimension], configuration.dimension_figure_decimals)))
        remaining.remove(next_dimension)
        current_r2 = weighted_r2(base, remaining, configuration.dimension_min_series)
    return collapse_order


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP
# ═══════════════════════════════════════════════════════════════════════════════════

def analyse_dimensions(series_rate: pd.DataFrame, series_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """η², unique contribution, ω² and the collapse order; checked and written. series_table (step 1.6) gives the
    dimension values of every series."""
    configuration.log_step_start(STEP_2_1_LABEL, STEP_2_1_NAME, STEP_2_1_PURPOSE, STEP_2_1_ACTIONS, STEP_2_1_OUTPUT)
    check_log = []
    mandatory_dims = configuration.business_mandatory_dims
    dimensions = mandatory_dims + configuration.extra_renovacion

    # [1] the base: series with an informative own rate, and their dimension values
    base = series_rate[(series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE) & series_rate[VALUE_COLUMN].notna()
                       & (series_rate[SIGN_COLUMN] == SIGN_NEUTRAL) & (series_rate[WEIGHT_COLUMN] > 0)]
    base = base.merge(series_table[[SERIES_ID_COLUMN] + dimensions], on=SERIES_ID_COLUMN)
    predictable_usd = series_rate.loc[series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE, "usd_por_predecir"].sum()
    configuration.log_action(STEP_2_1_LABEL, 1, f"base: {len(base):,} series (predecible, own rate, neutro), "
                                            f"{base['usd_por_predecir'].sum() / predictable_usd if predictable_usd else 0:.0%} "
                                            f"of the money of the predictable series")
    has_evidence = len(base) >= configuration.dimension_min_series

    # [2] the three figures of every dimension
    r2_all = weighted_r2(base, dimensions, configuration.dimension_min_series)
    dimension_rows = []
    for dimension in dimensions:
        dimension_rows.append({
            "dimension": dimension,
            "grupo": "mandatory" if dimension in mandatory_dims else "extra_renovacion",
            "valores": int(base[dimension].nunique()) if has_evidence else 0,
            "eta2_individual": round(weighted_eta2(base, dimension), configuration.dimension_figure_decimals) if has_evidence else 0.0,
            "contribucion_unica": round(max(0.0, r2_all - weighted_r2(base, [other for other in dimensions if other != dimension], configuration.dimension_min_series)),
                                        configuration.dimension_figure_decimals) if has_evidence else 0.0,
            "omega2": round(weighted_omega2(base, dimension), configuration.dimension_figure_decimals) if has_evidence else 0.0,
            "anulable": int(dimension in configuration.extra_renovacion)})
    decision = pd.DataFrame(dimension_rows)
    configuration.log_action(STEP_2_1_LABEL, 2, f"{len(dimensions)} dimensions measured; R² of all of them together "
                                            f"{r2_all:.3f}")

    # [3] the collapse order of the mandatory dimensions
    # a dimension with a single value (e.g. a generated level that ended in one group) has nothing to
    # collapse: step 2.2 puts it last, where its pass merges nothing
    without_variation = [dimension for dimension in mandatory_dims if base[dimension].nunique() <= 1]
    if without_variation:
        configuration.log_action(STEP_2_1_LABEL, 3, f"no variation, collapsed last by step 2.2 (its pass merges nothing): {without_variation}")
    collapse_order = sequential_collapse_order(base, [dimension for dimension in mandatory_dims if dimension not in without_variation],
                                               configuration)
    position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
    loss_of = dict(collapse_order)
    decision["orden_colapso"] = decision["dimension"].map(position_of).fillna(0).astype(int)
    decision["perdida_secuencial"] = decision["dimension"].map(loss_of)
    configuration.log_action(STEP_2_1_LABEL, 3, "collapse order: " + " → ".join(dimension for dimension, _ in collapse_order))

    # [4] the checks
    configuration.log_action(STEP_2_1_LABEL, 4, "checking the figures and the order")
    check_dimensions(base, decision, collapse_order, configuration, check_log)

    # [5] the decision table, written
    configuration.log_action(STEP_2_1_LABEL, 5, "writing the decision table")
    configuration.write_table(STEP_2_1_LABEL, check_log, decision, TABLE_DIMENSIONS)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_1_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_2_1_LABEL, STEP_2_1_NAME, check_log)

    # [7] the dimensions in collapse order
    configuration.log_action(STEP_2_1_LABEL, 7, "the dimensions (mandatory in collapse order: 1 goes first; extras are "
                                            "annulled before any mandatory):")
    shown = decision.assign(_order=decision["orden_colapso"].where(decision["orden_colapso"] > 0, 0))
    configuration.show_table(shown.sort_values(["anulable", "_order"], ascending=[False, True]).drop(columns="_order"))
    return decision


def check_dimensions(base: pd.DataFrame, decision: pd.DataFrame, collapse_order: list, configuration: Config,
                     check_log: list) -> None:
    """Checks 1 to 4."""
    # [1] enough series to measure anything
    configuration.log_check(STEP_2_1_LABEL, check_log, f"the base has at least {configuration.dimension_min_series} series",
                            len(base) >= configuration.dimension_min_series,
                            failure_detail=f"only {len(base)} series: every figure is 0 and the collapse order is the declared one",
                            context=f"{len(base):,} series", blocking=False)

    # [2] the figures are proportions
    figure_columns = ["eta2_individual", "contribucion_unica", "omega2"]
    out_of_range = decision[((decision[figure_columns] < 0) | (decision[figure_columns] > 1)).any(axis=1)]
    configuration.log_check(STEP_2_1_LABEL, check_log, "every figure is between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range)} dimensions with a figure outside [0, 1]",
                            examples=out_of_range)

    # [3] one position per mandatory dimension with variation, 1 to K; the ones without variation: 0 (collapsed last)
    mandatory_dims = configuration.business_mandatory_dims
    positions = sorted(position for position in decision.loc[decision["grupo"] == "mandatory", "orden_colapso"] if position > 0)
    without_variation = int((decision.loc[decision["grupo"] == "mandatory", "orden_colapso"] == 0).sum())
    configuration.log_check(STEP_2_1_LABEL, check_log, "every mandatory dimension with variation has one collapse position, 1 to K",
                            positions == list(range(1, len(mandatory_dims) - without_variation + 1)),
                            failure_detail=f"positions found: {positions}",
                            context=f"{len(positions)} with a pass · {without_variation} without variation")

    # [4] the hierarchy: a coarser level never goes before a finer one of its family
    position_of = {dimension: position for position, (dimension, _) in enumerate(collapse_order, 1)}
    broken = []
    ordered_dims = [dimension for dimension in mandatory_dims if dimension in position_of]      # the ones that are a pass
    for dimension in ordered_dims:
        family, level = family_and_level(dimension, ordered_dims)
        finer = [other for other in ordered_dims if family_and_level(other, ordered_dims) == (family, level + 1)]
        for finer_dimension in finer:
            if position_of[dimension] < position_of[finer_dimension]:
                broken.append(f"{dimension} before {finer_dimension}")
    configuration.log_check(STEP_2_1_LABEL, check_log, "a *_level_N never collapses before its *_level_(N+1)", not broken,
                            failure_detail=f"hierarchy broken: {broken}")


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.2 · LADDER GROUPS  (until 25-oct-2026: step_08_ladder_groups.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The ladder: the series are merged pass by pass into bigger groups
# until every group has enough support, and every group gets a credibility reference.
#
# THE IDEA: the forecast series of the raw (with its gaps) are the starting point. Each PASS gives
# every series a group id. Grouping the numbers by the id of any pass gives the same totals as
# the raw, but fewer and bigger series, with better binomial support. The report of the passes
# shows how much each pass improves the support.
#
# THE PASSES, in this order (a pass is skipped if it has nothing to do):
#   0            itself: the group id is the series itself
#   1            sign: the timevarying flags are summarised as their sign (SIG=negativo, …)
#   extras       one pass per extra_renovacion, the least informative first (unique contribution
#                of step 2.1): its value becomes '*'
#   mandatory    one pass per mandatory dim, in the collapse order of step 2.1 (the dim whose loss
#                costs the least R² first; a *_level_N before its *_level_N−1): its value becomes '*'
#
# THE RULE OF A PASS: group the series by the id of the previous pass. A group with a support
# ≥ support_floor (30) is CLOSED: it keeps its id in every later pass. The series of an open group
# take the id of this pass (one more dim set to '*'). So every pass is a partition: each series
# is in one group, and the totals by group always add up to the raw.
#   · a signed group (negativo / positivo) keeps its sign: it only goes through the mandatory
#     passes while the cumulative loss of R² is ≤ signed_ladder_max_loss
#   · a mixed series (flags of both signs) is never merged: it stays itself
#   · support of a group = the median, over its months with something due, of the units due
#     summed over its series
#
# AN ID ONLY CHANGES WHEN THE GROUP CHANGES: when a pass sets one more dim to '*' but the group
# keeps exactly the same series, it keeps the id it had. So reading the ids pass after pass, an id
# changes only where the series really joined others.
#
# THE STAGES: the passes are summarised in the 4 stages business reads (one column pair per stage
# in the core, stage_id and stage_support):
#   stage 0  raw      the forecast series itself (its gaps filled)
#   stage 1  sign     after the sign pass
#   stage 2  extras   after the last extra pass
#   stage 3  collapse after the last mandatory pass: the COMPOSITION, what is predicted
#   (stage 4, credibility, is step 2.3: it blends rates, it does not merge series)
# A stage that adds nothing repeats the id and the support of the stage before it.
#
# THE COMPOSITION of a series is its id after the last pass. The group lends its rate to every
# series in it (step 2.3). A final group below own_rate_floor (271) gets a CREDIBILITY REFERENCE:
# a wider group whose rate is blended with the group's own, z = n / (n + k). The reference is the
# first of these candidates that has more series than the group and reaches support_floor
# (else the widest one that has more series):
#   · the group's own id, counted over ALL the series (also the ones closed in earlier passes,
#     the big ones included)
#   · the id of each later pass its sign allows, counted over all the series
# A reference is a source of rate, not a group of the partition: the same series can be in the
# reference of many groups. Every series has ONE reference, so the series still add up by it,
# but the rate of a reference is computed with every series that matches it.
#
# Actions (logged as they are done):
#   1. the plan of the passes, from the decision of step 2.1
#   2. the estimable series with their sign and their columns
#   3. the id of every series at every pass (and of every possible pattern, for the references)
#   4. the passes: close the groups that reach the floor, move the others one pass up
#   5. the final group of every series, and the credibility reference of every final group
#   6. check the partition, the signs, the closed groups and the references      checks 1-4
#   7. write the passes, their summary, the merges and the composition members    checks 5-8
#   8. count the checks; stop if any failed
#   9. show how every pass improves the support, as a table
#
# Output: (the passes: series × pass with its group id, the summary by pass, the final group of
# every series with its reference, the members of every reference) · tables sff_ladder_steps,
# sff_ladder_summary, sff_ladder_merges, sff_composition_members (the final groups stay in memory).

STEP_2_2_LABEL = "2.2"
STEP_2_2_NAME = "LADDER GROUPS"
STEP_2_2_PURPOSE = ("merge the forecast series pass by pass (sign, extras, mandatory dims in the collapse order) until every "
                "group has enough support: every pass is a partition, so the totals always add up and the series get "
                "fewer and bigger; give every final group below the own-rate floor a wider credibility reference")
STEP_2_2_ACTIONS = ["the plan of the passes, from the decision of step 2.1",
                "the estimable series with their sign and their columns",
                "the id of every series at every pass (and of every possible pattern, for the references)",
                "the passes: close the groups that reach the floor, move the others one pass up",
                "the final group of every series, and the credibility reference of every final group",
                "check the partition, the signs, the closed groups and the references (checks 1-4)",
                "write the passes, their summary, the merges and the composition members (checks 5-8)",
                "count the checks; stop if any failed",
                "show how every pass improves the support, as a table"]
STEP_2_2_OUTPUT = ("series × pass with its group id · the summary by pass · the final group and the reference of every series · "
               "tables sff_ladder_steps, sff_ladder_summary, sff_ladder_merges, sff_composition_members")



def build_ladder_groups(rated_units: pd.DataFrame, series_rate: pd.DataFrame, series_table: pd.DataFrame,
                        dimension_decision: pd.DataFrame, configuration: Config) -> dict:
    """The passes of the ladder, their summary, the final group and reference of every series.

    INPUT:   rated_units (step 1.7: the rate of every unit, gaps included) · series_rate (step 1.7:
             route, sign, support of every series) · the series table (step 1.6: the columns of
             every series) · the dimension decision (step 2.1) · the Config.
    OUTPUT:  dict(steps, summary, groups, reference_members).
    RULES:   see the module header.
    EDGE CASES: no timevarying dim → no sign pass. No extra → no extra pass. The passes stop as
             soon as no group is open. A group that never reaches the floor keeps its last id.
    """
    configuration.log_step_start(STEP_2_2_LABEL, STEP_2_2_NAME, STEP_2_2_PURPOSE, STEP_2_2_ACTIONS, STEP_2_2_OUTPUT)
    check_log = []

    # [1] the plan of the passes
    plan = plan_of_the_passes(dimension_decision, configuration)
    merging = [step["name"] for step in plan if step["merges"]]
    reference_only = [step["name"] for step in plan if not step["merges"]]
    configuration.log_action(STEP_2_2_LABEL, 1, f"{len(merging)} merging passes: " + " → ".join(merging)
                             + (f" · searched only for a credibility reference: {', '.join(reference_only)}" if reference_only else "")
                             + f" · stage 3 merges at most {configuration.collapse_passes} dims · a signed group may lose up to "
                               f"{configuration.signed_ladder_max_loss} of R² · floor {configuration.support_floor:.0f} · "
                               f"own-rate floor {configuration.own_rate_floor:.0f}")

    # [2] the estimable series
    estimable = series_rate[series_rate[ROUTE_COLUMN] == ROUTE_PREDICTABLE][[SERIES_ID_COLUMN, SIGN_COLUMN, "usd_por_predecir"]]
    estimable = estimable.merge(series_table[[SERIES_ID_COLUMN] + configuration.rate_series_columns], on=SERIES_ID_COLUMN)
    estimable = estimable.set_index(SERIES_ID_COLUMN, drop=False)
    configuration.log_action(STEP_2_2_LABEL, 2, f"{len(estimable):,} estimable series · signs "
                                            f"{estimable[SIGN_COLUMN].value_counts().to_dict()}")

    # [3] the id of every series at every pass, as if it moved every time
    patterns = pd.DataFrame({step["step"]: pattern_of(estimable, configuration, step["summarise_sign"], step["starred"])
                             for step in plan})
    history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, configuration.period_col,
                                                              configuration.renewed_units_col, configuration.pipeline_units_col]]
    history = history[history[SERIES_ID_COLUMN].isin(estimable.index)]
    configuration.log_action(STEP_2_2_LABEL, 3, f"{len(plan)} patterns per series · {len(history):,} history months")

    # [4] the passes
    passes = run_the_passes(estimable, patterns, plan, history, configuration)
    steps, summary, final = passes["steps"], passes["summary"], passes["final"]
    stages = stages_of_the_series(steps, plan)
    members = composition_members(estimable, patterns, final)
    merges = ladder_merges(steps, series_rate, passes["pattern_stats"], patterns)
    lenders = members.merge(final[[COMPOSITION_ID_COLUMN]].reset_index(), on=[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN], how="left",
                            indicator=True)
    members["role"] = np.where(lenders["_merge"].to_numpy() == "both", "uses", "lends")
    improving = int(merges["improves"].sum()) if len(merges) else 0
    configuration.log_action(STEP_2_2_LABEL, 4, f"{len(summary)} passes run · ids {summary['groups'].iloc[0]:,} → {summary['groups'].iloc[-1]:,} · "
                                            f"{int(final['closed'].sum()):,} of {len(final):,} series reach the floor · "
                                            f"{len(merges):,} merges, {improving:,} improve the error of the series' monthly rate · "
                                            f"{int((members['role'] == 'lends').sum()):,} series lend their history to a composition they do not use")

    # [5] the final groups and their references
    groups, reference_members = credibility_references(estimable, patterns, plan, final, passes["pattern_stats"], configuration)
    with_reference = groups["credibility_ref_id"].notna()
    configuration.log_action(STEP_2_2_LABEL, 5, f"{groups[COMPOSITION_ID_COLUMN].nunique():,} final groups · "
                                            f"{int(with_reference.sum()):,} series take a credibility reference "
                                            f"({reference_members['credibility_ref_id'].nunique():,} references)")

    # [6] the checks
    configuration.log_action(STEP_2_2_LABEL, 6, "checking the partition, the signs, the closed groups and the references")
    check_the_ladder(steps, summary, groups, estimable, history, configuration, check_log)

    # [7] the tables
    configuration.log_action(STEP_2_2_LABEL, 7, "writing the passes, their summary, the merges and the composition members")
    configuration.write_table(STEP_2_2_LABEL, check_log, steps, TABLE_LADDER_STEPS)
    configuration.write_table(STEP_2_2_LABEL, check_log, summary, TABLE_LADDER_SUMMARY)
    configuration.write_table(STEP_2_2_LABEL, check_log, merges, TABLE_LADDER_MERGES)
    configuration.write_table(STEP_2_2_LABEL, check_log, members, TABLE_COMPOSITION_MEMBERS)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_2_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_2_2_LABEL, STEP_2_2_NAME, check_log)

    # [9] how every pass improves the support
    configuration.log_action(STEP_2_2_LABEL, 9, "how every pass improves the support (groups: fewer and bigger; the units due "
                                            "add up in every pass; pct_usd_*: money to predict in groups that reach the floor):")
    configuration.show_table(summary)
    configuration.logger.doc(f"[{STEP_2_2_LABEL}] the 4 stages (0 raw · 1 sign · 2 extras · 3 collapse: the composition), "
                             f"grouping the forecast series by the id of each stage:")
    summary_by_stage = stage_summary(stages, estimable, history, configuration)
    configuration.show_table(summary_by_stage)
    return dict(steps=steps, summary=summary, groups=groups, reference_members=reference_members, stages=stages,
                stage_summary=summary_by_stage, composition_members=members, merges=merges)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PLAN AND THE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════════

def plan_of_the_passes(dimension_decision: pd.DataFrame, configuration: Config) -> list:
    """The passes in order: itself, sign, one per extra, one per mandatory dim. Each pass says whether
    the sign is summarised, which dims are '*' so far, and whether a signed group may take it."""
    mandatory_decision = dimension_decision[dimension_decision["orden_colapso"] > 0].sort_values("orden_colapso")
    collapse_order = list(mandatory_decision["dimension"])
    collapse_order += [dimension for dimension in configuration.business_mandatory_dims if dimension not in collapse_order]
    collapse_loss = dict(zip(mandatory_decision["dimension"], mandatory_decision["perdida_secuencial"].fillna(1.0)))
    extras = dimension_decision[dimension_decision["grupo"] == "extra_renovacion"] if "grupo" in dimension_decision else dimension_decision.iloc[0:0]
    extra_order = list(extras.sort_values(["contribucion_unica", "dimension"])["dimension"])
    extra_order += [extra for extra in configuration.extra_renovacion if extra not in extra_order]

    plan = [dict(step=0, name="itself", summarise_sign=False, starred=frozenset(), signed_allowed=True)]
    if configuration.structural_timevarying_dims:
        plan.append(dict(step=len(plan), name="sign", summarise_sign=True, starred=frozenset(), signed_allowed=True))
    starred = set()
    for extra in extra_order:
        starred = starred | {extra}
        plan.append(dict(step=len(plan), name=f"extra {extra}", summarise_sign=True, starred=frozenset(starred),
                         signed_allowed=True))
    cumulative_loss = 0.0
    for collapse_index, dimension in enumerate(collapse_order, start=1):
        starred = starred | {dimension}
        cumulative_loss += float(collapse_loss.get(dimension, 1.0))
        signed_allowed = (configuration.signed_ladder_max_loss > 0
                          and cumulative_loss <= configuration.signed_ladder_max_loss + configuration.support_tolerance)
        # stage 3 merges at most collapse_passes dims; the passes after them are only searched for a
        # credibility reference (stage 4), never to merge
        plan.append(dict(step=len(plan), name=f"without {dimension}", summarise_sign=True, starred=frozenset(starred),
                         signed_allowed=signed_allowed, merges=collapse_index <= configuration.collapse_passes))
    for step in plan:
        step.setdefault("merges", True)
    return plan


def pattern_of(estimable: pd.DataFrame, configuration: Config, summarise_sign: bool, starred: set) -> pd.Series:
    """The group id of every series at one pass: the rate series columns in order; the timevarying
    block replaced by SIG=<sign> when summarised; '*' on the starred dims."""
    timevarying = list(configuration.structural_timevarying_dims)
    fields = []
    sign_written = False
    for column_name in configuration.rate_series_columns:
        if column_name in timevarying and summarise_sign:
            if not sign_written:
                fields.append(SIGN_TOKEN + estimable[SIGN_COLUMN].astype(str).to_numpy(dtype=object))
                sign_written = True
            continue
        if column_name in starred:
            fields.append(np.full(len(estimable), WILDCARD, dtype=object))
        else:
            fields.append(np.asarray(id_text(estimable[column_name]), dtype=object))
    if summarise_sign and not sign_written:            # no timevarying declared: the sign still separates groups
        fields.append(SIGN_TOKEN + estimable[SIGN_COLUMN].astype(str).to_numpy(dtype=object))
    joined = fields[0]
    for field in fields[1:]:
        joined = joined + ID_FIELD_SEPARATOR + field
    return pd.Series(joined, index=estimable.index)


def support_of_groups(history: pd.DataFrame, group_of_series: pd.Series, configuration: Config) -> pd.DataFrame:
    """Per group: support (median of the monthly units due summed over its series), units due and
    renewed in total, and its number of series."""
    joined = history.assign(_group=history[SERIES_ID_COLUMN].map(group_of_series)).dropna(subset=["_group"])
    monthly = (joined.groupby(["_group", configuration.period_col])
               [[configuration.renewed_units_col, configuration.pipeline_units_col]].sum().reset_index())
    with_pipeline = monthly[monthly[configuration.pipeline_units_col] > 0]
    result = pd.DataFrame({
        "support": with_pipeline.groupby("_group")[configuration.pipeline_units_col].median(),
        "renewed": monthly.groupby("_group")[configuration.renewed_units_col].sum(),
        "due": monthly.groupby("_group")[configuration.pipeline_units_col].sum(),
        "series": group_of_series.dropna().groupby(group_of_series.dropna()).size()})
    result[["support", "renewed", "due"]] = result[["support", "renewed", "due"]].fillna(0.0)
    result["rate"] = result["renewed"] / result["due"].where(result["due"] > 0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PASSES
# ═══════════════════════════════════════════════════════════════════════════════════

def run_the_passes(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, history: pd.DataFrame,
                   configuration: Config) -> dict:
    """The mechanical rule: at every pass EVERY forecast series is grouped by the same pattern (its dims
    with the ones removed so far set to '*'); each series uses the first pass whose group reaches the
    support floor. A series that reaches it on its own (pass 0) keeps its own id, but its history still
    counts in the groups of the others. Mixed series never merge. A series that never reaches the floor
    keeps the widest group it was allowed."""
    floor = configuration.support_floor
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    neutral = estimable[SIGN_COLUMN] == SIGN_NEUTRAL
    merge_passes = [step for step in plan if step["merges"]]

    # the support, rate and number of series of every pattern of every pass, over ALL the series
    pattern_stats = {}
    for step in plan:
        pattern_of_series = patterns[step["step"]] if step["step"] == 0 else patterns[step["step"]].where(~mixed)
        pattern_stats[step["step"]] = support_of_groups(history, pattern_of_series, configuration)

    # the pass every series uses: the first one (allowed for its sign) whose group reaches the floor
    chosen_step = pd.Series(np.nan, index=estimable.index)
    last_allowed = pd.Series(0, index=estimable.index)
    for step in merge_passes:
        allowed = (step["step"] == 0) | (~mixed & (neutral | step["signed_allowed"]))
        last_allowed[allowed] = step["step"]
        support = patterns[step["step"]].map(pattern_stats[step["step"]]["support"]).fillna(0.0)
        reaches = allowed & chosen_step.isna() & (support >= floor - configuration.support_tolerance)
        chosen_step[reaches] = step["step"]
    reached = chosen_step.notna()
    final_step = chosen_step.fillna(last_allowed).astype(int)

    # the id of every series at every pass: the pattern while it climbs, frozen once it has its group;
    # an id only changes when the group really gets new series (the same series: the previous id)
    ids, step_rows, summary_rows = {}, [], []
    usd_to_predict = estimable["usd_por_predecir"]
    total_usd = usd_to_predict.sum()
    previous_id, previous_count = None, None
    for step in merge_passes:
        climbing = step["step"] <= final_step
        allowed = (step["step"] == 0) | (~mixed & (neutral | step["signed_allowed"]))
        moving = climbing & allowed
        pattern = patterns[step["step"]]
        count = pattern.map(pattern_stats[step["step"]]["series"]).fillna(1)
        if previous_id is None:
            current_id = pattern.copy()
        else:
            current_id = previous_id.copy()
            new_series_joined = moving & (count > previous_count)
            current_id[new_series_joined] = pattern[new_series_joined]
        current_count = count.where(moving, previous_count if previous_count is not None else count)
        support = pattern.map(pattern_stats[step["step"]]["support"]).fillna(0.0)
        if previous_id is not None:
            support = support.where(moving, step_rows[-1].set_index(SERIES_ID_COLUMN)["group_support"])
        ids[step["step"]] = current_id
        has_group = step["step"] >= final_step
        step_rows.append(pd.DataFrame({SERIES_ID_COLUMN: estimable.index, "ladder_step": step["step"], "step_name": step["name"],
                                       "group_id": current_id.to_numpy(copy=True), "group_support": support.to_numpy(copy=True),
                                       "series_in_group": current_count.to_numpy(copy=True),
                                       "closed": (has_group & (support >= floor - configuration.support_tolerance)).astype(int).to_numpy()}))
        partition = support_of_groups(history, current_id, configuration)
        group_support = pd.Series(support.to_numpy(), index=current_id.to_numpy()).groupby(level=0).first()
        summary_rows.append({
            "ladder_step": step["step"], "step_name": step["name"], "groups": int(current_id.nunique()),
            "open_groups": int((group_support < floor - configuration.support_tolerance).sum()),
            "median_group_support": float(group_support.median()),
            "units_due": float(partition["due"].sum()),
            "pct_usd_floor": float(usd_to_predict[support >= floor - configuration.support_tolerance].sum() / total_usd) if total_usd else np.nan,
            "pct_usd_own_rate": float(usd_to_predict[support >= configuration.own_rate_floor].sum() / total_usd) if total_usd else np.nan})
        previous_id, previous_count = current_id, current_count

    # the composition of every series: its id at the pass it uses, with the series in its rate
    composition = ids[merge_passes[-1]["step"]]
    final_pattern = pd.Series(patterns.to_numpy()[np.arange(len(patterns)), patterns.columns.get_indexer(final_step.to_numpy())],
                              index=estimable.index)
    final = pd.DataFrame({COMPOSITION_ID_COLUMN: composition, "final_step": final_step, "final_pattern": final_pattern,
                          "closed": reached})
    return dict(steps=pd.concat(step_rows, ignore_index=True), summary=pd.DataFrame(summary_rows), final=final,
                pattern_stats=pattern_stats)


def composition_members(estimable: pd.DataFrame, patterns: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Every series in the rate of every composition: the series whose pattern, at the pass the composition
    uses, is the composition's (the series that use it and the ones that only lend their history)."""
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    rows = []
    for step_number, compositions in final.drop_duplicates([COMPOSITION_ID_COLUMN, "final_step", "final_pattern"]).groupby("final_step"):
        # every series of that pass with its pattern, then joined to the compositions that use that pass
        candidates = patterns[step_number] if step_number == 0 else patterns[step_number][~mixed]
        series_of_pattern = pd.DataFrame({"final_pattern": candidates.to_numpy(), SERIES_ID_COLUMN: candidates.index})
        rows.append(compositions[[COMPOSITION_ID_COLUMN, "final_pattern"]].merge(series_of_pattern, on="final_pattern")
                    [[COMPOSITION_ID_COLUMN, SERIES_ID_COLUMN]])
    return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)


def ladder_merges(steps: pd.DataFrame, series_rate: pd.DataFrame, pattern_stats: dict, patterns: pd.DataFrame) -> pd.DataFrame:
    """Every merge of every forecast series (a pass where its id changed): the group before and after,
    and whether it improves the prediction of its monthly rate: error alone √(p(1−p)/n) against error
    merged √(bias² + q(1−q)/N), bias = rate of the new group − its own rate."""
    ordered = steps.sort_values([SERIES_ID_COLUMN, "ladder_step"])
    ordered = ordered.assign(previous_id=ordered.groupby(SERIES_ID_COLUMN)["group_id"].shift(1),
                             previous_support=ordered.groupby(SERIES_ID_COLUMN)["group_support"].shift(1))
    changes = ordered[ordered["previous_id"].notna() & (ordered["group_id"] != ordered["previous_id"])].copy()
    if changes.empty:
        return changes
    own = series_rate.set_index(SERIES_ID_COLUMN)[["n_propio", "tasa_propia"]]
    changes = changes.join(own, on=SERIES_ID_COLUMN)
    changes["group_rate"] = [pattern_stats[int(step_number)]["rate"].get(patterns.loc[series_id, int(step_number)], np.nan)
                             for series_id, step_number in zip(changes[SERIES_ID_COLUMN], changes["ladder_step"])]
    own_rate, own_n = changes["tasa_propia"], changes["n_propio"].clip(lower=1)
    group_rate, group_n = changes["group_rate"], changes["group_support"].clip(lower=1)
    changes["bias_pp"] = 100 * (group_rate - own_rate)
    changes["error_alone_pp"] = 100 * np.sqrt(own_rate * (1 - own_rate) / own_n)
    changes["error_merged_pp"] = 100 * np.sqrt((group_rate - own_rate) ** 2 + group_rate * (1 - group_rate) / group_n)
    changes["improves"] = (changes["error_merged_pp"] < changes["error_alone_pp"]).astype(int)
    return changes[[SERIES_ID_COLUMN, "ladder_step", "step_name", "previous_id", "previous_support", "group_id", "group_support",
                    "series_in_group", "n_propio", "tasa_propia", "group_rate", "bias_pp", "error_alone_pp", "error_merged_pp",
                    "improves"]].rename(columns={"n_propio": "own_support", "tasa_propia": "own_rate"})


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STAGES
# ═══════════════════════════════════════════════════════════════════════════════════

STAGE_NAMES = {0: "raw", 1: "sign", 2: "extras", 3: "collapse"}


def stage_of_each_pass(plan: list) -> dict:
    """The stage of every pass: 0 itself, 1 the sign, 2 the extras, 3 the mandatory dims."""
    stage_of_pass = {}
    for step in plan:
        if step["step"] == 0:
            stage_of_pass[step["step"]] = 0
        elif step["name"] == "sign":
            stage_of_pass[step["step"]] = 1
        elif step["name"].startswith("extra"):
            stage_of_pass[step["step"]] = 2
        else:
            stage_of_pass[step["step"]] = 3
    return stage_of_pass


def stages_of_the_series(steps: pd.DataFrame, plan: list) -> pd.DataFrame:
    """Per forecast series: the id and the support at the end of every stage (the last pass run that
    belongs to the stage or an earlier one; a stage that adds nothing repeats the one before)."""
    stage_of_pass = stage_of_each_pass(plan)
    steps = steps.assign(stage=steps["ladder_step"].map(stage_of_pass))
    stage_columns = {}
    for stage in sorted(STAGE_NAMES):
        last_pass = steps[steps["stage"] <= stage].sort_values("ladder_step").groupby(SERIES_ID_COLUMN).tail(1)
        last_pass = last_pass.set_index(SERIES_ID_COLUMN)
        stage_columns[f"stage{stage}_id"] = last_pass["group_id"]
        stage_columns[f"stage{stage}_support"] = last_pass["group_support"]
    return pd.DataFrame(stage_columns).reset_index()


def stage_summary(stages: pd.DataFrame, estimable: pd.DataFrame, history: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per stage: groups, median support, money to predict in groups that reach the floor and the own-rate
    floor, and the units due (the same in every stage: each stage is a partition of the raw)."""
    usd = estimable.set_index(SERIES_ID_COLUMN)["usd_por_predecir"] if SERIES_ID_COLUMN in estimable.columns else estimable["usd_por_predecir"]
    total_usd = usd.sum()
    rows = []
    for stage, name in STAGE_NAMES.items():
        ids = stages.set_index(SERIES_ID_COLUMN)[f"stage{stage}_id"]
        support = stages.set_index(SERIES_ID_COLUMN)[f"stage{stage}_support"]
        groups = support_of_groups(history, ids, configuration)
        rows.append({"stage": stage, "stage_name": name, "groups": int(ids.nunique()),
                     "median_group_support": float(groups["support"].median()),
                     "units_due": float(groups["due"].sum()),
                     "pct_usd_floor": float(usd.reindex(ids.index)[support >= configuration.support_floor].sum() / total_usd) if total_usd else np.nan,
                     "pct_usd_own_rate": float(usd.reindex(ids.index)[support >= configuration.own_rate_floor].sum() / total_usd) if total_usd else np.nan})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CREDIBILITY REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════════

def credibility_references(estimable: pd.DataFrame, patterns: pd.DataFrame, plan: list, final: pd.DataFrame,
                           pattern_stats: dict, configuration: Config) -> tuple:
    """The composition of every series with its support, rate and series (every series in its rate) and,
    below the own-rate floor, its credibility reference: the first later pass (merging or not: stage 4 may
    look wider than stage 3 merges) allowed for its sign whose group has more series and reaches the floor,
    else the widest with more series. And the members of every reference."""
    floor, own_rate_floor = configuration.support_floor, configuration.own_rate_floor
    mixed = estimable[SIGN_COLUMN] == SIGN_MIXED
    users = final.groupby(COMPOSITION_ID_COLUMN).size()
    group_rows = []
    for (composition_id, step_number, pattern), members in final.groupby([COMPOSITION_ID_COLUMN, "final_step", "final_pattern"]):
        stats = pattern_stats[int(step_number)].loc[pattern]
        first_member = members.index[0]
        sign = estimable.loc[first_member, SIGN_COLUMN]
        chosen_step, chosen_id = None, None
        if stats["support"] < own_rate_floor - configuration.support_tolerance and sign != SIGN_MIXED:
            widest = None
            for step in plan:
                if step["step"] <= step_number or not (sign == SIGN_NEUTRAL or step["signed_allowed"]):
                    continue
                candidate_id = patterns.loc[first_member, step["step"]]
                candidate = pattern_stats[step["step"]].loc[candidate_id]
                if candidate["series"] > stats["series"]:
                    widest = (step["step"], candidate_id)
                    if candidate["support"] >= floor - configuration.support_tolerance:
                        break
            if widest is not None:
                chosen_step, chosen_id = widest
        row = {COMPOSITION_ID_COLUMN: composition_id, "final_step": int(step_number), "group_series": int(stats["series"]),
               "composition_users": int(users.get(composition_id, 0)), "group_support": float(stats["support"]),
               "group_rate": float(stats["rate"]), "credibility_ref_id": chosen_id, "credibility_ref_step": chosen_step}
        if chosen_id is not None:
            reference = pattern_stats[chosen_step].loc[chosen_id]
            row.update(ref_series=int(reference["series"]), ref_support=float(reference["support"]), ref_rate=float(reference["rate"]))
        group_rows.append(row)
    by_composition = pd.DataFrame(group_rows)
    for column_name in ("ref_series", "ref_support", "ref_rate"):
        if column_name not in by_composition.columns:
            by_composition[column_name] = np.nan

    groups = final[[COMPOSITION_ID_COLUMN]].reset_index().merge(by_composition, on=COMPOSITION_ID_COLUMN, how="left")
    reference_rows = []
    for (ref_step, ref_id), _ in by_composition.dropna(subset=["credibility_ref_id"]).groupby(["credibility_ref_step", "credibility_ref_id"]):
        members = patterns.index[(patterns[int(ref_step)] == ref_id) & ~mixed]
        reference_rows.append(pd.DataFrame({"credibility_ref_id": ref_id, SERIES_ID_COLUMN: members}))
    reference_members = (pd.concat(reference_rows, ignore_index=True) if reference_rows
                         else pd.DataFrame(columns=["credibility_ref_id", SERIES_ID_COLUMN]))
    return groups, reference_members


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════

def check_the_ladder(steps: pd.DataFrame, summary: pd.DataFrame, groups: pd.DataFrame, estimable: pd.DataFrame,
                     history: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 to 4."""
    # [1] every pass is a partition: one group per series, and the units due add up to the raw
    raw_units_due = float(history[configuration.pipeline_units_col].sum())
    one_group = steps.groupby("ladder_step")[SERIES_ID_COLUMN].agg(lambda ids: ids.is_unique and len(ids) == len(estimable)).all()
    adds_up = bool(((summary["units_due"] - raw_units_due).abs() <= configuration.money_tolerance).all())
    configuration.log_check(STEP_2_2_LABEL, check_log, "every pass is a partition: one group per series, the units due add up",
                            bool(one_group) and adds_up,
                            failure_detail=f"units due by pass {summary['units_due'].tolist()} vs {raw_units_due:,.0f}",
                            context=f"{len(summary)} passes · {raw_units_due:,.0f} units due in every one")

    # [2] a group never mixes signs
    sign_of_series = estimable[SIGN_COLUMN]
    signs_per_group = steps.assign(_sign=steps[SERIES_ID_COLUMN].map(sign_of_series)).groupby(["ladder_step", "group_id"])["_sign"].nunique()
    configuration.log_check(STEP_2_2_LABEL, check_log, "a group never mixes signs", bool((signs_per_group <= 1).all()),
                            failure_detail=f"{int((signs_per_group > 1).sum()):,} groups with more than one sign")

    # [3] a closed group keeps its id in every later pass, and the groups only get fewer
    ordered = steps.sort_values([SERIES_ID_COLUMN, "ladder_step"])
    previous_closed = ordered.groupby(SERIES_ID_COLUMN)["closed"].shift(1).fillna(0).astype(int)
    previous_id = ordered.groupby(SERIES_ID_COLUMN)["group_id"].shift(1)
    reopened = ordered[(previous_closed == 1) & (ordered["group_id"] != previous_id)]
    fewer = bool((summary["groups"].diff().fillna(0) <= 0).all())
    configuration.log_check(STEP_2_2_LABEL, check_log, "a closed group keeps its id, and every pass has fewer or equal groups",
                            reopened.empty and fewer,
                            failure_detail=f"{len(reopened):,} series left a closed group · groups by pass {summary['groups'].tolist()}")

    # [4] a reference is wider than its group
    with_reference = groups.dropna(subset=["credibility_ref_id"]).drop_duplicates(COMPOSITION_ID_COLUMN)
    narrower = with_reference[with_reference["ref_series"] <= with_reference["group_series"]]
    configuration.log_check(STEP_2_2_LABEL, check_log, "every credibility reference has more series than its group", narrower.empty,
                            failure_detail=f"{len(narrower):,} references not wider than their group",
                            context=f"{len(with_reference):,} groups with a reference")


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.3 · LADDER  (until 25-oct-2026: step_09_ladder.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# Every series takes the rate of its final group (step 2.2), reinforced by
# credibility with the group's reference; and gets its two errors and its risk level.
#
# "The group lends its rate to its series; a group without precision trusts its own rate in
# proportion to how much support it has, and the rest comes from a wider reference."
#
# THE RATE OF THE GROUP: Σ renewed / Σ due of all the series of the final group (step 2.2).
# THE CREDIBILITY (only for a final group below own_rate_floor, 271, with a reference):
#   z = n / (n + k), n = the support of the group, k = Bühlmann-Straub k of its reference,
#   estimated from the final groups that share that reference: within variance / between
#   variance of their rates (fewer than 3 groups: k_cred; no between variance: 10 × k_cred).
#   estimated rate = z · rate of the group + (1 − z) · rate of the reference
#   A group at or above own_rate_floor predicts alone: z = 1.
# Every series of a group gets the same estimated rate (the group lends it). Two errors, in pp:
#   · se_estimacion: the error of the ESTIMATE, √(z²·se_group² + (1−z)²·se_reference²)
#   · se_prediccion: √(se_estimacion² + se_binomial(rate, n_propio)²): next month still samples
#     with the series' OWN size. It never shrinks below its own binomial noise.
# THE RISK LEVEL: how the rate was obtained, by the pass where the group of the series was formed:
#   A  the series itself (pass 0) with own precision (≥ own_rate_floor), a year of history
#   A2 the same, less than a year
#   A3 the series itself with evidence (≥ support_floor) but not precision: credibility
#   B  merged in the sign or extra passes: its group shares EVERY mandatory dim with it
#   C  merged in a mandatory pass (a mandatory dim collapsed), or never reached the floor
#   S  signed and its group never reached the floor
#   M  mixed signs: never merged · D solo_futuro · N solo_historia
#
# Series that are not estimable (solo_historia, solo_futuro, mixto) get their row
# too: their own group, their own rate if they have one, and their level.
#
# Actions (logged as they are done):
#   1. the final group and the reference of every estimable series (step 2.2)
#   2. the credibility k of every reference, from the groups that share it
#   3. the rate and the two errors of every series (the others: their own rate, if any)
#   4. the risk level of every series
#   5. check the rates and the errors                                   checks 1-4
#   6. write the estimate of every series and the money by level        checks 5-6
#   7. count the checks; stop if any failed
#   8. show the money by risk level and what each level means, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every series has exactly one estimate
#    2. every series of a group has the same estimated rate
#    3. every estimated rate and every z is between 0 and 1
#    4. the prediction error is never below the estimation error
#    5. table sff_niveles_riesgo written and read back
#
# Output: one row per series with its final group, reference, k, z, estimated rate, two errors
# and risk level (in memory: it reaches the forecast series dimension) · the money by risk level ·
# table sff_niveles_riesgo.

STEP_2_3_LABEL = "2.3"
STEP_2_3_NAME = "LADDER"
STEP_2_3_PURPOSE = ("give every series the rate of its final group, reinforced by credibility with the group's reference "
                "when the group has no precision of its own; the error of that estimate, the error of next month's "
                "prediction, and the risk level of how the rate was obtained")
STEP_2_3_ACTIONS = ["the final group and the reference of every estimable series (step 2.2)",
                "the credibility k of every reference, from the groups that share it",
                "the rate and the two errors of every series (the others: their own rate, if any)",
                "the risk level of every series",
                "check the rates and the errors (checks 1-4)",
                "write the money by level (check 5)",
                "count the checks; stop if any failed",
                "show the money by risk level and what each level means, as tables"]
STEP_2_3_OUTPUT = ("one row per series: final group, reference, k, z, estimated rate, se_estimacion, se_prediccion, risk level · "
               "the money by level · table sff_niveles_riesgo")

PERCENTAGE_POINTS = 100

# What each risk level means, for the table of action 8 (the rule, then its meaning).
LEVEL_DEFINITIONS = [
    (LEVEL_OWN, "itself, n ≥ own_rate_floor, ≥ own_level_min_history_months", "own precision: predicts alone, z = 1"),
    (LEVEL_OWN_SHORT, "itself, n ≥ own_rate_floor, < own_level_min_history_months", "own precision but a short history: it has not seen every season"),
    (LEVEL_OWN_REINFORCED, "itself, support_floor ≤ n < own_rate_floor", "own evidence without precision: credibility with its reference"),
    (LEVEL_BORROWED, "merged in the sign or extra passes", "its group shares EVERY mandatory dim with it"),
    (LEVEL_FAR, "merged in a mandatory pass, or never reached the floor", "its group collapsed a mandatory dim"),
    (LEVEL_SIGNED_UNDER_FLOOR, "signed, its group never reached the floor", "the best rate of ITS sign, noisy: it may not mix with unsigned series"),
    (LEVEL_MIXED, "flags of both signs", "never merged: keeps its own rate; should be ≈ 0"),
    (LEVEL_NO_HISTORY, ROUTE_FUTURE_ONLY, "no rate to estimate: it will take a rate from its cell later"),
    (LEVEL_NO_IMPACT, ROUTE_HISTORY_ONLY, "nothing to predict: kept for the groups and the backtest"),
]


def binomial_se_pp(rates, supports) -> np.ndarray:
    """The binomial standard error of a rate, in pp: 100 · √(p(1 − p) / n), n at least 1."""
    rates = np.asarray(rates, dtype=float)
    supports = np.maximum(np.asarray(supports, dtype=float), 1.0)
    return PERCENTAGE_POINTS * np.sqrt(rates * (1 - rates) / supports)


def climb_the_ladder(series_rate: pd.DataFrame, ladder: dict, configuration: Config) -> tuple:
    """The rate of every series from its final group and its reference; the errors and the risk level.

    INPUT:   series_rate (step 1.7) · the ladder of step 2.2 (groups, summary) · the Config.
    OUTPUT:  (series_estimate: one row per series, money_by_level).
    RULES:   see the module header.
    EDGE CASES: a group with no reference, or at or above own_rate_floor, has z = 1. A series that
             is not estimable is its own group with its own rate.
    """
    configuration.log_step_start(STEP_2_3_LABEL, STEP_2_3_NAME, STEP_2_3_PURPOSE, STEP_2_3_ACTIONS, STEP_2_3_OUTPUT)
    check_log = []

    # [1] the final group and the reference of every estimable series
    groups = ladder["groups"]
    step_names = dict(zip(ladder["summary"]["ladder_step"], ladder["summary"]["step_name"]))
    configuration.log_action(STEP_2_3_LABEL, 1, f"{len(groups):,} estimable series in {groups[COMPOSITION_ID_COLUMN].nunique():,} "
                                            f"final groups; {int(groups['credibility_ref_id'].notna().sum()):,} of them "
                                            f"with a credibility reference")

    # [2] the credibility k of every reference
    k_by_reference = credibility_k(groups, configuration)
    configuration.log_action(STEP_2_3_LABEL, 2, f"k estimated for {int((k_by_reference != configuration.k_cred).sum()):,} "
                                            f"references; the others use k_cred = {configuration.k_cred:.0f}")

    # [3] the rate and the two errors of every series
    series_estimate = estimate_rates(series_rate, groups, k_by_reference, configuration)
    configuration.log_action(STEP_2_3_LABEL, 3, f"rate estimated for {int(series_estimate['tasa_estimada'].notna().sum()):,} "
                                            f"of {len(series_estimate):,} series · z < 1 in "
                                            f"{int((series_estimate['z'] < 1).sum()):,}")

    # [4] the risk level
    series_estimate["nivel_riesgo"] = risk_levels(series_estimate, step_names, configuration)
    configuration.log_action(STEP_2_3_LABEL, 4, f"levels: {series_estimate['nivel_riesgo'].value_counts().sort_index().to_dict()}")

    # [5] the checks
    configuration.log_action(STEP_2_3_LABEL, 5, "checking the rates and the errors")
    check_ladder(series_estimate, series_rate, configuration, check_log)

    # [6] the tables, written
    configuration.log_action(STEP_2_3_LABEL, 6, "writing the money by level")
    money_by_level = risk_levels_report(series_estimate)
    configuration.write_table(STEP_2_3_LABEL, check_log, money_by_level, TABLE_RISK_LEVELS)

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_3_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_2_3_LABEL, STEP_2_3_NAME, check_log)

    # [8] the money by level and what each level means
    configuration.log_action(STEP_2_3_LABEL, 8, "money to predict by risk level (error_pp: next month's prediction error "
                                            "of the level, weighted by money):")
    configuration.show_table(money_by_level)
    configuration.logger.doc(f"[{STEP_2_3_LABEL}] what each level means:")
    configuration.show_table(pd.DataFrame(LEVEL_DEFINITIONS, columns=["nivel", "regla", "significado"]))
    return series_estimate, money_by_level


def credibility_k_detail(groups: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Bühlmann-Straub k per credibility reference, with the numbers it is made of, from the
    compositions that share the reference (one row per composition):
      within  σ² = Σ w·p(1 − p) / Σ w: how much a composition's rate moves by chance (per unit due)
      between a  = (Σ w (p − p̄)² − (r − 1) σ²) / (W − Σ w² / W): how much the compositions differ for real
                   (the unbiased Bühlmann-Straub estimator; r compositions, w their support, W = Σ w)
      k = within / between · source 'estimated'
      fewer than 3 compositions: k_cred · source 'default'
      no real difference between them: 10 × k_cred (they take the reference's rate) · source 'homogeneous'"""
    siblings = groups.drop_duplicates(COMPOSITION_ID_COLUMN).dropna(subset=["credibility_ref_id", "group_rate"]).copy()
    siblings = siblings[siblings["group_support"] > 0]
    columns = ["credibility_ref_id", "compositions_using", "within_variance", "between_variance", "k", "k_source"]
    if siblings.empty:
        return pd.DataFrame(columns=columns)
    rates, weights = siblings["group_rate"], siblings["group_support"]
    siblings["_w"] = weights
    siblings["_wp"] = weights * rates
    grouped = siblings.groupby("credibility_ref_id")
    pooled_rate = grouped["_wp"].sum() / grouped["_w"].sum()
    siblings["_pooled"] = siblings["credibility_ref_id"].map(pooled_rate)
    siblings["_w_within"] = weights * rates * (1 - rates)
    siblings["_w_sq_dev"] = weights * (rates - siblings["_pooled"]) ** 2
    siblings["_w_squared"] = weights ** 2
    grouped = siblings.groupby("credibility_ref_id")
    compositions = grouped.size()
    total_weight = grouped["_w"].sum()
    within = grouped["_w_within"].sum() / total_weight
    effective_weight = total_weight - grouped["_w_squared"].sum() / total_weight
    detail = pd.DataFrame({"compositions_using": compositions,
                           "within_variance": within,
                           "between_variance": (grouped["_w_sq_dev"].sum() - (compositions - 1) * within)
                                               / effective_weight.where(effective_weight > 0)})
    has_between = detail["between_variance"] > configuration.min_between_variance
    detail["k"] = np.where(has_between, detail["within_variance"] / detail["between_variance"].where(has_between),
                           configuration.homogeneous_pool_factor * configuration.k_cred)
    detail["k_source"] = np.where(has_between, "estimated", "homogeneous")
    too_few = detail["compositions_using"] < configuration.min_siblings_for_k
    detail.loc[too_few, "k"] = configuration.k_cred
    detail.loc[too_few, "k_source"] = "default"
    return detail.reset_index()[columns]


def credibility_k(groups: pd.DataFrame, configuration: Config) -> pd.Series:
    """The k of every credibility reference (see credibility_k_detail)."""
    detail = credibility_k_detail(groups, configuration)
    return detail.set_index("credibility_ref_id")["k"] if len(detail) else pd.Series(dtype=float)


def estimate_rates(series_rate: pd.DataFrame, groups: pd.DataFrame, k_by_reference: pd.Series,
                   configuration: Config) -> pd.DataFrame:
    """The rate of the group (blended with its reference below own_rate_floor) and the two errors, per series."""
    estimate = series_rate.merge(groups, on=SERIES_ID_COLUMN, how="left")

    # a series that is not estimable is its own group, with its own rate
    not_estimable = estimate[COMPOSITION_ID_COLUMN].isna()
    estimate.loc[not_estimable, COMPOSITION_ID_COLUMN] = estimate.loc[not_estimable, SERIES_ID_COLUMN]
    estimate.loc[not_estimable, "final_step"] = 0
    estimate.loc[not_estimable, "group_series"] = 1
    estimate.loc[not_estimable, "group_support"] = estimate.loc[not_estimable, "n_propio"]
    estimate.loc[not_estimable, "group_rate"] = estimate.loc[not_estimable, "tasa_propia"]
    estimate["final_step"] = estimate["final_step"].astype(int)
    estimate["group_series"] = estimate["group_series"].astype(int)

    group_rate, group_support = estimate["group_rate"], estimate["group_support"].fillna(0.0)
    reference_rate, reference_support = estimate["ref_rate"], estimate["ref_support"]
    estimate["k"] = estimate["credibility_ref_id"].map(k_by_reference).fillna(configuration.k_cred)
    blends = (estimate["credibility_ref_id"].notna() & reference_rate.notna() & group_rate.notna()
              & (group_support < configuration.own_rate_floor - configuration.rounding_tolerance))

    z = np.where(blends, group_support / (group_support + estimate["k"]), 1.0)
    blended_rate = z * group_rate.fillna(0) + (1 - z) * reference_rate.fillna(0)
    blended_se = np.sqrt((z * binomial_se_pp(group_rate.fillna(0.5), group_support)) ** 2
                         + ((1 - z) * binomial_se_pp(reference_rate.fillna(0.5), reference_support.fillna(1))) ** 2)
    alone_se = binomial_se_pp(group_rate.fillna(0.5), group_support)

    estimate["z"] = np.round(z, 3)
    estimate["tasa_estimada"] = np.where(blends, blended_rate, group_rate)
    estimate["se_estimacion_pp"] = np.where(blends, blended_se, np.where(group_rate.notna(), alone_se, np.nan))
    own_noise = binomial_se_pp(np.nan_to_num(estimate["tasa_estimada"], nan=0.5), np.maximum(estimate["n_propio"], 1))
    estimate["se_prediccion_pp"] = np.where(estimate["tasa_estimada"].notna(),
                                            np.sqrt(estimate["se_estimacion_pp"] ** 2 + own_noise ** 2), np.nan)
    estimate["alcanzo_suelo"] = (group_support >= configuration.support_floor - configuration.rounding_tolerance).astype(int)
    # how much the credibility moves the rate of the composition (pp): 0 when it predicts alone
    estimate["credibility_effect_pp"] = PERCENTAGE_POINTS * (estimate["tasa_estimada"] - group_rate)
    return estimate


def risk_levels(estimate: pd.DataFrame, step_names: dict, configuration: Config) -> np.ndarray:
    """The risk level of every series: route first, then sign, then whether the group
    reached the floor, then the pass where its group was formed. The first condition that holds wins."""
    formed_in = estimate["final_step"].map(step_names).fillna("itself")
    merged_close = formed_in.str.startswith("sign") | formed_in.str.startswith("extra")
    itself = estimate["final_step"] == 0
    conditions = [
        estimate[ROUTE_COLUMN] == ROUTE_HISTORY_ONLY,
        (estimate[ROUTE_COLUMN] == ROUTE_FUTURE_ONLY) | (estimate["meses_historia"] == 0),
        estimate[SIGN_COLUMN] == SIGN_MIXED,
        (estimate["alcanzo_suelo"] == 0) & (estimate[SIGN_COLUMN] != SIGN_NEUTRAL),
        estimate["alcanzo_suelo"] == 0,
        itself & (estimate["group_support"] >= configuration.own_rate_floor)
        & (estimate["meses_historia"] >= configuration.own_level_min_history_months),
        itself & (estimate["group_support"] >= configuration.own_rate_floor),
        itself,
        merged_close]
    levels = [LEVEL_NO_IMPACT, LEVEL_NO_HISTORY, LEVEL_MIXED, LEVEL_SIGNED_UNDER_FLOOR, LEVEL_FAR,
              LEVEL_OWN, LEVEL_OWN_SHORT, LEVEL_OWN_REINFORCED, LEVEL_BORROWED]
    return np.select(conditions, levels, default=LEVEL_FAR)


def risk_levels_report(estimate: pd.DataFrame) -> pd.DataFrame:
    """Money to predict by risk level, with the level's prediction error weighted by money."""
    total_usd = estimate["usd_por_predecir"].sum()
    report_rows = []
    for level, level_series in estimate.groupby("nivel_riesgo"):
        with_error = level_series[level_series["se_prediccion_pp"].notna() & (level_series["usd_por_predecir"] > 0)]
        error_pp = (np.average(with_error["se_prediccion_pp"], weights=with_error["usd_por_predecir"])
                    if len(with_error) else np.nan)
        report_rows.append({"nivel_riesgo": level, "series": len(level_series),
                            "usd_por_predecir": level_series["usd_por_predecir"].sum(),
                            "pct_usd": level_series["usd_por_predecir"].sum() / total_usd if total_usd else 0.0,
                            "error_pp": error_pp})
    return pd.DataFrame(report_rows).sort_values("nivel_riesgo").reset_index(drop=True)


def check_ladder(estimate: pd.DataFrame, series_rate: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 to 4."""
    # [1] one estimate per series
    configuration.log_check(STEP_2_3_LABEL, check_log, "every series has exactly one estimate",
                            len(estimate) == len(series_rate) and estimate[SERIES_ID_COLUMN].is_unique,
                            failure_detail=f"{len(estimate):,} estimates for {len(series_rate):,} series",
                            context=f"{len(estimate):,} series")

    # [2] the group lends its rate: every series of a group has the same estimated rate
    spread = estimate.dropna(subset=["tasa_estimada"]).groupby(COMPOSITION_ID_COLUMN)["tasa_estimada"].agg(lambda rates: rates.max() - rates.min())
    configuration.log_check(STEP_2_3_LABEL, check_log, "every series of a group has the same estimated rate",
                            bool((spread <= configuration.rounding_tolerance).all()),
                            failure_detail=f"{int((spread > configuration.rounding_tolerance).sum()):,} groups with different rates")

    # [3] proportions
    out_of_range = estimate[(estimate["tasa_estimada"] < 0) | (estimate["tasa_estimada"] > 1)
                            | (estimate["z"] < 0) | (estimate["z"] > 1)]
    configuration.log_check(STEP_2_3_LABEL, check_log, "every estimated rate and every z is between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range):,} series out of [0, 1]",
                            examples=out_of_range[[SERIES_ID_COLUMN, "tasa_estimada", "z"]])

    # [4] the prediction never sheds the series' own noise
    shrinking = estimate[estimate["se_prediccion_pp"] < estimate["se_estimacion_pp"] - configuration.rounding_tolerance]
    configuration.log_check(STEP_2_3_LABEL, check_log, "the prediction error is never below the estimation error",
                            shrinking.empty,
                            failure_detail=f"{len(shrinking):,} series whose prediction error is below the estimate's")


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.4 · POOL SERIES  (until 25-oct-2026: step_10_pool_series.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The monthly series of every final group: what the backtest judges.
#
# After the ladder (steps 2.2-2.3) every estimable series belongs to ONE final group, and the
# group lends its rate to its series. The rate that will be forecast is the rate of that
# group, month by month. This step builds that monthly series: for every final group, the
# units due and renewed of every series in its rate (the ones that use it and the ones that
# only lend their history) are summed month by month, and its rate is Σ renewed / Σ due of the
# month. A lender is in more than one composition: summing every composition does NOT add up
# to the raw (the portfolio is taken from the series themselves, step 2.5).
#
# Only closed truth months enter (entrenamiento, examen): the gaps and the future have no
# rate. Every id also gets a reference row: its months, its support (the median units due
# of a month), its rate over all its months, how many series take their rate from it, the
# money they have to predict, and its GATE:
#   · nivel    support ≥ support_floor: the backtest will judge its techniques
#   · soporte  support below the floor: not judged, it will take the default technique
#
# Actions (logged as they are done):
#   1. the compositions of step 2.2 and every series in their rate (users and lenders)
#   2. the monthly series of every id (units due and renewed summed; rate)
#   3. the reference of every id: months, support, rate, series, money, gate
#   4. check the series and the reference                             checks 1-3
#   5. write the series and the reference                             checks 4-5
#   6. count the checks; stop if any failed
#   7. show the ids by gate and the largest ones, as tables
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every chosen estimation id has a monthly series
#    2. the support of every group equals its support in step 2.2
#    3. every monthly rate is between 0 and 1                         (warning only)
#    4-5. tables sff_pool_serie and sff_pool_referencia written and read back
#
# Output: (the monthly series of every id, long: id × month; the reference of every id)
# · tables sff_pool_serie, sff_pool_referencia.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_2_4_LABEL = "2.4"
STEP_2_4_NAME = "POOL SERIES"
STEP_2_4_PURPOSE = ("build the monthly series of the rate of every final group of the ladder, summing every "
                "series in it; this is the series the backtest judges and the forecast extends")
STEP_2_4_ACTIONS = ["the compositions of step 2.2 and every series in their rate (users and lenders)",
                "the monthly series of every id (units due and renewed summed; rate)",
                "the reference of every id: months, support, rate, series, money, gate",
                "check the series and the reference (checks 1-3)",
                "write the series and the reference (checks 4-5)",
                "count the checks; stop if any failed",
                "show the ids by gate and the largest ones, as tables"]
STEP_2_4_OUTPUT = "one row per id × closed month (units due, renewed, rate) · one row per id · tables sff_pool_serie, sff_pool_referencia"



def build_pool_series(rated_units: pd.DataFrame, ladder: dict, series_estimate: pd.DataFrame,
                      configuration: Config) -> tuple:
    """The monthly series and the reference of every final group; checked and written."""
    configuration.log_step_start(STEP_2_4_LABEL, STEP_2_4_NAME, STEP_2_4_PURPOSE, STEP_2_4_ACTIONS, STEP_2_4_OUTPUT)
    check_log = []
    period_column = configuration.period_col

    # [1] the compositions of step 2.2 and every series in their rate
    # every series in the rate of every composition: the ones that use it and the ones that only lend their history
    membership = ladder["composition_members"][[SERIES_ID_COLUMN, COMPOSITION_ID_COLUMN]]
    chosen = series_estimate[series_estimate[SERIES_ID_COLUMN].isin(set(membership[SERIES_ID_COLUMN]))]
    chosen_ids = set(membership[COMPOSITION_ID_COLUMN])
    configuration.log_action(STEP_2_4_LABEL, 1, f"{len(chosen_ids):,} compositions · {len(membership):,} series in their rates "
                                            f"({membership[SERIES_ID_COLUMN].nunique():,} distinct: a series may lend its history "
                                            f"to a composition it does not use)")

    # [2] the monthly series: every matching series summed, month by month
    history = rated_units[rated_units[RATE_COLUMN].notna()][[SERIES_ID_COLUMN, period_column, CALENDAR_ROLE_COLUMN,
                                                             configuration.renewed_units_col, configuration.pipeline_units_col]]
    joined = history.merge(membership, on=SERIES_ID_COLUMN)
    pool_series = (joined.groupby([COMPOSITION_ID_COLUMN, period_column])
                   .agg(rol=(CALENDAR_ROLE_COLUMN, "first"),
                        renovadas=(configuration.renewed_units_col, "sum"),
                        vencen=(configuration.pipeline_units_col, "sum"),
                        series_en_el_mes=(SERIES_ID_COLUMN, "nunique"))
                   .reset_index().sort_values([COMPOSITION_ID_COLUMN, period_column]).reset_index(drop=True))
    pool_series[RATE_COLUMN] = pool_series["renovadas"] / pool_series["vencen"].where(pool_series["vencen"] > 0)
    configuration.log_action(STEP_2_4_LABEL, 2, f"{len(pool_series):,} group × month rows; a group has "
                                            f"{pool_series.groupby(COMPOSITION_ID_COLUMN).size().median():.0f} months (median)")

    # [3] the reference of every id
    pool_reference = reference_of_pools(pool_series, chosen, membership, configuration)
    gate_counts = pool_reference["gate"].value_counts().to_dict()
    configuration.log_action(STEP_2_4_LABEL, 3, f"gates: {gate_counts} · the judged ids (nivel) carry "
                                            f"{pool_reference.loc[pool_reference['gate'] == GATE_LEVEL, 'usd_por_predecir'].sum() / max(pool_reference['usd_por_predecir'].sum(), 1):.0%} "
                                            f"of the money their series have to predict")

    # [4] the checks
    configuration.log_action(STEP_2_4_LABEL, 4, "checking the series and the reference")
    missing_ids = chosen_ids - set(pool_series[COMPOSITION_ID_COLUMN])
    configuration.log_check(STEP_2_4_LABEL, check_log, "every final group has a monthly series", not missing_ids,
                            failure_detail=f"{len(missing_ids):,} ids without months: {sorted(missing_ids)[:5]}",
                            context=f"{len(chosen_ids):,} ids")
    support_in_step_08 = (ladder["groups"].drop_duplicates(COMPOSITION_ID_COLUMN)
                          .set_index(COMPOSITION_ID_COLUMN)["group_support"].rename("n_pool_paso_10"))
    compared = pool_reference.join(support_in_step_08, on=COMPOSITION_ID_COLUMN)
    mismatched = compared[(compared["n_pool"] - compared["n_pool_paso_10"]).abs() > configuration.support_tolerance]
    configuration.log_check(STEP_2_4_LABEL, check_log, "the support of every group equals its support in step 2.2",
                            mismatched.empty,
                            failure_detail=f"{len(mismatched):,} groups whose support differs from step 2.2",
                            examples=mismatched[[COMPOSITION_ID_COLUMN, "n_pool", "n_pool_paso_10"]])
    out_of_range = pool_series[(pool_series[RATE_COLUMN] < 0) | (pool_series[RATE_COLUMN] > 1)]
    configuration.log_check(STEP_2_4_LABEL, check_log, "every monthly rate is between 0 and 1", out_of_range.empty,
                            failure_detail=f"{len(out_of_range):,} months with a rate outside [0, 1]", blocking=False,
                            examples=out_of_range)

    # [5] the tables, written
    configuration.log_action(STEP_2_4_LABEL, 5, "writing the series and the reference")
    configuration.write_table(STEP_2_4_LABEL, check_log, pool_series, TABLE_POOL_SERIES)
    configuration.write_table(STEP_2_4_LABEL, check_log, pool_reference, TABLE_POOL_REFERENCE)

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_4_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_2_4_LABEL, STEP_2_4_NAME, check_log)

    # [7] the ids by gate, and the largest ones
    configuration.log_action(STEP_2_4_LABEL, 7, f"final groups by gate (nivel: support ≥ {configuration.support_floor:.0f}, "
                                            f"judged by the backtest; soporte: below it, takes the default technique):")
    configuration.show_table(pool_reference.groupby("gate")
                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), series_que_lo_usan=("series_que_lo_usan", "sum"),
                                  usd_por_predecir=("usd_por_predecir", "sum"), meses_mediana=("meses", "median"))
                             .reset_index())
    configuration.logger.doc(f"[{STEP_2_4_LABEL}] the {configuration.largest_series_shown} groups with the most money to predict:")
    configuration.show_table(pool_reference.nlargest(configuration.largest_series_shown, "usd_por_predecir"))
    return pool_series, pool_reference


def reference_of_pools(pool_series: pd.DataFrame, chosen: pd.DataFrame, membership: pd.DataFrame,
                       configuration: Config) -> pd.DataFrame:
    """One row per id: months, support, rate, first and last month, series, money, gate."""
    period_column = configuration.period_col
    with_pipeline = pool_series[pool_series["vencen"] > 0]
    grouped = with_pipeline.groupby(COMPOSITION_ID_COLUMN)
    reference = pd.DataFrame({
        "meses": grouped.size(),
        "primer_mes": grouped[period_column].min(),
        "ultimo_mes": grouped[period_column].max(),
        "n_pool": grouped["vencen"].median(),
        "tasa_pool": grouped["renovadas"].sum() / grouped["vencen"].sum(),
        "series_en_el_pool": membership.groupby(COMPOSITION_ID_COLUMN)[SERIES_ID_COLUMN].nunique()})
    reference["series_que_lo_usan"] = chosen.groupby(COMPOSITION_ID_COLUMN).size()
    reference["usd_por_predecir"] = chosen.groupby(COMPOSITION_ID_COLUMN)["usd_por_predecir"].sum()
    reference = reference.reset_index().rename(columns={"index": COMPOSITION_ID_COLUMN})
    reference[["series_que_lo_usan", "usd_por_predecir"]] = reference[["series_que_lo_usan", "usd_por_predecir"]].fillna(0)
    reference["gate"] = np.where(reference["n_pool"] >= configuration.support_floor, GATE_LEVEL, GATE_SUPPORT)
    return reference


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.5 · DYNAMICS OF THE RATE  (until 25-oct-2026: step_11_dynamics.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The dynamics of the renewal rate: is there anything beyond noise?
#
# The rate of a month is k renewals out of n contracts: even if nothing changes, it moves by
# chance (the binomial error). This step asks, for the portfolio and for every estimation id
# with support, what moves the rate BEYOND that noise. It is DESCRIPTIVE: it labels, it does
# not restrict any technique (the backtest of step 2.6 lets every technique compete as far
# as its history allows, and decides).
#
# For every estimation id with support and at least dynamics_min_months (24) of history:
#   · φ (phi) = observed variance of the monthly rate / the variance chance alone would give
#     with a constant rate (mean of p(1 − p)/n). φ ≈ 1: the rate moves what the coin
#     predicts, nothing to model, the average is the best technique. φ > 1: something moves
#     it (a trend, a season, a change of level, a changing mix): there is room for a
#     time-series technique.
#   · TREND: the slope of the rate over time (weighted by the units of each month), in pp per
#     year, and its p-value. tendencia = +1 / −1 when significant, 0 otherwise.
#   · SEASONALITY: on the rate WITHOUT its trend, is the month of the year significant? An
#     F test: variance between the 12 calendar months vs within them. estacional = 1 when
#     significant. amplitud_pp: the highest month minus the lowest. meses_alto / meses_bajo:
#     the calendar months above / below the others beyond twice their error.
#   · CONSISTENCY: the correlation of the month profile of the first half of the years with
#     the second half: a real season repeats.
# The same for the PORTFOLIO as a whole (every estimable series, summed): its month profile.
# The significance level is dynamics_significance (5 %): no threshold in pp or in % of
# improvement; every test is relative to the noise of the series itself.
#
# Actions (logged as they are done):
#   1. the pools measured: support and history
#   2. φ, trend and seasonality of every pool
#   3. the month profile of the whole portfolio
#   4. check the attributes                                            checks 1-2
#   5. count the checks; stop if any failed
#   6. show the general verdict: the portfolio's profile and how many pools (and how much money)
#      show a trend, a season, φ > 1
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every pool measured has its attributes (φ ≥ 0, p-values between 0 and 1)
#    2. the portfolio profile covers the 12 calendar months
#
# Output: (the dynamics of every pool, the month profile of the portfolio), in memory: they reach the
# forecast series dimension (s2_5_*) and the report.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_2_5_LABEL = "2.5"
STEP_2_5_NAME = "DYNAMICS OF THE RATE"
STEP_2_5_PURPOSE = ("measure what moves the renewal rate beyond the binomial noise, for the portfolio and for every pool "
                "with support: φ (observed variation / noise), trend, seasonality and its months; descriptive: it "
                "does not restrict any technique, the backtest decides")
STEP_2_5_ACTIONS = ["the pools measured: support and history",
                "φ, trend and seasonality of every pool",
                "the month profile of the whole portfolio",
                "check the attributes (checks 1-2)",
                "count the checks; stop if any failed",
                "show the general verdict: the portfolio's profile and how many pools show a trend, a season, φ > 1"]
STEP_2_5_OUTPUT = "one row per pool (φ, trend, season, months high/low) · the portfolio's month profile (in memory)"

# ─── named constants ─────────────────────────────────────────────────────────────
MONTHS_PER_YEAR = 12


def measure_dynamics(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config,
                     rated_units: pd.DataFrame = None) -> tuple:
    """φ, trend and seasonality of every pool with support, and the portfolio's month profile. The portfolio is
    every real closed unit of rated_units (step 1.7) once: summing the compositions would count twice the series that
    lend their history to a composition they do not use."""
    configuration.log_step_start(STEP_2_5_LABEL, STEP_2_5_NAME, STEP_2_5_PURPOSE, STEP_2_5_ACTIONS, STEP_2_5_OUTPUT)
    check_log = []
    period_column = configuration.period_col
    truth_months = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (pool_series["vencen"] > 0)]

    # [1] the pools measured
    months_per_pool = truth_months.groupby(COMPOSITION_ID_COLUMN).size()
    measured_ids = [estimation_id for estimation_id in pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN]
                    if months_per_pool.get(estimation_id, 0) >= configuration.dynamics_min_months]
    configuration.log_action(STEP_2_5_LABEL, 1, f"{len(measured_ids):,} pools measured (support ≥ {configuration.support_floor:.0f} "
                                            f"and ≥ {configuration.dynamics_min_months} months) of {len(pool_reference):,}")

    # [2] the attributes of every pool
    dynamics_rows = []
    for estimation_id in measured_ids:
        monthly = truth_months[truth_months[COMPOSITION_ID_COLUMN] == estimation_id].sort_values(period_column)
        dynamics_rows.append({COMPOSITION_ID_COLUMN: estimation_id,
                              **dynamics_of_one_series(monthly, period_column, configuration)})
    pool_dynamics = pd.DataFrame(dynamics_rows)
    if len(pool_dynamics):
        pool_dynamics = pool_dynamics.merge(pool_reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
    configuration.log_action(STEP_2_5_LABEL, 2, f"φ median {pool_dynamics['phi'].median():.2f} · "
                                            f"{int(pool_dynamics['estacional'].sum())} seasonal · "
                                            f"{int((pool_dynamics['tendencia'] != 0).sum())} with a trend"
                             if len(pool_dynamics) else "no pool to measure")

    # [3] the portfolio: every series once, its month profile
    if rated_units is not None:
        real_closed = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                                  & (rated_units[configuration.pipeline_units_col] > 0)]
        portfolio_monthly = (real_closed.groupby(period_column)
                             .agg(renovadas=(configuration.renewed_units_col, "sum"), vencen=(configuration.pipeline_units_col, "sum"))
                             .reset_index())
    else:
        portfolio_monthly = truth_months.groupby(period_column)[["renovadas", "vencen"]].sum().reset_index()
    portfolio = dynamics_of_one_series(portfolio_monthly, period_column, configuration)
    portfolio_profile = pd.DataFrame(portfolio.pop("_profile"))
    configuration.log_action(STEP_2_5_LABEL, 3, f"portfolio: φ {portfolio['phi']:.2f} · month effect p = {portfolio['p_valor_mes']:.3f} · "
                                            f"amplitude {portfolio['amplitud_pp']:.1f} pp · trend {portfolio['tendencia_pp_ano']:+.1f} pp/year "
                                            f"(p = {portfolio['p_valor_tendencia']:.3f})")
    if len(pool_dynamics):
        pool_dynamics = pool_dynamics.drop(columns="_profile")

    # [4] the checks
    configuration.log_action(STEP_2_5_LABEL, 4, "checking the attributes")
    invalid = pool_dynamics[(pool_dynamics["phi"] < 0) | ~pool_dynamics["p_valor_mes"].between(0, 1)
                            | ~pool_dynamics["p_valor_tendencia"].between(0, 1)] if len(pool_dynamics) else pool_dynamics
    configuration.log_check(STEP_2_5_LABEL, check_log, "every pool measured has its attributes (φ ≥ 0, p-values in [0, 1])",
                            len(pool_dynamics) == len(measured_ids) and invalid.empty,
                            failure_detail=f"{len(invalid)} pools with invalid attributes", context=f"{len(pool_dynamics)} pools")
    configuration.log_check(STEP_2_5_LABEL, check_log, "the portfolio profile covers the 12 calendar months",
                            len(portfolio_profile) == MONTHS_PER_YEAR,
                            failure_detail=f"only {len(portfolio_profile)} calendar months")

    # [5] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_5_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_2_5_LABEL, STEP_2_5_NAME, check_log)

    # [7] the general verdict
    log_dynamics_report(pool_dynamics, portfolio, portfolio_profile, configuration)
    return pool_dynamics, portfolio_profile, portfolio


def dynamics_of_one_series(monthly: pd.DataFrame, period_column: str, configuration: Config) -> dict:
    """φ, trend (pp/year and p-value), seasonality on the detrended rate (F test), amplitude,
    months high and low, consistency between halves, and the 12-month profile."""
    renewed = monthly["renovadas"].to_numpy(dtype=float)
    due = monthly["vencen"].to_numpy(dtype=float)
    rates = renewed / due
    pooled_rate = renewed.sum() / due.sum()
    calendar_months = np.array([month.month for month in monthly[period_column]])
    years = np.array([month.year for month in monthly[period_column]])
    time_in_years = np.array([(month.ordinal - monthly[period_column].iloc[0].ordinal) / MONTHS_PER_YEAR
                              for month in monthly[period_column]])

    # φ: observed variance of the monthly rate / the binomial variance with a constant rate
    binomial_variance = np.mean(pooled_rate * (1 - pooled_rate) / due)
    phi = float(np.var(rates, ddof=1) / binomial_variance) if binomial_variance > 0 else np.nan

    # the trend: weighted least squares of the rate on time (weights = units due)
    weights = due / due.mean()
    design = np.column_stack([np.ones_like(time_in_years), time_in_years])
    weighted_design = design * np.sqrt(weights)[:, None]
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, rates * np.sqrt(weights), rcond=None)
    residuals = rates - design @ coefficients
    degrees_of_freedom = max(len(rates) - 2, 1)
    residual_variance = float(np.sum(weights * residuals ** 2) / degrees_of_freedom)
    slope_se = float(np.sqrt(residual_variance * np.linalg.inv(weighted_design.T @ weighted_design)[1, 1]))
    slope_t = coefficients[1] / slope_se if slope_se > 0 else 0.0
    trend_p_value = float(2 * stats.t.sf(abs(slope_t), degrees_of_freedom))
    trend_sign = int(np.sign(coefficients[1])) if trend_p_value < configuration.dynamics_significance else 0

    # the season: on the detrended rate, the 12 calendar months (weighted), F test
    detrended = pd.DataFrame({"mes": calendar_months, "residuo": residuals, "peso": weights})
    month_groups = detrended.groupby("mes")
    month_weight = month_groups["peso"].sum()
    month_effect = (detrended.assign(weighted=detrended["residuo"] * detrended["peso"]).groupby("mes")["weighted"].sum()
                    / month_weight)
    overall = float(np.sum(weights * residuals) / np.sum(weights))
    between = float(np.sum(month_weight * (month_effect - overall) ** 2))
    within = float(np.sum(weights * (residuals - detrended["mes"].map(month_effect).to_numpy()) ** 2))
    months_present = len(month_effect)
    between_df, within_df = max(months_present - 1, 1), max(len(rates) - months_present - 1, 1)
    f_statistic = (between / between_df) / (within / within_df) if within > 0 else 0.0
    month_p_value = float(stats.f.sf(f_statistic, between_df, within_df))
    seasonal = int(month_p_value < configuration.dynamics_significance)

    # the months high and low: beyond twice their own error
    month_count = month_groups.size()
    month_error = np.sqrt(within / within_df / month_count)
    deviation_pp = PERCENTAGE_POINTS * (month_effect - overall)
    high_months = [int(month) for month in month_effect.index
                   if month_effect[month] - overall > configuration.month_significance_sigmas * month_error[month]]
    low_months = [int(month) for month in month_effect.index
                  if overall - month_effect[month] > configuration.month_significance_sigmas * month_error[month]]

    # consistency: the month profile of the first half of the years vs the second half
    middle_year = np.median(np.unique(years))
    first_half = detrended[years <= middle_year].groupby("mes")["residuo"].mean()
    second_half = detrended[years > middle_year].groupby("mes")["residuo"].mean()
    shared = first_half.index.intersection(second_half.index)
    # a profile that does not move in one of the halves has no correlation (a series with the same rate every month)
    both_move = len(shared) >= configuration.dynamics_min_shared_months and first_half[shared].std() > 0 and second_half[shared].std() > 0
    consistency = float(np.corrcoef(first_half[shared], second_half[shared])[0, 1]) if both_move else np.nan

    profile = pd.DataFrame({"mes": month_effect.index.astype(int), "efecto_pp": deviation_pp.to_numpy(),
                            "error_pp": PERCENTAGE_POINTS * month_error.to_numpy(), "meses_observados": month_count.to_numpy(),
                            "tasa_media": (pooled_rate + month_effect.to_numpy() - overall)})
    return {"meses": int(len(rates)), "tasa_pool": float(pooled_rate), "phi": phi,
            "tendencia": trend_sign, "tendencia_pp_ano": float(PERCENTAGE_POINTS * coefficients[1]),
            "p_valor_tendencia": trend_p_value, "estacional": seasonal, "p_valor_mes": month_p_value,
            "amplitud_pp": float(deviation_pp.max() - deviation_pp.min()),
            "meses_alto": ",".join(str(month) for month in high_months), "meses_bajo": ",".join(str(month) for month in low_months),
            "consistencia": consistency, "_profile": profile}


def log_dynamics_report(pool_dynamics: pd.DataFrame, portfolio: dict, portfolio_profile: pd.DataFrame,
                        configuration: Config) -> None:
    """Action 6: the general verdict, as tables."""
    verdict = ("the portfolio rate HAS a month effect beyond noise" if portfolio["estacional"]
               else "the portfolio rate shows NO month effect beyond noise")
    configuration.log_action(STEP_2_5_LABEL, 6, f"general verdict: {verdict} (p = {portfolio['p_valor_mes']:.3f}, amplitude "
                                            f"{portfolio['amplitud_pp']:.1f} pp, consistency between halves "
                                            f"{portfolio['consistencia']:.2f}). Month profile of the portfolio (efecto_pp: "
                                            f"the month minus the year, without trend):")
    configuration.show_table(portfolio_profile)
    if len(pool_dynamics):
        total_usd = pool_dynamics["usd_por_predecir"].sum()
        def share(mask):
            return {"pools": int(mask.sum()), "usd_por_predecir": pool_dynamics.loc[mask, "usd_por_predecir"].sum(),
                    "pct_usd": pool_dynamics.loc[mask, "usd_por_predecir"].sum() / total_usd if total_usd else 0.0}
        summary = pd.DataFrame([
            {"atributo": f"φ > {configuration.phi_with_engine} (more than noise)", **share(pool_dynamics["phi"] > configuration.phi_with_engine)},
            {"atributo": "trend (+1 or −1)", **share(pool_dynamics["tendencia"] != 0)},
            {"atributo": "seasonal (month effect)", **share(pool_dynamics["estacional"] == 1)},
            {"atributo": "all pools measured", **share(pool_dynamics["phi"].notna())}])
        configuration.logger.doc(f"[{STEP_2_5_LABEL}] the pools with support, by attribute (the money their series have to predict):")
        configuration.show_table(summary)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.6 · BACKTEST OF THE RATE  (until 25-oct-2026: step_12_backtest.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# The backtest of the renewal rate: which technique predicts each pool
# best, how precise it is, and how wide its error band must be.
#
# HOW THE TEST WORKS (it is also written in the log, with the months of the run):
#   · WHAT IS JUDGED: the monthly series of every estimation id with support (gate 'nivel',
#     step 2.4). An id below the floor is not judged: it takes the default technique.
#   · TWO KINDS OF TARGET MONTHS, never mixed:
#       selection  the backtest_selection_months (6) closed months BEFORE the exam: the
#                  techniques are compared on them and the best one is CHOSEN
#       exam       the exam months of the calendar (test_months, 3): used only to MEASURE
#                  the chosen technique, which never saw them while being chosen
#   · NO PEEKING: for a target month T and a horizon h, every technique sees only the
#     months up to T − h, exactly as it would have been in real life h months earlier
#     (h = 1: next month; h = 6: six months ahead). It needs at least
#     backtest_min_history_months of history.
#   · THE ERROR: err_pp = 100 · (predicted − real rate). It is compared with the error chance
#     alone would cause in that month, the binomial error of the real rate with the month's
#     own units: err_norm = err_pp / se_binom_pp. |err_norm| ≈ 1 is as good as any forecast
#     can be; 3 means three times the noise.
#   · THE CHOICE, per id and horizon band (corto = h 1, medio_largo = h 2-6): the technique
#     with the lowest mean |err_norm| in the selection months REPLACES the default technique
#     (T3_ma3, the moving average of 3 months) only if it beats it by the band's margin
#     (0.10 near, 0 far). A technique competes where its minimum history is reached (the
#     seasonal ones need 13 and 24 months): only the history limits, no seasonality verdict.
#     The ranking of the techniques is computed on the targets where ALL of them competed.
#   · THE BAND, per technique and horizon: the 5 % and 95 % quantiles of err_norm in the
#     selection months, over every judged id. In the exam, 90 % of the errors of the chosen
#     technique should fall inside it.
#   · THE PRECISION, in the exam months: per id (chosen vs default technique) and for the TOTAL:
#     Σ predicted renewals vs Σ real renewals of all judged ids, month by month.
#
# OUTPUT TABLES (long format: one row per prediction, not one table or column per technique;
# each id competes with the techniques its history allows, and Power BI pivots it):
#   sff_backtest_predicciones  id × target month × horizon × technique: predicted, real, errors
#   sff_decision_tecnica       id × horizon band: the chosen technique, its error, the default technique's
#   sff_decision_bandas        technique × horizon: the error quantiles
#   sff_backtest_examen        id × horizon band: the chosen vs the default technique in the exam
#   sff_backtest_examen_total  exam month × horizon: the error of the total renewals
#
# Actions (logged as they are done):
#   1. the calendar of the test: selection and exam months, horizons, and how it works
#   2. the predictions of every technique for every judged id, target and horizon
#   3. the choice of every id and horizon band
#   4. the error bands of every technique and horizon
#   5. the precision in the exam: per id and for the total
#   6. check the test                                                 checks 1-5
#   7. write the five tables                                          checks 6-10
#   8. count the checks; stop if any failed
#   9. show the ranking of the techniques, the choices and the precision in the exam
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. no prediction saw its target month or anything after its origin
#    2. every estimation id has a decision in every horizon band
#    3. the default technique was predicted wherever another technique was
#    4. the ranking compares the techniques on common targets (every technique competed)
#    5. in the exam, the band holds at least 80 % of the chosen technique's errors (nominal 90 %)
#                                                                      (warning only)
#    6-10. the five tables written and read back
#
# Output: (the predictions, the decision per id and band, the error bands, the exam per id,
# the exam of the total) · the six tables above.

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_2_6_LABEL = "2.6"
STEP_2_6_NAME = "BACKTEST OF THE RATE"
STEP_2_6_PURPOSE = ("choose, for every pool with support and every horizon, the technique that predicts its renewal rate "
                "best in the months before the exam, then measure it in the exam months it never saw: its precision "
                "per pool and for the total, and the width its error band must have")
STEP_2_6_ACTIONS = ["the calendar of the test: selection and exam months, horizons, and how it works",
                "the predictions of every technique for every judged id, target and horizon",
                "the choice of every id and horizon band",
                "the error bands of every technique and horizon",
                "the precision in the exam: per id and for the total",
                "check the test (checks 1-5)",
                "write the five tables (checks 6-10)",
                "count the checks; stop if any failed",
                "show the ranking of the techniques, the choices and the precision in the exam"]
STEP_2_6_OUTPUT = ("predictions (id × target × horizon × technique) · the chosen technique per id and band · error bands · "
               "the exam per id and for the total · five tables")

# ─── named constants ─────────────────────────────────────────────────────────────
SUPPORT_ORIGIN = "sin_soporte"      # an id below the floor: not judged, it takes the default technique
PREDICTION_COLUMNS = [COMPOSITION_ID_COLUMN, "proposito", "mes_objetivo", "h", "origen", "ultimo_mes_visto", "tecnica",
                      "tasa_pred", "tasa_real", "vencen_real", "err_pp", "se_binom_pp", "err_norm"]


def band_of_horizon(horizon: int, horizon_bands: dict) -> str:
    """The horizon band of a horizon; beyond the last band, the last one."""
    for band_name, (low, high) in horizon_bands.items():
        if low <= horizon <= high:
            return band_name
    return list(horizon_bands)[-1]


def run_backtest(pool_series: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> dict:
    """The backtest of the rate: predictions, choice, bands and exam; checked and written."""
    configuration.log_step_start(STEP_2_6_LABEL, STEP_2_6_NAME, STEP_2_6_PURPOSE, STEP_2_6_ACTIONS, STEP_2_6_OUTPUT)
    check_log = []

    # [1] the calendar of the test, and how it works
    selection_months, exam_months = test_calendar(pool_series, configuration)
    log_how_the_test_works(selection_months, exam_months, pool_reference, configuration)

    # [2] the predictions
    judged_ids = set(pool_reference.loc[pool_reference["gate"] == GATE_LEVEL, COMPOSITION_ID_COLUMN])
    predictions = predict_every_target(pool_series, judged_ids, selection_months, exam_months, configuration)
    predictions["tramo_h"] = predictions["h"].map(lambda h: band_of_horizon(int(h), configuration.horizon_bands))
    configuration.log_action(STEP_2_6_LABEL, 2, f"{len(predictions):,} predictions: {predictions[COMPOSITION_ID_COLUMN].nunique():,} "
                                            f"ids × {predictions['mes_objetivo'].nunique()} target months × "
                                            f"{len(configuration.backtest_horizons)} horizons × up to {len(CATALOGUE)} techniques "
                                            f"(each where its history allows)")

    # [3] the choice
    decision = choose_techniques(predictions, pool_reference, configuration)
    judged_decision = decision[decision["tecnica_origen"] != SUPPORT_ORIGIN]
    configuration.log_action(STEP_2_6_LABEL, 3, f"choices: {judged_decision.groupby('tramo_h')['tecnica_origen'].value_counts().to_dict()}")

    # [4] the error bands of every technique and horizon (selection months)
    bands = error_bands(predictions, configuration)
    configuration.log_action(STEP_2_6_LABEL, 4, f"{len(bands)} bands (technique × horizon) from the selection months")

    # [5] the exam: per id and for the total
    exam_by_pool, exam_total, exam_rows = exam_precision(predictions, decision, bands, configuration)
    configuration.log_action(STEP_2_6_LABEL, 5, f"exam measured on {exam_by_pool[COMPOSITION_ID_COLUMN].nunique():,} ids and "
                                            f"{exam_total['mes_objetivo'].nunique()} months")

    # [6] the checks
    configuration.log_action(STEP_2_6_LABEL, 6, "checking the test")
    check_backtest(predictions, decision, pool_reference, exam_rows, configuration, check_log)

    # [7] the tables
    configuration.log_action(STEP_2_6_LABEL, 7, "writing the five tables")
    configuration.write_table(STEP_2_6_LABEL, check_log, predictions, TABLE_BACKTEST_PREDICTIONS)
    configuration.write_table(STEP_2_6_LABEL, check_log, decision, TABLE_TECHNIQUE_DECISION)
    configuration.write_table(STEP_2_6_LABEL, check_log, bands, TABLE_ERROR_BANDS)
    configuration.write_table(STEP_2_6_LABEL, check_log, exam_by_pool, TABLE_EXAM_BY_POOL)
    configuration.write_table(STEP_2_6_LABEL, check_log, exam_total, TABLE_EXAM_TOTAL)

    # [8] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_6_LABEL, 8, "counting the checks")
    configuration.log_check_summary(STEP_2_6_LABEL, STEP_2_6_NAME, check_log)

    # [9] what came out
    log_backtest_report(predictions, decision, exam_by_pool, exam_total, pool_reference, configuration)
    return dict(predictions=predictions, decision=decision, bands=bands, exam_by_pool=exam_by_pool, exam_total=exam_total)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CALENDAR AND ITS EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════════

def test_calendar(pool_series: pd.DataFrame, configuration: Config) -> tuple:
    """The selection months (the N closed months before the exam) and the exam months."""
    boundaries = configuration.calendar_boundaries()
    exam_months = pd.period_range(boundaries["test_start"], boundaries["current"] - 1, freq="M")
    selection_months = pd.period_range(boundaries["test_start"] - configuration.backtest_selection_months,
                                       boundaries["test_start"] - 1, freq="M")
    return list(selection_months), list(exam_months)


def log_how_the_test_works(selection_months: list, exam_months: list, pool_reference: pd.DataFrame,
                           configuration: Config) -> None:
    """Action 1: the calendar of this run and the rules of the test, in the log."""
    judged = pool_reference["gate"] == GATE_LEVEL
    largest_horizon = max(configuration.backtest_horizons)
    example_target = selection_months[-1]
    configuration.log_action(STEP_2_6_LABEL, 1, "how the test works:")
    explanation = [
        f"WHAT: the monthly rate of the {int(judged.sum()):,} estimation ids with support (of {len(pool_reference):,}); "
        f"the other {int((~judged).sum()):,} are not judged and take the default technique {configuration.default_technique}",
        f"SELECTION months {selection_months[0]}..{selection_months[-1]} ({len(selection_months)}): every technique predicts "
        f"them and the best one is CHOSEN",
        f"EXAM months {exam_months[0]}..{exam_months[-1]} ({len(exam_months)}): only to MEASURE the chosen technique, "
        f"which never saw them while being chosen",
        f"NO PEEKING: for target month T and horizon h a technique sees only the months up to T − h. Example: "
        f"T = {example_target}, h = {largest_horizon} → it sees up to {example_target - largest_horizon}; "
        f"h = 1 → up to {example_target - 1}. It needs ≥ {configuration.backtest_min_history_months} months of history",
        "ERROR: err_pp = 100·(predicted − real rate); err_norm = err_pp / binomial error of the real rate with the "
        "month's own units. |err_norm| ≈ 1 = as close as chance allows; 3 = three times the noise",
        f"CHOICE per id and band {list(configuration.horizon_bands)}: the lowest mean |err_norm| in the selection months "
        f"replaces {configuration.default_technique} only if it beats it by the margin "
        f"{configuration.default_margin_by_band}",
        "WHO COMPETES: every technique whose minimum history is reached (seasonal ones: 13 and 24 months); "
        "only the history limits, no seasonality verdict; the ranking uses the targets where all of them competed",
        f"BAND per technique and horizon: quantiles {configuration.band_low_quantile:.0%}-{configuration.band_high_quantile:.0%} "
        f"of err_norm in the selection months; in the exam ≈ {configuration.band_high_quantile - configuration.band_low_quantile:.0%} "
        f"of the errors should fall inside",
        "PRECISION in the exam: per id (chosen vs default technique) and of the TOTAL renewals of the judged ids, month by month"]
    for line in explanation:
        configuration.logger.doc(f"[{STEP_2_6_LABEL}]      {line}")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════════

def predict_every_target(pool_series: pd.DataFrame, judged_ids: set, selection_months: list, exam_months: list,
                         configuration: Config) -> pd.DataFrame:
    """One row per (id, target month, horizon, technique). The history of a prediction is
    every month of the id up to origin = target − h."""
    period_column = configuration.period_col
    purpose_of = {month.ordinal: PURPOSE_SELECTION for month in selection_months}
    purpose_of.update({month.ordinal: PURPOSE_EXAM for month in exam_months})
    judged_series = pool_series[pool_series[COMPOSITION_ID_COLUMN].isin(judged_ids) & pool_series[RATE_COLUMN].notna()
                                & (pool_series["vencen"] > 0)]
    rows = []
    for estimation_id, monthly in judged_series.groupby(COMPOSITION_ID_COLUMN, sort=False):
        monthly = monthly.sort_values(period_column)
        month_ordinals = np.array([month.ordinal for month in monthly[period_column]])
        calendar_months = np.array([month.month for month in monthly[period_column]])
        months = list(monthly[period_column])
        rates = monthly[RATE_COLUMN].to_numpy(dtype=float)
        units_due = monthly["vencen"].to_numpy(dtype=float)
        logit_rates = logit(rates)
        for target_position, target_ordinal in enumerate(month_ordinals):
            purpose = purpose_of.get(int(target_ordinal))
            if purpose is None:
                continue
            real_rate, real_units = rates[target_position], units_due[target_position]
            clipped = np.clip(real_rate, 0.5 / real_units, 1 - 0.5 / real_units)
            se_binomial_pp = PERCENTAGE_POINTS * np.sqrt(clipped * (1 - clipped) / real_units)
            for horizon in configuration.backtest_horizons:
                origin_ordinal = target_ordinal - horizon
                seen = month_ordinals <= origin_ordinal
                if seen.sum() < configuration.backtest_min_history_months:
                    continue
                history = logit_rates[seen]
                history_months = calendar_months[seen]
                last_seen = months[int(np.flatnonzero(seen)[-1])]
                for technique_id in eligible_techniques(len(history)):
                    predicted_logit = predict_logit(technique_id, history, history_months, horizon)
                    if not np.isfinite(predicted_logit):
                        continue
                    predicted_rate = float(inverse_logit(predicted_logit))
                    error_pp = PERCENTAGE_POINTS * (predicted_rate - real_rate)
                    rows.append((estimation_id, purpose, months[target_position], horizon,
                                 months[target_position] - horizon, last_seen, technique_id,
                                 predicted_rate, real_rate, real_units, error_pp, se_binomial_pp,
                                 error_pp / se_binomial_pp))
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHOICE, THE BANDS, THE EXAM
# ═══════════════════════════════════════════════════════════════════════════════════

def choose_techniques(predictions: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per id and band, on the selection months: a technique replaces the default technique only if, on the targets
    (month, horizon) BOTH predicted, its mean |err_norm| is lower than the default's by the band's margin and it has at
    least backtest_min_predictions of them; among the ones that do, the biggest improvement wins (ties: the catalogue
    order). Comparing on the shared targets matters: a technique that needs more history predicts fewer, later
    targets, and its mean over them is not comparable with the default's over all of them.
    Ids below the floor get the default technique (sin_soporte)."""
    default_technique = configuration.default_technique
    catalogue_order = {technique_id: position for position, technique_id in enumerate(CATALOGUE)}
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION].assign(
        abs_norm=lambda frame: frame["err_norm"].abs(), abs_pp=lambda frame: frame["err_pp"].abs())
    scores = (selection.groupby([COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
              .agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"), n_predicciones=("abs_norm", "size"))
              .reset_index())
    # every target of every id and band, one column per technique: the shared targets are where both are filled
    by_target = selection.pivot_table(index=[COMPOSITION_ID_COLUMN, "tramo_h", "mes_objetivo", "h"], columns="tecnica",
                                      values="abs_norm", aggfunc="first")
    decision_rows = []
    for (estimation_id, band_name), block in scores.groupby([COMPOSITION_ID_COLUMN, "tramo_h"]):
        block = block.set_index("tecnica")
        default_score = block.loc[default_technique, "err_norm_medio"] if default_technique in block.index else np.inf
        margin = float(configuration.default_margin_by_band.get(band_name, 0.0))
        chosen, origin = default_technique, DEFAULT_ORIGIN
        if default_technique in block.index:
            targets = by_target.loc[(estimation_id, band_name)]
            improvements = {}
            for technique_id in block.index:
                if technique_id == default_technique:
                    continue
                shared = targets[[technique_id, default_technique]].dropna()
                if len(shared) < configuration.backtest_min_predictions:
                    continue
                improvement = shared[default_technique].mean() - shared[technique_id].mean()
                if improvement > margin:
                    improvements[technique_id] = improvement
            if improvements:
                chosen = min(improvements, key=lambda technique_id: (-improvements[technique_id], catalogue_order[technique_id]))
                origin = CHAMPION_ORIGIN
        chosen_row = block.loc[chosen] if chosen in block.index else pd.Series(dict(err_norm_medio=np.nan, err_pp_medio=np.nan,
                                                                                    n_predicciones=0))
        decision_rows.append({COMPOSITION_ID_COLUMN: estimation_id, "tramo_h": band_name, "tecnica": chosen,
                              "tecnica_origen": origin, "err_norm_seleccion": chosen_row["err_norm_medio"],
                              "err_pp_seleccion": chosen_row["err_pp_medio"], "n_predicciones": int(chosen_row["n_predicciones"]),
                              "por_defecto_err_norm_seleccion": default_score})
    decision = pd.DataFrame(decision_rows)

    # every id and band gets a decision: the unjudged ones (and a judged id with no selection
    # month in a band) take the default technique
    all_pairs = pd.MultiIndex.from_product([pool_reference[COMPOSITION_ID_COLUMN], list(configuration.horizon_bands)],
                                           names=[COMPOSITION_ID_COLUMN, "tramo_h"]).to_frame(index=False)
    decision = all_pairs.merge(decision, on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
    unjudged = decision["tecnica"].isna()
    decision.loc[unjudged, "tecnica"] = default_technique
    decision.loc[unjudged, "tecnica_origen"] = SUPPORT_ORIGIN
    decision["n_predicciones"] = decision["n_predicciones"].fillna(0).astype(int)
    return decision


def error_bands(predictions: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """Per technique and horizon, the quantiles of err_norm in the selection months (every judged id)."""
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
    return (selection.groupby(["tecnica", "h"])["err_norm"]
            .agg(q_low_norm=lambda errors: errors.quantile(configuration.band_low_quantile),
                 q_high_norm=lambda errors: errors.quantile(configuration.band_high_quantile),
                 n_predicciones="size")
            .reset_index())


def exam_precision(predictions: pd.DataFrame, decision: pd.DataFrame, bands: pd.DataFrame, configuration: Config) -> tuple:
    """The chosen technique and the default technique in the exam months: per id and for the total."""
    default_technique = configuration.default_technique
    exam = predictions[predictions["proposito"] == PURPOSE_EXAM]
    chosen_rows = exam.merge(decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"]], on=[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica"])
    chosen_rows = chosen_rows.merge(bands[["tecnica", "h", "q_low_norm", "q_high_norm"]], on=["tecnica", "h"], how="left")
    chosen_rows["dentro_banda"] = ((chosen_rows["err_norm"] >= chosen_rows["q_low_norm"])
                                   & (chosen_rows["err_norm"] <= chosen_rows["q_high_norm"])).astype(int)
    default_rows = exam[exam["tecnica"] == default_technique]

    def summary(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (rows.assign(abs_pp=rows["err_pp"].abs(), abs_norm=rows["err_norm"].abs())
                .groupby([COMPOSITION_ID_COLUMN, "tramo_h"])
                .agg(**{f"{prefix}_err_pp_medio": ("abs_pp", "mean"), f"{prefix}_sesgo_pp": ("err_pp", "mean"),
                        f"{prefix}_err_norm_medio": ("abs_norm", "mean")}))

    exam_by_pool = summary(chosen_rows, "elegida").join(summary(default_rows, "por_defecto"), how="left").reset_index()
    exam_by_pool = exam_by_pool.merge(decision[[COMPOSITION_ID_COLUMN, "tramo_h", "tecnica", "tecnica_origen"]],
                                      on=[COMPOSITION_ID_COLUMN, "tramo_h"], how="left")
    exam_by_pool["mejora_pp"] = exam_by_pool["por_defecto_err_pp_medio"] - exam_by_pool["elegida_err_pp_medio"]
    exam_by_pool["dentro_banda"] = chosen_rows.groupby([COMPOSITION_ID_COLUMN, "tramo_h"])["dentro_banda"].mean().to_numpy()

    # the TOTAL: Σ predicted renewals vs Σ real renewals of every judged id, per exam month and horizon
    def total_of(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (rows.assign(pred_units=rows["tasa_pred"] * rows["vencen_real"], real_units=rows["tasa_real"] * rows["vencen_real"])
                .groupby(["mes_objetivo", "h"])
                .agg(**{f"{prefix}_renovadas_pred": ("pred_units", "sum"), "renovadas_reales": ("real_units", "sum"),
                        "vencen": ("vencen_real", "sum")}))

    exam_total = total_of(chosen_rows, "elegida").join(total_of(default_rows, "por_defecto")[["por_defecto_renovadas_pred"]]).reset_index()
    for prefix in ("elegida", "por_defecto"):
        exam_total[f"{prefix}_error_pct"] = (exam_total[f"{prefix}_renovadas_pred"] - exam_total["renovadas_reales"]) / exam_total["renovadas_reales"]
    return exam_by_pool, exam_total, chosen_rows


def check_backtest(predictions: pd.DataFrame, decision: pd.DataFrame, pool_reference: pd.DataFrame,
                   exam_rows: pd.DataFrame, configuration: Config, check_log: list) -> None:
    """Checks 1 to 5."""
    # [1] no peeking
    peeked = predictions[(predictions["ultimo_mes_visto"] > predictions["origen"])
                         | (predictions["origen"] >= predictions["mes_objetivo"])]
    configuration.log_check(STEP_2_6_LABEL, check_log, "no prediction saw its target month or anything after its origin",
                            peeked.empty, failure_detail=f"{len(peeked):,} predictions saw beyond their origin",
                            context=f"{len(predictions):,} predictions")

    # [2] a decision per id and band
    expected = len(pool_reference) * len(configuration.horizon_bands)
    configuration.log_check(STEP_2_6_LABEL, check_log, "every estimation id has a decision in every horizon band",
                            len(decision) == expected and decision["tecnica"].notna().all(),
                            failure_detail=f"{len(decision):,} decisions, {expected:,} expected",
                            context=f"{len(pool_reference):,} ids × {len(configuration.horizon_bands)} bands")

    # [3] the default technique is always there to compare with
    targets = predictions.groupby([COMPOSITION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].apply(set)
    without_default = int(sum(configuration.default_technique not in techniques for techniques in targets))
    configuration.log_check(STEP_2_6_LABEL, check_log, "the default technique was predicted wherever another technique was",
                            without_default == 0,
                            failure_detail=f"{without_default:,} targets without the default technique")

    # [4] a fair ranking: the techniques are compared on the targets where all of them competed
    common = common_targets(predictions)
    configuration.log_check(STEP_2_6_LABEL, check_log, "the ranking compares the techniques on common targets",
                            len(common) > 0,
                            failure_detail="no target where every technique competed: the ranking would mix different targets",
                            context=f"{len(common):,} of {predictions.groupby([COMPOSITION_ID_COLUMN, 'mes_objetivo', 'h']).ngroups:,} "
                                    f"targets have every technique; the others lack a history of 24 months")

    # [5] the band holds what it promises in the exam
    coverage = float(exam_rows["dentro_banda"].mean()) if len(exam_rows) else np.nan
    nominal = configuration.band_high_quantile - configuration.band_low_quantile
    configuration.log_check(STEP_2_6_LABEL, check_log,
                            f"in the exam, the band holds at least {configuration.min_exam_coverage:.0%} of the chosen technique's errors",
                            bool(coverage >= configuration.min_exam_coverage),
                            failure_detail=f"only {coverage:.0%} inside (nominal {nominal:.0%}): the band is too narrow",
                            context=f"{coverage:.0%} inside (nominal {nominal:.0%}, {len(exam_rows):,} exam predictions)",
                            blocking=False)


def common_targets(predictions: pd.DataFrame) -> pd.DataFrame:
    """The (id, target, horizon) where every technique of the catalogue competed."""
    techniques_per_target = predictions.groupby([COMPOSITION_ID_COLUMN, "mes_objetivo", "h"])["tecnica"].nunique()
    return techniques_per_target[techniques_per_target == len(CATALOGUE)].reset_index()[[COMPOSITION_ID_COLUMN, "mes_objetivo", "h"]]


def log_backtest_report(predictions: pd.DataFrame, decision: pd.DataFrame, exam_by_pool: pd.DataFrame,
                        exam_total: pd.DataFrame, pool_reference: pd.DataFrame, configuration: Config) -> None:
    """Action 9: the ranking, the choices and the precision in the exam, as tables."""
    configuration.log_action(STEP_2_6_LABEL, 9, "ranking of the techniques in the SELECTION months (mean |err_norm|: 1 = "
                                            "as close as chance allows; same targets for every technique):")
    selection = predictions[predictions["proposito"] == PURPOSE_SELECTION]
    common = common_targets(predictions)
    selection = selection.merge(common, on=[COMPOSITION_ID_COLUMN, "mes_objetivo", "h"])
    ranking = (selection.assign(abs_norm=selection["err_norm"].abs(), abs_pp=selection["err_pp"].abs())
               .groupby(["tramo_h", "tecnica"]).agg(err_norm_medio=("abs_norm", "mean"), err_pp_medio=("abs_pp", "mean"),
                                                    predicciones=("abs_norm", "size"))
               .reset_index().sort_values(["tramo_h", "err_norm_medio"]))
    configuration.show_table(ranking)

    configuration.logger.doc(f"[{STEP_2_6_LABEL}] the chosen technique per band (ids and the money their series predict):")
    chosen_money = decision.merge(pool_reference[[COMPOSITION_ID_COLUMN, "usd_por_predecir"]], on=COMPOSITION_ID_COLUMN)
    configuration.show_table(chosen_money.groupby(["tramo_h", "tecnica", "tecnica_origen"])
                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), usd_por_predecir=("usd_por_predecir", "sum"))
                             .reset_index().sort_values(["tramo_h", "usd_por_predecir"], ascending=[True, False]))

    configuration.logger.doc(f"[{STEP_2_6_LABEL}] precision in the EXAM, per band (mean over the judged ids; err in pp of rate; "
                             f"mejora = default technique's error − chosen's error):")
    configuration.show_table(exam_by_pool.groupby("tramo_h")
                             .agg(ids=(COMPOSITION_ID_COLUMN, "size"), elegida_err_pp=("elegida_err_pp_medio", "mean"),
                                  por_defecto_err_pp=("por_defecto_err_pp_medio", "mean"), mejora_pp=("mejora_pp", "mean"),
                                  elegida_sesgo_pp=("elegida_sesgo_pp", "mean"), dentro_banda=("dentro_banda", "mean"))
                             .reset_index())

    configuration.logger.doc(f"[{STEP_2_6_LABEL}] precision of the TOTAL in the exam (Σ predicted renewals vs Σ real of the "
                             f"judged ids; error_pct = (predicted − real) / real):")
    configuration.show_table(exam_total[["mes_objetivo", "h", "vencen", "renovadas_reales", "elegida_renovadas_pred",
                                         "elegida_error_pct", "por_defecto_renovadas_pred", "por_defecto_error_pct"]])


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 2.7 · THE BOX OF THE RATE  (until 25-oct-2026: step_13_rate_box.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE BOX OF THE RATE: the renewal rate of every forecast series h months ahead, learned with only
# the closed months up to an origin. One box that the forecast, the exam of every forecast series and the audit call,
# so what is examined is exactly what is used.
#
# THE INTERFACE
#     box = RateBox(rated_units, series_estimate, pool_series, backtest, reference_members, configuration)
#     rates = box.predict(rows, as_of)                 the rate of every row, learned up to as_of
#     rates = box.predict(rows, as_of, DEFAULT_MODE)   the same with the default technique for every composition
#     low, high = box.band(rates, units_due)           the interval of the rate (the error bands of step 2.6)
#   rows   future rows (or exam rows): the series (fs_id), the month, and the dims of the series (its mandatory dims and
#          its customer states, for a series with no composition)
#   rates  per row: h (month − as_of), the composition, the technique used, the composition's prediction
#          (tasa_pool_h), its level and its reference's level at as_of, z, the credibility shift (logit), the rate of
#          the series (tasa) and where it comes from (origen_tasa: pool · celda_mandatory · global)
#
# WHAT IS INSIDE THE BOX (what can be improved here without touching anything else)
#   · pool    the series has a composition with at least min_months closed months at as_of (1 for the forecast; the
#             exam asks exam_min_months_to_predict, 3: at an early origin a composition of 1-2 months is noise): the
#             technique chosen for that composition and horizon band (step 2.6; the default technique in DEFAULT_MODE or
#             when the chosen one cannot predict) predicts the composition's rate at h with the months up to as_of;
#             the series moves it toward its credibility reference by (1 − z) × (logit reference level − logit
#             composition level), both levels with the months up to as_of (prediction.py)
#   · celda   no composition (or too short a history): the level of its mandatory cell up to as_of; with
#             rate_fallback_with_states, of its mandatory cell AND its combination of customer states (a softcancel series
#             then never takes the level of a cell made mostly of neutral units; the exam says it predicts the total worse,
#             so it is off by default)
#   · global  not even that: the portfolio's level (with rate_fallback_with_states, first its combination's)
#
# THE STAGE 8.6 THE PIPELINE (label 2.7, after the backtest): the box is built once and checked; the forecast, the exam
# and the audit receive it.
#
# Actions (logged as they are done):
#   1. the box: the compositions and their closed months, the techniques chosen per band, the references
#   2. the rate of every composition at h = 1 at the last closed month, as a check of the box
#   3. check the box                                                    checks 1-2
#   4. count the checks; stop if any failed
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. every composition with enough closed months predicts a rate in [0, 1] at h = 1
#    2. the levels of the box at the last closed month are the ones of the ladder (step 2.3): the same composition
#       and reference rates
#
# Output: the box (RateBox) · no table.

# ─── the stage ───────────────────────────────────────────────────────────────────
STEP_2_7_LABEL = "2.7"
STEP_2_7_NAME = "THE BOX OF THE RATE"
STEP_2_7_PURPOSE = ("put the prediction of the renewal rate in ONE box: the technique chosen for every composition and "
                "horizon, moved toward its credibility reference, learned with the closed months up to any origin; the "
                "forecast, the exam and the audit call it")
STEP_2_7_ACTIONS = ["the box: the compositions and their closed months, the techniques chosen per band, the references",
                "the rate of every composition at h = 1 at the last closed month, as a check of the box",
                "check the box (checks 1-2)",
                "count the checks; stop if any failed"]
STEP_2_7_OUTPUT = "the box (RateBox) for the forecast, the exam and the audit"

CHOSEN_MODE = "chosen"           # the technique chosen by the backtest for the composition and band
DEFAULT_MODE = "default"         # the default technique for every composition (the cascade's por_defecto)
NO_COMPOSITION_TECHNIQUE = "mandatory_cell"


class RateBox:
    """THE BOX OF THE RATE (see the header of this module)."""

    def __init__(self, rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                 backtest: dict, reference_members: pd.DataFrame, configuration: Config):
        self.configuration = configuration
        period, due, renewed = configuration.period_col, configuration.pipeline_units_col, configuration.renewed_units_col
        self.decision = backtest["decision"].set_index([COMPOSITION_ID_COLUMN, "tramo_h"])["tecnica"].to_dict()
        self.bands = backtest["bands"]
        self.states = negative_states(configuration)
        # the compositions: their monthly rate (logit), renewed and due, indexed once
        composition_truth = pool_series[pool_series[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & pool_series[RATE_COLUMN].notna()
                                        & (pool_series["vencen"] > 0)]
        self.composition_index = index_history(composition_truth, COMPOSITION_ID_COLUMN, period, [RATE_COLUMN, "renovadas", "vencen"])
        for entry in self.composition_index.values():
            entry["rate_logit"] = logit(entry[RATE_COLUMN])
            entry["rate_logit"].flags.writeable = False
        # the closed months of every series (no gaps): the levels of the references, the cells and the portfolio
        closed = rated_units[(rated_units[SYNTHETIC_COLUMN] == 0) & rated_units[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                             & (rated_units[due] > 0)].copy()
        closed["_combinacion"] = state_combination(closed, self.states)
        closed["_celda"] = self.cell_key(closed)
        self.closed = closed
        self.reference_index = {}
        if reference_members is not None and len(reference_members):
            members = closed.merge(reference_members[[SERIES_ID_COLUMN, "credibility_ref_id"]], on=SERIES_ID_COLUMN)
            self.reference_index = index_history(members.groupby(["credibility_ref_id", period])[[renewed, due]].sum().reset_index(),
                                                 "credibility_ref_id", period, [renewed, due])
        self.estimate = series_estimate.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN)[
            [COMPOSITION_ID_COLUMN, "z", "credibility_ref_id"]]
        self._composition_cache, self._fallback_cache = {}, {}

    # ─── what the box knows at an origin ───
    def composition_history(self, composition_id, as_of) -> tuple:
        """(logit of its monthly rate, calendar months, Σ renewed, Σ due) of a composition up to as_of; None if unknown."""
        entry = self.composition_index.get(composition_id)
        if entry is None:
            return None
        known = months_known(entry, as_of)
        return (entry["rate_logit"][:known], entry["month"][:known], float(entry["renovadas"][:known].sum()),
                float(entry["vencen"][:known].sum()))

    def composition_prediction(self, composition_id, as_of, horizon: int, mode: str = CHOSEN_MODE,
                               min_months: int = 1) -> tuple:
        """(technique used, predicted rate, level at as_of) of a composition h months after as_of; (None, nan, nan)
        with fewer than min_months closed months."""
        key = (composition_id, as_of, int(horizon), mode, int(min_months))
        if key in self._composition_cache:
            return self._composition_cache[key]
        configuration = self.configuration
        history = self.composition_history(composition_id, as_of)
        if history is None or len(history[0]) < max(int(min_months), 1):
            result = (None, np.nan, np.nan)
        else:
            rate_logit, months, renewed, due = history
            band_name = band_of_horizon(int(horizon), configuration.horizon_bands)
            technique = (configuration.default_technique if mode == DEFAULT_MODE
                         else self.decision.get((composition_id, band_name), configuration.default_technique))
            technique, rate = predict_composition(rate_logit, months, technique, int(horizon), configuration.default_technique)
            result = (technique, rate, renewed / due if due > 0 else np.nan)
        self._composition_cache[key] = result
        return result

    def reference_level(self, reference_id, as_of) -> float:
        entry = self.reference_index.get(reference_id)
        if entry is None:
            return np.nan
        known = months_known(entry, as_of)
        due = entry[self.configuration.pipeline_units_col][:known].sum()
        return float(entry[self.configuration.renewed_units_col][:known].sum() / due) if due > 0 else np.nan

    def cell_key(self, rows: pd.DataFrame) -> pd.Series:
        """The fallback cell of every row: its mandatory dims, and its combination of customer states when
        rate_fallback_with_states (the exam of 25-oct on the like-real synthetic: with the states the error of the
        total at 6 months rises from 1.4 % to 1.5 %, and the honest one from 1.5 % to 1.7 %; off by default)."""
        cell = join_columns(rows, self.configuration.business_mandatory_dims)
        if self.configuration.rate_fallback_with_states:
            cell = cell + "|" + state_combination(rows, self.states)
        return cell

    def fallback_levels(self, as_of) -> tuple:
        """(level of every cell with its states, level of every combination of states, global level) up to as_of."""
        if as_of not in self._fallback_cache:
            renewed, due = self.configuration.renewed_units_col, self.configuration.pipeline_units_col
            known = self.closed[self.closed[self.configuration.period_col] <= as_of]
            by_cell = known.groupby("_celda")[[renewed, due]].sum()
            by_combination = known.groupby("_combinacion")[[renewed, due]].sum()
            self._fallback_cache[as_of] = (
                (by_cell[renewed] / by_cell[due].where(by_cell[due] > 0)).to_dict(),
                (by_combination[renewed] / by_combination[due].where(by_combination[due] > 0)).to_dict(),
                float(known[renewed].sum() / known[due].sum()) if known[due].sum() > 0 else np.nan)
        return self._fallback_cache[as_of]

    # ─── the prediction ───
    def predict(self, rows: pd.DataFrame, as_of, mode: str = CHOSEN_MODE, min_months: int = 1) -> pd.DataFrame:
        """The rate of every row h months after as_of (see the header). Same index as rows. A composition with fewer
        than min_months closed months at as_of does not predict (the row takes its cell with its states)."""
        configuration = self.configuration
        period = configuration.period_col
        result = pd.DataFrame(index=rows.index)
        result["h"] = rows[period].map(lambda month: (month - as_of).n).astype(int)
        estimate = self.estimate.reindex(rows[SERIES_ID_COLUMN].to_numpy())
        result[COMPOSITION_ID_COLUMN] = estimate[COMPOSITION_ID_COLUMN].to_numpy()
        result["z"] = estimate["z"].to_numpy()
        result["credibility_ref_id"] = estimate["credibility_ref_id"].to_numpy()
        predictions = {pair: self.composition_prediction(pair[0], as_of, pair[1], mode, min_months)
                       for pair in set(zip(result[COMPOSITION_ID_COLUMN].dropna(), result.loc[result[COMPOSITION_ID_COLUMN].notna(), "h"]))}
        pairs = list(zip(result[COMPOSITION_ID_COLUMN], result["h"]))
        result["tecnica"] = [predictions.get(pair, (None, np.nan, np.nan))[0] for pair in pairs]
        result["tasa_pool_h"] = np.array([predictions.get(pair, (None, np.nan, np.nan))[1] for pair in pairs], dtype=float)
        result["group_rate"] = np.array([predictions.get(pair, (None, np.nan, np.nan))[2] for pair in pairs], dtype=float)
        references = {reference: self.reference_level(reference, as_of) for reference in set(result["credibility_ref_id"].dropna())}
        result["ref_rate"] = result["credibility_ref_id"].map(references)
        rate, shift = shifted_rate(result["tasa_pool_h"], result["z"], result["ref_rate"], result["group_rate"],
                                   configuration.apply_credibility_shift)
        has_pool = result["tasa_pool_h"].notna().to_numpy()
        result["credibility_shift_logit"] = np.where(has_pool, shift, 0.0)
        result["tasa"] = np.where(has_pool, rate, np.nan)
        result["origen_tasa"] = np.where(has_pool, RATE_FROM_POOL, None)
        # no composition: its mandatory cell (with its states when rate_fallback_with_states), then the portfolio
        cell_level, combination_level, global_level = self.fallback_levels(as_of)
        combination = state_combination(rows, self.states)
        from_cell = self.cell_key(rows).map(cell_level)
        missing = result["tasa"].isna()
        result.loc[missing & from_cell.notna(), "origen_tasa"] = RATE_FROM_CELL
        result.loc[missing & from_cell.notna(), "tasa"] = from_cell
        still_missing = result["tasa"].isna()
        result.loc[still_missing, "origen_tasa"] = RATE_FROM_GLOBAL
        portfolio = (combination[still_missing].map(combination_level).fillna(global_level)
                     if configuration.rate_fallback_with_states else global_level)
        result.loc[still_missing, "tasa"] = portfolio
        result["tecnica"] = result["tecnica"].where(result["origen_tasa"] == RATE_FROM_POOL, NO_COMPOSITION_TECHNIQUE)
        return result

    def band(self, rates: pd.DataFrame, units_due) -> tuple:
        """(low, high) of the rate of every row: the error quantiles of its technique at its horizon (step 2.6; a row
        with no composition takes the default technique's) × the binomial error with its units due."""
        techniques = rates["tecnica"].where(rates["tecnica"] != NO_COMPOSITION_TECHNIQUE, self.configuration.default_technique)
        q_low, q_high = band_quantiles(self.bands, techniques, rates["h"], self.configuration)
        return rate_band(rates["tasa"], units_due, q_low, q_high)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STAGE: build the box and check it
# ═══════════════════════════════════════════════════════════════════════════════════

def build_the_rate_box(rated_units: pd.DataFrame, series_estimate: pd.DataFrame, pool_series: pd.DataFrame,
                       backtest: dict, ladder: dict, configuration: Config) -> RateBox:
    """The stage 2.7 of the pipeline: the box of the rate, checked."""
    configuration.log_step_start(STEP_2_7_LABEL, STEP_2_7_NAME, STEP_2_7_PURPOSE, STEP_2_7_ACTIONS, STEP_2_7_OUTPUT)
    check_log = []
    last_closed = configuration.calendar_boundaries()["current"] - 1

    # [1] the box
    box = RateBox(rated_units, series_estimate, pool_series, backtest, ladder.get("reference_members"), configuration)
    chosen = backtest["decision"].groupby("tramo_h")["tecnica"].value_counts()
    configuration.log_action(STEP_2_7_LABEL, 1, f"{len(box.composition_index):,} compositions with closed months · "
                                            f"{len(box.reference_index):,} credibility references · techniques per band: "
                                            + " · ".join(f"{band} {block.droplevel(0).head(3).to_dict()}"
                                                         for band, block in chosen.groupby(level=0)))

    # [2] every composition at h = 1 at the last closed month
    at_one = pd.DataFrame([(composition_id, *box.composition_prediction(composition_id, last_closed, 1))
                           for composition_id in box.composition_index],
                          columns=[COMPOSITION_ID_COLUMN, "tecnica", "tasa", "nivel"])
    predicted = at_one[at_one["tasa"].notna()]
    short = sum(len(entry["month"]) < configuration.exam_min_months_to_predict for entry in box.composition_index.values())
    configuration.log_action(STEP_2_7_LABEL, 2, f"{len(predicted):,} of {len(at_one):,} compositions predict at h = 1 for the "
                                            f"forecast · {short:,} have fewer than {configuration.exam_min_months_to_predict} "
                                            f"closed months: the exam does not predict them at an origin that short (their "
                                            f"series take the level of their cell and states there)")

    # [3] the checks
    configuration.log_action(STEP_2_7_LABEL, 3, "checking the box")
    configuration.log_check(STEP_2_7_LABEL, check_log, "every composition with enough closed months predicts a rate in [0, 1] at h = 1",
                            bool(predicted["tasa"].between(0, 1).all()),
                            failure_detail=f"{int((~predicted['tasa'].between(0, 1)).sum())} compositions outside [0, 1]",
                            context=f"{len(predicted):,} compositions")
    ladder_rates = series_estimate.dropna(subset=[COMPOSITION_ID_COLUMN, "group_rate"]).drop_duplicates(COMPOSITION_ID_COLUMN)
    levels = at_one.set_index(COMPOSITION_ID_COLUMN)["nivel"]
    compared = ladder_rates[ladder_rates[COMPOSITION_ID_COLUMN].isin(levels.dropna().index)]
    differences = (compared[COMPOSITION_ID_COLUMN].map(levels) - compared["group_rate"]).abs()
    configuration.log_check(STEP_2_7_LABEL, check_log,
                            "the levels of the box at the last closed month are the ones of the ladder (step 2.3)",
                            bool((differences <= configuration.rate_tolerance).all()), blocking=False,
                            failure_detail=f"{int((differences > configuration.rate_tolerance).sum())} compositions differ "
                                           f"(largest {differences.max():.4f})",
                            context=f"{len(compared):,} compositions compared")

    # [4] the count of the checks; stop if anything failed
    configuration.log_action(STEP_2_7_LABEL, 4, "counting the checks")
    configuration.log_check_summary(STEP_2_7_LABEL, STEP_2_7_NAME, check_log)
    return box


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 2 · THE BLOCK: run the steps in order, keep the results
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B2"
BLOCK_NAME = "RATE"
BLOCK_MISSION = "the renewal rate of any row of pipeline, at any horizon, with its band"
BLOCK_READS = ("rated_units", "series_rate", "series_table")
BLOCK_GIVES = ("dimension_decision", "ladder", "series_estimate", "money_by_level", "pool_series", "pool_reference",
               "pool_dynamics", "portfolio_profile", "portfolio_dynamics", "backtest", "rate_box")


class RateBlock:
    """BLOCK 2 · RATE (see the header of this module): its results are attributes, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.dimension_decision = None
        self.ladder = None
        self.series_estimate = None
        self.money_by_level = None
        self.pool_series = None
        self.pool_reference = None
        self.pool_dynamics = None
        self.portfolio_profile = None
        self.portfolio_dynamics = None
        self.backtest = None
        self.rate_box = None
        self.seconds = {}                 # how long every step took

    def run(self, rated_units: pd.DataFrame, series_rate: pd.DataFrame, series_table: pd.DataFrame) -> "RateBlock":
        """Every step of the block, in order."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 2, BLOCK_NAME, BLOCK_MISSION)

        started = time.time()
        self.dimension_decision = analyse_dimensions(series_rate, series_table, configuration)
        self.seconds["2.1 dimensions"] = time.time() - started

        started = time.time()
        self.ladder = build_ladder_groups(rated_units, series_rate, series_table, self.dimension_decision, configuration)
        self.seconds["2.2 ladder_groups"] = time.time() - started

        started = time.time()
        self.series_estimate, self.money_by_level = climb_the_ladder(series_rate, self.ladder, configuration)
        self.seconds["2.3 credibility"] = time.time() - started

        started = time.time()
        self.pool_series, self.pool_reference = build_pool_series(rated_units, self.ladder, self.series_estimate,
                                                                  configuration)
        self.seconds["2.4 pool_series"] = time.time() - started

        started = time.time()
        self.pool_dynamics, self.portfolio_profile, self.portfolio_dynamics = measure_dynamics(
            self.pool_series, self.pool_reference, configuration, rated_units=rated_units)
        self.seconds["2.5 dynamics"] = time.time() - started

        started = time.time()
        self.backtest = run_backtest(self.pool_series, self.pool_reference, configuration)
        self.seconds["2.6 backtest"] = time.time() - started

        started = time.time()
        self.rate_box = build_the_rate_box(rated_units, self.series_estimate, self.pool_series, self.backtest,
                                           self.ladder, configuration)
        self.seconds["2.7 rate_box"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
