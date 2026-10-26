"""
block_4_customer_states.py — BLOCK 4 · CUSTOMER STATES: how the portfolio matures before it falls due.

MISSION: say how many units of every future month will carry each customer state (softcancel, dormant…) when they
fall due. The extract has TODAY's states: a licence due in 10 months has not had the time to be softcancelled yet.

MECHANISM: the maturation curves of every state, learned from the closed months (the share of the final state already
visible k months before due), in one box (StateMaturation) that moves units between states without creating them.
The forecast (block 7) values what it moves; the exam of step 7.1 predicts the states too (the honest exam): that is
the exam of this block.

THE STEPS:
  4.1  maturation of the states   learn the curves and leave the box ready; show what it learned and how its curve did

READS (block 1): fine_table.
GIVES: state_maturation (the box).

THE INTERFACE:
  block = CustomerStatesBlock(configuration).run(fine_table)
  block.state_maturation    block.results()    block.seconds
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time

import numpy as np
import pandas as pd

from config import (ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, Config, DUE_UNITS_AT_DUE_COLUMN, ROLE_PROJECTION, ROLE_TEST,
                    SERIES_ID_COLUMN, UNITS_AT_DUE_COLUMN, UNITS_TODAY_COLUMN, WITH_DATES_COLUMN, join_columns,
                    negative_states, region_columns_of, state_combination)
from graficos import draw_maturation_by_region, draw_maturation_explained
from maturation import (CELL_LEVEL, GROUP_LEVEL, MaturationCurve, curve_history, examine_the_curve, group_dims,
                        is_in_state, visible_units)
from time_series_charts import chart_path, draw_without_stopping


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 4.1 · MATURATION OF THE CUSTOMER STATES  (until 25-oct-2026: step_17_customer_states.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE MATURATION OF THE CUSTOMER STATES: of the units that fall due in a future month, how many
# will carry a negative state (softcancel, dormant, not_installed and their combinations) WHEN they fall due?
#
# The extract keeps the LAST state of a state: a licence that falls due in ten months has not had the time to stop
# using the product or to be softcancelled, and a projected or simulated row has no states at all. The forecast with
# today's states renews all those units as neutral. This stage predicts how the states grow until the due month: the
# third prediction of the forecast, next to the rate and the uplift.
#
# ONE BOX, ONE INTERFACE. Everything that predicts the states lives here; the forecast (step 7.2) and the exam (step 7.1)
# only call it, so a better box makes a better forecast and a better exam without touching anything else:
#
#     box = StateMaturation(fine_table, as_of, configuration)    learns from the closed months up to as_of
#     table = box.mature(units)                                  predicts, for the units seen at as_of
#
#   units  one row per future row (or forecast series) and month, as seen at as_of: the dims of its series
#          (rate_series_columns, its states included), the month, unidades_hoy (the units due seen today), con_fechas
#          (True when the row carries the maturation columns of the dated states: a row of the extract) and those columns
#   table  one row per GROUP (the series without the states), month and COMBINATION of states (neutral, dormant,
#          softcancel, dormant+softcancel…) present today or receiving units: the dims of the group, the state columns
#          as the data writes them, unidades_hoy, unidades_al_vencer (THE PREDICTION), unidades_grupo (the units of
#          the group and month), cuota_final_<state> (the final share of every state the box aimed at in the group and
#          month) and nivel (where the final mix was measured: grupo · celda · cartera · sin_historia)
#   The units only move: Σ unidades_al_vencer = Σ unidades_hoy in every group and month. The box never values
#   anything: step 7.2 values the units that move (they leave a neutral row at its average price and renew at the rate
#   and the uplift of the series they arrive at); step 7.1 examines the box at the origin of every exam month.
#   The IDENTITY box (distribute_marks off, or no negative state) returns unidades_al_vencer = unidades_hoy: the forecast
#   is the one with today's states, and the report shows a maturation of 0.
#
# WHAT IS INSIDE THE BOX TODAY (what can be improved here without touching anything else)
#   For every group and month:
#   · the final mix: the share of every combination in the same calendar month of the last maturation_history_months
#     closed months (there the states are final): of the group; with fewer than maturation_min_units, of its mandatory
#     cell; with fewer still, of the portfolio
#   · a state with maturation columns (maturation_dims) takes its final share from its MATURATION CURVE (maturation.py)
#     on the rows with dates: V_h + (1 − V_h) × p(h), V_h the share already dated h months or more before due; the rows
#     without dates (projected, simulated) take its historical final share; the combinations that carry the state keep
#     their historical split
#   · a state only grows: the neutral units move to every combination with a state in proportion to what it lacks (final −
#     today, when positive), never more than the neutral units there are; no state ends above its final share through
#     a combination; a combination already above its final share stays as it is (the detector warns of it)
#   · the units in the state of today do not move (a dormant unit that will also take softcancel is not followed yet)
#
# THE STAGE 8.6 THE PIPELINE (label 4.1, before step 7.2): it learns the box at the last closed month and shows what it
# knows: the curve of every dated state and its exam, and what every combination of states renews. Step 7.2 calls the
# box and shows, with label 4.1, the maturation month by month (REPARTO DE ESTADOS DEL CLIENTE) and the detector of states above
# their final share. Step 7.1 calls honest_total_exam (here): the box at the origin of every exam month.
#
# Actions (logged as they are done):
#   1. the box: the closed months it learns from, the states with a curve and the states with a historical share
#   2. the curve of every dated state and its exam on the exam months
#   3. the maturation of every state by region (the first region level), as a table and a chart: one line per region
#   4. what every combination of states renews in the closed months it learns from
#   5. check the box                                                   checks 1-3
#   6. count the checks; stop if any failed
#
# THE MATURATION BY REGION (action 3, since 26-oct-2026). For every negative state and every value of the first region
# level, the share of the final state already visible k months before due (k = maturation_months … 0):
#   · a DATED state (maturation_dims, softcancel): the PAST curve, the average of the cohorts the box learns from
#     (the last maturation_history_months closed months), weighted by units: Σ V_k / Σ M per region. At k = 0 it is
#     100 %: the whole final state is seen when the licence falls due.
#   · an UNDATED state (dormant, not_installed): the past cannot be seen, the extract keeps only the last state. What
#     can be seen is the pipeline TODAY: of the units due k months ahead, the share already in the state, over the
#     final share of the region in the closed months. It is a picture of today, not an average of the past; the
#     forecast takes the historical final share for these states.
#   The dated states get the view of today too, in the table, to compare it with their past curve.
#   The first region level is region_columns_of(configuration)[0]; at most maturation_regions_shown regions (Config),
#   the largest by units due.
#
# Checks (logged as they are made, numbered, at the level of their status):
#    1. the box learns only from closed months up to the last closed one
#    2. every dated state's curve beats the states seen then in its exam, by ≤ maturation_curve_max_error_pp (WARN: a curve
#       that fails is not used, the state takes its historical share: rejected_maturation_dims; since 26-oct-2026)
#    3. every dated state has a curve learned from cohorts that carry it (WARN: without them its maturation is 0)
#
# Output: the box (StateMaturation) for step 7.2 · no table (the maturation of every row is in sff_nucleo, the maturation
# by month in sff_forecast_maduracion, written by step 7.2).

# ─── the stage ───────────────────────────────────────────────────────────────────
STEP_4_1_LABEL = "4.1"
STEP_4_1_NAME = "MATURATION OF THE CUSTOMER STATES"
STEP_4_1_PURPOSE = ("predict how many units of every future month will carry a negative state (softcancel, dormant…) when "
                "they fall due: one box that learns from the closed months and that the forecast (step 7.2) and the "
                "exam (step 7.1) call; here, what it learns and how well its curve did")
STEP_4_1_ACTIONS = ["the box: the closed months it learns from, the states with a curve and the states with a historical share",
                "the curve of every dated state and its exam on the exam months, and the chart that explains it (always shown): "
                "when the state is set, what the curve adds to today's pipeline, and its error by how far ahead",
                "the maturation of every state by region (first region level): table and chart, one line per region",
                "what every combination of states renews in the closed months it learns from",
                "check the box (checks 1-3)",
                "count the checks; stop if any failed"]
STEP_4_1_OUTPUT = "the box (StateMaturation) for step 7.2; the maturation is shown and written by step 7.2"

# ─── named constants ─────────────────────────────────────────────────────────────
NEUTRAL = "neutral"                              # the combination without any of the box's states
LEVEL_GROUP, LEVEL_CELL, LEVEL_PORTFOLIO = "grupo", "celda", "cartera"
LEVEL_NO_HISTORY = "sin_historia"                # no closed month of the calendar month: nothing moves
MATURATION_EXAM_HORIZONS = (1, 3, 6, 12)         # the horizons the curve is examined at (plus the backtest's)
PORTFOLIO_KEY = "cartera"
VIEW_PAST, VIEW_TODAY = "pasado", "hoy"          # the past curve (dated states) · the pipeline as seen today
MATURATION_BY_REGION_CHART = "04_1_maduracion_por_region"
MATURATION_EXPLAINED_CHART = "04_1_curva_de_maduracion_explicada"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE BOX
# ═══════════════════════════════════════════════════════════════════════════════════

def flag_values(fine_table: pd.DataFrame, state: str) -> tuple:
    """(active, inactive): how the data writes a state on and off (True / False, 1 / 0…)."""
    values = fine_table[state].dropna() if state in fine_table.columns else pd.Series(dtype=object)
    active = values[values.isin(ACTIVE_FLAG_VALUES)]
    inactive = values[~values.isin(ACTIVE_FLAG_VALUES)]
    return (active.mode().iloc[0] if len(active) else 1), (inactive.mode().iloc[0] if len(inactive) else 0)


class StateMaturation:
    """THE BOX: learns how the states grow from the closed months up to an origin (as_of) and predicts, for the units
    seen at that origin, the units of every combination of states when they fall due. See the header of this module
    for the interface and for what is inside."""

    def __init__(self, fine_table: pd.DataFrame, as_of, configuration: Config, states: list = None):
        self.configuration = configuration
        self.as_of = as_of
        self.states = list(states) if states is not None else negative_states(configuration)
        self.states = [state for state in self.states if state in fine_table.columns]
        self.identity = not configuration.distribute_marks or not self.states
        # the group of the box: the forecast series without the box's states
        self.group_columns = [column_name for column_name in configuration.rate_series_columns if column_name not in self.states]
        self.flags = {state: flag_values(fine_table, state) for state in self.states}
        self.minimum = configuration.maturation_min_units
        self.history = curve_history(fine_table, configuration, up_to=as_of) if not self.identity else fine_table.iloc[0:0]
        self.history_months = sorted(self.history[configuration.period_col].unique()) if len(self.history) else []
        self.curves = {}
        self.curve_exams = {}             # step 4.1: the exam of the curve of every dated state (the verdict reads it)
        self.maturation_by_region = None  # step 4.1: the maturation of every state by region (maturation_by_region)
        self.mix_units, self.mix_total, self.combination_sums = {}, {}, {}
        self.combinations_with_state = []
        if self.identity or self.history.empty:
            return
        self._learn_the_mix()
        # the states whose extract dates them: their final share comes from their curve on the rows with dates
        self.curves = {state: MaturationCurve(self.history, state, configuration)
                       for state in configuration.maturation_dims_in_use if state in self.states}

    def _learn_the_mix(self) -> None:
        """The final mix of the combinations in every calendar month, and the units and renewals of every
        combination, at the three levels (group, cell, portfolio)."""
        configuration = self.configuration
        units_column = configuration.pipeline_units_col
        # only the columns the mix reads (not a copy of the whole history: memory, 26-oct-2026)
        needed = list(dict.fromkeys([configuration.period_col, *self.states, *self.group_columns,
                                     *configuration.business_mandatory_dims, units_column, configuration.pipeline_usd_col,
                                     configuration.renewed_units_col, configuration.renewed_usd_col]))
        history = self.history[needed].copy()
        history["_combinacion"] = state_combination(history, self.states)
        history["_mes"] = history[configuration.period_col].map(lambda month: month.month)
        history[LEVEL_GROUP] = join_columns(history, self.group_columns)
        history[LEVEL_CELL] = join_columns(history, configuration.business_mandatory_dims)
        history[LEVEL_PORTFOLIO] = PORTFOLIO_KEY
        history["_renovadas"] = history[configuration.renewed_units_col].fillna(0.0)
        history["_renovado_usd"] = history[configuration.renewed_usd_col].fillna(0.0)
        for level in (LEVEL_GROUP, LEVEL_CELL, LEVEL_PORTFOLIO):
            self.mix_units[level] = history.groupby([level, "_mes", "_combinacion"])[units_column].sum().to_dict()
            self.mix_total[level] = history.groupby([level, "_mes"])[units_column].sum().to_dict()
            self.combination_sums[level] = (history.groupby([level, "_combinacion"])
                                            [[units_column, "_renovadas", configuration.pipeline_usd_col, "_renovado_usd"]]
                                            .sum().apply(tuple, axis=1).to_dict())
        self.combinations_with_state = sorted(set(history["_combinacion"]) - {NEUTRAL})

    # ─── what the box knows, in words (for the log and the report) ───
    def describe(self) -> str:
        if self.identity:
            return "identity box: the states do not mature (distribute_marks off or no negative state)"
        with_curve = list(self.curves)
        with_share = [state for state in self.states if state not in self.curves]
        months = f"{self.history_months[0]}..{self.history_months[-1]}" if self.history_months else "no closed month"
        return (f"learned from {months} · curve: {', '.join(with_curve) or '—'} · historical share: "
                f"{', '.join(with_share) or '—'}")

    def group_key(self, rows: pd.DataFrame) -> pd.Series:
        """The group of the box of every row: its series without the box's states."""
        return join_columns(rows, self.group_columns)

    def combination_rate(self, group: str, cell: str, combination: str) -> tuple:
        """(units rate, USD rate) of a combination in the closed months the box learns from: of the group, its cell or
        the portfolio, the first with at least maturation_min_units units (the portfolio with any)."""
        for level, key in ((LEVEL_GROUP, group), (LEVEL_CELL, cell), (LEVEL_PORTFOLIO, PORTFOLIO_KEY)):
            sums = self.combination_sums.get(level, {}).get((key, combination))
            if sums and (sums[0] >= self.minimum or (level == LEVEL_PORTFOLIO and sums[0] > 0)):
                return sums[1] / sums[0], (sums[3] / sums[2] if sums[2] > 0 else np.nan)
        return np.nan, np.nan

    # ─── the prediction ───
    def mature(self, units: pd.DataFrame) -> pd.DataFrame:
        """The units of every group, month and combination of states when they fall due (see the header)."""
        configuration = self.configuration
        period = configuration.period_col
        rows = units[units[UNITS_TODAY_COLUMN] > 0].copy()
        if rows.empty:
            return self._empty_table()
        rows["_combinacion"] = state_combination(rows, self.states)
        rows["_grupo"] = self.group_key(rows)
        rows["_celda"] = join_columns(rows, configuration.business_mandatory_dims)
        pair = ["_grupo", period]
        today_units = rows.groupby(pair + ["_combinacion"])[UNITS_TODAY_COLUMN].sum()
        today_total = rows.groupby(pair)[UNITS_TODAY_COLUMN].sum()
        today_by_pair = {}
        for (group, month, combination), value in today_units.items():
            today_by_pair.setdefault((group, month), {})[combination] = value

        moves, finals, levels = {}, {}, {}
        if not self.identity and self.mix_total:
            moves, finals, levels = self._the_moves(rows, today_by_pair, today_total)

        # the table: every combination present today or receiving units, in every group and month
        records = []
        for (group, month), today in today_by_pair.items():
            total = today_total[(group, month)]
            fractions = moves.get((group, month), {})
            neutral_units = today.get(NEUTRAL, 0.0)
            moved_units = neutral_units * sum(fractions.values())
            final_of_state = finals.get((group, month), {state: sum(value for combination, value in today.items()
                                                                  if state in combination.split("+")) / total
                                                           for state in self.states})
            for combination in sorted(set(today) | set(fractions)):
                units_today = today.get(combination, 0.0)
                if combination == NEUTRAL:
                    at_due = units_today - moved_units
                else:
                    at_due = units_today + neutral_units * fractions.get(combination, 0.0)
                records.append((group, month, combination, units_today, at_due, total,
                                levels.get((group, month), LEVEL_NO_HISTORY),
                                *[final_of_state[state] for state in self.states]))
        table = pd.DataFrame(records, columns=["_grupo", period, "combinacion", UNITS_TODAY_COLUMN, UNITS_AT_DUE_COLUMN,
                                               "unidades_grupo", "nivel"] + [f"cuota_final_{state}" for state in self.states])
        # the dims of the group, and the states of the combination as the data writes them
        dims_of_group = rows.drop_duplicates("_grupo").set_index("_grupo")[self.group_columns]
        table = table.join(dims_of_group, on="_grupo")
        for state in self.states:
            active, inactive = self.flags[state]
            carries = table["combinacion"].map(lambda combination: state in combination.split("+"))
            table[state] = [active if carried else inactive for carried in carries]
        return table

    def _the_moves(self, rows: pd.DataFrame, today_by_pair: dict, today_total: pd.Series) -> tuple:
        """For every group and month: the share of its neutral units that goes to every combination with a state, the final
        share of every state the box aims at, and the level of the final mix."""
        configuration = self.configuration
        period = configuration.period_col
        cell_of_group = rows.drop_duplicates("_grupo").set_index("_grupo")["_celda"].to_dict()
        curve_group_of_group = (rows.assign(_grupo_curva=join_columns(rows, group_dims(configuration)))
                                .drop_duplicates("_grupo").set_index("_grupo")["_grupo_curva"].to_dict())
        # the rows with dates see their states dated h months or more before due (V_h, the same as the curve and its
        # exam); the rows without dates (projected, simulated) see nothing and take the historical final share
        seen_share_of_state, dated_share_of_pair = {}, {}
        if self.curves:
            with_dates = rows[WITH_DATES_COLUMN].astype(bool) if WITH_DATES_COLUMN in rows.columns else pd.Series(False, index=rows.index)
            months_ahead = rows[period].map(lambda month: (month - self.as_of).n).to_numpy(dtype=int)
            dated_units = rows[UNITS_TODAY_COLUMN].where(with_dates, 0.0)
            dated_total = dated_units.groupby([rows["_grupo"], rows[period]]).sum()
            dated_share_of_pair = (dated_total / today_total).to_dict()
            dated_total_or_nan = dated_total.where(dated_total > 0)
            for state in self.curves:
                seen = pd.Series(visible_units(rows, state, months_ahead, configuration), index=rows.index).where(with_dates, 0.0)
                seen_of_pair = seen.groupby([rows["_grupo"], rows[period]]).sum()
                seen_share_of_state[state] = (seen_of_pair / dated_total_or_nan).fillna(0.0).to_dict()

        moves, finals, levels = {}, {}, {}
        for (group, month), today_units in today_by_pair.items():
            total_today = today_total[(group, month)]
            today = {combination: value / total_today for combination, value in today_units.items()}
            keys = {LEVEL_GROUP: group, LEVEL_CELL: cell_of_group[group], LEVEL_PORTFOLIO: PORTFOLIO_KEY}
            level = next((candidate for candidate in (LEVEL_GROUP, LEVEL_CELL, LEVEL_PORTFOLIO)
                          if self.mix_total[candidate].get((keys[candidate], month.month), 0.0) >= self.minimum), LEVEL_PORTFOLIO)
            total = self.mix_total[level].get((keys[level], month.month), 0.0)
            if total <= 0:
                continue
            levels[(group, month)] = level
            final_share = {combination: self.mix_units[level].get((keys[level], month.month, combination), 0.0) / total
                           for combination in self.combinations_with_state}
            # a dated state: the curve on the rows with dates, the historical share on the rows without them
            for state, curve in self.curves.items():
                carrying = [combination for combination in self.combinations_with_state if state in combination.split("+")]
                historical_of_state = sum(final_share[combination] for combination in carrying)
                today_of_state = seen_share_of_state[state].get((group, month), 0.0)
                probability, _ = curve.probability({GROUP_LEVEL: curve_group_of_group[group], CELL_LEVEL: cell_of_group[group]},
                                                   (month - self.as_of).n)
                expected_of_dated_rows = today_of_state + (1.0 - today_of_state) * probability
                dated_share = dated_share_of_pair.get((group, month), 0.0)
                expected_of_state = dated_share * expected_of_dated_rows + (1.0 - dated_share) * historical_of_state
                if historical_of_state > 0:
                    for combination in carrying:
                        final_share[combination] *= expected_of_state / historical_of_state
                elif state in final_share:
                    final_share[state] = expected_of_state
            finals[(group, month)] = {state: sum(share for combination, share in final_share.items()
                                                if state in combination.split("+")) for state in self.states}
            neutral_today = today.get(NEUTRAL, 0.0)
            if neutral_today <= 0:
                continue
            gaps = {combination: max(final_share[combination] - today.get(combination, 0.0), 0.0)
                    for combination in self.combinations_with_state}
            # no state may end above its own final share through a combination (a state already above it, like dormant
            # in a fresh cohort, does not grow inside dormant+softcancel): the combinations that carry it shrink to fit
            for state in self.states:
                carrying = [combination for combination in self.combinations_with_state if state in combination.split("+")]
                allowed = max(sum(final_share[combination] for combination in carrying)
                              - sum(today.get(combination, 0.0) for combination in carrying), 0.0)
                added = sum(gaps[combination] for combination in carrying)
                if added > allowed:
                    for combination in carrying:
                        gaps[combination] *= allowed / added
            gap_total = sum(gaps.values())
            if gap_total <= 0:
                continue
            # the share of the neutral units that moves: what the states lack over what is neutral, at most all of it;
            # computed once and split by weight (a sum of fractions of 1 can round above 1)
            moved_share = min(gap_total / neutral_today, 1.0)
            moves[(group, month)] = {combination: moved_share * gap / gap_total for combination, gap in gaps.items() if gap > 0}
        return moves, finals, levels

    def _empty_table(self) -> pd.DataFrame:
        columns = (["_grupo", self.configuration.period_col, "combinacion", UNITS_TODAY_COLUMN, UNITS_AT_DUE_COLUMN,
                    "unidades_grupo", "nivel"] + [f"cuota_final_{state}" for state in self.states] + self.group_columns + self.states)
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STAGE: learn the box, show what it knows
# ═══════════════════════════════════════════════════════════════════════════════════

def learn_the_maturation(fine_table: pd.DataFrame, configuration: Config) -> StateMaturation:
    """The stage 4.1 of the pipeline: the box learned at the last closed month, checked, and what it knows on screen."""
    configuration.log_step_start(STEP_4_1_LABEL, STEP_4_1_NAME, STEP_4_1_PURPOSE, STEP_4_1_ACTIONS, STEP_4_1_OUTPUT)
    check_log = []
    last_closed = configuration.calendar_boundaries()["current"] - 1

    # [1] the box
    box = StateMaturation(fine_table, last_closed, configuration)
    configuration.log_action(STEP_4_1_LABEL, 1, f"the box at {last_closed}: {box.describe()}")

    # [2] the curve of every dated state and its exam
    configuration.log_action(STEP_4_1_LABEL, 2, "the maturation curve of every dated state and its exam")
    box.curve_exams = show_maturation_curves(fine_table, configuration)
    curve_exams = sorted({int(h) for summary in box.curve_exams.values() for h in summary["h"]})

    # the curves that failed their exam are not used: the state takes its historical share (rejected_maturation_dims)
    for state in configuration.rejected_maturation_dims:
        box.curves.pop(state, None)
    show_the_curve_explained(fine_table, box, configuration)

    # [3] the maturation of every state by region
    box.maturation_by_region = show_maturation_by_region(fine_table, box, configuration)

    # [4] what every combination renews
    configuration.log_action(STEP_4_1_LABEL, 4, "what every combination of states renews in the closed months the box learns from")
    show_what_every_combination_renews(box, configuration)

    # [5] the checks
    configuration.log_action(STEP_4_1_LABEL, 5, "checking the box")
    last_learned = box.history_months[-1] if box.history_months else None
    configuration.log_check(STEP_4_1_LABEL, check_log, "the box learns only from closed months up to the last closed one",
                            last_learned is None or last_learned <= last_closed,
                            failure_detail=f"it learned from {last_learned}, after {last_closed}",
                            context=box.describe())
    rejected = list(configuration.rejected_maturation_dims)
    configuration.log_check(STEP_4_1_LABEL, check_log, "every dated state's curve beats the states seen then in its exam "
                            f"(and misses by ≤ {configuration.maturation_curve_max_error_pp:g} pp)",
                            not rejected, blocking=False,
                            failure_detail=f"{rejected}: the curve is not used, they take their historical share",
                            context=f"{len(configuration.maturation_dims_in_use)} curves used: "
                                    f"{', '.join(configuration.maturation_dims_in_use) or '—'}")
    without_cohorts = [state for state, curve in box.curves.items()
                       if curve.portfolio_table()["visible_del_final"].isna().all()]
    configuration.log_check(STEP_4_1_LABEL, check_log, "every dated state has a curve learned from cohorts that carry it",
                            not without_cohorts, blocking=False,
                            failure_detail=f"no cohort with the state for {without_cohorts}: their maturation is 0",
                            context=f"{len(box.curves)} dated states" + (f" · exam of the curve at h = "
                                                                       f"{sorted(curve_exams)}" if curve_exams else ""))

    # [6] the count of the checks; stop if anything failed
    configuration.log_action(STEP_4_1_LABEL, 6, "counting the checks")
    configuration.log_check_summary(STEP_4_1_LABEL, STEP_4_1_NAME, check_log)
    return box


def show_maturation_curves(fine_table: pd.DataFrame, configuration: Config) -> dict:
    """The maturation curve of every dated state and its exam on the exam months, on screen; returns the exam of the
    curve of every dated state ({state: one row per h}). Per state: the curve of the portfolio (visible_del_final: the share of the final state already set k months
    before due; p_tomar_estado: the probability that a unit not in the state then ends in the state) and the exam of the curve (the
    share of the state predicted with what was seen h months before against the final one; the states seen then alone,
    for comparison), with a VALORACIÓN line."""
    states = [state for state in configuration.maturation_dims if state in fine_table.columns]
    if not states:
        configuration.logger.doc(f"[{STEP_4_1_LABEL}] no state carries maturation columns (maturation_dims empty): every state "
                                 f"takes its historical share")
        return {}
    doc = configuration.logger.doc
    period = configuration.period_col
    last_closed = configuration.calendar_boundaries()["current"] - 1
    history = curve_history(fine_table, configuration, up_to=last_closed)
    history_months = sorted(history[period].unique())
    exam_months = sorted(fine_table.loc[fine_table[CALENDAR_ROLE_COLUMN] == ROLE_TEST, period].unique())
    horizons = sorted(set(configuration.backtest_horizons) | set(MATURATION_EXAM_HORIZONS))
    exams_of_states = {}
    doc(f"[{STEP_4_1_LABEL}] ══════════ MADURACIÓN DE LOS ESTADOS DEL CLIENTE: LA CURVA Y SU EXAMEN ══════════")
    for state in states:
        curve = MaturationCurve(history, state, configuration)
        doc(f"[{STEP_4_1_LABEL}] {state}: the curve of the portfolio over {history_months[0]}..{history_months[-1]} "
            f"(visible_del_final: share of the final {state} already set k months before due; p_tomar_estado: probability "
            f"that a unit not in the state k months before due ends in the state):")
        configuration.show_table(curve.portfolio_table().round(4))
        exam = examine_the_curve(fine_table, state, exam_months, horizons, configuration)
        if exam.empty:
            continue
        summary = exam.groupby("h")[["cuota_final", "cuota_vista_entonces", "cuota_curva", "error_curva_pp", "error_vista_pp"]].mean()
        summary["error_curva_abs_pp"] = exam.assign(_abs=exam["error_curva_pp"].abs()).groupby("h")["_abs"].mean()
        summary["error_vista_abs_pp"] = exam.assign(_abs=exam["error_vista_pp"].abs()).groupby("h")["_abs"].mean()
        doc(f"[{STEP_4_1_LABEL}] {state}: the exam of the curve in {exam_months[0]}..{exam_months[-1]} (its share of the units due "
            f"predicted with what was seen h months before, against the final one; cuota_vista_entonces: the states seen "
            f"then, with no curve), mean over the exam months:")
        configuration.show_table(summary.round(4).reset_index())
        exams_of_states[state] = summary.reset_index()
        worst = summary["error_curva_abs_pp"].max()
        seen_gap = summary["error_vista_pp"].abs().max()
        doc(f"[{STEP_4_1_LABEL}] VALORACIÓN {state}: with the curve the share of {state} is missed by at most {worst:.2f} pp "
            f"(mean per horizon); with the states seen then and no curve, by up to {seen_gap:.2f} pp"
            + (": the curve is what the forecast needs" if worst < seen_gap else ": the curve does not beat the states seen; look at it"))
        # the decision: the curve is used only if its exam beats the states seen then and stays under the threshold
        curve_error = float(exam["error_curva_pp"].abs().mean())
        seen_error = float(exam["error_vista_pp"].abs().mean())
        passes = curve_error < seen_error and curve_error <= configuration.maturation_curve_max_error_pp
        summary.attrs.update(curve_error=curve_error, seen_error=seen_error, passes=passes)
        exams_of_states[state].attrs.update(curve_error=curve_error, seen_error=seen_error, passes=passes)
        if not passes:
            configuration.rejected_maturation_dims.append(state)
            doc(f"[{STEP_4_1_LABEL}] DECISIÓN {state}: su curva NO se usa (error medio {curve_error:.2f} pp, frente a "
                f"{seen_error:.2f} pp con los estados vistos y un máximo de {configuration.maturation_curve_max_error_pp:g} pp): "
                f"toma su cuota histórica, como un estado sin fechas, en el forecast y en el examen honesto")
    doc(f"[{STEP_4_1_LABEL}] ══════════ FIN DE LA MADURACIÓN DE LOS ESTADOS DEL CLIENTE ══════════")
    return exams_of_states


# ─── action 3: the maturation of every state by region ──────────────────────────
def maturation_by_region(fine_table: pd.DataFrame, box: StateMaturation, configuration: Config) -> pd.DataFrame:
    """The maturation of every negative state by region (the first region level): one row per state, view, region
    and k (months before due), with

        visible_del_final   the share of the final state already visible k months before due
        cuota_final         the final share of the state in the region (closed months the box learns from)
        unidades            the units behind the value (the units in the final state for the past view, the units due
                            k months ahead for the view of today)

    view "pasado" (dated states only): Σ V_k / Σ M over the cohorts the box learns from, V_k the units already in the
    state k months or more before due, M the units in the state at the end.
    view "hoy" (every state): of the units due k months after the last closed month, the share in the state today,
    over the final share of the region. k = 1 … maturation_months (k = 0 is a closed month).
    The region "cartera" is the whole portfolio. Empty when the box learned nothing."""
    columns = ["estado", "vista", "region", "meses_antes", "visible_del_final", "cuota_final", "unidades"]
    if box.identity or box.history.empty:
        return pd.DataFrame(columns=columns)
    period, units_column = configuration.period_col, configuration.pipeline_units_col
    region_column = region_columns_of(configuration)[0]
    months = int(configuration.maturation_months)
    history = box.history

    # the regions shown: the largest by units due in the closed months the box learns from
    units_by_region = history.groupby(region_column)[units_column].sum().sort_values(ascending=False)
    regions = list(units_by_region.index[:configuration.maturation_regions_shown])

    # the pipeline as seen today: the future rows of the extract, k months after the last closed month
    future = fine_table[(fine_table[CALENDAR_ROLE_COLUMN] == ROLE_PROJECTION) & (fine_table[units_column] > 0)]
    months_ahead = future[period].map(lambda month: (month - box.as_of).n)
    future = future[(months_ahead >= 1) & (months_ahead <= months)]
    months_ahead = months_ahead[future.index]

    records = []
    for state in box.states:
        history_in_state = history[units_column].to_numpy(dtype=float) * is_in_state(history, state)
        future_in_state = future[units_column].to_numpy(dtype=float) * is_in_state(future, state)
        for region in regions + [PORTFOLIO_KEY]:
            in_region = (np.ones(len(history), dtype=bool) if region == PORTFOLIO_KEY
                         else (history[region_column] == region).to_numpy())
            units_due = float(history.loc[in_region, units_column].sum())
            units_final = float(history_in_state[in_region].sum())
            final_share = units_final / units_due if units_due > 0 else np.nan

            # the past curve of a dated state: Σ V_k / Σ M
            if state in configuration.maturation_dims:
                for k in range(months + 1):
                    visible = float(visible_units(history[in_region], state, k, configuration).sum())
                    records.append((state, VIEW_PAST, region, k, visible / units_final if units_final > 0 else np.nan,
                                    final_share, units_final))

            # the view of today: the share in the state k months ahead, over the final share
            future_in_region = (np.ones(len(future), dtype=bool) if region == PORTFOLIO_KEY
                                else (future[region_column] == region).to_numpy())
            for k in range(1, months + 1):
                at_k = future_in_region & (months_ahead == k).to_numpy()
                due_at_k = float(future.loc[at_k, units_column].sum())
                share_today = float(future_in_state[at_k].sum()) / due_at_k if due_at_k > 0 else np.nan
                records.append((state, VIEW_TODAY, region, k,
                                share_today / final_share if final_share and final_share > 0 else np.nan,
                                final_share, due_at_k))
    return pd.DataFrame(records, columns=columns)


def show_maturation_by_region(fine_table: pd.DataFrame, box: StateMaturation, configuration: Config) -> pd.DataFrame:
    """Action 3 on screen: for every state, the table k × region of the share of the final state already visible (the
    past curve for a dated state, the view of today for an undated one) with the final share of every region, and one
    chart with a panel per state and a line per region. Returns the long table (maturation_by_region)."""
    doc = configuration.logger.doc
    region_column = region_columns_of(configuration)[0]
    table = maturation_by_region(fine_table, box, configuration)
    configuration.log_action(STEP_4_1_LABEL, 3, f"the maturation of every state by {region_column}: "
                                                f"{table['region'].nunique() - 1 if len(table) else 0} regions and the portfolio")
    if table.empty:
        doc(f"[{STEP_4_1_LABEL}] no maturation by region: the box learned nothing (identity box or no closed month)")
        return table
    for state in box.states:
        view = VIEW_PAST if state in configuration.maturation_dims else VIEW_TODAY
        of_state = table[(table["estado"] == state) & (table["vista"] == view)]
        if of_state.empty:
            continue
        if view == VIEW_PAST:
            doc(f"[{STEP_4_1_LABEL}] {state} · the PAST curve by {region_column}: of the final {state}, the % already "
                f"visible k months before due (average of the cohorts {box.history_months[0]}..{box.history_months[-1]}, "
                f"weighted by units); last row: the final share of the state in the region")
        else:
            doc(f"[{STEP_4_1_LABEL}] {state} · no dates in the extract, so no past curve: the pipeline TODAY by {region_column}, "
                f"the % of the final {state} already visible in the units due k months ahead (share in the state today "
                f"over the final share of the region in {box.history_months[0]}..{box.history_months[-1]}); "
                f"last row: that final share")
        shown = of_state.pivot(index="meses_antes", columns="region", values="visible_del_final").sort_index(ascending=False)
        shown = (100 * shown).round(1)
        final_row = (100 * of_state.groupby("region")["cuota_final"].first()).round(1).rename("cuota_final_%")
        shown = pd.concat([shown, final_row.to_frame().T])
        shown.index = [f"k = {k}" if not isinstance(k, str) else k for k in shown.index]
        ordered = [region for region in of_state["region"].drop_duplicates() if region != PORTFOLIO_KEY] + [PORTFOLIO_KEY]
        shown = shown[ordered]
        shown.columns.name = None
        configuration.show_table(shown.reset_index().rename(columns={"index": "meses_antes"}))
    path = chart_path(configuration, MATURATION_BY_REGION_CHART)
    drawn = draw_without_stopping(lambda: draw_maturation_by_region(table, box, configuration, path), STEP_4_1_LABEL, configuration)
    if drawn:
        configuration.show_figure(STEP_4_1_LABEL, f"the maturation of every state by {region_column}", drawn)
    return table


def when_the_state_is_set(box, state: str, configuration: Config) -> tuple:
    """(when, tope) of one dated state, from the cohorts the box learns from: per k (months before due, 12 = "12 or
    more"), f = Σ units that took the state k months before / Σ units in the state at the end (the distribution of
    WHEN the final state is set), F = Σ f(k ≥ h) (the share already visible h months before), p = the probability
    that a unit not in the state h months before ends in it (the curve the forecast uses); tope = Σ units in the state
    / Σ units due."""
    months = int(configuration.maturation_months)
    units_column = configuration.pipeline_units_col
    history = box.history
    in_state = is_in_state(history, state)
    columns = configuration.maturation_columns_of(state)
    entered = history.loc[in_state, columns].fillna(0.0).sum().to_numpy(dtype=float)
    units_in_state = float(history.loc[in_state, units_column].sum())
    units_due = float(history[units_column].sum())
    distribution = entered / units_in_state if units_in_state > 0 else np.zeros(months + 1)
    visible = np.cumsum(distribution[::-1])[::-1]                    # F(h) = Σ f(k ≥ h)
    curve = MaturationCurve(history, state, configuration).portfolio_table().set_index("meses_antes")["p_tomar_estado"]
    when = pd.DataFrame({"meses_antes": np.arange(months + 1), "f": distribution, "F": visible})
    when["p"] = when["meses_antes"].map(curve).fillna(0.0).to_numpy(dtype=float)
    tope = units_in_state / units_due if units_due > 0 else np.nan
    return when, tope


def curve_table(box, configuration: Config) -> pd.DataFrame:
    """The table sff_curva_maduracion (written by step 7.2): one row per dated state and k (months before due, 12 =
    "12 or more"): parte_que_entra = f(k), parte_visible = F(k), p_entrar = p(k), the tope of the state, whether its
    curve is used, and the exam of the curve at that horizon (mean |error| of the final share, with the curve and
    without it: the states seen then), blank where the exam did not look. The SQL of the report reads it."""
    rows = []
    for state, exam in getattr(box, "curve_exams", {}).items():
        when, tope = when_the_state_is_set(box, state, configuration)
        exam_by_h = exam.set_index("h") if exam is not None and len(exam) else pd.DataFrame()
        for _, point in when.iterrows():
            k = int(point["meses_antes"])
            examined = k in exam_by_h.index
            rows.append({"estado": state, "meses_antes": k, "parte_que_entra": float(point["f"]),
                         "parte_visible": float(point["F"]), "p_entrar": float(point["p"]), "tope": float(tope),
                         "se_usa": int(state in box.curves),
                         "error_curva_pp": float(exam_by_h.loc[k, "error_curva_abs_pp"]) if examined else np.nan,
                         "error_sin_curva_pp": float(exam_by_h.loc[k, "error_vista_abs_pp"]) if examined else np.nan,
                         "cohortes": f"{box.history_months[0]}..{box.history_months[-1]}" if box.history_months else ""})
    return pd.DataFrame(rows, columns=["estado", "meses_antes", "parte_que_entra", "parte_visible", "p_entrar", "tope",
                                       "se_usa", "error_curva_pp", "error_sin_curva_pp", "cohortes"])


def maturation_explained(box, fine_table: pd.DataFrame, state: str, configuration: Config) -> dict:
    """The numbers of the explanation of the curve of one dated state, from the cohorts the box learns from
    (box.history: the last maturation_history_months closed months) and the pipeline of today:

      when      per k (months before due, 12 = "12 or more"): f(k) = Σ units that took the state k months before / Σ
                units in the state at the end — the distribution of WHEN the final state is set; F(h) = Σ f(k ≥ h), the
                share of the final state already visible h months before due; the tope π = Σ units in the state / Σ due
      today     per future month (h = months after the last closed one): the share in the state seen today, and what
                the curve adds, (1 − seen) × p(h), up to the expected final share
      exam      per h: the mean |error| of the final share in the exam of step 4.1, with the curve and with the states
                seen then (no curve)"""
    months = int(configuration.maturation_months)
    units_column, period = configuration.pipeline_units_col, configuration.period_col
    when, tope = when_the_state_is_set(box, state, configuration)
    curve = when.set_index("meses_antes")["p"]
    future = fine_table[(fine_table[CALENDAR_ROLE_COLUMN] == "proyeccion") & (fine_table[units_column] > 0)]
    months_ahead = future[period].map(lambda month: (month - box.as_of).n)
    future = future[(months_ahead >= 1) & (months_ahead <= months)]
    months_ahead = months_ahead[future.index].to_numpy(dtype=int)
    seen_units = visible_units(future, state, months_ahead, configuration)
    by_month = (pd.DataFrame({"mes": future[period].to_numpy(), "h": months_ahead,
                              "vencen": future[units_column].to_numpy(dtype=float), "vistas": seen_units})
                .groupby(["mes", "h"])[["vencen", "vistas"]].sum().reset_index())
    by_month["visto_hoy"] = by_month["vistas"] / by_month["vencen"]
    by_month["p"] = by_month["h"].map(curve).fillna(0.0)
    by_month["anade_la_curva"] = (1 - by_month["visto_hoy"]) * by_month["p"]

    exam = getattr(box, "curve_exams", {}).get(state)
    return dict(when=when, tope=tope, today=by_month, exam=exam,
                used=state in getattr(box, "curves", {}), history_months=box.history_months)


def show_the_curve_explained(fine_table: pd.DataFrame, box: StateMaturation, configuration: Config) -> None:
    """The chart that explains the curve, always shown (it is how the forecast is told): per dated state, the
    distribution of WHEN the final state is set, what the curve adds to the pipeline of today, and its error by horizon."""
    if box.identity or box.history.empty or not box.curve_exams:
        return
    explained = {state: maturation_explained(box, fine_table, state, configuration)
                 for state in configuration.maturation_dims if state in box.curve_exams}
    path = chart_path(configuration, MATURATION_EXPLAINED_CHART)
    drawn = draw_without_stopping(lambda: draw_maturation_explained(explained, configuration.maturation_curve_max_error_pp,
                                                                    path),
                                  STEP_4_1_LABEL, configuration)
    if drawn:
        configuration.show_figure(STEP_4_1_LABEL, "la curva de maduración explicada (cuándo, hoy, cuánto acierta)", drawn,
                                  always=True)


def show_what_every_combination_renews(box: StateMaturation, configuration: Config) -> None:
    """On screen: what every combination of states renews in the closed months the box learns from, in units and USD:
    the cost of a state against neutral."""
    if box.identity or box.history.empty:
        return
    doc = configuration.logger.doc
    units_column, usd_column = configuration.pipeline_units_col, configuration.pipeline_usd_col
    history = box.history.assign(_combinacion=state_combination(box.history, box.states))
    combinations = (history.groupby("_combinacion")
                    .agg(vencen_unidades=(units_column, "sum"), renovadas=(configuration.renewed_units_col, "sum"),
                         vencen_usd=(usd_column, "sum"), renovado_usd=(configuration.renewed_usd_col, "sum")))
    combinations["%_unidades"] = (100 * combinations["vencen_unidades"] / combinations["vencen_unidades"].sum()).round(1)
    combinations["tasa_unidades"] = (100 * combinations["renovadas"] / combinations["vencen_unidades"]).round(1)
    combinations["tasa_usd"] = (100 * combinations["renovado_usd"] / combinations["vencen_usd"]).round(1)
    combinations["usd_por_unidad"] = (combinations["vencen_usd"] / combinations["vencen_unidades"]).round(2)
    combinations = combinations.sort_values("vencen_unidades", ascending=False)
    doc(f"[{STEP_4_1_LABEL}] ══════════ LO QUE RENUEVA CADA COMBINACIÓN DE ESTADOS DEL CLIENTE ({', '.join(box.states)}) ══════════")
    doc(f"[{STEP_4_1_LABEL}] the last {len(box.history_months)} closed months (final states): the cost of every state against "
        f"neutral; usd_por_unidad = what falls due per unit (a state that leaves the neutral side leaves at the neutral "
        f"row's price):")
    configuration.show_table(combinations[["%_unidades", "tasa_unidades", "tasa_usd", "usd_por_unidad"]].reset_index()
                             .rename(columns={"_combinacion": "combinacion"}))
    doc(f"[{STEP_4_1_LABEL}] ══════════ FIN DE LO QUE RENUEVA CADA COMBINACIÓN DE ESTADOS DEL CLIENTE ══════════")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE MATURATION OF THE FORECAST, MONTH BY MONTH (step 7.2 calls these)
# ═══════════════════════════════════════════════════════════════════════════════════

def maturation_by_month(future: pd.DataFrame, box_tables: pd.DataFrame, box: StateMaturation, configuration: Config):
    """The table by month of the maturation over the whole future (extract and extension): every state today, expected
    when it falls due and at the final share the box aimed at; the forecast with today's states and the final one; the
    two vessels (the neutral side loses, the side with a state gains). None with the identity box.

    INPUT:   the future rows of step 7.2 (valued: esperado_*_hoy with today's states, esperado_* with the maturation,
             vencen_unidades_al_vencer) and the tables of the box (one per batch of rows, concatenated)."""
    if box.identity:
        return None
    period = configuration.period_col
    units_column = configuration.pipeline_units_col
    future = future.copy()
    future["_combinacion"] = state_combination(future, box.states)
    grouped = future.groupby(period)
    table = pd.DataFrame({"meses_vista": grouped["h"].min(), "vencen_unidades": grouped[units_column].sum(),
                          "vencen_usd": grouped[configuration.pipeline_usd_col].sum()})
    pairs = box_tables.drop_duplicates(["_grupo", period]) if len(box_tables) else box_tables
    for state in box.states:
        in_state = future[state].isin(ACTIVE_FLAG_VALUES)
        table[f"{state}_hoy"] = future[units_column].where(in_state, 0.0).groupby(future[period]).sum() / table["vencen_unidades"]
        table[f"{state}_esperado"] = (future[DUE_UNITS_AT_DUE_COLUMN].where(in_state, 0.0).groupby(future[period]).sum()
                                     / table["vencen_unidades"])
        final_units = ((pairs[f"cuota_final_{state}"] * pairs["unidades_grupo"]).groupby(pairs[period]).sum()
                       if len(pairs) else pd.Series(dtype=float))
        table[f"{state}_final"] = final_units.reindex(table.index).fillna(table[f"{state}_hoy"] * table["vencen_unidades"]) / table["vencen_unidades"]
    neutral = future["_combinacion"] == NEUTRAL
    table["neutro_hoy"] = future[units_column].where(neutral, 0.0).groupby(future[period]).sum() / table["vencen_unidades"]
    table["neutro_esperado"] = future[DUE_UNITS_AT_DUE_COLUMN].where(neutral, 0.0).groupby(future[period]).sum() / table["vencen_unidades"]
    table["unidades_migran"] = future["maduracion_unidades"].clip(lower=0.0).groupby(future[period]).sum()
    table["tasa_base"] = grouped["esperado_unidades_hoy"].sum() / table["vencen_unidades"]
    table["tasa_final"] = grouped["esperado_unidades"].sum() / table["vencen_unidades"]
    table["renovado_usd_base"] = grouped["esperado_usd_hoy"].sum()
    table["ajuste_usd"] = grouped["maduracion_delta_usd"].sum()
    table["renovado_usd_final"] = grouped["esperado_usd"].sum()
    table["ajuste_pct"] = table["ajuste_usd"] / table["renovado_usd_base"].where(table["renovado_usd_base"] > 0)
    table["sin_estado_usd_base"] = future["esperado_usd_hoy"].where(neutral, 0.0).groupby(future[period]).sum()
    table["sin_estado_usd_final"] = future["esperado_usd"].where(neutral, 0.0).groupby(future[period]).sum()
    table["con_estado_usd_base"] = future["esperado_usd_hoy"].where(~neutral, 0.0).groupby(future[period]).sum()
    table["con_estado_usd_final"] = future["esperado_usd"].where(~neutral, 0.0).groupby(future[period]).sum()
    return table.reset_index()


def show_maturation_by_month(table: pd.DataFrame, box: StateMaturation, configuration: Config,
                             step_label: str = STEP_4_1_LABEL) -> None:
    """The maturation month by month and by year on screen, with the two vessels."""
    doc = configuration.logger.doc
    share_columns = [f"{state}_{moment}" for state in box.states for moment in ("hoy", "esperado")] + ["neutro_hoy", "neutro_esperado"]
    shown = table[[configuration.period_col, "meses_vista"] + share_columns
                  + ["unidades_migran", "tasa_base", "tasa_final", "renovado_usd_base", "ajuste_usd", "renovado_usd_final",
                     "ajuste_pct"]].copy()
    for share_column in share_columns + ["tasa_base", "tasa_final", "ajuste_pct"]:
        shown[share_column] = (100 * shown[share_column]).round(1)
    for money_column in ["unidades_migran", "renovado_usd_base", "ajuste_usd", "renovado_usd_final"]:
        shown[money_column] = shown[money_column].round(0)
    doc(f"[{step_label}] ══════════ REPARTO DE ESTADOS DEL CLIENTE ({', '.join(box.states)}): EL FORECAST FINAL ══════════")
    doc(f"[{step_label}] {box.describe()}")
    doc(f"[{step_label}] mes a mes, en %: cada estado HOY (foto del extracto; las filas proyectadas y simuladas nacen sin "
        f"estado) y ESPERADO al vencer (la predicción de la caja); tasa_base = con los estados de hoy, tasa_final = el forecast. "
        f"Las unidades que migran salen de la fila neutra a su precio medio y renuevan a la tasa y el uplift de la serie a la "
        f"que llegan")
    configuration.show_table(shown)
    year_of = table[configuration.period_col].map(lambda month: month.year)
    by_year = table.groupby(year_of)[["renovado_usd_base", "ajuste_usd", "renovado_usd_final", "sin_estado_usd_base",
                                      "sin_estado_usd_final", "con_estado_usd_base", "con_estado_usd_final"]].sum()
    doc(f"[{step_label}] por año: el forecast con los estados de hoy, el ajuste y el forecast final, y los dos vasos "
        f"(sin estado pierde, con estado gana, la pipeline es la misma):")
    configuration.show_table(by_year.round(0).reset_index().rename(columns={configuration.period_col: "ano"}))
    for year, row in by_year.iterrows():
        if row["renovado_usd_base"] <= 0:
            continue
        doc(f"[{step_label}] VALORACIÓN {year}: con los estados de hoy ${row['renovado_usd_base']:,.0f}; con la maduración "
            f"de los estados del cliente (el forecast final) ${row['renovado_usd_final']:,.0f} "
            f"({row['ajuste_usd'] / row['renovado_usd_base']:+.1%}): sin estado ${row['sin_estado_usd_base'] - row['sin_estado_usd_final']:,.0f} "
            f"menos, con estado ${row['con_estado_usd_final'] - row['con_estado_usd_base']:,.0f} más")
    doc(f"[{step_label}] ══════════ FIN DEL REPARTO DE ESTADOS DEL CLIENTE ══════════")


def states_above_final(maturation: pd.DataFrame, box: StateMaturation, configuration: Config) -> pd.DataFrame:
    """The detector of states above their final share: one row per future month and state.

    OUTPUT:  period, estado, hoy and final (% of the units due), exceso_pp (hoy − final, in points; positive = the
             extract shows MORE in the state than there will be at due date), unidades_exceso, and impacto_usd_aprox: what the
             forecast lacks if the excess renews as neutral = units in excess × USD due per unit of the month ×
             (USD rate of the neutral − USD rate of the state), with the rates of the portfolio in the closed months
             the box learns from.
    RULES:   the extract records the LAST state of a state, so a state only grows: today's share can be below the final
             one (the states still to come) but never above. Above it, the state of the extract is early (a fresh cohort
             set to dormant before its first use) and the box cannot lower it. The excess is a NET difference over the
             whole month: the noise of small groups cancels out, a systematic early state does not."""
    period = configuration.period_col
    history = box.history
    usd_column = configuration.pipeline_usd_col
    renewed_usd = history[configuration.renewed_usd_col].fillna(0.0) if len(history) else pd.Series(dtype=float)

    def usd_rate(selected):
        due = history.loc[selected, usd_column].sum()
        return float(renewed_usd[selected].sum() / due) if due > 0 else np.nan

    neutral_rate = usd_rate(state_combination(history, box.states) == NEUTRAL) if len(history) else np.nan
    rows = []
    for state in box.states:
        state_rate = usd_rate(history[state].isin(ACTIVE_FLAG_VALUES)) if len(history) else np.nan
        rate_gap = neutral_rate - state_rate if not (np.isnan(neutral_rate) or np.isnan(state_rate)) else 0.0
        for _, month_row in maturation.iterrows():
            excess = month_row[f"{state}_hoy"] - month_row[f"{state}_final"]
            excess_units = excess * month_row["vencen_unidades"]
            usd_per_unit = month_row["vencen_usd"] / month_row["vencen_unidades"] if month_row["vencen_unidades"] > 0 else 0.0
            rows.append({period: month_row[period], "estado": state, "hoy": 100 * month_row[f"{state}_hoy"],
                         "final": 100 * month_row[f"{state}_final"], "exceso_pp": 100 * excess,
                         "unidades_exceso": excess_units,
                         "impacto_usd_aprox": max(excess_units, 0.0) * usd_per_unit * rate_gap})
    return pd.DataFrame(rows, columns=[period, "estado", "hoy", "final", "exceso_pp", "unidades_exceso", "impacto_usd_aprox"])


def show_states_above_final(detector: pd.DataFrame, configuration: Config, step_label: str = STEP_4_1_LABEL) -> None:
    """The detector on screen: the months and states past the threshold and what they cost the forecast, or OK."""
    doc = configuration.logger.doc
    period = configuration.period_col
    threshold = configuration.mark_excess_warning_pp
    doc(f"[{step_label}] ══════════ ESTADOS DEL CLIENTE POR ENCIMA DE SU PROPORCIÓN FINAL (DETECTOR) ══════════")
    doc(f"[{step_label}] un estado solo crece (el extracto guarda su último estado): HOY no puede estar por encima de su "
        f"proporción FINAL (la que la caja espera para su grupo y mes). Si lo está, el extracto pone el estado antes de tiempo (p. ej. "
        f"una cohorte recién vendida puesta en dormant antes de su primer uso) y la caja no puede bajarla. Umbral: "
        f"{threshold:.1f} pp; en %:")
    flagged = detector[detector["exceso_pp"] >= threshold].copy()
    if flagged.empty:
        doc(f"[{step_label}] VALORACIÓN: ningún estado supera su proporción final en {threshold:.1f} pp en ningún mes: OK")
    else:
        for column_name in ("hoy", "final", "exceso_pp"):
            flagged[column_name] = flagged[column_name].round(1)
        for column_name in ("unidades_exceso", "impacto_usd_aprox"):
            flagged[column_name] = flagged[column_name].round(0)
        configuration.show_table(flagged)
        by_year = flagged.groupby([flagged[period].map(lambda month: month.year), "estado"])["impacto_usd_aprox"].sum()
        for (year, state), impact in by_year.items():
            doc(f"[{step_label}] VALORACIÓN {year} · {state}: por encima de su proporción final en "
                f"{int(((flagged['estado'] == state) & (flagged[period].map(lambda month: month.year) == year)).sum())} meses; "
                f"si esas unidades renuevan como neutras, el forecast se queda corto en ≈ ${impact:,.0f}")
    doc(f"[{step_label}] ══════════ FIN DE ESTADOS DEL CLIENTE POR ENCIMA DE SU PROPORCIÓN FINAL ══════════")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE HONEST EXAM OF THE TOTAL (step 7.1 calls it): the box at the origin of every exam month
# ═══════════════════════════════════════════════════════════════════════════════════

def honest_total_exam(framework_rows: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The honest exam of the total: every exam month T and horizon h predicted with the states as they were h months
    before, and their maturation by THE SAME BOX the forecast uses, learned at the origin T − h.

    INPUT:   the framework predictions of step 7.1 (one row per forecast series × exam month × h: fs_id, pred_rate,
             due_units, real_units), the fine table (the maturation columns, the dims of every series).
    OUTPUT:  one row per exam month × h: the real renewals, and the predicted ones three ways
               techo_estados_finales  with the FINAL states, as step 7.1 has always predicted: the ceiling, what predicting
                                     the states perfectly would give
               estados_hoy            with the states seen at the origin and nothing more (no maturation)
               honesto               with the states seen at the origin and the box's maturation
    RULES:   only the dated states (maturation_dims) can be seen as they were: their units in the state fewer than h months
             before due go back to the series without the state (rewind). The other states stay FINAL (for them the exam
             is still the ceiling). Every series renews at the rate the framework predicted for it; a series with no
             prediction (a twin that had no units) at the rate of its combination in the closed months the box knows."""
    period = configuration.period_col
    dated = [state for state in configuration.maturation_dims_in_use
             if state in fine_table.columns and state in negative_states(configuration)]
    dims = [column for column in configuration.rate_series_columns if column in fine_table.columns]
    attributes = fine_table.drop_duplicates(SERIES_ID_COLUMN).set_index(SERIES_ID_COLUMN)[dims]
    timing_columns = [column for state in dated for column in configuration.maturation_columns_of(state)]
    timing = (fine_table.groupby([SERIES_ID_COLUMN, period])[timing_columns].sum(min_count=1).reset_index()
              if timing_columns else None)
    results = []
    for (target_month, horizon), rows in framework_rows.groupby([period, "h"]):
        origin = target_month - horizon
        box = StateMaturation(fine_table, origin, configuration, states=dated)
        frame = rows[[SERIES_ID_COLUMN, "pred_rate", "due_units", "real_units"]].join(attributes, on=SERIES_ID_COLUMN)
        if timing is not None:
            frame = frame.merge(timing[timing[period] == target_month].drop(columns=[period]), on=SERIES_ID_COLUMN, how="left")
        frame[period] = target_month
        ceiling = float((frame["pred_rate"] * frame["due_units"]).sum())
        rate_of_series = frame.groupby(SERIES_ID_COLUMN)["pred_rate"].mean().to_dict()
        seen = rewind_to_the_origin(frame, dated, int(horizon), box, configuration)

        def rate_of(series_rows):
            rates = join_columns(series_rows, configuration.rate_series_columns).map(rate_of_series)
            missing = rates.isna()
            if missing.any():
                groups = box.group_key(series_rows[missing])
                cells = join_columns(series_rows[missing], configuration.business_mandatory_dims)
                combinations = state_combination(series_rows[missing], box.states)
                rates[missing] = [box.combination_rate(group, cell, combination)[0]
                                  for group, cell, combination in zip(groups, cells, combinations)]
            return rates.fillna(0.0).to_numpy(dtype=float)

        today = float((seen[UNITS_TODAY_COLUMN].to_numpy(dtype=float) * rate_of(seen)).sum())
        table = box.mature(seen)
        honest = float((table[UNITS_AT_DUE_COLUMN].to_numpy(dtype=float) * rate_of(table)).sum()) if len(table) else today
        results.append({"mes": target_month, "h": horizon, "renovadas_reales": float(frame["real_units"].sum()),
                        "techo_estados_finales_pred": ceiling, "estados_hoy_pred": today, "honesto_pred": honest})
    exam = pd.DataFrame(results)
    for method in ("techo_estados_finales", "estados_hoy", "honesto"):
        exam[f"{method}_error_total"] = exam[f"{method}_pred"] / exam["renovadas_reales"] - 1
    return exam


def rewind_to_the_origin(frame: pd.DataFrame, dated: list, horizon: int, box: StateMaturation,
                         configuration: Config) -> pd.DataFrame:
    """The rows of an exam month as they were seen `horizon` months before due: for every dated state, the units in the state
    fewer than `horizon` months before due were not in the state yet, so they go back to the twin row without the state (its
    maturation columns at 0); the row with a state keeps only what was visible (its columns before `horizon` at 0, so the
    box sees exactly V_h). Returns the units input of the box (unidades_hoy, con_fechas)."""
    rows = frame.copy()
    rows[UNITS_TODAY_COLUMN] = rows["due_units"].astype(float)
    rows[WITH_DATES_COLUMN] = True
    for state in dated:
        columns = configuration.maturation_columns_of(state)
        for column_name in columns:
            if column_name not in rows.columns:
                rows[column_name] = 0.0
        rows[columns] = rows[columns].fillna(0.0)
        in_state = is_in_state(rows, state)
        units = rows[UNITS_TODAY_COLUMN].to_numpy(dtype=float)
        visible = np.minimum(visible_units(rows, state, horizon, configuration), units)
        hidden = np.where(in_state, units - visible, 0.0)
        # the rows with a state keep what was visible; the columns dated after the origin did not exist then
        kept = rows.copy()
        kept[UNITS_TODAY_COLUMN] = np.where(in_state, visible, units)
        not_yet = columns[:min(horizon, len(columns) - 1)]
        kept.loc[in_state, not_yet] = 0.0
        # the hidden units go back to the twin without the state
        moved = rows[in_state & (hidden > 0)].copy()
        moved[UNITS_TODAY_COLUMN] = hidden[in_state & (hidden > 0)]
        moved[state] = box.flags[state][1] if state in box.flags else 0
        moved[columns] = 0.0
        rows = pd.concat([kept, moved], ignore_index=True)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 4 · THE BLOCK: run the step, keep the box
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B4"
BLOCK_NAME = "CUSTOMER STATES"
BLOCK_MISSION = "how many units of every future month will carry each customer state when they fall due"
BLOCK_READS = ("fine_table",)
BLOCK_GIVES = ("state_maturation",)


class CustomerStatesBlock:
    """BLOCK 4 · CUSTOMER STATES (see the header of this module): its box is an attribute, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.state_maturation = None
        self.seconds = {}

    def run(self, fine_table: pd.DataFrame) -> "CustomerStatesBlock":
        """The step of the block."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 4, BLOCK_NAME, BLOCK_MISSION)

        started = time.time()
        self.state_maturation = learn_the_maturation(fine_table, configuration)
        self.seconds["4.1 customer_states"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
