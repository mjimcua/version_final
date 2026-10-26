"""
maturation.py — WHEN the states are set: the maturation curve of every state whose extract dates it.

A state (softcancel, dormant…) is not there from the start: a licence that falls due in twelve months may be
in the state next week, or in the month it falls due. The extract keeps the LAST state of a state, so the closed months
show every state already set, and the future months only the part set so far. With the maturation columns
(<state>_month_k: the units of the row in the state k months before due, k = 0 … maturation_months, the last one "that
many months or more") the history says how a cohort fills its states, and that is predicted like the rate and the
uplift: the third prediction of the forecast.

THE CURVE (per state, learned from closed cohorts, the states final)
  For every group and due month: U the units due, M the units in the state at the end, V_h the units already in the state h
  months before due (Σ of the columns k ≥ h). Then, summed over the cohorts:
      visible share   V_h / M                    how much of the final state is already seen h months before
      p(h)            Σ (M − V_h) / Σ (U − V_h)  the probability that a unit NOT in the state h months before due has
                                                 the state when it falls due
  A unit not in the state today, h months before its due month, ends in the state with probability p(h). The expected final
  units of the state are   V + (U − V) × p(h)   (V: in the state today; the states only grow).
  The groups are those of step 7.2 (the forecast series without its states), its mandatory cell and the
  portfolio: the first one with at least maturation_min_units units not in the state h months before.

WHERE IT IS USED
  Only inside the box of step 4.1 (block_4_customer_states.py): the final share of a dated state comes from its curve
  (today's share + the expected migration) on the rows with dates. The forecast (step 7.2) and the honest exam
  (step 7.1) reach it through the box, never directly.

h, THE MONTHS BEFORE DUE THAT ARE ALREADY KNOWN
  In the forecast: h = due month − last closed month (the current month: h = 1). In the exam: h = the horizon
  (a technique predicting T at horizon h sees the months up to T − h). A unit is visible iff it was in the state at
  least h months before due: Σ of the columns k ≥ h.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from config import ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, Config, TRUTH_ROLES, join_columns

# ─── named constants ─────────────────────────────────────────────────────────────
GROUP_LEVEL, CELL_LEVEL, PORTFOLIO_LEVEL = "_grupo", "_celda", "_cartera"
LEVELS = (GROUP_LEVEL, CELL_LEVEL, PORTFOLIO_LEVEL)
PORTFOLIO_KEY = "cartera"


def group_dims(configuration: Config) -> list:
    """The group of the curve: every dim of the forecast series but the timevarying ones (as step 7.2)."""
    return [column_name for column_name in configuration.rate_series_columns
            if column_name not in configuration.structural_timevarying_dims]


def add_level_keys(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The keys of the three levels of the curve on every row."""
    keyed = rows.copy()
    keyed[GROUP_LEVEL] = join_columns(keyed, group_dims(configuration))
    keyed[CELL_LEVEL] = join_columns(keyed, configuration.business_mandatory_dims)
    keyed[PORTFOLIO_LEVEL] = PORTFOLIO_KEY
    return keyed


def is_in_state(rows: pd.DataFrame, state: str) -> np.ndarray:
    return rows[state].isin(ACTIVE_FLAG_VALUES).to_numpy() if state in rows.columns else np.zeros(len(rows), dtype=bool)


def visible_units(rows: pd.DataFrame, state: str, months_before, configuration: Config) -> np.ndarray:
    """The units of every row already in the state `months_before` months before due (one value, or one per row):
    Σ of its columns k ≥ months_before; 0 on a row without the state. Past the last column, only the last one
    (it means that many months or more)."""
    columns = configuration.maturation_columns_of(state)
    # rows without the columns (the projected and simulated rows of the extended horizon) have nothing dated: 0
    timing = rows.reindex(columns=columns).fillna(0.0).to_numpy(dtype=float)
    seen_from_the_end = np.cumsum(timing[:, ::-1], axis=1)[:, ::-1]          # [:, k] = Σ of the columns ≥ k
    first = np.clip(np.broadcast_to(np.asarray(months_before, dtype=int), (len(rows),)), 0, len(columns) - 1)
    visible = seen_from_the_end[np.arange(len(rows)), first] if len(rows) else np.zeros(0)
    return np.where(is_in_state(rows, state), visible, 0.0)


class MaturationCurve:
    """The curve of one state, learned from closed cohorts (their states final): for every level key and h the
    sums Σ (M − V_h) and Σ (U − V_h); p(h) is their ratio at the first level with enough support."""

    def __init__(self, closed_rows: pd.DataFrame, state: str, configuration: Config):
        self.state = state
        self.configuration = configuration
        self.months = int(configuration.maturation_months)
        self.minimum = configuration.maturation_min_units
        units_column = configuration.pipeline_units_col
        rows = closed_rows                # read, not copied: the keys of the levels are built apart (memory, 26-oct-2026)
        level_keys = {GROUP_LEVEL: join_columns(rows, group_dims(configuration)).to_numpy(),
                      CELL_LEVEL: join_columns(rows, configuration.business_mandatory_dims).to_numpy(),
                      PORTFOLIO_LEVEL: np.full(len(rows), PORTFOLIO_KEY, dtype=object)}
        in_state = is_in_state(rows, state)
        columns = configuration.maturation_columns_of(state)
        timing = rows[columns].fillna(0.0).to_numpy(dtype=float) * in_state[:, None]
        seen_from_the_end = np.cumsum(timing[:, ::-1], axis=1)[:, ::-1]        # V_h of every row, h = 0 … months
        units = rows[units_column].to_numpy(dtype=float)
        units_in_state = units * in_state
        # per row: still to be in the state after h (M − V_h) and not in the state h months before (U − V_h)
        to_come = units_in_state[:, None] - seen_from_the_end
        not_in_state_yet = units[:, None] - seen_from_the_end
        self.sums = {}
        for level in LEVELS:
            keys = level_keys[level]
            frame = pd.DataFrame(np.hstack([to_come, not_in_state_yet, units_in_state[:, None], seen_from_the_end]))
            frame["_key"] = keys
            grouped = frame.groupby("_key").sum()
            width = self.months + 1
            self.sums[level] = {key: (values[:width], values[width:2 * width], values[2 * width], values[2 * width + 1:])
                                for key, values in zip(grouped.index, grouped.to_numpy())}

    def probability(self, keys: dict, months_before: int) -> tuple:
        """(p, level): the probability that a unit not in the state `months_before` months before due ends in the state, at
        the first level with at least maturation_min_units units not in the state then (the portfolio otherwise)."""
        h = int(np.clip(months_before, 0, self.months))
        for level in LEVELS:
            sums = self.sums[level].get(keys.get(level, PORTFOLIO_KEY))
            if sums is None:
                continue
            to_come, not_in_state_yet = sums[0][h], sums[1][h]
            if not_in_state_yet >= self.minimum or (level == PORTFOLIO_LEVEL and not_in_state_yet > 0):
                return float(max(to_come, 0.0) / not_in_state_yet), level
        return 0.0, PORTFOLIO_LEVEL

    def portfolio_table(self) -> pd.DataFrame:
        """The curve of the whole portfolio: h, the share of the final state already seen, p(h)."""
        to_come, not_in_state_yet, units_in_state, seen = self.sums[PORTFOLIO_LEVEL][PORTFOLIO_KEY]
        return pd.DataFrame({"meses_antes": np.arange(self.months + 1),
                             "visible_del_final": seen / units_in_state if units_in_state > 0 else np.nan,
                             "p_tomar_estado": np.where(not_in_state_yet > 0, to_come / np.maximum(not_in_state_yet, 1e-9), np.nan)})


def closed_rows_of(fine_table: pd.DataFrame, configuration: Config, up_to=None) -> pd.DataFrame:
    """The closed rows (truth months) with something due, optionally only up to a month (inclusive)."""
    rows = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES) & (fine_table[configuration.pipeline_units_col] > 0)]
    if up_to is not None:
        rows = rows[rows[configuration.period_col] <= up_to]
    return rows


def curve_history(fine_table: pd.DataFrame, configuration: Config, up_to) -> pd.DataFrame:
    """The cohorts a curve is learned from at an origin: the last maturation_history_months closed months up to it
    (inclusive), the same window the forecast learns its curve with (step 7.2), so the exam examines that curve."""
    rows = closed_rows_of(fine_table, configuration, up_to=up_to)
    months = sorted(rows[configuration.period_col].unique())[-configuration.maturation_history_months:]
    return rows[rows[configuration.period_col].isin(months)]


def examine_the_curve(fine_table: pd.DataFrame, state: str, exam_months: list, horizons: list,
                      configuration: Config) -> pd.DataFrame:
    """The exam of the curve on the closed months: for every exam month T and horizon h, the units of the state
    predicted with what was seen h months before (V_h + (U − V_h) × p(h), the curve learned with the cohorts up to
    T − h) against the units in the state at the end. Also what was seen then (V_h, the states of "today")."""
    period, units_column = configuration.period_col, configuration.pipeline_units_col
    closed = closed_rows_of(fine_table, configuration)
    rows = []
    for target_month in exam_months:
        month_rows = add_level_keys(closed[closed[period] == target_month], configuration)
        if month_rows.empty:
            continue
        for horizon in horizons:
            curve = MaturationCurve(curve_history(fine_table, configuration, target_month - horizon), state, configuration)
            grouped = month_rows.assign(_con_estado=month_rows[units_column] * is_in_state(month_rows, state),
                                        _visibles=visible_units(month_rows, state, horizon, configuration))
            grouped = grouped.groupby([GROUP_LEVEL, CELL_LEVEL])[[units_column, "_con_estado", "_visibles"]].sum().reset_index()
            probability = np.array([curve.probability({GROUP_LEVEL: group, CELL_LEVEL: cell}, horizon)[0]
                                    for group, cell in zip(grouped[GROUP_LEVEL], grouped[CELL_LEVEL])])
            predicted = grouped["_visibles"] + (grouped[units_column] - grouped["_visibles"]) * probability
            total = grouped[units_column].sum()
            rows.append({"mes": target_month, "h": horizon, "vencen": total,
                         "cuota_final": grouped["_con_estado"].sum() / total,
                         "cuota_vista_entonces": grouped["_visibles"].sum() / total,
                         "cuota_curva": predicted.sum() / total})
    exam = pd.DataFrame(rows)
    if len(exam):
        exam["error_curva_pp"] = 100 * (exam["cuota_curva"] - exam["cuota_final"])
        exam["error_vista_pp"] = 100 * (exam["cuota_vista_entonces"] - exam["cuota_final"])
    return exam
