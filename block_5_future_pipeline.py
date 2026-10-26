"""
block_5_future_pipeline.py — BLOCK 5 · FUTURE PIPELINE: the rows of pipeline the extract cannot have yet.

MISSION: fill the pipeline that will fall due 12 months after the simulation window (current month → December), which
the extract cannot have because it is created by what happens in the window: the renewals of the 1-year licences due
in it, and the acquisitions sold in it.

MECHANISM: a projected renewal falls due the same month next year with the units and USD the forecast expects it to
renew; an acquisition is simulated as what its group sold the same month a year before × the level of its acquisition
value × its value per unit. The acquisitions are examined on the closed months (as if each one were the current
month). The rows leave the block WITHOUT a prediction: block 7 values them with the boxes of blocks 2, 3 and 4.

THE STEPS:
  5.1  future pipeline   the projected renewals and the simulated acquisitions, compared with the year before and
                         examined

READS (block 1): fine_table.
GIVES: future_pipeline (the box, the comparison with the year before, the exam of the acquisitions).

THE INTERFACE:
  block = FuturePipelineBlock(configuration).run(fine_table)
  block.future_pipeline    block.results()    block.seconds
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time

import numpy as np
import pandas as pd

from config import (ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, Config, PIPELINE_ORIGIN_COLUMN, PIPELINE_PROJECTED,
                    PIPELINE_SIMULATED, ROLE_PROJECTION, SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN,
                    discount_bucket_labels, is_one_year, join_columns)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 5.1 · FUTURE PIPELINE  (until 25-oct-2026: step_18_future_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE FUTURE PIPELINE: the rows that will fall due and that the extract cannot have
# yet, in one box with its exam.
#
# THE QUESTION: what falls due 12 months after the SIMULATION WINDOW (from the current month, included, to
# simulation_end, December)? The extract cannot have it: it is created by events of the window. Two kinds:
#   · proyectada  every 1-year licence (term_column = one_year_term_value) due in the window renews as the
#                 forecast expects; its renewal falls due the same month next year with the same dims, units =
#                 expected renewed units, value = expected renewed USD, discount renewal_reentry_discount (0); an
#                 acquisition that renews is no longer one (acquisition_column → renewed_acquisition_value); the
#                 dims of dims_after_renewal take their declared value; the customer states are neutral (nothing
#                 is known yet of the renewal's dormant, softcancel…)
#   · simulada    the acquisitions of the window, per value of acquisition_column in acquisition_values apart: per
#                 group of dims, units = what the group sold the same month a year before (the shape) × the
#                 AGGREGATE level of its acquisition value (last acquisition_level_window_months sold over the same
#                 months a year before; per acquisition_level_dims when declared, never per fine group), value =
#                 units × value per unit (last acquisition_auv_window_months), customer states off, a 1-year term,
#                 discount acquisition_discount (0.4). An acquisition sold in month m is read in the extract as
#                 the 1-year acquisition rows due in m + 12.
#
# THE INTERFACE (what the forecast calls):
#   box = FuturePipeline(fine_table, configuration)
#   box.new_rows(extract_future)   the rows of the window (proyectada + simulada), with their ids, NOT predicted:
#                                  extract_future are the extract's future rows already predicted (the proyectada
#                                  rows are their expected renewals); the forecast predicts and matures the new rows
#   box.simulated                  the simulated acquisitions (they do not depend on the forecast: built here)
#   box.levels                     the level of every acquisition value (and key)
# A window that is empty, or an extract without acquisition_column, gives no rows of that kind.
#
# THE EXAM OF THE ACQUISITIONS (new: before this stage the simulation had none). For every origin o of the last
# acquisition_exam_origins closed months: the box is built as if o were the current month (the level and the
# value per unit with the months before o only) and it simulates the sales of o, o + 1 … up to the last closed
# month; they are compared with what was really sold. The baseline is 'the same as a year before' (level 1). The
# error is |simulated − real| / real of the total of a month, per acquisition value and months after the origin.
#
# Actions (logged as they are done):
#   1. the acquisitions read as sales and their level
#   2. the simulated acquisitions of the window, against the same months a year before
#   3. the exam of the acquisitions on the closed months
#   4. check the box                                                    checks 1-3
#   5. count the checks; stop if any failed
#   6. show the levels, the comparison and the exam
#
# Checks (logged as they are made, numbered, at the level of their status):
#   1. every value a projected renewal takes after renewing exists in the history (WARN: its retention series
#      can be found)
#   2. the simulated acquisitions are between acquisition_ratio_low and acquisition_ratio_high times what was sold
#      in the same months a year before (WARN; the section ADQUISICIÓN SIMULADA FRENTE AL AÑO ANTERIOR)
#   3. the exam simulates only with months before its origin (no month at or after the origin is read)
#
# Output: the box (FuturePipeline) for the forecast · the exam of the acquisitions (in memory, for the report).

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_5_1_LABEL = "5.1"
STEP_5_1_NAME = "FUTURE PIPELINE"
STEP_5_1_PURPOSE = ("build the rows that will fall due 12 months after the simulation window and the extract cannot have "
                "yet (renewals of 1-year licences and acquisitions), in one box the forecast calls; and examine the "
                "simulated acquisitions on the closed months")
STEP_5_1_ACTIONS = ["the acquisitions read as sales and their level",
                "the simulated acquisitions of the window, against the same months a year before",
                "the exam of the acquisitions on the closed months",
                "check the box (checks 1-3)",
                "count the checks; stop if any failed",
                "show the levels, the comparison and the exam"]
STEP_5_1_OUTPUT = "the box (FuturePipeline) for the forecast · the exam of the acquisitions (in memory)"

# ─── named constants ─────────────────────────────────────────────────────────────
LEVEL_OWN, LEVEL_PORTFOLIO, LEVEL_NO_REFERENCE = "propio", "cartera", "sin_referencia"
PORTFOLIO_KEY = "cartera"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE BOX
# ═══════════════════════════════════════════════════════════════════════════════════

class FuturePipeline:
    """The future pipeline of the window (see the header): built once from the fine table, asked by the forecast."""

    def __init__(self, fine_table: pd.DataFrame, configuration: Config):
        self.configuration = configuration
        self.window = configuration.simulation_window
        self.last_due_month = fine_table[configuration.period_col].max()
        self.has_term = bool(configuration.term_column) and configuration.term_column in fine_table.columns
        # the 'off' value of every customer state, as the data writes it
        self.inactive = {column: inactive_value(fine_table[column]) for column in configuration.structural_timevarying_dims
                         if column in fine_table.columns}
        self.carried = carried_columns(configuration, fine_table.columns)
        self.events, self.dims_of_group, self.group_columns = acquisition_events(fine_table, configuration)
        self.levels = (acquisition_levels(self.events, self.dims_of_group, configuration, last_due_month=self.last_due_month)
                       if len(self.events) else pd.DataFrame())
        self.simulated = (simulate_acquisitions(self, self.window) if self.window else pd.DataFrame())

    def describe(self) -> str:
        window = f"{self.window[0]}..{self.window[-1]}" if self.window else "—"
        return (f"window {window} · {len(self.simulated):,} simulated acquisition rows"
                + (f" (${self.simulated[self.configuration.pipeline_usd_col].sum():,.0f} due)" if len(self.simulated) else ""))

    def renewals(self, extract_future: pd.DataFrame) -> pd.DataFrame:
        """proyectada: the expected renewals of the 1-year licences due in the window, due 12 months later."""
        configuration = self.configuration
        period = configuration.period_col
        carried = [column for column in self.carried if column in extract_future.columns]
        renewing = extract_future[extract_future[period].isin(self.window) & is_one_year(extract_future, configuration)
                                  & (extract_future["esperado_unidades"] > 0)]
        projected = renewing[carried].copy()
        projected[period] = renewing[period].to_numpy() + configuration.one_year_term_months
        projected[configuration.pipeline_units_col] = renewing["esperado_unidades"].to_numpy()
        projected[configuration.pipeline_usd_col] = renewing["esperado_usd"].to_numpy()
        if configuration.discount_value_column:
            projected[configuration.discount_value_column] = configuration.renewal_reentry_discount
        # an acquisition that renews is no longer an acquisition: its next due date is a retention
        acquisition = configuration.acquisition_column
        if acquisition and acquisition in projected.columns and configuration.renewed_acquisition_value is not None:
            was_acquisition = projected[acquisition].isin(configuration.acquisition_values)
            projected[acquisition] = projected[acquisition].astype(object)
            projected.loc[was_acquisition, acquisition] = configuration.renewed_acquisition_value
        # the other dims a renewal changes (the previous operation is now a renewal), as the Config declares them
        for column_name, value_after_renewal in configuration.dims_after_renewal.items():
            if column_name in projected.columns:
                projected[column_name] = projected[column_name].astype(object)
                projected[column_name] = value_after_renewal
        # the customer states of a renewal are neutral: nothing is known yet of its dormant, softcancel…
        for column_name, value in self.inactive.items():
            if column_name in projected.columns:
                projected[column_name] = value
        projected[PIPELINE_ORIGIN_COLUMN] = PIPELINE_PROJECTED
        return projected

    def new_rows(self, extract_future: pd.DataFrame) -> pd.DataFrame:
        """The rows of the window (proyectada + simulada) with their ids, not predicted (the forecast predicts them)."""
        if not self.window:
            return pd.DataFrame()
        frames = [frame for frame in (self.renewals(extract_future), self.simulated) if len(frame)]
        if not frames:
            return pd.DataFrame()
        return with_ids(pd.concat(frames, ignore_index=True), self.configuration)


def carried_columns(configuration: Config, available) -> list:
    """The dims a row of the window carries: the series, the uplift dims, the term and the acquisition column."""
    carried = list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion
                                 + ([configuration.term_column] if configuration.term_column else [])
                                 + ([configuration.acquisition_column] if configuration.acquisition_column else [])))
    return [column for column in carried if column in available]


def is_acquisition(rows: pd.DataFrame, configuration: Config) -> pd.Series:
    """The acquisition rows (acquisition_column in acquisition_values); none when it is not declared."""
    if not configuration.acquisition_column or not configuration.acquisition_values:
        return pd.Series(False, index=rows.index)
    return rows[configuration.acquisition_column].isin(configuration.acquisition_values)


def inactive_value(values: pd.Series):
    """The value that means 'state off' in a timevarying column, as the data writes it (False, 0, "0")."""
    inactive = values[~values.isin(ACTIVE_FLAG_VALUES)].dropna()
    return inactive.mode().iloc[0] if len(inactive) else 0


def acquisition_events(fine_table: pd.DataFrame, configuration: Config) -> tuple:
    """The 1-year acquisitions read as SALES: an acquisition sold in month m is the 1-year acquisition rows due in
    m + 12. One row per group of dims, acquisition value and month of sale, for the months already sold (before the
    current month): units and value. Returns (events, the dims of every group, the group columns).

    The group keeps the grain of the pipeline: every mandatory dim, every extra_renovacion and every
    extra_revalorizacion, but not the customer states nor the term (an acquisition is simulated without states and
    as a 1-year licence)."""
    period, current = configuration.period_col, configuration.calendar_boundaries()["current"]
    timevarying = list(configuration.structural_timevarying_dims)
    carried = list(dict.fromkeys(configuration.rate_series_columns + configuration.extra_revalorizacion
                                 + ([configuration.acquisition_column] if configuration.acquisition_column else [])))
    group_columns = [column for column in carried if column in fine_table.columns and column not in timevarying
                     and column != configuration.term_column and column != configuration.acquisition_column]
    empty = (pd.DataFrame(columns=["_grupo", configuration.acquisition_column or "_valor", "_mes_evento", "unidades", "valor"]),
             pd.DataFrame(columns=group_columns), group_columns)
    if not configuration.acquisition_column or not configuration.acquisition_values:
        return empty
    acquired = fine_table[is_acquisition(fine_table, configuration) & is_one_year(fine_table, configuration)].copy()
    if acquired.empty:
        return empty
    acquired["_mes_evento"] = acquired[period] - configuration.one_year_term_months
    acquired = acquired[acquired["_mes_evento"] < current]
    acquired["_grupo"] = join_columns(acquired, group_columns)
    events = (acquired.groupby(["_grupo", configuration.acquisition_column, "_mes_evento"])
              [[configuration.pipeline_units_col, configuration.pipeline_usd_col]].sum().reset_index()
              .rename(columns={configuration.pipeline_units_col: "unidades", configuration.pipeline_usd_col: "valor"}))
    dims_of_group = acquired.drop_duplicates("_grupo").set_index("_grupo")[group_columns]
    return events, dims_of_group, group_columns


def acquisition_levels(events: pd.DataFrame, dims_of_group: pd.DataFrame, configuration: Config,
                       last_due_month=None, as_of=None) -> pd.DataFrame:
    """The growth of the acquisitions: the units sold in the last acquisition_level_window_months months before as_of
    (the current month by default) over the units sold in the same months a year before, per acquisition value and
    LEVEL KEY (acquisition_level_dims; none = the whole portfolio of that acquisition value).

    OUTPUT:  one row per acquisition value and level key: unidades_recientes, unidades_antes, nivel and origen
             ("propio": the key has acquisition_level_min_units units a year before; "cartera": fewer, it takes the
             level of its acquisition value; "sin_referencia": not even that, level 1).
    RULES:   the level is NEVER measured per fine group: a group of a few units a year before gives a ratio of small
             numbers that only goes up (a group that sold 2 and now 40 has level 20, and multiplies its whole month by
             20), and summed over thousands of groups it multiplied the Pure-New acquisitions of Kamelot by 6 to 24.
             The fine group gives the SHAPE (what it sold in the same month a year before); the level is aggregate.
             A recent month of sales is only read when the extract reaches its due date (month + 12, up to
             last_due_month): an extract that ends earlier has not seen those sales, and that is not a fall to zero
             (no month seen: level 1, "sin_referencia")."""
    as_of = as_of if as_of is not None else configuration.calendar_boundaries()["current"]
    acquisition = configuration.acquisition_column
    level_dims = [column for column in configuration.acquisition_level_dims if column in dims_of_group.columns]
    recent_months = list(pd.period_range(as_of - configuration.acquisition_level_window_months, as_of - 1, freq="M"))
    if last_due_month is not None:
        recent_months = [month for month in recent_months if month + configuration.one_year_term_months <= last_due_month]
    before_months = [month - configuration.one_year_term_months for month in recent_months]
    events = events.copy()
    events["_clave_nivel"] = (join_columns(dims_of_group.loc[events["_grupo"]].reset_index(drop=True), level_dims).to_numpy()
                              if level_dims else PORTFOLIO_KEY)
    rows = []
    for acquisition_value, block in events.groupby(acquisition):
        recent_total = block.loc[block["_mes_evento"].isin(recent_months), "unidades"].sum()
        before_total = block.loc[block["_mes_evento"].isin(before_months), "unidades"].sum()
        portfolio_level = recent_total / before_total if before_total > 0 and recent_months else np.nan
        keys = sorted(block["_clave_nivel"].unique()) if level_dims else [PORTFOLIO_KEY]
        for key in keys:
            part = block[block["_clave_nivel"] == key] if level_dims else block
            recent = part.loc[part["_mes_evento"].isin(recent_months), "unidades"].sum()
            before = part.loc[part["_mes_evento"].isin(before_months), "unidades"].sum()
            if before >= configuration.acquisition_level_min_units and before > 0 and recent_months:
                level, origin = recent / before, LEVEL_OWN
            elif np.isfinite(portfolio_level):
                level, origin = portfolio_level, LEVEL_PORTFOLIO
            else:
                level, origin = 1.0, LEVEL_NO_REFERENCE
            rows.append({acquisition: acquisition_value, "clave_nivel": key, "unidades_recientes": float(recent),
                         "unidades_antes": float(before), "nivel": float(level), "origen": origin})
    return pd.DataFrame(rows, columns=[acquisition, "clave_nivel", "unidades_recientes", "unidades_antes", "nivel", "origen"])


def simulate_acquisitions(box: FuturePipeline, months: list, as_of=None, levels: pd.DataFrame = None) -> pd.DataFrame:
    """The acquisitions sold in `months`, one line per group of dims and acquisition value, due 12 months later.

    RULES:   units of a group in month m = what the group sold in m − 12 (the shape: the same month a year before,
             Black Friday keeps its size) × the level of its acquisition value (acquisition_levels at as_of: aggregate,
             never per fine group). A group that sold nothing in m − 12 sells nothing in m: the total of the month is
             exactly what was sold a year before × the level. Value = units × the value per unit of the group over
             the last acquisition_auv_window_months months of sales before as_of (of its acquisition value when the
             group has none). Customer states off, a 1-year term, acquisition_discount.
             as_of = the current month (the forecast) or an origin of the exam: nothing at or after as_of is read
             except the shape (m − 12, always before as_of for a month of the window)."""
    configuration = box.configuration
    if not configuration.acquisition_column or not configuration.acquisition_values or box.events.empty:
        return pd.DataFrame()
    period = configuration.period_col
    as_of = as_of if as_of is not None else configuration.calendar_boundaries()["current"]
    acquisition = configuration.acquisition_column
    events, dims_of_group = box.events, box.dims_of_group
    levels = levels if levels is not None else box.levels
    level_dims = [column for column in configuration.acquisition_level_dims if column in dims_of_group.columns]
    key_of_group = (join_columns(dims_of_group, level_dims) if level_dims
                    else pd.Series(PORTFOLIO_KEY, index=dims_of_group.index))
    level_of = {(row[acquisition], row["clave_nivel"]): row["nivel"] for _, row in levels.iterrows()}
    auv_start = as_of - configuration.acquisition_auv_window_months

    rows = []
    for acquisition_value, block in events.groupby(acquisition):
        auv_window = block[(block["_mes_evento"] >= auv_start) & (block["_mes_evento"] < as_of)]
        auv = auv_window.groupby("_grupo")["valor"].sum() / auv_window.groupby("_grupo")["unidades"].sum().replace(0, np.nan)
        total_auv = auv_window["valor"].sum() / auv_window["unidades"].sum() if auv_window["unidades"].sum() > 0 else np.nan
        for month in months:
            same_month = block[block["_mes_evento"] == month - configuration.one_year_term_months]
            for group, units_a_year_before in zip(same_month["_grupo"], same_month["unidades"]):
                level = level_of.get((acquisition_value, key_of_group[group]), 1.0)
                units = units_a_year_before * level
                if not np.isfinite(units) or units <= 0:
                    continue
                unit_value = auv.get(group, np.nan)
                unit_value = unit_value if np.isfinite(unit_value) else total_auv
                row = dims_of_group.loc[group].to_dict()
                row.update({acquisition: acquisition_value, period: month + configuration.one_year_term_months,
                            configuration.pipeline_units_col: float(units), configuration.pipeline_usd_col: float(units * unit_value)})
                rows.append(row)
    simulated = pd.DataFrame(rows)
    if simulated.empty:
        return simulated
    for column_name, value in box.inactive.items():
        simulated[column_name] = value
    if box.has_term:
        simulated[configuration.term_column] = configuration.one_year_term_value
    if configuration.discount_value_column:
        simulated[configuration.discount_value_column] = configuration.acquisition_discount
    simulated[PIPELINE_ORIGIN_COLUMN] = PIPELINE_SIMULATED
    return simulated


def with_ids(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The ids of a row of the window: its series, its forecast unit, its discount bucket and its uplift cell."""
    rows = rows.copy()
    rows[SERIES_ID_COLUMN] = join_columns(rows, configuration.rate_series_columns)
    rows[UNIT_ID_COLUMN] = rows[SERIES_ID_COLUMN] + "|" + rows[configuration.period_col].astype(str)
    if configuration.discount_value_column:
        rows[configuration.discount_bucket_column] = discount_bucket_labels(rows[configuration.discount_value_column],
                                                                           configuration.discount_bucket_edges)
    rows[UPLIFT_CELL_ID_COLUMN] = join_columns(rows, configuration.uplift_cell_columns)
    rows[CALENDAR_ROLE_COLUMN] = ROLE_PROJECTION
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════
# THE COMPARISON AND THE EXAM OF THE ACQUISITIONS
# ═══════════════════════════════════════════════════════════════════════════════════

def acquisition_comparison(box: FuturePipeline) -> pd.DataFrame:
    """The simulated acquisitions of the window against the same months a year before, per acquisition value and
    month of sale: units and USD sold a year before, units and USD simulated, and their ratio. A ratio far from the
    level means the simulation is not the shape × the level."""
    configuration = box.configuration
    acquisition, period = configuration.acquisition_column, configuration.period_col
    if not acquisition or not box.window:
        return pd.DataFrame()
    rows = []
    simulated = box.simulated
    for acquisition_value in configuration.acquisition_values:
        for month in box.window:
            before = box.events[(box.events[acquisition] == acquisition_value)
                                & (box.events["_mes_evento"] == month - configuration.one_year_term_months)]
            now = (simulated[(simulated[acquisition] == acquisition_value)
                             & (simulated[period] == month + configuration.one_year_term_months)] if len(simulated) else pd.DataFrame())
            units_before, usd_before = float(before["unidades"].sum()), float(before["valor"].sum())
            units_now = float(now[configuration.pipeline_units_col].sum()) if len(now) else 0.0
            usd_now = float(now[configuration.pipeline_usd_col].sum()) if len(now) else 0.0
            rows.append({acquisition: acquisition_value, "mes_venta": month, "vence": month + configuration.one_year_term_months,
                         "unidades_ano_anterior": units_before, "usd_ano_anterior": usd_before,
                         "unidades_simuladas": units_now, "usd_simulado": usd_now,
                         "ratio_unidades": units_now / units_before if units_before > 0 else np.nan})
    return pd.DataFrame(rows)


def acquisition_exam(box: FuturePipeline) -> dict:
    """THE EXAM OF THE ACQUISITIONS (see the header): the box rebuilt at every origin o of the last
    acquisition_exam_origins closed months simulates the months o … the last closed one; against the real sales.

    OUTPUT:  dict(months: one row per acquisition value, origin and month of sale: h (months after the origin, 0 =
             the origin itself), real, simulated and a-year-before units and USD; summary: per acquisition value and h,
             the mean |error| of the simulation and of the baseline (a year before) over real, in units and USD, and
             the months examined; read_after_origin: how many months at or after an origin the levels read (must be 0)).
    EDGE CASES: no acquisition column or no sales → empty tables. A month whose sales the extract does not reach
             (its due date after the last month of the extract) is not examined."""
    configuration = box.configuration
    acquisition = configuration.acquisition_column
    empty = dict(months=pd.DataFrame(), summary=pd.DataFrame(), read_after_origin=0)
    if not acquisition or box.events.empty or configuration.acquisition_exam_origins <= 0:
        return empty
    current = configuration.calendar_boundaries()["current"]
    last_seen_sale = box.last_due_month - configuration.one_year_term_months
    last_examinable = min(current - 1, last_seen_sale)
    origins = [last_examinable - back for back in range(configuration.acquisition_exam_origins - 1, -1, -1)]
    rows, read_after_origin = [], 0
    for origin in origins:
        levels = acquisition_levels(box.events, box.dims_of_group, configuration, last_due_month=box.last_due_month,
                                    as_of=origin)
        recent = pd.period_range(origin - configuration.acquisition_level_window_months, origin - 1, freq="M")
        read_after_origin += int(sum(month >= origin for month in recent))
        months = list(pd.period_range(origin, last_examinable, freq="M"))
        simulated = simulate_acquisitions(box, months, as_of=origin, levels=levels)
        for acquisition_value in configuration.acquisition_values:
            events_of_value = box.events[box.events[acquisition] == acquisition_value]
            for month in months:
                real = events_of_value[events_of_value["_mes_evento"] == month]
                before = events_of_value[events_of_value["_mes_evento"] == month - configuration.one_year_term_months]
                now = (simulated[(simulated[acquisition] == acquisition_value)
                                 & (simulated[configuration.period_col] == month + configuration.one_year_term_months)]
                       if len(simulated) else pd.DataFrame())
                rows.append({acquisition: acquisition_value, "origen": origin, "mes_venta": month, "h": (month - origin).n,
                             "unidades_real": float(real["unidades"].sum()), "usd_real": float(real["valor"].sum()),
                             "unidades_simuladas": float(now[configuration.pipeline_units_col].sum()) if len(now) else 0.0,
                             "usd_simulado": float(now[configuration.pipeline_usd_col].sum()) if len(now) else 0.0,
                             "unidades_ano_anterior": float(before["unidades"].sum()),
                             "usd_ano_anterior": float(before["valor"].sum())})
    months_table = pd.DataFrame(rows)
    if months_table.empty:
        return empty
    examined = months_table[months_table["unidades_real"] > 0].copy()
    for measure, real in (("unidades", "unidades_real"), ("usd", "usd_real")):
        simulated_column = "unidades_simuladas" if measure == "unidades" else "usd_simulado"
        examined[f"error_{measure}"] = (examined[simulated_column] - examined[real]).abs() / examined[real]
        examined[f"error_base_{measure}"] = (examined[f"{measure}_ano_anterior"] - examined[real]).abs() / examined[real]
    summary = (examined.groupby([acquisition, "h"])
               .agg(meses=("mes_venta", "size"), error_unidades=("error_unidades", "mean"),
                    error_base_unidades=("error_base_unidades", "mean"), error_usd=("error_usd", "mean"),
                    error_base_usd=("error_base_usd", "mean"),
                    sesgo_unidades=("unidades_simuladas", "sum"), real_unidades=("unidades_real", "sum"))
               .reset_index())
    summary["sesgo_unidades"] = summary["sesgo_unidades"] / summary["real_unidades"] - 1.0
    summary = summary.drop(columns=["real_unidades"])
    return dict(months=months_table, summary=summary, read_after_origin=read_after_origin)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STAGE: build the box, examine it, check it
# ═══════════════════════════════════════════════════════════════════════════════════

def build_the_future_pipeline(fine_table: pd.DataFrame, configuration: Config) -> dict:
    """The stage 5.1 of the pipeline: the box of the future pipeline, its exam and its checks.
    Returns dict(box=FuturePipeline, comparison, exam)."""
    configuration.log_step_start(STEP_5_1_LABEL, STEP_5_1_NAME, STEP_5_1_PURPOSE, STEP_5_1_ACTIONS, STEP_5_1_OUTPUT)
    check_log = []

    # [1] the box: acquisitions read as sales and their level
    box = FuturePipeline(fine_table, configuration)
    configuration.log_action(STEP_5_1_LABEL, 1, f"{len(box.events):,} (group, acquisition value, month) sales · "
                                            f"{len(box.levels):,} levels" if len(box.events) else "no acquisitions declared or sold")

    # [2] the simulated acquisitions of the window
    comparison = acquisition_comparison(box)
    configuration.log_action(STEP_5_1_LABEL, 2, box.describe())

    # [3] the exam of the acquisitions
    exam = acquisition_exam(box)
    configuration.log_action(STEP_5_1_LABEL, 3, f"{exam['summary']['meses'].sum() if len(exam['summary']) else 0:,} "
                                            f"(acquisition value, origin, month) examined")

    # [4] the checks
    configuration.log_action(STEP_5_1_LABEL, 4, "checking the box")
    after_renewal = dict(configuration.dims_after_renewal)
    if configuration.acquisition_column and configuration.renewed_acquisition_value is not None:
        after_renewal[configuration.acquisition_column] = configuration.renewed_acquisition_value
    history_values = {column_name: set(fine_table[column_name].dropna().astype(str).unique())
                      for column_name in after_renewal if column_name in fine_table.columns}
    unseen = {column_name: value for column_name, value in after_renewal.items()
              if column_name in history_values and str(value) not in history_values[column_name]}
    configuration.log_check(STEP_5_1_LABEL, check_log,
                            "every value a projected renewal takes after renewing exists in the history "
                            "(its retention series can be found)",
                            not unseen,
                            failure_detail="; ".join(f"{column_name} = {value!r} is not in the data (values: "
                                                     f"{sorted(history_values[column_name])[:8]})"
                                                     for column_name, value in unseen.items()),
                            blocking=False, context=f"{after_renewal}" if after_renewal else "nothing declared")
    if len(comparison):
        by_value = comparison.groupby(configuration.acquisition_column)[["unidades_ano_anterior", "unidades_simuladas"]].sum()
        by_value["ratio"] = by_value["unidades_simuladas"] / by_value["unidades_ano_anterior"].where(by_value["unidades_ano_anterior"] > 0)
        outside = by_value[(by_value["ratio"] < configuration.acquisition_ratio_low) | (by_value["ratio"] > configuration.acquisition_ratio_high)]
        configuration.log_check(STEP_5_1_LABEL, check_log,
                                f"the simulated acquisitions are between {configuration.acquisition_ratio_low:g} and "
                                f"{configuration.acquisition_ratio_high:g} times what was sold in the same months a year before",
                                outside.empty,
                                failure_detail=" · ".join(f"{value}: {row['ratio']:.2f}×" for value, row in outside.iterrows())
                                               + " · see ADQUISICIÓN SIMULADA FRENTE AL AÑO ANTERIOR",
                                blocking=False,
                                context=" · ".join(f"{value} {row['ratio']:.2f}×" for value, row in by_value.iterrows()))
    configuration.log_check(STEP_5_1_LABEL, check_log,
                            "the exam simulates only with months before its origin",
                            exam["read_after_origin"] == 0,
                            failure_detail=f"{exam['read_after_origin']} months at or after an origin were read",
                            context=f"{configuration.acquisition_exam_origins} origins")

    # [5] the count of the checks; stop if anything failed
    configuration.log_action(STEP_5_1_LABEL, 5, "counting the checks")
    configuration.log_check_summary(STEP_5_1_LABEL, STEP_5_1_NAME, check_log)

    # [6] on screen
    configuration.log_action(STEP_5_1_LABEL, 6, "the levels, the comparison and the exam")
    show_acquisitions(box, comparison, configuration)
    show_acquisition_exam(exam, configuration)
    return dict(box=box, comparison=comparison, exam=exam)


def show_acquisitions(box: FuturePipeline, comparison: pd.DataFrame, configuration: Config) -> None:
    """The simulated acquisitions against the same months a year before, and the level they took, on screen."""
    if comparison is None or comparison.empty:
        return
    doc = configuration.logger.doc
    acquisition = configuration.acquisition_column
    doc(f"[{STEP_5_1_LABEL}] ══════════ ADQUISICIÓN SIMULADA FRENTE AL AÑO ANTERIOR ══════════")
    doc(f"[{STEP_5_1_LABEL}] una venta del mes m se simula como lo que su grupo vendió en m − 12 × el nivel de su valor de "
        f"adquisición ({', '.join(configuration.acquisition_level_dims) or 'toda la cartera'}): unidades vendidas en los "
        f"últimos {configuration.acquisition_level_window_months} meses / las de los mismos meses un año antes")
    shown_levels = box.levels.copy()
    shown_levels["nivel"] = shown_levels["nivel"].round(3)
    configuration.show_table(shown_levels)
    if (box.levels["origen"] == LEVEL_NO_REFERENCE).any():
        doc(f"[{STEP_5_1_LABEL}] AVISO: el extracto no llega al vencimiento de las ventas de los últimos "
            f"{configuration.acquisition_level_window_months} meses (o no hubo ventas un año antes): su nivel es 1, se simula "
            f"exactamente lo vendido un año antes")
    shown = comparison.copy()
    for column_name in ("unidades_ano_anterior", "usd_ano_anterior", "unidades_simuladas", "usd_simulado"):
        shown[column_name] = shown[column_name].round(0)
    shown["ratio_unidades"] = shown["ratio_unidades"].round(3)
    configuration.show_table(shown)
    for value, block in comparison.groupby(acquisition):
        before, now = block["usd_ano_anterior"].sum(), block["usd_simulado"].sum()
        doc(f"[{STEP_5_1_LABEL}] VALORACIÓN {value}: vendido un año antes ${before:,.0f} ({block['unidades_ano_anterior'].sum():,.0f} "
            f"unidades); simulado ${now:,.0f} ({block['unidades_simuladas'].sum():,.0f} unidades), "
            f"{block['unidades_simuladas'].sum() / max(block['unidades_ano_anterior'].sum(), 1e-9):.2f}× en unidades")
    doc(f"[{STEP_5_1_LABEL}] ══════════ FIN DE ADQUISICIÓN SIMULADA FRENTE AL AÑO ANTERIOR ══════════")


def show_acquisition_exam(exam: dict, configuration: Config) -> None:
    """The exam of the acquisitions, on screen: per acquisition value and months after the origin, the error of the
    simulation against the baseline 'the same as a year before'."""
    summary = exam["summary"]
    if summary is None or summary.empty:
        return
    doc = configuration.logger.doc
    acquisition = configuration.acquisition_column
    doc(f"[{STEP_5_1_LABEL}] ══════════ EXAMEN DE LA CAPTACIÓN ══════════")
    doc(f"[{STEP_5_1_LABEL}] en cada uno de los últimos {configuration.acquisition_exam_origins} meses cerrados (el origen) se "
        f"simula la captación como si fuese el mes en curso (nivel y valor por unidad solo con los meses anteriores) y se "
        f"compara con lo vendido; h = meses tras el origen. Baseline: lo mismo que un año antes (nivel 1)")
    shown = summary.copy()
    for column_name in ("error_unidades", "error_base_unidades", "error_usd", "error_base_usd", "sesgo_unidades"):
        shown[column_name] = (100 * shown[column_name]).round(1)
    configuration.show_table(shown.rename(columns={"error_unidades": "error_unidades_pct", "error_base_unidades": "base_unidades_pct",
                                                   "error_usd": "error_usd_pct", "error_base_usd": "base_usd_pct",
                                                   "sesgo_unidades": "sesgo_unidades_pct"}))
    for value, block in summary.groupby(acquisition):
        weights = block["meses"]
        error = float((block["error_usd"] * weights).sum() / weights.sum())
        baseline = float((block["error_base_usd"] * weights).sum() / weights.sum())
        verdict = ("igual que" if abs(error - baseline) < 5e-4 else "mejor que" if error < baseline else "PEOR que")
        doc(f"[{STEP_5_1_LABEL}] CAPTACIÓN {value}: error medio en USD {error:.1%} frente a {baseline:.1%} de la baseline "
            f"(un año antes): la simulación es {verdict} la baseline")
    doc(f"[{STEP_5_1_LABEL}] ══════════ FIN DE EXAMEN DE LA CAPTACIÓN ══════════")


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 5 · THE BLOCK: run the step, keep the box
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B5"
BLOCK_NAME = "FUTURE PIPELINE"
BLOCK_MISSION = ("the rows of pipeline that will fall due 12 months after the simulation window and the extract cannot "
                 "have yet: projected renewals and simulated acquisitions, examined")
BLOCK_READS = ("fine_table",)
BLOCK_GIVES = ("future_pipeline",)


class FuturePipelineBlock:
    """BLOCK 5 · FUTURE PIPELINE (see the header of this module): its box is an attribute, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.future_pipeline = None
        self.seconds = {}

    def run(self, fine_table: pd.DataFrame) -> "FuturePipelineBlock":
        """The step of the block."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 5, BLOCK_NAME, BLOCK_MISSION)

        started = time.time()
        self.future_pipeline = build_the_future_pipeline(fine_table, configuration)
        self.seconds["5.1 future_pipeline"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
