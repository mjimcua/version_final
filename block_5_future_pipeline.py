"""
block_5_future_pipeline.py — BLOCK 5 · ACQUISITION: the new customers of the year, everything about them in one step.

MISSION: the acquisitions the extract cannot have yet. An acquisition sold from the current month to December falls
due 12 months later (a 1-year licence): it is pipeline of next year that does not exist yet. The block reads the
acquisitions already sold, simulates the ones still to sell, completes the year up to the acquisition team's target
when there is one, examines the simulation and draws it.

MECHANISM: an acquisition is simulated as what its group sold the same month a year before × the level of its
acquisition value × its value per unit. With acquisition_targets_file, every region is completed up to its target of
the year in every term (sold + simulated = target, the simulated split by terms in the proportion of the year's
sales); only the 1-year part falls due inside the horizon and makes rows. The acquisitions are examined on the closed
months (as if each one were the current month). The rows leave the block WITHOUT a prediction: block 7 values them
with the boxes of blocks 2, 3 and 4.

WHAT IS NOT HERE: the renewal of the real pipeline still to fall due this year (current month → December). It is
forecast in step 7.2 like every future row of the extract, and its expected renewal of the 1-year licences, due again
12 months later, is built there too (the rows proyectada): it needs that forecast.

THE STEPS:
  5.1  acquisition   sales read, level, simulation, the acquisition team's targets, exam and charts

READS (block 1): fine_table.
GIVES: future_pipeline (the box of the acquisitions, the comparison with the year before, the exam).

THE INTERFACE:
  block = FuturePipelineBlock(configuration).run(fine_table)
  block.future_pipeline    block.results()    block.seconds
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time

import numpy as np
import pandas as pd

from config import (ACTIVE_FLAG_VALUES, CALENDAR_ROLE_COLUMN, Config, PIPELINE_ORIGIN_COLUMN,
                    PIPELINE_SIMULATED, ROLE_PROJECTION, SERIES_ID_COLUMN, UNIT_ID_COLUMN, UPLIFT_CELL_ID_COLUMN,
                    discount_bucket_labels, is_one_year, join_columns)


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 5.1 · ACQUISITION  (until 25-oct-2026: step_18_future_pipeline.py; until 27-oct-2026 it also built the
#            projected renewals, now in step 7.2)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE ACQUISITIONS OF THE YEAR: read, simulated, completed up to the target, examined and drawn, in one box.
#
# THE QUESTION: what will be sold to new customers from the current month to December (the SIMULATION WINDOW,
# simulation_end), and so fall due 12 months later? Per value of acquisition_column in acquisition_values apart:
# per group of dims, units = what the group sold the same month a year before (the shape) × the AGGREGATE level of its
# acquisition value (last acquisition_level_window_months sold over the same months a year before; per
# acquisition_level_dims when declared, never per fine group), value = units × value per unit (last
# acquisition_auv_window_months), customer states off, a 1-year term, discount acquisition_discount (0.4). An
# acquisition sold in month m is read in the extract as the 1-year acquisition rows due in m + 12.
#
# THE TARGET (acquisition_targets_file, acquisition_target_column): the acquisition team's forecast of the year per
# region, in every term. What is missing (target − sold this year in every term) is split by terms in the proportion
# of the sales of the year; the 1-year simulation of the region is scaled by ONE factor up to its 1-year part (its mix
# of products and months does not change); the 2- and 3-year parts are counted, but fall due after the horizon.
#
# THE INTERFACE (what the forecast calls):
#   box = FuturePipeline(fine_table, configuration)
#   box.acquisition_rows()   the simulated acquisitions of the window, with their ids, NOT predicted (step 7.2 predicts
#                            and matures them, next to the projected renewals it builds itself)
#   box.simulated            the simulated acquisitions        box.levels    the level of every acquisition value
#   box.targets              the table of the target per region (empty without acquisition_targets_file)
# A window that is empty, or an extract without acquisition_column, gives no rows.
#
# THE EXAM OF THE ACQUISITIONS. For every origin o of the last acquisition_exam_origins closed months: the box is built
# as if o were the current month (the level and the value per unit with the months before o only) and it simulates the
# sales of o, o + 1 … up to the last closed month; they are compared with what was really sold. The baseline is 'the
# same as a year before' (level 1). The error is |simulated − real| / real of the total of a month, per acquisition
# value and months after the origin. It examines the method, not the target (a target cannot be examined on the past).
#
# Actions (logged as they are done):
#   1. the acquisitions read as sales and their level
#   2. the simulated acquisitions of the window, against the same months a year before (and up to the target)
#   3. the exam of the acquisitions on the closed months
#   4. check the box                                                    checks 1-2 (3 with targets)
#   5. count the checks; stop if any failed
#   6. show the levels, the comparison, the target and the exam
#   7. draw the acquisitions month by month: in total, and by term (always shown)
#
# Checks (logged as they are made, numbered, at the level of their status):
#   1. the simulated acquisitions are between acquisition_ratio_low and acquisition_ratio_high times what was sold
#      in the same months a year before (WARN; the section ADQUISICIÓN SIMULADA FRENTE AL AÑO ANTERIOR)
#   2. with targets: every region ends the year within acquisition_target_tolerance of its target (WARN)
#   3. the exam simulates only with months before its origin (no month at or after the origin is read)
#
# Output: the box (FuturePipeline) for the forecast · the exam of the acquisitions (in memory, for the report).

# ─── the step ────────────────────────────────────────────────────────────────────
STEP_5_1_LABEL = "5.1"
STEP_5_1_NAME = "ACQUISITION"
STEP_5_1_PURPOSE = ("the acquisitions sold from the current month to December (due 12 months later, pipeline of next year "
                "that the extract cannot have yet): read what was sold, simulate the rest, complete the year up to the "
                "acquisition team's target, examine the simulation on the closed months and draw it")
STEP_5_1_ACTIONS = ["the acquisitions read as sales and their level",
                "the simulated acquisitions of the window, against the same months a year before (and up to the target)",
                "the exam of the acquisitions on the closed months",
                "check the box (checks 1-2, 3 with targets)",
                "count the checks; stop if any failed",
                "show the levels, the comparison, the target and the exam",
                "draw the acquisitions month by month, in total and by term (always shown)"]
STEP_5_1_OUTPUT = "the box of the acquisitions (FuturePipeline) for the forecast · the exam of the acquisitions (in memory)"

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
        # the acquisition team's targets (acquisition_targets_file): the year of every region completed up to its target
        self.targets = (complete_to_the_targets(self, fine_table)
                        if configuration.acquisition_targets_file and len(self.events) else pd.DataFrame())

    def describe(self) -> str:
        window = f"{self.window[0]}..{self.window[-1]}" if self.window else "—"
        return (f"window {window} · {len(self.simulated):,} simulated acquisition rows"
                + (f" (${self.simulated[self.configuration.pipeline_usd_col].sum():,.0f} due)" if len(self.simulated) else ""))

    def acquisition_rows(self) -> pd.DataFrame:
        """The simulated acquisitions of the window with their ids, not predicted (step 7.2 predicts and matures them)."""
        if not self.window or self.simulated.empty:
            return pd.DataFrame()
        return with_ids(self.simulated.copy(), self.configuration)


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


# ─── the acquisition team's targets: the simulation completes the year of every region up to its target ───
TARGET_ADJUSTED = "ajustado"                    # sold + simulated = its target
TARGET_ALREADY_REACHED = "objetivo_ya_superado" # sold already reaches the target: nothing more is simulated
TARGET_WITHOUT_SHAPE = "sin_forma"              # a gap to fill but nothing simulated to scale (no sales a year before)
TARGET_NOT_IN_DATA = "region_sin_datos"         # a region of the targets the data does not have (a name to check)
TARGET_MISSING = "sin_objetivo"                 # a region of the data without target: its simulation stays as it is


def read_acquisition_targets(configuration: Config) -> pd.Series:
    """The targets of the year per region, in USD: the CSV of acquisition_targets_file (region, objetivo) × the unit."""
    targets = pd.read_csv(configuration.acquisition_targets_file)
    targets.columns = [str(column).strip().lower() for column in targets.columns]
    targets["region"] = targets["region"].astype(str).str.strip()
    values = pd.to_numeric(targets["objetivo"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return (values * float(configuration.acquisition_target_unit_usd)).groupby(targets["region"]).sum()


def term_in_years(terms: pd.Series) -> pd.Series:
    """The years of a term written as text ('1 year', '2 year' → 1, 2); 1 when it says none."""
    years = pd.to_numeric(terms.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return years.fillna(1).astype(int)


def acquisitions_by_term(fine_table: pd.DataFrame, configuration: Config, region_column: str) -> tuple:
    """Every acquisition of the extract read as a SALE, with its term: an acquisition of t years sold in month m is the
    acquisition row of term t due in m + 12 t. Returns (the sales: region, term in years, month of sale, USD; the last due
    month of the extract)."""
    period = configuration.period_col
    acquired = fine_table[is_acquisition(fine_table, configuration)]
    if configuration.term_column and configuration.term_column in acquired:
        years = term_in_years(acquired[configuration.term_column]).to_numpy()
    else:
        years = np.ones(len(acquired), dtype=int)
    sales = pd.DataFrame({"region": acquired[region_column].astype(str).to_numpy(), "plazo": years,
                          "mes_venta": acquired[period].to_numpy() - configuration.one_year_term_months * years,
                          "usd": acquired[configuration.pipeline_usd_col].fillna(0.0).to_numpy()})
    return sales, fine_table[period].max()


def term_mix(fine_table: pd.DataFrame, configuration: Config, region_column: str) -> tuple:
    """The sales of THIS YEAR by region and term (January to the last closed month) and the proportion of every term.

    The proportion is the one of this year when the extract already shows every term of it (a 3-year licence sold in
    the last closed month falls due 36 months later: the extract has to reach that month). When it does not, the
    proportion is the one of the last 12 months of sales whose every term is visible, and the sales of this year are
    estimated from the 1-year ones (sold 1 year / its proportion). acquisition_target_all_terms = False: only 1 year.

    Returns (sold this year: region × term, USD; the proportion: region × term; where the proportion comes from;
    whether the sales of this year are read or estimated)."""
    current = configuration.calendar_boundaries()["current"]
    sales, last_due = acquisitions_by_term(fine_table, configuration, region_column)
    if not configuration.acquisition_target_all_terms:
        sales = sales[sales["plazo"] == 1]
    terms = sorted(sales["plazo"].unique()) if len(sales) else [1]
    longest = int(max(terms))
    this_year = sales[(sales["mes_venta"].map(lambda month: month.year) == current.year) & (sales["mes_venta"] < current)]
    sold = this_year.pivot_table(index="region", columns="plazo", values="usd", aggfunc="sum").reindex(columns=terms).fillna(0.0)
    every_term_visible = (current - 1) + configuration.one_year_term_months * longest <= last_due
    if every_term_visible:
        mix = sold.div(sold.sum(axis=1).where(sold.sum(axis=1) > 0), axis=0)
        source = f"las ventas de {current.year} (enero a {current - 1}), todos los plazos vistos en el extracto"
        return sold, mix, source, True
    last_complete = last_due - configuration.one_year_term_months * longest
    window = list(pd.period_range(last_complete - 11, last_complete, freq="M"))
    reference = sales[sales["mes_venta"].isin(window)]
    by_term = reference.pivot_table(index="region", columns="plazo", values="usd", aggfunc="sum").reindex(columns=terms).fillna(0.0)
    mix = by_term.div(by_term.sum(axis=1).where(by_term.sum(axis=1) > 0), axis=0)
    portfolio = by_term.sum() / by_term.sum().sum() if by_term.sum().sum() > 0 else pd.Series(1.0, index=[1])
    regions = sorted(set(sold.index) | set(mix.index))
    mix = mix.reindex(regions)
    mix = mix.apply(lambda row: row if row.notna().all() else portfolio.reindex(terms).fillna(0.0), axis=1)
    # the sales of this year in every term, from the 1-year ones (the only ones the extract shows this year)
    one_year_sold = sold.reindex(regions).get(1, pd.Series(0.0, index=regions)).fillna(0.0)
    share_one = mix.get(1, pd.Series(1.0, index=regions)).replace(0, np.nan)
    estimated = mix.mul((one_year_sold / share_one).fillna(0.0), axis=0)
    source = (f"las ventas de {window[0]}..{window[-1]} (el extracto llega a {last_due}: no muestra todavía los plazos de "
              f"{longest} años vendidos en {current.year}); lo vendido este año en 2 y 3 años se estima desde el de 1 año")
    return estimated, mix, source, False


def complete_to_the_targets(box: "FuturePipeline", fine_table: pd.DataFrame) -> pd.DataFrame:
    """The year of every region up to the acquisition team's target, in EVERY term (the target counts them all):

        sold this year (January to the last closed month, every term) + simulated (current month to December, every
        term) = target, the simulated split by terms in the proportion of the sales of the year.

    Only the 1-year part falls due inside the horizon, so only it becomes rows of the window: the 1-year simulated rows
    of the region are multiplied by ONE factor up to it (its mix of products, acquisition values and months stays the
    one of the simulation). The 2- and 3-year parts are counted in the table, so the 1-year part is the right one, but
    they fall due after the horizon and make no rows. Returns the table of every region (box.simulated changes in place)."""
    configuration = box.configuration
    region_column = configuration.acquisition_target_column
    targets = read_acquisition_targets(configuration)
    sold, mix, source, read_in_extract = term_mix(fine_table, configuration, region_column)
    terms = list(mix.columns) if len(mix.columns) else [1]
    simulated = box.simulated
    simulated_before = (simulated[configuration.pipeline_usd_col].groupby(simulated[region_column].astype(str)).sum()
                        if len(simulated) and region_column in simulated else pd.Series(dtype=float))
    known_regions = set(fine_table[region_column].dropna().astype(str).unique())
    rows = []
    for region in sorted(set(targets.index) | set(simulated_before.index) | set(sold.index)):
        target = float(targets.get(region, np.nan))
        sold_by_term = sold.loc[region] if region in sold.index else pd.Series(0.0, index=terms)
        sold_usd = float(sold_by_term.sum())
        share = mix.loc[region] if region in mix.index else pd.Series([1.0] + [0.0] * (len(terms) - 1), index=terms)
        share = share.fillna(0.0)
        share_one = float(share.get(1, 1.0))
        before = float(simulated_before.get(region, 0.0))
        missing = target - sold_usd if np.isfinite(target) else np.nan
        if region not in known_regions and np.isfinite(target):
            state, factor = TARGET_NOT_IN_DATA, np.nan
        elif not np.isfinite(target):
            state, factor = TARGET_MISSING, 1.0
        elif missing <= 0:
            state, factor = TARGET_ALREADY_REACHED, 0.0
        elif before <= 0 or share_one <= 0:
            state, factor = TARGET_WITHOUT_SHAPE, np.nan
        else:
            state, factor = TARGET_ADJUSTED, missing * share_one / before
        if np.isfinite(factor) and len(simulated):
            in_region = (simulated[region_column].astype(str) == region).to_numpy()
            for column_name in (configuration.pipeline_units_col, configuration.pipeline_usd_col):
                simulated.loc[in_region, column_name] = simulated.loc[in_region, column_name] * factor
        one_year_after = before * factor if np.isfinite(factor) else before
        if state == TARGET_ADJUSTED:
            simulated_by_term = {term: missing * float(share.get(term, 0.0)) for term in terms}
        else:
            simulated_by_term = {term: (one_year_after if term == 1 else 0.0) for term in terms}
        row = {"region": region, "objetivo_usd": target, "vendido_hasta_hoy_usd": sold_usd}
        row.update({f"vendido_{term}_anos_usd": float(sold_by_term.get(term, 0.0)) for term in terms})
        row.update({f"parte_{term}_anos": float(share.get(term, 0.0)) for term in terms})
        row.update({"falta_usd": missing, "simulado_por_el_metodo_1_ano_usd": before, "factor_1_ano": factor})
        row.update({f"simulado_{term}_anos_usd": float(simulated_by_term[term]) for term in terms})
        final = sold_usd + sum(simulated_by_term.values())
        row.update({"final_usd": final,
                    "diferencia_pct": (final / target - 1) if np.isfinite(target) and target > 0 else np.nan,
                    "estado": state})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["origen_de_la_proporcion"] = source
    table.attrs["vendido_leido_en_el_extracto"] = read_in_extract
    box.simulated = simulated[simulated[configuration.pipeline_units_col] > 0].reset_index(drop=True) if len(simulated) else simulated
    return table


def show_acquisition_targets(table: pd.DataFrame, configuration: Config) -> None:
    """The acquisitions completed up to the acquisition team's targets, region by region and term by term, on screen."""
    if table is None or table.empty:
        return
    doc = configuration.logger.doc
    doc(f"[{STEP_5_1_LABEL}] ══════════ ADQUISICIÓN HASTA EL OBJETIVO DEL EQUIPO DE ADQUISICIÓN ══════════")
    doc(f"[{STEP_5_1_LABEL}] por {configuration.acquisition_target_column}: el objetivo del año ({configuration.acquisition_targets_file}), "
        f"en todos los plazos; lo vendido de enero al último mes cerrado; lo que falta, repartido por plazos en la proporción de "
        f"{table.attrs.get('origen_de_la_proporcion')}. Solo la parte de 1 año vence en el horizonte: la simulación de 1 año se "
        f"escala por un factor por región (la mezcla de productos no cambia); la de 2 y 3 años se cuenta, pero vence después")
    shown = table.copy()
    for column_name in shown.columns:
        if column_name.endswith("_usd"):
            shown[column_name] = shown[column_name].round(0)
        elif column_name.startswith("parte_") or column_name == "factor_1_ano":
            shown[column_name] = shown[column_name].round(3)
    shown["diferencia_pct"] = (100 * shown["diferencia_pct"]).round(2)
    configuration.show_table(shown)
    method = table["simulado_por_el_metodo_1_ano_usd"].sum()
    one_year = table["simulado_1_anos_usd"].sum() if "simulado_1_anos_usd" in table else method
    every_term = sum(table[column].sum() for column in table.columns if column.startswith("simulado_") and column.endswith("_anos_usd"))
    doc(f"[{STEP_5_1_LABEL}] VALORACIÓN: hasta diciembre faltan ${table['falta_usd'].clip(lower=0).sum():,.0f} de objetivo; se "
        f"simulan ${every_term:,.0f} en todos los plazos, de ellos ${one_year:,.0f} de 1 año (el método simulaba ${method:,.0f}"
        + (f", {one_year / method - 1:+.1%})" if method > 0 else ")"))
    if not table.attrs.get("vendido_leido_en_el_extracto", True):
        doc(f"[{STEP_5_1_LABEL}] AVISO: el extracto no muestra todavía las ventas de 2 y 3 años de este año (vencen después de su "
            f"último mes): lo vendido en esos plazos se ha estimado desde el de 1 año con la proporción de referencia")
    for state, text in ((TARGET_NOT_IN_DATA, "regiones del objetivo que no están en los datos (revisar el nombre)"),
                        (TARGET_MISSING, "regiones de los datos sin objetivo (se quedan con la simulación del método)"),
                        (TARGET_ALREADY_REACHED, "regiones que ya han vendido su objetivo (no se simula más)"),
                        (TARGET_WITHOUT_SHAPE, "regiones con hueco pero sin ventas de 1 año un año antes que escalar")):
        names = table.loc[table["estado"] == state, "region"].tolist()
        if names:
            doc(f"[{STEP_5_1_LABEL}] AVISO: {text}: {', '.join(names)}")
    doc(f"[{STEP_5_1_LABEL}] {int((table['estado'] == TARGET_ADJUSTED).sum())} regiones ajustadas · "
        f"══════════ FIN DE ADQUISICIÓN HASTA EL OBJETIVO ══════════")

# ─── the acquisitions month by month, in total and by term: what is drawn at the end of step 5.1 ───
KIND_REAL, KIND_ESTIMATED, KIND_SIMULATED = "real", "estimado", "simulado"
ACQUISITION_CHART = "05_1_adquisicion_mes_a_mes"


def acquisitions_month_by_month(box: "FuturePipeline", fine_table: pd.DataFrame) -> pd.DataFrame:
    """The acquisitions by month of sale and term, in USD, from January of acquisition_chart_history_years before to December:

      real       sold, read in the extract (a t-year licence sold in m falls due in m + 12 t: the extract shows it)
      estimado   sold, but its term falls due after the last month of the extract: the 1-year sales of that month ×
                 the proportion of the term over the 1-year ones in the last 12 months with every term visible
      simulado   from the current month to December: the 1-year simulation (after the target); the 2- and 3-year
                 parts of the target (or, without target, the 1-year simulation × the same proportion), spread over the
                 months like the 1-year simulation

    One row per month and term: mes_venta, plazo, usd, tipo."""
    configuration = box.configuration
    current = configuration.calendar_boundaries()["current"]
    period = configuration.period_col
    region_column = configuration.acquisition_target_column or (configuration.business_mandatory_dims or [period])[0]
    sales, last_due = acquisitions_by_term(fine_table, configuration, region_column)
    if sales.empty:
        return pd.DataFrame(columns=["mes_venta", "plazo", "usd", "tipo"])
    terms = sorted(int(term) for term in sales["plazo"].unique())
    first_month = pd.Period(f"{current.year - configuration.acquisition_chart_history_years}-01", freq="M")
    months = list(pd.period_range(first_month, box.window[-1] if box.window else current - 1, freq="M"))
    visible = sales["mes_venta"] + (configuration.one_year_term_months * sales["plazo"]).to_numpy() <= last_due
    sold = sales[visible & (sales["mes_venta"] < current)].groupby(["mes_venta", "plazo"])["usd"].sum()
    # the proportion of every term over the 1-year sales, in the last 12 months whose every term is visible
    longest = max(terms)
    last_complete = last_due - configuration.one_year_term_months * longest
    reference_months = list(pd.period_range(last_complete - 11, last_complete, freq="M"))
    reference = sales[sales["mes_venta"].isin(reference_months)].groupby("plazo")["usd"].sum()
    over_one_year = (reference / reference.get(1, np.nan)).fillna(0.0) if reference.get(1, 0) > 0 else pd.Series(0.0, index=terms)
    simulated_one_year = (box.simulated[configuration.pipeline_usd_col].groupby(
        box.simulated[period] - configuration.one_year_term_months).sum() if len(box.simulated) else pd.Series(dtype=float))
    targets = getattr(box, "targets", pd.DataFrame())
    rows = []
    for month in months:
        for term in terms:
            if month < current:
                value = float(sold.get((month, term), 0.0))
                kind = KIND_REAL
                if month + configuration.one_year_term_months * term > last_due:
                    value, kind = float(sold.get((month, 1), 0.0)) * float(over_one_year.get(term, 0.0)), KIND_ESTIMATED
            else:
                one_year = float(simulated_one_year.get(month, 0.0))
                if term == 1:
                    value = one_year
                elif len(targets) and f"simulado_{term}_anos_usd" in targets:
                    share_of_month = one_year / simulated_one_year.sum() if simulated_one_year.sum() > 0 else 0.0
                    value = float(targets[f"simulado_{term}_anos_usd"].sum()) * share_of_month
                else:
                    value = one_year * float(over_one_year.get(term, 0.0))
                kind = KIND_SIMULATED
            rows.append({"mes_venta": month, "plazo": term, "usd": value, "tipo": kind})
    return pd.DataFrame(rows)


def draw_the_acquisitions(box: "FuturePipeline", fine_table: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """The two charts of the acquisitions, always shown: the total month by month with the total of every year, and the
    same by term (1, 2 and 3 years). Returns the months drawn (mes_venta, plazo, usd, tipo), for the report."""
    from graficos import draw_acquisitions_by_month
    from time_series_charts import chart_path, draw_without_stopping
    months = acquisitions_month_by_month(box, fine_table)
    if months.empty:
        return months
    target = (float(read_acquisition_targets(configuration).sum()) if configuration.acquisition_targets_file else None)
    current = configuration.calendar_boundaries()["current"]
    path = chart_path(configuration, ACQUISITION_CHART)
    drawn = draw_without_stopping(lambda: draw_acquisitions_by_month(months, current, target, path), STEP_5_1_LABEL, configuration)
    if drawn:
        configuration.show_figure(STEP_5_1_LABEL, "la adquisición mes a mes, en total y por plazo", drawn, always=True)
    return months


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
    if len(box.targets):
        adjusted = box.targets[box.targets["estado"] == TARGET_ADJUSTED]
        off = adjusted[adjusted["diferencia_pct"].abs() > configuration.acquisition_target_tolerance]
        problems = box.targets[box.targets["estado"].isin([TARGET_NOT_IN_DATA, TARGET_WITHOUT_SHAPE])]
        configuration.log_check(STEP_5_1_LABEL, check_log,
                                f"every region of the acquisition team's targets ends the year within "
                                f"{configuration.acquisition_target_tolerance:.0%} of "
                                f"its target (sold + simulated)",
                                off.empty and problems.empty,
                                failure_detail=" · ".join([f"{row['region']}: {row['diferencia_pct']:+.1%}" for _, row in off.iterrows()]
                                                          + [f"{row['region']}: {row['estado']}" for _, row in problems.iterrows()])
                                               + " · see ADQUISICIÓN HASTA EL OBJETIVO",
                                blocking=False,
                                context=f"{len(adjusted)} regions adjusted · 1-year simulated "
                                        f"${box.targets['simulado_por_el_metodo_1_ano_usd'].sum():,.0f} → "
                                        f"${box.targets.get('simulado_1_anos_usd', box.targets['simulado_por_el_metodo_1_ano_usd']).sum():,.0f}")
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
    show_acquisition_targets(box.targets, configuration)
    show_acquisition_exam(exam, configuration)

    # [7] the acquisitions month by month, in total and by term (always shown)
    configuration.log_action(STEP_5_1_LABEL, 7, "the acquisitions month by month, in total and by term")
    by_month = draw_the_acquisitions(box, fine_table, configuration)
    return dict(box=box, comparison=comparison, exam=exam, targets=box.targets, by_month=by_month)


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
