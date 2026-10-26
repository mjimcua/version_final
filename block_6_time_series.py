"""
block_6_time_series.py — BLOCK 6 · TIME SERIES: the retail-to-subscription universe, by region.

MISSION: give what the retail-to-subscription universe still brings this year, the whole of next year, and the
pipeline of next year it creates, by region, with a band.

MECHANISM: the universe comes with its result only (what converted), so it is predicted with its own series: the same
month a year before × the level of the region, with the level rule that best simulated the closed months (the exam
compares four rules on the same origins); next year repeats this one by default; what is projected this year
falls due next year as pipeline (40 % off, renewed at 100 %, at the rate of the region); the simulation is examined on
the closed months and its error by horizon is the band of every row. Less science than blocks 2 to 5, on purpose: a
piece that fits with the rest and can be improved on its own.

THE STEPS:
  6.1  time series universe   real, projected, simulated and re-entry rows by region and month, the exam, the chart
                              with the prediction (the same chart step 1.2 draws with the history)

READS (block 1): time_series_rows, fine_table.
GIVES: time_series_universe (the box: its rows with their band, its exam). Writes sff_ts_examen.

THE INTERFACE:
  block = TimeSeriesBlock(configuration).run(time_series_rows, fine_table)
  block.time_series_universe    block.results()    block.seconds

  The box can also be built alone: TimeSeriesUniverse(time_series_rows, fine_table, configuration).

THE BOX, IN DETAIL (until 25-oct-2026: time_series_universe.py) — see the section of step 6.1 below.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import time

import numpy as np
import pandas as pd

from config import (CALENDAR_ROLE_COLUMN, Config, TABLE_TS_EXAM, TRUTH_ROLES, TS_LEVEL_RULES, TS_PROJECTED, TS_REAL,
                    TS_REENTRY, TS_REGION_KEY, TS_SIMULATED, join_columns, region_columns_of)
from time_series_charts import draw_the_prediction_of_the_time_series, draw_without_stopping


# ═══════════════════════════════════════════════════════════════════════════════════
# STEP 6.1 · TIME SERIES UNIVERSE  (until 25-oct-2026: time_series_universe.py)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE TIME_SERIES UNIVERSE (retail to subscription): one piece that simulates what the
# universe still brings this year and next year, and the pipeline it creates, and hands the forecast its rows.
#
# WHAT IT IS: the rows flagged time_series are the revenue of the retail-to-subscription campaigns (licences bought in
# retail that move to an online subscription with an offer). They come with the RESULT only (renewed units and USD:
# what converted); nobody knows what they were worth before. There is no signal proportional to the people who move
# and the campaigns are not modelled: it is the best effort with the series itself, BY REGION (ts_region_columns,
# coarse to fine; the projection is made at the finest level). Less science than the rest of the forecast, on purpose:
# a piece that fits with everything else and can be improved on its own.
#
# THE ROWS IT GIVES (region × month; every one carries revenue and its band):
#   · ts_real         the closed months of this year, as they are (revenue: what converted)
#   · ts_proyectado   the rest of this year (the simulation window, current month included):
#                        units = units of the same month last year × level of the region
#                        level = the rule the exam chose (TS_LEVEL_RULES, config.py): the region's own (Σ units of the
#                                last ts_level_window_months closed months / Σ the same months a year before), the
#                                region's moved towards the total's by credibility, the total's, or none (1)
#                        value = units × value per unit of the region (last ts_auv_window_months closed months)
#   · ts_simulado     the WHOLE of next year, month by month, with the same rule on this year's months (real where they
#                     are closed, projected where they are not): units = units of the same month this year ×
#                     ts_next_year_level (1.0 by default: next year repeats this one, the campaigns in their month;
#                     None: the level of the region again, the growth of its last months kept one more year — a campaign
#                     inside the level window then multiplies the whole of next year)
#   · ts_reentrada    the projected months of this year falling due again next year, the same month, as PIPELINE: units
#                     and value as converted (ts_acquisition_discount, 40 % off); renewed at 100 %: value to renew =
#                     value / (1 − 0.4); renewed revenue = value to renew × the renewal rate of the region in the normal
#                     pipeline (finest region level that has one, last ts_rate_window_months closed months; else global).
#                     The conversions of the CLOSED months are already in the extract as pipeline of next year: they do
#                     not re-enter (they would be counted twice). Next year's conversions fall due the year after: out of
#                     the horizon.
#   A region with no same month a year before or no level takes its share of the projected TOTAL (all regions).
#
# THE BAND (the same principle as the official band of the forecast: the error measured on months already closed):
#   the simulation is examined on the last ts_exam_origins closed months: at every origin the level and the value per
#   unit are taken with the months before it only, the months from the origin to the last closed one are simulated, and
#   the error of the TOTAL revenue of a month is measured at every horizon h (months after the last month known). Every
#   level rule is examined on the same origins; the one with the least mean error simulates (ts_level_rule fixes one),
#   and its error is the band — chosen and measured on the same months, so the band is somewhat optimistic. A row
#   takes error(h) × its revenue as its half band (beyond the last horizon examined, the last one: a lower bound); a
#   month more than 12 months ahead is simulated on a month that was simulated itself, so it adds that month's error in
#   quadrature; a re-entry takes the error of the month it comes from (its volume was simulated then; the error of the
#   rate of the region is not in the band). The forecast adds these half bands to its official band (step 7.2).
#
# THE INTERFACE:
#   universe = TimeSeriesUniverse(time_series_rows, fine_table, configuration)
#   universe.table          every row above (the time_series table of the core and of the total)
#   universe.exam           the exam of the simulation: per origin and month, and the error per horizon (of the rule
#                           chosen); by_rule: every rule × horizon (the table sff_ts_examen); rule: the rule chosen
#   universe.level_rule     the level rule that simulates
#   universe.by_month()     per future month: revenue of ts_proyectado / ts_simulado, pipeline and expected renewals of
#                           ts_reentrada, and their half bands — what the forecast adds to its months
#   No time_series row: every table is empty and the forecast adds nothing.

# ─── named constants ─────────────────────────────────────────────────────────────
MONTHS_PER_YEAR = 12
REGION_KEY = TS_REGION_KEY          # the key of a region: its levels joined, coarse to fine
TABLE_COLUMNS = ["origen", "unidades", "valor", "revenue", "revenue_bajo", "revenue_alto", "h", "nivel", "valor_medio",
                 "por_cuota", "mes_origen", "descuento", "valor_a_renovar", "tasa", "nivel_tasa", "tasa_global",
                 "unidades_renovadas"]


# ─── the step ────────────────────────────────────────────────────────────────────
STEP_6_1_LABEL = "6.1"
STEP_6_1_NAME = "TIME SERIES UNIVERSE"
STEP_6_1_PURPOSE = ("simulate, by region, the retail-to-subscription revenue the extract cannot have yet: the rest of this "
                "year, the whole of next year and the pipeline next year it creates (the re-entry of what is projected), "
                "with a band from the exam of the simulation on the closed months")
STEP_6_1_ACTIONS = ["the time_series rows by region and month",
                "this year: the level and the value per unit of every region, and the projected months",
                "next year: the simulated months and the re-entry of the projected ones, at the rate of the region",
                "the exam of the simulation on the closed months, for every level rule: the rule chosen and the band "
                "of every row",
                "check the universe (checks 1-4)",
                "write the exam of the simulation (check 5)",
                "count the checks; stop if any failed",
                "show the universe, its exam and its chart with the prediction (the same chart step 1.2 draws with the "
                "history)"]
STEP_6_1_OUTPUT = ("the box (TimeSeriesUniverse): region × month rows with their band, for the forecast and the core (in "
                   "memory); sff_ts_examen (level rule × horizon: the error of the simulated total, the rule chosen)")


class TimeSeriesUniverse:
    """THE TIME_SERIES UNIVERSE (see the header): built once, before the forecast; the forecast adds its months."""

    def __init__(self, time_series_rows: pd.DataFrame, fine_table: pd.DataFrame, configuration: Config):
        self.configuration = configuration
        self.region_columns = region_columns_of(configuration)
        self.current = configuration.calendar_boundaries()["current"]
        self.last_closed = self.current - 1
        self.next_year = self.current.year + 1
        period = configuration.period_col

        # the series of every region, month by month: what converted (renewed units and USD)
        rows = time_series_rows if time_series_rows is not None else pd.DataFrame()
        if len(rows):
            keyed = rows.assign(**{REGION_KEY: join_columns(rows, self.region_columns)})
            self.monthly = (keyed.groupby([REGION_KEY, period])[[configuration.renewed_units_col, configuration.renewed_usd_col]]
                            .sum(min_count=1).fillna(0).reset_index()
                            .rename(columns={configuration.renewed_units_col: "unidades", configuration.renewed_usd_col: "valor"}))
            self.levels_of_region = keyed[[REGION_KEY] + self.region_columns].drop_duplicates(REGION_KEY).set_index(REGION_KEY)
        else:
            self.monthly = pd.DataFrame(columns=[REGION_KEY, period, "unidades", "valor"])
            self.levels_of_region = pd.DataFrame(columns=self.region_columns)
        self.closed_monthly = self.monthly[self.monthly[period] < self.current]

        # the exam of the simulation, every level rule on the same origins: the rule chosen simulates, its error is the band
        self.exam = examine_the_simulation(self.closed_monthly, configuration)
        self.level_rule = self.exam["rule"]

        # this year: real and projected; next year: simulated; the re-entry of the projected months
        self.parameters = region_parameters(self.closed_monthly, self.current, configuration, self.level_rule)
        window = [month for month in configuration.simulation_window if month.year == self.current.year]
        self.projected, self.fallback_regions = project_months(self.closed_monthly, self.parameters, window, configuration)
        real = self.closed_monthly[self.closed_monthly[period].map(lambda month: month.year) == self.current.year]
        self.real = real.assign(origen=TS_REAL)
        this_year = pd.concat([self.real[[REGION_KEY, period, "unidades", "valor"]],
                               self.projected[[REGION_KEY, period, "unidades", "valor"]]], ignore_index=True)
        next_year_months = list(pd.period_range(f"{self.next_year}-01", f"{self.next_year}-12", freq="M"))
        next_year_parameters = self.parameters.copy()
        next_year_parameters.attrs = dict(self.parameters.attrs)
        if configuration.ts_next_year_level is not None:
            next_year_parameters["nivel"] = float(configuration.ts_next_year_level)
            next_year_parameters.attrs["total_nivel"] = float(configuration.ts_next_year_level)
        self.simulated, _ = project_months(this_year, next_year_parameters, next_year_months, configuration)
        self.simulated["origen"] = TS_SIMULATED
        self.rates, self.global_rate = region_rates(fine_table, self.current, configuration)
        self.reentry = reentry_rows(self.projected.join(self.levels_of_region, on=REGION_KEY) if len(self.projected)
                                    else self.projected, self.rates, self.global_rate, configuration)

        # every row with its band (the error of the exam at its horizon)
        self.table = self.assemble_the_table()

    # ─── the table ───────────────────────────────────────────────────────────────
    def assemble_the_table(self) -> pd.DataFrame:
        """Every row of the universe (real, projected, simulated, re-entry) with its revenue, its band and its region."""
        configuration, period = self.configuration, self.configuration.period_col
        frames = []
        if len(self.real):
            frames.append(self.real.assign(revenue=self.real["valor"]))
        if len(self.projected):
            frames.append(self.projected.assign(revenue=self.projected["valor"]))
        if len(self.simulated):
            frames.append(self.simulated.assign(revenue=self.simulated["valor"]))
        if len(self.reentry):
            frames.append(self.reentry)
        if not frames:
            return pd.DataFrame(columns=self.region_columns + [REGION_KEY, period] + TABLE_COLUMNS)
        table = pd.concat(frames, ignore_index=True)
        table = table.drop(columns=self.region_columns, errors="ignore").join(self.levels_of_region, on=REGION_KEY)
        # h: months after the last closed one; a re-entry takes the h of the month it was simulated in
        simulated_in = table["mes_origen"].where(table["origen"] == TS_REENTRY, table[period]) if "mes_origen" in table \
            else table[period]
        table["h"] = [(month - self.last_closed).n for month in simulated_in]
        error_of_h = self.error_at_horizon(table["h"])
        is_real = table["origen"] == TS_REAL
        # an error above 100 % would take the low end below 0: no month converts less than nothing
        table["revenue_bajo"] = np.where(is_real, table["revenue"], table["revenue"] * np.clip(1 - error_of_h, 0, None))
        table["revenue_alto"] = np.where(is_real, table["revenue"], table["revenue"] * (1 + error_of_h))
        ordered = self.region_columns + [REGION_KEY, period] + [column for column in TABLE_COLUMNS if column in table.columns]
        return table.reindex(columns=ordered)

    def error_at_horizon(self, horizons) -> np.ndarray:
        """The error of the simulated total at every horizon: the exam's, interpolated, held beyond the last horizon
        examined (a lower bound there); 0 without an exam. Beyond 12 months the month is simulated on a month that was
        simulated itself (the same month a year before was projected): its error is that month's and one more year's,
        in quadrature: √(error(h − 12)² + error(h)²)."""
        errors = self.exam["by_horizon"]
        horizons = np.asarray(list(horizons), dtype=float)
        if errors.empty:
            return np.zeros(len(horizons))
        examined_h, examined_error = errors["h"].to_numpy(dtype=float), errors["error_usd"].to_numpy(dtype=float)
        error = np.interp(horizons, examined_h, examined_error)
        on_a_simulated_month = horizons > MONTHS_PER_YEAR
        base_error = np.interp(horizons - MONTHS_PER_YEAR, examined_h, examined_error)
        return np.where(on_a_simulated_month, np.sqrt(base_error ** 2 + error ** 2), error)

    def by_month(self) -> pd.DataFrame:
        """Per future month, what the forecast adds: revenue of the projected and simulated conversions, pipeline and
        expected renewals of the re-entries, and their half bands."""
        period = self.configuration.period_col
        future = self.table[self.table["origen"] != TS_REAL]
        if future.empty:
            return pd.DataFrame(columns=[period, "ingresos_ts_usd", "vence_ts_reentrada_usd", "esperado_ts_reentrada",
                                         "error_ts_usd"])
        is_revenue = future["origen"].isin([TS_PROJECTED, TS_SIMULATED])
        is_reentry = future["origen"] == TS_REENTRY
        grouped = future.assign(_ingresos=future["revenue"].where(is_revenue, 0.0),
                                _vence=future["valor"].where(is_reentry, 0.0),
                                _esperado=future["revenue"].where(is_reentry, 0.0),
                                _error=future["revenue_alto"] - future["revenue"]).groupby(period)
        return pd.DataFrame({"ingresos_ts_usd": grouped["_ingresos"].sum(), "vence_ts_reentrada_usd": grouped["_vence"].sum(),
                             "esperado_ts_reentrada": grouped["_esperado"].sum(), "error_ts_usd": grouped["_error"].sum()}
                            ).reset_index()

    def real_revenue_by_year(self) -> pd.Series:
        """The revenue of the closed months of the universe, by year."""
        period = self.configuration.period_col
        return self.closed_monthly.groupby(self.closed_monthly[period].map(lambda month: month.year))["valor"].sum()


# ═══════════════════════════════════════════════════════════════════════════════════
# THE RULES: level and value per unit, the projection, the rate of the re-entry
# ═══════════════════════════════════════════════════════════════════════════════════

def region_parameters(closed_monthly: pd.DataFrame, as_of, configuration: Config, rule: str = "region") -> pd.DataFrame:
    """Per region, with the months before as_of only: level (last ts_level_window_months vs the same months a year
    before; only the months whose year before is inside the history), value per unit (last ts_auv_window_months), share
    of units (same window) and the range of its monthly value per unit.

    The level by rule (TS_LEVEL_RULES): region = its own (nivel_region); credibilidad = z · its own + (1 − z) · the
    total's, z = units a year before / (units a year before + the mean of the regions); total = the total's; sin_nivel
    = 1. A region without a level of its own (nothing a year before) keeps none: it goes by its share of the total."""
    period, region = configuration.period_col, REGION_KEY
    known = closed_monthly[closed_monthly[period] < as_of]
    # the months of the level window whose month a year before is inside the history (before the first month of the
    # extract nothing is known: it is not a month with 0 sold)
    first_known = known[period].min() if len(known) else as_of
    level_months = [month for month in pd.period_range(as_of - configuration.ts_level_window_months, as_of - 1, freq="M")
                    if month - MONTHS_PER_YEAR >= first_known]
    last_year_months = [month - MONTHS_PER_YEAR for month in level_months]
    auv_window = known[known[period] >= as_of - configuration.ts_auv_window_months]
    recent = known[known[period].isin(level_months)].groupby(region)["unidades"].sum()
    before = known[known[period].isin(last_year_months)].groupby(region)["unidades"].sum()
    grouped = auv_window.groupby(region)
    parameters = pd.DataFrame({"unidades_recientes": recent, "unidades_ano_anterior": before,
                               "valor_medio": grouped["valor"].sum() / grouped["unidades"].sum().replace(0, np.nan),
                               "cuota_unidades": grouped["unidades"].sum() / auv_window["unidades"].sum()})
    monthly_auv = auv_window.assign(_auv=auv_window["valor"] / auv_window["unidades"].replace(0, np.nan)).groupby(region)["_auv"]
    parameters["valor_medio_min"], parameters["valor_medio_max"] = monthly_auv.min(), monthly_auv.max()
    # a region with no row in the level window sold nothing there: 0 recent units, so a level of 0 (not "unknown",
    # which would project it by its share); without units a year before, the level stays unknown
    parameters["unidades_recientes"] = parameters["unidades_recientes"].fillna(0.0)
    parameters["nivel_region"] = parameters["unidades_recientes"] / parameters["unidades_ano_anterior"].replace(0, np.nan)
    total_level = float(recent.sum() / before.sum()) if before.sum() > 0 else np.nan
    has_level = parameters["nivel_region"].notna()
    credibility = parameters["unidades_ano_anterior"] / (parameters["unidades_ano_anterior"]
                                                         + parameters["unidades_ano_anterior"].mean())
    parameters["z_nivel"] = credibility.where(has_level)
    if rule == "region":
        parameters["nivel"] = parameters["nivel_region"]
    elif rule == "credibilidad":
        parameters["nivel"] = credibility * parameters["nivel_region"] + (1 - credibility) * total_level
    elif rule == "total":
        parameters["nivel"] = pd.Series(total_level, index=parameters.index).where(has_level)
    elif rule == "sin_nivel":
        parameters["nivel"] = pd.Series(1.0, index=parameters.index).where(has_level)
        total_level = 1.0 if np.isfinite(total_level) else total_level
    else:
        raise ValueError(f"unknown level rule {rule}: one of {list(TS_LEVEL_RULES)}")
    parameters.attrs["regla_nivel"] = rule
    parameters.attrs["total_nivel"] = total_level
    parameters.attrs["total_valor_medio"] = (float(auv_window["valor"].sum() / auv_window["unidades"].sum())
                                             if auv_window["unidades"].sum() > 0 else np.nan)
    return parameters


def project_months(base_monthly: pd.DataFrame, parameters: pd.DataFrame, months: list, configuration: Config) -> tuple:
    """Units of a region in a month = its units in the same month a year before (in base_monthly: real or already
    projected) × its level; value = units × its value per unit. A region with no same month a year before or no level:
    its share of the projected TOTAL (all regions). Returns (the projected rows, the regions projected by share)."""
    period, region = configuration.period_col, REGION_KEY
    columns = [region, period, "origen", "unidades", "valor", "nivel", "valor_medio", "por_cuota"]
    if base_monthly.empty or parameters.empty or not months:
        return pd.DataFrame(columns=columns), set()
    last_year = base_monthly.groupby([region, period])["unidades"].sum()
    total_last_year = base_monthly.groupby(period)["unidades"].sum()
    rows, fallback_regions = [], set()
    for month in months:
        total_units = total_last_year.get(month - MONTHS_PER_YEAR, np.nan) * parameters.attrs["total_nivel"]
        for region_value, region_parameters_row in parameters.iterrows():
            same_month = last_year.get((region_value, month - MONTHS_PER_YEAR), np.nan)
            level, auv = region_parameters_row["nivel"], region_parameters_row["valor_medio"]
            by_share = not (np.isfinite(same_month) and np.isfinite(level))
            if by_share:
                fallback_regions.add(region_value)
                units = total_units * region_parameters_row["cuota_unidades"]
                auv = auv if np.isfinite(auv) else parameters.attrs["total_valor_medio"]
            else:
                units = same_month * level
            if not np.isfinite(units):
                continue
            rows.append({region: region_value, period: month, "origen": TS_PROJECTED, "unidades": float(units),
                         "valor": float(units * auv), "nivel": level if not by_share else parameters.attrs["total_nivel"],
                         "valor_medio": auv, "por_cuota": int(by_share)})
    return pd.DataFrame(rows, columns=columns), fallback_regions


def region_rates(fine_table: pd.DataFrame, current, configuration: Config) -> tuple:
    """The renewal rate of the normal pipeline at every region level (Σ renewed / Σ due units of the last
    ts_rate_window_months closed months): {depth: Series by the levels joined up to that depth}; and the global rate."""
    region_columns = region_columns_of(configuration)
    window = fine_table[fine_table[CALENDAR_ROLE_COLUMN].isin(TRUTH_ROLES)
                        & (fine_table[configuration.period_col] >= current - configuration.ts_rate_window_months)]
    rates = {}
    for depth in range(1, len(region_columns) + 1):
        keys = join_columns(window, region_columns[:depth])
        grouped = window.groupby(keys)
        rates[depth] = (grouped[configuration.renewed_units_col].sum()
                        / grouped[configuration.pipeline_units_col].sum().replace(0, np.nan)).dropna()
    global_rate = float(window[configuration.renewed_units_col].sum() / window[configuration.pipeline_units_col].sum())
    return rates, global_rate


def reentry_rows(projected: pd.DataFrame, rates: dict, global_rate: float, configuration: Config) -> pd.DataFrame:
    """Every projected month falls due the same month next year: value to renew = value / (1 − discount),
    renewed units = units × rate, revenue = value to renew × rate."""
    region, region_columns = REGION_KEY, region_columns_of(configuration)
    if projected.empty:
        return pd.DataFrame(columns=[region, configuration.period_col, "origen"])
    reentry = projected[[region] + region_columns + [configuration.period_col, "unidades", "valor"]].copy()
    reentry["mes_origen"] = reentry[configuration.period_col]
    reentry[configuration.period_col] = reentry[configuration.period_col] + MONTHS_PER_YEAR
    reentry["origen"] = TS_REENTRY
    reentry["descuento"] = configuration.ts_acquisition_discount           # the pipeline: units and value, 40 % off
    reentry["valor_a_renovar"] = reentry["valor"] / (1 - configuration.ts_acquisition_discount)   # renewed at 100 %
    # the rate climbs the levels: the finest level that has one, then coarser, then the global rate
    reentry["tasa"] = np.nan
    reentry["nivel_tasa"] = "global"
    for depth in range(len(region_columns), 0, -1):
        level_rate = join_columns(reentry, region_columns[:depth]).map(rates[depth])
        take = reentry["tasa"].isna() & level_rate.notna()
        reentry.loc[take, "tasa"] = level_rate[take]
        reentry.loc[take, "nivel_tasa"] = region_columns[depth - 1]
    reentry["tasa_global"] = reentry["tasa"].isna().astype(int)
    reentry["tasa"] = reentry["tasa"].fillna(global_rate)
    reentry["unidades_renovadas"] = reentry["unidades"] * reentry["tasa"]
    reentry["revenue"] = reentry["valor_a_renovar"] * reentry["tasa"]
    return reentry


# ═══════════════════════════════════════════════════════════════════════════════════
# THE EXAM OF THE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════════

def examine_the_simulation(closed_monthly: pd.DataFrame, configuration: Config) -> dict:
    """Every level rule (TS_LEVEL_RULES; only ts_level_rule when it is fixed) examined on the same origins, and the
    rule with the least mean error over the horizons chosen (a tie: the first in TS_LEVEL_RULES).

    OUTPUT:  dict(rule: the rule chosen; by_rule: rule × h: months, mean |error|, chosen (1/0) — the table
             sff_ts_examen; months, by_horizon and read_after_origin of the rule chosen, see exam_of_a_rule).
    EDGE CASES: no closed month or fewer than a year of history → empty tables and the first rule (no band)."""
    rules = [configuration.ts_level_rule] if configuration.ts_level_rule else list(TS_LEVEL_RULES)
    exams = {rule: exam_of_a_rule(closed_monthly, configuration, rule) for rule in rules}
    mean_error = {rule: float(exam["by_horizon"]["error_usd"].mean()) for rule, exam in exams.items()
                  if len(exam["by_horizon"])}
    chosen = min(mean_error, key=lambda rule: (mean_error[rule], rules.index(rule))) if mean_error else rules[0]
    by_rule = pd.concat([exam["by_horizon"].assign(regla_nivel=rule, error_medio_regla=mean_error.get(rule, np.nan),
                                                   elegida=int(rule == chosen))
                         for rule, exam in exams.items()], ignore_index=True) if mean_error else \
        pd.DataFrame(columns=["regla_nivel", "h", "meses", "error_usd", "error_medio_regla", "elegida"])
    by_rule = by_rule.reindex(columns=["regla_nivel", "h", "meses", "error_usd", "error_medio_regla", "elegida"])
    read_after_origin = sum(exam["read_after_origin"] for exam in exams.values())
    return dict(exams[chosen], rule=chosen, by_rule=by_rule, read_after_origin=read_after_origin)


def exam_of_a_rule(closed_monthly: pd.DataFrame, configuration: Config, rule: str) -> dict:
    """At every origin o of the last ts_exam_origins closed months, the level rule with the months before o only
    simulates the months o … the last closed one; the error of the TOTAL revenue of every month against what converted.

    OUTPUT:  dict(months: origin × month: h, real and simulated revenue; by_horizon: per h, the mean |error| over real
             and the months examined; read_after_origin: months at or after an origin the rules read, must be 0).
    EDGE CASES: no closed month or fewer than a year of history → empty tables (the rows take no band)."""
    period = configuration.period_col
    empty = dict(months=pd.DataFrame(columns=["origen", period, "h", "real_usd", "simulado_usd"]),
                 by_horizon=pd.DataFrame(columns=["h", "meses", "error_usd"]), read_after_origin=0)
    if closed_monthly.empty or configuration.ts_exam_origins <= 0:
        return empty
    last_closed = closed_monthly[period].max()
    origins = [last_closed - back for back in range(configuration.ts_exam_origins - 1, -1, -1)]
    rows, read_after_origin = [], 0
    real_by_month = closed_monthly.groupby(period)["valor"].sum()
    for origin in origins:
        known = closed_monthly[closed_monthly[period] < origin]
        read_after_origin += int((known[period] >= origin).sum())
        parameters = region_parameters(known, origin, configuration, rule)
        months = list(pd.period_range(origin, last_closed, freq="M"))
        simulated, _ = project_months(known, parameters, months, configuration)
        simulated_by_month = simulated.groupby(period)["valor"].sum() if len(simulated) else pd.Series(dtype=float)
        for month in months:
            rows.append({"origen": origin, period: month, "h": (month - (origin - 1)).n,
                         "real_usd": float(real_by_month.get(month, 0.0)),
                         "simulado_usd": float(simulated_by_month.get(month, 0.0))})
    months_table = pd.DataFrame(rows)
    examined = months_table[months_table["real_usd"] > 0]
    if examined.empty:
        return dict(empty, months=months_table)
    examined = examined.assign(error_usd=(examined["simulado_usd"] - examined["real_usd"]).abs() / examined["real_usd"])
    by_horizon = examined.groupby("h").agg(meses=("error_usd", "size"), error_usd=("error_usd", "mean")).reset_index()
    return dict(months=months_table, by_horizon=by_horizon, read_after_origin=read_after_origin)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE STEP: build the box, check it, show it
# ═══════════════════════════════════════════════════════════════════════════════════

def build_the_time_series_universe(time_series_rows: pd.DataFrame, fine_table: pd.DataFrame,
                                   configuration: Config) -> TimeSeriesUniverse:
    """The step: the box of the time_series universe, its exam, its checks, its table and its chart.

    INPUT:   the time_series rows (separated in step 1.2) · the fine table (the rate of every region in the normal
             pipeline, for the re-entry) · the Config.
    OUTPUT:  the box (TimeSeriesUniverse).
    EDGE CASES: no time_series row → an empty box: the forecast adds nothing, no check is made."""
    configuration.log_step_start(STEP_6_1_LABEL, STEP_6_1_NAME, STEP_6_1_PURPOSE, STEP_6_1_ACTIONS, STEP_6_1_OUTPUT)
    check_log = []
    universe = TimeSeriesUniverse(time_series_rows, fine_table, configuration)
    period = configuration.period_col

    # [1]-[4] what the box built
    configuration.log_action(STEP_6_1_LABEL, 1, f"{len(universe.monthly):,} region × month rows, "
                                            f"{universe.monthly[REGION_KEY].nunique()} regions at the finest level "
                                            f"({universe.region_columns[-1]})" if len(universe.monthly)
                             else "no time_series row: the universe adds nothing")
    configuration.log_action(STEP_6_1_LABEL, 2, f"{len(universe.projected):,} projected rows "
                                            f"{universe.current}..{universe.current.year}-12: "
                                            f"${universe.projected['valor'].sum() if len(universe.projected) else 0:,.0f} · "
                                            f"{len(universe.fallback_regions)} regions by share of the total")
    configuration.log_action(STEP_6_1_LABEL, 3, f"{len(universe.simulated):,} simulated rows of {universe.next_year} "
                                            f"(${universe.simulated['valor'].sum() if len(universe.simulated) else 0:,.0f}, "
                                            f"next-year level {configuration.ts_next_year_level}) · "
                                            f"{len(universe.reentry):,} re-entry rows: "
                                            f"${universe.reentry['valor'].sum() if len(universe.reentry) else 0:,.0f} of pipeline, "
                                            f"${universe.reentry['revenue'].sum() if len(universe.reentry) else 0:,.0f} expected")
    examined = universe.exam["by_rule"]
    mean_by_rule = examined.groupby("regla_nivel", sort=False)["error_medio_regla"].first() if len(examined) else None
    configuration.log_action(STEP_6_1_LABEL, 4, (f"level rule chosen: {universe.level_rule} · mean error of the "
                                                 f"simulated total: " + " · ".join(
        f"{rule} {error:.0%}" for rule, error in mean_by_rule.items())) if mean_by_rule is not None
                             else "not enough closed months to examine the simulation: its rows go without a band")

    # [5] the checks
    configuration.log_action(STEP_6_1_LABEL, 5, "checking the universe")
    if len(universe.table):
        check_the_universe(universe, STEP_6_1_LABEL, check_log)
    else:
        configuration.log_not_evaluated(STEP_6_1_LABEL, check_log, "the checks of the time_series universe",
                                        "no time_series row")

    # [6] the exam of the simulation, written (every rule × horizon, the rule chosen)
    configuration.log_action(STEP_6_1_LABEL, 6, f"writing {configuration.table_prefix}{TABLE_TS_EXAM}")
    if len(universe.exam["by_rule"]):
        configuration.write_table(STEP_6_1_LABEL, check_log, universe.exam["by_rule"], TABLE_TS_EXAM)
    else:
        configuration.log_not_evaluated(STEP_6_1_LABEL, check_log, f"table {configuration.table_prefix}{TABLE_TS_EXAM} "
                                        "written and read back", "no exam of the simulation")

    # [7] the count of the checks; stop if anything failed
    configuration.log_action(STEP_6_1_LABEL, 7, "counting the checks")
    configuration.log_check_summary(STEP_6_1_LABEL, STEP_6_1_NAME, check_log)

    # [8] on screen
    configuration.log_action(STEP_6_1_LABEL, 8, "the universe, its exam and its chart")
    show_the_universe(universe, STEP_6_1_LABEL)
    if len(universe.table):
        draw_without_stopping(lambda: draw_the_prediction_of_the_time_series(universe, STEP_6_1_LABEL, configuration),
                              STEP_6_1_LABEL, configuration)
    return universe


def check_the_universe(universe: TimeSeriesUniverse, step_label: str, check_log: list) -> None:
    """The checks of the universe: no re-entry from a real month; the value per unit of the projected months inside
    the region's history (warning); every region with a re-entry has a rate of its own (warning); the exam reads
    only the months before its origins."""
    configuration, period = universe.configuration, universe.configuration.period_col
    table = universe.table
    real_months = set(zip(table.loc[table["origen"] == TS_REAL, REGION_KEY], table.loc[table["origen"] == TS_REAL, period]))
    reentry = universe.reentry
    from_real = [key for key in zip(reentry.get(REGION_KEY, []), reentry.get("mes_origen", [])) if key in real_months]
    configuration.log_check(step_label, check_log, "no ts_reentrada row comes from a ts_real month", not from_real,
                            failure_detail=f"{len(from_real)} re-entries from real months (they would be counted twice)",
                            context=f"{len(reentry)} re-entry rows")
    projected = universe.projected
    if len(projected):
        implied = projected.groupby(REGION_KEY)["valor"].sum() / projected.groupby(REGION_KEY)["unidades"].sum()
        low = universe.parameters["valor_medio_min"].reindex(implied.index)
        high = universe.parameters["valor_medio_max"].reindex(implied.index)
        outside = implied[(implied < low - 1e-9) | (implied > high + 1e-9)]
        configuration.log_check(step_label, check_log,
                                "the implied value per unit of the projected months is inside the region's history",
                                outside.empty, blocking=False,
                                failure_detail=f"{len(outside)} regions outside their monthly range: {list(outside.index)[:5]}")
    else:
        configuration.log_not_evaluated(step_label, check_log, "the implied value per unit is inside the region's history",
                                        "nothing projected")
    global_regions = sorted(set(reentry.loc[reentry.get("tasa_global", pd.Series(dtype=int)) == 1, REGION_KEY])) if len(reentry) else []
    rate_levels = reentry["nivel_tasa"].value_counts().to_dict() if len(reentry) else {}
    configuration.log_check(step_label, check_log, "every region with a re-entry has a rate of some region level",
                            not global_regions, blocking=False,
                            failure_detail=f"{len(global_regions)} regions take the global rate: {global_regions[:5]}",
                            context=f"re-entry rows by the level of their rate: {rate_levels}")
    configuration.log_check(step_label, check_log, "the exam of the time_series simulation reads only months before its origins",
                            universe.exam["read_after_origin"] == 0,
                            failure_detail=f"{universe.exam['read_after_origin']} months at or after an origin were read",
                            context=f"{configuration.ts_exam_origins} origins")


def show_the_universe(universe: TimeSeriesUniverse, step_label: str) -> None:
    """On screen: the rows by origin and year, the exam per horizon and the regions with the most projected value."""
    configuration, period = universe.configuration, universe.configuration.period_col
    doc = configuration.logger.doc
    table = universe.table
    if table.empty:
        doc(f"[{step_label}] no time_series row: the universe adds nothing to the forecast")
        return
    doc(f"[{step_label}] ══════════ EL UNIVERSO TIME_SERIES (RETAIL A SUSCRIPCIÓN), POR REGIONES ══════════")
    summary = (table.assign(ano=table[period].map(lambda month: month.year))
               .groupby(["ano", "origen"]).agg(filas=("revenue", "size"), unidades=("unidades", "sum"), valor=("valor", "sum"),
                                               revenue=("revenue", "sum"), revenue_bajo=("revenue_bajo", "sum"),
                                               revenue_alto=("revenue_alto", "sum")).round(0).reset_index())
    configuration.show_table(summary)
    if len(universe.exam["by_horizon"]):
        doc(f"[{step_label}] el examen de la simulación (en cada uno de los últimos {configuration.ts_exam_origins} meses "
            f"cerrados se simula con los meses anteriores y se compara el total del mes con lo que convirtió; h = meses "
            f"tras el último conocido), con cada regla del nivel sobre los mismos orígenes: simula la de menor error "
            f"medio, {universe.level_rule}, y su error de cada h es la media banda de las filas a ese horizonte")
        by_rule = universe.exam["by_rule"]
        shown = (by_rule.assign(error_pct=100 * by_rule["error_usd"])
                 .pivot_table(index="regla_nivel", columns="h", values="error_pct", sort=False).round(1))
        shown.columns = [f"h{int(h)}" for h in shown.columns]
        shown["media"] = (100 * by_rule.groupby("regla_nivel", sort=False)["error_medio_regla"].first()).round(1)
        shown["elegida"] = by_rule.groupby("regla_nivel", sort=False)["elegida"].first()
        configuration.show_table(shown.reset_index())
    else:
        doc(f"[{step_label}] AVISO: no hay meses cerrados suficientes para examinar la simulación: sus filas van sin banda")
    if universe.fallback_regions:
        doc(f"[{step_label}] AVISO: {len(universe.fallback_regions)} regiones sin el mismo mes del año anterior o sin nivel "
            f"se proyectan por su cuota del total: {sorted(universe.fallback_regions)[:10]}")
    if len(universe.projected):
        doc(f"[{step_label}] las {configuration.top_regions_shown} regiones con más valor proyectado este año:")
        configuration.show_table(universe.projected.join(universe.levels_of_region, on=REGION_KEY)
                                 .groupby(universe.region_columns)
                                 .agg(meses=(period, "size"), unidades=("unidades", "sum"), valor=("valor", "sum"),
                                      nivel=("nivel", "first"), valor_medio=("valor_medio", "first"),
                                      por_cuota=("por_cuota", "max"))
                                 .reset_index().sort_values("valor", ascending=False).head(configuration.top_regions_shown))
    doc(f"[{step_label}] ══════════ FIN DEL UNIVERSO TIME_SERIES ══════════")


# ═══════════════════════════════════════════════════════════════════════════════════
# BLOCK 6 · THE BLOCK: run the step, keep the box
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_LABEL = "B6"
BLOCK_NAME = "TIME SERIES"
BLOCK_MISSION = ("what the retail-to-subscription universe still brings this year, the whole of next year and the "
                 "pipeline of next year it creates, by region, with a band")
BLOCK_READS = ("time_series_rows", "fine_table")
BLOCK_GIVES = ("time_series_universe",)


class TimeSeriesBlock:
    """BLOCK 6 · TIME SERIES (see the header of this module): its box is an attribute, in memory."""

    def __init__(self, configuration: Config):
        self.configuration = configuration
        self.time_series_universe = None
        self.seconds = {}

    def run(self, time_series_rows: pd.DataFrame, fine_table: pd.DataFrame) -> "TimeSeriesBlock":
        """The step of the block."""
        configuration = self.configuration
        configuration.log_block_start(BLOCK_LABEL, 6, BLOCK_NAME, BLOCK_MISSION)

        started = time.time()
        self.time_series_universe = build_the_time_series_universe(time_series_rows, fine_table, configuration)
        self.seconds["6.1 time_series_universe"] = time.time() - started
        return self

    def results(self) -> dict:
        """The results under the names the pipeline gives them."""
        return {name: getattr(self, name) for name in BLOCK_GIVES}
