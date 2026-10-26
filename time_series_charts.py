"""
time_series_charts.py — The time_series universe (retail to subscription) as a picture: the same chart twice.

  ts_1_2_historia    drawn in step 1.2, when the universe is separated: what converted month by month, before any
                     prediction. To see its evolution, its volume and how stable it is.
  ts_6_1_prediccion  drawn in step 6.1, when the universe is simulated: the same chart, with the rest of this year,
                     the whole of next year and the re-entry added. To see whether the prediction continues the
                     history or breaks it.

THE FOUR PANELS (the same in both charts, the same months on the x axis):
  1. revenue USD per month       the total and the regions of the coarsest level with the most revenue
                                 (top_regions_charted, the rest added as "resto"); in the prediction, the simulated
                                 months dashed with the band of the exam, and the expected renewals of the re-entry
  2. units per month             the total (the volume)
  3. value per unit              USD / units of the month: a jump here is a change of price or of mix
  4. the level                   units of the last ts_level_window_months months / the same months a year before:
                                 the number the projection multiplies by. A level far from 1 or jumping from one
                                 month to the next is a series the rule will follow badly

A chart never stops the run: if it cannot be drawn, the step says why and goes on.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

import matplotlib
import numpy as np
import pandas as pd

from config import Config, TS_PROJECTED, TS_REENTRY, TS_REGION_KEY, TS_SIMULATED, region_columns_of
from graficos import (AQUA, BLUE, CHART_FOLDER, INK, INK_MUTED, INK_SECONDARY, ORANGE, VIOLET, money_label, new_figure,
                      save, titles)


# ─── named constants ─────────────────────────────────────────────────────────────
MONTHS_PER_YEAR = 12
REGION_COLOURS = [ORANGE, AQUA, VIOLET, INK_MUTED]
HISTORY_CHART = "ts_1_2_historia"
PREDICTION_CHART = "ts_6_1_prediccion"
REST_OF_REGIONS = "resto"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TWO CHARTS
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_the_history_of_the_time_series(time_series_rows: pd.DataFrame, step_label: str, configuration: Config):
    """Step 1.2: what the universe converted in the closed months, by month. Returns the path, or None."""
    if time_series_rows is None or time_series_rows.empty:
        return None
    period = configuration.period_col
    current = configuration.calendar_boundaries()["current"]
    closed = time_series_rows[time_series_rows[period] < current]
    real = monthly_series(closed, configuration)
    if real.empty:
        return None
    stability = stability_of(real, configuration)
    subtitle = (f"Lo que convirtió cada mes cerrado, sin predicción. Últimos 12 meses: {money_label(stability['usd_12m'])}, "
                f"variación mes a mes típica {stability['variacion_tipica']:.0%}; nivel (últimos "
                f"{configuration.ts_level_window_months} meses frente al año anterior) entre {stability['nivel_min']:.2f} y "
                f"{stability['nivel_max']:.2f} en el último año.")
    path = chart_path(configuration, HISTORY_CHART)
    draw_the_four_panels(real, None, None, configuration, path,
                         "Retail a suscripción: la serie antes de predecir", subtitle)
    configuration.show_figure(step_label, "the time_series universe month by month (evolution, volume, stability)", path)
    return path


def draw_the_prediction_of_the_time_series(universe, step_label: str, configuration: Config):
    """Step 6.1: the same chart with the simulated months (rest of this year and next year) and the re-entry."""
    if universe is None or universe.table.empty:
        return None
    real_rows = (universe.closed_monthly
                 .rename(columns={"unidades": configuration.renewed_units_col, "valor": configuration.renewed_usd_col})
                 .join(universe.levels_of_region, on=TS_REGION_KEY))
    real = monthly_series(real_rows, configuration)
    table = universe.table
    simulated_rows = table[table["origen"].isin([TS_PROJECTED, TS_SIMULATED])]
    simulated = monthly_series(simulated_rows.rename(columns={"unidades": configuration.renewed_units_col,
                                                              "revenue": configuration.renewed_usd_col}), configuration)
    band = simulated_rows.groupby(configuration.period_col)[["revenue_bajo", "revenue_alto"]].sum()
    reentry = table[table["origen"] == TS_REENTRY].groupby(configuration.period_col)["revenue"].sum()
    first_error = float(universe.exam["by_horizon"]["error_usd"].iloc[0]) if len(universe.exam["by_horizon"]) else np.nan
    next_level = configuration.ts_next_year_level
    subtitle = (f"Real hasta {universe.last_closed}; discontinuo: lo simulado (regla del nivel elegida por el examen: "
                f"{universe.level_rule}, valor por unidad de los últimos {universe.auv_window} meses; resto de "
                f"{universe.current.year} y {universe.next_year} entero, nivel del año siguiente "
                f"{'el de la región' if next_level is None else f'{next_level:g}'}"
                f"{' = repite este año' if next_level == 1 else ''}), con la banda del examen "
                f"(error a 1 mes: {first_error:.0%}); punteado: la renovación esperada de la reentrada.")
    path = chart_path(configuration, PREDICTION_CHART)
    draw_the_four_panels(real, simulated, dict(band=band, reentry=reentry), configuration, path,
                         "Retail a suscripción: la serie con su predicción", subtitle)
    configuration.show_figure(step_label, "the time_series universe with its prediction (the same chart as step 1.2)", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════════
# THE SERIES BEHIND THE PANELS
# ═══════════════════════════════════════════════════════════════════════════════════

def monthly_series(rows: pd.DataFrame, configuration: Config) -> pd.DataFrame:
    """One row per month: units and USD of the total, and the USD of every region of the coarsest level."""
    period = configuration.period_col
    if rows.empty:
        return pd.DataFrame()
    region = region_columns_of(configuration)[0]
    grouped = rows.groupby(period)
    monthly = pd.DataFrame({"unidades": grouped[configuration.renewed_units_col].sum(),
                            "usd": grouped[configuration.renewed_usd_col].sum()})
    if region in rows.columns:
        by_region = rows.groupby([period, rows[region].fillna("(sin región)").astype(str)])[configuration.renewed_usd_col].sum()
        monthly = monthly.join(by_region.unstack(fill_value=0.0).add_prefix("usd_region_"))
    return monthly.sort_index()


def level_of_months(units: pd.Series, configuration: Config) -> pd.Series:
    """Per month: units of the last ts_level_window_months months (that one included) / the same months a year before."""
    every_month = units.reindex(pd.period_range(units.index.min(), units.index.max(), freq="M"), fill_value=0.0)
    window = configuration.ts_level_window_months
    recent = every_month.rolling(window).sum()
    before = recent.shift(MONTHS_PER_YEAR)
    return (recent / before.where(before > 0)).dropna()


def stability_of(real: pd.DataFrame, configuration: Config) -> dict:
    """The numbers of the subtitle: USD of the last 12 months, the typical month-to-month change of the USD, and the
    range of the level in the last year."""
    last_year = real.tail(MONTHS_PER_YEAR)
    change = last_year["usd"].pct_change().abs().replace([np.inf, -np.inf], np.nan).dropna()
    level = level_of_months(real["unidades"], configuration).tail(MONTHS_PER_YEAR)
    return dict(usd_12m=float(last_year["usd"].sum()),
                variacion_tipica=float(change.median()) if len(change) else np.nan,
                nivel_min=float(level.min()) if len(level) else np.nan,
                nivel_max=float(level.max()) if len(level) else np.nan)


# ═══════════════════════════════════════════════════════════════════════════════════
# THE DRAWING
# ═══════════════════════════════════════════════════════════════════════════════════

def draw_the_four_panels(real: pd.DataFrame, simulated, extras, configuration: Config, path: str, title: str,
                         subtitle: str) -> str:
    """Revenue (total and regions), units, value per unit and level; the simulated months dashed when given."""
    figure, axes = new_figure(11, 11, rows=4, columns=1, sharex=True)
    usd_axis, units_axis, auv_axis, level_axis = axes
    as_time = lambda months: [month.to_timestamp() for month in months]

    # 1 · revenue: the total and the regions with the most revenue
    usd_axis.plot(as_time(real.index), real["usd"], color=BLUE, linewidth=2.2, label="total")
    region_columns = [column for column in real.columns if column.startswith("usd_region_")]
    ranked = real[region_columns].sum().sort_values(ascending=False).index.tolist()
    shown = ranked[:configuration.top_regions_charted] if len(ranked) > 1 else []     # one region = the total
    for colour, column in zip(REGION_COLOURS, shown):
        usd_axis.plot(as_time(real.index), real[column], color=colour, linewidth=1.2,
                      label=column.replace("usd_region_", ""))
    if len(shown) and len(ranked) > len(shown):
        usd_axis.plot(as_time(real.index), real[ranked[len(shown):]].sum(axis=1), color=INK_MUTED, linewidth=1,
                      linestyle=":", label=REST_OF_REGIONS)
    if simulated is not None and len(simulated):
        joined = pd.concat([real.tail(1), simulated])            # the dashed line starts where the real one ends
        usd_axis.plot(as_time(joined.index), joined["usd"], color=BLUE, linewidth=2.2, linestyle="--", label="simulado")
        band = extras["band"].reindex(simulated.index)
        usd_axis.fill_between(as_time(simulated.index), band["revenue_bajo"], band["revenue_alto"], color=BLUE, alpha=0.15,
                              linewidth=0, label="banda del examen")
        if len(extras["reentry"]):
            usd_axis.plot(as_time(extras["reentry"].index), extras["reentry"], color=INK, linewidth=1.4, linestyle=":",
                          label="renovación esperada de la reentrada")
    usd_axis.set_ylabel("USD al mes", color=INK_SECONDARY)
    usd_axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: money_label(value)))
    usd_axis.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)

    # 2 · units, 3 · value per unit, 4 · the level: the total only
    whole = pd.concat([real, simulated]) if simulated is not None and len(simulated) else real
    whole = whole[~whole.index.duplicated(keep="first")]
    real_months = set(real.index)
    level = level_of_months(whole["unidades"], configuration)
    panels = [(units_axis, whole["unidades"], "unidades al mes", lambda value, _: f"{value:,.0f}"),
              (auv_axis, whole["usd"] / whole["unidades"].where(whole["unidades"] > 0), "USD por unidad",
               lambda value, _: f"${value:,.2f}"),
              (level_axis, level, f"nivel ({configuration.ts_level_window_months} meses / año anterior)",
               lambda value, _: f"{value:.2f}")]
    for axis, series, label, formatter in panels:
        observed = series[[month in real_months for month in series.index]]
        axis.plot(as_time(observed.index), observed, color=BLUE, linewidth=1.8)
        predicted = series[[month not in real_months for month in series.index]]
        if len(predicted) and len(observed):
            joined = pd.concat([observed.tail(1), predicted])
            axis.plot(as_time(joined.index), joined, color=BLUE, linewidth=1.8, linestyle="--")
        axis.set_ylabel(label, color=INK_SECONDARY)
        axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(formatter))
    level_axis.axhline(1.0, color=INK_SECONDARY, linewidth=0.8)
    current = configuration.calendar_boundaries()["current"].to_timestamp()
    for axis in axes:
        axis.axvline(current, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
    titles(figure, title, subtitle)
    return save(figure, path, top=0.93)


def chart_path(configuration: Config, name: str) -> str:
    folder = os.path.join(configuration.output_folder or ".", CHART_FOLDER)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{name}.png")


def draw_without_stopping(draw, step_label: str, configuration: Config):
    """A chart never stops the run: on any error the step says why the chart is missing and goes on."""
    try:
        return draw()
    except Exception as error:
        import matplotlib.pyplot as plt
        plt.close("all")
        configuration.logger.doc(f"[{step_label}] chart not drawn: {type(error).__name__}: {error}")
        return None
