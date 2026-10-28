"""
informe_forecast.py — THE FORECAST TOLD: one business question, one SQL and its charts.

MISSION: tell the forecast as a story, question by question, over the tables a run already wrote (sff_*). Nothing of
the framework runs again: every answer is a SQL query (SQL Server; it also runs on the SQLite of the synthetics) that
can be pasted in SSMS or Power BI, its result as a CSV, and one or more charts drawn from that same result.

THE STORY (the chapters, in the order it is told):
  1  the real data         what the extract brings, what other areas have too and can be triangulated
  2  what I simulate       the maturation curves, the renewal of what falls due to December, the acquisitions and the
                           retail to subscription
  3  what I built          the pipeline of next year by its four origins, without double counting
  4  the result            how the current year and the next one end, with their band
  5  how to break it down  pipeline → rate → uplift → money at any level with its bands; whether the band was honest;
                           what the customer states cost
  6  the real bands        how the bands of the series add up, learned from the exam
  7  the components        the pipeline audited component by component, month by month, from the real past to next
                           December, with the uplift of every component with and without price increases (its
                           charts are in English and every chart has its annex of numbers under it)
  8  next year             where our next year differs from the retention team's forecast and why: the rates, the
                           uplift and the end of the price increases, the price reference combinations, the
                           composition of the pipeline, the maturation of the states and the horizon
  9  monographs           one business question each: the answer first (in text, <number>_respuesta.md), then the
                           charts from the general to the detail. 9.1: how much next year changes because of the
                           acquisition of this year (the first renewal: volume, price at sale, rate and uplift)

GROUPING: by region (the first and the last level of region_columns_of, tr_regional_level_1 and tr_regional_level_3 in
production) with the subtotal of every first-level region (its finer level = 'TOTAL') and the TOTAL, as the final
table of step 8.6. Every column is a sum or a ratio of sums, so the rows of a finer level add up to their region.

THE BAND THAT CAN BE BROKEN DOWN (composition_band_query): the rows of one composition (the same rate) move together,
different compositions move apart: band = √(Σ per composition (Σ deviations of its rows)²); the error of the volume of
the simulated acquisitions (s7_2_error_captacion_usd) moves every acquisition row together: it is added linearly and
then in quadrature. The sum of the bands of the rows (the linear band) is shown as the worst case.

THE INTERFACE:
  from informe_forecast import report
  results = report(configuration)                    every question
  results = report(configuration, ["3.1", "4.1"])    some of them
  results["3.1"].table   results["3.1"].sql   results["3.1"].charts
  python informe_forecast.py parecido 2026-10        the like-real synthetic (or: sintetico; nothing = production)

WHAT IT WRITES (salida/informe/): <number>_<name>.sql, .csv and .png per question.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import math
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import inspect

from config import (CORE_RENEWED_UNITS_TO_PREFIX, CORE_RENEWED_USD_TO_PREFIX, TERM_DESTINATIONS, TABLE_BUSINESS_SUMMARY, TABLE_CORE, TABLE_FORECAST_MATURATION, TABLE_MATURATION_CURVE, Config, running_in_notebook,
                    write_frame_to_sql)
from graficos import (AQUA, BLUE, INK, INK_MUTED, INK_SECONDARY, NEUTRAL_FILL, ORANGE, VIOLET, money_label,
                      new_figure, save, titles)
from report_queries import table_in_sql, year_by_region_columns
from term_transitions import term_years

# ─── named constants ─────────────────────────────────────────────────────────────
REPORT_FOLDER = "informe"
TABLE_SERIES = "forecast_series"
TOTAL_LABEL = "TOTAL"
SMALL_MULTIPLE_COLUMNS = 3
YEAR_COLOURS = ["#c9c7bf", "#a9a69c", "#7f7c72", BLUE, ORANGE]      # older years in greys, the last two in colour
ORIGIN_COLOURS = {"real": BLUE, "renovacion_proyectada": AQUA, "retail_to_subscription": VIOLET,
                  "adquisicion_simulada": ORANGE}
MONTH_NAMES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CONTEXT: the names of the tables and columns of this run
# ═══════════════════════════════════════════════════════════════════════════════════

class ReportContext:
    """What every query needs: the table names, the region levels, the years, z, the customer states and the columns
    the core really has (a run before 26-oct-2026 has no s7_2_error_captacion_usd and no sff_curva_maduracion)."""

    def __init__(self, configuration: Config, acquisition_plan_units: Optional[float] = None,
                 product_column: str = "tr_product_level_1", retention_reference_file: Optional[str] = None):
        self.configuration = configuration
        self.retention_reference_file = retention_reference_file     # 8.1: None = forecast_retencion_2027.csv
        self.engine = configuration.sql_engine
        self.core = table_in_sql(configuration, TABLE_CORE)
        self.series = table_in_sql(configuration, TABLE_SERIES)
        self.maturation = table_in_sql(configuration, TABLE_FORECAST_MATURATION)
        self.curve = table_in_sql(configuration, TABLE_MATURATION_CURVE)
        region_levels = year_by_region_columns(configuration)
        self.region = region_levels[0]
        self.subregion = region_levels[-1] if len(region_levels) > 1 else None
        current = configuration.calendar_boundaries()["current"]
        self.current_month = str(current)
        self.current_year = current.year
        self.next_year = current.year + 1
        self.z = float(configuration.z)
        self.acquisition_plan_units = acquisition_plan_units
        inspector = inspect(self.engine)
        schema = configuration.sql_schema
        self.tables = set(inspector.get_table_names(schema=schema))
        core_name = f"{configuration.table_prefix}{TABLE_CORE}"
        self.core_columns = {column["name"] for column in inspector.get_columns(core_name, schema=schema)}
        self.product = product_column if product_column in self.core_columns else None
        self.states = [state for state in configuration.structural_timevarying_dims if state in self.core_columns]
        self.group_dims = [dim for dim in configuration.business_mandatory_dims if dim in self.core_columns]
        # one folder per database: the synthetics (SQLite files) do not overwrite each other's report
        database = self.engine.url.database if self.engine.url.drivername.startswith("sqlite") else None
        suffix = f"_{os.path.splitext(os.path.basename(database))[0]}" if database else ""
        self.output_folder = os.path.join(configuration.output_folder, REPORT_FOLDER + suffix)
        self.answered = {}                  # number → the table of every question answered (a chart may read another's)
        os.makedirs(self.output_folder, exist_ok=True)

    def has_table(self, table_name: str) -> bool:
        return f"{self.configuration.table_prefix}{table_name}" in self.tables

    def acquisition_filter(self, alias: str = "") -> str:
        """The rows of a 1-year acquisition: acquisition_column in acquisition_values (and the 1-year term)."""
        configuration = self.configuration
        prefix = f"{alias}." if alias else ""
        if not configuration.acquisition_column or not configuration.acquisition_values:
            return "1 = 0"
        values = ", ".join(sql_literal(value) for value in configuration.acquisition_values)
        column = f"{prefix}{configuration.acquisition_column}"
        numeric = all(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
                      for value in configuration.acquisition_values)
        if numeric:            # a core with the column as text can hold '1.0' and 1 at once: compared as numbers
            column = f"CAST({column} AS FLOAT)"
        condition = f"{column} IN ({values})"
        if configuration.term_column:
            condition += f" AND {prefix}{configuration.term_column} = {sql_literal(configuration.one_year_term_value)}"
        return condition

    def sale_month(self, alias: str = "") -> str:
        """The month an acquisition was sold: the month of its row − 12 ('2026-03' for a row of '2027-03'), as text."""
        prefix = f"{alias}." if alias else ""
        return f"CONCAT(CAST({prefix}forecast_year - 1 AS VARCHAR(4)), SUBSTRING({prefix}{self.configuration.period_col}, 5, 3))"


def sql_literal(value) -> str:
    """A value of the Config as a SQL literal: numbers as they are, text quoted."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


# ═══════════════════════════════════════════════════════════════════════════════════
# THE TWO SHAPES OF QUERY
# ═══════════════════════════════════════════════════════════════════════════════════

def with_subtotals(context: ReportContext, base_sql: str, measures: list, keys: list = (), ctes: list = (),
                   levels: tuple = None, comment: str = "") -> str:
    """base_sql: one SELECT whose columns are the levels (region and subregion, named as in the core), the keys
    (anio, mes…) and the values the measures add up. Gives one row per region × subregion × keys, then the subtotal
    of every region (subregion = 'TOTAL'), then the TOTAL; orden_total and orden_region keep that order in Power BI.
    ctes: other common table expressions the base reads, as (name, sql), written before it."""
    first, second = levels if levels is not None else (context.region, context.subregion)
    keys = list(keys)
    key_list = "".join(f", {key}" for key in keys)
    measures_sql = ",\n       ".join(measures)
    total_group = f"\nGROUP BY {', '.join(keys)}" if keys else ""
    order = ", ".join(str(position) for position in range(1, 5 + len(keys)))
    with_list = ",\n".join(f"{name} AS (\n{textwrap.indent(sql.strip(), '    ')}\n)" for name, sql in list(ctes) + [("base", base_sql)])
    header = f"-- {comment}\n" if comment else ""
    if second is None:
        return f"""{header}WITH {with_list}
SELECT 0 AS orden_total, {first}, 0 AS orden_region, {first} AS nivel_2{key_list},
       {measures_sql}
FROM base
GROUP BY {first}{key_list}
UNION ALL
SELECT 1, '{TOTAL_LABEL}', 2, '{TOTAL_LABEL}'{key_list},
       {measures_sql}
FROM base{total_group}
ORDER BY {order};"""
    return f"""{header}WITH {with_list}
SELECT 0 AS orden_total, {first}, 0 AS orden_region, {second}{key_list},
       {measures_sql}
FROM base
GROUP BY {first}, {second}{key_list}
UNION ALL
SELECT 0, {first}, 1, '{TOTAL_LABEL}'{key_list},
       {measures_sql}
FROM base
GROUP BY {first}{key_list}
UNION ALL
SELECT 1, '{TOTAL_LABEL}', 2, '{TOTAL_LABEL}'{key_list},
       {measures_sql}
FROM base{total_group}
ORDER BY {order};"""


# the values every row of composition_band_query carries (on top of the components of the question)
BAND_VALUES = {
    "dinero_usd": "COALESCE(n.forecast_renewed_USD, 0)",
    "desviacion_baja_usd": "COALESCE(n.forecast_renewed_USD, 0) - COALESCE(n.forecast_renewed_USD_low, n.forecast_renewed_USD, 0)",
    "desviacion_alta_usd": "COALESCE(n.forecast_renewed_USD_high, n.forecast_renewed_USD, 0) - COALESCE(n.forecast_renewed_USD, 0)",
    "pipeline_al_vencer_usd": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                              "THEN COALESCE(n.forecast_to_renew_USD_at_due, 0) ELSE 0 END",
    "unidades_al_vencer": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                          "THEN COALESCE(n.forecast_to_renew_units_at_due, 0) ELSE 0 END",
    "unidades_esperadas": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                          "THEN COALESCE(n.s7_2_esperado_unidades, 0) ELSE 0 END",
    "varianza_binomial": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                         "THEN COALESCE(n.forecast_to_renew_units_at_due, 0) * COALESCE(n.s7_2_tasa, 0) * (1 - COALESCE(n.s7_2_tasa, 0)) ELSE 0 END",
    "desviacion_baja_unidades": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                                "THEN COALESCE(n.forecast_to_renew_units_at_due, 0) * (COALESCE(n.s7_2_tasa, 0) - COALESCE(n.s7_2_tasa_baja, n.s7_2_tasa, 0)) ELSE 0 END",
    "desviacion_alta_unidades": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                                "THEN COALESCE(n.forecast_to_renew_units_at_due, 0) * (COALESCE(n.s7_2_tasa_alta, n.s7_2_tasa, 0) - COALESCE(n.s7_2_tasa, 0)) ELSE 0 END",
    "esperado_usd_pipeline": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                             "THEN COALESCE(n.s7_2_esperado_usd, 0) ELSE 0 END",
    "base_uplift_usd": "CASE WHEN n.forecast_status = 'forecast' AND n.forecast_universe = 'pipeline' "
                       "THEN COALESCE(n.s7_2_vencen_usd_al_vencer, 0) * COALESCE(n.s7_2_tasa, 0) ELSE 0 END",
    "vencen_unidades_reales": "CASE WHEN n.forecast_status = 'actual' AND n.forecast_universe = 'pipeline' "
                              "THEN COALESCE(n.forecast_to_renew_units, 0) ELSE 0 END",
    "renovadas_unidades_reales": "CASE WHEN n.forecast_status = 'actual' AND n.forecast_universe = 'pipeline' "
                                 "THEN COALESCE(n.forecast_renewed_units, 0) ELSE 0 END",
}


def composition_band_query(context: ReportContext, where: str, components: dict, keys: dict = None,
                           levels: tuple = None, comment: str = "") -> str:
    """The money of a set of rows of the core (where) by region, subregion and keys, with its subtotals, its components
    (name → SQL expression over the core, alias n) and the bands that can be broken down:
      dinero_usd ± banda by composition: √(Σ per composition (Σ deviations of its rows)² + (Σ error of acquisition)²)
      the linear band (every row wrong at once) as the worst case
      the rate of the pipeline rows (Σ expected / Σ due at due date) with its binomial band (z √(Σ n p (1 − p)) / Σ n)
      and the band of the framework (the deviations in units summed by composition), the uplift
      (Σ expected USD / Σ USD due at due date × rate) and the real rate of the closed months.
    The band of a region is computed at the region (the rows of a composition in two subregions add up first), never
    by adding the bands of its subregions."""
    first, second = levels if levels is not None else (context.region, context.subregion)
    keys = dict(keys or {})
    error_of_acquisition = ("COALESCE(n.s7_2_error_captacion_usd, 0)" if "s7_2_error_captacion_usd" in context.core_columns
                            else "0")
    composition = (f"CASE WHEN f.s2_2_stage3_id IS NOT NULL THEN CAST(f.s2_2_stage3_id AS VARCHAR(200)) "
                   f"WHEN n.s1_4_fs_id IS NOT NULL THEN CONCAT('fs|', n.s1_4_fs_id) "
                   f"ELSE CONCAT('ts|', n.{context.region}) END")
    values = {**components, **BAND_VALUES, "error_captacion_usd": error_of_acquisition}
    level_columns = [first] + ([second] if second else [])
    row_select = ",\n       ".join([f"n.{column} AS {column}" for column in level_columns]
                                   + [f"{expression} AS {name}" for name, expression in keys.items()]
                                   + [f"{composition} AS composicion"]
                                   + [f"{expression} AS {name}" for name, expression in values.items()])
    rows_sql = f"""SELECT {row_select}
FROM {context.core} AS n
LEFT JOIN {context.series} AS f ON f.s1_4_fs_id = n.s1_4_fs_id
WHERE {where}"""
    key_names = list(keys)

    def level_query(group_columns: list, labels: list) -> str:
        grouping = group_columns + key_names
        inner_group = ", ".join(grouping + ["composicion"])
        inner_sums = ",\n              ".join(f"SUM({name}) AS {name}" for name in values)
        outer_select = ",\n       ".join(labels + key_names + band_measures(context, list(components)))
        outer_group = f"\nGROUP BY {', '.join(grouping)}" if grouping else ""
        return f"""SELECT {outer_select}
FROM (SELECT {inner_group},
              {inner_sums}
      FROM filas
      GROUP BY {inner_group}) AS por_composicion{outer_group}"""

    if second:
        parts = [level_query([first, second], ["0 AS orden_total", first, "0 AS orden_region", second]),
                 level_query([first], ["0 AS orden_total", first, "1 AS orden_region", f"'{TOTAL_LABEL}' AS {second}"]),
                 level_query([], ["1 AS orden_total", f"'{TOTAL_LABEL}' AS {first}", "2 AS orden_region",
                                  f"'{TOTAL_LABEL}' AS {second}"])]
    else:
        parts = [level_query([first], ["0 AS orden_total", first, "0 AS orden_region", f"{first} AS nivel_2"]),
                 level_query([], ["1 AS orden_total", f"'{TOTAL_LABEL}' AS {first}", "2 AS orden_region",
                                  f"'{TOTAL_LABEL}' AS nivel_2"])]
    order = ", ".join(str(position) for position in range(1, 5 + len(key_names)))
    header = f"-- {comment}\n" if comment else ""
    return (f"{header}WITH filas AS (\n{textwrap.indent(rows_sql, '    ')}\n)\n"
            + "\nUNION ALL\n".join(parts) + f"\nORDER BY {order};")


def band_measures(context: ReportContext, components: list) -> list:
    """The columns of composition_band_query from the sums per composition (por_composicion)."""
    z = context.z
    band_low = "SQRT(SUM(desviacion_baja_usd * desviacion_baja_usd) + SUM(error_captacion_usd) * SUM(error_captacion_usd))"
    band_high = "SQRT(SUM(desviacion_alta_usd * desviacion_alta_usd) + SUM(error_captacion_usd) * SUM(error_captacion_usd))"
    real_rate = "SUM(renovadas_unidades_reales) / NULLIF(SUM(vencen_unidades_reales), 0)"
    expected_rate = "SUM(unidades_esperadas) / NULLIF(SUM(unidades_al_vencer), 0)"
    return ([f"SUM({name}) AS {name}" for name in components]
            + ["SUM(dinero_usd) AS dinero_usd",
               f"SUM(dinero_usd) - {band_low} AS banda_baja_usd",
               f"SUM(dinero_usd) + {band_high} AS banda_alta_usd",
               "SUM(dinero_usd) - SUM(desviacion_baja_usd) - SUM(error_captacion_usd) AS banda_lineal_baja_usd",
               "SUM(dinero_usd) + SUM(desviacion_alta_usd) + SUM(error_captacion_usd) AS banda_lineal_alta_usd",
               "SUM(error_captacion_usd) AS error_captacion_usd",
               "SUM(pipeline_al_vencer_usd) AS pipeline_al_vencer_usd",
               "SUM(unidades_al_vencer) AS unidades_al_vencer",
               f"{expected_rate} AS tasa_prevista",
               f"{expected_rate} - {z} * SQRT(SUM(varianza_binomial)) / NULLIF(SUM(unidades_al_vencer), 0) AS tasa_binomial_baja",
               f"{expected_rate} + {z} * SQRT(SUM(varianza_binomial)) / NULLIF(SUM(unidades_al_vencer), 0) AS tasa_binomial_alta",
               f"{expected_rate} - SQRT(SUM(desviacion_baja_unidades * desviacion_baja_unidades)) / NULLIF(SUM(unidades_al_vencer), 0) AS tasa_framework_baja",
               f"{expected_rate} + SQRT(SUM(desviacion_alta_unidades * desviacion_alta_unidades)) / NULLIF(SUM(unidades_al_vencer), 0) AS tasa_framework_alta",
               "SUM(esperado_usd_pipeline) / NULLIF(SUM(base_uplift_usd), 0) AS uplift",
               "SUM(vencen_unidades_reales) AS vencen_unidades_reales",
               f"{real_rate} AS tasa_real",
               f"{real_rate} - {z} * SQRT({real_rate} * (1 - {real_rate}) / NULLIF(SUM(vencen_unidades_reales), 0)) AS tasa_real_binomial_baja",
               f"{real_rate} + {z} * SQRT({real_rate} * (1 - {real_rate}) / NULLIF(SUM(vencen_unidades_reales), 0)) AS tasa_real_binomial_alta"])


def source_is(*sources) -> str:
    return "n.forecast_pipeline_source IN (" + ", ".join(f"'{source}'" for source in sources) + ")"


def money_of(condition: str, column: str = "forecast_renewed_USD") -> str:
    return f"CASE WHEN {condition} THEN COALESCE(n.{column}, 0) ELSE 0 END"


WITH_MONEY = "n.origen_fila NOT IN ('hueco', 'ts_sin_resultado')"


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 1 · THE REAL DATA
# ═══════════════════════════════════════════════════════════════════════════════════

def sql_1_1(context: ReportContext) -> str:
    """1.1 · the real pipeline due every year, as the extract brings it, and what really renewed."""
    region, subregion = context.region, context.subregion
    closed = "s1_3_rol <> 'proyeccion'"
    base = f"""SELECT {region}{f', {subregion}' if subregion else ''}, forecast_year AS anio,
       CASE WHEN {closed} THEN COALESCE(s1_1_vencen_usd, 0) ELSE 0 END AS vence_usd_meses_cerrados,
       CASE WHEN {closed} THEN 0 ELSE COALESCE(s1_1_vencen_usd, 0) END AS vence_usd_meses_que_quedan,
       COALESCE(s1_1_vencen_unidades, 0) AS vence_unidades,
       CASE WHEN {closed} THEN COALESCE(s1_3_vencen_unidades, 0) ELSE 0 END AS vence_unidades_meses_cerrados,
       CASE WHEN {closed} THEN COALESCE(s1_3_renovadas_unidades, 0) ELSE 0 END AS renovadas_unidades_real,
       CASE WHEN {closed} THEN COALESCE(s1_3_renovado_usd, 0) ELSE 0 END AS renovado_usd_real
FROM {context.core}
WHERE origen_fila = 'raw'"""
    measures = ["SUM(vence_usd_meses_cerrados) + SUM(vence_usd_meses_que_quedan) AS vence_usd",
                "SUM(vence_usd_meses_cerrados) AS vence_usd_meses_cerrados",
                "SUM(vence_usd_meses_que_quedan) AS vence_usd_meses_que_quedan",
                "SUM(vence_unidades) AS vence_unidades",
                "SUM(renovado_usd_real) AS renovado_usd_real",
                "SUM(renovadas_unidades_real) AS renovadas_unidades_real",
                "SUM(renovadas_unidades_real) / NULLIF(SUM(vence_unidades_meses_cerrados), 0) AS tasa_unidades_real"]
    return with_subtotals(context, base, measures, ["anio"],
                          comment="1.1 · la pipeline real que vence cada año (como viene en el extracto) y lo renovado real")


def sql_1_2(context: ReportContext) -> str:
    """1.2 · the real pipeline month by month over the whole extract."""
    period = context.configuration.period_col
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, forecast_year AS anio,
       SUBSTRING({period}, 6, 2) AS mes, {period} AS periodo,
       COALESCE(s1_1_vencen_usd, 0) AS vence_usd, COALESCE(s1_1_vencen_unidades, 0) AS vence_unidades
FROM {context.core}
WHERE origen_fila = 'raw'"""
    return with_subtotals(context, base, ["SUM(vence_usd) AS vence_usd", "SUM(vence_unidades) AS vence_unidades"],
                          ["anio", "mes", "periodo"], comment="1.2 · la pipeline real mes a mes (como viene en el extracto)")


def sql_1_3(context: ReportContext) -> str:
    """1.3 · the real pipeline of the months to come: raw, what the forecast uses after the calendar, what it wiped."""
    period = context.configuration.period_col
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, forecast_year AS anio, {period} AS periodo,
       COALESCE(s1_1_vencen_usd, 0) AS raw_usd, COALESCE(s1_3_vencen_usd, 0) AS usado_usd,
       COALESCE(s1_1_vencen_unidades, 0) AS raw_unidades, COALESCE(s1_3_vencen_unidades, 0) AS usado_unidades
FROM {context.core}
WHERE origen_fila = 'raw' AND s1_3_rol = 'proyeccion'"""
    measures = ["SUM(raw_usd) AS raw_usd", "SUM(usado_usd) AS usado_usd", "SUM(raw_usd) - SUM(usado_usd) AS borrado_usd",
                "SUM(raw_unidades) AS raw_unidades", "SUM(usado_unidades) AS usado_unidades",
                "SUM(raw_unidades) - SUM(usado_unidades) AS borrado_unidades"]
    return with_subtotals(context, base, measures, ["anio", "periodo"],
                          comment="1.3 · la pipeline real de los meses que vienen: raw, la que usa el forecast tras el calendario y la borrada")


def sql_1_4(context: ReportContext) -> str:
    """1.4 · the real renewal rate of every closed month, in units and USD, with its binomial band."""
    period, z = context.configuration.period_col, context.z
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, {period} AS periodo,
       COALESCE(s1_3_vencen_unidades, 0) AS vencen_unidades, COALESCE(s1_3_renovadas_unidades, 0) AS renovadas_unidades,
       COALESCE(s1_3_vencen_usd, 0) AS vence_usd, COALESCE(s1_3_renovado_usd, 0) AS renovado_usd
FROM {context.core}
WHERE origen_fila = 'raw' AND s1_3_rol <> 'proyeccion'"""
    rate = "SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)"
    measures = ["SUM(vencen_unidades) AS vencen_unidades", "SUM(renovadas_unidades) AS renovadas_unidades",
                "SUM(vence_usd) AS vence_usd", "SUM(renovado_usd) AS renovado_usd",
                f"{rate} AS tasa_unidades",
                f"{rate} - {z} * SQRT({rate} * (1 - {rate}) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_baja",
                f"{rate} + {z} * SQRT({rate} * (1 - {rate}) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_alta",
                "SUM(renovado_usd) / NULLIF(SUM(vence_usd), 0) AS tasa_usd"]
    return with_subtotals(context, base, measures, ["periodo"],
                          comment="1.4 · la tasa de renovación real de cada mes cerrado, con su banda binomial")


def acquisition_base(context: ReportContext, level_columns: list) -> str:
    """The real acquisitions sold before the current month, by month of sale (the month of the row − 12)."""
    period = context.configuration.period_col
    return f"""SELECT {', '.join(level_columns)}, forecast_year - 1 AS anio_venta, SUBSTRING({period}, 6, 2) AS mes,
       COALESCE(s1_1_vencen_unidades, 0) AS vendidas_unidades, COALESCE(s1_1_vencen_usd, 0) AS vendido_usd_como_pipeline
FROM {context.core}
WHERE origen_fila = 'raw' AND {context.acquisition_filter()}
  AND {context.sale_month()} < '{context.current_month}'"""


ACQUISITION_MEASURES = ["SUM(vendidas_unidades) AS vendidas_unidades",
                        "SUM(vendido_usd_como_pipeline) AS vendido_usd_como_pipeline"]


def sql_1_5(context: ReportContext) -> str:
    """1.5 · the real acquisitions by month of sale, by region."""
    levels = [context.region] + ([context.subregion] if context.subregion else [])
    return with_subtotals(context, acquisition_base(context, levels), ACQUISITION_MEASURES, ["anio_venta", "mes"],
                          comment="1.5 · la adquisición real por mes de venta (el mes de la fila − 12), en unidades y en "
                                  "el USD que vencerá como pipeline")


def sql_1_5b(context: ReportContext) -> Optional[str]:
    """1.5b · the real acquisitions by month of sale, by product."""
    if context.product is None:
        return None
    return with_subtotals(context, acquisition_base(context, [context.product]), ACQUISITION_MEASURES,
                          ["anio_venta", "mes"], levels=(context.product, None),
                          comment="1.5b · la adquisición real por mes de venta, por producto")


def sql_1_6(context: ReportContext) -> str:
    """1.6 · the real retail to subscription by month and region."""
    period = context.configuration.period_col
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, {period} AS periodo,
       COALESCE(forecast_renewed_USD, 0) AS ingresos_usd, COALESCE(forecast_renewed_units, 0) AS convertidas_unidades
FROM {context.core}
WHERE origen_fila = 'ts_real'"""
    return with_subtotals(context, base, ["SUM(ingresos_usd) AS ingresos_usd", "SUM(convertidas_unidades) AS convertidas_unidades"],
                          ["periodo"], comment="1.6 · el retail to subscription real por mes")


def sql_1_7(context: ReportContext) -> str:
    """1.7 · the customer states TODAY in the real pipeline still to fall due."""
    period = context.configuration.period_col
    state_columns = "".join(f",\n       CASE WHEN {state} = 1 THEN COALESCE(s1_3_vencen_unidades, 0) ELSE 0 END AS {state}_unidades"
                            for state in context.states)
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, {period} AS periodo,
       COALESCE(s1_3_vencen_unidades, 0) AS vencen_unidades{state_columns}
FROM {context.core}
WHERE origen_fila = 'raw' AND s1_3_rol = 'proyeccion' AND s1_3_vencen_unidades > 0"""
    measures = ["SUM(vencen_unidades) AS vencen_unidades"]
    for state in context.states:
        measures += [f"SUM({state}_unidades) AS {state}_unidades",
                     f"SUM({state}_unidades) / NULLIF(SUM(vencen_unidades), 0) AS {state}_parte"]
    return with_subtotals(context, base, measures, ["periodo"],
                          comment="1.7 · los estados del cliente HOY en la pipeline real que queda por vencer")


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 2 · WHAT I SIMULATE TO BUILD
# ═══════════════════════════════════════════════════════════════════════════════════

def sql_2_1(context: ReportContext) -> Optional[str]:
    """2.1 · the maturation curve of every dated state and its exam (sff_curva_maduracion, from 26-oct-2026)."""
    if not context.has_table(TABLE_MATURATION_CURVE):
        return None
    return f"""-- 2.1 · la curva de maduración de cada estado con fechas y su examen por antelación
SELECT estado, meses_antes, parte_que_entra, parte_visible, p_entrar, tope, se_usa, error_curva_pp, error_sin_curva_pp, cohortes
FROM {context.curve}
ORDER BY estado, meses_antes;"""


def sql_2_2(context: ReportContext) -> Optional[str]:
    """2.2 · how the states of what falls due will mature, month by month (sff_forecast_maduracion)."""
    if not context.has_table(TABLE_FORECAST_MATURATION):
        return None
    return f"""-- 2.2 · por mes: la parte en cada estado hoy, la esperada al vencer y lo que resta (ajuste_usd)
SELECT *
FROM {context.maturation}
ORDER BY {context.configuration.period_col};"""


def sql_2_3(context: ReportContext) -> str:
    """2.3 · what falls due from the current month to December, what we expect to renew, and what comes back as
    pipeline 12 months later (pipeline_proyectada, placed at the month it comes from)."""
    period = context.configuration.period_col
    levels = f"{context.region}{f', {context.subregion}' if context.subregion else ''}"
    base = f"""SELECT {levels}, {period} AS mes_que_vence,
       COALESCE(forecast_to_renew_USD, 0) AS vence_usd, COALESCE(forecast_renewed_USD, 0) AS renovado_esperado_usd,
       0 AS vuelve_como_pipeline_usd
FROM {context.core}
WHERE forecast_pipeline_source = 'pipeline_real_esperado' AND forecast_year = {context.current_year}
UNION ALL
SELECT {levels}, {context.sale_month()} AS mes_que_vence,
       0, 0, COALESCE(forecast_to_renew_USD, 0)
FROM {context.core}
WHERE origen_fila = 'proyectada'"""
    measures = ["SUM(vence_usd) AS vence_usd", "SUM(renovado_esperado_usd) AS renovado_esperado_usd",
                "SUM(vuelve_como_pipeline_usd) AS vuelve_como_pipeline_usd"]
    return with_subtotals(context, base, measures, ["mes_que_vence"],
                          comment="2.3 · lo que vence hasta diciembre, lo que esperamos que renueve y lo que vuelve como pipeline 12 meses después")


def sql_2_4(context: ReportContext) -> str:
    """2.4 · the acquisitions: real sales (last year and this year to date) and the simulated ones, by month of sale."""
    period = context.configuration.period_col
    levels = f"{context.region}{f', {context.subregion}' if context.subregion else ''}"
    base = f"""SELECT {levels}, forecast_year - 1 AS anio_venta, SUBSTRING({period}, 6, 2) AS mes,
       COALESCE(s1_1_vencen_unidades, 0) AS real_unidades, COALESCE(s1_1_vencen_usd, 0) AS real_usd,
       0 AS simulada_unidades, 0 AS simulada_usd
FROM {context.core}
WHERE origen_fila = 'raw' AND {context.acquisition_filter()}
  AND {context.sale_month()} < '{context.current_month}' AND forecast_year - 1 >= {context.current_year - 1}
UNION ALL
SELECT {levels}, forecast_year - 1, SUBSTRING({period}, 6, 2),
       0, 0, COALESCE(forecast_to_renew_units, 0), COALESCE(forecast_to_renew_USD, 0)
FROM {context.core}
WHERE origen_fila = 'simulada'"""
    measures = ["SUM(real_unidades) AS real_unidades", "SUM(simulada_unidades) AS simulada_unidades",
                "SUM(real_usd) AS real_usd_como_pipeline", "SUM(simulada_usd) AS simulada_usd_como_pipeline"]
    return with_subtotals(context, base, measures, ["anio_venta", "mes"],
                          comment="2.4 · la adquisición por mes de venta: real (el año anterior y este hasta hoy) y simulada")


def sql_2_5(context: ReportContext) -> str:
    """2.5 · the retail to subscription: real, projected and simulated, with its band."""
    period = context.configuration.period_col
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, {period} AS periodo,
       CASE WHEN origen_fila = 'ts_real' THEN COALESCE(forecast_renewed_USD, 0) ELSE 0 END AS real_usd,
       CASE WHEN origen_fila <> 'ts_real' THEN COALESCE(forecast_renewed_USD, 0) ELSE 0 END AS simulado_usd,
       CASE WHEN origen_fila <> 'ts_real' THEN COALESCE(forecast_renewed_USD_low, 0) ELSE 0 END AS simulado_bajo_usd,
       CASE WHEN origen_fila <> 'ts_real' THEN COALESCE(forecast_renewed_USD_high, 0) ELSE 0 END AS simulado_alto_usd
FROM {context.core}
WHERE origen_fila IN ('ts_real', 'ts_proyectado', 'ts_simulado')"""
    measures = ["SUM(real_usd) AS real_usd", "SUM(simulado_usd) AS simulado_usd",
                "SUM(simulado_bajo_usd) AS simulado_bajo_usd", "SUM(simulado_alto_usd) AS simulado_alto_usd"]
    return with_subtotals(context, base, measures, ["periodo"],
                          comment="2.5 · el retail to subscription: real, proyectado (lo que queda del año) y simulado, con su banda")


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 · WHAT I BUILT: THE PIPELINE OF NEXT YEAR
# ═══════════════════════════════════════════════════════════════════════════════════

PIPELINE_ORIGINS = {"real": "pipeline_real_esperado", "renovacion_proyectada": "pipeline_proyectada",
                    "retail_to_subscription": "ts_reentrada", "adquisicion_simulada": "pipeline_simulada"}


def sql_3_1(context: ReportContext) -> str:
    """3.1 · the pipeline of next year, month by month, by its four origins (USD and units)."""
    period = context.configuration.period_col
    columns = []
    for name, source in PIPELINE_ORIGINS.items():
        columns.append(f"CASE WHEN forecast_pipeline_source = '{source}' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS {name}_usd")
        columns.append(f"CASE WHEN forecast_pipeline_source = '{source}' THEN COALESCE(forecast_to_renew_units, 0) ELSE 0 END AS {name}_unidades")
    sources = ", ".join(f"'{source}'" for source in PIPELINE_ORIGINS.values())
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, {period} AS periodo,
       {(',' + chr(10) + '       ').join(columns)}
FROM {context.core}
WHERE forecast_year = {context.next_year} AND forecast_pipeline_source IN ({sources})"""
    measures = [f"SUM({name}_usd) AS {name}_usd" for name in PIPELINE_ORIGINS]
    measures.append(" + ".join(f"SUM({name}_usd)" for name in PIPELINE_ORIGINS) + " AS total_usd")
    measures += [f"SUM({name}_unidades) AS {name}_unidades" for name in PIPELINE_ORIGINS]
    measures.append(" + ".join(f"SUM({name}_unidades)" for name in PIPELINE_ORIGINS) + " AS total_unidades")
    return with_subtotals(context, base, measures, ["periodo"],
                          comment=f"3.1 · la pipeline de {context.next_year} por mes y por sus cuatro orígenes")


def sql_3_2(context: ReportContext) -> str:
    """3.2 · the double count: what the calendar wiped from next year against what replaces it."""
    period = context.configuration.period_col
    base = f"""SELECT {context.region}{f', {context.subregion}' if context.subregion else ''}, {period} AS periodo,
       CASE WHEN origen_fila = 'raw' THEN COALESCE(s1_1_vencen_usd, 0) - COALESCE(s1_3_vencen_usd, 0) ELSE 0 END AS borrado_usd,
       CASE WHEN origen_fila = 'proyectada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS renovacion_proyectada_usd,
       CASE WHEN origen_fila = 'simulada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS adquisicion_simulada_usd
FROM {context.core}
WHERE forecast_year = {context.next_year} AND origen_fila IN ('raw', 'proyectada', 'simulada')"""
    measures = ["SUM(borrado_usd) AS borrado_usd", "SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd",
                "SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd",
                "SUM(renovacion_proyectada_usd) + SUM(adquisicion_simulada_usd) - SUM(borrado_usd) AS sustituto_menos_borrado_usd"]
    return with_subtotals(context, base, measures, ["periodo"],
                          comment=f"3.2 · {context.next_year}: lo que el calendario borró frente a lo que lo sustituye (sin doble conteo)")


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 · THE RESULT
# ═══════════════════════════════════════════════════════════════════════════════════

def components_of_year(context: ReportContext, year: int) -> dict:
    """The parts of the money of a year: the current year has real and expected; the next one only expected."""
    if year == context.current_year:
        return {"renovado_real_usd": money_of(source_is("pipeline_renovado_real")),
                "esperado_pipeline_real_usd": money_of(source_is("pipeline_real_esperado")),
                "retail_to_subscription_real_usd": money_of(source_is("ts_real")),
                "retail_to_subscription_proyectado_usd": money_of(source_is("ts_proyectado"))}
    return {"renueva_pipeline_real_usd": money_of(source_is("pipeline_real_esperado")),
            "renovacion_proyectada_usd": money_of(source_is("pipeline_proyectada")),
            "adquisicion_simulada_usd": money_of(source_is("pipeline_simulada")),
            "retail_to_subscription_usd": money_of(source_is("ts_reentrada", "ts_simulado", "ts_proyectado"))}


def monthly_components() -> dict:
    return {"pipeline_usd": "CASE WHEN n.forecast_universe = 'pipeline' THEN COALESCE(n.forecast_to_renew_USD, 0) ELSE 0 END",
            "renovado_real_usd": money_of("n.forecast_universe = 'pipeline' AND n.forecast_status = 'actual'"),
            "renovado_previsto_usd": money_of("n.forecast_universe = 'pipeline' AND n.forecast_status = 'forecast'"),
            "retail_to_subscription_usd": money_of("n.forecast_universe = 'time_series'")}


def sql_4_1(context: ReportContext) -> str:
    year = context.current_year
    return composition_band_query(context, f"n.forecast_year = {year} AND {WITH_MONEY}", components_of_year(context, year),
                                  comment=f"4.1 · cómo acaba {year}: sus partes y la banda por composición")


def sql_4_2(context: ReportContext) -> str:
    year, period = context.current_year, context.configuration.period_col
    return composition_band_query(context, f"n.forecast_year = {year} AND {WITH_MONEY}", monthly_components(),
                                  keys={"periodo": f"n.{period}"}, comment=f"4.2 · {year} mes a mes: pipeline, renovado real y previsto, tasa y bandas")


def sql_4_3(context: ReportContext) -> str:
    year = context.next_year
    return composition_band_query(context, f"n.forecast_year = {year} AND {WITH_MONEY}", components_of_year(context, year),
                                  comment=f"4.3 · cómo acaba {year}: sus partes y la banda por composición")


def sql_4_4(context: ReportContext) -> str:
    year, period = context.next_year, context.configuration.period_col
    return composition_band_query(context, f"n.forecast_year = {year} AND {WITH_MONEY}", monthly_components(),
                                  keys={"periodo": f"n.{period}"}, comment=f"4.4 · {year} mes a mes: pipeline, renovado previsto, tasa y bandas")


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 · HOW TO BREAK IT DOWN
# ═══════════════════════════════════════════════════════════════════════════════════

def sql_5_1(context: ReportContext) -> str:
    year = context.next_year
    return composition_band_query(context, f"n.forecast_year = {year} AND n.forecast_status = 'forecast' AND {WITH_MONEY}", {},
                                  comment=f"5.1 · {year} por región: pipeline al vencer, tasa y sus bandas, uplift, dinero y su banda")


def sql_5_1b(context: ReportContext) -> str:
    year = context.current_year
    return composition_band_query(context, f"n.forecast_year = {year} AND n.forecast_status = 'forecast' AND {WITH_MONEY}", {},
                                  comment=f"5.1b · lo que queda de {year} por región: la misma cadena")


def sql_5_2(context: ReportContext) -> Optional[str]:
    if context.product is None:
        return None
    year = context.next_year
    second = "tr_product_level_2" if "tr_product_level_2" in context.core_columns else None
    return composition_band_query(context, f"n.forecast_year = {year} AND n.forecast_status = 'forecast' AND {WITH_MONEY}", {},
                                  levels=(context.product, second),
                                  comment=f"5.2 · {year} por producto: la misma cadena")


def sql_5_3(context: ReportContext) -> str:
    """5.3 · was the band honest? The exam of every forecast series, added up by region."""
    levels = f"{context.region}{f', {context.subregion}' if context.subregion else ''}"
    level_columns_m = f"m.{context.region}" + (f", m.{context.subregion}" if context.subregion else "")
    base = f"""SELECT {level_columns_m},
       COALESCE(f.s7_1_exam_predictions, 0) AS predicciones, COALESCE(f.s7_1_exam_in_band, 0) AS dentro_de_la_banda,
       COALESCE(f.s7_1_exam_abs_err_units, 0) AS error_absoluto_unidades, COALESCE(f.s7_1_exam_real_units, 0) AS reales_unidades,
       COALESCE(f.s7_1_raw_predictions, 0) AS predicciones_serie_sola, COALESCE(f.s7_1_raw_in_band, 0) AS dentro_serie_sola
FROM (SELECT DISTINCT s1_4_fs_id, {levels} FROM {context.core} WHERE origen_fila = 'raw' AND s1_4_fs_id IS NOT NULL) AS m
JOIN {context.series} AS f ON f.s1_4_fs_id = m.s1_4_fs_id"""
    measures = ["SUM(predicciones) AS predicciones", "SUM(dentro_de_la_banda) AS dentro_de_la_banda",
                "SUM(dentro_de_la_banda) / NULLIF(SUM(predicciones), 0) AS cobertura",
                "SUM(dentro_serie_sola) / NULLIF(SUM(predicciones_serie_sola), 0) AS cobertura_serie_sola",
                "SUM(error_absoluto_unidades) / NULLIF(SUM(reales_unidades), 0) AS wape"]
    return with_subtotals(context, base, measures,
                          comment="5.3 · ¿ha sido honesta la banda? La cobertura del examen y el WAPE, sumados por región")


def sql_5_4(context: ReportContext) -> str:
    """5.4 · what the customer states cost: what is still to mature takes off, and the potential (the units with a
    state at due date renewing as the neutral rows of their group and month)."""
    period = context.configuration.period_col
    levels = f"{context.region}{f', {context.subregion}' if context.subregion else ''}"
    states = context.states
    neutral = " AND ".join(f"COALESCE({state}, 0) = 0" for state in states) or "1 = 1"
    any_state = " OR ".join(f"s.{state} = 1" for state in states) or "1 = 0"
    group = context.group_dims
    join_on = " AND ".join([f"g.{dim} = s.{dim}" for dim in group] + [f"g.{period} = s.{period}"])
    ctes = [("neutra", f"""SELECT {', '.join(group + [period])},
       SUM(COALESCE(s7_2_esperado_unidades, 0)) / NULLIF(SUM(COALESCE(forecast_to_renew_units_at_due, 0)), 0) AS tasa_neutra
FROM {context.core}
WHERE forecast_status = 'forecast' AND forecast_universe = 'pipeline' AND {neutral}
GROUP BY {', '.join(group + [period])}""")]
    level_columns_s = f"s.{context.region}" + (f", s.{context.subregion}" if context.subregion else "")
    base = f"""SELECT {levels}, forecast_year AS anio,
       -COALESCE(forecast_renewed_USD_maturation_adj, 0) AS resta_lo_que_falta_por_madurar_usd, 0 AS potencial_usd
FROM {context.core}
WHERE forecast_status = 'forecast' AND forecast_year IN ({context.current_year}, {context.next_year})
UNION ALL
SELECT {level_columns_s}, s.forecast_year,
       0, COALESCE(s.s7_2_vencen_usd_al_vencer, 0) * COALESCE(s.s3_1_uplift, 1) * (g.tasa_neutra - COALESCE(s.s7_2_tasa, 0))
FROM {context.core} AS s
JOIN neutra AS g ON {join_on}
WHERE s.forecast_status = 'forecast' AND s.forecast_universe = 'pipeline' AND ({any_state})
  AND s.forecast_year IN ({context.current_year}, {context.next_year})"""
    measures = ["SUM(resta_lo_que_falta_por_madurar_usd) AS resta_lo_que_falta_por_madurar_usd",
                "SUM(potencial_usd) AS potencial_usd"]
    return with_subtotals(context, base, measures, ["anio"], ctes=ctes,
                          comment="5.4 · lo que cuestan los estados: lo que resta lo que falta por madurar y el potencial "
                                  "(las unidades con estado al vencer renovando como las neutras de su grupo y mes)")


# ═══════════════════════════════════════════════════════════════════════════════════
# THE CHARTS
# ═══════════════════════════════════════════════════════════════════════════════════

def total_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["orden_total"] == 1]


def region_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The subtotal of every first-level region (orden_region = 1), or its only rows when there is one level."""
    if (table["orden_region"] == 1).any():
        return table[table["orden_region"] == 1]
    return table[table["orden_total"] == 0]


def stability_text(values: pd.Series) -> str:
    """The coefficient of variation of a monthly series and its trend per year, as the chart writes them."""
    values = pd.Series(values, dtype=float).dropna()
    if len(values) < 3 or values.mean() == 0:
        return ""
    slope = np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0]
    return f"CV {100 * values.std() / values.mean():.0f} % · tendencia {100 * 12 * slope / values.mean():+.0f} %/año"


def money_axis(axis) -> None:
    axis.yaxis.set_major_formatter(lambda value, _: money_label(value))


def percent_axis(axis, decimals: int = 0) -> None:
    axis.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.{decimals}f} %")


def small_multiples(regions: list, height_per_row: float = 2.8, same_scale: bool = True):
    rows = max(1, math.ceil(len(regions) / SMALL_MULTIPLE_COLUMNS))
    figure, axes = new_figure(15, height_per_row * rows + 1.2, rows, SMALL_MULTIPLE_COLUMNS, sharey=same_scale)
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[len(regions):]:
        axis.set_visible(False)
    return figure, axes


def chart_path(context: ReportContext, number: str, name: str) -> str:
    return os.path.join(context.output_folder, f"{number.replace('.', '_')}_{name}.png")


def draw_1_1(table, context, number):
    total = total_rows(table).sort_values("anio")
    figure, axis = new_figure(11, 5)
    positions = np.arange(len(total))
    axis.bar(positions - 0.2, total["vence_usd_meses_cerrados"], 0.38, color=BLUE, label="vence en meses cerrados")
    axis.bar(positions - 0.2, total["vence_usd_meses_que_quedan"], 0.38, bottom=total["vence_usd_meses_cerrados"],
             color=BLUE, alpha=0.35, label="vence en los meses que quedan (lo que ya trae el extracto)")
    axis.bar(positions + 0.2, total["renovado_usd_real"], 0.38, color=AQUA, label="renovado real")
    for position, (_, row) in zip(positions, total.iterrows()):
        axis.annotate(money_label(row["vence_usd"]), (position - 0.2, row["vence_usd"]), ha="center", va="bottom",
                      fontsize=8.5, color=INK_SECONDARY)
        if row["renovado_usd_real"] > 0:
            axis.annotate(money_label(row["renovado_usd_real"]), (position + 0.2, row["renovado_usd_real"]), ha="center",
                          va="bottom", fontsize=8.5, color=INK_SECONDARY)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(int(year)) for year in total["anio"]])
    money_axis(axis)
    axis.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    titles(figure, f"{number} · La pipeline real que vence cada año y lo renovado",
           "Como viene en el extracto: sin estimar nada. La parte clara de los años que no han acabado es la pipeline que ya está en el extracto.")
    paths = [save(figure, chart_path(context, number, "pipeline_real_por_anio"))]

    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names)
    for axis, name in zip(axes, names):
        of_region = regions[regions[context.region] == name].sort_values("anio")
        positions = np.arange(len(of_region))
        axis.bar(positions - 0.2, of_region["vence_usd"], 0.38, color=BLUE, alpha=0.7)
        axis.bar(positions + 0.2, of_region["renovado_usd_real"], 0.38, color=AQUA)
        axis.set_xticks(positions)
        axis.set_xticklabels([str(int(year)) for year in of_region["anio"]], fontsize=8)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
        money_axis(axis)
    titles(figure, f"{number}a · La pipeline real por región (azul: vence; verde: renovado real)", "Misma escala en todos los paneles.")
    paths.append(save(figure, chart_path(context, number, "pipeline_real_por_region"), top=0.9))
    return paths


def lines_per_year(axis, frame: pd.DataFrame, value: str, current_month: str, year_column: str = "anio") -> None:
    """One line per year over the months of the year; the months not closed yet, dashed."""
    years = sorted(frame[year_column].unique())
    colours = YEAR_COLOURS[-len(years):] if len(years) <= len(YEAR_COLOURS) else [INK_MUTED] * (len(years) - 2) + YEAR_COLOURS[-2:]
    for year, colour in zip(years, colours):
        of_year = frame[frame[year_column] == year].sort_values("mes")
        months = of_year["mes"].astype(int).to_numpy()
        values = of_year[value].to_numpy(dtype=float)
        if "periodo" in of_year:
            closed = (of_year["periodo"].astype(str) < current_month).to_numpy()
        else:
            closed = np.ones(len(of_year), dtype=bool)
        axis.plot(months[closed], values[closed], color=colour, linewidth=2, marker="o", markersize=3, label=str(int(year)))
        if (~closed).any():
            start = max(0, int(np.argmax(~closed)) - 1) if closed.any() else 0
            axis.plot(months[start:], values[start:], color=colour, linewidth=1.6, linestyle="--")
    axis.set_xticks(range(1, 13))
    axis.set_xticklabels(MONTH_NAMES, fontsize=8)
    axis.set_ylim(bottom=0)


def draw_1_2(table, context, number):
    total = total_rows(table)
    figure, axis = new_figure(11, 5)
    lines_per_year(axis, total, "vence_usd", context.current_month)
    money_axis(axis)
    axis.legend(fontsize=8.5, frameon=False, ncol=6, loc="upper left")
    titles(figure, f"{number} · La pipeline real mes a mes, un año por línea",
           "Discontinua: los meses que no han cerrado (lo que el extracto ya trae; los lejanos aún se llenarán). "
           "Si noviembre es un pico, debe repetirse cada año.")
    paths = [save(figure, chart_path(context, number, "pipeline_real_mes_a_mes"))]
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names)
    for axis, name in zip(axes, names):
        lines_per_year(axis, regions[regions[context.region] == name], "vence_usd", context.current_month)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
        money_axis(axis)
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    titles(figure, f"{number}a · La pipeline real mes a mes por región", "Misma escala en todos los paneles.")
    paths.append(save(figure, chart_path(context, number, "pipeline_real_por_region"), top=0.9))
    return paths


def draw_1_3(table, context, number):
    total = total_rows(table).sort_values("periodo")
    figure, axis = new_figure(13, 5)
    positions = np.arange(len(total))
    axis.bar(positions, total["usado_usd"], color=BLUE, label="la usa el forecast (tras el calendario)")
    axis.bar(positions, total["borrado_usd"], bottom=total["usado_usd"], color=BLUE, alpha=0.25, hatch="///",
             edgecolor=BLUE, linewidth=0, label="borrada: vendida o renovada desde el mes en curso (se proyecta entera)")
    axis.set_xticks(positions)
    axis.set_xticklabels(total["periodo"].astype(str), rotation=60, fontsize=7.5)
    money_axis(axis)
    axis.legend(fontsize=8.5, frameon=False, loc="upper right")
    by_year = total.groupby("anio")[["raw_usd", "usado_usd", "borrado_usd"]].sum()
    subtitle = " · ".join(f"{int(year)}: {money_label(row['raw_usd'])} en el extracto, {money_label(row['usado_usd'])} usados"
                          for year, row in by_year.iterrows())
    titles(figure, f"{number} · La pipeline real que hay hoy para los meses que vienen", subtitle)
    return [save(figure, chart_path(context, number, "pipeline_real_que_viene"))]


def draw_1_4(table, context, number):
    total = total_rows(table).sort_values("periodo")
    figure, axis = new_figure(13, 4.6)
    positions = np.arange(len(total))
    axis.fill_between(positions, total["tasa_binomial_baja"], total["tasa_binomial_alta"], color=BLUE, alpha=0.18,
                      linewidth=0, label=f"banda binomial (z = {context.z:g})")
    axis.plot(positions, total["tasa_unidades"], color=BLUE, linewidth=2, label="tasa en unidades")
    axis.plot(positions, total["tasa_usd"], color=AQUA, linewidth=1.6, linestyle="--", label="tasa en dólares")
    step = max(1, len(total) // 18)
    axis.set_xticks(positions[::step])
    axis.set_xticklabels(total["periodo"].astype(str).to_numpy()[::step], rotation=60, fontsize=7.5)
    axis.set_ylim(0, 1)
    percent_axis(axis)
    axis.legend(fontsize=8.5, frameon=False, loc="lower left")
    titles(figure, f"{number} · La tasa de renovación real, mes cerrado a mes cerrado",
           f"Eje de 0 a 100 %. Unidades: {stability_text(total['tasa_unidades'])}. La banda binomial es lo que oscilaría "
           f"solo por el número de licencias.")
    paths = [save(figure, chart_path(context, number, "tasa_real"))]
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names)
    for axis, name in zip(axes, names):
        of_region = regions[regions[context.region] == name].sort_values("periodo")
        positions = np.arange(len(of_region))
        axis.fill_between(positions, of_region["tasa_binomial_baja"], of_region["tasa_binomial_alta"], color=BLUE, alpha=0.18, linewidth=0)
        axis.plot(positions, of_region["tasa_unidades"], color=BLUE, linewidth=1.8)
        axis.set_ylim(0, 1)
        percent_axis(axis)
        axis.set_xticks([])
        axis.set_title(f"{name}\n{stability_text(of_region['tasa_unidades'])}", fontsize=9, color=INK, loc="left")
    titles(figure, f"{number}a · La tasa real por región, con su banda binomial", "Eje de 0 a 100 % en todos los paneles.")
    paths.append(save(figure, chart_path(context, number, "tasa_real_por_region"), top=0.9))
    return paths


def draw_1_5(table, context, number, level_column=None, name="adquisicion_real"):
    level_column = level_column or context.region
    total = total_rows(table)
    figure, axis = new_figure(11, 5)
    lines_per_year(axis, total.rename(columns={"anio_venta": "anio"}), "vendidas_unidades", context.current_month)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    axis.legend(fontsize=8.5, frameon=False, ncol=6, loc="upper left")
    by_year = total.groupby("anio_venta")["vendidas_unidades"].sum()
    titles(figure, f"{number} · La adquisición real por mes de venta, un año por línea (unidades)",
           " · ".join(f"{int(year)}: {value:,.0f}" for year, value in by_year.items()) + " unidades (el año en curso, hasta el último mes cerrado)")
    paths = [save(figure, chart_path(context, number, name))]
    regions = region_rows(table)
    names = list(regions[level_column].drop_duplicates())
    figure, axes = small_multiples(names)
    for axis, region in zip(axes, names):
        of_region = regions[regions[level_column] == region].rename(columns={"anio_venta": "anio"})
        lines_per_year(axis, of_region, "vendidas_unidades", context.current_month)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
        monthly = of_region.sort_values(["anio", "mes"])["vendidas_unidades"]
        axis.set_title(f"{region}\n{stability_text(monthly)}", fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    titles(figure, f"{number}{'a' if level_column == context.region else ''} · La adquisición real por "
                   f"{'región' if level_column == context.region else 'producto'} (unidades)", "Misma escala en todos los paneles.")
    paths.append(save(figure, chart_path(context, number, f"{name}_por_nivel"), top=0.9))
    return paths


def draw_1_5b(table, context, number):
    return draw_1_5(table, context, number, level_column=context.product, name="adquisicion_real_por_producto")


def draw_1_6(table, context, number):
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names)
    for axis, name in zip(axes, names):
        of_region = regions[regions[context.region] == name].sort_values("periodo")
        positions = np.arange(len(of_region))
        axis.plot(positions, of_region["ingresos_usd"], color=VIOLET, linewidth=1.8)
        axis.set_ylim(bottom=0)
        money_axis(axis)
        step = max(1, len(of_region) // 6)
        axis.set_xticks(positions[::step])
        axis.set_xticklabels(of_region["periodo"].astype(str).to_numpy()[::step], fontsize=7, rotation=45)
        axis.set_title(f"{name}\n{stability_text(of_region['ingresos_usd'])}", fontsize=9, color=INK, loc="left")
    total = total_rows(table).sort_values("periodo")
    titles(figure, f"{number} · El retail to subscription real por región, mes a mes",
           f"Eje desde 0 y misma escala. Total: {stability_text(total['ingresos_usd'])}.")
    return [save(figure, chart_path(context, number, "retail_to_subscription_real"), top=0.9)]


def draw_1_7(table, context, number):
    total = total_rows(table).sort_values("periodo")
    figure, axis = new_figure(13, 4.6)
    positions = np.arange(len(total))
    width = 0.8 / max(1, len(context.states))
    colours = [VIOLET, ORANGE, INK_MUTED, AQUA]
    for index, state in enumerate(context.states):
        axis.bar(positions + (index - (len(context.states) - 1) / 2) * width, total[f"{state}_parte"], width,
                 color=colours[index % len(colours)], label=state)
    axis.set_xticks(positions)
    axis.set_xticklabels(total["periodo"].astype(str), rotation=60, fontsize=7.5)
    percent_axis(axis)
    axis.legend(fontsize=8.5, frameon=False)
    titles(figure, f"{number} · Los estados del cliente HOY en lo que queda por vencer (parte de las unidades)",
           "Lo lejano lleva menos estado porque aún no ha madurado: el capítulo 2 dice cuánto se espera al vencer.")
    return [save(figure, chart_path(context, number, "estados_hoy"))]


def draw_2_1(table, context, number):
    states = list(table["estado"].drop_duplicates())
    figure, axes = new_figure(14, 4.4 * len(states), len(states), 2)
    axes = np.atleast_2d(axes)
    for row, state in enumerate(states):
        of_state = table[table["estado"] == state].sort_values("meses_antes")
        months = int(of_state["meses_antes"].max())
        axis = axes[row, 0]
        axis.bar(of_state["meses_antes"], 100 * of_state["parte_que_entra"], color=VIOLET, alpha=0.35, width=0.8,
                 label="parte del estado final que se puso k meses antes")
        axis.plot(of_state["meses_antes"], 100 * of_state["parte_visible"], color=VIOLET, linewidth=2.2, marker="o",
                  markersize=3.5, label="parte del final ya visible h meses antes")
        axis.set_xticks(range(months + 1))
        axis.set_xticklabels([f"{k}+" if k == months else str(k) for k in range(months + 1)], fontsize=8)
        axis.invert_xaxis()
        axis.set_ylim(0, 130)
        axis.set_yticks(range(0, 101, 20))
        used = int(of_state["se_usa"].iloc[0]) == 1
        axis.set_title(f"{state.upper()} · cohortes {of_state['cohortes'].iloc[0]} · acaba en él el "
                       f"{100 * of_state['tope'].iloc[0]:.1f} %" + ("" if used else " · LA CURVA NO SE USA"),
                       fontsize=10, color=INK if used else ORANGE, loc="left")
        axis.set_xlabel("meses antes de vencer", color=INK_SECONDARY)
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")
        axis = axes[row, 1]
        examined = of_state.dropna(subset=["error_curva_pp"])
        positions = np.arange(len(examined))
        bars = axis.bar(positions - 0.18, examined["error_curva_pp"], 0.36, color=VIOLET, label="con la curva")
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in examined["error_curva_pp"]], fontsize=8, padding=2)
        bars = axis.bar(positions + 0.18, examined["error_sin_curva_pp"], 0.36, color=NEUTRAL_FILL, label="sin curva (lo visto entonces)")
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in examined["error_sin_curva_pp"]], fontsize=8, padding=2)
        axis.axhline(context.configuration.maturation_curve_max_error_pp, color=ORANGE, linestyle=":", linewidth=1,
                     label="umbral para usar la curva")
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{int(h)} {'mes' if int(h) == 1 else 'meses'}" for h in examined["meses_antes"]])
        axis.set_ylabel("error medio de la cuota final (pp)", color=INK_SECONDARY)
        axis.set_title("¿cuánto acierta según la antelación?", fontsize=10, color=INK, loc="left")
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    titles(figure, f"{number} · Cómo maduran los estados del cliente y cuánto acierta la curva",
           "Suponemos que las cohortes futuras maduran como estas: es la base del ajuste por estados del forecast.")
    return [save(figure, chart_path(context, number, "curvas_de_maduracion"), top=0.9 if len(states) > 1 else 0.84)]


def draw_2_2(table, context, number):
    period = context.configuration.period_col
    states = [column[:-4] for column in table.columns if column.endswith("_hoy") and column != "neutro_hoy"]
    table = table.sort_values(period)
    figure, axes = new_figure(14, 3.2 * len(states) + 3, len(states) + 1, 1)
    axes = np.atleast_1d(axes)
    positions = np.arange(len(table))
    labels = table[period].astype(str)
    for axis, state in zip(axes, states):
        today, expected = table[f"{state}_hoy"], table[f"{state}_esperado"]
        axis.bar(positions, today, color=VIOLET, label="en el estado hoy")
        axis.bar(positions, (expected - today).clip(lower=0), bottom=today, color=VIOLET, alpha=0.3,
                 label="lo que se espera añadir hasta vencer")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=60, fontsize=7)
        percent_axis(axis)
        axis.set_title(state, fontsize=10, color=INK, loc="left")
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    axis = axes[-1]
    axis.bar(positions, table["ajuste_usd"], color=ORANGE)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=60, fontsize=7)
    money_axis(axis)
    axis.set_title(f"lo que resta al forecast lo que falta por madurar: {money_label(table['ajuste_usd'].sum())} en total",
                   fontsize=10, color=INK, loc="left")
    titles(figure, f"{number} · Cuánto madurarán los estados de lo que vence, mes a mes", "Parte de las unidades que vencen cada mes.")
    return [save(figure, chart_path(context, number, "maduracion_por_mes"), top=0.93)]


def draw_2_3(table, context, number):
    total = total_rows(table).sort_values("mes_que_vence")
    total = total[(total["vence_usd"] > 0) | (total["vuelve_como_pipeline_usd"] > 0)]
    figure, axis = new_figure(11, 5)
    positions = np.arange(len(total))
    for offset, column, colour, label in ((-0.27, "vence_usd", BLUE, "vence (pipeline real)"),
                                          (0, "renovado_esperado_usd", AQUA, "renovado esperado"),
                                          (0.27, "vuelve_como_pipeline_usd", VIOLET, "vuelve a vencer 12 meses después (1 año)")):
        bars = axis.bar(positions + offset, total[column], 0.26, color=colour, label=label)
        axis.bar_label(bars, labels=[money_label(value) for value in total[column]], fontsize=7.5, padding=2)
    axis.set_xticks(positions)
    axis.set_xticklabels(total["mes_que_vence"].astype(str))
    money_axis(axis)
    axis.legend(fontsize=8.5, frameon=False)
    titles(figure, f"{number} · Lo que vence hasta diciembre, lo que renovará y lo que vuelve como pipeline",
           "Solo las licencias de 1 año vuelven a vencer el mismo mes del año siguiente, con el valor con que renuevan.")
    return [save(figure, chart_path(context, number, "renovacion_proyectada"))]


def draw_2_4(table, context, number):
    total = total_rows(table)
    previous, current = context.current_year - 1, context.current_year
    figure, axis = new_figure(12, 5)
    months = range(1, 13)
    real = total.pivot_table(index="mes", columns="anio_venta", values="real_unidades", aggfunc="sum").reindex(
        [f"{month:02d}" for month in months]).fillna(0.0)
    simulated = total.pivot_table(index="mes", columns="anio_venta", values="simulada_unidades", aggfunc="sum").reindex(
        [f"{month:02d}" for month in months]).fillna(0.0)
    positions = np.arange(12)
    last_year = real[previous] if previous in real else pd.Series(0.0, index=real.index)
    this_year = real[current] if current in real else pd.Series(0.0, index=real.index)
    simulated_this = simulated[current] if current in simulated else pd.Series(0.0, index=real.index)
    axis.bar(positions - 0.2, last_year, 0.38, color=NEUTRAL_FILL, label=f"{previous} real")
    axis.bar(positions + 0.2, this_year, 0.38, color=BLUE, label=f"{current} real")
    axis.bar(positions + 0.2, simulated_this, 0.38, bottom=this_year, color=ORANGE, alpha=0.6, label=f"{current} simulado")
    axis.set_xticks(positions)
    axis.set_xticklabels(MONTH_NAMES)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    axis.legend(fontsize=8.5, frameon=False)
    year_total = this_year.sum() + simulated_this.sum()
    subtitle = (f"{current}: {this_year.sum():,.0f} vendidas + {simulated_this.sum():,.0f} simuladas = {year_total:,.0f} unidades · "
                f"{previous}: {last_year.sum():,.0f} ({100 * (year_total / last_year.sum() - 1):+.1f} %)" if last_year.sum() else "")
    if context.acquisition_plan_units:
        axis.set_title(f"previsión del equipo de adquisición: {context.acquisition_plan_units:,.0f} unidades "
                       f"({100 * (year_total / context.acquisition_plan_units - 1):+.1f} % la nuestra)", fontsize=10,
                       color=INK, loc="left")
    titles(figure, f"{number} · La adquisición simulada frente a la real (unidades por mes de venta)", subtitle)
    return [save(figure, chart_path(context, number, "adquisicion_simulada"))]


def draw_2_5(table, context, number):
    def one(axis, frame):
        frame = frame.sort_values("periodo")
        positions = np.arange(len(frame))
        simulated = frame["simulado_usd"] > 0
        axis.plot(positions[~simulated.to_numpy()], frame.loc[~simulated, "real_usd"], color=VIOLET, linewidth=1.8)
        axis.plot(positions[simulated.to_numpy()], frame.loc[simulated, "simulado_usd"], color=VIOLET, linewidth=1.8, linestyle="--")
        axis.fill_between(positions[simulated.to_numpy()], frame.loc[simulated, "simulado_bajo_usd"],
                          frame.loc[simulated, "simulado_alto_usd"], color=VIOLET, alpha=0.15, linewidth=0)
        axis.set_ylim(bottom=0)
        money_axis(axis)
        step = max(1, len(frame) // 8)
        axis.set_xticks(positions[::step])
        axis.set_xticklabels(frame["periodo"].astype(str).to_numpy()[::step], fontsize=7, rotation=45)
    figure, axis = new_figure(12, 4.5)
    one(axis, total_rows(table))
    titles(figure, f"{number} · El retail to subscription: real (sólido) y simulado (discontinuo, con su banda)", "Eje desde 0.")
    paths = [save(figure, chart_path(context, number, "retail_to_subscription_simulado"))]
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names)
    for axis, name in zip(axes, names):
        one(axis, regions[regions[context.region] == name])
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
    titles(figure, f"{number}a · El retail to subscription por región", "Misma escala en todos los paneles.")
    paths.append(save(figure, chart_path(context, number, "retail_to_subscription_por_region"), top=0.9))
    return paths


def draw_3_1(table, context, number):
    total = total_rows(table).sort_values("periodo")
    figure, axes = new_figure(14, 8, 2, 1, gridspec_kw={"height_ratios": [3, 2]})
    positions = np.arange(len(total))
    width = 0.2
    for index, (name, colour) in enumerate(ORIGIN_COLOURS.items()):
        axes[0].bar(positions + (index - 1.5) * width, total[f"{name}_usd"], width, color=colour, label=name.replace("_", " "))
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(total["periodo"].astype(str))
    money_axis(axes[0])
    axes[0].legend(fontsize=8.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    bars = axes[1].bar(positions, total["total_usd"], color=INK_MUTED)
    axes[1].bar_label(bars, labels=[money_label(value) for value in total["total_usd"]], fontsize=8, padding=2)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(total["periodo"].astype(str))
    money_axis(axes[1])
    axes[1].set_title(f"el total por mes: {money_label(total['total_usd'].sum())} en el año", fontsize=10, color=INK, loc="left")
    shares = {name: total[f"{name}_usd"].sum() / total["total_usd"].sum() for name in ORIGIN_COLOURS}
    titles(figure, f"{number} · La pipeline de {context.next_year} por sus cuatro orígenes",
           " · ".join(f"{name.replace('_', ' ')} {100 * share:.0f} %" for name, share in shares.items()))
    return [save(figure, chart_path(context, number, "pipeline_por_origen"), top=0.9)]


def draw_3_2(table, context, number):
    total = total_rows(table).sort_values("periodo")
    total = total[(total["borrado_usd"] != 0) | (total["renovacion_proyectada_usd"] != 0) | (total["adquisicion_simulada_usd"] != 0)]
    figure, axis = new_figure(11, 5)
    positions = np.arange(len(total))
    axis.bar(positions - 0.2, total["borrado_usd"], 0.38, color=BLUE, alpha=0.3, hatch="///", edgecolor=BLUE, linewidth=0,
             label="borrado: lo que el extracto ya traía de este mes")
    axis.bar(positions + 0.2, total["renovacion_proyectada_usd"], 0.38, color=AQUA, label="renovación proyectada")
    axis.bar(positions + 0.2, total["adquisicion_simulada_usd"], 0.38, bottom=total["renovacion_proyectada_usd"], color=ORANGE,
             label="adquisición simulada")
    for position, (_, row) in zip(positions, total.iterrows()):
        top = max(row["borrado_usd"], row["renovacion_proyectada_usd"] + row["adquisicion_simulada_usd"])
        axis.annotate(f"{'+' if row['sustituto_menos_borrado_usd'] >= 0 else ''}{money_label(row['sustituto_menos_borrado_usd'])}",
                      (position, top), ha="center", va="bottom", fontsize=9,
                      color=INK if row["sustituto_menos_borrado_usd"] >= 0 else ORANGE)
    axis.set_xticks(positions)
    axis.set_xticklabels(total["periodo"].astype(str))
    money_axis(axis)
    axis.legend(fontsize=8.5, frameon=False)
    short = total[total["sustituto_menos_borrado_usd"] < 0]
    verdict = ("lo que sustituye a lo borrado es mayor en todos los meses: no hay doble conteo y se proyecta el mes entero"
               if short.empty else f"en {', '.join(short['periodo'].astype(str))} lo proyectado es MENOR que lo que ya traía el extracto: revisar")
    titles(figure, f"{number} · Sin doble conteo: lo borrado frente a lo que lo sustituye", verdict)
    return [save(figure, chart_path(context, number, "doble_conteo"))]


PART_COLOURS = {"renovado_real_usd": BLUE, "esperado_pipeline_real_usd": "#8ab4ea", "renueva_pipeline_real_usd": BLUE,
                "renovacion_proyectada_usd": AQUA, "adquisicion_simulada_usd": ORANGE,
                "retail_to_subscription_usd": VIOLET, "retail_to_subscription_real_usd": VIOLET,
                "retail_to_subscription_proyectado_usd": "#a99ee8"}


def waterfall(axis, parts: dict, total: float, low: float, high: float) -> None:
    """A bridge: every part on top of the previous ones, then the total with its band as a whisker."""
    running = 0.0
    labels = []
    parts = {label: value for label, value in parts.items() if abs(value) > 0}      # a part this run does not have: out
    for index, (label, value) in enumerate(parts.items()):
        axis.bar(index, value, bottom=running, color=PART_COLOURS.get(label, INK_MUTED), width=0.6)
        axis.annotate(money_label(value), (index, running + value), ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        running += value
        labels.append(label.replace("_usd", "").replace("_", " "))
    position = len(parts)
    axis.bar(position, total, color=INK_MUTED, width=0.6)
    axis.errorbar(position, total, yerr=[[total - low], [high - total]], color=INK, capsize=6, linewidth=1.4)
    axis.annotate(f"{money_label(total)}\n[{money_label(low)} – {money_label(high)}]", (position, high), ha="center",
                  va="bottom", fontsize=9, color=INK)
    axis.set_xticks(range(len(parts) + 1))
    axis.set_xticklabels(labels + ["total"], fontsize=8.5)
    money_axis(axis)


def point_and_whisker(axis, frame: pd.DataFrame, label_column: str) -> None:
    positions = np.arange(len(frame))
    axis.errorbar(positions, frame["dinero_usd"], yerr=[frame["dinero_usd"] - frame["banda_lineal_baja_usd"],
                                                         frame["banda_lineal_alta_usd"] - frame["dinero_usd"]],
                  fmt="none", ecolor=NEUTRAL_FILL, elinewidth=6, label="banda lineal (todo falla a la vez)")
    axis.errorbar(positions, frame["dinero_usd"], yerr=[frame["dinero_usd"] - frame["banda_baja_usd"],
                                                         frame["banda_alta_usd"] - frame["dinero_usd"]],
                  fmt="o", color=INK, ecolor=INK, capsize=4, label="banda por composición")
    axis.set_xticks(positions)
    axis.set_xticklabels(frame[label_column].astype(str), rotation=45, ha="right", fontsize=8)
    axis.set_ylim(bottom=0)
    money_axis(axis)


def draw_year(table, context, number, year, parts_names):
    total = total_rows(table).iloc[0]
    figure, axes = new_figure(15, 5.2, 1, 2, gridspec_kw={"width_ratios": [3, 2]})
    waterfall(axes[0], {name: total[name] for name in parts_names}, total["dinero_usd"], total["banda_baja_usd"],
              total["banda_alta_usd"])
    point_and_whisker(axes[1], region_rows(table)[region_rows(table)["orden_total"] == 0], context.region)
    axes[1].legend(fontsize=7.5, frameon=False)
    axes[1].set_title("por región", fontsize=10, color=INK, loc="left")
    titles(figure, f"{number} · Cómo acaba {year}: {money_label(total['dinero_usd'])} "
                   f"[{money_label(total['banda_baja_usd'])} – {money_label(total['banda_alta_usd'])}]",
           "La banda es la de composición (las filas con la misma tasa fallan juntas; el error de la adquisición, todo a la vez). "
           f"La lineal, el peor caso: [{money_label(total['banda_lineal_baja_usd'])} – {money_label(total['banda_lineal_alta_usd'])}].")
    return [save(figure, chart_path(context, number, f"como_acaba_{year}"))]


def draw_4_1(table, context, number):
    return draw_year(table, context, number, context.current_year, list(components_of_year(context, context.current_year)))


def draw_4_3(table, context, number):
    return draw_year(table, context, number, context.next_year, list(components_of_year(context, context.next_year)))


def draw_month_by_month(table, context, number, year):
    total = total_rows(table).sort_values("periodo")
    figure, axes = new_figure(14, 8.5, 2, 1)
    positions = np.arange(len(total))
    axes[0].bar(positions - 0.2, total["pipeline_usd"], 0.38, color=NEUTRAL_FILL, label="pipeline que vence")
    axes[0].bar(positions + 0.2, total["renovado_real_usd"], 0.38, color=BLUE, label="renovado real")
    axes[0].bar(positions + 0.2, total["renovado_previsto_usd"], 0.38, bottom=total["renovado_real_usd"], color=BLUE, alpha=0.35,
                label="renovado previsto")
    forecast_months = total["renovado_previsto_usd"] > 0
    renewed = total["renovado_real_usd"] + total["renovado_previsto_usd"]
    band_low = renewed - (total["dinero_usd"] - total["banda_baja_usd"])
    band_high = renewed + (total["banda_alta_usd"] - total["dinero_usd"])
    axes[0].errorbar(positions[forecast_months.to_numpy()] + 0.2, renewed[forecast_months],
                     yerr=[(renewed - band_low)[forecast_months], (band_high - renewed)[forecast_months]],
                     fmt="none", ecolor=INK, capsize=3, linewidth=1)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(total["periodo"].astype(str))
    axes[0].set_ylim(0, 1.2 * float(max(total["pipeline_usd"].max(), band_high.max(), 1.0)))     # room for the legend
    money_axis(axes[0])
    axes[0].legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    axes[0].set_title("pipeline y renovado (con la banda por composición en los meses previstos)", fontsize=10, color=INK, loc="left")
    axis = axes[1]
    real = total["tasa_real"].notna() & (total["vencen_unidades_reales"] > 0)
    axis.plot(positions[real.to_numpy()], total.loc[real, "tasa_real"], color=BLUE, linewidth=2, marker="o", label="tasa real")
    forecast = total["tasa_prevista"].notna() & (total["unidades_al_vencer"] > 0)
    axis.fill_between(positions[forecast.to_numpy()], total.loc[forecast, "tasa_framework_baja"],
                      total.loc[forecast, "tasa_framework_alta"], color=ORANGE, alpha=0.15, linewidth=0, label="banda del framework")
    axis.fill_between(positions[forecast.to_numpy()], total.loc[forecast, "tasa_binomial_baja"],
                      total.loc[forecast, "tasa_binomial_alta"], color=ORANGE, alpha=0.35, linewidth=0, label="banda binomial")
    axis.plot(positions[forecast.to_numpy()], total.loc[forecast, "tasa_prevista"], color=ORANGE, linewidth=2, linestyle="--",
              marker="o", label="tasa prevista")
    axis.set_xticks(positions)
    axis.set_xticklabels(total["periodo"].astype(str))
    # the axis is cut around the rates (at 0-100 % a band of 1-2 pp cannot be seen): the title says so
    shown = pd.concat([total.loc[real, ["tasa_real", "tasa_real_binomial_baja", "tasa_real_binomial_alta"]].stack(),
                       total.loc[forecast, ["tasa_framework_baja", "tasa_framework_alta", "tasa_binomial_baja",
                                            "tasa_binomial_alta"]].stack()]).dropna()
    low, high = (float(shown.min()) - 0.05, float(shown.max()) + 0.05) if len(shown) else (0.0, 1.0)
    axis.set_ylim(max(0.0, low), min(1.0, high))
    percent_axis(axis, decimals=1)
    axis.legend(fontsize=8.5, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    axis.set_title(f"la tasa de renovación en unidades (eje recortado: {100 * max(0.0, low):.0f}–{100 * min(1.0, high):.0f} %)",
                   fontsize=10, color=INK, loc="left")
    titles(figure, f"{number} · {year} mes a mes", f"Renovado del año: {money_label(renewed.sum())}; "
                                                    f"retail to subscription aparte: {money_label(total['retail_to_subscription_usd'].sum())}.")
    return [save(figure, chart_path(context, number, f"{year}_mes_a_mes"), top=0.92)]


def draw_4_2(table, context, number):
    return draw_month_by_month(table, context, number, context.current_year)


def draw_4_4(table, context, number):
    return draw_month_by_month(table, context, number, context.next_year)


def draw_chain(table, context, number, first, second, year):
    detail = table[table["orden_region"] == 0].copy()
    label_column = second if second and second in detail else first
    detail["_etiqueta"] = (detail[first].astype(str) + " · " + detail[label_column].astype(str)) if label_column != first else detail[first].astype(str)
    detail = detail.sort_values([first, "dinero_usd"], ascending=[True, False])
    figure, axis = new_figure(max(10, 0.45 * len(detail) + 3), 5.5)
    point_and_whisker(axis, detail, "_etiqueta")
    axis.legend(fontsize=8, frameon=False)
    total = total_rows(table).iloc[0]
    titles(figure, f"{number} · {year}: el dinero de cada {('subregión' if label_column != first else 'nivel')} y entre qué valores se moverá",
           f"Total {money_label(total['dinero_usd'])} [{money_label(total['banda_baja_usd'])} – {money_label(total['banda_alta_usd'])}] · "
           f"tasa {100 * total['tasa_prevista']:.1f} % (binomial ±{100 * (total['tasa_binomial_alta'] - total['tasa_prevista']):.2f} pp, "
           f"framework ±{100 * (total['tasa_framework_alta'] - total['tasa_prevista']):.2f} pp) · uplift {total['uplift']:.3f}")
    return [save(figure, chart_path(context, number, f"cadena_{year}"))]


def draw_5_1(table, context, number):
    return draw_chain(table, context, number, context.region, context.subregion, context.next_year)


def draw_5_1b(table, context, number):
    return draw_chain(table, context, number, context.region, context.subregion, context.current_year)


def draw_5_2(table, context, number):
    second = "tr_product_level_2" if "tr_product_level_2" in table.columns else None
    return draw_chain(table, context, number, context.product, second, context.next_year)


def draw_5_3(table, context, number):
    regions = region_rows(table)
    regions = regions[regions["orden_total"] == 0]
    total = total_rows(table).iloc[0]
    nominal = math.erf(context.z / math.sqrt(2))
    figure, axis = new_figure(11, 4.8)
    positions = np.arange(len(regions))
    bars = axis.bar(positions, regions["cobertura"], color=BLUE, width=0.6)
    axis.bar_label(bars, labels=[f"{100 * value:.0f} %" for value in regions["cobertura"]], fontsize=8.5, padding=2)
    axis.axhline(nominal, color=ORANGE, linestyle="--", linewidth=1.2, label=f"cobertura nominal ({100 * nominal:.0f} %)")
    axis.set_xticks(positions)
    axis.set_xticklabels(regions[context.region].astype(str), rotation=30, ha="right")
    axis.set_ylim(0, 1.1)
    percent_axis(axis)
    axis.legend(fontsize=8.5, frameon=False, loc="lower right")
    titles(figure, f"{number} · ¿Ha sido honesta la banda? La parte del examen que cayó dentro, por región",
           f"Total: {100 * total['cobertura']:.0f} % dentro (la serie sola: {100 * total['cobertura_serie_sola']:.0f} %) · "
           f"WAPE {100 * total['wape']:.1f} %. Muy por debajo de la nominal: la banda se queda corta en esa región.")
    return [save(figure, chart_path(context, number, "banda_honesta"))]


def draw_5_4(table, context, number):
    regions = region_rows(table)
    regions = regions[regions["orden_total"] == 0]
    years = sorted(regions["anio"].unique())
    figure, axes = new_figure(14, 4.8, 1, len(years), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, year in zip(axes, years):
        of_year = regions[regions["anio"] == year]
        positions = np.arange(len(of_year))
        axis.bar(positions - 0.2, of_year["resta_lo_que_falta_por_madurar_usd"], 0.38, color=ORANGE,
                 label="resta lo que falta por madurar")
        axis.bar(positions + 0.2, of_year["potencial_usd"], 0.38, color=AQUA, label="potencial si renovasen como neutros")
        axis.set_xticks(positions)
        axis.set_xticklabels(of_year[context.region].astype(str), rotation=30, ha="right", fontsize=8)
        money_axis(axis)
        total = table[(table["orden_total"] == 1) & (table["anio"] == year)].iloc[0]
        axis.set_title(f"{int(year)}: resta {money_label(total['resta_lo_que_falta_por_madurar_usd'])} · potencial "
                       f"{money_label(total['potencial_usd'])}", fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=8, frameon=False)
    titles(figure, f"{number} · Lo que cuestan dormant y softcancel",
           "El potencial es un techo («si se corrigiera todo»): las unidades con estado al vencer, a la tasa de las neutras de su grupo y mes.")
    return [save(figure, chart_path(context, number, "coste_de_los_estados"))]


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 6 · THE REAL BANDS: HOW THE ERRORS ADD UP, LEARNED FROM THE EXAM
# ═══════════════════════════════════════════════════════════════════════════════════
# The exam of step 7.1 (sff_series_exam_detail, method framework) predicted every forecast series in the exam months
# at h = 1 and h = 6, with its band, and knows what really renewed. Adding those predictions up at a level (series,
# composition, subregion, region, total), the error of the level is REAL; the band of the level can be built by four
# rules from the bands of its series:
#   lineal       Σ deviations                                   every series wrong at once
#   composicion  √(Σ per composition (Σ deviations)²)           a composition wrong at once, compositions apart
#   cuadratura   √(Σ deviations²)                               every series apart
#   binomial     z √(Σ n p (1 − p))                             the noise of the volume only
# A rule is right at a level when the real error falls inside its band as often as z promises (90 % with 1.645) and
# the error is about as big as the band says (cociente_error_banda = RMS(error) / RMS(semibanda / z) ≈ 1; above 1 the
# band is too narrow, below too wide). Two time bases: every month apart (mes) and the exam months added up (acumulado,
# what a year does: there a composition is wrong in the same direction in every month, a series too).

TABLE_REAL_BANDS = "bandas_reales"
TABLE_REAL_BANDS_SUMMARY = "bandas_reales_resumen"
BAND_RULES = ["lineal", "composicion", "cuadratura", "binomial"]
EXAM_LEVELS = ["serie", "composicion", "subregion", "region", "total"]
TABLE_SERIES_EXAM_DETAIL = "series_exam_detail"


def real_bands_ctes(context: ReportContext) -> str:
    """The common table expressions of the real bands: the exam of every series with its region (serie) and every level
    × group × month × h with its real error and the band of every rule (bandas)."""
    region, subregion = context.region, context.subregion or context.region
    detail = table_in_sql(context.configuration, TABLE_SERIES_EXAM_DETAIL)
    z = context.z

    def level(name: str, group: Optional[str], monthly: bool) -> str:
        period_key = ["periodo"] if monthly else []
        inner_keys = ([group] if group else []) + (["composition_id"] if group != "composition_id" else []) + period_key + ["h"]
        outer_keys = (["grupo"] if group else []) + period_key + ["h"]
        group_select = f"{group} AS grupo, " if group else ""
        group_value = "grupo" if group else f"'{TOTAL_LABEL}'"
        period_value = "periodo" if monthly else "'acumulado'"
        composition_select = "" if group == "composition_id" else "composition_id, "
        return f"""SELECT '{name}' AS nivel, {group_value} AS grupo, {period_value} AS periodo, h,
       SUM(series) AS series, COUNT(*) AS composiciones,
       SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
       SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
       SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
       SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
       SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
       {z} * SQRT(SUM(varianza_binomial)) AS binomial
FROM (SELECT {group_select}{composition_select}{', '.join(period_key + ['h'])},
             COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
             SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
             SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
             SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
             SUM(varianza_binomial) AS varianza_binomial
      FROM {'serie' if monthly else 'serie_acumulada'}
      GROUP BY {', '.join(inner_keys)}) AS por_composicion
GROUP BY {', '.join(outer_keys)}"""

    groups = [("serie", "fs_id"), ("composicion", "composition_id"),
              ("subregion", "subregion_completa"), ("region", "region"), ("total", None)]
    levels = "\nUNION ALL\n".join(level(name, group, monthly) for monthly in (True, False) for name, group in groups)
    return f"""WITH mapa AS (
    SELECT DISTINCT s1_4_fs_id, {region}, {subregion} FROM {context.core}
    WHERE origen_fila = 'raw' AND s1_4_fs_id IS NOT NULL
),
serie AS (
    SELECT d.fs_id, d.composition_id, m.{region} AS region,
           CONCAT(m.{region}, ' · ', m.{subregion}) AS subregion_completa, d.period AS periodo, d.h,
           COALESCE(d.pred_units, 0) AS predichas, COALESCE(d.real_units, 0) AS reales,
           COALESCE(d.pred_usd, 0) AS predicho_usd, COALESCE(d.real_usd, 0) AS real_usd,
           COALESCE(d.due_units, 0) * (COALESCE(d.pred_rate, 0) - COALESCE(d.band_low, d.pred_rate, 0)) AS desviacion_baja,
           COALESCE(d.due_units, 0) * (COALESCE(d.band_high, d.pred_rate, 0) - COALESCE(d.pred_rate, 0)) AS desviacion_alta,
           COALESCE(d.due_units, 0) * (COALESCE(d.pred_rate, 0) - COALESCE(d.band_low, d.pred_rate, 0))
               * COALESCE(d.due_units, 0) * (COALESCE(d.pred_rate, 0) - COALESCE(d.band_low, d.pred_rate, 0)) AS cuadrado_baja,
           COALESCE(d.due_units, 0) * (COALESCE(d.band_high, d.pred_rate, 0) - COALESCE(d.pred_rate, 0))
               * COALESCE(d.due_units, 0) * (COALESCE(d.band_high, d.pred_rate, 0) - COALESCE(d.pred_rate, 0)) AS cuadrado_alta,
           COALESCE(d.due_units, 0) * COALESCE(d.pred_rate, 0) * (1 - COALESCE(d.pred_rate, 0)) AS varianza_binomial
    FROM {detail} AS d
    JOIN mapa AS m ON m.s1_4_fs_id = d.fs_id
    WHERE d.method = 'framework'
),
serie_acumulada AS (
    -- the exam months added up per series: a series is wrong in the same direction every month (linear over the
    -- months), its binomial noise is independent month to month (its variance adds up)
    SELECT fs_id, composition_id, region, subregion_completa, h,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
           SUM(desviacion_baja) * SUM(desviacion_baja) AS cuadrado_baja, SUM(desviacion_alta) * SUM(desviacion_alta) AS cuadrado_alta,
           SUM(varianza_binomial) AS varianza_binomial
    FROM serie
    GROUP BY fs_id, composition_id, region, subregion_completa, h
),
niveles AS (
{textwrap.indent(levels, '    ')}
),
bandas AS (
    SELECT niveles.*,
           CASE WHEN error < 0 THEN lineal_baja ELSE lineal_alta END AS semibanda_lineal,
           CASE WHEN error < 0 THEN composicion_baja ELSE composicion_alta END AS semibanda_composicion,
           CASE WHEN error < 0 THEN cuadratura_baja ELSE cuadratura_alta END AS semibanda_cuadratura,
           binomial AS semibanda_binomial
    FROM niveles
)"""


def sql_6_1(context: ReportContext) -> Optional[str]:
    """6.1 · the real bands: every level × group × month (and the exam months added up) × h, its real error and the band
    of every rule."""
    if not context.has_table(TABLE_SERIES_EXAM_DETAIL):
        return None
    return (f"-- 6.1 · las bandas reales: el error real del examen (7.1, método framework) a cada nivel y la banda de cada regla\n"
            f"{real_bands_ctes(context)}\nSELECT * FROM bandas\nORDER BY nivel, grupo, periodo, h;")


def sql_6_2(context: ReportContext) -> Optional[str]:
    """6.2 · per level, time base, h and rule: how often the real error fell inside the band and how big it was."""
    if not context.has_table(TABLE_SERIES_EXAM_DETAIL):
        return None
    z = context.z
    per_rule = "\nUNION ALL\n".join(
        f"""SELECT nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END AS base, h, '{rule}' AS regla,
       COUNT(*) AS observaciones,
       AVG(CASE WHEN ABS(error) <= semibanda_{rule} THEN 1.0 ELSE 0.0 END) AS cobertura,
       SQRT(AVG(error * error)) / NULLIF(SQRT(AVG(semibanda_{rule} * semibanda_{rule})) / {z}, 0) AS cociente_error_banda,
       AVG(ABS(error)) AS error_medio_absoluto, AVG(semibanda_{rule}) AS semibanda_media,
       AVG(error) AS error_medio, AVG(error) / NULLIF(SQRT(AVG(error * error)), 0) AS parte_sesgo
FROM bandas
GROUP BY nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END, h"""
        for rule in BAND_RULES)
    return (f"-- 6.2 · el resumen de las bandas reales: cobertura y cociente error/banda por nivel, base, h y regla "
            f"(1 = honesta; la cobertura prometida es la de z = {z:g})\n{real_bands_ctes(context)},\nresumen AS (\n"
            f"{textwrap.indent(per_rule, '    ')}\n)\nSELECT * FROM resumen\nORDER BY base, h, regla, nivel;")


AGGREGATE_LEVELS = ["subregion", "region", "total"]       # where the compositions meet: the correlation is learned there


def implied_correlation(frame: pd.DataFrame, z: float) -> float:
    """The correlation between the errors of different compositions that the real errors imply. With the same
    correlation ρ between every pair of compositions, the variance of a group is Σ σ_c² + ρ Σ_{c≠c'} σ_c σ_c' =
    composicion² + ρ (lineal² − composicion²) in units of (semibanda / z)²; so ρ = (z² · mean(error²) − mean(composicion²))
    / (mean(lineal²) − mean(composicion²)). 0 = the compositions are independent (the composition rule); 1 = they are
    wrong at once (the linear rule)."""
    composition = frame["semibanda_composicion"].astype(float) ** 2
    linear = frame["semibanda_lineal"].astype(float) ** 2
    room = (linear - composition).mean()
    return float((z * z * (frame["error"].astype(float) ** 2).mean() - composition.mean()) / room) if room > 0 else np.nan


def calibration_factors(detail: pd.DataFrame, z: float) -> pd.DataFrame:
    """Per level, time base, h and rule: the real coverage and the factor k by which the band must be multiplied to
    cover the real errors as z promises (the quantile at the nominal coverage of |error| / semibanda; 1 = honest, above
    1 too narrow). (In SQL Server: PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ABS(error) / semibanda_x).)
    Plus the rule LEARNED from the data, 'aprendida': √(composicion² + ρ (lineal² − composicion²)), with ρ the correlation
    between compositions implied by the errors of the aggregate levels (subregion, region and total pooled) of that time
    base and h; its coverage is in sample, so it says whether ONE ρ fits every level, not that it will hold."""
    nominal = math.erf(z / math.sqrt(2))
    detail = detail.assign(base=np.where(detail["periodo"].astype(str) == "acumulado", "acumulado", "mes"))
    rows = []
    for (base, horizon), panel in detail.groupby(["base", "h"]):
        rho = implied_correlation(panel[panel["nivel"].isin(AGGREGATE_LEVELS)], z)
        learned = np.sqrt(panel["semibanda_composicion"].astype(float) ** 2 + max(0.0, min(1.0, rho if rho == rho else 0.0))
                          * (panel["semibanda_lineal"].astype(float) ** 2 - panel["semibanda_composicion"].astype(float) ** 2).clip(lower=0))
        panel = panel.assign(semibanda_aprendida=learned)
        for level, frame in panel.groupby("nivel"):
            for rule in BAND_RULES + ["aprendida"]:
                half = frame[f"semibanda_{rule}"].astype(float)
                ratio = (frame["error"].abs() / half.where(half > 0)).dropna()
                rows.append({"nivel": level, "base": base, "h": int(horizon), "regla": rule, "observaciones": len(frame),
                             "cobertura": float((frame["error"].abs() <= half).mean()),
                             "factor_k": float(ratio.quantile(nominal)) if len(ratio) else np.nan,
                             "correlacion_implicita_del_nivel": implied_correlation(frame, z),
                             "correlacion_aprendida": rho})
    return pd.DataFrame(rows)


def draw_6_1(table, context, number):
    """At the region level, month by month (h = 1): the real error against the band of every rule."""
    regions = table[(table["nivel"] == "region") & (table["periodo"] != "acumulado") & (table["h"] == 1)]
    total = table[(table["nivel"] == "total") & (table["periodo"] != "acumulado") & (table["h"] == 1)]
    frames = [("TOTAL", total)] + [(name, regions[regions["grupo"] == name]) for name in regions["grupo"].drop_duplicates()]
    figure, axes = small_multiples([name for name, _ in frames], height_per_row=3.0, same_scale=False)
    for axis, (name, frame) in zip(axes, frames):
        frame = frame.sort_values("periodo")
        # the scale of the error and of the narrow bands; the linear band, much wider, runs off the panel
        reach = 1.6 * float(max(frame["error"].abs().max(), frame["composicion_alta"].max(), frame["cuadratura_alta"].max(), 1.0))
        axis.set_ylim(-reach, reach)
        positions = np.arange(len(frame))
        for offset, rule, colour in ((-0.24, "lineal", NEUTRAL_FILL), (-0.08, "composicion", INK),
                                     (0.08, "cuadratura", AQUA), (0.24, "binomial", ORANGE)):
            axis.errorbar(positions + offset, np.zeros(len(frame)),
                          yerr=[frame[f"{rule}_baja"] if rule != "binomial" else frame["binomial"],
                                frame[f"{rule}_alta"] if rule != "binomial" else frame["binomial"]],
                          fmt="none", ecolor=colour, elinewidth=3 if rule == "lineal" else 1.6, capsize=0 if rule == "lineal" else 3,
                          label=f"banda {rule}")
        axis.scatter(positions, frame["error"], color=BLUE, zorder=5, s=28, label="error real (reales − predichas)")
        axis.axhline(0, color=INK_MUTED, linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(frame["periodo"].astype(str), fontsize=7, rotation=45)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    titles(figure, f"{number} · El error real de cada mes del examen frente a la banda de cada regla (h = 1, unidades renovadas)",
           "Cada panel es una región (el primero, el total). Un punto fuera de una banda es un mes que esa regla no cubrió. "
           "Escala propia por panel, a la medida del error: la banda lineal, mucho más ancha, se sale del panel.")
    for axis in axes:
        axis.tick_params(labelleft=True)
    return [save(figure, chart_path(context, number, "error_real_frente_a_bandas"), top=0.9)]


def draw_6_2(table, context, number, detail: pd.DataFrame = None):
    """Per level (x), the coverage and the factor k of every rule, for every time base and h."""
    detail = detail if detail is not None else context.answered.get("6.1")
    if detail is None:
        detail = pd.read_sql(sql_6_1(context), context.engine)
    factors = calibration_factors(detail, context.z)
    factors.to_csv(os.path.join(context.output_folder, f"{number.replace('.', '_')}_factor_k.csv"), index=False)
    nominal = math.erf(context.z / math.sqrt(2))
    panels = [(base, horizon) for base in ("mes", "acumulado") for horizon in sorted(table["h"].unique())]
    figure, axes = new_figure(15, 8.5, 2, len(panels), sharey="row")
    axes = np.atleast_2d(axes)
    colours = {"lineal": INK_MUTED, "composicion": INK, "cuadratura": AQUA, "binomial": ORANGE, "aprendida": VIOLET}
    learned_text = []
    for column, (base, horizon) in enumerate(panels):
        factors_of_panel = factors[(factors["base"] == base) & (factors["h"] == horizon)]
        for rule in BAND_RULES + ["aprendida"]:
            of_rule = factors_of_panel[factors_of_panel["regla"] == rule].set_index("nivel").reindex(EXAM_LEVELS)
            style = dict(marker="o", color=colours[rule], label=rule, linewidth=2.4 if rule == "aprendida" else 1.4,
                         linestyle="--" if rule == "aprendida" else "-")
            axes[0, column].plot(range(len(EXAM_LEVELS)), of_rule["cobertura"], **style)
            axes[1, column].plot(range(len(EXAM_LEVELS)), of_rule["factor_k"], **style)
        rho = factors_of_panel["correlacion_aprendida"].iloc[0] if len(factors_of_panel) else np.nan
        learned_text.append(f"{'mes' if base == 'mes' else 'acumulado'} h={horizon}: ρ = {rho:.2f}")
        observations = factors_of_panel.groupby("nivel")["observaciones"].max().reindex(EXAM_LEVELS)
        for row in (0, 1):
            axes[row, column].set_xticks(range(len(EXAM_LEVELS)))
            axes[row, column].set_xticklabels([f"{level}\n(n={int(count) if count == count else 0})"
                                               for level, count in zip(EXAM_LEVELS, observations)], fontsize=7)
        axes[0, column].axhline(nominal, color=ORANGE, linestyle="--", linewidth=1)
        axes[0, column].set_ylim(0, 1.05)
        percent_axis(axes[0, column])
        axes[0, column].set_title(f"{'cada mes' if base == 'mes' else 'los meses del examen sumados'} · h = {horizon}",
                                  fontsize=10, color=INK, loc="left")
        axes[1, column].axhline(1, color=ORANGE, linestyle="--", linewidth=1)
        axes[1, column].set_yscale("log")
        axes[1, column].yaxis.set_major_formatter(lambda value, _: f"{value:g}")
    axes[0, 0].set_ylabel("cobertura real (dentro de la banda)", color=INK_SECONDARY)
    axes[1, 0].set_ylabel("factor k para cubrir el " + f"{100 * nominal:.0f} %", color=INK_SECONDARY)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    titles(figure, f"{number} · Qué regla de suma acierta a cada nivel: la cobertura real y el factor que le falta",
           f"Arriba, la parte de los errores reales que cayó dentro de la banda (la línea: el {100 * nominal:.0f} % prometido). "
           "Abajo, por cuánto habría que multiplicar la banda para cubrirlo (1 = honesta; por encima, se queda corta). "
           "n = observaciones: con pocas (el total), la cobertura es ruidosa. Aprendida = √(composición² + ρ (lineal² − "
           "composición²)), ρ la correlación entre composiciones que implican los errores: " + " · ".join(learned_text) + ".")
    return [save(figure, chart_path(context, number, "que_regla_acierta"), top=0.88)]


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 7 · THE PIPELINE AUDITED BY ITS COMPONENTS (added 28-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE QUESTION: the pipeline of next year comes out smaller than the one of this year. Is that what the past says,
# or does one component break the line? Every component of the pipeline is drawn month by month, from the real past
# to December of next year, so the future can be read against the real trend: a component that suddenly goes up or
# down where the real data stops is the one to audit.
#
# THE COMPONENTS OF THE PIPELINE (what falls due in a month, in USD):
#   renewal_1y              rows of the extract, 1-year term, not an acquisition: what renewed 12 months before
#   acquisition_1y          rows of the extract, 1-year term, an acquisition (acquisition_column in acquisition_values)
#   multi_year              rows of the extract with a 2- or 3-year term
#   projected_renewal       step 7.2: the expected renewal of the 1-year pipeline that falls due from the current
#                           month to December, due again 12 months later
#   simulated_acquisition   step 5.1: the 1-year acquisitions simulated from the current month to December
#   retail_to_subscription  step 6.1: the re-entry of the retail to subscription universe
#   wiped                   not a component: the 1-year pipeline of the extract that step 1.3 wipes (it is only
#                           partly built) and the projected renewals and simulated acquisitions replace
# The pipeline of the past that comes from retail to subscription cannot be told apart in the extract yet (it is
# inside the renewals): that is an improvement still to do.
#
# THE UPLIFT OF EVERY COMPONENT (renewed USD over the USD due of the units that renewed):
#   closed months   forecast_renewed_USD / forecast_to_renew_USD_renewed (the real uplift)
#   future months   s7_2_esperado_usd / (s7_2_vencen_usd_al_vencer × s7_2_tasa) (the uplift the forecast applies)
# With and without price increases: a row is inside a price increase when its price group (price_group_dims) and
# its month fall inside the cycle (price_cycle_months) of an increase of sff_price_increase_calendar. The uplift
# without increases leaves those rows out: it is the stability the increases do not move.
#
# THE QUESTIONS:
#   7.1  the components of the pipeline month by month, real past and future, total and by region
#   7.2  the pipeline of every region in the last year, the current one and the next one, by component
#   7.3  does what renews come back as pipeline 12 months later? (the real ratio against the projected one)
#   7.4  the uplift of every component month by month, with and without price increases
# Every chart has its annex under it: the numbers of what the chart draws, also written as <number>_anexo*.csv.
# The questions are in Spanish; the charts of this chapter are in English.

COMPONENTS = {
    "renewal_1y": ("1-year renewals (extract)", BLUE),
    "multi_year": ("multi-year licences (extract)", "#8ab4ea"),
    "acquisition_1y": ("1-year acquisitions (extract)", "#f4a582"),
    "projected_renewal": ("projected renewals (step 7.2)", AQUA),
    "simulated_acquisition": ("simulated acquisitions (step 5.1)", ORANGE),
    "retail_to_subscription": ("retail to subscription re-entry (step 6.1)", VIOLET),
}
BUILT_COMPONENTS = ["projected_renewal", "simulated_acquisition", "retail_to_subscription"]   # what the framework builds
UPLIFT_COMPONENTS = ["renewal_1y", "multi_year", "acquisition_1y", "projected_renewal", "simulated_acquisition"]
PIPELINE_SOURCES = ["pipeline_renovado_real", "pipeline_real_esperado", "pipeline_proyectada", "pipeline_simulada"]
TABLE_PRICE_INCREASE_CALENDAR = "price_increase_calendar"
ENGLISH_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ANNEX_FONT_SIZE = 8
ANNEX_GRID_LINE = "#e6e5e0"
ANNEX_HEADER_FILL = "#f1f0ec"


# ─── the pieces of SQL ───────────────────────────────────────────────────────────
def level_columns(context: ReportContext, alias: str = "n") -> str:
    """The region columns of the SELECT: region and subregion (or region alone)."""
    columns = [context.region] + ([context.subregion] if context.subregion else [])
    return ", ".join(f"{alias}.{column}" for column in columns)


def is_one_year_term(context: ReportContext, alias: str = "n") -> str:
    """A row of a 1-year licence (every row when the extract has no term column)."""
    configuration = context.configuration
    if not configuration.term_column or configuration.term_column not in context.core_columns:
        return "1 = 1"
    one_year = sql_literal(configuration.one_year_term_value)
    return f"COALESCE({alias}.{configuration.term_column}, {one_year}) = {one_year}"


def is_acquisition(context: ReportContext, alias: str = "n") -> str:
    """A row of an acquisition (acquisition_column in acquisition_values)."""
    configuration = context.configuration
    if (not configuration.acquisition_column or not configuration.acquisition_values
            or configuration.acquisition_column not in context.core_columns):
        return "1 = 0"
    values = ", ".join(sql_literal(value) for value in configuration.acquisition_values)
    return f"{acquisition_column_as_written(context, alias)} IN ({values})"


def acquisition_column_as_written(context: ReportContext, alias: str = "n") -> str:
    """The acquisition column as the values of the Config compare with it. With numeric values (newcust = 1) the
    column is read as a number: a core written with the column as text can hold '1.0' and 1 at once."""
    configuration = context.configuration
    column = f"{alias}.{configuration.acquisition_column}"
    numeric = all(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
                  for value in configuration.acquisition_values)
    return f"CAST({column} AS FLOAT)" if numeric else column


def pipeline_component(context: ReportContext, alias: str = "n") -> str:
    """The component of the pipeline a row belongs to (COMPONENTS). The rows built by the framework are told by their
    source; the rows of the extract by their term and by whether they are an acquisition."""
    return f"""CASE
           WHEN {alias}.forecast_pipeline_source = 'pipeline_proyectada' THEN 'projected_renewal'
           WHEN {alias}.forecast_pipeline_source = 'pipeline_simulada' THEN 'simulated_acquisition'
           WHEN {alias}.forecast_pipeline_source = 'ts_reentrada' THEN 'retail_to_subscription'
           WHEN NOT ({is_one_year_term(context, alias)}) THEN 'multi_year'
           WHEN {is_acquisition(context, alias)} THEN 'acquisition_1y'
           ELSE 'renewal_1y'
       END"""


def month_number(expression: str) -> str:
    """'2026-10' as a number of months (2026 × 12 + 10), to compare months in SQL Server and in SQLite alike."""
    return f"(CAST(SUBSTRING({expression}, 1, 4) AS INT) * 12 + CAST(SUBSTRING({expression}, 6, 2) AS INT))"


@dataclass
class PriceIncreaseLookup:
    """The months inside a price increase, to join to the core: ctes (the months of every cycle, one row per price
    group and month, written before the base of a query), join (the LEFT JOIN of the core to them) and flag (1 when
    the row is inside an increase, 0 otherwise). A join on equal values, not a correlated EXISTS row by row: on a core
    of millions of rows the EXISTS was most of the time of the report (28-oct-2026)."""
    ctes: list
    join: str
    flag: str


INCREASE_MONTHS_CTE = "meses_en_subida"


def price_increase_lookup(context: ReportContext, alias: str = "n") -> PriceIncreaseLookup:
    """1 when the row's price group and month are inside the cycle of a price increase of the calendar (step 8.4),
    0 otherwise (and always 0 when this run has no calendar)."""
    configuration = context.configuration
    if not context.has_table(TABLE_PRICE_INCREASE_CALENDAR):
        return PriceIncreaseLookup(ctes=[], join="", flag="0")
    calendar = table_in_sql(configuration, TABLE_PRICE_INCREASE_CALENDAR)
    inspector = inspect(context.engine)
    calendar_columns = {column["name"] for column in inspector.get_columns(
        f"{configuration.table_prefix}{TABLE_PRICE_INCREASE_CALENDAR}", schema=configuration.sql_schema)}
    group_dims = [dim for dim in configuration.price_group_dims if dim in calendar_columns and dim in context.core_columns]
    # every month of every cycle: the start of the increase + 0 … price_cycle_months − 1
    offsets = " UNION ALL ".join(f"SELECT {offset} AS desfase" for offset in range(configuration.price_cycle_months))
    group_select = "".join(f"c.{dim}, " for dim in group_dims)
    months_sql = f"""SELECT DISTINCT {group_select}{month_number('c.inicio')} + d.desfase AS mes_numero
FROM {calendar} c
CROSS JOIN ({offsets}) d"""
    same_group = "".join(f"ms.{dim} = {alias}.{dim} AND " for dim in group_dims)
    join = (f"LEFT JOIN {INCREASE_MONTHS_CTE} ms ON {same_group}"
            f"ms.mes_numero = {month_number(f'{alias}.{configuration.period_col}')}")
    return PriceIncreaseLookup(ctes=[(INCREASE_MONTHS_CTE, months_sql)], join=join,
                               flag="CASE WHEN ms.mes_numero IS NOT NULL THEN 1 ELSE 0 END")


def sum_of_component(component: str, value: str, extra_condition: str = "") -> str:
    condition = f"componente = '{component}'" + (f" AND {extra_condition}" if extra_condition else "")
    return f"SUM(CASE WHEN {condition} THEN {value} ELSE 0 END)"


# ─── 7.1 · the components month by month ─────────────────────────────────────────
def components_base(context: ReportContext, period_column: str = "periodo") -> str:
    """Every row with pipeline: its region, its month, its component, the USD and units due, and what step 1.3 wiped."""
    period = context.configuration.period_col
    sources = ", ".join(f"'{source}'" for source in PIPELINE_SOURCES + ["ts_reentrada"])
    return f"""SELECT {level_columns(context)}, n.forecast_year AS anio, n.{period} AS {period_column},
       {pipeline_component(context)} AS componente,
       COALESCE(n.forecast_to_renew_USD, 0) AS vence_usd,
       COALESCE(n.forecast_to_renew_units, 0) AS vence_unidades,
       CASE WHEN n.origen_fila = 'raw' AND n.s1_3_rol = 'proyeccion'
            THEN COALESCE(n.s1_1_vencen_usd, 0) - COALESCE(n.s1_3_vencen_usd, 0) ELSE 0 END AS borrado_usd
FROM {context.core} n
WHERE n.forecast_pipeline_source IN ({sources})"""


def component_measures() -> list:
    measures = [f"{sum_of_component(component, 'vence_usd')} AS {component}_usd" for component in COMPONENTS]
    measures += ["SUM(vence_usd) AS total_usd",
                 " + ".join(sum_of_component(component, "vence_usd") for component in BUILT_COMPONENTS) + " AS built_usd",
                 "SUM(borrado_usd) AS wiped_usd",
                 "SUM(vence_unidades) AS total_unidades"]
    return measures


def sql_7_1(context: ReportContext) -> str:
    """7.1 · the components of the pipeline month by month, from the first month of the extract to next December."""
    return with_subtotals(context, components_base(context), component_measures(), ["anio", "periodo"],
                          comment="7.1 · la pipeline mes a mes por componente: el pasado real y lo que viene")


# ─── 7.2 · the pipeline of every region by year ──────────────────────────────────
def sql_7_2(context: ReportContext) -> str:
    """7.2 · the pipeline of the last year, the current one and the next one, by region and component."""
    base = components_base(context) + f"\n  AND n.forecast_year BETWEEN {context.current_year - 1} AND {context.next_year}"
    measures = component_measures() + [
        "SUM(CASE WHEN componente IN ('acquisition_1y', 'simulated_acquisition') THEN vence_usd ELSE 0 END) AS acquisition_usd"]
    return with_subtotals(context, base, measures, ["anio"],
                          comment=f"7.2 · la pipeline de {context.current_year - 1}, {context.current_year} y "
                                  f"{context.next_year} por región y componente: cuánto es simulado")


# ─── 7.3 · what renews comes back as pipeline 12 months later ────────────────────
def sql_7_3(context: ReportContext) -> str:
    """7.3 · the 1-year renewal pipeline of every month against what renewed 12 months before (real in the closed
    months, expected in the future). Their ratio in the real past is what the projection should keep."""
    period = context.configuration.period_col
    levels = level_columns(context)
    renewed_sources = ", ".join(f"'{source}'" for source in PIPELINE_SOURCES)
    a_year_later = f"CONCAT(CAST(n.forecast_year + 1 AS VARCHAR(4)), SUBSTRING(n.{period}, 5, 3))"
    component = pipeline_component(context)
    one_year = is_one_year_term(context)
    base = f"""SELECT {levels}, {a_year_later} AS periodo,
       CASE WHEN {one_year} THEN COALESCE(n.forecast_renewed_USD, 0) ELSE 0 END AS renovado_12m_antes_usd,
       CASE WHEN {one_year} THEN 0 ELSE COALESCE(n.forecast_renewed_USD, 0) END AS renovado_varios_anos_12m_antes_usd,
       0 AS renewal_1y_usd, 0 AS projected_renewal_usd, 0 AS r2s_reentrada_usd
FROM {context.core} n
WHERE n.forecast_pipeline_source IN ({renewed_sources})
  AND n.forecast_year + 1 <= {context.next_year}
UNION ALL
SELECT {levels}, n.{period} AS periodo,
       0 AS renovado_12m_antes_usd, 0 AS renovado_varios_anos_12m_antes_usd,
       CASE WHEN {component} = 'renewal_1y' THEN COALESCE(n.forecast_to_renew_USD, 0) ELSE 0 END AS renewal_1y_usd,
       CASE WHEN {component} = 'projected_renewal' THEN COALESCE(n.forecast_to_renew_USD, 0) ELSE 0 END AS projected_renewal_usd,
       CASE WHEN {component} = 'retail_to_subscription' THEN COALESCE(n.forecast_to_renew_USD, 0) ELSE 0 END AS r2s_reentrada_usd
FROM {context.core} n
WHERE n.forecast_pipeline_source IN ('pipeline_renovado_real', 'pipeline_real_esperado', 'pipeline_proyectada', 'ts_reentrada')"""
    measures = ["SUM(renovado_12m_antes_usd) AS renovado_12m_antes_usd",
                "SUM(renovado_varios_anos_12m_antes_usd) AS renovado_varios_anos_12m_antes_usd",
                "SUM(renewal_1y_usd) AS renewal_1y_usd",
                "SUM(projected_renewal_usd) AS projected_renewal_usd",
                "SUM(r2s_reentrada_usd) AS r2s_reentrada_usd",
                "(SUM(renewal_1y_usd) + SUM(projected_renewal_usd)) / NULLIF(SUM(renovado_12m_antes_usd), 0) AS ratio_pipeline_sobre_renovado"]
    return with_subtotals(context, base, measures, ["periodo"],
                          comment="7.3 · ¿lo renovado vuelve como pipeline 12 meses después? Pipeline de renovación de 1 año "
                                  "frente a lo renovado 12 meses antes")


# ─── 7.4 · the uplift of every component, with and without price increases ──────
def sql_7_4(context: ReportContext) -> str:
    """7.4 · the uplift of every component month by month (real in the closed months, applied in the future), with
    every row and without the rows inside a price increase."""
    configuration = context.configuration
    period = configuration.period_col
    sources = ", ".join(f"'{source}'" for source in PIPELINE_SOURCES)
    increases = price_increase_lookup(context)
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio, n.{period} AS periodo,
       {pipeline_component(context)} AS componente,
       {increases.flag} AS en_subida,
       CASE WHEN n.forecast_status = 'actual' THEN COALESCE(n.forecast_renewed_USD, 0)
            ELSE COALESCE(n.s7_2_esperado_usd, 0) END AS renovado_usd,
       {renewal_base_usd(context)} AS base_uplift_usd
FROM {context.core} n
{increases.join}
WHERE n.forecast_pipeline_source IN ({sources})"""
    measures = []
    for component in UPLIFT_COMPONENTS + ["total"]:
        def total_or(value, extra=""):
            if component == "total":
                return f"SUM(CASE WHEN {extra or '1 = 1'} THEN {value} ELSE 0 END)"
            return sum_of_component(component, value, extra)
        measures += [
            f"{total_or('renovado_usd')} AS {component}_renovado_usd",
            f"{total_or('base_uplift_usd')} AS {component}_base_usd",
            f"{total_or('renovado_usd')} / NULLIF({total_or('base_uplift_usd')}, 0) AS {component}_uplift",
            f"{total_or('renovado_usd', 'en_subida = 0')} AS {component}_renovado_sin_subida_usd",
            f"{total_or('base_uplift_usd', 'en_subida = 0')} AS {component}_base_sin_subida_usd",
            f"{total_or('renovado_usd', 'en_subida = 0')} / NULLIF({total_or('base_uplift_usd', 'en_subida = 0')}, 0) "
            f"AS {component}_uplift_sin_subida"]
    measures.append("SUM(CASE WHEN en_subida = 1 THEN base_uplift_usd ELSE 0 END) / NULLIF(SUM(base_uplift_usd), 0) "
                    "AS parte_en_subida")
    return with_subtotals(context, base, measures, ["anio", "periodo"], ctes=increases.ctes,
                          comment="7.4 · el uplift de cada componente mes a mes, con y sin subidas de precio")


# ─── the charts of chapter 7 (in English) and their annexes ─────────────────────
def as_months(frame: pd.DataFrame, column: str = "periodo") -> pd.DataFrame:
    """The rows in month order, with the month as a Period."""
    frame = frame.copy()
    frame["_month"] = pd.PeriodIndex(frame[column].astype(str), freq="M")
    return frame.sort_values("_month")


def month_axis(axis, months: pd.PeriodIndex, compact: bool = False) -> None:
    """A tick every January and July ('Jan 2024'); compact, for the small panels: every January, the year alone."""
    shown_months = (1,) if compact else (1, 7)
    positions = [position for position, month in enumerate(months) if month.month in shown_months]
    axis.set_xticks(positions)
    if compact:
        labels = [str(months[position].year) for position in positions]
    else:
        labels = [f"{ENGLISH_MONTHS[months[position].month - 1]} {months[position].year}" for position in positions]
    axis.set_xticklabels(labels, fontsize=7.5, rotation=0)
    axis.set_xlim(-0.5, len(months) - 0.5)


def mark_the_future(axis, months: pd.PeriodIndex, current_month: str, label: bool = True) -> None:
    """A light band over the months from the current one: what the extract does not know yet."""
    current = pd.Period(current_month, freq="M")
    future = [position for position, month in enumerate(months) if month >= current]
    if not future:
        return
    axis.axvspan(future[0] - 0.5, len(months) - 0.5, color=NEUTRAL_FILL, alpha=0.35, linewidth=0, zorder=0)
    axis.axvline(future[0] - 0.5, color=INK_MUTED, linewidth=1, linestyle=":")
    if label:
        axis.annotate("forecast →", (future[0] - 0.3, 1.0), xycoords=("data", "axes fraction"), fontsize=8,
                      color=INK_SECONDARY, va="top")


def complete_years(monthly: pd.DataFrame) -> list:
    """The years with their 12 months in the table (the first month of the extract alone is not a year)."""
    counts = monthly.groupby(monthly["_month"].map(lambda month: month.year))["_month"].nunique()
    return [int(year) for year, months in counts.items() if months == 12]


def growth(new: float, old: float) -> str:
    if not np.isfinite(old) or old == 0:
        return "—"
    return f"{100 * (new / old - 1):+.1f} %"


def annex_table(axis, frame: pd.DataFrame, title: str = "", font_size: int = ANNEX_FONT_SIZE,
                first_column_width: float = 0.22) -> None:
    """The annex of a chart: a table of text under it (the numbers of what the chart draws)."""
    axis.axis("off")
    if frame.empty:
        return
    shown = frame.reset_index()
    shown.columns = [""] + list(shown.columns[1:])          # the row labels have no header
    columns = len(shown.columns)
    other_width = (1 - first_column_width) / max(1, columns - 1)
    table = axis.table(cellText=shown.astype(str).values, colLabels=[str(column) for column in shown.columns],
                       loc="upper center", cellLoc="right", colLoc="right",
                       colWidths=[first_column_width] + [other_width] * (columns - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(ANNEX_GRID_LINE)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor(ANNEX_HEADER_FILL)
        if column == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    if title:
        axis.set_title(title, fontsize=9, color=INK, loc="left")


def write_annex(context: ReportContext, number: str, name: str, frame: pd.DataFrame) -> str:
    """The annex also as a CSV next to the question, and printed: the numbers can be read without the chart."""
    path = os.path.join(context.output_folder, f"{number.replace('.', '_')}_anexo_{name}.csv")
    frame.to_csv(path)
    print(f"[informe] {number} · annex {name}:")
    print(frame.to_string())
    return path


def yearly_annex(monthly: pd.DataFrame, rows: dict, money: bool = True) -> pd.DataFrame:
    """One row per line of the chart (rows: label → column), one column per complete year, and the growth of the
    last two years. The real trend (every complete year up to the current one) against next year: the coherence."""
    years = complete_years(monthly)
    by_year = monthly.groupby(monthly["_month"].map(lambda month: month.year))
    table = {}
    for label, column in rows.items():
        sums = by_year[column].sum()
        line = {str(year): (money_label(sums.get(year, np.nan)) if money else f"{sums.get(year, np.nan):.3f}") for year in years}
        if len(years) >= 2:
            line[f"{years[-1]} vs {years[-2]}"] = growth(sums.get(years[-1], np.nan), sums.get(years[-2], np.nan))
        if len(years) >= 3:
            line[f"{years[-2]} vs {years[-3]}"] = growth(sums.get(years[-2], np.nan), sums.get(years[-3], np.nan))
        table[label] = line
    return pd.DataFrame(table).T


def real_trend_text(monthly: pd.DataFrame, column: str, current_year: int, next_year: int) -> str:
    """The yearly growth of the real years (up to the current one) against the growth of next year."""
    years = [year for year in complete_years(monthly) if year <= current_year]
    by_year = monthly.groupby(monthly["_month"].map(lambda month: month.year))[column].sum()
    if len(years) < 2 or next_year not in by_year.index:
        return ""
    first, last = years[0], years[-1]
    yearly = (by_year[last] / by_year[first]) ** (1 / (last - first)) - 1 if by_year[first] > 0 else np.nan
    return (f"Real trend {first}→{last}: {100 * yearly:+.1f} %/year · {next_year} vs {current_year}: "
            f"{growth(by_year[next_year], by_year[current_year])}")


def stacked_components(axis, monthly: pd.DataFrame, components: list) -> None:
    positions = np.arange(len(monthly))
    values = [monthly[f"{component}_usd"].to_numpy(dtype=float) for component in components]
    axis.stackplot(positions, *values, colors=[COMPONENTS[component][1] for component in components],
                   labels=[COMPONENTS[component][0] for component in components], alpha=0.9, linewidth=0)


def draw_7_1(table, context, number):
    present = [component for component in COMPONENTS if table[f"{component}_usd"].abs().sum() > 0]
    total = as_months(total_rows(table))
    months = pd.PeriodIndex(total["_month"])
    positions = np.arange(len(total))

    # the total: the components stacked, the change against the same month a year before, and the annex
    figure, axes = new_figure(15, 12, 3, 1, gridspec_kw={"height_ratios": [3.2, 1.3, 2.2]})
    stacked_components(axes[0], total, present)
    with_wiped = total["total_usd"] + total["wiped_usd"]
    if total["wiped_usd"].sum() > 0:
        axes[0].plot(positions, with_wiped, color=INK_SECONDARY, linewidth=1, linestyle="--",
                     label="extract before step 1.3 (with the partial pipeline it wipes)")
    mark_the_future(axes[0], months, context.current_month)
    month_axis(axes[0], months)
    money_axis(axes[0])
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    axes[0].set_title("Pipeline due per month, by component (USD)", fontsize=10, color=INK, loc="left")

    a_year_before = total["total_usd"].shift(12)
    change = total["total_usd"] / a_year_before - 1
    current = pd.Period(context.current_month, freq="M")
    colours = [BLUE if month < current else ORANGE for month in months]
    axes[1].bar(positions, change.fillna(0), color=colours, width=0.8)
    axes[1].axhline(0, color=INK_MUTED, linewidth=0.8)
    mark_the_future(axes[1], months, context.current_month, label=False)
    month_axis(axes[1], months)
    percent_axis(axes[1])
    axes[1].set_title("Total pipeline against the same month a year before (blue: real · orange: forecast)",
                      fontsize=10, color=INK, loc="left")

    rows = {COMPONENTS[component][0]: f"{component}_usd" for component in present}
    rows["TOTAL pipeline"] = "total_usd"
    rows["of which built by the framework"] = "built_usd"
    if total["wiped_usd"].sum() > 0:
        rows["wiped by step 1.3 (not in the total)"] = "wiped_usd"
    annex = yearly_annex(total, rows)
    annex_table(axes[2], annex, "Annex · pipeline due per year and component")
    write_annex(context, number, "total", annex)
    titles(figure, f"{number} · The pipeline month by month, by component: real past and forecast",
           real_trend_text(total, "total_usd", context.current_year, context.next_year)
           + ". Each component should continue its own real line where the forecast starts.")
    paths = [save(figure, chart_path(context, number, "componentes_total"), top=0.93)]

    # by region: the same components, every region on its own scale, with its two years in the title
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names, height_per_row=3.2, same_scale=False)
    annex_rows = {}
    for axis, name in zip(axes, names):
        of_region = as_months(regions[regions[context.region] == name])
        stacked_components(axis, of_region, present)
        region_months = pd.PeriodIndex(of_region["_month"])
        mark_the_future(axis, region_months, context.current_month, label=False)
        month_axis(axis, region_months, compact=True)
        money_axis(axis)
        by_year = of_region.groupby(of_region["_month"].map(lambda month: month.year))["total_usd"].sum()
        this_year, next_year = by_year.get(context.current_year, np.nan), by_year.get(context.next_year, np.nan)
        axis.set_title(f"{name} · {context.current_year} {money_label(this_year)} → {context.next_year} "
                       f"{money_label(next_year)} ({growth(next_year, this_year)})", fontsize=9, color=INK, loc="left")
        annex_rows[name] = real_trend_text(of_region, "total_usd", context.current_year, context.next_year)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    titles(figure, f"{number}a · The pipeline month by month by region, by component",
           "Each region on its own scale. The grey band is the forecast.")
    paths.append(save(figure, chart_path(context, number, "componentes_por_region"), top=0.9))
    write_annex(context, number, "tendencia_por_region",
                pd.DataFrame({"real trend against next year": annex_rows}))
    return paths


def draw_7_2(table, context, number):
    years = [context.current_year - 1, context.current_year, context.next_year]
    present = [component for component in COMPONENTS if table[f"{component}_usd"].abs().sum() > 0]
    regions = pd.concat([region_rows(table), total_rows(table)])
    names = list(region_rows(table)[context.region].drop_duplicates())

    def value(name, year, column):
        rows = regions[(regions[context.region] == name) & (regions["anio"] == year)]
        return float(rows[column].sum()) if len(rows) else 0.0

    # the bars: every region, its three years, the components stacked; the annex under it
    figure, axes = new_figure(15, 11, 2, 1, gridspec_kw={"height_ratios": [3, 2.2]})
    width = 0.26
    for year_position, year in enumerate(years):
        offset = (year_position - 1) * width
        bottom = np.zeros(len(names))
        for component in present:
            heights = np.array([value(name, year, f"{component}_usd") for name in names])
            axes[0].bar(np.arange(len(names)) + offset, heights, width * 0.95, bottom=bottom, color=COMPONENTS[component][1],
                        label=COMPONENTS[component][0] if year_position == 1 else None,
                        alpha=1.0 if year != context.current_year - 1 else 0.55)
            bottom += heights
        for position, height in enumerate(bottom):
            axes[0].annotate(f"{year}\n{money_label(height)}", (position + offset, height), ha="center", va="bottom",
                             fontsize=6.5, color=INK_SECONDARY)
    axes[0].set_xticks(np.arange(len(names)))
    axes[0].set_xticklabels([str(name) for name in names], fontsize=9)
    money_axis(axes[0])
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_title(f"Pipeline due per region: {years[0]} (pale), {years[1]} and {years[2]}", fontsize=10, color=INK, loc="left")

    annex = {}
    for name in names + [TOTAL_LABEL]:
        this_year, next_year = value(name, years[1], "total_usd"), value(name, years[2], "total_usd")
        built = value(name, years[2], "built_usd")
        annex[name] = {str(years[0]): money_label(value(name, years[0], "total_usd")),
                       str(years[1]): money_label(this_year), str(years[2]): money_label(next_year),
                       f"{years[2]} vs {years[1]}": growth(next_year, this_year),
                       f"{years[1]} vs {years[0]}": growth(this_year, value(name, years[0], "total_usd")),
                       f"built {years[2]}": money_label(built),
                       "built share": f"{100 * built / next_year:.0f} %" if next_year else "—",
                       f"acq. due {years[1]} (sold {years[1] - 1})": money_label(value(name, years[1], "acquisition_usd")),
                       f"acq. due {years[2]} (sold {years[2] - 1})": money_label(value(name, years[2], "acquisition_usd")),
                       f"R2S re-entry pipeline {years[2]}": money_label(value(name, years[2], "retail_to_subscription_usd"))}
    annex = pd.DataFrame(annex).T
    annex_table(axes[1], annex, "Annex · pipeline per region (built = projected renewals + simulated acquisitions + "
                                "retail to subscription; acq. = the 1-year acquisitions that FALL DUE in the year, sold the "
                                "year before: extract + simulated; the 2- and 3-year ones fall due later. R2S here is only "
                                "the pipeline of its re-entry: its revenue of the year has no pipeline, see 4.3 and 8.1b)",
                first_column_width=0.12)
    write_annex(context, number, "por_region", annex)
    titles(figure, f"{number} · How much pipeline every region has, and how much of it is built",
           "The pipeline that falls due in the year. Next year mixes what the extract already has with what the framework builds.")
    paths = [save(figure, chart_path(context, number, "pipeline_por_region"), top=0.92)]

    # the subregions: only the annex, a table per subregion (the finer level adds up to its region)
    if context.subregion:
        fine = table[table["orden_region"] == 0]
        rows = {}
        for (name, subregion), block in fine.groupby([context.region, context.subregion], sort=True):
            def of_year(year, column, block=block):
                return float(block.loc[block["anio"] == year, column].sum())
            this_year, next_year = of_year(years[1], "total_usd"), of_year(years[2], "total_usd")
            rows[f"{name} · {subregion}"] = {
                str(years[0]): money_label(of_year(years[0], "total_usd")), str(years[1]): money_label(this_year),
                str(years[2]): money_label(next_year), f"{years[2]} vs {years[1]}": growth(next_year, this_year),
                "built share": f"{100 * of_year(years[2], 'built_usd') / next_year:.0f} %" if next_year else "—",
                f"acq. due {years[2]} (sold {years[2] - 1})": money_label(of_year(years[2], "acquisition_usd")),
                f"R2S re-entry pipeline {years[2]}": money_label(of_year(years[2], "retail_to_subscription_usd"))}
        subregions = pd.DataFrame(rows).T
        figure, axis = new_figure(15, 0.26 * len(subregions) + 1.6)
        annex_table(axis, subregions, font_size=7.5, first_column_width=0.25)
        titles(figure, f"{number}a · Annex · pipeline per subregion", "The subregions add up to their region.")
        paths.append(save(figure, chart_path(context, number, "pipeline_por_subregion"), top=0.95))
        write_annex(context, number, "por_subregion", subregions)
    return paths


def draw_7_3(table, context, number):
    total = as_months(total_rows(table))
    total = total[total["renovado_12m_antes_usd"] > 0]
    months = pd.PeriodIndex(total["_month"])
    positions = np.arange(len(total))
    figure, axes = new_figure(15, 11, 3, 1, gridspec_kw={"height_ratios": [3, 1.4, 1.6]})
    axes[0].stackplot(positions, total["renewal_1y_usd"], total["projected_renewal_usd"],
                      colors=[BLUE, AQUA], labels=[COMPONENTS["renewal_1y"][0], COMPONENTS["projected_renewal"][0]],
                      alpha=0.85, linewidth=0)
    axes[0].plot(positions, total["renovado_12m_antes_usd"], color=INK, linewidth=1.6,
                 label="1-year renewals of 12 months before (real; expected from the current month)")
    axes[0].plot(positions, total["renovado_12m_antes_usd"] + total["renovado_varios_anos_12m_antes_usd"], color=INK,
                 linewidth=1.2, linestyle="--", label="+ the multi-year renewals of 12 months before (if they renew as 1 year)")
    mark_the_future(axes[0], months, context.current_month)
    month_axis(axes[0], months)
    money_axis(axes[0])
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_title("1-year renewal pipeline due per month against what renewed 12 months before", fontsize=10,
                      color=INK, loc="left")
    ratio = (total["renewal_1y_usd"] + total["projected_renewal_usd"]) / total["renovado_12m_antes_usd"]
    current = pd.Period(context.current_month, freq="M")
    ratio_with_multi_year = ((total["renewal_1y_usd"] + total["projected_renewal_usd"])
                             / (total["renovado_12m_antes_usd"] + total["renovado_varios_anos_12m_antes_usd"]))
    axes[1].plot(positions, ratio, color=BLUE, linewidth=1.8, marker="o", markersize=2.5, label="over the 1-year renewals")
    axes[1].plot(positions, ratio_with_multi_year, color=INK_MUTED, linewidth=1.4, linestyle="--",
                 label="over the 1-year + multi-year renewals")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].axhline(1, color=INK_MUTED, linewidth=0.8)
    mark_the_future(axes[1], months, context.current_month, label=False)
    month_axis(axes[1], months)
    axes[1].set_title("Pipeline over renewed 12 months before (1 = what renews comes back exactly)", fontsize=10,
                      color=INK, loc="left")
    renewal_months = months - 12
    closed = renewal_months < current
    real_ratio = ratio[np.asarray(closed)]
    rows = {"renewed 12 months before (1 year)": "renovado_12m_antes_usd",
            "renewed 12 months before (multi-year)": "renovado_varios_anos_12m_antes_usd",
            COMPONENTS["renewal_1y"][0]: "renewal_1y_usd", COMPONENTS["projected_renewal"][0]: "projected_renewal_usd"}
    annex = yearly_annex(total, rows)
    by_year = total.groupby(total["_month"].map(lambda month: month.year))
    ratios = (by_year["renewal_1y_usd"].sum() + by_year["projected_renewal_usd"].sum()) / by_year["renovado_12m_antes_usd"].sum()
    annex.loc["ratio pipeline / renewed"] = {**{str(year): f"{ratios.get(year, np.nan):.3f}" for year in complete_years(total)}}
    annex = annex.fillna("")
    annex_table(axes[2], annex, "Annex · per year of the pipeline")
    write_annex(context, number, "total", annex)
    subtitle = (f"Real ratio (renewal month closed): mean {real_ratio.mean():.3f}, from {real_ratio.min():.3f} to "
                f"{real_ratio.max():.3f}. The projected renewals come back at exactly what they renew (ratio 1): "
                f"a real ratio above 1 is pipeline the projection does not build; below 1, pipeline it builds in excess.") if len(real_ratio) else ""
    titles(figure, f"{number} · Does what renews come back as pipeline 12 months later?", subtitle)
    paths = [save(figure, chart_path(context, number, "renovado_vuelve_como_pipeline"), top=0.92)]

    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names, height_per_row=2.8, same_scale=True)
    for axis, name in zip(axes, names):
        of_region = as_months(regions[regions[context.region] == name])
        of_region = of_region[of_region["renovado_12m_antes_usd"] > 0]
        region_months = pd.PeriodIndex(of_region["_month"])
        region_ratio = (of_region["renewal_1y_usd"] + of_region["projected_renewal_usd"]) / of_region["renovado_12m_antes_usd"]
        axis.plot(np.arange(len(of_region)), region_ratio, color=BLUE, linewidth=1.5)
        axis.axhline(1, color=INK_MUTED, linewidth=0.8)
        mark_the_future(axis, region_months, context.current_month, label=False)
        month_axis(axis, region_months, compact=True)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
    titles(figure, f"{number}a · Pipeline over renewed 12 months before, by region", "Same scale in every panel.")
    paths.append(save(figure, chart_path(context, number, "renovado_vuelve_por_region"), top=0.9))
    paths.append(draw_what_the_projection_misses(table, context, number))
    return paths


def real_ratio_of(rows: pd.DataFrame, months: list) -> dict:
    """The real ratio of a set of pipeline months: their 1-year renewal pipeline over what renewed 12 months before,
    in 1 year and with the multi-year renewals too."""
    chosen = rows[rows["_month"].isin(months)]
    renewed_one_year = chosen["renovado_12m_antes_usd"].sum()
    renewed_multi_year = chosen["renovado_varios_anos_12m_antes_usd"].sum()
    pipeline = chosen["renewal_1y_usd"].sum() + chosen["projected_renewal_usd"].sum()
    return {"ratio": safe_ratio(pipeline, renewed_one_year),
            "ratio_with_multi_year": safe_ratio(pipeline, renewed_one_year + renewed_multi_year),
            "multi_year_share": safe_ratio(renewed_multi_year, renewed_one_year)}


def draw_what_the_projection_misses(table, context, number) -> str:
    """The projected renewals come back at exactly what they renew (ratio 1). The real pipeline of the same months a
    year before came back at its real ratio: above 1, the extract builds pipeline that the projection does not (a
    multi-year licence that renews as 1 year, a customer that comes back…). At the real ratio of the same months,
    what would the projection build, per region and subregion?"""
    current = pd.Period(context.current_month, freq="M")
    groups = pd.concat([region_rows(table), total_rows(table)])
    if context.subregion:
        fine = table[table["orden_region"] == 0].copy()
        fine[context.region] = fine[context.region].astype(str) + " · " + fine[context.subregion].astype(str)
        groups = pd.concat([groups, fine])
    annex = {}
    for name, block in groups.groupby(context.region, sort=False):
        block = as_months(block)
        projected = block[block["projected_renewal_usd"] > 0]
        projected_months = list(projected["_month"])
        if not projected_months:
            continue
        same_months_before = [month - 12 for month in projected_months]
        last_closed = [month for month in block["_month"] if current <= month + 0 and month - 12 < current]
        same = real_ratio_of(block, same_months_before)
        last_twelve = real_ratio_of(block, last_closed[-12:])
        projected_usd = float(projected["projected_renewal_usd"].sum())
        # the real ratio also carries the retail to subscription that converted 12 months before (its pipeline is
        # inside the renewals of the extract); for next year it is built apart (step 6.1): not counted twice
        reentry_usd = float(projected["r2s_reentrada_usd"].sum())
        extra = projected_usd * (same["ratio"] - 1) if np.isfinite(same["ratio"]) else np.nan
        missing = extra - reentry_usd if np.isfinite(extra) else np.nan
        annex[str(name)] = {
            f"projected renewals {context.next_year}": money_label(projected_usd),
            "real ratio, same months a year before": ratio_text(same["ratio"]),
            "real ratio, last 12 months": ratio_text(last_twelve["ratio"]),
            "with multi-year renewals": ratio_text(same["ratio_with_multi_year"]),
            "multi-year / 1-year renewed": percent_text(same["multi_year_share"], 0),
            "extra at the real ratio": money_label(extra),
            "R2S re-entry already built": money_label(reentry_usd),
            "missing": money_label(missing),
            "_missing": missing}
    annex = pd.DataFrame(annex).T
    total_missing = annex.loc[TOTAL_LABEL, "_missing"] if TOTAL_LABEL in annex.index else np.nan
    shown = annex.drop(columns=["_missing"])
    region_count = region_rows(table)[context.region].nunique()
    write_annex(context, number, "lo_que_falta", shown)
    figure, axis = new_figure(15, 0.3 * min(len(shown), region_count + 1) + 2.4)
    annex_table(axis, shown.head(region_count + 1), font_size=8, first_column_width=0.16)
    titles(figure, f"{number}b · What the projected renewals miss if the real ratio holds",
           f"The projection brings back exactly what renews (ratio 1). At the real ratio of the same months a year before, "
           f"and without the retail to subscription re-entry the framework already builds apart, next year would have "
           f"{money_label(abs(total_missing))} {'more' if total_missing >= 0 else 'less'} pipeline. 'With multi-year renewals' "
           f"near 1 says the extra comes from multi-year licences that renew as 1 year. The subregions are in the CSV of the annex.")
    return save(figure, chart_path(context, number, "lo_que_falta_a_la_proyeccion"), top=0.85)


def draw_7_4(table, context, number):
    total = as_months(total_rows(table))
    months = pd.PeriodIndex(total["_month"])
    positions = np.arange(len(total))
    present = [component for component in UPLIFT_COMPONENTS if total[f"{component}_base_usd"].sum() > 0]
    has_increases = total["parte_en_subida"].fillna(0).sum() > 0
    figure, axes = new_figure(15, 13, 3, 1, gridspec_kw={"height_ratios": [2.4, 2.4, 2.4]})
    for axis, suffix, title in ((axes[0], "uplift", "Uplift with every row (price increases included)"),
                                (axes[1], "uplift_sin_subida", "Uplift without the rows inside a price increase")):
        for component in present:
            axis.plot(positions, total[f"{component}_{suffix}"], color=COMPONENTS[component][1], linewidth=1.5,
                      label=COMPONENTS[component][0])
        axis.plot(positions, total[f"total_{suffix}"], color=INK, linewidth=2.2, label="TOTAL")
        axis.axhline(1, color=INK_MUTED, linewidth=0.8)
        mark_the_future(axis, months, context.current_month, label=axis is axes[0])
        month_axis(axis, months)
        axis.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    drawn = total[[f"{component}_{suffix}" for component in present + ["total"]
                   for suffix in ("uplift", "uplift_sin_subida")]].to_numpy(dtype=float)
    low, high = np.nanmin(drawn), np.nanmax(drawn)
    for axis in axes[:2]:              # the same scale in both panels, around the uplifts (not from 0)
        axis.set_ylim(min(low, 1.0) - 0.02, high + 0.02)

    years = complete_years(total)
    by_year = total.groupby(total["_month"].map(lambda month: month.year))
    annex = {}
    for component in present + ["total"]:
        label = COMPONENTS[component][0] if component in COMPONENTS else "TOTAL"
        for suffix, name in (("", "with increases"), ("_sin_subida", "without increases")):
            if suffix and not has_increases:
                continue
            renewed = by_year[f"{component}_renovado{suffix}_usd"].sum()
            base = by_year[f"{component}_base{suffix}_usd"].sum()
            line = {}
            for year in years:
                value = renewed.get(year, np.nan) / base.get(year, np.nan) if base.get(year, 0) > 0 else np.nan
                line[str(year)] = f"{value:.3f}" if np.isfinite(value) else "—"
            annex[f"{label} · {name}"] = line
    share = by_year["parte_en_subida"].mean()
    annex["share of the base inside an increase"] = {str(year): f"{100 * share.get(year, np.nan):.0f} %" for year in years}
    annex = pd.DataFrame(annex).T
    annex_table(axes[2], annex, "Annex · uplift per year (renewed USD over the USD due of what renews)",
                first_column_width=0.36)
    write_annex(context, number, "total", annex)
    subtitle = ("Closed months: the real uplift. From the current month: the uplift the forecast applies. "
                + ("A row is inside a price increase when its price group and month fall in the cycle of an increase of "
                   "the calendar of step 8.4." if has_increases else
                   "This run has no price increase in its calendar: the two panels are the same."))
    titles(figure, f"{number} · The uplift of every component month by month, with and without price increases", subtitle)
    paths = [save(figure, chart_path(context, number, "uplift_por_componente"), top=0.93)]

    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names, height_per_row=2.8, same_scale=True)
    for axis, name in zip(axes, names):
        of_region = as_months(regions[regions[context.region] == name])
        region_months = pd.PeriodIndex(of_region["_month"])
        axis.plot(np.arange(len(of_region)), of_region["total_uplift"], color=INK, linewidth=1.6, label="with increases")
        if has_increases:
            axis.plot(np.arange(len(of_region)), of_region["total_uplift_sin_subida"], color=ORANGE, linewidth=1.4,
                      linestyle="--", label="without increases")
        axis.axhline(1, color=INK_MUTED, linewidth=0.8)
        mark_the_future(axis, region_months, context.current_month, label=False)
        month_axis(axis, region_months, compact=True)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    titles(figure, f"{number}a · The uplift of every region, with and without price increases", "Same scale in every panel.")
    paths.append(save(figure, chart_path(context, number, "uplift_por_region"), top=0.9))
    return paths


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 8 · UNDERSTANDING NEXT YEAR (added 28-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════════════
# THE QUESTION: the retention team's forecast of next year is reasonable, and ours is a baseline well below it.
# With the pipeline of the current year well built, where can the difference come from? Every possible cause is
# looked at against the real past, month by month:
#
#   8.1  where the difference with the retention team's forecast is, by subregion, and how our next year moves from
#        the current one: pipeline × renewal rate (in USD) × uplift, plus retail to subscription. For every
#        subregion, the rate and the uplift the reference would need with our pipeline
#   8.2  are the renewal rates of one year and the next alike? (one line per year)
#   8.3  is the effect of the price increases ending? The uplift of every year, with and without the rows inside
#        an increase, and the money the increases gave in the current year that next year no longer has
#   8.4  the price reference combinations (subregion × product × term × band, the way the increases are made)
#        month by month: uplift, price due and price paid, with the months inside an increase marked
#   8.5  is the composition of the pipeline of next year different? For every dimension, the change of the money
#        split into volume, mix (the composition at the yields of the current year) and performance
#   8.6  do the customer states mature well? The share of every state in what falls due: final in the closed
#        months, today and at due date in the future (a state without a maturation curve keeps today's share)
#   8.7  does the horizon of more than a year fall short? The rate predicted for every future month by months
#        ahead, against the real rate of the same month a year and two years before, and the exam by horizon
#   8.8  which region has the most favourable composition? Its pipeline at the yields of the whole portfolio, by
#        segment and by every dimension alone (product, term, band, renewal type, states, acquisition, price increase)
#   8.9  where a price increase was detected, and since when: the detector of every price group month by month and
#        the calendar of the increases, to compare with the tariff changes that were made
#   8.10 which term do the licences renew to? The share of every destination term and its uplift, month by month
#        (with the six columns of the term transitions in the extract)
#
# THE MONEY OF A ROW, the same in every question (USD):
#   pipeline   forecast_to_renew_USD_at_due: what falls due, after the maturation of the states (same total)
#   base       the USD due of what renews: real in the closed months (forecast_to_renew_USD_renewed), expected in
#              the future (s7_2_vencen_usd_al_vencer × s7_2_tasa)
#   renewed    forecast_renewed_USD: real in the closed months, expected in the future
#   rate (USD) = base / pipeline · uplift = renewed / base · renewed = pipeline × rate × uplift
#
# THE REFERENCE (forecast_retencion_2027.csv, next to this file): the retention team's forecast of next year per
# subregion (tr_regional_level_3), in thousands of USD, with the region of the team. Its LATAM line included Canada;
# the file has the subregions only. It is compared with our renewals plus retail to subscription (the reference is
# taken as the whole revenue of the year; the table also shows the renewals alone).

RETENTION_REFERENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast_retencion_2027.csv")
RETENTION_REFERENCE_UNIT_USD = 1000.0       # the reference is in thousands of USD
NOT_IN_REFERENCE = "not in the reference"
# the price reference combinations to follow in 8.4 (the increases are made by region, product, term and band): the
# ones written here (each value is found inside the value of the extract, without case: "premium" finds
# "Kaspersky Premium") and the biggest ones of the current year
REFERENCE_PRICE_CELLS = [{"subregion": "DACH", "product": "premium", "term": "1 year", "band": "10"}]
REFERENCE_PRICE_CELL_COUNT = 8
BAND_COLUMN = "tr_band"
TABLE_PORTFOLIO_EXAM_REPORT = "examen_cartera"      # step 7.1: the exam of the total of the portfolio by horizon
COMPOSITION_TOP_VALUES = 8                  # the values of a dimension drawn in 8.5 (the rest add up to "other")
YEAR_LINE_COLOURS = {-3: "#c9c7bf", -2: "#a9a69c", -1: "#7f7c72", 0: BLUE, 1: ORANGE}   # years around the current one


def due_usd(alias: str = "n") -> str:
    return f"COALESCE({alias}.forecast_to_renew_USD_at_due, {alias}.forecast_to_renew_USD, 0)"


def due_units(alias: str = "n") -> str:
    return f"COALESCE({alias}.forecast_to_renew_units_at_due, {alias}.forecast_to_renew_units, 0)"


def renewal_base_usd(context: ReportContext, alias: str = "n") -> str:
    """The USD due of what renews: real in the closed months, expected in the future (the base of the uplift)."""
    if "forecast_to_renew_USD_renewed" in context.core_columns:
        real_base = f"COALESCE({alias}.forecast_to_renew_USD_renewed, 0)"
    else:   # without the exact base: the renewed units at the row's price due
        real_base = (f"COALESCE({alias}.forecast_renewed_units, 0) * COALESCE({alias}.forecast_to_renew_USD, 0) "
                     f"/ NULLIF({alias}.forecast_to_renew_units, 0)")
    return (f"CASE WHEN {alias}.forecast_status = 'actual' THEN {real_base} "
            f"ELSE COALESCE({alias}.s7_2_vencen_usd_al_vencer, 0) * COALESCE({alias}.s7_2_tasa, 0) END")


def money_columns(context: ReportContext, alias: str = "n") -> str:
    """The money of a pipeline row as the base of a query: pipeline, base, renewed, units due and renewed."""
    return f"""{due_usd(alias)} AS pipeline_usd,
       {renewal_base_usd(context, alias)} AS base_usd,
       COALESCE({alias}.forecast_renewed_USD, 0) AS renovado_usd,
       {due_units(alias)} AS vencen_unidades,
       COALESCE({alias}.forecast_renewed_units, 0) AS renovadas_unidades"""


MONEY_MEASURES = ["SUM(pipeline_usd) AS pipeline_usd", "SUM(base_usd) AS base_usd", "SUM(renovado_usd) AS renovado_usd",
                  "SUM(vencen_unidades) AS vencen_unidades", "SUM(renovadas_unidades) AS renovadas_unidades",
                  "SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) AS tasa_unidades",
                  "SUM(base_usd) / NULLIF(SUM(pipeline_usd), 0) AS tasa_usd",
                  "SUM(renovado_usd) / NULLIF(SUM(base_usd), 0) AS uplift"]


def in_pipeline_sources(alias: str = "n") -> str:
    return f"{alias}.forecast_pipeline_source IN ({', '.join(repr(source) for source in PIPELINE_SOURCES)})"


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator and np.isfinite(denominator) else np.nan


def year_colour(year: int, current_year: int) -> str:
    return YEAR_LINE_COLOURS.get(year - current_year, INK_MUTED)


def english_month_axis(axis) -> None:
    axis.set_xticks(range(1, 13))
    axis.set_xticklabels(ENGLISH_MONTHS, fontsize=8)


def lines_by_year(axis, frame: pd.DataFrame, value: str, context: ReportContext) -> None:
    """One line per year over the 12 months; the months from the current one on, dashed (the forecast)."""
    current = pd.Period(context.current_month, freq="M")
    for year in sorted(frame["anio"].unique()):
        of_year = frame[frame["anio"] == year].sort_values("mes")
        months = of_year["mes"].astype(int).to_numpy()
        values = of_year[value].to_numpy(dtype=float)
        closed = np.array([pd.Period(f"{int(year)}-{month:02d}", freq="M") < current for month in months])
        colour = year_colour(int(year), context.current_year)
        width = 2.2 if year >= context.current_year else 1.5
        if closed.any():
            axis.plot(months[closed], values[closed], color=colour, linewidth=width, label=str(int(year)))
        if (~closed).any():
            start = max(0, int(np.argmax(~closed)) - 1) if closed.any() else 0
            axis.plot(months[start:], values[start:], color=colour, linewidth=width, linestyle="--",
                      label=None if closed.any() else f"{int(year)} (forecast)")
    english_month_axis(axis)


# ─── 8.1 · against the retention team's forecast ────────────────────────────────
def read_retention_reference(path: str = RETENTION_REFERENCE_FILE) -> pd.DataFrame:
    """The retention team's forecast per subregion, in USD (empty when the file is not there)."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["region", "referencia_usd", "region_del_equipo"])
    reference = pd.read_csv(path)
    reference["referencia_usd"] = reference["referencia"].astype(float) * RETENTION_REFERENCE_UNIT_USD
    return reference[["region", "referencia_usd", "region_del_equipo"]]


def sql_8_1(context: ReportContext) -> str:
    """8.1 · the current year and the next one by subregion: pipeline, base, renewed and retail to subscription."""
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio,
       CASE WHEN n.forecast_universe = 'pipeline' THEN {due_usd()} ELSE 0 END AS pipeline_usd,
       CASE WHEN n.forecast_universe = 'pipeline' THEN {renewal_base_usd(context)} ELSE 0 END AS base_usd,
       CASE WHEN n.forecast_universe = 'pipeline' THEN COALESCE(n.forecast_renewed_USD, 0) ELSE 0 END AS renovado_usd,
       CASE WHEN n.forecast_universe = 'pipeline' THEN {due_units()} ELSE 0 END AS vencen_unidades,
       CASE WHEN n.forecast_universe = 'pipeline' THEN COALESCE(n.forecast_renewed_units, 0) ELSE 0 END AS renovadas_unidades,
       CASE WHEN n.forecast_universe = 'time_series' THEN COALESCE(n.forecast_renewed_USD, 0) ELSE 0 END AS r2s_usd,
       {concept_columns()}
FROM {context.core} n
WHERE n.forecast_year IN ({context.current_year}, {context.next_year})
  AND ({in_pipeline_sources()} OR n.forecast_universe = 'time_series')"""
    measures = (MONEY_MEASURES + ["SUM(r2s_usd) AS r2s_usd", "SUM(renovado_usd) + SUM(r2s_usd) AS total_usd"]
                + [f"SUM({concept}) AS {concept}" for concept in CONCEPTS])
    return with_subtotals(context, base, measures, ["anio"],
                          comment=f"8.1 · {context.current_year} y {context.next_year} por subregión: pipeline, tasa, "
                                  f"uplift y retail to subscription (para comparar con el forecast del equipo de retención)")


# every concept of the money of a year (they add up to total_usd, the total of sff_resumen_negocio)
CONCEPTS = {"pipeline_real_usd": ("renewals of the pipeline of the extract", BLUE,
                                  "n.forecast_pipeline_source IN ('pipeline_renovado_real', 'pipeline_real_esperado')"),
            "renovacion_proyectada_usd": ("renewals of the projected renewals (7.2)", AQUA,
                                          "n.forecast_pipeline_source = 'pipeline_proyectada'"),
            "adquisicion_simulada_usd": ("renewals of the simulated acquisitions (5.1)", ORANGE,
                                         "n.forecast_pipeline_source = 'pipeline_simulada'"),
            "r2s_ingresos_usd": ("retail to subscription: revenue (6.1)", VIOLET,
                                 "n.forecast_universe = 'time_series' AND n.forecast_pipeline_source <> 'ts_reentrada'"),
            "r2s_reentrada_usd": ("retail to subscription: renewal of its re-entry (6.1)", "#a99ff0",
                                  "n.forecast_pipeline_source = 'ts_reentrada'")}


CONCEPT_SHORT_NAMES = {"pipeline_real_usd": "extract pipeline", "renovacion_proyectada_usd": "projected (7.2)",
                       "adquisicion_simulada_usd": "acquisitions (5.1)", "r2s_ingresos_usd": "R2S revenue",
                       "r2s_reentrada_usd": "R2S re-entry"}       # the headers of the annex


def concept_columns() -> str:
    return ",\n       ".join(f"CASE WHEN {condition} THEN COALESCE(n.forecast_renewed_USD, 0) ELSE 0 END AS {concept}"
                              for concept, (_, _, condition) in CONCEPTS.items())


def bridge_of(row_now: dict, row_next: dict) -> dict:
    """From the current year to the next: renewed = pipeline × rate × uplift, changed one factor at a time."""
    renewed_now = row_now["renovado_usd"]
    pipeline_ratio = safe_ratio(row_next["pipeline_usd"], row_now["pipeline_usd"])
    rate_ratio = safe_ratio(safe_ratio(row_next["base_usd"], row_next["pipeline_usd"]),
                            safe_ratio(row_now["base_usd"], row_now["pipeline_usd"]))
    after_pipeline = renewed_now * pipeline_ratio if np.isfinite(pipeline_ratio) else row_next["renovado_usd"]
    after_rate = after_pipeline * rate_ratio if np.isfinite(rate_ratio) else after_pipeline
    return {"pipeline_effect": after_pipeline - renewed_now, "rate_effect": after_rate - after_pipeline,
            "uplift_effect": row_next["renovado_usd"] - after_rate, "r2s_effect": row_next["r2s_usd"] - row_now["r2s_usd"]}


def subregion_comparison(table: pd.DataFrame, context: ReportContext, reference: pd.DataFrame) -> pd.DataFrame:
    """One row per subregion: our two years, the reference, the gap, the bridge and the implied rate and uplift."""
    level = context.subregion or context.region
    fine = table[table["orden_total"] == 0]
    if context.subregion:
        fine = fine[fine["orden_region"] == 0]
    rows = []
    for (region, name), block in fine.groupby([context.region, level], sort=False):
        by_year = {int(year): row for year, row in block.set_index("anio").iterrows()}
        now = by_year.get(context.current_year)
        upcoming = by_year.get(context.next_year)
        if now is None or upcoming is None:
            continue
        rows.append(dict(region=region, subregion=name, **comparison_values(now, upcoming)))
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    comparison = comparison.merge(reference, how="outer", left_on="subregion", right_on="region", suffixes=("", "_ref"))
    comparison["subregion"] = comparison["subregion"].fillna(comparison["region_ref"])
    comparison["region_del_equipo"] = comparison["region_del_equipo"].fillna(NOT_IN_REFERENCE)
    comparison = comparison.drop(columns=["region_ref"])
    comparison["en_los_datos"] = comparison["total_next"].notna()
    numeric = [column for column in comparison.columns if column not in ("region", "subregion", "region_del_equipo", "en_los_datos")]
    comparison[numeric] = comparison[numeric].astype(float)
    with_reference = comparison["referencia_usd"].notna() & comparison["en_los_datos"]
    comparison.loc[with_reference, "gap_usd"] = (comparison["total_next"] - comparison["referencia_usd"])[with_reference]
    comparison["gap_pct"] = comparison["gap_usd"] / comparison["referencia_usd"]
    renewals_needed = comparison["referencia_usd"] - comparison["r2s_next"]
    comparison["rate_needed"] = renewals_needed / (comparison["pipeline_next"] * comparison["uplift_next"])
    comparison["uplift_needed"] = renewals_needed / (comparison["pipeline_next"] * comparison["rate_usd_next"])
    comparison["pipeline_needed_usd"] = renewals_needed / (comparison["rate_usd_next"] * comparison["uplift_next"])
    return comparison.sort_values("gap_usd", na_position="last")


def comparison_values(now, upcoming) -> dict:
    values = {"pipeline_now": now["pipeline_usd"], "pipeline_next": upcoming["pipeline_usd"],
              "rate_usd_now": safe_ratio(now["base_usd"], now["pipeline_usd"]),
              "rate_usd_next": safe_ratio(upcoming["base_usd"], upcoming["pipeline_usd"]),
              "uplift_now": safe_ratio(now["renovado_usd"], now["base_usd"]),
              "uplift_next": safe_ratio(upcoming["renovado_usd"], upcoming["base_usd"]),
              "renewed_now": now["renovado_usd"], "renewed_next": upcoming["renovado_usd"],
              "r2s_now": now["r2s_usd"], "r2s_next": upcoming["r2s_usd"],
              "total_now": now["total_usd"], "total_next": upcoming["total_usd"]}
    values.update({f"{concept}_next": upcoming[concept] for concept in CONCEPTS})
    values.update(bridge_of(dict(now), dict(upcoming)))
    return values


def percent_text(value: float, decimals: int = 1) -> str:
    return f"{100 * value:.{decimals}f} %" if np.isfinite(value) else "—"


def ratio_text(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "—"


def comparison_annex(comparison: pd.DataFrame, context: ReportContext, label_column: str) -> pd.DataFrame:
    now, upcoming = context.current_year, context.next_year
    annex = {}
    for _, row in comparison.iterrows():
        annex[str(row[label_column])] = {
            f"ours {now}": money_label(row["total_now"]), f"ours {upcoming}": money_label(row["total_next"]),
            "reference": money_label(row["referencia_usd"]), "gap": money_label(row["gap_usd"]),
            "gap %": percent_text(row["gap_pct"]),
            "Δ pipeline": money_label(row["pipeline_effect"]), "Δ rate": money_label(row["rate_effect"]),
            "Δ uplift": money_label(row["uplift_effect"]), "Δ R2S": money_label(row["r2s_effect"]),
            f"rate {now}→{upcoming}": f"{percent_text(row['rate_usd_now'])} → {percent_text(row['rate_usd_next'])}",
            "rate needed": percent_text(row["rate_needed"]),
            f"uplift {now}→{upcoming}": f"{ratio_text(row['uplift_now'])} → {ratio_text(row['uplift_next'])}",
            "uplift needed": ratio_text(row["uplift_needed"])}
    return pd.DataFrame(annex).T


def team_region_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """The subregions added up to the regions of the retention team (the rates and uplifts recomputed from sums)."""
    sums = ["pipeline_now", "pipeline_next", "renewed_now", "renewed_next", "r2s_now", "r2s_next", "total_now",
            "total_next", "referencia_usd", "pipeline_effect", "rate_effect", "uplift_effect", "r2s_effect"]
    sums += [f"{concept}_next" for concept in CONCEPTS]
    known = comparison[comparison["en_los_datos"]].copy()
    for column, numerator, denominator in (("base_now", "rate_usd_now", "pipeline_now"), ("base_next", "rate_usd_next", "pipeline_next")):
        known[column] = known[numerator] * known[denominator]
    grouped = known.groupby("region_del_equipo")[sums + ["base_now", "base_next"]].sum(min_count=1)
    grouped.loc["TOTAL"] = grouped.sum(numeric_only=True)
    grouped["rate_usd_now"] = grouped["base_now"] / grouped["pipeline_now"]
    grouped["rate_usd_next"] = grouped["base_next"] / grouped["pipeline_next"]
    grouped["uplift_now"] = grouped["renewed_now"] / grouped["base_now"]
    grouped["uplift_next"] = grouped["renewed_next"] / grouped["base_next"]
    with_reference = known[known["referencia_usd"].notna()].groupby("region_del_equipo")["total_next"].sum()
    with_reference.loc["TOTAL"] = with_reference.sum()
    grouped["gap_usd"] = with_reference.reindex(grouped.index) - grouped["referencia_usd"]
    grouped["gap_pct"] = grouped["gap_usd"] / grouped["referencia_usd"]
    renewals_needed = grouped["referencia_usd"] - grouped["r2s_next"]
    grouped["rate_needed"] = renewals_needed / (grouped["pipeline_next"] * grouped["uplift_next"])
    grouped["uplift_needed"] = renewals_needed / (grouped["pipeline_next"] * grouped["rate_usd_next"])
    return grouped.reset_index()


def draw_concepts_against_reference(teams: pd.DataFrame, context: ReportContext, number: str) -> str:
    """Next year by region of the retention team: every concept of ours stacked, and the reference (one number per
    region, all concepts together) as a black mark; the annex has every concept, ours in total and the gap."""
    upcoming = context.next_year
    teams = teams.set_index("region_del_equipo")
    regions = [name for name in teams.index if name != TOTAL_LABEL]
    figure, axes = new_figure(15, 10, 2, 1, gridspec_kw={"height_ratios": [2.6, 2]})
    positions = np.arange(len(regions))
    left = np.zeros(len(regions))
    for concept, (label, colour, _) in CONCEPTS.items():
        widths = teams.loc[regions, f"{concept}_next"].fillna(0).to_numpy(dtype=float)
        axes[0].barh(positions, widths, left=left, color=colour, label=label, height=0.6)
        left += widths
    reference = teams.loc[regions, "referencia_usd"].to_numpy(dtype=float)
    axes[0].scatter(reference, positions, marker="D", color=INK, zorder=3, label="retention team (all concepts)")
    for position, (ours, theirs) in enumerate(zip(left, reference)):
        gap = f" · gap {money_label(ours - theirs)} ({percent_text(safe_ratio(ours - theirs, theirs), 0)})" if np.isfinite(theirs) else ""
        axes[0].annotate(f" ours {money_label(ours)}{gap}", (max(ours, np.nan_to_num(theirs)), position), va="center",
                         fontsize=8, color=INK_SECONDARY)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(regions, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(lambda value, _: money_label(value))
    axes[0].set_xlim(0, 1.3 * max(np.nanmax(left), np.nanmax(np.nan_to_num(reference))))
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[0].set_title(f"{upcoming} by region of the retention team: every concept of ours against their total",
                      fontsize=10, color=INK, loc="left")
    annex = {}
    for name in regions + [TOTAL_LABEL]:
        row = teams.loc[name]
        line = {CONCEPT_SHORT_NAMES[concept]: money_label(row[f"{concept}_next"]) for concept in CONCEPTS}
        line["ours, every concept"] = money_label(row["total_next"])
        line["retention team"] = money_label(row["referencia_usd"])
        line["gap"] = money_label(row["gap_usd"])
        line["gap %"] = percent_text(row["gap_pct"])
        annex[str(name)] = line
    annex = pd.DataFrame(annex).T
    annex_table(axes[1], annex, "Annex · every concept of ours, its total and the gap with the retention team",
                font_size=7.5, first_column_width=0.1)
    write_annex(context, number, "por_concepto", annex)
    total = teams.loc[TOTAL_LABEL]
    titles(figure, f"{number}b · Ours {upcoming} with every concept against the retention team's forecast",
           f"Ours: {money_label(total['total_next'])} = renewals of the pipeline of the extract, of the projected renewals and "
           f"of the simulated acquisitions + retail to subscription (its revenue and the renewal of its re-entry): the total "
           f"of sff_resumen_negocio. The reference is not split by concept.")
    return save(figure, chart_path(context, number, "conceptos_frente_a_la_referencia"), top=0.92)


def draw_8_1(table, context, number):
    reference = read_retention_reference(context.retention_reference_file or RETENTION_REFERENCE_FILE)
    comparison = subregion_comparison(table, context, reference)
    if comparison.empty:
        return []
    paths = []
    now, upcoming = context.current_year, context.next_year
    known = comparison[comparison["en_los_datos"]]
    totals = known[["total_now", "total_next", "pipeline_effect", "rate_effect", "uplift_effect", "r2s_effect"]].sum()
    reference_total = float(comparison["referencia_usd"].sum()) if comparison["referencia_usd"].notna().any() else np.nan

    # the bridge of the total, and the gap of every subregion with the reference
    figure, axes = new_figure(15, 12, 2, 1, gridspec_kw={"height_ratios": [1.6, 3]})
    steps = [(f"ours {now}", totals["total_now"], "total"), ("pipeline", totals["pipeline_effect"], "step"),
             ("renewal rate", totals["rate_effect"], "step"), ("uplift", totals["uplift_effect"], "step"),
             ("retail to subscription", totals["r2s_effect"], "step"), (f"ours {upcoming}", totals["total_next"], "total")]
    if np.isfinite(reference_total):
        steps += [("gap to the reference", reference_total - totals["total_next"], "step"),
                  (f"retention team {upcoming}", reference_total, "total")]
    level = 0.0
    for position, (label, amount, kind) in enumerate(steps):
        if kind == "total":
            axes[0].bar(position, amount, color=INK_SECONDARY if "retention" in label else BLUE, width=0.6)
            level = amount
            axes[0].annotate(money_label(amount), (position, amount), ha="center", va="bottom", fontsize=8.5, color=INK)
        else:
            bottom = level if amount >= 0 else level + amount
            axes[0].bar(position, abs(amount), bottom=bottom, color=AQUA if amount >= 0 else ORANGE, width=0.6)
            axes[0].annotate(f"{'+' if amount >= 0 else ''}{money_label(amount)}", (position, level + max(amount, 0)),
                             ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
            level += amount
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels([label for label, _, _ in steps], fontsize=8.5)
    money_axis(axes[0])
    low = min(totals["total_now"], totals["total_next"], reference_total if np.isfinite(reference_total) else np.inf)
    axes[0].set_ylim(bottom=max(0, 0.8 * low))
    axes[0].set_title(f"From {now} to {upcoming}: renewed = pipeline × renewal rate (USD) × uplift, one factor at a time",
                      fontsize=10, color=INK, loc="left")

    gaps = comparison[comparison["gap_usd"].notna()].sort_values("gap_usd")
    positions = np.arange(len(gaps))
    axes[1].barh(positions, gaps["gap_usd"], color=[ORANGE if gap < 0 else AQUA for gap in gaps["gap_usd"]])
    for position, (_, row) in zip(positions, gaps.iterrows()):
        axes[1].annotate(f" {money_label(row['gap_usd'])} ({percent_text(row['gap_pct'], 0)})",
                         (row["gap_usd"], position), va="center", ha="left" if row["gap_usd"] >= 0 else "right",
                         fontsize=7.5, color=INK_SECONDARY)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([f"{row['subregion']} ({row['region_del_equipo']})" for _, row in gaps.iterrows()], fontsize=8)
    axes[1].axvline(0, color=INK_MUTED, linewidth=0.8)
    axes[1].xaxis.set_major_formatter(lambda value, _: money_label(value))
    axes[1].invert_yaxis()
    if len(gaps):
        low_gap, high_gap = float(gaps["gap_usd"].min()), float(gaps["gap_usd"].max())
        axes[1].set_xlim(min(low_gap, 0) * 1.25, max(high_gap, 0) * 1.25 + 1)      # room for the labels
    else:
        axes[1].annotate("No subregion of the reference is in the data: check the names of the reference file.",
                         (0.5, 0.5), xycoords="axes fraction", ha="center", fontsize=10, color=INK_SECONDARY)
    axes[1].set_title(f"Our {upcoming} minus the retention team's forecast, by subregion (largest shortfall first)",
                      fontsize=10, color=INK, loc="left")
    missing = comparison[~comparison["en_los_datos"]]["subregion"].tolist()
    subtitle = (f"Ours = renewals + retail to subscription. Gap of the total: {money_label(totals['total_next'] - reference_total)} "
                f"({percent_text(safe_ratio(totals['total_next'] - reference_total, reference_total))})."
                if np.isfinite(reference_total) else "Without the reference file (forecast_retencion_2027.csv).")
    if missing:
        subtitle += f" Subregions of the reference not found in the data: {', '.join(map(str, missing))}."
    titles(figure, f"{number} · Where our {upcoming} differs from the retention team's forecast", subtitle)
    paths.append(save(figure, chart_path(context, number, "frente_al_equipo_de_retencion"), top=0.92))
    paths.append(draw_concepts_against_reference(team_region_comparison(comparison), context, number))

    # the annexes: by region of the team and by subregion
    teams = team_region_comparison(comparison)
    team_annex = comparison_annex(teams, context, "region_del_equipo")
    sub_annex = comparison_annex(comparison[comparison["en_los_datos"]], context, "subregion")
    figure, axes = new_figure(17, 0.27 * (len(team_annex) + len(sub_annex)) + 3.5, 2, 1,
                              gridspec_kw={"height_ratios": [len(team_annex) + 2, len(sub_annex) + 2]})
    annex_table(axes[0], team_annex, "Annex · by region of the retention team", font_size=7.5, first_column_width=0.1)
    annex_table(axes[1], sub_annex, "Annex · by subregion (largest shortfall first). Rate needed: the rate that reaches "
                                    "the reference with our pipeline and uplift; uplift needed: with our pipeline and rate",
                font_size=7.5, first_column_width=0.1)
    titles(figure, f"{number}a · Annex · the gap and what moves from {now} to {upcoming}",
           "Δ pipeline, Δ rate, Δ uplift and Δ R2S add up to ours next year minus ours this year.")
    paths.append(save(figure, chart_path(context, number, "anexo_diferencias"), top=0.96))
    write_annex(context, number, "por_region_del_equipo", team_annex)
    comparison.to_csv(os.path.join(context.output_folder, f"{number.replace('.', '_')}_anexo_por_subregion.csv"), index=False)
    print(f"[informe] {number} · annex por_subregion (the 10 largest shortfalls):")
    print(sub_annex.head(10).to_string())
    return paths


# ─── 8.2 · the renewal rates of every year ──────────────────────────────────────
def sql_8_2(context: ReportContext) -> str:
    """8.2 · the renewal rate month by month of the last years, the current one and the next one."""
    period = context.configuration.period_col
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio, CAST(SUBSTRING(n.{period}, 6, 2) AS INT) AS mes,
       n.{period} AS periodo,
       {money_columns(context)}
FROM {context.core} n
WHERE {in_pipeline_sources()} AND n.forecast_year >= {context.current_year - 3}"""
    return with_subtotals(context, base, MONEY_MEASURES, ["anio", "mes", "periodo"],
                          comment="8.2 · la tasa de renovación mes a mes de cada año (real y prevista)")


def yearly_values(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    by_year = frame.groupby("anio")[[numerator, denominator]].sum()
    return by_year[numerator] / by_year[denominator].replace(0, np.nan)


def draw_8_2(table, context, number):
    total = total_rows(table)
    figure, axes = new_figure(15, 11, 3, 1, gridspec_kw={"height_ratios": [2.4, 2.4, 1.6]})
    lines_by_year(axes[0], total, "tasa_unidades", context)
    percent_axis(axes[0], 1)
    axes[0].legend(fontsize=8, frameon=False, ncol=5, loc="lower left")
    axes[0].set_title("Renewal rate in units (renewed units / units due); dashed: forecast", fontsize=10, color=INK, loc="left")
    lines_by_year(axes[1], total, "tasa_usd", context)
    percent_axis(axes[1], 1)
    axes[1].set_title("Renewal rate in USD (USD due of what renews / USD due)", fontsize=10, color=INK, loc="left")
    regions = pd.concat([region_rows(table), total])
    annex = {}
    for name, block in regions.groupby(context.region, sort=False):
        units = yearly_values(block, "renovadas_unidades", "vencen_unidades")
        usd = yearly_values(block, "base_usd", "pipeline_usd")
        annex[str(name)] = {f"{int(year)}": f"{percent_text(units.get(year, np.nan))} · {percent_text(usd.get(year, np.nan))}"
                            for year in sorted(block["anio"].unique())}
    annex = pd.DataFrame(annex).T
    annex_table(axes[2], annex, "Annex · rate per year: in units · in USD (the current year: real months + forecast)",
                first_column_width=0.16)
    write_annex(context, number, "tasa_por_anio", annex)
    titles(figure, f"{number} · Are the renewal rates of one year and the next alike?",
           "One line per year over its months. A forecast year below the real ones, with the same pipeline, is less money.")
    paths = [save(figure, chart_path(context, number, "tasa_por_anio"), top=0.93)]
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names, height_per_row=2.8, same_scale=False)
    for axis, name in zip(axes, names):
        lines_by_year(axis, regions[regions[context.region] == name], "tasa_unidades", context)
        percent_axis(axis)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    titles(figure, f"{number}a · The renewal rate in units by region, one line per year", "Each region on its own scale.")
    paths.append(save(figure, chart_path(context, number, "tasa_por_region"), top=0.9))
    return paths


# ─── 8.3 · the price increases: is their effect ending? ─────────────────────────
def sql_8_3(context: ReportContext) -> str:
    """8.3 · the uplift month by month of every year, with and without the rows inside a price increase."""
    period = context.configuration.period_col
    increases = price_increase_lookup(context)
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio, CAST(SUBSTRING(n.{period}, 6, 2) AS INT) AS mes,
       n.{period} AS periodo,
       {increases.flag} AS en_subida,
       {renewal_base_usd(context)} AS base_usd,
       COALESCE(n.forecast_renewed_USD, 0) AS renovado_usd
FROM {context.core} n
{increases.join}
WHERE {in_pipeline_sources()} AND n.forecast_year >= {context.current_year - 3}"""
    measures = ["SUM(base_usd) AS base_usd", "SUM(renovado_usd) AS renovado_usd",
                "SUM(CASE WHEN en_subida = 1 THEN base_usd ELSE 0 END) AS base_en_subida_usd",
                "SUM(CASE WHEN en_subida = 1 THEN renovado_usd ELSE 0 END) AS renovado_en_subida_usd",
                "SUM(renovado_usd) / NULLIF(SUM(base_usd), 0) AS uplift",
                "SUM(CASE WHEN en_subida = 0 THEN renovado_usd ELSE 0 END) / "
                "NULLIF(SUM(CASE WHEN en_subida = 0 THEN base_usd ELSE 0 END), 0) AS uplift_sin_subida"]
    return with_subtotals(context, base, measures, ["anio", "mes", "periodo"], ctes=increases.ctes,
                          comment="8.3 · el uplift mes a mes de cada año, con y sin las filas dentro de una subida de precio")


def increase_effect(block: pd.DataFrame) -> dict:
    """Per year: the uplift with and without increases, the share of the base inside one, and the money the
    increases add (what the rows inside renewed minus what they would renew at the uplift without increases)."""
    by_year = block.groupby("anio")[["base_usd", "renovado_usd", "base_en_subida_usd", "renovado_en_subida_usd"]].sum()
    outside_base = by_year["base_usd"] - by_year["base_en_subida_usd"]
    outside_renewed = by_year["renovado_usd"] - by_year["renovado_en_subida_usd"]
    without = outside_renewed / outside_base.replace(0, np.nan)
    effect = by_year["renovado_en_subida_usd"] - by_year["base_en_subida_usd"] * without
    return {"uplift": by_year["renovado_usd"] / by_year["base_usd"].replace(0, np.nan), "uplift_without": without,
            "share_inside": by_year["base_en_subida_usd"] / by_year["base_usd"].replace(0, np.nan), "effect_usd": effect}


def draw_8_3(table, context, number):
    total = total_rows(table)
    figure, axes = new_figure(15, 12, 3, 1, gridspec_kw={"height_ratios": [2.4, 2.4, 2]})
    lines_by_year(axes[0], total, "uplift", context)
    axes[0].legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    axes[0].set_title("Uplift with every row (renewed USD / USD due of what renews)", fontsize=10, color=INK, loc="left")
    lines_by_year(axes[1], total, "uplift_sin_subida", context)
    axes[1].set_title("Uplift without the rows inside a price increase", fontsize=10, color=INK, loc="left")
    values = total[["uplift", "uplift_sin_subida"]].to_numpy(dtype=float)
    for axis in axes[:2]:
        axis.set_ylim(np.nanmin(values) - 0.02, np.nanmax(values) + 0.02)
    groups = pd.concat([region_rows(table), total])
    if context.subregion:
        groups = pd.concat([groups, table[table["orden_region"] == 0].assign(
            **{context.region: lambda frame: frame[context.region].astype(str) + " · " + frame[context.subregion].astype(str)})])
    annex = {}
    years = sorted(int(year) for year in total["anio"].unique())
    for name, block in groups.groupby(context.region, sort=False):
        effect = increase_effect(block)
        line = {str(year): f"{ratio_text(effect['uplift'].get(year, np.nan))} / {ratio_text(effect['uplift_without'].get(year, np.nan))}"
                for year in years}
        for year in (context.current_year - 1, context.current_year, context.next_year):
            line[f"increase share {year}"] = percent_text(effect["share_inside"].get(year, np.nan), 0)
            line[f"increase effect {year}"] = money_label(effect["effect_usd"].get(year, np.nan))
        annex[str(name)] = line
    annex = pd.DataFrame(annex).T
    shown = annex.head(region_rows(table)[context.region].nunique() + 1)       # the regions and the TOTAL
    annex_table(axes[2], shown, "Annex · uplift per year with / without increases; share of the base inside an increase "
                                "and the money the increases add", font_size=7.5, first_column_width=0.1)
    write_annex(context, number, "efecto_de_las_subidas", annex)
    current_effect = increase_effect(total)["effect_usd"]
    subtitle = (f"The increases added {money_label(current_effect.get(context.current_year, np.nan))} in {context.current_year} "
                f"and add {money_label(current_effect.get(context.next_year, np.nan))} in {context.next_year}: "
                "a cycle that ends is money next year does not have. "
                + ("" if context.has_table(TABLE_PRICE_INCREASE_CALENDAR) else "This run has no calendar of increases."))
    titles(figure, f"{number} · Is the effect of the price increases ending?", subtitle)
    paths = [save(figure, chart_path(context, number, "efecto_de_las_subidas"), top=0.93)]
    regions = region_rows(table)
    names = list(regions[context.region].drop_duplicates())
    figure, axes = small_multiples(names, height_per_row=2.8, same_scale=True)
    for axis, name in zip(axes, names):
        lines_by_year(axis, regions[regions[context.region] == name], "uplift", context)
        axis.set_title(str(name), fontsize=10, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    titles(figure, f"{number}a · The uplift by region, one line per year", "Same scale in every panel.")
    paths.append(save(figure, chart_path(context, number, "uplift_por_region"), top=0.9))
    if context.subregion:
        print(f"[informe] {number} · the subregions are in {number.replace('.', '_')}_anexo_efecto_de_las_subidas.csv")
    return paths


# ─── 8.4 · the price reference combinations ─────────────────────────────────────
def price_reference_dims(context: ReportContext) -> dict:
    """The columns of a price reference combination (the way the increases are made), the ones the core has."""
    configuration = context.configuration
    dims = {"subregion": context.subregion or context.region, "product": context.product,
            "term": configuration.term_column, "band": BAND_COLUMN}
    return {key: column for key, column in dims.items() if column and column in context.core_columns}


def sql_8_4(context: ReportContext) -> str:
    """8.4 · every price reference combination month by month: units, pipeline, base, renewed and the increase."""
    period = context.configuration.period_col
    dims = price_reference_dims(context)
    dim_columns = ", ".join(f"n.{column} AS {key}" for key, column in dims.items())
    keys = ", ".join(dims) + ", anio, periodo"
    increases = price_increase_lookup(context)
    increase_ctes = "".join(f"{name} AS (\n{textwrap.indent(sql.strip(), '    ')}\n),\n" for name, sql in increases.ctes)
    return f"""-- 8.4 · las combinaciones de referencia de precio (subregión × producto × plazo × banda) mes a mes
WITH {increase_ctes}base AS (
    SELECT {dim_columns}, n.forecast_year AS anio, n.{period} AS periodo,
           {increases.flag} AS en_subida,
           {money_columns(context)}
    FROM {context.core} n
    {increases.join}
    WHERE {in_pipeline_sources()} AND n.forecast_year >= {context.current_year - 3}
)
SELECT {keys},
       SUM(pipeline_usd) AS pipeline_usd, SUM(base_usd) AS base_usd, SUM(renovado_usd) AS renovado_usd,
       SUM(vencen_unidades) AS vencen_unidades, SUM(renovadas_unidades) AS renovadas_unidades,
       MAX(en_subida) AS en_subida
FROM base
GROUP BY {keys}
ORDER BY {keys};"""


def chosen_price_cells(table: pd.DataFrame, context: ReportContext) -> list:
    """The combinations written in REFERENCE_PRICE_CELLS that exist, and the biggest of the current year."""
    keys = [key for key in ("subregion", "product", "term", "band") if key in table.columns]
    sizes = (table[table["anio"] == context.current_year].groupby(keys)["pipeline_usd"].sum().sort_values(ascending=False))
    chosen = []
    for wanted in REFERENCE_PRICE_CELLS:
        matches = sizes.index.to_frame(index=False)
        mask = pd.Series(True, index=matches.index)
        for key, value in wanted.items():
            if key in matches.columns:
                mask &= matches[key].astype(str).str.lower().str.contains(str(value).lower(), regex=False)
        for _, match in matches[mask].iterrows():
            cell = tuple(match[key] for key in keys)
            if cell not in chosen:
                chosen.append(cell)
    for cell in sizes.index[:REFERENCE_PRICE_CELL_COUNT]:
        cell = cell if isinstance(cell, tuple) else (cell,)
        if cell not in chosen:
            chosen.append(cell)
    return chosen[:REFERENCE_PRICE_CELL_COUNT + len(REFERENCE_PRICE_CELLS)]


def shade_increases(axis, of_cell: pd.DataFrame) -> None:
    """The months inside a price increase, shaded."""
    for position, inside in enumerate(of_cell["en_subida"].to_numpy()):
        if inside:
            axis.axvspan(position - 0.5, position + 0.5, color=ORANGE, alpha=0.12, linewidth=0, zorder=0)


def draw_8_4(table, context, number):
    keys = [key for key in ("subregion", "product", "term", "band") if key in table.columns]
    cells = chosen_price_cells(table, context)
    if not cells:
        return []
    figure_uplift, axes_uplift = small_multiples([str(cell) for cell in cells], height_per_row=3.0, same_scale=False)
    figure_price, axes_price = small_multiples([str(cell) for cell in cells], height_per_row=3.0, same_scale=False)
    annex = {}
    for axis_uplift, axis_price, cell in zip(axes_uplift, axes_price, cells):
        mask = np.ones(len(table), dtype=bool)
        for key, value in zip(keys, cell):
            mask &= (table[key] == value).to_numpy()
        of_cell = as_months(table[mask])
        months = pd.PeriodIndex(of_cell["_month"])
        positions = np.arange(len(of_cell))
        uplift = of_cell["renovado_usd"] / of_cell["base_usd"].replace(0, np.nan)
        price_due = of_cell["pipeline_usd"] / of_cell["vencen_unidades"].replace(0, np.nan)
        price_paid = of_cell["renovado_usd"] / of_cell["renovadas_unidades"].replace(0, np.nan)
        name = " · ".join(str(value) for value in cell)
        for axis in (axis_uplift, axis_price):
            shade_increases(axis, of_cell)
            mark_the_future(axis, months, context.current_month, label=False)
            month_axis(axis, months, compact=True)
            axis.set_title(name, fontsize=8.5, color=INK, loc="left")
        axis_uplift.plot(positions, uplift, color=INK, linewidth=1.6, label="uplift")
        axis_uplift.axhline(1, color=INK_MUTED, linewidth=0.8)
        axis_price.plot(positions, price_due, color=BLUE, linewidth=1.5, label="USD due per unit due")
        axis_price.plot(positions, price_paid, color=AQUA, linewidth=1.5, label="USD paid per renewed unit")
        money_axis(axis_price)
        by_year = of_cell.groupby("anio")[["pipeline_usd", "base_usd", "renovado_usd", "vencen_unidades", "renovadas_unidades"]].sum()
        line = {}
        for year, row in by_year.iterrows():
            line[str(int(year))] = (f"{ratio_text(safe_ratio(row['renovado_usd'], row['base_usd']))} · "
                                    f"{money_label(safe_ratio(row['pipeline_usd'], row['vencen_unidades']))} · "
                                    f"{percent_text(safe_ratio(row['renovadas_unidades'], row['vencen_unidades']), 0)}")
        annex[name] = line
    axes_uplift[0].legend(fontsize=7, frameon=False, loc="upper left")
    axes_price[0].legend(fontsize=7, frameon=False, loc="upper left")
    annex = pd.DataFrame(annex).T
    titles(figure_uplift, f"{number} · The price reference combinations: the uplift month by month",
           "Subregion · product · term · band. Orange: months inside a price increase; grey: the forecast. "
           "After an increase the uplift steps up for a cycle and then comes back: next year may not have that step.")
    paths = [save(figure_uplift, chart_path(context, number, "combinaciones_uplift"), top=0.88)]
    titles(figure_price, f"{number}a · The price reference combinations: price due and price paid per unit",
           "Blue: USD due per unit due (the tariff of the licence that expires); green: USD paid per renewed unit.")
    paths.append(save(figure_price, chart_path(context, number, "combinaciones_precio"), top=0.88))
    figure, axis = new_figure(15, 0.3 * len(annex) + 1.8)
    annex_table(axis, annex, "uplift · USD due per unit · rate in units, per year", font_size=7.5, first_column_width=0.3)
    titles(figure, f"{number}b · Annex · the price reference combinations per year", "")
    paths.append(save(figure, chart_path(context, number, "anexo_combinaciones"), top=0.95))
    write_annex(context, number, "combinaciones", annex)
    return paths


# ─── 8.5 · the composition of the pipeline ──────────────────────────────────────
def composition_dims(context: ReportContext) -> list:
    """The dimensions whose mix is compared: the business dims but the regions, the customer states, the acquisition
    column and the component of the pipeline."""
    configuration = context.configuration
    region_columns = set(year_by_region_columns(configuration)) | set(configuration.ts_region_columns or [])
    dims = [dim for dim in configuration.business_mandatory_dims if dim not in region_columns]
    dims += [state for state in context.states if state not in dims]
    if configuration.acquisition_column and configuration.acquisition_column not in dims:
        dims.append(configuration.acquisition_column)
    return [dim for dim in dims if dim in context.core_columns]


def sql_8_5(context: ReportContext) -> str:
    """8.5 · the pipeline, base and renewed of the current year and the next one by the value of every dimension."""
    parts = []
    for dim in composition_dims(context) + ["componente"]:
        value = pipeline_component(context) if dim == "componente" else f"CAST(n.{dim} AS VARCHAR(200))"
        parts.append(f"""SELECT '{dim}' AS dimension, {value} AS valor, n.forecast_year AS anio,
       {money_columns(context)}
FROM {context.core} n
WHERE {in_pipeline_sources()} AND n.forecast_year IN ({context.current_year}, {context.next_year})""")
    union = "\nUNION ALL\n".join(parts)
    return f"""-- 8.5 · la composición de la pipeline de {context.current_year} y {context.next_year} por cada dimensión
WITH base AS (
{textwrap.indent(union, '    ')}
)
SELECT dimension, valor, anio,
       SUM(pipeline_usd) AS pipeline_usd, SUM(base_usd) AS base_usd, SUM(renovado_usd) AS renovado_usd,
       SUM(vencen_unidades) AS vencen_unidades, SUM(renovadas_unidades) AS renovadas_unidades
FROM base
GROUP BY dimension, valor, anio
ORDER BY dimension, valor, anio;"""


def mix_split(of_dimension: pd.DataFrame, now: int, upcoming: int) -> tuple:
    """For one dimension: the table of its values (pipeline share and yield of both years) and the change of the
    renewed money split into volume, mix and performance:
       volume       (pipeline next − pipeline now) × yield now
       mix          Σ pipeline next of a value × its yield now − pipeline next × yield now
       performance  renewed next − Σ pipeline next of a value × its yield now
    yield = renewed / pipeline. The three add up to renewed next − renewed now."""
    wide = of_dimension.pivot_table(index="valor", columns="anio", values=["pipeline_usd", "renovado_usd"], aggfunc="sum").fillna(0.0)
    pipeline_now, pipeline_next = wide[("pipeline_usd", now)], wide[("pipeline_usd", upcoming)]
    renewed_now, renewed_next = wide[("renovado_usd", now)], wide[("renovado_usd", upcoming)]
    yield_total_now = safe_ratio(renewed_now.sum(), pipeline_now.sum())
    yield_now = (renewed_now / pipeline_now.replace(0, np.nan)).fillna(yield_total_now)
    at_yields_now = float((pipeline_next * yield_now).sum())
    split = {"volume": (pipeline_next.sum() - pipeline_now.sum()) * yield_total_now,
             "mix": at_yields_now - pipeline_next.sum() * yield_total_now,
             "performance": renewed_next.sum() - at_yields_now}
    values = pd.DataFrame({"share_now": pipeline_now / pipeline_now.sum(), "share_next": pipeline_next / pipeline_next.sum(),
                           "yield_now": renewed_now / pipeline_now.replace(0, np.nan),
                           "yield_next": renewed_next / pipeline_next.replace(0, np.nan),
                           "pipeline_now": pipeline_now, "pipeline_next": pipeline_next})
    return values.sort_values("pipeline_now", ascending=False), split


def draw_8_5(table, context, number):
    now, upcoming = context.current_year, context.next_year
    table = table.copy()
    table["valor"] = table["valor"].fillna("(empty)").astype(str)
    dims = list(table["dimension"].drop_duplicates())
    splits, value_tables = {}, {}
    for dim in list(dims):
        values, split = mix_split(table[table["dimension"] == dim], now, upcoming)
        if len(values) < 2:            # one value only (a level that groups everything): no mix to see
            dims.remove(dim)
            continue
        splits[dim], value_tables[dim] = split, values
    figure, axes = small_multiples(dims, height_per_row=3.0, same_scale=False)
    for axis, dim in zip(axes, dims):
        values = value_tables[dim]
        if len(values) > COMPOSITION_TOP_VALUES:
            top = values.head(COMPOSITION_TOP_VALUES - 1)
            rest = values.iloc[COMPOSITION_TOP_VALUES - 1:]
            other = pd.DataFrame({"share_now": [rest["share_now"].sum()], "share_next": [rest["share_next"].sum()],
                                  "yield_now": [np.nan], "yield_next": [np.nan]}, index=["other"])
            values = pd.concat([top, other])
        positions = np.arange(len(values))
        axis.barh(positions - 0.2, values["share_now"], 0.38, color=BLUE, label=str(now))
        axis.barh(positions + 0.2, values["share_next"], 0.38, color=ORANGE, label=str(upcoming))
        for position, (_, row) in zip(positions, values.iterrows()):
            if np.isfinite(row["yield_now"]):
                axis.annotate(f" yield {percent_text(row['yield_now'], 0)} → {percent_text(row['yield_next'], 0)}",
                              (max(row["share_now"], row["share_next"]), position), va="center", fontsize=6.5, color=INK_SECONDARY)
        axis.set_yticks(positions)
        axis.set_yticklabels([str(label)[:28] for label in values.index], fontsize=7)
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f} %")
        split = splits[dim]
        axis.set_title(f"{dim} · mix {money_label(split['mix'])} · performance {money_label(split['performance'])}",
                       fontsize=8.5, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    titles(figure, f"{number} · Is the pipeline of {upcoming} made of the same as the one of {now}?",
           f"Share of the pipeline of every value in {now} (blue) and {upcoming} (orange), and its yield (renewed / pipeline). "
           "Mix: what the change of composition costs at the yields of this year; performance: what the yields of every value change.")
    paths = [save(figure, chart_path(context, number, "composicion_por_dimension"), top=0.9)]

    summary = pd.DataFrame(splits).T[["volume", "mix", "performance"]]
    figure, axes = new_figure(15, 9, 2, 1, gridspec_kw={"height_ratios": [2.4, 2]})
    positions = np.arange(len(summary))
    axes[0].bar(positions - 0.2, summary["mix"], 0.38, color=VIOLET, label="mix (composition)")
    axes[0].bar(positions + 0.2, summary["performance"], 0.38, color=ORANGE, label="performance (yield of every value)")
    axes[0].axhline(0, color=INK_MUTED, linewidth=0.8)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(summary.index, fontsize=8, rotation=20, ha="right")
    money_axis(axes[0])
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title(f"Change of the renewed money {now} → {upcoming}, split by every dimension (the volume is the same in all)",
                      fontsize=10, color=INK, loc="left")
    annex = summary.apply(lambda column: column.map(money_label))
    annex["volume + mix + performance"] = summary.sum(axis=1).map(money_label)
    annex_table(axes[1], annex, "Annex · volume, mix and performance per dimension", first_column_width=0.25)
    write_annex(context, number, "mezcla", annex)
    titles(figure, f"{number}a · How much of the change is composition and how much is the yield of every value", "")
    paths.append(save(figure, chart_path(context, number, "mezcla_y_rendimiento"), top=0.93))
    details = pd.concat({dim: values for dim, values in value_tables.items()}, names=["dimension", "valor"])
    details.to_csv(os.path.join(context.output_folder, f"{number.replace('.', '_')}_anexo_valores.csv"))
    return paths


# ─── 8.6 · the customer states: do they mature well? ────────────────────────────
def sql_8_6(context: ReportContext) -> Optional[str]:
    """8.6 · every customer state month by month: units due in it today and at due date, and its renewals."""
    if not context.states:
        return None
    period = context.configuration.period_col
    columns = []
    for state in context.states:
        columns += [f"CASE WHEN n.{state} = 1 THEN COALESCE(n.forecast_to_renew_units, 0) ELSE 0 END AS {state}_hoy",
                    f"CASE WHEN n.{state} = 1 THEN {due_units()} ELSE 0 END AS {state}_al_vencer",
                    f"CASE WHEN n.{state} = 1 THEN COALESCE(n.forecast_renewed_units, 0) ELSE 0 END AS {state}_renovadas"]
    neutral = " AND ".join(f"COALESCE(n.{state}, 0) <> 1" for state in context.states)
    columns += [f"CASE WHEN {neutral} THEN {due_units()} ELSE 0 END AS neutras_al_vencer",
                f"CASE WHEN {neutral} THEN COALESCE(n.forecast_renewed_units, 0) ELSE 0 END AS neutras_renovadas"]
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio, CAST(SUBSTRING(n.{period}, 6, 2) AS INT) AS mes,
       n.{period} AS periodo,
       {due_units()} AS vencen_unidades,
       {(',' + chr(10) + '       ').join(columns)}
FROM {context.core} n
WHERE {in_pipeline_sources()} AND n.forecast_year >= {context.current_year - 3}"""
    measures = ["SUM(vencen_unidades) AS vencen_unidades", "SUM(neutras_al_vencer) AS neutras_al_vencer",
                "SUM(neutras_renovadas) AS neutras_renovadas"]
    for state in context.states:
        measures += [f"SUM({state}_hoy) AS {state}_hoy", f"SUM({state}_al_vencer) AS {state}_al_vencer",
                     f"SUM({state}_renovadas) AS {state}_renovadas",
                     f"SUM({state}_al_vencer) / NULLIF(SUM(vencen_unidades), 0) AS {state}_parte"]
    return with_subtotals(context, base, measures, ["anio", "mes", "periodo"],
                          comment="8.6 · cada estado del cliente mes a mes: unidades hoy y al vencer, y sus renovaciones")


def states_with_curve(context: ReportContext) -> list:
    if not context.has_table(TABLE_MATURATION_CURVE):
        return list(context.configuration.maturation_dims or [])
    curve = pd.read_sql(f"SELECT DISTINCT estado, se_usa FROM {context.curve}", context.engine)
    return sorted(curve[curve["se_usa"] == 1]["estado"].astype(str).unique())


def draw_8_6(table, context, number):
    total = as_months(total_rows(table))
    months = pd.PeriodIndex(total["_month"])
    positions = np.arange(len(total))
    with_curve = states_with_curve(context)
    figure, axes = new_figure(15, 3.2 * len(context.states) + 3, len(context.states) + 1, 1,
                              gridspec_kw={"height_ratios": [2.4] * len(context.states) + [1.8]})
    annex = {}
    for axis, state in zip(axes, context.states):
        share_at_due = total[f"{state}_al_vencer"] / total["vencen_unidades"].replace(0, np.nan)
        share_today = total[f"{state}_hoy"] / total["vencen_unidades"].replace(0, np.nan)
        current = pd.Period(context.current_month, freq="M")
        closed = np.array([month < current for month in months])
        axis.plot(positions[closed], share_at_due[closed], color=BLUE, linewidth=2, label="real, final (closed months)")
        axis.plot(positions[~closed], share_today[~closed], color=INK_MUTED, linewidth=1.5, linestyle=":",
                  label="today (what the extract shows)")
        axis.plot(positions[~closed], share_at_due[~closed], color=ORANGE, linewidth=2, linestyle="--",
                  label="expected at due date (forecast)")
        a_year_before = share_at_due.shift(12)
        axis.plot(positions[~closed], a_year_before[~closed], color=BLUE, linewidth=1, alpha=0.5,
                  label="real, same month a year before")
        mark_the_future(axis, months, context.current_month, label=axis is axes[0])
        month_axis(axis, months)
        percent_axis(axis, 1)
        curve_text = ("dated: follows its maturation curve" if state in with_curve
                      else "not dated: no curve, it goes to its historical share")
        axis.set_title(f"{state} · share of the units due ({curve_text})", fontsize=10, color=INK, loc="left")
        by_year = total.groupby("anio")[[f"{state}_al_vencer", "vencen_unidades", f"{state}_renovadas",
                                         "neutras_al_vencer", "neutras_renovadas"]].sum()
        line = {}
        for year, row in by_year.iterrows():
            line[f"share {int(year)}"] = percent_text(safe_ratio(row[f"{state}_al_vencer"], row["vencen_unidades"]))
        current_year_row = by_year.loc[context.current_year] if context.current_year in by_year.index else None
        if current_year_row is not None:
            state_rate = safe_ratio(current_year_row[f"{state}_renovadas"], current_year_row[f"{state}_al_vencer"])
            neutral_rate = safe_ratio(current_year_row["neutras_renovadas"], current_year_row["neutras_al_vencer"])
            line[f"rate in state {context.current_year}"] = percent_text(state_rate, 0)
            line[f"rate neutral {context.current_year}"] = percent_text(neutral_rate, 0)
            next_units = by_year["vencen_unidades"].get(context.next_year, np.nan)
            line[f"units lost per +1 pp {context.next_year}"] = f"{0.01 * next_units * (neutral_rate - state_rate):,.0f}"
        annex[state] = line
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    annex = pd.DataFrame(annex).T
    annex_table(axes[-1], annex, "Annex · share of every state per year (closed months final, future at due date), "
                                 "its rate and the units one more point of share costs", font_size=7.5,
                first_column_width=0.12)
    write_annex(context, number, "estados", annex)
    titles(figure, f"{number} · Do the customer states mature well?",
           f"States with a dated maturation curve: {', '.join(with_curve) if with_curve else 'none'}; the rest go to the share "
           "they had in the past. A state whose real final share is higher than the one the forecast expects at due date is "
           "renewal the forecast counts and will not come (and lower, the opposite).")
    return [save(figure, chart_path(context, number, "maduracion_de_los_estados"), top=0.94)]


# ─── 8.7 · the horizon: does more than a year ahead fall short? ─────────────────
def sql_8_7(context: ReportContext) -> str:
    """8.7 · the rate of every month (real in the closed ones, predicted in the future) and the months ahead."""
    period = context.configuration.period_col
    return f"""-- 8.7 · la tasa de cada mes (real o prevista) y los meses de antelación de la predicción
SELECT n.forecast_year AS anio, n.{period} AS periodo,
       MAX(n.s7_2_h) AS meses_vista,
       SUM({due_units()}) AS vencen_unidades,
       SUM(COALESCE(n.forecast_renewed_units, 0)) AS renovadas_unidades,
       SUM(COALESCE(n.forecast_renewed_units, 0)) / NULLIF(SUM({due_units()}), 0) AS tasa_unidades,
       SUM({renewal_base_usd(context)}) / NULLIF(SUM({due_usd()}), 0) AS tasa_usd
FROM {context.core} n
WHERE {in_pipeline_sources()} AND n.forecast_year >= {context.current_year - 3}
GROUP BY n.forecast_year, n.{period}
ORDER BY n.{period};"""


def draw_8_7(table, context, number):
    monthly = as_months(table)
    rate = pd.Series(monthly["tasa_unidades"].to_numpy(dtype=float), index=pd.PeriodIndex(monthly["_month"]))
    current = pd.Period(context.current_month, freq="M")
    future = [month for month in rate.index if month >= current]
    if not future:
        return []
    figure, axes = new_figure(15, 11, 3, 1, gridspec_kw={"height_ratios": [2.4, 2.2, 1.6]})
    positions = np.arange(len(future))
    ahead = monthly.set_index(pd.PeriodIndex(monthly["_month"]))["meses_vista"]
    labels = [f"{ENGLISH_MONTHS[month.month - 1]} {month.year}\nh={int(np.nan_to_num(ahead.get(month, 0)))}" for month in future]
    predicted = [rate.get(month) for month in future]
    year_before = [rate.get(month - 12, np.nan) for month in future]
    two_years_before = [rate.get(month - 24, np.nan) for month in future]
    axes[0].plot(positions, predicted, color=ORANGE, linewidth=2.2, marker="o", markersize=3, label="predicted")
    axes[0].plot(positions, year_before, color=BLUE, linewidth=1.6, label="same month a year before (real, or predicted)")
    axes[0].plot(positions, two_years_before, color=INK_MUTED, linewidth=1.2, label="same month two years before (real)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, fontsize=7)
    percent_axis(axes[0], 1)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_title("Renewal rate in units of every future month, by months ahead (h)", fontsize=10, color=INK, loc="left")
    # the last REAL rate of the same calendar month: a year before when that month is closed, two years before if not
    last_real = [rate.get(month - 12) if month - 12 < current else rate.get(month - 24, np.nan) for month in future]
    difference = np.array(predicted, dtype=float) - np.array(last_real, dtype=float)
    axes[1].bar(positions, 100 * difference, color=[ORANGE if value < 0 else AQUA for value in difference])
    axes[1].axhline(0, color=INK_MUTED, linewidth=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:+.1f} pp")
    axes[1].set_title("Predicted minus the last real rate of the same calendar month (a year before; two years before "
                      "from h = 13) in pp: a slope with h is the technique drifting with the horizon",
                      fontsize=10, color=INK, loc="left")
    exam_annex = horizon_exam(context)
    annex_table(axes[2], exam_annex, "Annex · the exam of the portfolio by horizon (the framework, closed months)",
                first_column_width=0.3)
    write_annex(context, number, "examen_por_horizonte", exam_annex)
    slope = np.polyfit(positions[np.isfinite(difference)], difference[np.isfinite(difference)], 1)[0] if np.isfinite(difference).sum() > 2 else np.nan
    titles(figure, f"{number} · Does the horizon of more than a year fall short?",
           f"The forecast reaches h = {int(np.nanmax(ahead.to_numpy(dtype=float)))} months; the exam judges the horizons it "
           f"has (the further ones take the band of the last). Slope of the difference: {100 * 12 * slope:+.2f} pp per year "
           "of horizon." if np.isfinite(slope) else "")
    return [save(figure, chart_path(context, number, "horizonte"), top=0.92)]


def horizon_exam(context: ReportContext) -> pd.DataFrame:
    """The exam of the portfolio (step 2.6/7.1): the error of the framework on the total, by horizon."""
    if not context.has_table(TABLE_PORTFOLIO_EXAM_REPORT):
        return pd.DataFrame()
    exam = pd.read_sql(f"SELECT * FROM {table_in_sql(context.configuration, TABLE_PORTFOLIO_EXAM_REPORT)}", context.engine)
    error_column = "framework_error_total" if "framework_error_total" in exam.columns else None
    if error_column is None:
        return pd.DataFrame()
    rows = {}
    for horizon, block in exam.groupby("h"):
        errors = block[error_column].astype(float)
        rows[f"h = {int(horizon)}"] = {"months": len(block), "bias (mean error)": percent_text(errors.mean(), 2),
                                       "mean absolute error": percent_text(errors.abs().mean(), 2),
                                       "error in USD (mean)": money_label(block["framework_error_usd"].astype(float).mean())
                                       if "framework_error_usd" in block else "—"}
    return pd.DataFrame(rows).T



# ─── 8.8 · which region has the most favourable composition ─────────────────────
# THE QUESTION: two regions with the same pipeline do not bring the same money if one has more of what renews at more
# value (a product, a term, a band, auto renewal), less of the customer states, or more pipeline still inside a price
# increase. The composition of a region is judged at the yields of the whole portfolio:
#   composition index of a region = Σ its pipeline of a segment × the yield of that segment in the portfolio
#                                   / its pipeline / the yield of the portfolio
#   (yield = renewed / pipeline, the rate and the uplift together; 1 = the composition of the portfolio, above 1 a
#   composition that renews more, below 1 one that renews less)
#   actual index = the yield of the region / the yield of the portfolio
#   performance  = actual / composition: how the region renews against what its composition promises
# The index of one dimension uses that dimension alone (its values at the yields of the portfolio): it says WHICH part
# of the composition helps or hurts. The pipeline is the one at due date, so the customer states are already matured.
COMPOSITION_INDEX_DIMS_ORDER = ["product", "term", "band", "renewal_type", "acquisition", "price_increase"]


def composition_segment_columns(context: ReportContext) -> dict:
    """The columns that make a segment of the composition: name in the query → SQL expression over the core."""
    configuration = context.configuration
    columns = {}
    if context.product:
        columns["product"] = f"n.{context.product}"
    if configuration.term_column and configuration.term_column in context.core_columns:
        columns["term"] = f"n.{configuration.term_column}"
    if BAND_COLUMN in context.core_columns:
        columns["band"] = f"n.{BAND_COLUMN}"
    if "tr_renewal_type" in context.core_columns:
        columns["renewal_type"] = "n.tr_renewal_type"
    for state in context.states:
        columns[state] = f"n.{state}"
    columns["acquisition"] = f"CASE WHEN {is_acquisition(context)} THEN 1 ELSE 0 END"
    return columns


def sql_8_8(context: ReportContext) -> str:
    """8.8 · the pipeline, base and renewed of every region and segment, the current year and the next one."""
    increases = price_increase_lookup(context)
    segments = composition_segment_columns(context)
    segment_select = ",\n           ".join(f"{expression} AS {name}" for name, expression in segments.items())
    levels = [context.region] + ([context.subregion] if context.subregion else [])
    keys = ", ".join(levels + ["anio"] + list(segments) + ["price_increase"])
    increase_ctes = "".join(f"{name} AS (\n{textwrap.indent(sql.strip(), '    ')}\n),\n" for name, sql in increases.ctes)
    return f"""-- 8.8 · la composición de cada región por segmento (producto, plazo, banda, tipo de renovación, estados,
--       adquisición y subida de precio), {context.current_year} y {context.next_year}
WITH {increase_ctes}base AS (
    SELECT {level_columns(context)}, n.forecast_year AS anio,
           {segment_select},
           {increases.flag} AS price_increase,
           {due_usd()} AS pipeline_usd,
           {renewal_base_usd(context)} AS base_usd,
           COALESCE(n.forecast_renewed_USD, 0) AS renovado_usd
    FROM {context.core} n
    {increases.join}
    WHERE {in_pipeline_sources()} AND n.forecast_year IN ({context.current_year}, {context.next_year})
)
SELECT {keys},
       SUM(pipeline_usd) AS pipeline_usd, SUM(base_usd) AS base_usd, SUM(renovado_usd) AS renovado_usd
FROM base
GROUP BY {keys}
ORDER BY {keys};"""


def composition_indexes(of_year: pd.DataFrame, group_column: str, segment_dims: list) -> pd.DataFrame:
    """For one year: per group (region or subregion) its composition index with every segment, with every dimension
    alone, the index of the rate and of the uplift, its actual index and its performance."""
    frame = of_year.copy()
    frame["_segment"] = frame[segment_dims].astype(str).agg("|".join, axis=1)
    portfolio = {"pipeline": frame["pipeline_usd"].sum(), "base": frame["base_usd"].sum(), "renewed": frame["renovado_usd"].sum()}
    portfolio_yield = safe_ratio(portfolio["renewed"], portfolio["pipeline"])
    portfolio_rate = safe_ratio(portfolio["base"], portfolio["pipeline"])
    portfolio_uplift = safe_ratio(portfolio["renewed"], portfolio["base"])

    def at_portfolio_values(key_columns, value: str, weight: str) -> pd.Series:
        """Σ weight of the group in a key × the portfolio's value of that key / Σ weight of the group."""
        by_key = frame.groupby(key_columns)[["pipeline_usd", "base_usd", "renovado_usd"]].sum()
        if value == "yield":
            per_key = by_key["renovado_usd"] / by_key["pipeline_usd"].replace(0, np.nan)
        elif value == "rate":
            per_key = by_key["base_usd"] / by_key["pipeline_usd"].replace(0, np.nan)
        else:
            per_key = by_key["renovado_usd"] / by_key["base_usd"].replace(0, np.nan)
        fallback = {"yield": portfolio_yield, "rate": portfolio_rate, "uplift": portfolio_uplift}[value]
        weighted = frame.join(per_key.rename("_value").fillna(fallback), on=key_columns)
        weighted["_weighted"] = weighted[weight] * weighted["_value"]
        sums = weighted.groupby(group_column)[["_weighted", weight]].sum()
        return sums["_weighted"] / sums[weight].replace(0, np.nan)

    groups = frame.groupby(group_column)[["pipeline_usd", "base_usd", "renovado_usd"]].sum()
    result = pd.DataFrame(index=groups.index)
    result["pipeline_usd"] = groups["pipeline_usd"]
    result["composition"] = at_portfolio_values("_segment", "yield", "pipeline_usd") / portfolio_yield
    result["composition_rate"] = at_portfolio_values("_segment", "rate", "pipeline_usd") / portfolio_rate
    result["composition_uplift"] = at_portfolio_values("_segment", "uplift", "base_usd") / portfolio_uplift
    for dim in segment_dims:
        result[dim] = at_portfolio_values(dim, "yield", "pipeline_usd") / portfolio_yield
    result["actual"] = (groups["renovado_usd"] / groups["pipeline_usd"].replace(0, np.nan)) / portfolio_yield
    result["performance"] = result["actual"] / result["composition"]
    result["inside_increase_share"] = (frame[frame["price_increase"] == 1].groupby(group_column)["pipeline_usd"].sum()
                                       .reindex(result.index).fillna(0) / result["pipeline_usd"])
    return result


def index_text(value: float) -> str:
    return f"{100 * (value - 1):+.1f} %" if np.isfinite(value) else "—"


def draw_8_8(table, context, number):
    now, upcoming = context.current_year, context.next_year
    segment_dims = [column for column in table.columns
                    if column not in (context.region, context.subregion, "anio", "pipeline_usd", "base_usd", "renovado_usd")]
    shown_dims = ([dim for dim in COMPOSITION_INDEX_DIMS_ORDER if dim in segment_dims]
                  + [dim for dim in segment_dims if dim not in COMPOSITION_INDEX_DIMS_ORDER])
    group_column = context.subregion or context.region
    by_year = {year: composition_indexes(table[table["anio"] == year], group_column, segment_dims) for year in (now, upcoming)}
    regions_by_year = {year: composition_indexes(table[table["anio"] == year], context.region, segment_dims)
                       for year in (now, upcoming)}
    paths = []

    # the heat map of next year: every subregion (largest first) × every dimension alone, the whole composition,
    # what it really renews and its performance; one point = 1 % more (or less) yield than the portfolio's
    upcoming_index = by_year[upcoming].sort_values("pipeline_usd", ascending=False)
    columns = shown_dims + ["composition", "actual", "performance"]
    values = (upcoming_index[columns].to_numpy(dtype=float) - 1) * 100
    figure, axis = new_figure(15, 0.32 * len(upcoming_index) + 2.6)
    limit = max(1.0, np.nanpercentile(np.abs(values), 95)) if np.isfinite(values).any() else 1.0
    colours = matplotlib.colors.LinearSegmentedColormap.from_list("composition", [ORANGE, "#fcfcfb", BLUE])
    image = axis.imshow(values, aspect="auto", cmap=colours, vmin=-limit, vmax=limit)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(column, row, f"{values[row, column]:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels([label.replace("_", " ") for label in columns], fontsize=8, rotation=20, ha="right")
    axis.set_yticks(range(len(upcoming_index)))
    axis.set_yticklabels([f"{name} ({money_label(pipeline)})" for name, pipeline in upcoming_index["pipeline_usd"].items()],
                         fontsize=7.5)
    axis.axvline(len(shown_dims) - 0.5, color=INK, linewidth=1)
    axis.grid(False)
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.01).set_label("% against the portfolio", fontsize=8)
    titles(figure, f"{number} · Which subregion has the most favourable composition in {upcoming}?",
           "Every dimension alone at the yields of the whole portfolio (yield = renewed / pipeline): blue, a composition that "
           "renews more than the portfolio's; orange, less. Composition: every dimension together; actual: what the "
           "subregion renews; performance: actual against what its composition promises. Pipeline at due date: the states "
           "already matured.")
    paths.append(save(figure, chart_path(context, number, "composicion_por_subregion"), top=0.9))

    # the regions: composition this year and next year, and what they really renew
    regions_now, regions_next = regions_by_year[now], regions_by_year[upcoming]
    names = list(regions_next.sort_values("pipeline_usd", ascending=False).index)
    figure, axes = new_figure(15, 10, 2, 1, gridspec_kw={"height_ratios": [2.4, 2]})
    positions = np.arange(len(names))
    for offset, (label, frame, column, colour) in enumerate((
            (f"composition {now}", regions_now, "composition", BLUE),
            (f"composition {upcoming}", regions_next, "composition", ORANGE),
            (f"actual {now}", regions_now, "actual", INK_MUTED),
            (f"actual {upcoming}", regions_next, "actual", INK))):
        heights = 100 * (frame.reindex(names)[column].to_numpy(dtype=float) - 1)
        axes[0].bar(positions + (offset - 1.5) * 0.2, heights, 0.19, color=colour, label=label)
    axes[0].axhline(0, color=INK_MUTED, linewidth=0.8)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([str(name) for name in names], fontsize=9)
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:+.0f} %")
    axes[0].legend(fontsize=8, frameon=False, ncol=4)
    axes[0].set_title("Yield of the composition and actual yield against the portfolio (0 = the portfolio)",
                      fontsize=10, color=INK, loc="left")
    annex = {}
    for name in names:
        line = {}
        for year, frame in ((now, regions_now), (upcoming, regions_next)):
            row = frame.loc[name] if name in frame.index else None
            if row is None:
                continue
            line[f"composition {year}"] = index_text(row["composition"])
            line[f"rate {year}"] = index_text(row["composition_rate"])
            line[f"uplift {year}"] = index_text(row["composition_uplift"])
            line[f"actual {year}"] = index_text(row["actual"])
            line[f"inside an increase {year}"] = percent_text(row["inside_increase_share"], 0)
        best = regions_next.loc[name, shown_dims].astype(float) if name in regions_next.index else pd.Series(dtype=float)
        if len(best.dropna()):
            line[f"helps most {upcoming}"] = f"{best.idxmax()} {index_text(best.max())}"
            line[f"hurts most {upcoming}"] = f"{best.idxmin()} {index_text(best.min())}"
        annex[str(name)] = line
    annex = pd.DataFrame(annex).T
    annex_table(axes[1], annex, "Annex · composition (all, its rate and its uplift), actual, share of the pipeline inside "
                                "a price increase, and the dimension that helps and hurts most", font_size=7,
                first_column_width=0.08)
    write_annex(context, number, "por_region", annex)
    titles(figure, f"{number}a · The composition of every region, {now} and {upcoming}",
           "A region whose composition worsens from one year to the next brings less money with the same pipeline; "
           "one inside a price increase this year and not the next loses its step.")
    paths.append(save(figure, chart_path(context, number, "composicion_por_region"), top=0.92))
    pd.concat({str(year): frame for year, frame in by_year.items()}, names=["anio", group_column]).to_csv(
        os.path.join(context.output_folder, f"{number.replace('.', '_')}_anexo_por_subregion.csv"))
    return paths


# ─── 8.9 · where a price increase was detected, and since when ──────────────────
# THE QUESTION: did the framework see the price increases there were? Step 8.4 writes the detector month by month
# (sff_price_increase_monitor: per price group, the uplift of the isolated renewals, the average of its previous 12
# months and the step over it) and one row per increase it confirmed (sff_price_increase_calendar: the first month
# of the step, its size, the uplift before and in the cycle). An increase is a month price_step_threshold or more above
# the average of its previous 12, price_persistence_months months in a row, in a group with
# price_group_min_renewed_units renewals a month. The chart puts every group with an increase next to its detector,
# so the dates can be compared with the tariff changes that were made.
TABLE_PRICE_INCREASE_MONITOR = "price_increase_monitor"
PRICE_INCREASE_GROUPS_DRAWN = 12             # the groups with the most money renewed inside their increases


def sql_8_9(context: ReportContext) -> Optional[str]:
    """8.9 · the detector of the price increases, month by month, for every price group (step 8.4)."""
    if not context.has_table(TABLE_PRICE_INCREASE_MONITOR):
        return None
    monitor = table_in_sql(context.configuration, TABLE_PRICE_INCREASE_MONITOR)
    return f"""-- 8.9 · el detector de subidas de precio mes a mes por grupo de precio (paso 8.4)
SELECT *
FROM {monitor}
ORDER BY price_group, period;"""


def draw_8_9(table, context, number):
    configuration = context.configuration
    calendar = (pd.read_sql(f"SELECT * FROM {table_in_sql(configuration, TABLE_PRICE_INCREASE_CALENDAR)}", context.engine)
                if context.has_table(TABLE_PRICE_INCREASE_CALENDAR) else pd.DataFrame())
    paths = []
    cycle = configuration.price_cycle_months
    if len(calendar):
        calendar["inicio"] = pd.PeriodIndex(calendar["inicio"].astype(str), freq="M")
        calendar["fin"] = calendar["inicio"] + (cycle - 1)
    # the groups drawn: the ones with an increase, the most money renewed inside it first
    if len(calendar):
        ranking = calendar.groupby("price_group")["usd_renovado_ciclo"].sum().sort_values(ascending=False)
        groups = list(ranking.index[:PRICE_INCREASE_GROUPS_DRAWN])
    else:
        groups = []
    monitor = table.copy()
    monitor["_month"] = pd.PeriodIndex(monitor["period"].astype(str), freq="M")
    if groups:
        figure, axes = small_multiples(groups, height_per_row=3.0, same_scale=False)
        for axis, group in zip(axes, groups):
            of_group = monitor[monitor["price_group"] == group].sort_values("_month")
            months = pd.PeriodIndex(of_group["_month"])
            positions = np.arange(len(of_group))
            detector = of_group["uplift_isolated"] if of_group["uplift_isolated"].notna().any() else of_group["uplift"]
            axis.plot(positions, of_group["uplift"], color=INK_MUTED, linewidth=1, label="uplift (every renewal)")
            axis.plot(positions, detector, color=INK, linewidth=1.6, label="uplift the detector reads (isolated renewals)")
            axis.plot(positions, of_group["uplift_previous_12"], color=BLUE, linewidth=1.2, linestyle="--",
                      label="average of the previous 12 months")
            increases_of_group = calendar[calendar["price_group"] == group].sort_values("inicio")
            for order, (_, increase) in enumerate(increases_of_group.iterrows()):
                inside = [position for position, month in enumerate(months) if increase["inicio"] <= month <= increase["fin"]]
                if inside:
                    axis.axvspan(inside[0] - 0.5, inside[-1] + 0.5, color=ORANGE, alpha=0.15, linewidth=0, zorder=0)
                    # one label under the other, so two increases close in time can be read
                    axis.annotate(f"{increase['inicio']} {increase['escalon']:+.1%}", (inside[0], 1.0 - 0.08 * order),
                                  xycoords=("data", "axes fraction"), fontsize=7, color=ORANGE, va="top")
            month_axis(axis, months, compact=True)
            axis.set_title(str(group).replace("|", " · "), fontsize=8, color=INK, loc="left")
        figure.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="upper right", ncol=3,
                      bbox_to_anchor=(0.99, 0.9))
        titles(figure, f"{number} · Where a price increase was detected, and since when",
               f"Orange: the {cycle} months of every increase (its first month and its step written above). An increase is a "
               f"month {configuration.price_step_threshold:.0%} or more above the average of its previous 12 months, "
               f"{configuration.price_persistence_months} months in a row, in a group with "
               f"{configuration.price_group_min_renewed_units:,.0f} renewals a month or more. Price group: "
               f"{' × '.join(configuration.price_group_dims) or 'the portfolio'}.")
        paths.append(save(figure, chart_path(context, number, "subidas_detectadas"), top=0.88))

    # the annex: every increase of the calendar, the most recent first
    if len(calendar):
        annex = calendar.sort_values(["inicio", "escalon"], ascending=[False, False])
        annex = pd.DataFrame({"price group": annex["price_group"].astype(str).str.replace("|", " · ", regex=False),
                              "from": annex["inicio"].astype(str), "to": annex["fin"].astype(str),
                              "step": annex["escalon"].map(lambda value: f"{value:+.1%}"),
                              "uplift before": annex["uplift_antes"].map(ratio_text),
                              "uplift in the cycle": annex["uplift_ciclo"].map(ratio_text),
                              "months seen": annex["meses_vistos"].astype(int),
                              "USD renewed in the cycle": annex["usd_renovado_ciclo"].map(money_label)}).set_index("price group")
    else:
        annex = pd.DataFrame({"note": ["no increase in the calendar of this run"]}, index=["—"])
    figure, axis = new_figure(15, 0.27 * len(annex) + 1.8)
    annex_table(axis, annex, font_size=7.5, first_column_width=0.3)
    titles(figure, f"{number}a · Annex · every price increase detected (sff_price_increase_calendar)",
           "Compare the dates with the tariff changes that were made: an increase made and not here is one the detector "
           "did not see (too small, or too few renewals in its group).")
    paths.append(save(figure, chart_path(context, number, "calendario_de_subidas"), top=0.94))
    write_annex(context, number, "calendario_de_subidas", annex)
    return paths


# ─── 8.10 · the term a licence renews to ────────────────────────────────────────
# THE QUESTION: of what renews, how much renews to 1, 2 or 3 years, and at what uplift? A 1-year licence that renews
# to 2 years brings more money this year and a gap next year; a 2-year licence that renews to 1 year brings pipeline
# next year at a lower price. Read from the six columns of the term transitions in the closed months of the core (the
# step 7.2 projects with their share of the last term_transition_window_months months: sff_transiciones_plazo).
def sql_8_10(context: ReportContext) -> Optional[str]:
    """8.10 · the renewals of every closed month by origin term and destination term: units, USD and the base."""
    destinations = [short_name for short_name, _ in TERM_DESTINATIONS]
    needed = [f"{CORE_RENEWED_UNITS_TO_PREFIX}{name}" for name in destinations] + [f"{CORE_RENEWED_USD_TO_PREFIX}{name}" for name in destinations]
    term_column = context.configuration.term_column
    if not term_column or not all(column in context.core_columns for column in needed):
        return None
    period = context.configuration.period_col
    columns = []
    for name in destinations:
        units, usd = f"n.{CORE_RENEWED_UNITS_TO_PREFIX}{name}", f"n.{CORE_RENEWED_USD_TO_PREFIX}{name}"
        columns += [f"COALESCE({units}, 0) AS unidades_a_{name}", f"COALESCE({usd}, 0) AS usd_a_{name}",
                    f"COALESCE({units}, 0) * COALESCE(n.forecast_to_renew_USD, 0) / NULLIF(n.forecast_to_renew_units, 0) "
                    f"AS base_a_{name}"]
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio, n.{period} AS periodo,
       n.{term_column} AS plazo_origen,
       {(',' + chr(10) + '       ').join(columns)}
FROM {context.core} n
WHERE n.forecast_status = 'actual' AND n.forecast_universe = 'pipeline'"""
    measures = []
    for name in destinations:
        measures += [f"SUM(unidades_a_{name}) AS unidades_a_{name}", f"SUM(usd_a_{name}) AS usd_a_{name}",
                     f"SUM(base_a_{name}) AS base_a_{name}",
                     f"SUM(usd_a_{name}) / NULLIF(SUM(base_a_{name}), 0) AS uplift_a_{name}"]
    return with_subtotals(context, base, measures, ["anio", "periodo", "plazo_origen"],
                          comment="8.10 · a qué plazo renuevan las licencias: por plazo de origen y de destino, mes a mes")


def draw_8_10(table, context, number):
    destinations = [short_name for short_name, _ in TERM_DESTINATIONS]
    total = total_rows(table).copy()
    total["plazo_origen"] = total["plazo_origen"].astype(str)
    origins = sorted(total["plazo_origen"].unique())
    if not origins:
        return []
    destination_colours = {"1y": BLUE, "2y": AQUA, "3y": VIOLET}
    figure, axes = new_figure(15, 3.2 * len(origins) + 3, len(origins) + 1, 1,
                              gridspec_kw={"height_ratios": [2.4] * len(origins) + [2.2]})
    annex = {}
    for axis, origin in zip(axes, origins):
        of_origin = as_months(total[total["plazo_origen"] == origin])
        months = pd.PeriodIndex(of_origin["_month"])
        units = of_origin[[f"unidades_a_{name}" for name in destinations]].sum(axis=1).replace(0, np.nan)
        for name in destinations:
            axis.plot(np.arange(len(of_origin)), of_origin[f"unidades_a_{name}"] / units, color=destination_colours[name],
                      linewidth=1.8, label=f"renews to {name.replace('y', ' year')}{'s' if name != '1y' else ''}")
        month_axis(axis, months)
        percent_axis(axis)
        axis.set_ylim(0, 1)
        axis.set_title(f"From {origin}: share of the renewed units by the term they renew to", fontsize=10, color=INK, loc="left")
        by_year = of_origin.groupby("anio")[[f"{kind}_a_{name}" for name in destinations for kind in ("unidades", "usd", "base")]].sum()
        for year, row in by_year.iterrows():
            renewed_units = sum(row[f"unidades_a_{name}"] for name in destinations)
            line = {}
            for name in destinations:
                share = safe_ratio(row[f"unidades_a_{name}"], renewed_units)
                line[f"to {name}: share · uplift"] = f"{percent_text(share, 0)} · {ratio_text(safe_ratio(row[f'usd_a_{name}'], row[f'base_a_{name}']))}"
                line[f"to {name}: USD"] = money_label(row[f"usd_a_{name}"])
            annex[f"from {origin} · {int(year)}"] = line
    axes[0].legend(fontsize=8, frameon=False, loc="center left")
    annex = pd.DataFrame(annex).T
    annex_table(axes[-1], annex, "Annex · per origin term and year: the share of the renewed units to every term, its uplift "
                                 "(USD renewed / USD due of those units) and its USD", font_size=7, first_column_width=0.14)
    write_annex(context, number, "transiciones", annex)
    titles(figure, f"{number} · Which term do the licences renew to?",
           "A 1-year licence that renews to 2 or 3 years brings more money now and leaves a gap next year; one of 2 or 3 "
           "years that renews to 1 year comes back next year at its own uplift. Step 7.2 projects with the share of the "
           f"last {context.configuration.term_transition_window_months} closed months (sff_transiciones_plazo).")
    return [save(figure, chart_path(context, number, "transiciones_de_plazo"), top=0.93)]


# ═══════════════════════════════════════════════════════════════════════════════════
# CHAPTER 9 · MONOGRAPHS: ONE BUSINESS QUESTION, ITS ANSWER FIRST, THEN ITS DETAIL (added 28-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════════════
# A monograph answers one question of the business. First the answer, in a few lines of text (printed and written as
# <number>_respuesta.md), then the charts that explain it, from the general to the detail, every one with its annex.
#
# 9.1 · HOW MUCH DOES NEXT YEAR CHANGE BECAUSE OF THE ACQUISITION OF THIS YEAR?
# An acquisition is sold with a big discount. When it renews for the first time (12 months later, for a 1-year
# licence) it renews at a lower rate than a retention, but at a much higher uplift: the one that renews pays the full
# price. So the pipeline that comes from the acquisitions of a year weighs on the renewed money of the next one, and
# not in proportion to its USD: its volume, the price it was sold at (the discount), its rate and its uplift all count.
# If this year acquires less (or more in 2- and 3-year licences, which do not fall due next year), next year has less
# of that pipeline, and the renewal rate does not show it.
#
# THE TWO PIPELINES OF EVERY YEAR (the rows of the pipeline: real, expected, projected and simulated):
#   acquisition   the first renewal of what was acquired: acquisition_column in acquisition_values, and the
#                 acquisitions simulated by step 5.1 (they fall due 12 months after the simulation window)
#   retention     the rest: licences that had already renewed, and the renewals projected by step 7.2
# The 1-year licences are the answer; the multi-year ones are in the annex (an acquisition of 2 years falls due 24
# months after its sale). The retail to subscription re-entry is not in either: it is its own universe (6.1).
#
# THE MONEY, as in chapter 8: renewed = units due × USD due per unit × rate (USD) × uplift
#   USD due per unit   the price it was sold at (for an acquisition, with its discount)
#   rate (USD)         base / pipeline: the USD due of what renews over the USD due
#   uplift             renewed / base: what the one that renews pays over what it was paying
#
# THE ANSWERS:
#   the premium over retention (every year): the renewed money of the acquisition − its pipeline × the yield of the
#   retention (yield = renewed / pipeline). It splits in two:
#       rate gap         pipeline × (rate acq − rate ret) × uplift ret      (negative when it renews less)
#       uplift premium   pipeline × rate acq × (uplift acq − uplift ret)     (positive when it pays more)
#   the bridge of the renewed money of the acquisition, current year → next year, one factor at a time, the
#   conditions first and the volume last:
#       price at sale    U now × (p next − p now) × r now × u now
#       rate             U now × p next × (r next − r now) × u now
#       uplift           U now × p next × r next × (u next − u now)
#       volume           (U next − U now) × p next × r next × u next
#   The four add up to renewed next − renewed now, and the volume effect is the counterfactual: had the year that
#   feeds next year acquired the units of the one before, at its own conditions, next year would renew −volume more.
#   The sales of acquisition by term, January to the last closed month of every year: an acquisition of t years sold
#   in month m is the acquisition row of term t due in m + 12 t (as step 5.1 reads them). A term whose due months the
#   extract does not reach yet is not visible.
MONOGRAPH_YEARS_BACK = 3                    # 9.1: the years before the current one that are drawn
ORIGIN_ACQUISITION = "acquisition"
ORIGIN_RETENTION = "retention"
TERM_ONE_YEAR = "1y"
TERM_MULTI_YEAR = "multi_year"
FACTOR_SUMS = ["pipeline_usd", "base_usd", "renovado_usd", "vencen_unidades", "renovadas_unidades", "simulado_usd",
               "simulado_base_usd", "simulado_renovado_usd", "descuento_por_usd", "descuento_peso_usd",
               "simulado_descuento_por_usd", "simulado_descuento_peso_usd"]
SIMULATED_UPLIFT_WARNING = 0.05            # 9.1: the simulated acquisition renews at an uplift 5 % or more away from the
                                           # acquisition of the extract: a warning (its discount is not the real one?)
BRIDGE_EFFECTS = {"price_at_sale": ("price at sale (USD per unit due)", VIOLET), "rate": ("rate", ORANGE),
                  "uplift": ("uplift", AQUA), "volume": ("volume (units due)", BLUE)}
BRIDGE_SHORT_NAMES = {"price_at_sale": "price at sale", "rate": "rate effect", "uplift": "uplift effect",
                      "volume": "volume"}                 # the headers of the annex by region


def acquisition_or_retention(context: ReportContext, alias: str = "n") -> str:
    """The origin of a pipeline row: the first renewal of an acquisition, or a retention."""
    return f"""CASE
           WHEN {alias}.forecast_pipeline_source = 'pipeline_simulada' THEN '{ORIGIN_ACQUISITION}'
           WHEN {alias}.forecast_pipeline_source = 'pipeline_proyectada' THEN '{ORIGIN_RETENTION}'
           WHEN {is_acquisition(context, alias)} THEN '{ORIGIN_ACQUISITION}'
           ELSE '{ORIGIN_RETENTION}'
       END"""


def sql_9_1(context: ReportContext) -> Optional[str]:
    """9.1 · the pipeline, base and renewed money of every month by origin (acquisition or retention) and term, from
    MONOGRAPH_YEARS_BACK years before the current one to next December."""
    if is_acquisition(context) == "1 = 0":          # no acquisition column in this run: nothing to tell apart
        return None
    period = context.configuration.period_col
    first_year = context.current_year - MONOGRAPH_YEARS_BACK
    discount_column = context.configuration.discount_value_column
    # the discount it was sold with, weighted by the USD due of the rows that carry it (a row of the maturation of a
    # state carries part of the USD due, but no discount: it is left out of the weight). 0 without a discount column
    with_discount = discount_column and discount_column in context.core_columns
    discount_times_usd = (f"COALESCE(n.{discount_column} * n.forecast_to_renew_USD, 0)" if with_discount else "0")
    discount_weight = (f"CASE WHEN n.{discount_column} IS NULL THEN 0 ELSE COALESCE(n.forecast_to_renew_USD, 0) END"
                       if with_discount else "0")
    base = f"""SELECT {level_columns(context)}, n.forecast_year AS anio, n.{period} AS periodo,
       {acquisition_or_retention(context)} AS origen,
       CASE WHEN {is_one_year_term(context)} THEN '{TERM_ONE_YEAR}' ELSE '{TERM_MULTI_YEAR}' END AS plazo,
       {money_columns(context)},
       CASE WHEN n.forecast_pipeline_source = 'pipeline_simulada' THEN {due_usd()} ELSE 0 END AS simulado_usd,
       CASE WHEN n.forecast_pipeline_source = 'pipeline_simulada' THEN {renewal_base_usd(context)} ELSE 0 END AS simulado_base_usd,
       CASE WHEN n.forecast_pipeline_source = 'pipeline_simulada' THEN COALESCE(n.forecast_renewed_USD, 0) ELSE 0 END AS simulado_renovado_usd,
       {discount_times_usd} AS descuento_por_usd, {discount_weight} AS descuento_peso_usd,
       CASE WHEN n.forecast_pipeline_source = 'pipeline_simulada' THEN {discount_times_usd} ELSE 0 END AS simulado_descuento_por_usd,
       CASE WHEN n.forecast_pipeline_source = 'pipeline_simulada' THEN {discount_weight} ELSE 0 END AS simulado_descuento_peso_usd
FROM {context.core} n
WHERE {in_pipeline_sources()}
  AND n.forecast_year BETWEEN {first_year} AND {context.next_year}"""
    measures = MONEY_MEASURES + ["SUM(simulado_usd) AS simulado_usd", "SUM(simulado_base_usd) AS simulado_base_usd",
                                 "SUM(simulado_renovado_usd) AS simulado_renovado_usd",
                                 "SUM(descuento_por_usd) AS descuento_por_usd", "SUM(descuento_peso_usd) AS descuento_peso_usd",
                                 "SUM(simulado_descuento_por_usd) AS simulado_descuento_por_usd",
                                 "SUM(simulado_descuento_peso_usd) AS simulado_descuento_peso_usd"]
    return with_subtotals(context, base, measures, ["anio", "periodo", "origen", "plazo"],
                          comment="9.1 · la primera renovación de la adquisición frente a la retención: pipeline, base y "
                                  f"renovado mes a mes, de {first_year} a {context.next_year}")


def sql_9_1_sales(context: ReportContext) -> str:
    """9.1 · the rows of acquisition of the extract by due month and term: every one is a sale of its due month − 12 ×
    its term in years (the sale month is computed in Python from the term written as text)."""
    configuration = context.configuration
    period = configuration.period_col
    levels = level_columns(context)
    term = configuration.term_column if configuration.term_column in context.core_columns else None
    term_select = f"n.{term}" if term else f"{sql_literal(configuration.one_year_term_value)}"
    term_group = f", n.{term}" if term else ""
    return f"""-- 9.1 · las ventas de adquisición: cada fila de adquisición del extracto es una venta de su mes − 12 × su plazo
SELECT {levels}, n.{period} AS periodo, {term_select} AS plazo,
       SUM(COALESCE(n.s1_1_vencen_unidades, 0)) AS vendidas_unidades,
       SUM(COALESCE(n.s1_1_vencen_usd, 0)) AS vendido_usd
FROM {context.core} n
WHERE n.origen_fila = 'raw' AND {is_acquisition(context)}
GROUP BY {levels}, n.{period}{term_group}
ORDER BY {levels}, n.{period}{term_group};"""


def money_factors(sums) -> dict:
    """The factors of renewed = units × USD per unit × rate × uplift, from the sums of a block of rows."""
    units, pipeline = float(sums["vencen_unidades"]), float(sums["pipeline_usd"])
    base, renewed = float(sums["base_usd"]), float(sums["renovado_usd"])
    return {"units": units, "pipeline": pipeline, "base": base, "renewed": renewed,
            "renewed_units": float(sums["renovadas_unidades"]), "simulated": float(sums.get("simulado_usd", 0.0)),
            "simulated_uplift": safe_ratio(float(sums.get("simulado_renovado_usd", 0.0)), float(sums.get("simulado_base_usd", 0.0))),
            "extract_uplift": safe_ratio(renewed - float(sums.get("simulado_renovado_usd", 0.0)),
                                         base - float(sums.get("simulado_base_usd", 0.0))),
            "discount": safe_ratio(float(sums.get("descuento_por_usd", 0.0)), float(sums.get("descuento_peso_usd", 0.0))),
            "simulated_discount": safe_ratio(float(sums.get("simulado_descuento_por_usd", 0.0)),
                                             float(sums.get("simulado_descuento_peso_usd", 0.0))),
            "extract_discount": safe_ratio(float(sums.get("descuento_por_usd", 0.0)) - float(sums.get("simulado_descuento_por_usd", 0.0)),
                                           float(sums.get("descuento_peso_usd", 0.0)) - float(sums.get("simulado_descuento_peso_usd", 0.0))),
            "price": safe_ratio(pipeline, units), "rate": safe_ratio(base, pipeline), "uplift": safe_ratio(renewed, base),
            "rate_units": safe_ratio(float(sums["renovadas_unidades"]), units),
            "yield": safe_ratio(renewed, pipeline),
            "renewed_price": safe_ratio(renewed, float(sums["renovadas_unidades"]))}


def premium_over_retention(acquisition: dict, retention: dict) -> dict:
    """What the acquisition renews over what the same pipeline would renew at the yield of the retention, split into
    the rate gap and the uplift premium (they add up to the premium)."""
    pipeline = acquisition["pipeline"]
    rate_gap = pipeline * (acquisition["rate"] - retention["rate"]) * retention["uplift"]
    uplift_premium = pipeline * acquisition["rate"] * (acquisition["uplift"] - retention["uplift"])
    at_retention_yield = pipeline * retention["rate"] * retention["uplift"]
    if not all(np.isfinite(value) for value in (rate_gap, uplift_premium, at_retention_yield)):
        return {"at_retention_yield": np.nan, "rate_gap": np.nan, "uplift_premium": np.nan, "premium": np.nan}
    return {"at_retention_yield": at_retention_yield, "rate_gap": rate_gap, "uplift_premium": uplift_premium,
            "premium": acquisition["renewed"] - at_retention_yield}


def acquisition_bridge(now: dict, upcoming: dict) -> dict:
    """The change of the renewed money from one year to the next, one factor at a time: price at sale, rate and uplift
    at the volume of now, then the volume at the conditions of next (so the volume effect is the counterfactual).
    Without the four factors in both years, the whole change goes to the volume."""
    factors = ("units", "price", "rate", "uplift")
    change = upcoming["renewed"] - now["renewed"]
    if not all(np.isfinite(year[factor]) for year in (now, upcoming) for factor in factors):
        return {"price_at_sale": 0.0, "rate": 0.0, "uplift": 0.0, "volume": change}
    units_now, price_now, rate_now, uplift_now = (now[factor] for factor in factors)
    units_next, price_next, rate_next, uplift_next = (upcoming[factor] for factor in factors)
    return {"price_at_sale": units_now * (price_next - price_now) * rate_now * uplift_now,
            "rate": units_now * price_next * (rate_next - rate_now) * uplift_now,
            "uplift": units_now * price_next * rate_next * (uplift_next - uplift_now),
            "volume": (units_next - units_now) * price_next * rate_next * uplift_next}


def factors_by_year(rows: pd.DataFrame, origin: str, term: str = TERM_ONE_YEAR) -> dict:
    """{year: money_factors} of the rows of one origin and term."""
    chosen = rows[(rows["origen"] == origin) & (rows["plazo"] == term)]
    sums = chosen.groupby("anio")[FACTOR_SUMS].sum()
    return {int(year): money_factors(row) for year, row in sums.iterrows()}


def acquisition_sales_by_term(context: ReportContext) -> tuple:
    """The sales of acquisition of every year by term, January to the last closed month (the same months every year,
    so the years compare), in units and USD; and whether every term of every year is visible in the extract.
    Returns (the table: year × term, the SQL that reads it)."""
    sql = sql_9_1_sales(context)
    rows = pd.read_sql(sql, context.engine)
    if rows.empty:
        return pd.DataFrame(), sql
    rows["_due"] = pd.PeriodIndex(rows["periodo"].astype(str), freq="M")
    years_of_term = term_years(rows["plazo"]).fillna(1).astype(int)
    rows["anos_plazo"] = years_of_term
    rows["_sale"] = [due - 12 * years for due, years in zip(rows["_due"], years_of_term)]
    last_due = rows["_due"].max()
    current = pd.Period(context.current_month, freq="M")
    months_to_date = current.month - 1 if current.month > 1 else 12      # January: the whole previous year
    last_year = context.current_year if current.month > 1 else context.current_year - 1
    years = list(range(context.current_year - MONOGRAPH_YEARS_BACK, last_year + 1))
    lines = []
    for year in years:
        for term_years_value in sorted(rows["anos_plazo"].unique()):
            of_year = rows[(rows["anos_plazo"] == term_years_value)
                           & rows["_sale"].map(lambda sale: sale.year == year and sale.month <= months_to_date)]
            last_sale = pd.Period(f"{year}-{months_to_date:02d}", freq="M")
            visible = last_sale + 12 * term_years_value <= last_due
            lines.append({"anio_venta": year, "plazo_anos": int(term_years_value), "visible": visible,
                          "vendidas_unidades": float(of_year["vendidas_unidades"].sum()) if visible else np.nan,
                          "vendido_usd": float(of_year["vendido_usd"].sum()) if visible else np.nan})
    table = pd.DataFrame(lines)
    table.attrs["months_to_date"] = months_to_date
    return table, sql


def bridge_waterfall(axis, start_label: str, start: float, effects: dict, end_label: str, end: float) -> None:
    """From the renewed money of one year to the next: the start, every effect on top of the previous ones, the end."""
    labels = [start_label]
    axis.bar(0, start, color=INK_MUTED, width=0.6)
    axis.annotate(money_label(start), (0, start), ha="center", va="bottom", fontsize=8.5, color=INK)
    running = start
    levels = [start, end]                       # every height the bars reach: the axis keeps room for their labels
    for position, (effect, value) in enumerate(effects.items(), start=1):
        levels.append(running + value)
        label, colour = BRIDGE_EFFECTS[effect]
        axis.bar(position, value, bottom=running, color=colour, width=0.6)
        axis.annotate(money_label(value), (position, max(running, running + value)), ha="center", va="bottom",
                      fontsize=8.5, color=INK_SECONDARY)
        running += value
        labels.append(label)
    axis.bar(len(effects) + 1, end, color=INK, width=0.6)
    axis.annotate(money_label(end), (len(effects) + 1, end), ha="center", va="bottom", fontsize=8.5, color=INK)
    labels.append(end_label)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, fontsize=8)
    money_axis(axis)
    highest, lowest = max(levels), min(levels)
    if lowest > 0:                              # all above zero: start the axis near the lowest bar, not at zero
        bottom = lowest - 0.6 * (highest - lowest) if highest > lowest else 0.85 * lowest
        bottom = max(0.0, bottom)
        axis.set_ylim(bottom=bottom, top=highest + 0.1 * (highest - bottom))


def year_label(year: int, context: ReportContext) -> str:
    if year == context.next_year:
        return f"{year}\n(forecast)"
    if year == context.current_year:
        return f"{year}\n(partly forecast)"
    return str(year)


def unit_price_text(value: float) -> str:
    return f"${value:,.2f}" if np.isfinite(value) else "—"


def growth_text(new: float, old: float) -> str:
    return f"{100 * (new / old - 1):+.1f} %" if np.isfinite(old) and old and np.isfinite(new) else "—"


def acquisition_answer(context: ReportContext, acquisition: dict, retention: dict, bridge: dict,
                       sales: pd.DataFrame) -> list:
    """The answer of 9.1 in a few lines, in Spanish, for the business: what changes, why, against the retention, and
    the sales of this year by term."""
    now, upcoming = context.current_year, context.next_year
    acquisition_now, acquisition_next = acquisition.get(now), acquisition.get(upcoming)
    if acquisition_now is None or acquisition_next is None:
        return [f"Sin pipeline de adquisición a 1 año en {now} o {upcoming}: no hay respuesta."]
    change = acquisition_next["renewed"] - acquisition_now["renewed"]
    renewed_one_year = {year: acquisition[year]["renewed"] + retention[year]["renewed"]
                        for year in (now, upcoming) if year in retention}
    lines = [f"PREGUNTA: ¿Cuánto cambia {upcoming} por la adquisición de {now}?", "",
             f"RESPUESTA: la primera renovación de la adquisición (licencias de 1 año) renueva "
             f"{money_label(acquisition_now['renewed'])} en {now} y {money_label(acquisition_next['renewed'])} en {upcoming}: "
             f"{money_label(change)} ({growth_text(acquisition_next['renewed'], acquisition_now['renewed'])})."]
    if all(year in renewed_one_year for year in (now, upcoming)):
        lines.append(f"Es el {percent_text(safe_ratio(acquisition_now['renewed'], renewed_one_year[now]))} del renovado de 1 año "
                     f"en {now} y el {percent_text(safe_ratio(acquisition_next['renewed'], renewed_one_year[upcoming]))} en {upcoming}.")
    pipeline_change = acquisition_next["pipeline"] - acquisition_now["pipeline"]
    lines += ["", f"EN DÓLARES: la pipeline de adquisición a 1 año (lo vendido un año antes, al precio de venta) pasa de "
                  f"{money_label(acquisition_now['pipeline'])} a {money_label(acquisition_next['pipeline'])} "
                  f"({money_label(pipeline_change)}); de la de {upcoming}, {money_label(acquisition_next['simulated'])} es la "
                  f"adquisición simulada de lo que queda de {now} (paso 5.1). Cada $1M de esa pipeline renueva "
                  f"{money_label(1e6 * acquisition_next['yield'])} en {upcoming}, así que la diferencia de pipeline vale "
                  f"{money_label(pipeline_change * acquisition_next['yield'])} en {upcoming}; el resto "
                  f"({money_label(change - pipeline_change * acquisition_next['yield'])}) es que cada dólar renueva distinto "
                  f"(tasa × revalorización: {percent_text(acquisition_now['yield'])} → {percent_text(acquisition_next['yield'])})."]
    simulated_uplift, extract_uplift = acquisition_next["simulated_uplift"], acquisition_next["extract_uplift"]
    if (np.isfinite(simulated_uplift) and np.isfinite(extract_uplift) and extract_uplift > 0
            and abs(simulated_uplift / extract_uplift - 1) >= SIMULATED_UPLIFT_WARNING):
        configuration = context.configuration
        lines.append(f"AVISO: en {upcoming} la adquisición simulada renueva con uplift {ratio_text(simulated_uplift)} y la del "
                     f"extracto con {ratio_text(extract_uplift)}. La simulada se vende con el descuento "
                     f"acquisition_discount = {configuration.acquisition_discount:.0%} ({configuration.discount_value_column}: "
                     f"{percent_text(acquisition_next['simulated_discount'])} ponderado por USD); la del extracto tiene "
                     f"{percent_text(acquisition_next['extract_discount'])}. Si el descuento no es el real, la revalorización "
                     f"de la simulada tampoco: revisa acquisition_discount con el descuento real de la adquisición.")
    lines += ["", f"POR QUÉ (de {now} a {upcoming}, un factor cada vez, el volumen al final):",
              f"  - volumen: {money_label(bridge['volume'])}. Las unidades que vencen pasan de "
              f"{acquisition_now['units']:,.0f} a {acquisition_next['units']:,.0f} "
              f"({growth_text(acquisition_next['units'], acquisition_now['units'])}). Si se hubieran adquirido las "
              f"unidades del año anterior, {upcoming} renovaría {money_label(-bridge['volume'])} más.",
              f"  - precio de venta: {money_label(bridge['price_at_sale'])}. El USD que vence por unidad pasa de "
              f"{unit_price_text(acquisition_now['price'])} a {unit_price_text(acquisition_next['price'])} "
              f"(más descuento en la venta = menos USD por unidad, y más margen para la revalorización).",
              f"  - tasa: {money_label(bridge['rate'])}. Tasa en USD {percent_text(acquisition_now['rate'])} → "
              f"{percent_text(acquisition_next['rate'])} (en unidades {percent_text(acquisition_now['rate_units'])} → "
              f"{percent_text(acquisition_next['rate_units'])}).",
              f"  - revalorización: {money_label(bridge['uplift'])}. Uplift {ratio_text(acquisition_now['uplift'])} → "
              f"{ratio_text(acquisition_next['uplift'])}; el que renueva paga "
              f"{unit_price_text(acquisition_now['renewed_price'])} → {unit_price_text(acquisition_next['renewed_price'])} por unidad."]
    lines += ["", "FRENTE A LA RETENCIÓN (lo que la adquisición renueva sobre lo que renovaría a la rentabilidad de la retención):"]
    for year in (now, upcoming):
        if year not in acquisition or year not in retention:
            continue
        premium = premium_over_retention(acquisition[year], retention[year])
        lines.append(f"  - {year}: tasa en unidades {percent_text(acquisition[year]['rate_units'])} frente a "
                     f"{percent_text(retention[year]['rate_units'])}; uplift {ratio_text(acquisition[year]['uplift'])} frente a "
                     f"{ratio_text(retention[year]['uplift'])}. Prima de revalorización {money_label(premium['uplift_premium'])}, "
                     f"hueco de tasa {money_label(premium['rate_gap'])}: neto {money_label(premium['premium'])}.")
    yield_next = acquisition_next["yield"]
    if np.isfinite(yield_next):
        lines.append(f"  - Cada $1M de pipeline de adquisición a 1 año renueva {money_label(1e6 * yield_next)} en {upcoming}; "
                     f"de retención, {money_label(1e6 * retention.get(upcoming, {}).get('yield', np.nan))}.")
    if len(sales):
        months = sales.attrs.get("months_to_date", 12)
        last_year = int(sales["anio_venta"].max())
        lines += ["", f"VENTAS DE ADQUISICIÓN (enero a {MONTH_NAMES[months - 1]} de cada año, USD vendido):"]
        for term_value in sorted(sales["plazo_anos"].unique()):
            of_term = sales[sales["plazo_anos"] == term_value].set_index("anio_venta")
            this_year, previous_year = of_term.loc[last_year], of_term.loc[last_year - 1] if last_year - 1 in of_term.index else None
            if not this_year["visible"]:
                lines.append(f"  - {term_value} año(s): las de {last_year} aún no se ven en el extracto (vencen después de su "
                             f"último mes); el paso 5.1 las estima con la proporción por plazo.")
                continue
            previous_text = (f" frente a {money_label(previous_year['vendido_usd'])} en {last_year - 1} "
                             f"({growth_text(this_year['vendido_usd'], previous_year['vendido_usd'])})"
                             if previous_year is not None and previous_year["visible"] else "")
            lines.append(f"  - {term_value} año(s): {money_label(this_year['vendido_usd'])} en {last_year}{previous_text}.")
        lines.append("  Lo vendido a 2 y 3 años este año no vence el año que viene: es pipeline que falta en él.")
    return lines


def draw_9_1(table, context, number):
    now, upcoming = context.current_year, context.next_year
    total = total_rows(table)
    acquisition = factors_by_year(total, ORIGIN_ACQUISITION)
    retention = factors_by_year(total, ORIGIN_RETENTION)
    years = sorted(set(acquisition) & set(retention))
    if now not in acquisition or upcoming not in acquisition:
        return []
    bridge = acquisition_bridge(acquisition[now], acquisition[upcoming])
    sales, sales_sql = acquisition_sales_by_term(context)
    with open(os.path.join(context.output_folder, f"{number.replace('.', '_')}_ventas_por_plazo.sql"), "w", encoding="utf-8") as handle:
        handle.write(sales_sql + "\n")

    # the answer, first: printed and written next to the charts
    answer_lines = acquisition_answer(context, acquisition, retention, bridge, sales)
    answer_path = os.path.join(context.output_folder, f"{number.replace('.', '_')}_respuesta.md")
    with open(answer_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(answer_lines) + "\n")
    print(f"[informe] {number} · respuesta:")
    for line in answer_lines:
        print(f"[informe]    {line}")
    paths = []

    # ─── 9.1 · the answer in two charts: the premium of every year and the bridge to next year ───
    figure, axes = new_figure(15, 17, 3, 1, gridspec_kw={"height_ratios": [2.2, 2.2, 3.2]})
    positions = np.arange(len(years))
    premiums = {year: premium_over_retention(acquisition[year], retention[year]) for year in years}
    width = 0.2
    axes[0].bar(positions - 1.5 * width, [acquisition[year]["renewed"] for year in years], width, color=INK,
                label="renewed by the acquisition (first renewal)")
    axes[0].bar(positions - 0.5 * width, [premiums[year]["at_retention_yield"] for year in years], width, color=INK_MUTED,
                label="the same pipeline at the yield of the retention")
    axes[0].bar(positions + 0.5 * width, [premiums[year]["uplift_premium"] for year in years], width, color=AQUA,
                label="uplift premium (it pays more when it renews)")
    axes[0].bar(positions + 1.5 * width, [premiums[year]["rate_gap"] for year in years], width, color=ORANGE,
                label="rate gap (it renews less)")
    axes[0].axhline(0, color=INK_MUTED, linewidth=0.8)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([year_label(year, context) for year in years], fontsize=8.5)
    money_axis(axes[0])
    axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    axes[0].set_title("The first renewal of the 1-year acquisitions against what the same pipeline would renew at the "
                      "yield of the retention", fontsize=10, color=INK, loc="left")
    bridge_waterfall(axes[1], f"renewed {now}", acquisition[now]["renewed"], bridge, f"renewed {upcoming}",
                     acquisition[upcoming]["renewed"])
    axes[1].set_title(f"From {now} to {upcoming}: the renewed money of the acquisition, one factor at a time (the volume "
                      "last: it is what the units not acquired cost)", fontsize=10, color=INK, loc="left")
    annex = {}
    for year in years:
        acq, ret, premium = acquisition[year], retention[year], premiums[year]
        annex[str(year)] = {
            "acquisition · units due": f"{acq['units']:,.0f}",
            "acquisition · USD due (pipeline)": money_label(acq["pipeline"]),
            "acquisition · USD due per unit (price at sale)": unit_price_text(acq["price"]),
            "acquisition · discount at sale (weighted by USD)": percent_text(acq["discount"]),
            "acquisition · rate in units": percent_text(acq["rate_units"]),
            "acquisition · rate in USD": percent_text(acq["rate"]),
            "acquisition · uplift": ratio_text(acq["uplift"]),
            "acquisition · renewed USD per renewed unit": unit_price_text(acq["renewed_price"]),
            "acquisition · renewed USD": money_label(acq["renewed"]),
            "acquisition · yield (renewed / pipeline)": percent_text(acq["yield"]),
            "retention · USD due per unit": unit_price_text(ret["price"]),
            "retention · discount (weighted by USD)": percent_text(ret["discount"]),
            "retention · rate in units": percent_text(ret["rate_units"]),
            "retention · uplift": ratio_text(ret["uplift"]),
            "retention · yield": percent_text(ret["yield"]),
            "retention · renewed USD": money_label(ret["renewed"]),
            "acquisition share of the 1-year renewed USD": percent_text(safe_ratio(acq["renewed"], acq["renewed"] + ret["renewed"])),
            "premium over retention": money_label(premium["premium"]),
            "   of which uplift premium": money_label(premium["uplift_premium"]),
            "   of which rate gap": money_label(premium["rate_gap"]),
        }
    annex = pd.DataFrame(annex)
    annex[f"{upcoming} vs {now}"] = ""
    for label, key in (("acquisition · units due", "units"), ("acquisition · USD due (pipeline)", "pipeline"),
                       ("acquisition · renewed USD", "renewed")):
        annex.loc[label, f"{upcoming} vs {now}"] = growth_text(acquisition[upcoming][key], acquisition[now][key])
    for effect, (label, _) in BRIDGE_EFFECTS.items():
        annex.loc[f"bridge {now} → {upcoming} · {label}", f"{upcoming} vs {now}"] = money_label(bridge[effect])
    annex = annex.fillna("")
    annex_table(axes[2], annex, "Annex · the 1-year pipeline by origin, every year (the current one: real to the last "
                                "closed month, forecast after; the next one: forecast)", font_size=7.5, first_column_width=0.3)
    write_annex(context, number, "adquisicion_frente_a_retencion", annex)
    titles(figure, f"{number} · How much does {upcoming} change because of the acquisition of {now}?",
           f"The first renewal of the acquisitions renews {money_label(acquisition[now]['renewed'])} in {now} and "
           f"{money_label(acquisition[upcoming]['renewed'])} in {upcoming} ({money_label(bridge['volume'] + bridge['price_at_sale'] + bridge['rate'] + bridge['uplift'])}): "
           f"volume {money_label(bridge['volume'])}, price at sale {money_label(bridge['price_at_sale'])}, rate "
           f"{money_label(bridge['rate'])}, uplift {money_label(bridge['uplift'])}. renewed = units × USD per unit × "
           "rate × uplift; the acquisition is sold with a discount, renews less and pays more when it renews.")
    paths.append(save(figure, chart_path(context, number, "adquisicion_en_la_renovacion"), top=0.93))

    # ─── 9.1a · by region: where the change of next year comes from ───
    regions_table = region_rows(table)
    region_column = context.region
    region_lines = {}
    for region, of_region in regions_table.groupby(region_column):
        acquisition_of_region = factors_by_year(of_region, ORIGIN_ACQUISITION)
        retention_of_region = factors_by_year(of_region, ORIGIN_RETENTION)
        if now not in acquisition_of_region or upcoming not in acquisition_of_region:
            continue
        region_bridge = acquisition_bridge(acquisition_of_region[now], acquisition_of_region[upcoming])
        premium_next = (premium_over_retention(acquisition_of_region[upcoming], retention_of_region[upcoming])
                        if upcoming in retention_of_region else {"premium": np.nan})
        region_lines[str(region)] = {"now": acquisition_of_region[now], "next": acquisition_of_region[upcoming],
                                     "bridge": region_bridge, "premium_next": premium_next["premium"]}
    if region_lines:
        order = sorted(region_lines, key=lambda region: region_lines[region]["next"]["renewed"] - region_lines[region]["now"]["renewed"])
        figure, axes = new_figure(15, 0.8 * len(order) + 5.5, 2, 1,
                                  gridspec_kw={"height_ratios": [max(2.0, 0.6 * len(order)), 0.3 * len(order) + 1.2]})
        positions = np.arange(len(order))
        bar_height = 0.2
        for offset, (effect, (label, colour)) in zip((-1.5, -0.5, 0.5, 1.5), BRIDGE_EFFECTS.items()):
            axes[0].barh(positions + offset * bar_height, [region_lines[region]["bridge"][effect] for region in order],
                         bar_height, color=colour, label=label)
        changes = [region_lines[region]["next"]["renewed"] - region_lines[region]["now"]["renewed"] for region in order]
        axes[0].scatter(changes, positions, color=INK, marker="D", s=22, zorder=3, label=f"change {now} → {upcoming}")
        axes[0].axvline(0, color=INK_MUTED, linewidth=0.8)
        axes[0].set_yticks(positions)
        axes[0].set_yticklabels(order, fontsize=8)
        axes[0].xaxis.set_major_formatter(lambda value, _: money_label(value))
        axes[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        axes[0].set_title(f"The change of the renewed money of the acquisition {now} → {upcoming} by region, split by factor",
                          fontsize=10, color=INK, loc="left")
        region_annex = pd.DataFrame({region: {
            f"renewed {now}": money_label(line["now"]["renewed"]), f"renewed {upcoming}": money_label(line["next"]["renewed"]),
            "change": money_label(line["next"]["renewed"] - line["now"]["renewed"]),
            **{BRIDGE_SHORT_NAMES[effect]: money_label(line["bridge"][effect]) for effect in BRIDGE_EFFECTS},
            "units due": f"{line['now']['units']:,.0f} → {line['next']['units']:,.0f}",
            "USD/unit due": f"{unit_price_text(line['now']['price'])} → {unit_price_text(line['next']['price'])}",
            "uplift": f"{ratio_text(line['now']['uplift'])} → {ratio_text(line['next']['uplift'])}",
            f"premium {upcoming}": money_label(line["premium_next"])} for region, line in region_lines.items()}).T
        region_annex = region_annex.loc[order[::-1]]
        annex_table(axes[1], region_annex, "Annex · by region", font_size=7, first_column_width=0.14)
        write_annex(context, number, "por_region", region_annex)
        titles(figure, f"{number}a · Where does the change of the acquisition come from, by region?",
               "Volume: the units due (what was acquired a year before); price at sale: the USD due per unit (the "
               "discount it was sold with); rate and uplift: how much of it renews and at what price.")
        paths.append(save(figure, chart_path(context, number, "adquisicion_por_region"), top=0.93))

    # ─── 9.1b · month by month, and the sales of acquisition by term ───
    monthly = total[(total["origen"] == ORIGIN_ACQUISITION) & (total["plazo"] == TERM_ONE_YEAR)].copy()
    monthly["mes"] = monthly["periodo"].astype(str).str[5:7].astype(int)
    monthly = monthly.groupby(["anio", "mes"])[FACTOR_SUMS].sum().reset_index()
    monthly["uplift"] = monthly["renovado_usd"] / monthly["base_usd"].replace(0, np.nan)
    figure, axes = new_figure(15, 13, 3, 1, gridspec_kw={"height_ratios": [2.2, 2.2, 1.0]})
    lines_by_year(axes[0], monthly, "renovado_usd", context)
    money_axis(axes[0])
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[0].set_title("Renewed USD of the first renewal of the 1-year acquisitions, by due month (dashed: forecast)",
                      fontsize=10, color=INK, loc="left")
    lines_by_year(axes[1], monthly, "uplift", context)
    axes[1].set_title("Its uplift (renewed / USD due of what renews), by due month", fontsize=10, color=INK, loc="left")
    if len(sales):
        sales_annex = sales.copy()
        sales_annex["term"] = sales_annex["plazo_anos"].map(lambda years: f"{years} year" + ("s" if years > 1 else ""))
        sales_annex["value"] = [f"{money_label(usd)} · {units:,.0f} u" if visible else "not visible yet"
                                for usd, units, visible in zip(sales_annex["vendido_usd"], sales_annex["vendidas_unidades"],
                                                               sales_annex["visible"])]
        sales_annex = sales_annex.pivot(index="term", columns="anio_venta", values="value")
        sales_annex.columns = [str(column) for column in sales_annex.columns]
        months = sales.attrs.get("months_to_date", 12)
        annex_table(axes[2], sales_annex, f"Annex · acquisition sold by term, January to {ENGLISH_MONTHS[months - 1]} of "
                                          "every year (USD as pipeline · units). Only the 1-year part falls due next year",
                    font_size=8, first_column_width=0.14)
        write_annex(context, number, "ventas_por_plazo", sales_annex)
    else:
        axes[2].axis("off")
    titles(figure, f"{number}b · The acquisition month by month, and what was sold by term",
           "A month where the first renewal falls is a month whose acquisition a year before was lower; a sale of 2 or 3 "
           "years is pipeline that does not come back next year.")
    paths.append(save(figure, chart_path(context, number, "adquisicion_mes_a_mes"), top=0.93))
    return paths


# ═══════════════════════════════════════════════════════════════════════════════════
# THE QUESTIONS, IN THE ORDER OF THE STORY
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class Question:
    number: str
    chapter: str
    title: str
    name: str
    build_sql: Callable
    draw: Optional[Callable] = None
    persist_as: Optional[str] = None      # a table of the database the answer is also written to (sff_<name>)


@dataclass
class Answer:
    question: Question
    sql: Optional[str] = None
    table: Optional[pd.DataFrame] = None
    charts: list = field(default_factory=list)
    note: str = ""


CHAPTER_REAL = "1 · El dato real"
CHAPTER_SIMULATED = "2 · Lo que simulo para construir"
CHAPTER_BUILT = "3 · Lo que he construido"
CHAPTER_RESULT = "4 · El resultado"
CHAPTER_BREAKDOWN = "5 · Cómo desgranarlo"
CHAPTER_REAL_BANDS = "6 · Las bandas reales"
CHAPTER_COMPONENTS = "7 · La pipeline auditada por componentes"
CHAPTER_NEXT_YEAR = "8 · Entender el año que viene"
CHAPTER_MONOGRAPHS = "9 · Monográficos: una pregunta de negocio, su respuesta y su detalle"

QUESTIONS = [
    Question("1.1", CHAPTER_REAL, "¿Cuánta pipeline real vence cada año y cuánto se ha renovado?", "pipeline_real_por_anio", sql_1_1, draw_1_1),
    Question("1.2", CHAPTER_REAL, "¿Hay continuidad mes a mes en la pipeline real?", "pipeline_real_mes_a_mes", sql_1_2, draw_1_2),
    Question("1.3", CHAPTER_REAL, "¿Cuánta pipeline real hay hoy para los meses que vienen?", "pipeline_real_que_viene", sql_1_3, draw_1_3),
    Question("1.4", CHAPTER_REAL, "¿Cómo ha evolucionado la tasa de renovación real?", "tasa_real", sql_1_4, draw_1_4),
    Question("1.5", CHAPTER_REAL, "¿Cómo evoluciona la adquisición real (por región)?", "adquisicion_real", sql_1_5, draw_1_5),
    Question("1.5b", CHAPTER_REAL, "¿Cómo evoluciona la adquisición real por producto?", "adquisicion_real_por_producto", sql_1_5b, draw_1_5b),
    Question("1.6", CHAPTER_REAL, "¿Es estable el retail to subscription real?", "retail_to_subscription_real", sql_1_6, draw_1_6),
    Question("1.7", CHAPTER_REAL, "¿Cuánto dormant y softcancel hay hoy en lo que vence?", "estados_hoy", sql_1_7, draw_1_7),
    Question("2.1", CHAPTER_SIMULATED, "¿Cómo maduran los estados y cuánto acierta la curva?", "curvas_de_maduracion", sql_2_1, draw_2_1),
    Question("2.2", CHAPTER_SIMULATED, "¿Cuánto madurarán los estados de lo que vence?", "maduracion_por_mes", sql_2_2, draw_2_2),
    Question("2.3", CHAPTER_SIMULATED, "¿Qué renovará de lo que vence hasta diciembre y cuánto vuelve como pipeline?", "renovacion_proyectada", sql_2_3, draw_2_3),
    Question("2.4", CHAPTER_SIMULATED, "¿Cuánta adquisición simulo y cuadra con la real y la previsión?", "adquisicion_simulada", sql_2_4, draw_2_4),
    Question("2.5", CHAPTER_SIMULATED, "¿Qué retail to subscription simulo?", "retail_to_subscription_simulado", sql_2_5, draw_2_5),
    Question("3.1", CHAPTER_BUILT, "La pipeline del año que viene por sus cuatro orígenes", "pipeline_por_origen", sql_3_1, draw_3_1),
    Question("3.2", CHAPTER_BUILT, "¿Hay doble conteo con lo que el extracto ya trae?", "doble_conteo", sql_3_2, draw_3_2),
    Question("4.1", CHAPTER_RESULT, "¿Cómo acaba el año en curso y de qué se compone?", "como_acaba_el_anio", sql_4_1, draw_4_1),
    Question("4.2", CHAPTER_RESULT, "El año en curso mes a mes: pipeline, renovado y tasa", "anio_en_curso_mes_a_mes", sql_4_2, draw_4_2),
    Question("4.3", CHAPTER_RESULT, "¿Cómo acaba el año que viene y de dónde sale?", "como_acaba_el_siguiente", sql_4_3, draw_4_3),
    Question("4.4", CHAPTER_RESULT, "El año que viene mes a mes", "siguiente_mes_a_mes", sql_4_4, draw_4_4),
    Question("5.1", CHAPTER_BREAKDOWN, "La cadena por región y subregión del año que viene, con sus bandas", "cadena_por_region", sql_5_1, draw_5_1),
    Question("5.1b", CHAPTER_BREAKDOWN, "La cadena por región de lo que queda del año en curso", "cadena_por_region_anio_en_curso", sql_5_1b, draw_5_1b),
    Question("5.2", CHAPTER_BREAKDOWN, "La cadena por producto del año que viene", "cadena_por_producto", sql_5_2, draw_5_2),
    Question("5.3", CHAPTER_BREAKDOWN, "¿Ha sido honesta la banda?", "banda_honesta", sql_5_3, draw_5_3),
    Question("5.4", CHAPTER_BREAKDOWN, "¿Cuánto cuestan dormant y softcancel? El potencial", "coste_de_los_estados", sql_5_4, draw_5_4),
    Question("6.1", CHAPTER_REAL_BANDS, "Las bandas reales: el error del examen a cada nivel frente a la banda de cada regla",
             "bandas_reales", sql_6_1, draw_6_1, persist_as=TABLE_REAL_BANDS),
    Question("6.2", CHAPTER_REAL_BANDS, "¿Qué regla de suma acierta a cada nivel?", "bandas_reales_resumen", sql_6_2, draw_6_2,
             persist_as=TABLE_REAL_BANDS_SUMMARY),
    Question("7.1", CHAPTER_COMPONENTS, "¿De qué se compone la pipeline mes a mes, del pasado real al año que viene?",
             "componentes_mes_a_mes", sql_7_1, draw_7_1),
    Question("7.2", CHAPTER_COMPONENTS, "¿Cuánta pipeline tiene cada región este año y el que viene, y cuánto es simulado?",
             "pipeline_por_region", sql_7_2, draw_7_2),
    Question("7.3", CHAPTER_COMPONENTS, "¿Lo renovado vuelve como pipeline 12 meses después?",
             "renovado_vuelve_como_pipeline", sql_7_3, draw_7_3),
    Question("7.4", CHAPTER_COMPONENTS, "¿Es estable el uplift de cada componente, con y sin subidas de precio?",
             "uplift_por_componente", sql_7_4, draw_7_4),
    Question("8.1", CHAPTER_NEXT_YEAR, "¿Dónde están las mayores diferencias con el forecast del equipo de retención?",
             "frente_al_equipo_de_retencion", sql_8_1, draw_8_1),
    Question("8.2", CHAPTER_NEXT_YEAR, "¿Son parecidas las tasas de renovación de un año a otro?", "tasa_por_anio", sql_8_2, draw_8_2),
    Question("8.3", CHAPTER_NEXT_YEAR, "¿Se está acabando el efecto de las subidas de precio en la revalorización?",
             "efecto_de_las_subidas", sql_8_3, draw_8_3),
    Question("8.4", CHAPTER_NEXT_YEAR, "¿Cómo evolucionan las combinaciones de referencia de precio (región × producto × plazo × banda)?",
             "combinaciones_de_precio", sql_8_4, draw_8_4),
    Question("8.5", CHAPTER_NEXT_YEAR, "¿Es distinta la composición de la pipeline del año que viene?",
             "composicion_de_la_pipeline", sql_8_5, draw_8_5),
    Question("8.6", CHAPTER_NEXT_YEAR, "¿Maduran bien los estados del cliente (dormant, softcancel, no instalado)?",
             "maduracion_de_los_estados", sql_8_6, draw_8_6),
    Question("8.7", CHAPTER_NEXT_YEAR, "¿Se queda corto el horizonte de más de un año?", "horizonte", sql_8_7, draw_8_7),
    Question("8.8", CHAPTER_NEXT_YEAR, "¿Qué región tiene la composición más favorable?", "composicion_por_region",
             sql_8_8, draw_8_8),
    Question("8.9", CHAPTER_NEXT_YEAR, "¿Dónde se han detectado subidas de precio y desde cuándo?", "subidas_detectadas",
             sql_8_9, draw_8_9),
    Question("8.10", CHAPTER_NEXT_YEAR, "¿A qué plazo renuevan las licencias?", "transiciones_de_plazo", sql_8_10, draw_8_10),
    Question("9.1", CHAPTER_MONOGRAPHS, "¿Cuánto cambia el año que viene por la adquisición de este año?",
             "adquisicion_en_la_renovacion", sql_9_1, draw_9_1),
]


def answer(question: Question, context: ReportContext, draw: bool = True) -> Answer:
    """One question: its SQL (written to a .sql file), its result (to a .csv) and its charts."""
    result = Answer(question)
    result.sql = question.build_sql(context)
    if result.sql is None:
        result.note = "no se puede responder con las tablas de esta ejecución (falta una tabla o una columna)"
        return result
    base_name = os.path.join(context.output_folder, f"{question.number.replace('.', '_')}_{question.name}")
    with open(f"{base_name}.sql", "w", encoding="utf-8") as handle:
        handle.write(result.sql + "\n")
    result.table = pd.read_sql(result.sql, context.engine)
    context.answered[question.number] = result.table
    result.table.to_csv(f"{base_name}.csv", index=False)
    if question.persist_as:            # a table to learn from in SQL and Power BI, next to the tables of the run
        physical_name = f"{context.configuration.table_prefix}{question.persist_as}"
        write_frame_to_sql(result.table, physical_name, context.engine, context.configuration.sql_schema)
        result.note = f"escrita en {physical_name}"
    if draw and question.draw is not None and len(result.table):
        result.charts = question.draw(result.table, context, question.number)
    return result


def show_what_the_core_has(context: ReportContext) -> None:
    """Before the questions: the money of the core by year and universe, so a missing universe (no retail to
    subscription rows, a run that did not finish) is seen at once, not inferred from a total that comes out low."""
    summary = pd.read_sql(f"""SELECT forecast_year AS anio, forecast_universe AS universo, COUNT(*) AS filas,
       SUM(COALESCE(forecast_to_renew_USD, 0)) AS pipeline_usd, SUM(COALESCE(forecast_renewed_USD, 0)) AS renovado_usd
FROM {context.core}
WHERE forecast_year BETWEEN {context.current_year - 1} AND {context.next_year}
GROUP BY forecast_year, forecast_universe
ORDER BY forecast_year, forecast_universe""", context.engine)
    print(f"[informe] lo que trae {context.core} (mes en curso {context.current_month}):")
    for _, row in summary.iterrows():
        print(f"[informe]    {int(row['anio'])} · {row['universo']:<12} {int(row['filas']):>10,} filas · "
              f"pipeline {money_label(row['pipeline_usd'])} · renovado / ingresos {money_label(row['renovado_usd'])}")
    if context.has_table(TABLE_BUSINESS_SUMMARY):     # the same money as step 7.2 wrote it: they must agree
        business = pd.read_sql(f"SELECT ano, total_usd FROM {table_in_sql(context.configuration, TABLE_BUSINESS_SUMMARY)}",
                               context.engine).set_index("ano")["total_usd"]
        core_by_year = summary.groupby("anio")["renovado_usd"].sum()
        for year, core_total in core_by_year.items():
            written = float(business.get(year, np.nan))
            agree = np.isfinite(written) and abs(core_total - written) <= 1e-6 * max(1.0, abs(written))
            print(f"[informe]    {int(year)} · núcleo {money_label(core_total)} · sff_resumen_negocio {money_label(written)}"
                  + ("" if agree else "  ← NO COINCIDEN: el núcleo y el resumen no son de la misma ejecución"))
    else:
        print(f"[informe] AVISO: {context.configuration.table_prefix}{TABLE_BUSINESS_SUMMARY} no está en este schema: la "
              f"ejecución no llegó a escribir el paso 7.2, o lo escribió en otro schema o con otro prefijo. El núcleo "
              f"puede ser de otra ejecución o haber quedado a medias: vuelve a ejecutar el framework y mira el final del log.")
    if "time_series" not in set(summary["universo"]):
        print("[informe] AVISO: el núcleo no tiene filas time_series (retail to subscription): las preguntas del año "
              "salen sin él. Mira el paso 6.1 en el log de la ejecución.")


def report(configuration: Config, which: list = None, acquisition_plan_units: Optional[float] = None,
           show: bool = True, retention_reference_file: Optional[str] = None) -> dict:
    """Every question of the story (or those of which), in order: {number: Answer}. In a notebook the charts are shown.
    retention_reference_file: the retention team's forecast of next year for 8.1 (None = forecast_retencion_2027.csv)."""
    context = ReportContext(configuration, acquisition_plan_units=acquisition_plan_units,
                            retention_reference_file=retention_reference_file)
    show_what_the_core_has(context)
    answers = {}
    for question in QUESTIONS:
        if which and question.number not in which:
            continue
        started = time.time()
        try:
            with matplotlib.rc_context({"text.parse_math": False}):       # "$" in a label is a dollar, not maths
                result = answer(question, context)
        except Exception as error:          # one question that fails does not stop the story: it says why
            result = Answer(question, note=f"falló: {type(error).__name__}: {error}")
        answers[question.number] = result
        status = (f"{len(result.table)} filas · {len(result.charts)} gráficos" + (f" · {result.note}" if result.note else "")
                  if result.table is not None else result.note) + f" · {time.time() - started:.0f} s"
        print(f"[informe] {question.number} {question.title} → {status}")
        if show and running_in_notebook() and result.charts:
            from IPython.display import Image, display
            for path in result.charts:
                display(Image(filename=path))
    print(f"[informe] SQL, CSV y PNG en {context.output_folder}")
    return answers


if __name__ == "__main__":
    # python informe_forecast.py [sintetico | parecido [YYYY-MM]] [numero …]
    # only the Config asked for is imported: a main.py with production_configuration alone is enough for production
    import main
    arguments = sys.argv[1:]
    numbers = [argument for argument in arguments if argument[:1].isdigit() and "." in argument]
    months = [argument for argument in arguments if argument[:1].isdigit() and "-" in argument]
    if "sintetico" in arguments:
        run_configuration = main.synthetic_configuration()
    elif "parecido" in arguments:
        run_configuration = main.like_real_configuration(*months[:1])
    else:
        run_configuration = main.production_configuration()
    report(run_configuration, numbers or None)
