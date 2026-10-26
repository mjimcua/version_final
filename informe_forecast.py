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
  (6, the method of the bands, is text in the document; chapters 4 and 5 use it through composition_band_query)

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
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import inspect

from config import (TABLE_CORE, TABLE_FORECAST_MATURATION, TABLE_MATURATION_CURVE, Config, running_in_notebook,
                    write_frame_to_sql)
from graficos import (AQUA, BLUE, INK, INK_MUTED, INK_SECONDARY, NEUTRAL_FILL, ORANGE, VIOLET, money_label,
                      new_figure, save, titles)
from report_queries import table_in_sql, year_by_region_columns

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
                 product_column: str = "tr_product_level_1"):
        self.configuration = configuration
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
        condition = f"{prefix}{configuration.acquisition_column} IN ({values})"
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


def report(configuration: Config, which: list = None, acquisition_plan_units: Optional[float] = None,
           show: bool = True) -> dict:
    """Every question of the story (or those of which), in order: {number: Answer}. In a notebook the charts are shown."""
    context = ReportContext(configuration, acquisition_plan_units=acquisition_plan_units)
    answers = {}
    for question in QUESTIONS:
        if which and question.number not in which:
            continue
        try:
            with matplotlib.rc_context({"text.parse_math": False}):       # "$" in a label is a dollar, not maths
                result = answer(question, context)
        except Exception as error:          # one question that fails does not stop the story: it says why
            result = Answer(question, note=f"falló: {type(error).__name__}: {error}")
        answers[question.number] = result
        status = (f"{len(result.table)} filas · {len(result.charts)} gráficos" + (f" · {result.note}" if result.note else "")
                  if result.table is not None else result.note)
        print(f"[informe] {question.number} {question.title} → {status}")
        if show and running_in_notebook() and result.charts:
            from IPython.display import Image, display
            for path in result.charts:
                display(Image(filename=path))
    print(f"[informe] SQL, CSV y PNG en {context.output_folder}")
    return answers


if __name__ == "__main__":
    # python informe_forecast.py [sintetico | parecido [YYYY-MM]] [numero …]
    from main import LIKE_REAL_KEYWORD, SYNTHETIC_KEYWORD, like_real_configuration, production_configuration, synthetic_configuration
    arguments = sys.argv[1:]
    numbers = [argument for argument in arguments if argument[:1].isdigit() and "." in argument]
    months = [argument for argument in arguments if argument[:1].isdigit() and "-" in argument]
    if SYNTHETIC_KEYWORD in arguments:
        run_configuration = synthetic_configuration()
    elif LIKE_REAL_KEYWORD in arguments:
        run_configuration = like_real_configuration(*months[:1])
    else:
        run_configuration = production_configuration()
    report(run_configuration, numbers or None)
