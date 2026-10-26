"""
report_queries.py — The SQL that reproduces every aggregated report from the detail tables.

A report with few rows because it is very aggregated (per month, per role, per year and origin, per
method) is not a source of data: it is a query over the detail tables the run writes. The step shows the
report and prints the query, so anyone can reproduce it in SQL, or in Power BI, from the same tables, and
the report never becomes a second copy of the data.

Every query reads only tables the run writes (sff_nucleo, sff_series_exam_detail) and uses plain SQL that
SQL Server and SQLite both run (the tests run every query on SQLite and compare it with the report).

  new_rows_query          step 8.1  the rows the framework adds or wipes, month × origin   ← sff_nucleo
  roles_query             step 1.3  the calendar per role                                 ← sff_nucleo
  calendar_query          step 1.3  the calendar per month                                ← sff_nucleo
  forecast_total_query    step 7.3  the total per year and origin (sff_forecast_total)    ← sff_nucleo
  exam_by_month_query     step 7.1  the total of every exam month per method              ← sff_series_exam_detail
  exam_summary_query      step 7.1  the exam per method and horizon                       ← sff_series_exam_detail
  year_by_region_query    step 8.6  the year by region: renewed so far and expected to its end ← sff_nucleo

The reports computed with medians or quantiles (the ladder summary, the error bands) or from the JSON of
the generated levels have no query here: they are not a plain sum over a detail table.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
from config import TABLE_CORE, TABLE_SERIES_EXAM_DETAIL, region_columns_of


def table_in_sql(configuration, table_name: str) -> str:
    """The name of a table of the run as SQL writes it: schema.prefix+name (or prefix+name)."""
    physical_name = f"{configuration.table_prefix}{table_name}"
    return f"{configuration.sql_schema}.{physical_name}" if configuration.sql_schema else physical_name


def new_rows_query(configuration) -> str:
    """Step 8.1: the gaps, what the calendar wiped and what the forecast created, month × origin."""
    core = table_in_sql(configuration, TABLE_CORE)
    period = configuration.period_col
    return f"""-- step 8.1 · the rows the framework adds or wipes, month × origin
SELECT {period} AS mes, s1_3_rol AS rol, 'hueco' AS origen, COUNT(*) AS filas, COUNT(DISTINCT s1_4_fs_id) AS series,
       0 AS unidades, 0 AS usd, NULL AS esperado_usd
FROM {core} WHERE origen_fila = 'hueco' GROUP BY {period}, s1_3_rol
UNION ALL
SELECT {period}, s1_3_rol, 'resultado_adelantado_borrado', COUNT(*), COUNT(DISTINCT s1_4_fs_id),
       SUM(s1_1_renovadas_unidades - COALESCE(s1_3_renovadas_unidades, 0)),
       SUM(s1_1_renovado_usd - COALESCE(s1_3_renovado_usd, 0)), NULL
FROM {core} WHERE origen_fila = 'raw' AND COALESCE(s1_1_renovadas_unidades, 0) > COALESCE(s1_3_renovadas_unidades, 0)
GROUP BY {period}, s1_3_rol
UNION ALL
SELECT {period}, s1_3_rol, 'pipeline_parcial_borrada', COUNT(*), COUNT(DISTINCT s1_4_fs_id),
       SUM(s1_1_vencen_unidades - s1_3_vencen_unidades), SUM(s1_1_vencen_usd - s1_3_vencen_usd), NULL
FROM {core} WHERE origen_fila = 'raw' AND s1_1_vencen_unidades > s1_3_vencen_unidades
GROUP BY {period}, s1_3_rol
UNION ALL
SELECT {period}, s1_3_rol, origen_fila, COUNT(*), COUNT(DISTINCT s1_4_fs_id),
       SUM(forecast_to_renew_units), SUM(forecast_to_renew_USD), SUM(s7_2_esperado_usd)
FROM {core} WHERE origen_fila IN ('proyectada', 'simulada') GROUP BY {period}, s1_3_rol, origen_fila
UNION ALL
SELECT {period}, s1_3_rol, origen_fila, COUNT(*), COUNT(DISTINCT s1_4_fs_id),
       SUM(forecast_to_renew_units_at_due), SUM(forecast_to_renew_USD_at_due), SUM(s7_2_esperado_usd)
FROM {core} WHERE origen_fila = 'maduracion' GROUP BY {period}, s1_3_rol, origen_fila
ORDER BY mes, origen;"""


def roles_query(configuration) -> str:
    """Step 1.3: the calendar per role (the rows of the extract)."""
    core = table_in_sql(configuration, TABLE_CORE)
    period = configuration.period_col
    return f"""-- step 1.3 · the calendar per role
SELECT s1_3_rol AS rol, COUNT(DISTINCT {period}) AS meses, MIN({period}) AS desde, MAX({period}) AS hasta, COUNT(*) AS filas,
       SUM(s1_3_vencen_unidades) AS unidades_vencen, SUM(s1_3_renovadas_unidades) AS unidades_renovadas,
       SUM(s1_3_renovadas_unidades) / SUM(s1_3_vencen_unidades) AS tasa_unidades
FROM {core} WHERE origen_fila = 'raw'
GROUP BY s1_3_rol ORDER BY MIN({period});"""


def calendar_query(configuration) -> str:
    """Step 1.3: the calendar per month."""
    core = table_in_sql(configuration, TABLE_CORE)
    period = configuration.period_col
    return f"""-- step 1.3 · the calendar per month
SELECT {period}, MIN(s1_3_rol) AS rol, MAX(s1_3_es_mes_en_curso) AS es_mes_en_curso, COUNT(*) AS filas,
       SUM(s1_3_vencen_unidades) AS unidades_vencen, SUM(s1_3_vencen_usd) AS usd_vence,
       SUM(s1_3_renovadas_unidades) AS unidades_renovadas, SUM(s1_3_renovado_usd) AS usd_renovado,
       SUM(s1_1_renovadas_unidades) AS s0_renovados_unidades, SUM(s1_1_renovado_usd) AS s0_renovados_usd,
       SUM(s1_3_renovadas_unidades) / SUM(s1_3_vencen_unidades) AS tasa_unidades
FROM {core} WHERE origen_fila = 'raw'
GROUP BY {period} ORDER BY {period};"""


def forecast_total_query(configuration) -> str:
    """Step 7.3: the total per year and origin, and the TOTAL of every year (the content of sff_forecast_total)."""
    core = table_in_sql(configuration, TABLE_CORE)
    first_year = configuration.calendar_boundaries()["current"].year
    # the rows that carry no money in the answer are left out: a gap (a month with nothing due) and an original
    # time_series row of a month not closed yet (its result is replaced by its projection)
    without_money = "origen_fila NOT IN ('hueco', 'ts_sin_resultado')"
    return f"""-- step 7.3 · the total per year and origin (sff_forecast_total)
SELECT forecast_year AS ano, forecast_pipeline_source AS origen, SUM(forecast_to_renew_USD) AS usd_vence, SUM(forecast_renewed_USD) AS usd_renovado
FROM {core} WHERE forecast_year >= {first_year} AND {without_money}
GROUP BY forecast_year, forecast_pipeline_source
UNION ALL
SELECT forecast_year, 'TOTAL', SUM(forecast_to_renew_USD), SUM(forecast_renewed_USD)
FROM {core} WHERE forecast_year >= {first_year} AND {without_money}
GROUP BY forecast_year
ORDER BY ano, origen;"""


def exam_by_month_query(configuration, methods: list) -> str:
    """Step 7.1: per exam month and horizon, the real renewals and every method's prediction and error of the total."""
    detail = table_in_sql(configuration, TABLE_SERIES_EXAM_DETAIL)
    period = configuration.period_col
    reference = methods[0]
    per_method = ",\n       ".join(
        f"SUM(CASE WHEN method = '{method}' THEN pred_units END) AS {method}_pred,\n"
        f"       SUM(CASE WHEN method = '{method}' THEN pred_units END) / SUM(CASE WHEN method = '{reference}' THEN real_units END) - 1"
        f" AS {method}_error_total" for method in methods)
    return f"""-- step 7.1 · the total of every exam month and horizon, per method
SELECT {period} AS mes, h, SUM(CASE WHEN method = '{reference}' THEN real_units END) AS renovadas_reales,
       {per_method}
FROM {detail}
GROUP BY {period}, h ORDER BY {period}, h;"""


def exam_summary_query(configuration) -> str:
    """Step 7.1: per method and horizon, the mean error of the total, series-by-series WAPE, the error
    against the noise and the share in the interval."""
    detail = table_in_sql(configuration, TABLE_SERIES_EXAM_DETAIL)
    period = configuration.period_col
    return f"""-- step 7.1 · the exam per method and horizon
WITH month_total AS (
    SELECT method, h, {period}, SUM(pred_units) / SUM(real_units) - 1 AS error_total
    FROM {detail} GROUP BY method, h, {period})
SELECT d.method AS metodo, d.h,
       (SELECT AVG(ABS(m.error_total)) FROM month_total m WHERE m.method = d.method AND m.h = d.h) AS error_total_medio,
       (SELECT AVG(m.error_total) FROM month_total m WHERE m.method = d.method AND m.h = d.h) AS sesgo_total_medio,
       SUM(ABS(d.err_units)) / SUM(d.real_units) AS wape_series,
       SQRT(AVG(d.err_pp * d.err_pp)) / NULLIF(SQRT(AVG(d.noise_pp * d.noise_pp)), 0) AS error_vs_ruido,
       AVG(d.in_band) AS en_intervalo
FROM {detail} d
GROUP BY d.method, d.h ORDER BY d.h, error_total_medio;"""


def year_by_region_columns(configuration) -> list:
    """The two region levels of the year by region: the first and the last of region_columns_of (with the
    production Config, tr_regional_level_1 and tr_regional_level_3); one level when only one is declared."""
    regions = region_columns_of(configuration)
    return list(dict.fromkeys([regions[0], regions[-1]]))


def year_by_region_query(configuration) -> str:
    """Step 8.6, the final table: the USD renewed in the current year by region, in two columns: what really
    renewed in the closed months (forecast_status = 'actual', up to the last closed month) and what is expected
    from the current month to December (forecast_status = 'forecast'), and their sum. One row per region of the
    first and last level, then the subtotal of every region of the first level (its finer level = 'TOTAL'), then the
    TOTAL of the year.

    The same rows as the total of step 7.3: a gap (nothing due) and an original time_series row of a month not
    closed yet (replaced by its projection) carry no money and are left out. The TOTAL equals the TOTAL row of
    sff_forecast_total for the year (step 8.6 checks it). Plain SQL (UNION ALL, no ROLLUP): SQL Server and SQLite."""
    core = table_in_sql(configuration, TABLE_CORE)
    year = configuration.calendar_boundaries()["current"].year
    without_money = "origen_fila NOT IN ('hueco', 'ts_sin_resultado')"
    first_level, *finer = year_by_region_columns(configuration)
    last_level = finer[-1] if finer else None
    real = "SUM(CASE WHEN forecast_status = 'actual' THEN forecast_renewed_USD ELSE 0 END)"
    expected = "SUM(CASE WHEN forecast_status = 'forecast' THEN forecast_renewed_USD ELSE 0 END)"
    total = "SUM(forecast_renewed_USD)"
    money = f"{real} AS usd_real_hasta_hoy, {expected} AS usd_esperado_hasta_fin_de_ano, {total} AS usd_total_ano"
    where = f"WHERE forecast_year = {year} AND {without_money}"
    detail = (f"""SELECT 0 AS orden_total, {first_level} AS {first_level}, 0 AS orden_region, {last_level} AS {last_level}, {money}
FROM {core} {where}
GROUP BY {first_level}, {last_level}
UNION ALL
""" if last_level else "")
    last_label = f", 'TOTAL' AS {last_level}" if last_level else ""   # the subtotal of a region: its finer level says TOTAL
    return f"""-- step 8.6 · the year {year} by region: renewed so far (closed months) and expected to December
{detail}SELECT 0 AS orden_total, {first_level} AS {first_level}, 1 AS orden_region{last_label}, {money}
FROM {core} {where}
GROUP BY {first_level}
UNION ALL
SELECT 1, 'TOTAL', 2{", 'TOTAL'" if last_level else ""}, {money}
FROM {core} {where}
ORDER BY 1, 2, 3, 4;"""
