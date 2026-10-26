-- 1.6 · el retail to subscription real por mes
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS periodo,
           COALESCE(forecast_renewed_USD, 0) AS ingresos_usd, COALESCE(forecast_renewed_units, 0) AS convertidas_unidades
    FROM sff_nucleo
    WHERE origen_fila = 'ts_real'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, periodo,
       SUM(ingresos_usd) AS ingresos_usd,
       SUM(convertidas_unidades) AS convertidas_unidades
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', periodo,
       SUM(ingresos_usd) AS ingresos_usd,
       SUM(convertidas_unidades) AS convertidas_unidades
FROM base
GROUP BY tr_regional_level_1, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', periodo,
       SUM(ingresos_usd) AS ingresos_usd,
       SUM(convertidas_unidades) AS convertidas_unidades
FROM base
GROUP BY periodo
ORDER BY 1, 2, 3, 4, 5;
