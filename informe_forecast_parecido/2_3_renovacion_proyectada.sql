-- 2.3 · lo que vence hasta diciembre, lo que esperamos que renueve y lo que vuelve como pipeline 12 meses después
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS mes_que_vence,
           COALESCE(forecast_to_renew_USD, 0) AS vence_usd, COALESCE(forecast_renewed_USD, 0) AS renovado_esperado_usd,
           0 AS vuelve_como_pipeline_usd
    FROM sff_nucleo
    WHERE forecast_pipeline_source = 'pipeline_real_esperado' AND forecast_year = 2026
    UNION ALL
    SELECT tr_regional_level_1, tr_regional_level_3, CONCAT(CAST(forecast_year - 1 AS VARCHAR(4)), SUBSTRING(period, 5, 3)) AS mes_que_vence,
           0, 0, COALESCE(forecast_to_renew_USD, 0)
    FROM sff_nucleo
    WHERE origen_fila = 'proyectada'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, mes_que_vence,
       SUM(vence_usd) AS vence_usd,
       SUM(renovado_esperado_usd) AS renovado_esperado_usd,
       SUM(vuelve_como_pipeline_usd) AS vuelve_como_pipeline_usd
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, mes_que_vence
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', mes_que_vence,
       SUM(vence_usd) AS vence_usd,
       SUM(renovado_esperado_usd) AS renovado_esperado_usd,
       SUM(vuelve_como_pipeline_usd) AS vuelve_como_pipeline_usd
FROM base
GROUP BY tr_regional_level_1, mes_que_vence
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', mes_que_vence,
       SUM(vence_usd) AS vence_usd,
       SUM(renovado_esperado_usd) AS renovado_esperado_usd,
       SUM(vuelve_como_pipeline_usd) AS vuelve_como_pipeline_usd
FROM base
GROUP BY mes_que_vence
ORDER BY 1, 2, 3, 4, 5;
