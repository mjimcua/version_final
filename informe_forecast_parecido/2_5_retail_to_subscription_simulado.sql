-- 2.5 · el retail to subscription: real, proyectado (lo que queda del año) y simulado, con su banda
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS periodo,
           CASE WHEN origen_fila = 'ts_real' THEN COALESCE(forecast_renewed_USD, 0) ELSE 0 END AS real_usd,
           CASE WHEN origen_fila <> 'ts_real' THEN COALESCE(forecast_renewed_USD, 0) ELSE 0 END AS simulado_usd,
           CASE WHEN origen_fila <> 'ts_real' THEN COALESCE(forecast_renewed_USD_low, 0) ELSE 0 END AS simulado_bajo_usd,
           CASE WHEN origen_fila <> 'ts_real' THEN COALESCE(forecast_renewed_USD_high, 0) ELSE 0 END AS simulado_alto_usd
    FROM sff_nucleo
    WHERE origen_fila IN ('ts_real', 'ts_proyectado', 'ts_simulado')
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, periodo,
       SUM(real_usd) AS real_usd,
       SUM(simulado_usd) AS simulado_usd,
       SUM(simulado_bajo_usd) AS simulado_bajo_usd,
       SUM(simulado_alto_usd) AS simulado_alto_usd
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', periodo,
       SUM(real_usd) AS real_usd,
       SUM(simulado_usd) AS simulado_usd,
       SUM(simulado_bajo_usd) AS simulado_bajo_usd,
       SUM(simulado_alto_usd) AS simulado_alto_usd
FROM base
GROUP BY tr_regional_level_1, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', periodo,
       SUM(real_usd) AS real_usd,
       SUM(simulado_usd) AS simulado_usd,
       SUM(simulado_bajo_usd) AS simulado_bajo_usd,
       SUM(simulado_alto_usd) AS simulado_alto_usd
FROM base
GROUP BY periodo
ORDER BY 1, 2, 3, 4, 5;
