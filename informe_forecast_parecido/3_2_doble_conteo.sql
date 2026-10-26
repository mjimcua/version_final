-- 3.2 · 2027: lo que el calendario borró frente a lo que lo sustituye (sin doble conteo)
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS periodo,
           CASE WHEN origen_fila = 'raw' THEN COALESCE(s1_1_vencen_usd, 0) - COALESCE(s1_3_vencen_usd, 0) ELSE 0 END AS borrado_usd,
           CASE WHEN origen_fila = 'proyectada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS renovacion_proyectada_usd,
           CASE WHEN origen_fila = 'simulada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS adquisicion_simulada_usd
    FROM sff_nucleo
    WHERE forecast_year = 2027 AND origen_fila IN ('raw', 'proyectada', 'simulada')
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, periodo,
       SUM(borrado_usd) AS borrado_usd,
       SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd,
       SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd,
       SUM(renovacion_proyectada_usd) + SUM(adquisicion_simulada_usd) - SUM(borrado_usd) AS sustituto_menos_borrado_usd
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', periodo,
       SUM(borrado_usd) AS borrado_usd,
       SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd,
       SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd,
       SUM(renovacion_proyectada_usd) + SUM(adquisicion_simulada_usd) - SUM(borrado_usd) AS sustituto_menos_borrado_usd
FROM base
GROUP BY tr_regional_level_1, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', periodo,
       SUM(borrado_usd) AS borrado_usd,
       SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd,
       SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd,
       SUM(renovacion_proyectada_usd) + SUM(adquisicion_simulada_usd) - SUM(borrado_usd) AS sustituto_menos_borrado_usd
FROM base
GROUP BY periodo
ORDER BY 1, 2, 3, 4, 5;
