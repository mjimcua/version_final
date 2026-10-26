-- 1.2 · la pipeline real mes a mes (como viene en el extracto)
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year AS anio,
           SUBSTRING(period, 6, 2) AS mes, period AS periodo,
           COALESCE(s1_1_vencen_usd, 0) AS vence_usd, COALESCE(s1_1_vencen_unidades, 0) AS vence_unidades
    FROM sff_nucleo
    WHERE origen_fila = 'raw'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, anio, mes, periodo,
       SUM(vence_usd) AS vence_usd,
       SUM(vence_unidades) AS vence_unidades
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, anio, mes, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', anio, mes, periodo,
       SUM(vence_usd) AS vence_usd,
       SUM(vence_unidades) AS vence_unidades
FROM base
GROUP BY tr_regional_level_1, anio, mes, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', anio, mes, periodo,
       SUM(vence_usd) AS vence_usd,
       SUM(vence_unidades) AS vence_unidades
FROM base
GROUP BY anio, mes, periodo
ORDER BY 1, 2, 3, 4, 5, 6, 7;
