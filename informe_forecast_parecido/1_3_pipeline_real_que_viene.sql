-- 1.3 · la pipeline real de los meses que vienen: raw, la que usa el forecast tras el calendario y la borrada
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year AS anio, period AS periodo,
           COALESCE(s1_1_vencen_usd, 0) AS raw_usd, COALESCE(s1_3_vencen_usd, 0) AS usado_usd,
           COALESCE(s1_1_vencen_unidades, 0) AS raw_unidades, COALESCE(s1_3_vencen_unidades, 0) AS usado_unidades
    FROM sff_nucleo
    WHERE origen_fila = 'raw' AND s1_3_rol = 'proyeccion'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, anio, periodo,
       SUM(raw_usd) AS raw_usd,
       SUM(usado_usd) AS usado_usd,
       SUM(raw_usd) - SUM(usado_usd) AS borrado_usd,
       SUM(raw_unidades) AS raw_unidades,
       SUM(usado_unidades) AS usado_unidades,
       SUM(raw_unidades) - SUM(usado_unidades) AS borrado_unidades
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, anio, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', anio, periodo,
       SUM(raw_usd) AS raw_usd,
       SUM(usado_usd) AS usado_usd,
       SUM(raw_usd) - SUM(usado_usd) AS borrado_usd,
       SUM(raw_unidades) AS raw_unidades,
       SUM(usado_unidades) AS usado_unidades,
       SUM(raw_unidades) - SUM(usado_unidades) AS borrado_unidades
FROM base
GROUP BY tr_regional_level_1, anio, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', anio, periodo,
       SUM(raw_usd) AS raw_usd,
       SUM(usado_usd) AS usado_usd,
       SUM(raw_usd) - SUM(usado_usd) AS borrado_usd,
       SUM(raw_unidades) AS raw_unidades,
       SUM(usado_unidades) AS usado_unidades,
       SUM(raw_unidades) - SUM(usado_unidades) AS borrado_unidades
FROM base
GROUP BY anio, periodo
ORDER BY 1, 2, 3, 4, 5, 6;
