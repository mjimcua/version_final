-- 1.1 · la pipeline real que vence cada año (como viene en el extracto) y lo renovado real
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year AS anio,
           CASE WHEN s1_3_rol <> 'proyeccion' THEN COALESCE(s1_1_vencen_usd, 0) ELSE 0 END AS vence_usd_meses_cerrados,
           CASE WHEN s1_3_rol <> 'proyeccion' THEN 0 ELSE COALESCE(s1_1_vencen_usd, 0) END AS vence_usd_meses_que_quedan,
           COALESCE(s1_1_vencen_unidades, 0) AS vence_unidades,
           CASE WHEN s1_3_rol <> 'proyeccion' THEN COALESCE(s1_3_vencen_unidades, 0) ELSE 0 END AS vence_unidades_meses_cerrados,
           CASE WHEN s1_3_rol <> 'proyeccion' THEN COALESCE(s1_3_renovadas_unidades, 0) ELSE 0 END AS renovadas_unidades_real,
           CASE WHEN s1_3_rol <> 'proyeccion' THEN COALESCE(s1_3_renovado_usd, 0) ELSE 0 END AS renovado_usd_real
    FROM sff_nucleo
    WHERE origen_fila = 'raw'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, anio,
       SUM(vence_usd_meses_cerrados) + SUM(vence_usd_meses_que_quedan) AS vence_usd,
       SUM(vence_usd_meses_cerrados) AS vence_usd_meses_cerrados,
       SUM(vence_usd_meses_que_quedan) AS vence_usd_meses_que_quedan,
       SUM(vence_unidades) AS vence_unidades,
       SUM(renovado_usd_real) AS renovado_usd_real,
       SUM(renovadas_unidades_real) AS renovadas_unidades_real,
       SUM(renovadas_unidades_real) / NULLIF(SUM(vence_unidades_meses_cerrados), 0) AS tasa_unidades_real
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, anio
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', anio,
       SUM(vence_usd_meses_cerrados) + SUM(vence_usd_meses_que_quedan) AS vence_usd,
       SUM(vence_usd_meses_cerrados) AS vence_usd_meses_cerrados,
       SUM(vence_usd_meses_que_quedan) AS vence_usd_meses_que_quedan,
       SUM(vence_unidades) AS vence_unidades,
       SUM(renovado_usd_real) AS renovado_usd_real,
       SUM(renovadas_unidades_real) AS renovadas_unidades_real,
       SUM(renovadas_unidades_real) / NULLIF(SUM(vence_unidades_meses_cerrados), 0) AS tasa_unidades_real
FROM base
GROUP BY tr_regional_level_1, anio
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', anio,
       SUM(vence_usd_meses_cerrados) + SUM(vence_usd_meses_que_quedan) AS vence_usd,
       SUM(vence_usd_meses_cerrados) AS vence_usd_meses_cerrados,
       SUM(vence_usd_meses_que_quedan) AS vence_usd_meses_que_quedan,
       SUM(vence_unidades) AS vence_unidades,
       SUM(renovado_usd_real) AS renovado_usd_real,
       SUM(renovadas_unidades_real) AS renovadas_unidades_real,
       SUM(renovadas_unidades_real) / NULLIF(SUM(vence_unidades_meses_cerrados), 0) AS tasa_unidades_real
FROM base
GROUP BY anio
ORDER BY 1, 2, 3, 4, 5;
