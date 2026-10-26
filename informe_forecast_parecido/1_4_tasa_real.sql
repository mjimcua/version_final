-- 1.4 · la tasa de renovación real de cada mes cerrado, con su banda binomial
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS periodo,
           COALESCE(s1_3_vencen_unidades, 0) AS vencen_unidades, COALESCE(s1_3_renovadas_unidades, 0) AS renovadas_unidades,
           COALESCE(s1_3_vencen_usd, 0) AS vence_usd, COALESCE(s1_3_renovado_usd, 0) AS renovado_usd
    FROM sff_nucleo
    WHERE origen_fila = 'raw' AND s1_3_rol <> 'proyeccion'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, periodo,
       SUM(vencen_unidades) AS vencen_unidades,
       SUM(renovadas_unidades) AS renovadas_unidades,
       SUM(vence_usd) AS vence_usd,
       SUM(renovado_usd) AS renovado_usd,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) AS tasa_unidades,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) - 1.645 * SQRT(SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) * (1 - SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_baja,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) + 1.645 * SQRT(SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) * (1 - SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_alta,
       SUM(renovado_usd) / NULLIF(SUM(vence_usd), 0) AS tasa_usd
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', periodo,
       SUM(vencen_unidades) AS vencen_unidades,
       SUM(renovadas_unidades) AS renovadas_unidades,
       SUM(vence_usd) AS vence_usd,
       SUM(renovado_usd) AS renovado_usd,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) AS tasa_unidades,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) - 1.645 * SQRT(SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) * (1 - SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_baja,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) + 1.645 * SQRT(SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) * (1 - SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_alta,
       SUM(renovado_usd) / NULLIF(SUM(vence_usd), 0) AS tasa_usd
FROM base
GROUP BY tr_regional_level_1, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', periodo,
       SUM(vencen_unidades) AS vencen_unidades,
       SUM(renovadas_unidades) AS renovadas_unidades,
       SUM(vence_usd) AS vence_usd,
       SUM(renovado_usd) AS renovado_usd,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) AS tasa_unidades,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) - 1.645 * SQRT(SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) * (1 - SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_baja,
       SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) + 1.645 * SQRT(SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0) * (1 - SUM(renovadas_unidades) / NULLIF(SUM(vencen_unidades), 0)) / NULLIF(SUM(vencen_unidades), 0)) AS tasa_binomial_alta,
       SUM(renovado_usd) / NULLIF(SUM(vence_usd), 0) AS tasa_usd
FROM base
GROUP BY periodo
ORDER BY 1, 2, 3, 4, 5;
