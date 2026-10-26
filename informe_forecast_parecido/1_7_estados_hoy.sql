-- 1.7 · los estados del cliente HOY en la pipeline real que queda por vencer
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS periodo,
           COALESCE(s1_3_vencen_unidades, 0) AS vencen_unidades,
           CASE WHEN dormant = 1 THEN COALESCE(s1_3_vencen_unidades, 0) ELSE 0 END AS dormant_unidades,
           CASE WHEN softcancel = 1 THEN COALESCE(s1_3_vencen_unidades, 0) ELSE 0 END AS softcancel_unidades,
           CASE WHEN not_installed = 1 THEN COALESCE(s1_3_vencen_unidades, 0) ELSE 0 END AS not_installed_unidades
    FROM sff_nucleo
    WHERE origen_fila = 'raw' AND s1_3_rol = 'proyeccion' AND s1_3_vencen_unidades > 0
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, periodo,
       SUM(vencen_unidades) AS vencen_unidades,
       SUM(dormant_unidades) AS dormant_unidades,
       SUM(dormant_unidades) / NULLIF(SUM(vencen_unidades), 0) AS dormant_parte,
       SUM(softcancel_unidades) AS softcancel_unidades,
       SUM(softcancel_unidades) / NULLIF(SUM(vencen_unidades), 0) AS softcancel_parte,
       SUM(not_installed_unidades) AS not_installed_unidades,
       SUM(not_installed_unidades) / NULLIF(SUM(vencen_unidades), 0) AS not_installed_parte
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', periodo,
       SUM(vencen_unidades) AS vencen_unidades,
       SUM(dormant_unidades) AS dormant_unidades,
       SUM(dormant_unidades) / NULLIF(SUM(vencen_unidades), 0) AS dormant_parte,
       SUM(softcancel_unidades) AS softcancel_unidades,
       SUM(softcancel_unidades) / NULLIF(SUM(vencen_unidades), 0) AS softcancel_parte,
       SUM(not_installed_unidades) AS not_installed_unidades,
       SUM(not_installed_unidades) / NULLIF(SUM(vencen_unidades), 0) AS not_installed_parte
FROM base
GROUP BY tr_regional_level_1, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', periodo,
       SUM(vencen_unidades) AS vencen_unidades,
       SUM(dormant_unidades) AS dormant_unidades,
       SUM(dormant_unidades) / NULLIF(SUM(vencen_unidades), 0) AS dormant_parte,
       SUM(softcancel_unidades) AS softcancel_unidades,
       SUM(softcancel_unidades) / NULLIF(SUM(vencen_unidades), 0) AS softcancel_parte,
       SUM(not_installed_unidades) AS not_installed_unidades,
       SUM(not_installed_unidades) / NULLIF(SUM(vencen_unidades), 0) AS not_installed_parte
FROM base
GROUP BY periodo
ORDER BY 1, 2, 3, 4, 5;
