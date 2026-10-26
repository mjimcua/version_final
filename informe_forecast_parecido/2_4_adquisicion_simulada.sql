-- 2.4 · la adquisición por mes de venta: real (el año anterior y este hasta hoy) y simulada
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year - 1 AS anio_venta, SUBSTRING(period, 6, 2) AS mes,
           COALESCE(s1_1_vencen_unidades, 0) AS real_unidades, COALESCE(s1_1_vencen_usd, 0) AS real_usd,
           0 AS simulada_unidades, 0 AS simulada_usd
    FROM sff_nucleo
    WHERE origen_fila = 'raw' AND net_new IN ('Acquisition_Not-New', 'Acquisition_Pure-New') AND tr_term = '1 year'
      AND CONCAT(CAST(forecast_year - 1 AS VARCHAR(4)), SUBSTRING(period, 5, 3)) < '2026-10' AND forecast_year - 1 >= 2025
    UNION ALL
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year - 1, SUBSTRING(period, 6, 2),
           0, 0, COALESCE(forecast_to_renew_units, 0), COALESCE(forecast_to_renew_USD, 0)
    FROM sff_nucleo
    WHERE origen_fila = 'simulada'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, anio_venta, mes,
       SUM(real_unidades) AS real_unidades,
       SUM(simulada_unidades) AS simulada_unidades,
       SUM(real_usd) AS real_usd_como_pipeline,
       SUM(simulada_usd) AS simulada_usd_como_pipeline
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, anio_venta, mes
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', anio_venta, mes,
       SUM(real_unidades) AS real_unidades,
       SUM(simulada_unidades) AS simulada_unidades,
       SUM(real_usd) AS real_usd_como_pipeline,
       SUM(simulada_usd) AS simulada_usd_como_pipeline
FROM base
GROUP BY tr_regional_level_1, anio_venta, mes
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', anio_venta, mes,
       SUM(real_unidades) AS real_unidades,
       SUM(simulada_unidades) AS simulada_unidades,
       SUM(real_usd) AS real_usd_como_pipeline,
       SUM(simulada_usd) AS simulada_usd_como_pipeline
FROM base
GROUP BY anio_venta, mes
ORDER BY 1, 2, 3, 4, 5, 6;
