-- 1.5 · la adquisición real por mes de venta (el mes de la fila − 12), en unidades y en el USD que vencerá como pipeline
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year - 1 AS anio_venta, SUBSTRING(period, 6, 2) AS mes,
           COALESCE(s1_1_vencen_unidades, 0) AS vendidas_unidades, COALESCE(s1_1_vencen_usd, 0) AS vendido_usd_como_pipeline
    FROM sff_nucleo
    WHERE origen_fila = 'raw' AND net_new IN ('Acquisition_Not-New', 'Acquisition_Pure-New') AND tr_term = '1 year'
      AND CONCAT(CAST(forecast_year - 1 AS VARCHAR(4)), SUBSTRING(period, 5, 3)) < '2026-10'
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, anio_venta, mes,
       SUM(vendidas_unidades) AS vendidas_unidades,
       SUM(vendido_usd_como_pipeline) AS vendido_usd_como_pipeline
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, anio_venta, mes
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', anio_venta, mes,
       SUM(vendidas_unidades) AS vendidas_unidades,
       SUM(vendido_usd_como_pipeline) AS vendido_usd_como_pipeline
FROM base
GROUP BY tr_regional_level_1, anio_venta, mes
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', anio_venta, mes,
       SUM(vendidas_unidades) AS vendidas_unidades,
       SUM(vendido_usd_como_pipeline) AS vendido_usd_como_pipeline
FROM base
GROUP BY anio_venta, mes
ORDER BY 1, 2, 3, 4, 5, 6;
