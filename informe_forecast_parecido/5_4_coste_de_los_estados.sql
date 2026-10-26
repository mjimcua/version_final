-- 5.4 · lo que cuestan los estados: lo que resta lo que falta por madurar y el potencial (las unidades con estado al vencer renovando como las neutras de su grupo y mes)
WITH neutra AS (
    SELECT tr_regional_level_1, tr_regional_level_2, tr_regional_level_3, tr_product_level_1, tr_product_level_2, tr_renewal_type, tr_term, tr_term_level_1, tr_band, tr_band_level_1, tr_master_partner_code, period,
           SUM(COALESCE(s7_2_esperado_unidades, 0)) / NULLIF(SUM(COALESCE(forecast_to_renew_units_at_due, 0)), 0) AS tasa_neutra
    FROM sff_nucleo
    WHERE forecast_status = 'forecast' AND forecast_universe = 'pipeline' AND COALESCE(dormant, 0) = 0 AND COALESCE(softcancel, 0) = 0 AND COALESCE(not_installed, 0) = 0
    GROUP BY tr_regional_level_1, tr_regional_level_2, tr_regional_level_3, tr_product_level_1, tr_product_level_2, tr_renewal_type, tr_term, tr_term_level_1, tr_band, tr_band_level_1, tr_master_partner_code, period
),
base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, forecast_year AS anio,
           -COALESCE(forecast_renewed_USD_maturation_adj, 0) AS resta_lo_que_falta_por_madurar_usd, 0 AS potencial_usd
    FROM sff_nucleo
    WHERE forecast_status = 'forecast' AND forecast_year IN (2026, 2027)
    UNION ALL
    SELECT s.tr_regional_level_1, s.tr_regional_level_3, s.forecast_year,
           0, COALESCE(s.s7_2_vencen_usd_al_vencer, 0) * COALESCE(s.s3_1_uplift, 1) * (g.tasa_neutra - COALESCE(s.s7_2_tasa, 0))
    FROM sff_nucleo AS s
    JOIN neutra AS g ON g.tr_regional_level_1 = s.tr_regional_level_1 AND g.tr_regional_level_2 = s.tr_regional_level_2 AND g.tr_regional_level_3 = s.tr_regional_level_3 AND g.tr_product_level_1 = s.tr_product_level_1 AND g.tr_product_level_2 = s.tr_product_level_2 AND g.tr_renewal_type = s.tr_renewal_type AND g.tr_term = s.tr_term AND g.tr_term_level_1 = s.tr_term_level_1 AND g.tr_band = s.tr_band AND g.tr_band_level_1 = s.tr_band_level_1 AND g.tr_master_partner_code = s.tr_master_partner_code AND g.period = s.period
    WHERE s.forecast_status = 'forecast' AND s.forecast_universe = 'pipeline' AND (s.dormant = 1 OR s.softcancel = 1 OR s.not_installed = 1)
      AND s.forecast_year IN (2026, 2027)
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, anio,
       SUM(resta_lo_que_falta_por_madurar_usd) AS resta_lo_que_falta_por_madurar_usd,
       SUM(potencial_usd) AS potencial_usd
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, anio
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', anio,
       SUM(resta_lo_que_falta_por_madurar_usd) AS resta_lo_que_falta_por_madurar_usd,
       SUM(potencial_usd) AS potencial_usd
FROM base
GROUP BY tr_regional_level_1, anio
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', anio,
       SUM(resta_lo_que_falta_por_madurar_usd) AS resta_lo_que_falta_por_madurar_usd,
       SUM(potencial_usd) AS potencial_usd
FROM base
GROUP BY anio
ORDER BY 1, 2, 3, 4, 5;
