-- 5.3 · ¿ha sido honesta la banda? La cobertura del examen y el WAPE, sumados por región
WITH base AS (
    SELECT m.tr_regional_level_1, m.tr_regional_level_3,
           COALESCE(f.s7_1_exam_predictions, 0) AS predicciones, COALESCE(f.s7_1_exam_in_band, 0) AS dentro_de_la_banda,
           COALESCE(f.s7_1_exam_abs_err_units, 0) AS error_absoluto_unidades, COALESCE(f.s7_1_exam_real_units, 0) AS reales_unidades,
           COALESCE(f.s7_1_raw_predictions, 0) AS predicciones_serie_sola, COALESCE(f.s7_1_raw_in_band, 0) AS dentro_serie_sola
    FROM (SELECT DISTINCT s1_4_fs_id, tr_regional_level_1, tr_regional_level_3 FROM sff_nucleo WHERE origen_fila = 'raw' AND s1_4_fs_id IS NOT NULL) AS m
    JOIN sff_forecast_series AS f ON f.s1_4_fs_id = m.s1_4_fs_id
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3,
       SUM(predicciones) AS predicciones,
       SUM(dentro_de_la_banda) AS dentro_de_la_banda,
       SUM(dentro_de_la_banda) / NULLIF(SUM(predicciones), 0) AS cobertura,
       SUM(dentro_serie_sola) / NULLIF(SUM(predicciones_serie_sola), 0) AS cobertura_serie_sola,
       SUM(error_absoluto_unidades) / NULLIF(SUM(reales_unidades), 0) AS wape
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL',
       SUM(predicciones) AS predicciones,
       SUM(dentro_de_la_banda) AS dentro_de_la_banda,
       SUM(dentro_de_la_banda) / NULLIF(SUM(predicciones), 0) AS cobertura,
       SUM(dentro_serie_sola) / NULLIF(SUM(predicciones_serie_sola), 0) AS cobertura_serie_sola,
       SUM(error_absoluto_unidades) / NULLIF(SUM(reales_unidades), 0) AS wape
FROM base
GROUP BY tr_regional_level_1
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL',
       SUM(predicciones) AS predicciones,
       SUM(dentro_de_la_banda) AS dentro_de_la_banda,
       SUM(dentro_de_la_banda) / NULLIF(SUM(predicciones), 0) AS cobertura,
       SUM(dentro_serie_sola) / NULLIF(SUM(predicciones_serie_sola), 0) AS cobertura_serie_sola,
       SUM(error_absoluto_unidades) / NULLIF(SUM(reales_unidades), 0) AS wape
FROM base
ORDER BY 1, 2, 3, 4;
