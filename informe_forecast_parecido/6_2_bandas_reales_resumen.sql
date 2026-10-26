-- 6.2 · el resumen de las bandas reales: cobertura y cociente error/banda por nivel, base, h y regla (1 = honesta; la cobertura prometida es la de z = 1.645)
WITH mapa AS (
    SELECT DISTINCT s1_4_fs_id, tr_regional_level_1, tr_regional_level_3 FROM sff_nucleo
    WHERE origen_fila = 'raw' AND s1_4_fs_id IS NOT NULL
),
serie AS (
    SELECT d.fs_id, d.composition_id, m.tr_regional_level_1 AS region,
           CONCAT(m.tr_regional_level_1, ' · ', m.tr_regional_level_3) AS subregion_completa, d.period AS periodo, d.h,
           COALESCE(d.pred_units, 0) AS predichas, COALESCE(d.real_units, 0) AS reales,
           COALESCE(d.pred_usd, 0) AS predicho_usd, COALESCE(d.real_usd, 0) AS real_usd,
           COALESCE(d.due_units, 0) * (COALESCE(d.pred_rate, 0) - COALESCE(d.band_low, d.pred_rate, 0)) AS desviacion_baja,
           COALESCE(d.due_units, 0) * (COALESCE(d.band_high, d.pred_rate, 0) - COALESCE(d.pred_rate, 0)) AS desviacion_alta,
           COALESCE(d.due_units, 0) * (COALESCE(d.pred_rate, 0) - COALESCE(d.band_low, d.pred_rate, 0))
               * COALESCE(d.due_units, 0) * (COALESCE(d.pred_rate, 0) - COALESCE(d.band_low, d.pred_rate, 0)) AS cuadrado_baja,
           COALESCE(d.due_units, 0) * (COALESCE(d.band_high, d.pred_rate, 0) - COALESCE(d.pred_rate, 0))
               * COALESCE(d.due_units, 0) * (COALESCE(d.band_high, d.pred_rate, 0) - COALESCE(d.pred_rate, 0)) AS cuadrado_alta,
           COALESCE(d.due_units, 0) * COALESCE(d.pred_rate, 0) * (1 - COALESCE(d.pred_rate, 0)) AS varianza_binomial
    FROM sff_series_exam_detail AS d
    JOIN mapa AS m ON m.s1_4_fs_id = d.fs_id
    WHERE d.method = 'framework'
),
serie_acumulada AS (
    -- the exam months added up per series: a series is wrong in the same direction every month (linear over the
    -- months), its binomial noise is independent month to month (its variance adds up)
    SELECT fs_id, composition_id, region, subregion_completa, h,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
           SUM(desviacion_baja) * SUM(desviacion_baja) AS cuadrado_baja, SUM(desviacion_alta) * SUM(desviacion_alta) AS cuadrado_alta,
           SUM(varianza_binomial) AS varianza_binomial
    FROM serie
    GROUP BY fs_id, composition_id, region, subregion_completa, h
),
niveles AS (
    SELECT 'serie' AS nivel, grupo AS grupo, periodo AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT fs_id AS grupo, composition_id, periodo, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie
          GROUP BY fs_id, composition_id, periodo, h) AS por_composicion
    GROUP BY grupo, periodo, h
    UNION ALL
    SELECT 'composicion' AS nivel, grupo AS grupo, periodo AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT composition_id AS grupo, periodo, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie
          GROUP BY composition_id, periodo, h) AS por_composicion
    GROUP BY grupo, periodo, h
    UNION ALL
    SELECT 'subregion' AS nivel, grupo AS grupo, periodo AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT subregion_completa AS grupo, composition_id, periodo, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie
          GROUP BY subregion_completa, composition_id, periodo, h) AS por_composicion
    GROUP BY grupo, periodo, h
    UNION ALL
    SELECT 'region' AS nivel, grupo AS grupo, periodo AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT region AS grupo, composition_id, periodo, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie
          GROUP BY region, composition_id, periodo, h) AS por_composicion
    GROUP BY grupo, periodo, h
    UNION ALL
    SELECT 'total' AS nivel, 'TOTAL' AS grupo, periodo AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT composition_id, periodo, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie
          GROUP BY composition_id, periodo, h) AS por_composicion
    GROUP BY periodo, h
    UNION ALL
    SELECT 'serie' AS nivel, grupo AS grupo, 'acumulado' AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT fs_id AS grupo, composition_id, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie_acumulada
          GROUP BY fs_id, composition_id, h) AS por_composicion
    GROUP BY grupo, h
    UNION ALL
    SELECT 'composicion' AS nivel, grupo AS grupo, 'acumulado' AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT composition_id AS grupo, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie_acumulada
          GROUP BY composition_id, h) AS por_composicion
    GROUP BY grupo, h
    UNION ALL
    SELECT 'subregion' AS nivel, grupo AS grupo, 'acumulado' AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT subregion_completa AS grupo, composition_id, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie_acumulada
          GROUP BY subregion_completa, composition_id, h) AS por_composicion
    GROUP BY grupo, h
    UNION ALL
    SELECT 'region' AS nivel, grupo AS grupo, 'acumulado' AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT region AS grupo, composition_id, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie_acumulada
          GROUP BY region, composition_id, h) AS por_composicion
    GROUP BY grupo, h
    UNION ALL
    SELECT 'total' AS nivel, 'TOTAL' AS grupo, 'acumulado' AS periodo, h,
           SUM(series) AS series, COUNT(*) AS composiciones,
           SUM(predichas) AS predichas, SUM(reales) AS reales, SUM(reales) - SUM(predichas) AS error,
           SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
           SUM(desviacion_baja) AS lineal_baja, SUM(desviacion_alta) AS lineal_alta,
           SQRT(SUM(desviacion_baja * desviacion_baja)) AS composicion_baja, SQRT(SUM(desviacion_alta * desviacion_alta)) AS composicion_alta,
           SQRT(SUM(cuadrado_baja)) AS cuadratura_baja, SQRT(SUM(cuadrado_alta)) AS cuadratura_alta,
           1.645 * SQRT(SUM(varianza_binomial)) AS binomial
    FROM (SELECT composition_id, h,
                 COUNT(*) AS series, SUM(predichas) AS predichas, SUM(reales) AS reales,
                 SUM(predicho_usd) AS predicho_usd, SUM(real_usd) AS real_usd,
                 SUM(desviacion_baja) AS desviacion_baja, SUM(desviacion_alta) AS desviacion_alta,
                 SUM(cuadrado_baja) AS cuadrado_baja, SUM(cuadrado_alta) AS cuadrado_alta,
                 SUM(varianza_binomial) AS varianza_binomial
          FROM serie_acumulada
          GROUP BY composition_id, h) AS por_composicion
    GROUP BY h
),
bandas AS (
    SELECT niveles.*,
           CASE WHEN error < 0 THEN lineal_baja ELSE lineal_alta END AS semibanda_lineal,
           CASE WHEN error < 0 THEN composicion_baja ELSE composicion_alta END AS semibanda_composicion,
           CASE WHEN error < 0 THEN cuadratura_baja ELSE cuadratura_alta END AS semibanda_cuadratura,
           binomial AS semibanda_binomial
    FROM niveles
),
resumen AS (
    SELECT nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END AS base, h, 'lineal' AS regla,
           COUNT(*) AS observaciones,
           AVG(CASE WHEN ABS(error) <= semibanda_lineal THEN 1.0 ELSE 0.0 END) AS cobertura,
           SQRT(AVG(error * error)) / NULLIF(SQRT(AVG(semibanda_lineal * semibanda_lineal)) / 1.645, 0) AS cociente_error_banda,
           AVG(ABS(error)) AS error_medio_absoluto, AVG(semibanda_lineal) AS semibanda_media,
           AVG(error) AS error_medio, AVG(error) / NULLIF(SQRT(AVG(error * error)), 0) AS parte_sesgo
    FROM bandas
    GROUP BY nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END, h
    UNION ALL
    SELECT nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END AS base, h, 'composicion' AS regla,
           COUNT(*) AS observaciones,
           AVG(CASE WHEN ABS(error) <= semibanda_composicion THEN 1.0 ELSE 0.0 END) AS cobertura,
           SQRT(AVG(error * error)) / NULLIF(SQRT(AVG(semibanda_composicion * semibanda_composicion)) / 1.645, 0) AS cociente_error_banda,
           AVG(ABS(error)) AS error_medio_absoluto, AVG(semibanda_composicion) AS semibanda_media,
           AVG(error) AS error_medio, AVG(error) / NULLIF(SQRT(AVG(error * error)), 0) AS parte_sesgo
    FROM bandas
    GROUP BY nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END, h
    UNION ALL
    SELECT nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END AS base, h, 'cuadratura' AS regla,
           COUNT(*) AS observaciones,
           AVG(CASE WHEN ABS(error) <= semibanda_cuadratura THEN 1.0 ELSE 0.0 END) AS cobertura,
           SQRT(AVG(error * error)) / NULLIF(SQRT(AVG(semibanda_cuadratura * semibanda_cuadratura)) / 1.645, 0) AS cociente_error_banda,
           AVG(ABS(error)) AS error_medio_absoluto, AVG(semibanda_cuadratura) AS semibanda_media,
           AVG(error) AS error_medio, AVG(error) / NULLIF(SQRT(AVG(error * error)), 0) AS parte_sesgo
    FROM bandas
    GROUP BY nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END, h
    UNION ALL
    SELECT nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END AS base, h, 'binomial' AS regla,
           COUNT(*) AS observaciones,
           AVG(CASE WHEN ABS(error) <= semibanda_binomial THEN 1.0 ELSE 0.0 END) AS cobertura,
           SQRT(AVG(error * error)) / NULLIF(SQRT(AVG(semibanda_binomial * semibanda_binomial)) / 1.645, 0) AS cociente_error_banda,
           AVG(ABS(error)) AS error_medio_absoluto, AVG(semibanda_binomial) AS semibanda_media,
           AVG(error) AS error_medio, AVG(error) / NULLIF(SQRT(AVG(error * error)), 0) AS parte_sesgo
    FROM bandas
    GROUP BY nivel, CASE WHEN periodo = 'acumulado' THEN 'acumulado' ELSE 'mes' END, h
)
SELECT * FROM resumen
ORDER BY base, h, regla, nivel;
