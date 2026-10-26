-- 3.1 · la pipeline de 2027 por mes y por sus cuatro orígenes
WITH base AS (
    SELECT tr_regional_level_1, tr_regional_level_3, period AS periodo,
           CASE WHEN forecast_pipeline_source = 'pipeline_real_esperado' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS real_usd,
           CASE WHEN forecast_pipeline_source = 'pipeline_real_esperado' THEN COALESCE(forecast_to_renew_units, 0) ELSE 0 END AS real_unidades,
           CASE WHEN forecast_pipeline_source = 'pipeline_proyectada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS renovacion_proyectada_usd,
           CASE WHEN forecast_pipeline_source = 'pipeline_proyectada' THEN COALESCE(forecast_to_renew_units, 0) ELSE 0 END AS renovacion_proyectada_unidades,
           CASE WHEN forecast_pipeline_source = 'ts_reentrada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS retail_to_subscription_usd,
           CASE WHEN forecast_pipeline_source = 'ts_reentrada' THEN COALESCE(forecast_to_renew_units, 0) ELSE 0 END AS retail_to_subscription_unidades,
           CASE WHEN forecast_pipeline_source = 'pipeline_simulada' THEN COALESCE(forecast_to_renew_USD, 0) ELSE 0 END AS adquisicion_simulada_usd,
           CASE WHEN forecast_pipeline_source = 'pipeline_simulada' THEN COALESCE(forecast_to_renew_units, 0) ELSE 0 END AS adquisicion_simulada_unidades
    FROM sff_nucleo
    WHERE forecast_year = 2027 AND forecast_pipeline_source IN ('pipeline_real_esperado', 'pipeline_proyectada', 'ts_reentrada', 'pipeline_simulada')
)
SELECT 0 AS orden_total, tr_regional_level_1, 0 AS orden_region, tr_regional_level_3, periodo,
       SUM(real_usd) AS real_usd,
       SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd,
       SUM(retail_to_subscription_usd) AS retail_to_subscription_usd,
       SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd,
       SUM(real_usd) + SUM(renovacion_proyectada_usd) + SUM(retail_to_subscription_usd) + SUM(adquisicion_simulada_usd) AS total_usd,
       SUM(real_unidades) AS real_unidades,
       SUM(renovacion_proyectada_unidades) AS renovacion_proyectada_unidades,
       SUM(retail_to_subscription_unidades) AS retail_to_subscription_unidades,
       SUM(adquisicion_simulada_unidades) AS adquisicion_simulada_unidades,
       SUM(real_unidades) + SUM(renovacion_proyectada_unidades) + SUM(retail_to_subscription_unidades) + SUM(adquisicion_simulada_unidades) AS total_unidades
FROM base
GROUP BY tr_regional_level_1, tr_regional_level_3, periodo
UNION ALL
SELECT 0, tr_regional_level_1, 1, 'TOTAL', periodo,
       SUM(real_usd) AS real_usd,
       SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd,
       SUM(retail_to_subscription_usd) AS retail_to_subscription_usd,
       SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd,
       SUM(real_usd) + SUM(renovacion_proyectada_usd) + SUM(retail_to_subscription_usd) + SUM(adquisicion_simulada_usd) AS total_usd,
       SUM(real_unidades) AS real_unidades,
       SUM(renovacion_proyectada_unidades) AS renovacion_proyectada_unidades,
       SUM(retail_to_subscription_unidades) AS retail_to_subscription_unidades,
       SUM(adquisicion_simulada_unidades) AS adquisicion_simulada_unidades,
       SUM(real_unidades) + SUM(renovacion_proyectada_unidades) + SUM(retail_to_subscription_unidades) + SUM(adquisicion_simulada_unidades) AS total_unidades
FROM base
GROUP BY tr_regional_level_1, periodo
UNION ALL
SELECT 1, 'TOTAL', 2, 'TOTAL', periodo,
       SUM(real_usd) AS real_usd,
       SUM(renovacion_proyectada_usd) AS renovacion_proyectada_usd,
       SUM(retail_to_subscription_usd) AS retail_to_subscription_usd,
       SUM(adquisicion_simulada_usd) AS adquisicion_simulada_usd,
       SUM(real_usd) + SUM(renovacion_proyectada_usd) + SUM(retail_to_subscription_usd) + SUM(adquisicion_simulada_usd) AS total_usd,
       SUM(real_unidades) AS real_unidades,
       SUM(renovacion_proyectada_unidades) AS renovacion_proyectada_unidades,
       SUM(retail_to_subscription_unidades) AS retail_to_subscription_unidades,
       SUM(adquisicion_simulada_unidades) AS adquisicion_simulada_unidades,
       SUM(real_unidades) + SUM(renovacion_proyectada_unidades) + SUM(retail_to_subscription_unidades) + SUM(adquisicion_simulada_unidades) AS total_unidades
FROM base
GROUP BY periodo
ORDER BY 1, 2, 3, 4, 5;
