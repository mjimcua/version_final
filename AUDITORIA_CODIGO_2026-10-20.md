# Auditoría del código · 20-oct-2026

La auditoría se repartió en tres bloques de pasos (00–09 y Config, 10–14 y 19, 15–22 y núcleo), con revisión de solo lectura y cada hallazgo confirmado con el código o con un script. Abajo, primero lo que cambiaba números, que ya está arreglado; después lo que no se arregla y por qué.

## Arreglado: cambiaba números

| # | Dónde | Qué pasaba | Arreglo |
|---|---|---|---|
| 1 | Paso 14, elección de técnica | Cada técnica se puntuaba sobre sus propios meses objetivo y la técnica por defecto sobre todos los suyos. Una técnica que necesita más historia predice menos meses y más tardíos, y podía ganar sin ser mejor en los meses comunes: en una prueba, T15 salía campeona con 1,05 frente a 2,0, siendo peor en los 3 objetivos comunes. | Se compara cada técnica con la técnica por defecto solo en los objetivos (mes, horizonte) que predicen las dos, con un mínimo de `backtest_min_predictions`. Gana la mejora mayor. |
| 2 | Paso 11, k de credibilidad | El estimador de Bühlmann-Straub restaba r·σ² en vez de (r−1)·σ² y dividía por W en vez de W − Σw²/W. Así infravaloraba la varianza entre composiciones: más grupos salían "homogéneos" (k = 10 × k_cred), z quedaba demasiado bajo y la referencia pesaba de más. | Estimador insesgado: a = (Σw(p−p̄)² − (r−1)σ²) / (W − Σw²/W), con σ² la media ponderada de p(1−p). |
| 3 | Paso 17, maduración | Lo visible hoy salía del flag, que también incluye lo marcado en el mes en curso (k = h−1). La curva vuelve a sumar ese tramo como "por venir", así que la marca salía inflada y el forecast más bajo. El examen, en cambio, usaba V_h: forecast y examen no medían lo mismo. | El forecast parte de V_h (columnas k ≥ h), igual que la curva y su examen. |
| 4 | Paso 01, niveles | El JSON de niveles se reutilizaba aunque se hubiera entrenado con un mes en curso posterior: al ejecutar un mes anterior, sus grupos veían su examen y su futuro. Tampoco se regeneraba al cambiar `support_floor`. | Se regenera si el fichero es de un mes en curso posterior o si cambia el suelo de soporte. |
| 5 | Paso 20, time series | Una región sin ventas en la ventana del nivel quedaba con nivel "desconocido" y se proyectaba por cuota (171 unidades en la prueba). | Sin filas recientes, el nivel es 0. |
| 6 | Paso 19, serie sola (raw) | La media de logits de tasas mensuales de 0 % o 100 % (2 de 2) llevaba la serie sola muy lejos: con p = 0,7 y 2 unidades al mes predecía 0,81. La mejora del framework sobre la serie sola salía inflada. | La tasa mensual entra con corrección de continuidad: (renovadas + ½)/(vencen + 1). |
| 7 | Paso 13, cartera | El perfil de la cartera sumaba las composiciones y contaba dos veces a las series que prestan su historia. | La cartera sale de cada serie una sola vez. |
| 8 | Paso 08, huecos | Un hueco en un mes de examen heredaba el rol del mes real anterior (entrenamiento). | El rol es el del calendario. |
| 9 | Examen honesto (maturation.py) | La curva examinada se aprendía con toda la historia hasta el origen, y el forecast con los últimos 12 meses: se examinaba una curva distinta de la que se usa. | El examen usa la misma ventana (`maturation_history_months`). |
| 10 | Núcleo | Las filas proyectadas y simuladas perdían las dimensiones que empiezan por "s" (`softcancel`, `sku`): unos $5,4M esperados sin softcancel en el sintético. | Solo se excluyen los bloques de paso (`s00_`, `s17_`…). |
| 11 | AUD, técnicas | Una técnica salía "tested" con meses que solo estaban en el examen. | "Tested" solo si predijo en los meses de selección de ese tramo. |

## Arreglado: mensajes y textos

- Se quitaron las cabeceras desfasadas tras quitar los pasos 05 y 07 (núcleo, paso 22, paso 15, paso 02, pipeline), las menciones a claves hash, la "serie_temporal" y el "sin pase".
- El examen honesto avisa de qué marcas no traen fecha: para ellas sigue siendo el techo.
- La leyenda del núcleo dice que el ajuste de la maduración ya está dentro del forecast, y que `s11_tasa_estimada` es un diagnóstico.
- El diagnóstico de grano ya no presenta las columnas `<marca>_month_k` como dimensiones.
- Hay tablas de versiones anteriores que esta ya no escribe. Al arrancar, el run avisa si siguen en la base de datos e imprime cómo borrarlas; no las borra.

## No arreglado: decidido

- **El examen no es del todo fuera de muestra.** La escalera, k, los gates, la técnica elegida y las bandas se estiman con todos los meses cerrados, incluidos los de examen; a 6 meses la elección ve meses posteriores al origen. Re-estimarlo todo por origen multiplica el tiempo. Queda declarado en el paso 19, en el informe y en el resumen para negocio: el framework sale algo optimista y la baseline no tiene esa ventaja.
- **Las unidades que migran en el reparto de marcas** renuevan a la tasa USD histórica de su combinación, sin deflactar subidas. El efecto es de segundo orden: la subida por la parte que migra.
- **Comprobaciones que no pueden fallar por construcción** (02 c1, 04 c1, 06 c1, 08 c1 y c4, 09 c3, 10 c1 y c4, 14 c1, 19 c1–4). Se mantienen como autocomprobación.
- **Un total nulo con partes no nulas en el paso 01.** El nulo ya se avisa como "renovación nula"; compararlo como 0 duplicaría el aviso.

## Cambios de concepto en la misma entrega

- **"Retador" pasa a ser "técnica por defecto"** en código, columnas, pantalla, informe, explorador y documentación (`default_technique`, `default_margin_by_band`, `por_defecto_*`).
- **La baseline es la referencia.** Celda mandatory con los últimos `baseline_months` = 3 meses. El examen del paso 19 tiene ahora la cascada baseline → serie sola → escalera con la técnica por defecto → framework, también en USD. Cada método lleva su ganancia frente a la baseline en USD al mes y su consistencia (en cuántos meses yerra menos).
- **El resumen para negocio** (`resumen_negocio.py`) sale al final del run y al principio del informe.
- **Los gráficos** (`graficos.py`, en `salida/graficos/`) van enlazados en el informe.
