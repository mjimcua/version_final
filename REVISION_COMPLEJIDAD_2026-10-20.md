# Revisión de complejidad · 20-oct-2026

Pregunta: qué partes son demasiado complejas para lo que dan, han quedado obsoletas o se muestran por pantalla sin decidir nada. El inventario ha sido paso a paso: qué produce cada uno, quién lo lee (otro paso, el núcleo, Power BI, el informe o nadie) y qué sale solo por pantalla. Las series temporales (paso 20) quedan fuera por decisión tuya.

## Ejecutado (los candidatos claros)

| Qué | Por qué | Ganancia |
|---|---|---|
| El paso 03 ya no escribe `sff_fact_fine` | Nadie la lee: `sff_nucleo` ya lleva cada fila fina con sus ids, y no está en el modelo de datos | **~3,5 min** de los 45 del run real (216 s) |
| El diagnóstico de marcas del paso 17 se reduce a una tabla | Sus tablas A (marcas hoy frente a final) y C (tamaño del efecto) repiten el REPARTO DE MARCAS. Se queda solo la B: lo que renueva cada combinación (neutral 71 %, dormant 45 %, softcancel 28 %), que no sale en ningún otro sitio | −90 líneas, una sección menos |
| Regla del contrato: de tres sitios a una tabla | `contract_check` lleva ahora la realización por población: neutras, con marca e isolated. Desaparecen la tabla A duplicada y la C (con/sin subida), que confundía la mezcla de tramos; esa pregunta se mide ahora en el examen (paso 16). La regla 1/(1−d) se calcula en una sola función (`contract_rule`), no en cinco | Una tabla en vez de tres; una fórmula |
| Restos muertos | Campo `contract_tolerance` (no lo leía nadie) y su columna `pct_usd_dentro_2pct`; método `Config.read_time_series` (no se llamaba); bloque de constantes duplicado en `vocabulario.py` (los `ORIGIN_*` sin uso); una línea que no hacía nada en el paso 11; `credibility_shift_logit`, que AUD calculaba y luego tiraba; 13 imports sin uso | Código más corto y sin pistas falsas |
| Incoherencia arreglada | `uplift_mandatory_dims=[]` daba celdas distintas en la Config (`is not None`) y en los pasos 15/17 (`or`). Ahora `None` y `[]` significan lo mismo en todas partes: todas las mandatory | Un bug latente menos |
| Texto obsoleto del informe | El capítulo 8 daba como pendientes el horizonte extendido y la baseline de la hoja, que ya están. Ahora lista los pendientes reales | — |

## Propuestos, por ganancia — EJECUTADOS el 20-oct (noche) a petición de Miguel Angel: los puntos 1 a 5, más los pares del 09 y la ficha por serie. Detalle en DOC_modelo_datos.md, "Lo que NO se escribe"

1. **AUD duplica el examen del paso 19.** `series_backtest` y `series_technique_summary` recalculan la predicción del framework en los meses de examen. El check 10 de AUD solo comprueba que coincide con la del 19, y fuera de los tests nadie lee esas dos tablas. AUD es el **26 % del run** (701 s). Propuesta: que AUD lea el detalle del 19 en vez de recalcularlo. Es la mayor ganancia de tiempo que queda.
2. **El paso 07 (cota de soporte) entero.** Su única salida es la columna `s07_moe_pp_max` del núcleo, que no lee nadie (ni el explorer, ni Power BI, ni el informe). Su propio docstring dice "Nothing later computes with it".
3. **Las claves hash del 03 y las lookups del 05.** Las claves `fs_key`, `fu_key`, `uplift_cell_key` y `fila_key` solo las verifican los checks del 05. El núcleo une por `s03_fs_id` en texto ("no hash keys"). `lookup_fu` y `lookup_uplift_cell` solo las lee un test, y `lookup_fs` se puede sacar de `series_table`.
4. **La columna `universo` en 04/06/08.** Desde que el 01 separa el time_series, siempre vale `normal`. Las comprobaciones del 04 y del 06 que la miran no pueden fallar, y los filtros de 08/10/11 siempre dan verdadero.
5. **23 tablas SQL que se escriben y ningún código lee, y que no están en el modelo de Power BI:**
   - dimension_levels, dimension_level_values;
   - calendario, fact_fu, lookup_fs/fu/uplift_cell;
   - series, fu_soporte, fu_huecos, series_tasa;
   - decision_eta2_pares, ladder_groups, series_estimacion;
   - composition_dynamics, estacionalidad_cartera, dim_tecnica;
   - forecast, time_series, series_backtest, series_technique_summary, ficha_serie.

   Pregunta para ti: ¿cuáles consultas a mano en SQL? Las que no, se dejan de escribir. `ficha_serie` es un subconjunto de `sff_forecast_series` con el mismo grano.
6. **Duplicados menores:**
   - `stage_summary` (paso 10) repite `summary`;
   - `exam_total` (paso 14) solo lo usa un respaldo del paso 18 que nunca se ejecuta;
   - hay tres rankings de técnicas con definiciones distintas (paso 14, informe y compare_with_real);
   - la tabla por rol de los meses sale cuatro veces (00, 02, 04, 07);
   - el bloque `ISOLATED: COHERENCIA` del 01 repite comprobaciones del mismo paso.

## Lo que mantendría aunque solo se muestre

- **Las formas del uplift estadístico (paso 16)** —toda la historia, ventana, ventana sin subidas—. Validaron la decisión del 18-oct y cuestan dos estimaciones extra. Las quitaría después del próximo run real, cuando la decisión de la regla del contrato esté cerrada.
- **El informe de revalorización y las series con subidas (paso 15).** No deciden, pero son el material de la presentación de descuento y MSRP.
- **La dinámica (paso 13).** No decide nada, pero es donde se mide la tendencia y la estacionalidad de la cartera (la subida de 4 pp de la tasa).
- **La homogeneidad del uplift (paso 15).** Parecía obsoleta, pero no lo es: el forecast valora a los renovadores al precio medio de la fila, y `ratio_seleccion` mide justo ese sesgo.

## Opciones de la Config que ninguna configuración activa

`distribute_marks=False`, `apply_credibility_shift=False`, `simulation_end`, `level_merge_max_pp`, `acquisition_level_dims`, `isolated_pipeline_units_col` y la base aproximada del uplift. Son ramas vivas que no se ejercitan en ningún run. Se pueden quitar cuando tengas claro que no vas a volver a ellas.
