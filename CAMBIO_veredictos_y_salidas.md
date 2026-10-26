# CAMBIO · Veredicto por bloque, perfil de la pipeline e informe por bloques (25-oct-2026)

## En una frase

Cada bloque cierra con su **veredicto**: las hipótesis en que descansa, contrastadas con los datos de la ejecución, con métrica, referencia y veredicto. El bloque 1 deja además el **perfil de la pipeline**.

El informe pasa a tener **un capítulo por bloque**, precedido del veredicto, y el resumen para negocio dice de qué está hecha la banda y qué hipótesis no se sostienen. Ningún número del forecast cambia.

## Por qué

En el sintético las hipótesis se cumplen por construcción, y las ganancias del framework frente a la baseline no son significativas a 1 mes. Con datos reales hay que poder ver, en cada ejecución y sin leer 2.000 líneas, qué hipótesis se sostienen. El análisis completo está en `REVISION_FRAMEWORK_2026-10-25.md`.

## Qué hay de nuevo

### `verdicts.py` (módulo nuevo)

- `verdict_of_block(bloque, resultados, Config)`: las filas del veredicto de un bloque, con estas columnas:
  - `bloque`, `paso`, `tipo` (hipótesis · examen · dato · aviso);
  - `pregunta`, `metrica`, `valor`, `referencia`;
  - `veredicto` (se cumple · no se cumple · no concluyente · dato · revisar) y `detalle`.
- `record_the_verdict`: el orquestador lo llama tras cada bloque. El veredicto queda en `configuration.block_verdicts` y sale en pantalla. Nunca para la ejecución: si no se puede calcular, avisa y sigue.
- `paired_comparison` y `gain_verdict`: comparan dos métodos en los mismos meses con un test de signos y un t pareado. Una ganancia solo "se cumple" si es significativa (p < 0,10). Con 6 meses de examen eso exige 6 de 6, o un t pareado claro.
- `pipeline_profile`: el perfil de la pipeline (`sff_perfil_datos`).

### Las preguntas de cada bloque

| Bloque | Preguntas |
|---|---|
| B1 | volumen y su variación; historia propia del dinero a predecir (≥ 90 %); concentración; soporte propio; variación de la tasa de la cartera frente a su ruido; descuento desconocido; estados de hoy; peso del universo time series; comprobaciones del bloque |
| B2 | la escalera conoce mejor la tasa; dinero propio y prestado; la k de credibilidad sale de los datos o es la por defecto; las fusiones mejoran; señal más allá del ruido; la técnica elegida mejora a la por defecto en el examen (por tramo, con test); la banda de la técnica cubre (con el mismo umbral que la comprobación 2.6, `min_exam_coverage`) |
| B3 | la regla del contrato frente a la estadística; realización de la regla; precio por estado; la forma del uplift de la Config es la mejor en el examen; error del precio del total; subidas en curso |
| B4 | la curva predice la cuota final de cada estado fechado; estados sin fecha |
| B5 | por valor de captación, la simulación frente a "lo mismo que hace un año"; lo simulado frente al año anterior |
| B6 | la regla del nivel elegida y su error; lo que añade al año siguiente |
| B7 | por horizonte, el framework frente a la baseline (en unidades y en USD renovado), frente a la escalera con la técnica por defecto y frente a la serie sola (con tests); WAPE serie a serie frente a la serie sola; si el error del total es sesgo; si predecir los estados mejora el total; cambio de nivel de la tasa en el examen; qué cubre la banda; la maduración por año; de qué está hecha la banda (aviso si una parte pesa en la banda más del doble que en el dinero); si la confianza de cada fila está ordenada por el examen; reparto por confianza |
| B8 | la cadena cuadra; el modelo de Power BI puede hacer cada relación; tablas escritas |

### Tablas

| Tabla | Paso | Qué es |
|---|---|---|
| `sff_veredicto_bloques` | 8.6 | bloque × pregunta |
| `sff_perfil_datos` | 8.6 | sección × dimensión × valor |
| `sff_resumen_negocio` | 7.2 | añade `banda_tasa_usd`, `banda_precio_usd`, `banda_tasa_y_precio_usd`, `banda_captacion_usd` y `banda_time_series_usd` |

### Informe (8.6, 9 capítulos)

| Capítulo | Contenido |
|---|---|
| 0 | el veredicto: tabla de preguntas, lo que no se cumple y la banda desglosada |
| 1 · B1 | el extracto y el perfil |
| 2 · B2 | soporte, dinámica y precisión |
| 3 · B3 | precio, formas del uplift y error del precio por mes |
| 4 · B4 | examen de la curva, examen honesto y maduración por año (nuevo) |
| 5 · B5 | pipeline futura y examen de la captación (antes, dentro del forecast) |
| 6 · B6 | examen de las reglas del nivel y universo por año (antes, dentro del forecast) |
| 7 · B7 | forecast por mes y año, origen de la tasa y total por origen |
| 8 · B8 | filas añadidas o borradas, validación, relaciones de Power BI, dónde está cada serie y qué validar con datos reales |

### Resumen para negocio

- **"LA BANDA"**: por año, tasa y precio · captación · time series, frente a lo que pesa cada parte en el dinero. Dice que es un error típico, no un rango del 90 %.
- **"LO QUE HAY QUE VALIDAR"**: cuántas hipótesis se cumplen y las primeras que no.
- La línea de técnicas ya no dice "ganaron su sitio en el backtest": dice lo que dice el examen.

### Pantalla

- **Al cerrar cada bloque:** su veredicto. El del B1 lleva antes el perfil: cartera, concentración, soporte, historia y una línea por dimensión.
- **8.4:** solo los parámetros cambiados.
- **3.2:** 2 series de 24 meses (antes 5 de 36: `price_series_shown`, `price_series_months_shown`).
- **Maduración:** las tablas que calcula el 7.2 salen con la etiqueta [7.2].

### Datos que ahora se guardan para el veredicto

- `StateMaturation.curve_exams`: el examen de la curva de cada estado fechado (4.1).
- `uplift_verdict["formas_uplift"]`: el examen de las formas del uplift (3.3).

## Comprobaciones

- **8.6 pasa de 1 a 3 comprobaciones:** capítulos, más `sff_veredicto_bloques` y `sff_perfil_datos` escritas y releídas.
- **Tests:**
  - `test_2_5_8_6_dynamics_report`: 9 capítulos, los ocho veredictos en pantalla, el perfil y las reglas de los tests de signos (6 de 6 → p = 0,031; 5 de 6 → p = 0,22).
  - `test_resumen_graficos`: "LA BANDA" y "LO QUE HAY QUE VALIDAR".
  - `test_blocks_3_to_5_and_8_5`: nuevo título del capítulo del forecast.

## Lo que no cambia

El forecast, la banda oficial y todas las tablas anteriores tienen los mismos números. En el parecido al real:
- 2026: 26,447 M$ ± 221 k$;
- 2027: 25,766 M$ ± 945 k$.
