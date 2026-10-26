# Arquitectura por etapas · 25-oct-2026

Continúa `ARQUITECTURA_ETAPAS_2026-10-24.md`. Ese documento proponía cuatro cambios; aquí están hechos, con dos más: cada etapa tiene un número (en el orden en que corre) y "marcas" pasa a llamarse **estados del cliente**.

## Conclusión

- **Cada predicción vive ahora en una caja con interfaz propia**, y cada caja tiene su examen:

| Predicción | Caja (etapa) | Interfaz | Se examina en |
|---|---|---|---|
| Tasa | `RateBox` (13) | `predict(filas, origen)`, `band(tasas, unidades)` | 12 (por composición) · 19 (por serie y total) · 23 |
| Precio | uplift por celda y por estado (14) | `cells`, `state_factors` | 16 |
| Estados del cliente | `StateMaturation` (17) | `mature(unidades)` | 17 (curva) · 19 (examen honesto) |
| Pipeline futura | `FuturePipeline` (18) | `new_rows(futuro)`, `simulated` | 18 (examen de la captación, **nuevo**) |

- **El paso 20 (el forecast) solo monta:** pide a cada caja, multiplica y suma. La banda oficial del total es la del examen (19), no la de cuadratura.
- **Lo que se examina es lo que se usa.** El forecast, el examen y la auditoría llaman a la misma caja de la tasa. La única diferencia es explícita: en el examen, una composición necesita 3 meses cerrados en el origen para predecir (`exam_min_months_to_predict`); en el forecast basta con 1, como antes.
- **Al usar la misma caja, el examen juzgó un cambio de la ronda anterior y lo descartó:** el repuesto de tasa con los estados del cliente (una serie softcancel sin composición tomaba el nivel de su celda *y* su estado). Con él, el error del total a 6 meses sube de 1,38 % a 1,53 %, y el del examen honesto de 1,53 % a 1,65 %. Encogerlo hacia el nivel de su combinación casi no ayuda (1,46 % con k = 1.000). Queda como opción, `rate_fallback_with_states`, desactivada por defecto.
- **El examen nuevo de la captación encuentra un problema** (sintético parecido al real):
  - Pure-New: error de 1,1 % en USD frente a 7,0 % de la baseline "lo mismo que hace un año";
  - Not-New: **20,4 % frente a 10,8 %**, con sesgo creciente (−12 % el mismo mes, −35 % a 5 meses). El nivel de los últimos 3 meses sigue una caída temporal y la arrastra a todo el horizonte.
  - En el run real ese nivel era 0,45×. Es lo primero que hay que mirar con datos reales.

## Las etapas, numeradas

| Etapa | Misión | Tipo | Lee | Ofrece a |
|---|---|---|---|---|
| 00 | el extracto encaja con la Config | datos | extracto | 01 |
| 01 | separa time series, valida valores, `_level_1` | datos + patrones | 00 | 02, 18 |
| 02 | rol de cada mes; borra lo que aún no se sabe | datos | 01 | 03 |
| 03 | tabla fina e ids | datos | 02 | casi todas |
| 04 | forecast units | datos | 03 | 05, 06, 20 |
| 05 | ruta de cada serie | datos | 04 | 06–08 |
| 06 | series de tasa: huecos, tasa propia, soporte, signo | datos | 04, 05 | 07–11, 13 |
| 07 | qué dimensión separa la tasa | patrones | 06 | 08 |
| 08–09 | escalera y credibilidad | predicción (tasa) | 06, 07 | 10, 13 |
| 10 | serie mensual de cada composición | predicción (tasa) | 08, 09 | 11, 12, 13 |
| 11 | dinámica (no decide) | patrones | 10 | 23, 24, 27 |
| 12 | técnica por composición y horizonte; bandas | examen (tasa) | 10 | 13 |
| **13** | **caja de la tasa** | predicción (tasa) | 06, 09, 10, 12 | 19, 20, 23 |
| 14 | uplift por celda y por estado; subidas en juego | predicción (precio) | 03 | 15, 16, 20 |
| **15** | **diagnósticos del precio para negocio** | patrones | 14 | 27 |
| 16 | examen del uplift | examen (precio) | 03 | 20 |
| **17** | **caja de los estados del cliente** | predicción (estados) | 03 | 19, 20 |
| **18** | **caja de la pipeline futura y caja del universo time_series, cada una con su examen** | predicción (pipeline, time series) + examen | 01, 03 | 20, 21, 24, 27 |
| 19 | examen de cada serie y del total | examen (tasa, estados) | 13, 17 | 20, 24, 26, 27 |
| 20 | forecast: monta y suma (el universo time_series incluido en meses, años y banda) | montaje | 13, 14, 16–19 | 21–26 |
| 21 | total por año y origen, igual al total del forecast | montaje | 18, 20 | 24, 27 |
| 22 | filas añadidas y borradas | informe | 03, 06, 20 | 27 |
| 23 | auditoría | salida | 08–13, 20 | 24, 27 |
| 24 | el núcleo | salida | casi todas | 25, 26, 27 |
| 25 | Power BI | salida | 24 | — |
| 26 | validación de la cadena | control | casi todas | 27 |
| 27 | informe y resumen para negocio | salida | todas | — |

La tabla de equivalencias (número antiguo → nuevo, módulo y prefijo de columna) está en `DOC_modelo_datos.md`.

## Lo que cambia para quien usa la salida

- **Columnas del núcleo renombradas:** el prefijo es el número de la etapa que las crea. Por ejemplo, `s17_*` pasa a `s20_*`, `s10_stage*` a `s08_stage*`, `s11_*` a `s09_*`, `s13_*` a `s11_*`, `s14_*` a `s12_*` y `s15_*` a `s14_*`. Los nombres de las tablas `sff_*` no cambian.
- **Columnas y textos de "marca" → "estado":** `factor_marca` → `factor_estado`, la vía `estadistica_marca` → `estadistica_estado`, `realizacion_con_marca` → `realizacion_con_estado`, `marcas_hoy` → `estados_hoy`, la columna `marca` del detector → `estado`, la página "Marcas" del explorador → "Estados del cliente".
- **Se mantienen** los campos de la Config que fija el notebook: `maturation_dims`, `distribute_marks`, `mark_excess_warning_pp`.
- **Hay que volver a ejecutar** antes de abrir el explorador desde SQL: las tablas de una ejecución anterior tienen los nombres viejos.

## Hecho hoy, además

- **Filas de predicción de los estados agregadas:** una por serie de destino, mes y origen de la pipeline, con el uplift ponderado por USD. Es exacto (un test lo comprueba) porque la tasa solo depende de la serie y del horizonte.
- **Banda oficial en el titular del informe y en el resumen para negocio:** `error_examen_usd` por año.
- **Paso 14 partido en dos:** el 14 estima (lo que leen 16 y 20) y el 15 explica (homogeneidad, regla del contrato, revalorización, subidas). Los números no cambian.

## Comprobación en el sintético parecido al real (antes → ahora)

- **El forecast cambia solo por el repuesto revertido:** 2027 pasa de $24,756M a $24,799M (+0,17 %), y cada mes entre +0,1 % y +0,25 %. Sin ese cambio, la caja de la tasa y la agregación dan exactamente los mismos números (comprobado: 0,000 % en los 15 meses).
- **Examen:**
  - framework a 1 mes 0,98 % y a 6 meses 1,38 %;
  - honesto a 1 mes 1,03 % y a 6 meses 1,53 %;
  - la baseline, 1,08 % y 2,79 %.
- **Banda oficial de 2027:** ±$372.000 (1,5 % de lo esperado). La de cuadratura que se publicaba antes era ±$146.000.
- **Núcleo:** 694.483 → 617.496 filas (−11 %). Las filas de predicción de los estados pasan de unas 160.000 a unas 107.000.
- **Tiempo:** 283 s → 266–308 s en dos ejecuciones; la diferencia es ruido de la máquina, sobre todo en la escritura del núcleo en SQL.
- **Tests:** 26 baterías, 444 comprobaciones, todas bien (dos nuevas: el examen de la captación y la agregación exacta).
