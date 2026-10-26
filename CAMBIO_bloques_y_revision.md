# CAMBIO · 8 bloques con pasos numerados dentro del bloque, y revisión de la lógica (25-oct-2026)

## En una frase

El framework pasa de 28 etapas numeradas 00–27 a **8 bloques funcionales con 29 pasos numerados dentro de su bloque** (1.1 … 8.6). Cada bloque tiene misión, mecanismo, lo que lee y lo que entrega, y vive en su módulo `block_<n>_<nombre>.py`.

La revisión de la lógica frente a lo que se espera del framework dejó seis correcciones:

1. el forecast usa la misma regla de meses mínimos que el examen;
2. la banda oficial incluye el error del precio y el de la captación simulada;
3. las relaciones del modelo de Power BI se escriben y se comprueban sobre los datos;
4. la validación 8.5 compara lo comparable;
5. el nivel del universo time series lo elige un examen;
6. cada fila futura dice de dónde sale su uplift.

## Los bloques

| Bloque | Misión | Pasos | Módulo |
|---|---|---|---|
| B1 · Preparación | dejar el extracto verificado y en las tablas de las que parte toda predicción | 1.1–1.7 | `block_1_preparation.py` |
| B2 · Tasa | la mejor predicción de la tasa de renovación, a cualquier horizonte, con banda medida | 2.1–2.7 | `block_2_rate.py` |
| B3 · Precio | a qué precio renueva quien renueva, con la vía que mejor lo predice y su banda | 3.1–3.3 | `block_3_price.py` |
| B4 · Estados del cliente | cuántas unidades llevarán cada estado al vencer | 4.1 | `block_4_customer_states.py` |
| B5 · Pipeline futura | la pipeline que el extracto aún no puede tener: renovaciones proyectadas y captación simulada | 5.1 | `block_5_future_pipeline.py` |
| B6 · Time series | el universo de retail a suscripción por regiones, con banda | 6.1 | `block_6_time_series.py` |
| B7 · Examen y montaje | examen en condiciones de forecast (7.1), forecast con banda oficial (7.2), total por año y origen (7.3) | 7.1–7.3 | `block_7_exam_and_assembly.py` |
| B8 · Salida y control | núcleo, satélites, Power BI, validación e informe; no cambia ningún número | 8.1–8.6 | `block_8_output.py` |

Equivalencias con la numeración anterior:

- 00–06 → 1.1–1.7
- 07–13 → 2.1–2.7
- 14–16 → 3.1–3.3
- 17 → 4.1
- 18 → 5.1; el universo time series, que antes iba dentro del 18, pasa a 6.1
- 19–21 → 7.1–7.3
- 22–27 → 8.1–8.6

El prefijo de las columnas del núcleo sigue al paso que las crea: `s03_` → `s1_4_`, `s20_` → `s7_2_`, etc. Las columnas `s0_` (copia del extracto) no cambian. La tabla completa está en `DOC_modelo_datos.md`.

## La revisión: qué se espera y qué hace

| Se espera | Qué hace el framework | Veredicto |
|---|---|---|
| La mejor forma de predecir la **tasa** | La escalera presta historia solo donde baja el error (2.1–2.3). Un backtest elige la técnica por composición y horizonte en meses de selección y la mide en meses de examen (2.6). La caja 2.7 es la que usan el examen, el forecast y la auditoría. | Correcto. Corregido: el forecast no aplicaba la regla de meses mínimos que sí aplica el examen (`exam_min_months_to_predict`). Ahora la aplican los dos; afecta a 15 de 1.326 composiciones. |
| La mejor forma de predecir la **revalorización** | Dos vías, contrato y estadística de la celda, más el factor por estados. El backtest 3.3 elige con el uplift reestimado sin los meses de examen. | Correcto. Faltaba su error en la banda. Ahora 3.3 mide el error del precio del total de un mes con las vías elegidas (`error_precio_total`) y entra en la banda 7.2. |
| Ajustar la **maduración** de la cartera con curvas de los estados | La caja 4.1 aprende curvas de los meses cerrados y mueve unidades entre estados sin crearlas. El examen honesto del 7.1 predice también los estados. | Correcto. Su error está dentro del error de la tasa del examen honesto. dormant y not_installed no traen fecha y usan la proporción histórica (lo dice el informe). |
| **Rellenar huecos** de pipeline suponiendo adquisición y renovación | La caja 5.1 crea las renovaciones proyectadas y la captación simulada; el bloque 7 las valora con 2, 3 y 4. La captación se examina en 6 orígenes. | Correcto. Faltaba su error en la banda. Ahora entra el error del examen de captación a su horizonte (`error_captacion_usd`). |
| **Probar en las mismas situaciones** | El examen 7.1 usa la misma caja, la misma regla de meses mínimos y los estados predichos. 3.3, 5.1 y 6.1 reestiman con los meses anteriores a cada origen. | Correcto tras la corrección de meses mínimos. Límite conocido: la estructura (escalera, técnica) se estima con todos los meses cerrados, así que el examen es algo optimista (lo dice el informe). |
| **Bandas de error** para negocio | La banda oficial se construye con errores medidos en meses cerrados: √(tasa² + precio²) + captación + time series. Una renovación proyectada suma además el error del mes en que se predijo su volumen. | Completa. Antes solo llevaba tasa y time series. |
| **Auditable en Power BI** con tablas y relaciones | El núcleo tiene una fila por registro y cada decisión como columna con el prefijo de su paso. Las satélites van por ids. 8.4 escribe `sff_relaciones_modelo` con las 15 relaciones del modelo, comprobadas sobre los datos (clave única en el lado 1, filas huérfanas en el lado n). | Nuevo: la tabla de relaciones. Nuevo en 6.1: `sff_ts_examen`. |

## Las correcciones, en detalle

### 1. Misma regla de meses mínimos en examen y forecast (7.2)

`RateBox.predict` recibe `min_months = exam_min_months_to_predict` (3) también en el forecast. Una composición con menos meses cerrados no predice, y sus filas toman la tasa de su celda, igual que en el examen. Lo examinado es lo que se usa.

### 2. La banda oficial reúne todos los errores medidos (7.2)

`error_examen_usd = √(error_tasa_usd² + error_precio_usd²) + error_captacion_usd + error_ts_usd`

- `error_tasa_usd`: el error del total del examen 7.1 a su horizonte × lo esperado. Una renovación proyectada usa √(e(h)² + e(h−12)²).
- `error_precio_usd`: `error_precio_total` del veredicto 3.3 × lo esperado.
- `error_captacion_usd`: el error del examen de captación 5.1 a su horizonte de venta, por fila simulada.
- `error_ts_usd`: la media banda de las filas time series (6.1).

Tasa y precio se suman en cuadratura porque son independientes sobre la misma fila. Captación y time series se suman lineales porque son volúmenes distintos y su dirección es desconocida.

### 3. Relaciones del modelo de Power BI (8.4)

`MODEL_RELATIONS` declara las 15 relaciones del modelo. 8.4 las comprueba sobre las tablas que escribió la ejecución (`configuration.written_tables`) y escribe `sff_relaciones_modelo`, con una fila por relación:

- tablas y columnas de los dos lados;
- filas de cada lado;
- si la clave del lado 1 es única;
- cuántas filas del lado n no encuentran su clave.

Checks 9–12 del 8.4.

### 4. Validación 8.5, check 4

Comprueba que toma su tasa del pool solo en las filas de composiciones con meses suficientes, la misma regla del punto 1.

### 5. El nivel del universo time series lo elige un examen (6.1)

**El fallo.** El nivel de una región era Σ unidades de sus 3 últimos meses / Σ los mismos meses un año antes, sin contraste. En el parecido al real, Italia tuvo un pico en julio de 2026 y su nivel salió 4,34. Eso proyectó 156 k$ en 2026-10, donde lo normal son unos 60 k$. El pico se copió después a 2027-10 dos veces: como simulado (156 k$) y como reentrada (182 k$ esperados de 260 k$ de pipeline).

**La corrección**, con el mismo principio que la técnica de la tasa: el examen decide. Se examinan cuatro reglas sobre los mismos orígenes (`TS_LEVEL_RULES`, `config.py`):

| Regla | Nivel de una región |
|---|---|
| `region` | el suyo (la regla anterior) |
| `credibilidad` | z · el suyo + (1 − z) · el del total, con z = unidades un año antes / (unidades un año antes + la media de las regiones) |
| `total` | el del total, para todas |
| `sin_nivel` | 1: el mismo mes del año anterior, tal cual |

Simula la regla de menor error medio del total del mes en el examen, y su error por horizonte es la banda. `ts_level_rule` permite fijar una. El examen de todas las reglas se escribe en `sff_ts_examen` (check 5 del 6.1).

**El examen en el parecido al real** (error medio del total del mes):

| Regla | 6 orígenes (el de la ejecución) | 12 orígenes (prueba aparte) |
|---|---|---|
| `region` (la anterior) | 47,8 % | 137,1 % |
| `credibilidad` | 54,4 % | 89,0 % |
| `total` | 61,3 % | 81,6 % |
| **`sin_nivel`** | **41,8 %** | **56,4 %** |
| nivel acotado a [0,5; 2] (descartada: límites arbitrarios y nunca la mejor) | 48,4 % | 66,8 % |

`sin_nivel` gana con las dos ventanas. Es el resultado habitual en series ruidosas: el estacional ingenuo es difícil de batir. El universo time series del parecido al real es una muestra del 3 % de las renovaciones, más ruidosa que una serie real de conversiones. Con los datos reales, el examen puede elegir otra regla.

**Límite:** la regla se elige y su error se mide en los mismos 6 orígenes, así que la banda es algo optimista. Con 6 orígenes no hay meses para separar selección y examen.

**Efecto en el parecido al real** (antes → ahora):

| | 2026 | 2027 |
|---|---|---|
| Revenue time series simulado | 264 k$ → 210 k$ | 798 k$ → 743 k$ |
| Renovación esperada de la reentrada | — | 305 k$ → 224 k$ |
| Total | 26,502 M$ → 26,447 M$ | 25,902 M$ → 25,766 M$ |
| Banda oficial | ±310 k$ → ±221 k$ | ±1,180 M$ → ±0,945 M$ |

El mes de 2026-10 deja de llevar el pico de Italia: su error time series baja de 162 k$ a 65 k$.

### 6. Cada fila futura dice de dónde sale su uplift (7.2 → `s3_1_origen_uplift`)

Una fila extendida (proyectada o simulada) puede tener una celda de uplift que nunca se vio. Entonces toma el uplift de su celda mandatory o, si no hay, el global, y en Power BI queda en blanco en la relación con `sff_uplift_celda`. La columna nueva del núcleo `s3_1_origen_uplift` dice de dónde sale el uplift de cada fila:

- `celda`: su fila de `sff_uplift_celda`;
- `celda_mandatory`;
- `global`;
- `contrato`;
- `varias`: en una fila agregada de la maduración.

## Hallazgos para decidir

- **A 1 mes el framework no bate a la baseline en el parecido al real.** El error del total es 1,0 % frente a 1,1 %, pero mejor solo en el 50 % de los meses (+991 $/mes de error). A 6 meses sí la bate: 1,4 % frente a 2,8 %, mejor en el 83 % de los meses.
- **El precio falla más que la tasa** en el sintético (3,4 % frente a 2,1 %), con un sesgo sistemático de unos −3 %. No se corrige el sesgo: se mide y entra en la banda. Corregirlo sería una decisión de modelo.
- **Relaciones con filas huérfanas** (avisos del check 11 del 8.4):
  - Las series que no suben la escalera no tienen composición (`sff_forecast_series[s2_2_stage3_id]`, 7.402 filas). Es esperado.
  - Hay 56.273 filas del núcleo cuyo `uplift_cell_id` no está en `sff_uplift_celda`. Esperado:
    - 46.360 son filas agregadas de la maduración, con `uplift_cell_id = varias`. Agregan varias celdas, y su uplift es la media ponderada exacta.
    - 9.913 son filas proyectadas o simuladas con una celda que nunca se vio; toman la de su celda mandatory.
    - Ahora las dos clases las explica `s3_1_origen_uplift`.
- **No se persiste el bloque 2** para reanudar: reanudar el bloque 1 desde SQL resultó más lento que recalcularlo.
