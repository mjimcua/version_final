# SFF · los 8 bloques y sus 29 pasos: qué entra, qué sale, qué comprueban

Oct 26, 2026 · @Miguel Angel

El framework tiene **8 bloques funcionales** y **29 pasos**. Cada bloque tiene una misión, un mecanismo para alcanzarla, lo que lee y lo que entrega, y vive en un módulo (`block_<n>_<nombre>.py`) con su clase. Los pasos se numeran dentro de su bloque y en el orden en que corren: el bloque 4 tiene el paso 4.1; el 7, los pasos 7.1 a 7.3. Cada paso mejora los datos, el conocimiento de los patrones o la predicción; las predicciones tienen su examen. El prefijo de una columna del núcleo es el paso que la crea, con guion bajo (`s7_2_esperado_usd`).

## Cómo leer este documento

- **Entra:** lo que el paso lee del contexto de la ejecución, con el paso que lo produce entre paréntesis.
- **Sale:** lo que deja para las siguientes, y las tablas `sff_*` que escribe.
- **Depende de:** el paso que tiene que haber corrido antes, y por qué.
- **Acciones:** las acciones, en el orden en que el paso las registra. Todas acaban contando sus comprobaciones y parando si alguna falla.
- **Comprobaciones:** las de la última ejecución del sintético parecido al real, con su número. Todas bloquean salvo las marcadas como aviso. "Se escribe y se relee" = la tabla se escribe y se vuelve a leer para confirmar que está entera.

&#91;embedded content: los 8 bloques · quién alimenta a quién\]

Una horizontal, el bloque 1 (1.1–1.7), prepara el dato una vez. Después cinco verticales predicen por separado, cada una con su examen: V1 la tasa (bloque 2, 2.1–2.7), V2 el uplift (bloque 3, 3.1–3.3), V3 los estados al vencer (bloque 4, 4.1), V4 la pipeline que el extracto aún no tiene (bloque 5, 5.1) y V5 el universo time series (bloque 6, 6.1). V1, V2 y V3 dicen cómo renueva una fila de pipeline; V4 añade filas, que el bloque 7 valora con V1, V2 y V3; V5 es otro universo, con su propia predicción y su propia banda. El bloque 7 examina (7.1), monta (7.2) y comprueba el total (7.3); el bloque 8 solo escribe y comprueba.

**Cada bloque cierra con su veredicto** (`verdicts.py`, desde el 25-oct): las hipótesis en que descansa, contrastadas con los datos de la ejecución, cada una con su métrica, su referencia y un veredicto (se cumple · no se cumple · no concluyente · revisar · dato). Las ganancias se contrastan con un test de signos y un t pareado sobre los meses de examen. El bloque 1 deja además el perfil de la pipeline. Las dos cosas quedan en `sff_veredicto_bloques` y `sff_perfil_datos` (paso 8.6).

**Al empezar** (líneas `[ORQ]`, desde el 26-oct) la ejecución no lista los parámetros: dice lo que implica la Config. Son los meses de cada rol y lo que ve cada horizonte del examen, desde qué mes se borra la pipeline de 1 año, de qué dims está hecha cada cosa, qué estados tienen curva y qué elige el examen. Los valores están en `sff_parametros` (8.4).

**La consola** (desde el 26-oct): con `console_log_level = "WARNING"` y un `log_file`, el notebook solo enseña avisos, fallos y la tabla final, y el fichero guarda todas las líneas y tablas. Es lo que trae la Config de producción: con el extracto real, la salida completa dejaba al navegador sin memoria.

## Bloque 1 · Preparación (pasos 1.1 a 1.7)

**Misión:** dejar el extracto verificado y en las tablas de las que parte toda predicción. No predice nada.

**Mecanismo:** comprueba el extracto contra la Config (estructura, después valores), aparta el universo time series, da su rol a cada mes (lo que aún no se sabe se borra, para que el futuro se vea como será), nombra cada fila por sus contextos, suma las filas de cada unit, da su ruta a cada serie y prepara su tasa. Corre una vez por ejecución y se puede persistir y reanudar (`persist_block`, `resume_block_from_sql`).

**Memoria** (desde el 26-oct): de 1.1 a 1.4 cada paso hace una tabla del tamaño del extracto, y la del paso anterior se libera en cuanto existe la nueva. Al cerrar el bloque solo queda la tabla fina y `extract_summary` (filas, columnas, unidades y USD que vencen), lo único que leen del extracto la 8.5 y la 8.6. Ya no se entregan `raw_extract`, `validated_extract`, `leveled_raw` ni `calendared_raw`.

**Lee:** el extracto y la Config. · **Entrega:** `fine_table`, `forecast_units`, `series_table`, `rated_units`, `series_rate`, `time_series_rows`.

### 1.1 · Validar el extracto

Comprueba que el extracto encaja con la Config antes de leer un solo valor.

- **Entra:** el extracto (`raw_extract`, que lee el orquestador).
- **Sale:** `validated_extract`: las mismas filas y columnas, con `period` convertido a mes. No escribe tablas. El extracto leído se libera al acabar el bloque: queda su resumen, extract\_summary.
- **Depende de:** nada.
- **Acciones:**
  1. Lee los roles de columna que declara la Config.
  2. Compara las columnas del extracto con esos roles.
  3. Lee los meses del extracto y les ajusta el calendario.
  4. Cuenta las comprobaciones y para si alguna falla.
  5. Convierte `period` a mes.
  6. Muestra las columnas por rol y el calendario.
  7. Resume lo leído: filas, unidades y USD de pipeline (lo que vence) y lo renovado, por universo y rol del calendario, con un total. Es un recuento: los valores se comprueban en la 1.2.
- **Comprobaciones (10):**
  1. El extracto tiene filas.
  2. Ningún nombre de columna se repite.
  3. Toda columna del extracto tiene un rol en la Config.
  4. Toda columna declarada está en el extracto (las ignoradas pueden faltar; los niveles generados los crea la 1.2).
  5. Toda fila tiene mes.
  6. Todo valor de `period` es un mes.
  7. `current_month` está declarado.
  8. El mes en curso cae dentro de los meses del extracto.
  9. El calendario deja meses para entrenar.
  10. No falta ningún mes entre el primero y el último.

### 1.2 · Valores y niveles de las dimensiones

Separa el universo time\_series, valida que los valores se pueden usar para calcular y genera el nivel grueso `<columna>_level_1` de las dimensiones de `leveled_dims`.

- **Entra:** `validated_extract` (1.1) y el JSON de niveles de ejecuciones anteriores, si existe.
- **Sale:** `leveled_raw` (el universo de renovaciones con sus `_level_1`) y `time_series_rows` (el universo time\_series). Guarda el JSON de grupos; no escribe tablas.
- **Depende de:** 1.1 (columnas con rol y `period` como mes).
- **Acciones:**
  1. Comprueba la marca de time\_series y separa los dos universos.
  2. Acota las comprobaciones de renovación a los meses cerrados (desde el mes en curso son parciales y la 1.3 las borra).
  3. Comprueba el dinero: pipeline en todos los meses, renovaciones en los cerrados.
  4. Comprueba las dimensiones y los estados del cliente (columnas timevarying).
  5. Comprueba el descuento exacto, si está declarado.
  6. Busca lo que es posible pero merece una mirada (avisos).
  7. Decide, por dimensión con nivel generado, si usa el JSON o genera.
  8. Genera los grupos de cada dimensión que los necesita (meses de entrenamiento).
  9. Rellena `<columna>_level_1`.
  10. Comprueba que el JSON conoce todos los valores.
  11. Prepara los niveles como tablas.
  12. Cuenta las comprobaciones y para si alguna falla.
  13. Informa del dinero de los meses cerrados y dibuja el universo time\_series mes a mes, antes de predecir nada: ingresos (total y regiones), unidades, valor por unidad y nivel (gráfico `ts_1_2_historia`).
  14. Muestra el criterio, los grupos (tasa por año), los valores y las decisiones de fusión.
- **Comprobaciones (53 en el sintético parecido al real; el número depende de las columnas declaradas):**
  - 1 · `flag_time_series` es 0 / 1.
  - 2 · Toda fila time\_series tiene todos los niveles de región.
  - 3 · Toda fila time\_series cerrada tiene unidades y valor (aviso).
  - 4 a 7 · Unidades y USD que vencen: sin nulos ni negativos, en todos los meses.
  - 8 a 30 · Cada medida de renovación (unidades, USD, base exacta, sin softcancel, isolated y sus seis tramos): sin negativos y nunca por encima de su total, en meses cerrados. Es una comprobación por medida y regla.
  - 31 · El momento de las isolated da una varianza ≥ 0.
  - 32 · Los 6 tramos de ratio suman la base sin softcancel.
  - 33 · Los seis tramos suman la base isolated en total.
  - 34 · Una fila sin renovación llega como 0, no como nulo.
  - 35 · Unidades y USD renovados son nulos a la vez.
  - 36 · Ninguna renovación gratis (unidades > 0 con USD ≤ 0).
  - 37 · Ningún USD renovado sin unidades renovadas.
  - 38 · La base exacta es 0 donde no renovó nada.
  - 39 · La base exacta es > 0 donde algo renovó.
  - 40 · Ninguna dimensión tiene un valor nulo o vacío.
  - 41 a 43 · Cada estado del cliente (dormant, softcancel, not\_installed) es 0 / 1.
  - 44 · Las columnas de maduración de softcancel son numéricas.
  - 45 · Nunca son negativas.
  - 46 · Con softcancel = 1 suman las unidades que vencen.
  - 47 · Con softcancel = 0 son nulas o 0.
  - 48 · Ninguna unidad de un mes futuro tiene softcancel fechado después del mes en curso.
  - 49 · El descuento está entre 0 y 1 (o nulo = desconocido).
  - 50 · Ninguna fila renueva más unidades de las que vencen (aviso).
  - 51 · Ninguna fila vence con unidades y USD = 0 (aviso).
  - 52 · Ninguna fila está repetida en todas sus columnas (aviso).
  - 53 · El JSON conoce todos los valores del extracto (aviso).
  - Con filas time series, además: toda fila time series tiene las dimensiones de la serie en la que renueva (mandatory y extra\_renovacion) y el JSON conoce sus valores (avisos).

### 1.3 · Aplicar el calendario

Da a cada fila su rol en el tiempo y hace que el futuro parezca no haber empezado.

- **Entra:** `leveled_raw` (1.2).
- **Sale:** `calendared_raw`: el extracto con seis columnas nuevas (`rol`, `es_mes_en_curso`, `s0_renovados_*`, `s0_vencen_*`). No escribe tablas.
- **Depende de:** 1.2 (dinero ya validado; sin él, leer un nulo como 0 escondería un error).
- **Acciones:**
  1. Guarda las renovaciones y la pipeline del extracto en las columnas `s0_`.
  2. Da a cada fila su rol (entrenamiento, examen, proyección) y la marca de mes en curso.
  3. En los meses cerrados, lee una renovación nula como 0.
  4. Desde el mes en curso, borra las renovaciones ya contabilizadas y la pipeline de las licencias de 1 año vendidas o renovadas desde ese mes (vencen 12 meses después): aún no se conoce y se proyecta.
  5. Comprueba el calendario y el dinero.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra el calendario por rol.
- **Comprobaciones (9):**
  1. Toda fila tiene uno de los tres roles.
  2. El entrenamiento tiene al menos 12 meses.
  3. Todo mes de examen tiene filas.
  4. El mes en curso tiene pipeline.
  5. La pipeline no cambia salvo en las filas que aún no se conocen (unidades y USD).
  6. Ninguna fila de 1 año creada desde el mes en curso conserva su pipeline.
  7. Las renovaciones de los meses cerrados no cambian (un nulo se lee como 0).
  8. Ninguna fila desde el mes en curso conserva una renovación.
  9. Las columnas `s0_` guardan intactas las renovaciones y la pipeline del extracto.

### 1.4 · Tabla fina

Nombra cada fila por los contextos de los que toma sus números. No agrega nada.

- **Entra:** `calendared_raw` (1.3).
- **Sale:** `fine_table`: el extracto con el tramo de descuento y tres ids, `fs_id` (serie de tasa), `fu_id` (serie en su mes) y `uplift_cell_id` (celda de precio). Vive en memoria; el núcleo (8.3) escribe cada fila.
- **Depende de:** 1.3 (el rol de cada mes).
- **Acciones:**
  1. Construye `fs_id` con las columnas de la serie de tasa.
  2. Construye `fu_id` = `fs_id` | mes.
  3. Deriva el tramo de descuento del descuento exacto.
  4. Construye `uplift_cell_id` con las columnas de la celda de uplift.
  5. Comprueba el tramo, el grano y que no cambió nada más.
  6. Guarda la tabla fina en memoria.
  7. Cuenta las comprobaciones y para si alguna falla.
  8. Muestra las filas por tramo y unas cuantas filas con sus ids.
- **Comprobaciones (3):**
  1. Todo descuento exacto cae en un tramo (uno nulo, en `sin_dato`).
  2. Una fila por forecast unit, valores de revalorización y descuento exacto (el grano del extracto).
  3. Mismas filas y mismo orden; ninguna columna de la 1.3 cambia.

### 1.5 · Forecast units

Suma las filas finas de cada forecast unit (una serie en su mes). La tasa se estudia por unit, porque los valores de revalorización cambian el precio, no la tasa.

- **Entra:** `fine_table` (1.4).
- **Sale:** `forecast_units`: una fila por unit con sus cuatro medidas sumadas y el número de filas finas. En memoria.
- **Depende de:** 1.4 (`fs_id`, `fu_id`).
- **Acciones:**
  1. Agrupa las filas finas por unit.
  2. Toma la descripción de cada unit (serie, mes, rol) de su primera fila.
  3. Suma las cuatro medidas y cuenta las filas finas.
  4. Comprueba las units y que el dinero se conserva.
  5. Cuenta las comprobaciones y para si alguna falla.
  6. Muestra las units por rol y las filas finas por unit.
- **Comprobaciones (3):**
  1. Las cuatro medidas se conservan: Σ filas finas = Σ units.
  2. Las renovaciones son números en los meses cerrados y nulas desde el mes en curso.
  3. Toda unit tiene algo que vence.

### 1.6 · Series y rutas

Describe cada serie por los meses que tiene y decide cómo se trata: se estima, solo sirve de historia o se predice desde sus parientes. No se tira ninguna.

- **Entra:** `forecast_units` (1.5).
- **Sale:** `series_table`: una fila por serie con su cobertura, su ruta (`predecible`, `solo_historia`, `solo_futuro`) y el USD que tiene por predecir. En memoria.
- **Depende de:** 1.5 (los meses y el rol de cada unit).
- **Acciones:**
  1. Agrupa las units por serie.
  2. Cuenta los meses de cada serie en cada rol, y su primer y último mes.
  3. Asigna cobertura y ruta.
  4. Calcula el dinero que cada serie tiene por predecir (lo que vence desde el mes en curso).
  5. Comprueba las series.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra las series y el dinero por ruta.
- **Comprobaciones (2):**
  1. Toda unit pertenece a una serie de la tabla, y toda serie tiene ruta.
  2. Toda serie con algo por predecir tiene pipeline por predecir (aviso).

### 1.7 · Series de tasa

Prepara la tasa de renovación: rellena los huecos de la historia, calcula la tasa solo donde es verdad y resume cada serie.

- **Entra:** `forecast_units` (1.5) y `series_table` (1.6).
- **Sale:** `rated_units` (las units con filas hueco y la tasa de cada unit) y `series_rate` (una fila por serie: soporte `n_propio`, tasa propia, error binomial de Wilson y signo). En memoria; las filas hueco llegan al núcleo.
- **Depende de:** 1.5 (las units) y 1.6 (la ruta: solo las series estimables reciben huecos).
- **Acciones:**
  1. Encuentra las series que reciben huecos y el tramo de su historia.
  2. Añade las filas hueco (tasa indefinida, nunca 0 %).
  3. Calcula la tasa de cada unit.
  4. Resume cada serie: soporte, tasa propia, error binomial.
  5. Asigna el signo de cada serie.
  6. Comprueba los huecos, las tasas y el resumen.
  7. Cuenta las comprobaciones y para si alguna falla.
  8. Muestra las series por soporte y las mayores.
- **Comprobaciones (4):**
  1. Las filas hueco caen solo dentro de la historia de series estimables, sin repetir una unit.
  2. Solo hay tasa en meses cerrados reales con algo que vence.
  3. Toda tasa está entre 0 y 1.
  4. Toda serie tiene su fila de resumen.

## Bloque 2 · Tasa (pasos 2.1 a 2.7)

**Misión:** la mejor predicción de la tasa de renovación que permite la historia, para cualquier fila de pipeline y a cualquier horizonte, con una banda medida.

**Mecanismo:** una forecast serie sola es casi todo ruido binomial. El bloque toma prestada historia solo donde baja el error: la escalera fusiona la serie con las que menos difieren en tasa hasta tener soporte (2.1–2.3); cada grupo final es una serie mensual (2.4), descrita sin decidir nada (2.5); un backtest elige por grupo y horizonte la técnica que mejor predice meses que no vio, y el ancho de su banda (2.6); todo va a una caja a la que se le pide la tasa (2.7). El examen del forecast entero con esta caja, frente a la baseline, es el 7.1.

**Lee:** `rated_units`, `series_rate`, `series_table` (bloque 1). · **Entrega:** `rate_box` y, para la auditoría, el núcleo y el informe, la escalera, las estimaciones, las composiciones, la dinámica y el backtest.

### 2.1 · Dimensiones

Mide cuánto separa la tasa cada dimensión (η², contribución única, ω²) y fija el orden en que la escalera colapsará las dimensiones mandatory: primero la que menos R² cuesta perder.

- **Entra:** `series_rate` (1.7) y `series_table` (1.6).
- **Sale:** `dimension_decision`: una fila por dimensión con sus cifras, su orden de colapso y su pérdida secuencial. Escribe `sff_decision_eta2`.
- **Depende de:** 1.7 (la tasa propia y el soporte de cada serie).
- **Acciones:**
  1. Construye la base: las series cuya tasa propia informa, con sus dimensiones.
  2. Calcula η², contribución única y ω² de cada dimensión.
  3. Calcula el orden de colapso de las mandatory.
  4. Comprueba las cifras y el orden.
  5. Escribe la tabla de la decisión.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra las dimensiones en orden de colapso.
- **Comprobaciones (5):**
  1. La base tiene al menos 3 series.
  2. Toda cifra está entre 0 y 1.
  3. Toda dimensión mandatory con variación tiene una posición de colapso, de 1 a K.
  4. Un `*_level_N` nunca colapsa antes que su `*_level_(N+1)`.
  5. `sff_decision_eta2` se escribe y se relee.

### 2.2 · Grupos de la escalera

Fusiona las forecast series pasada a pasada (signo → extras → dimensiones mandatory en el orden de la 2.1) hasta que cada grupo tiene soporte. Cada pasada es una partición: los totales siempre cuadran.

- **Entra:** `rated_units` (1.7), `series_rate` (1.7), `series_table` (1.6) y `dimension_decision` (2.1).
- **Sale:** `ladder`: el id de cada serie en cada pasada, el resumen por pasada, el grupo final (la composición) y la referencia de credibilidad de cada serie. Escribe `sff_ladder_steps`, `sff_ladder_summary`, `sff_ladder_merges` y `sff_composition_members`.
- **Depende de:** 2.1 (el orden de colapso) y 1.7 (soporte y signo de cada serie).
- **Acciones:**
  1. Planifica las pasadas a partir de la decisión de la 2.1.
  2. Toma las series estimables con su signo y sus columnas.
  3. Calcula el id de cada serie en cada pasada (y de cada patrón posible, para las referencias).
  4. Recorre las pasadas: cierra los grupos que llegan al suelo de soporte y sube los demás.
  5. Fija el grupo final de cada serie y la referencia de credibilidad de cada grupo final.
  6. Comprueba la partición, los signos, los grupos cerrados y las referencias.
  7. Escribe las pasadas, su resumen, las fusiones y los miembros de cada composición.
  8. Cuenta las comprobaciones y para si alguna falla.
  9. Muestra cómo mejora el soporte en cada pasada.
- **Comprobaciones (8):**
  1. Cada pasada es una partición: un grupo por serie, y las unidades que vencen suman.
  2. Un grupo nunca mezcla signos.
  3. Un grupo cerrado conserva su id, y cada pasada tiene igual o menos grupos.
  4. Toda referencia de credibilidad tiene más series que su grupo.
  5. a 8. `sff_ladder_steps`, `sff_ladder_summary`, `sff_ladder_merges` y `sff_composition_members` se escriben y se releen.

### 2.3 · Credibilidad

Da a cada serie la tasa de su grupo final, reforzada por credibilidad con la referencia cuando el grupo no tiene precisión propia. Añade los dos errores y el nivel de riesgo.

- **Entra:** `series_rate` (1.7) y `ladder` (2.2).
- **Sale:** `series_estimate` (una fila por serie: grupo final, referencia, k, z, tasa estimada, `se_estimacion`, `se_prediccion`, nivel de riesgo) y `money_by_level`. Escribe `sff_niveles_riesgo`.
- **Depende de:** 2.2 (grupo final y referencia).
- **Acciones:**
  1. Toma el grupo final y la referencia de cada serie estimable.
  2. Calcula la k de credibilidad de cada referencia, con los grupos que la comparten.
  3. Calcula la tasa y los dos errores de cada serie (las demás: su tasa propia, si la tienen).
  4. Asigna el nivel de riesgo de cada serie.
  5. Comprueba las tasas y los errores.
  6. Escribe el dinero por nivel.
  7. Cuenta las comprobaciones y para si alguna falla.
  8. Muestra el dinero por nivel de riesgo y qué significa cada nivel.
- **Comprobaciones (5):**
  1. Toda serie tiene exactamente una estimación.
  2. Todas las series de un grupo tienen la misma tasa estimada.
  3. Toda tasa estimada y toda z están entre 0 y 1.
  4. El error de predicción nunca es menor que el de estimación.
  5. `sff_niveles_riesgo` se escribe y se relee.

### 2.4 · Serie mensual de cada composición

Construye la serie mensual de la tasa de cada grupo final, sumando todas sus series. Es la serie que el backtest juzga y la caja de la tasa prolonga.

- **Entra:** `rated_units` (1.7), `ladder` (2.2) y `series_estimate` (2.3).
- **Sale:** `pool_series` (una fila por composición × mes cerrado: vencen, renovadas, tasa) y `pool_reference` (una fila por composición: meses, soporte, tasa, series, dinero y *gate*). Escribe `sff_pool_serie` y `sff_pool_referencia`.
- **Depende de:** 2.2 (quién forma cada composición) y 1.7 (las unidades de cada mes).
- **Acciones:**
  1. Toma las composiciones de la 2.2 y todas las series que entran en su tasa (usuarias y prestamistas).
  2. Construye la serie mensual de cada composición.
  3. Construye la ficha de cada composición.
  4. Comprueba la serie y la ficha.
  5. Escribe la serie y la ficha.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra las composiciones por *gate* y las mayores.
- **Comprobaciones (5):**
  1. Todo grupo final tiene serie mensual.
  2. El soporte de cada grupo es el mismo que en la 2.2.
  3. Toda tasa mensual está entre 0 y 1.
  4. y 5. `sff_pool_serie` y `sff_pool_referencia` se escriben y se releen.

### 2.5 · Dinámica de la tasa

Mide qué mueve la tasa más allá del ruido binomial: φ (variación observada / ruido), tendencia y estacionalidad. Es descriptiva: no restringe ninguna técnica; decide el backtest.

- **Entra:** `pool_series` (2.4), `pool_reference` (2.4) y `rated_units` (1.7).
- **Sale:** `pool_dynamics` (una fila por composición: φ, tendencia, estacionalidad, meses alto y bajo), `portfolio_profile` y `portfolio_dynamics` (el perfil mensual de la cartera). En memoria; la dinámica llega al núcleo y a la auditoría.
- **Depende de:** 2.4. Nadie decide con su salida: la leen la auditoría (8.2), el núcleo (8.3) y el informe (8.6).
- **Acciones:**
  1. Selecciona las composiciones que se miden (soporte e historia).
  2. Calcula φ, tendencia y estacionalidad de cada una.
  3. Calcula el perfil mensual de toda la cartera.
  4. Comprueba los atributos.
  5. Cuenta las comprobaciones y para si alguna falla.
  6. Muestra el veredicto general: el perfil de la cartera y cuántas composiciones tienen tendencia, estacionalidad o φ > 1.
- **Comprobaciones (2):**
  1. Toda composición medida tiene sus atributos (φ ≥ 0, p-valores en \[0, 1\]).
  2. El perfil de la cartera cubre los 12 meses del calendario.

### 2.6 · Backtest de la tasa

Elige, para cada composición con soporte y cada horizonte, la técnica que mejor predice en los meses de selección. Después la mide en los meses de examen, que no ha visto, y fija el ancho de su banda de error.

- **Entra:** `pool_series` (2.4) y `pool_reference` (2.4).
- **Sale:** `backtest`: las predicciones (composición × mes × horizonte × técnica), la técnica elegida por composición y tramo de horizonte, las bandas de error y el examen por composición y del total. Escribe `sff_backtest_predicciones`, `sff_decision_tecnica`, `sff_decision_bandas`, `sff_backtest_examen` y `sff_backtest_examen_total`.
- **Depende de:** 2.4 (la serie mensual de cada composición y su *gate*).
- **Acciones:**
  1. Fija el calendario de la prueba: meses de selección y de examen, y horizontes.
  2. Predice con cada técnica cada composición juzgada, mes y horizonte.
  3. Elige la técnica de cada composición y tramo de horizonte.
  4. Calcula las bandas de error de cada técnica y horizonte.
  5. Mide la precisión en el examen, por composición y del total.
  6. Comprueba la prueba.
  7. Escribe las cinco tablas.
  8. Cuenta las comprobaciones y para si alguna falla.
  9. Muestra el ranking de técnicas, las elecciones y la precisión del examen.
- **Comprobaciones (10):**
  1. Ninguna predicción vio su mes objetivo ni nada posterior a su origen.
  2. Toda composición tiene decisión en cada tramo de horizonte.
  3. La técnica por defecto se predijo dondequiera que se predijo otra.
  4. El ranking compara las técnicas sobre los mismos objetivos.
  5. En el examen, la banda contiene al menos el 80 % de los errores de la técnica elegida.
  6. a 10. Las cinco tablas se escriben y se releen.

### 2.7 · La caja de la tasa (`RateBox`)

Mete la predicción de la tasa en una sola caja: la técnica elegida de cada composición y horizonte, movida hacia su referencia de credibilidad y aprendida con los meses cerrados hasta cualquier origen. La usan el examen (7.1), el forecast (7.2) y la auditoría (8.2), así que lo que se examina es lo que se usa.

- **Entra:** `rated_units` (1.7), `series_estimate` (2.3), `pool_series` (2.4), `backtest` (2.6) y `ladder` (2.2).
- **Sale:** `rate_box`, con `predict(filas, origen)` (tasa, técnica, origen de la tasa) y `band(tasas, unidades)` (su intervalo). No escribe tablas.
- **Depende de:** 2.6 (técnica elegida y bandas), 2.4 (historia de cada composición) y 2.3 (z y referencia de cada serie).
- **Una serie sin composición** toma el nivel de su celda mandatory y, si no lo hay, el de la cartera. Con `rate_fallback_with_states` usaría también sus estados del cliente; está apagado porque el examen lo hace peor (1,38 % → 1,53 % de error del total a 6 meses).
- **Acciones:**
  1. Construye la caja: las composiciones y sus meses cerrados, las técnicas elegidas por tramo y las referencias.
  2. Predice la tasa de cada composición a h = 1 en el último mes cerrado, como prueba.
  3. Comprueba la caja.
  4. Cuenta las comprobaciones y para si alguna falla.
- **Comprobaciones (2):**
  1. Toda composición con meses cerrados predice a h = 1 una tasa en \[0, 1\].
  2. Los niveles de la caja en el último mes cerrado son los de la escalera (2.3).

## Bloque 3 · Precio (pasos 3.1 a 3.3)

**Misión:** a qué precio renueva quien renueva (la revalorización: USD renovado sobre el USD que vencía), con la vía que mejor lo predice y su banda.

**Mecanismo:** dos candidatos por fila: la regla del contrato 1 / (1 − descuento) donde se conoce el descuento, y el uplift estadístico de su celda (sus renovadores pasados, sin las subidas de precio, que se aplican aparte mientras dura su ciclo). Un backtest en los meses de examen, con el uplift estimado antes, decide qué vía usa el forecast, si el factor de los estados ayuda y cuánto falla el precio del total de un mes (la banda del precio).

**Lee:** `fine_table` (bloque 1). · **Entrega:** celdas de uplift, factores por estado, subidas en curso y el veredicto (con el error del precio).

### 3.1 · Uplift

Estima a qué precio renueva un renovador frente a lo que le vencía: el uplift estadístico de cada celda, el factor de precio de cada combinación de estados del cliente y la comprobación de la regla del contrato 1 / (1 − descuento).

- **Entra:** `fine_table` (1.4).
- **Sale:** `uplift_cells` (uplift de cada celda, con su origen — propia, padre, celda, global — y su banda bootstrap), `uplift_state_factors` (factor por combinación de estados), `contract_check`, `price_increases` (las subidas de precio en juego) y `uplift_basis` (renovadores, ventana y monitor de precios). Solo lo lee la 3.2: se queda dentro del bloque y se libera al acabar la 3.2. Escribe `sff_uplift_celda` y `sff_uplift_contrato_check`.
- **Depende de:** 1.4 (`uplift_cell_id`, tramo de descuento, rol de cada mes).
- **Acciones:**
  1. Toma las filas renovadoras de los meses cerrados, con la base de su uplift.
  2. Detecta las subidas de precio de cada grupo, fija la ventana del uplift y deflacta sus meses; calcula el uplift de cada celda (propia, padre, celda, global), su banda y el factor de precio de cada combinación de estados.
  3. Contrasta la regla del contrato con las renovaciones pasadas de descuento conocido.
  4. Comprueba los uplifts.
  5. Escribe las tablas.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra el uplift por origen y el precio por estado del cliente.
- **Comprobaciones (6):**
  1. Toda celda de uplift de la tabla fina tiene uplift (también las futuras).
  2. Todo uplift está en \[0,01, 3,0\].
  3. La banda contiene el uplift de cada celda que la tiene.
  4. Ningún grupo de precio acaba su historia con un escalón sin confirmar (por encima del 3 % menos de 3 meses). Es un aviso: si fuera una subida, el forecast no la lleva.
  5. y 6. `sff_uplift_celda` y `sff_uplift_contrato_check` se escriben y se releen.

### 3.2 · Diagnósticos del precio

Explica el precio de los renovadores para negocio. Ninguna predicción lee lo que sale de aquí.

- **Entra:** `fine_table` (1.4), `uplift_basis` (3.1) y `contract_check` (3.1).
- **Sale:** `uplift_homogeneity` (por celda: precio medio de los renovadores frente al de la celda; solo con la base exacta). Escribe `sff_uplift_homogeneidad`. El resto sale por pantalla.
- **Depende de:** 3.1 (sus renovadores, su ventana y su monitor de precios; no los recalcula).
- **Acciones:**
  1. Calcula la homogeneidad de cada celda.
  2. Escribe la homogeneidad.
  3. Cuenta las comprobaciones y para si alguna falla.
  4. Lee la regla del contrato para negocio: quién la cumple y el precio de lista implícito de cada tramo.
  5. Lee la revalorización de los renovadores: población, distribución, reparto por descuento y camino mensual.
  6. Muestra las subidas de precio y la ventana del uplift: el mes de cada ciclo y dónde se pierde su efecto.
- **Comprobaciones (1):**
  1. `sff_uplift_homogeneidad` se escribe y se relee.

### 3.3 · Backtest del uplift

Decide, en los meses de examen, qué uplift predice mejor el precio de las renovaciones donde se conoce el descuento: el estadístico de la celda, la regla del contrato, o el contrato en las filas neutras y el estadístico en las que tienen un estado. También decide si se aplica el factor por estado del cliente. Su veredicto gobierna el forecast.

- **Entra:** `fine_table` (1.4).
- **Sale:** `uplift_backtest` (error de cada vía, en total y por tramo de descuento) y `uplift_verdict` (`via_usada_con_descuento`, `factor_estado` y `error_precio_total`: el error medio del precio del total de un mes con las vías que usará el forecast, la banda del precio). Escribe `sff_backtest_uplift`.
- **Depende de:** 1.4. Reestima el uplift con los meses anteriores al examen, en vez de usar el de la 3.1, para no ver el examen.
- **Acciones:**
  1. Detecta las subidas conocidas antes del examen y estima el uplift estadístico con los meses anteriores, de todas las formas (toda la historia, la ventana, la ventana sin subidas).
  2. Predice las renovaciones de los meses de examen por las dos vías (y por cada forma de la estadística).
  3. Mide el error de cada vía y da el veredicto; lo mismo para el precio por estado del cliente; mide el error del precio del total de cada mes de examen con las vías elegidas (la banda del precio que usa el 7.2).
  4. Comprueba la prueba.
  5. Escribe el resultado.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra la comparación y las formas del uplift estadístico, en total y mes a mes.
- **Comprobaciones (3):**
  1. El uplift estadístico se estimó sin los meses de examen (de todas las formas).
  2. Las dos vías se compararon sobre las mismas filas.
  3. `sff_backtest_uplift` se escribe y se relee.

## Bloque 4 · Estados del cliente (paso 4.1)

**Misión:** cuántas unidades de cada mes futuro llevarán cada estado (softcancel, dormant…) al vencer. El extracto trae los estados de hoy: una licencia que vence dentro de 10 meses aún no ha tenido tiempo de hacer softcancel.

**Mecanismo:** curvas de maduración aprendidas de los meses cerrados, en una caja (`StateMaturation`) que mueve unidades entre estados sin crearlas. El bloque 7 valora lo que mueve; el examen honesto del 7.1 también predice los estados: es el examen de este bloque.

**Lee:** `fine_table` (bloque 1). · **Entrega:** `state_maturation`.

### 4.1 · Estados del cliente (`StateMaturation`)

Predice cuántas unidades de cada mes futuro llevarán un estado negativo (softcancel, dormant…) cuando venzan. Es una caja que aprende de los meses cerrados y que llaman el examen (7.1) y el forecast (7.2). Con `distribute_marks` apagado es la identidad: maduración 0.

- **Entra:** `fine_table` (1.4).
- **Sale:** `state_maturation`, con `mature(unidades)`: por grupo (la serie sin sus estados), mes y combinación, las unidades de hoy y las unidades al vencer. También deja maturation\_by\_region, la maduración por región de la acción 3. No escribe tablas; la 7.2 escribe su curva (sff\_curva\_maduracion, con curve\_table) y `sff_forecast_maduracion`.
- **Depende de:** 1.4, y de las columnas de maduración (`<estado>_month_k`) que valida la 1.2.
- **Acciones:**
  1. Construye la caja: los meses cerrados de los que aprende, los estados con curva (fechados) y los que usan la proporción histórica.
  2. Calcula la curva de cada estado fechado y su examen en los meses de examen. Decide si la curva se usa. Dibuja el gráfico que explica la curva (04\_1\_curva\_de\_maduracion\_explicada), que se ve siempre, también con la consola en silencio. Tiene una fila por estado y tres paneles: cuándo se pone el estado (la distribución f(k) de los meses antes de vencer y la parte ya visible F(h)); la pipeline de hoy (lo visto más lo que añade la curva, frente al tope histórico); y el error de la cuota final según la antelación, con curva y sin ella, frente al umbral. Los números salen de maturation\_explained.
  3. Muestra la maduración de cada estado por región del primer nivel (tr\_regional\_level\_1): una tabla por estado y el gráfico 04\_1\_maduracion\_por\_region, una línea por región. softcancel, con fechas: la curva del pasado, media de las cohortes cerradas. dormant y not\_installed, sin fechas: la pipeline vista hoy sobre la cuota final de la región.
  4. Muestra lo que renueva cada combinación de estados en los meses de los que aprende.
  5. Comprueba la caja.
  6. Cuenta las comprobaciones y para si alguna falla.
- **Comprobaciones (3):**
  1. La caja aprende solo de meses cerrados, hasta el último.
  2. La curva de cada estado fechado falla su cuota final menos que el estado visto entonces, y por menos de maturation\_curve\_max\_error\_pp (5 pp). Aviso: si no, la curva no se usa y el estado toma su cuota histórica en el forecast y en el examen honesto (desde el 26-oct: la curva de dormant del dato real fallaba 70 pp y restaba un 17,6 % a 2026).
  3. Todo estado fechado tiene una curva aprendida de cohortes que lo llevan (aviso: si no, su maduración es 0).

## Bloque 5 · Adquisición (paso 5.1)

**Misión:** la adquisición del año en un solo paso: lo ya vendido a clientes nuevos, lo que falta por vender del mes en curso a diciembre (la ventana) y, si hay objetivo del equipo de adquisición, hasta dónde llega. Lo vendido en la ventana vence 12 meses después: es pipeline del año siguiente que el extracto aún no tiene.

**Mecanismo:** una venta se simula como lo que vendió su grupo el mismo mes del año anterior × el nivel de su valor de adquisición × su valor por unidad. Con el CSV de objetivos, cada subregión se completa hasta su objetivo en todos los plazos; solo la parte de 1 año vence en el horizonte y hace filas. La simulación se examina en los meses cerrados como si cada uno fuera el mes en curso. Las filas salen sin predecir: el bloque 7 las valora con los bloques 2, 3 y 4, y el error del examen entra en la banda. **Lo que no está aquí:** la pipeline real que vence entre el mes en curso y diciembre no es adquisición. El paso 7.2 la pronostica y proyecta su renovación de 1 año, que vuelve a vencer 12 meses después (filas proyectadas), porque necesita ese forecast.

**Lee:** `fine_table` (bloque 1) y, si está configurado, `objetivos_adquisicion_2026.csv`. · **Entrega:** `future_pipeline` (la caja, la tabla hasta el objetivo, la adquisición mes a mes, la comparación con el año anterior y el examen).

### 5.1 · Adquisición (`FuturePipeline`)

Lee lo vendido en 1, 2 y 3 años, simula la ventana (mes en curso → diciembre) y la completa hasta el objetivo del equipo de adquisición. Después examina el método en los meses cerrados y dibuja la adquisición mes a mes. El error del examen entra en la banda oficial (7.2).

- **Entra:** `fine_table` (1.4) y, con `acquisition_targets_file`, el CSV de objetivos (subregión de `tr_regional_level_3`, miles de USD, todos los plazos).
- **Sale:** `future_pipeline`: la caja, con `acquisition_rows()` (las filas simuladas de 1 año, sin predecir; el paso 7.2 las junta con las proyectadas), los niveles, la tabla hasta el objetivo, la adquisición mes a mes, la comparación con el año anterior y el examen. Gráfico `05_1_adquisicion_mes_a_mes.png`. No escribe tablas; el examen sale en el informe.
- **Depende de:** 1.4. Una licencia de t años vendida en el mes m es la fila de adquisición de plazo t que vence en m + 12 t.
- **Acciones:**
  1. Lee las ventas de adquisición por plazo y calcula su nivel (últimos 3 meses sobre los mismos meses un año antes).
  2. Simula la captación de la ventana y la compara con los mismos meses del año anterior. Si hay objetivos del equipo de adquisición (acquisition\_targets\_file, por acquisition\_target\_column), completa el año de cada región hasta su objetivo. Lo hace con un factor por región sobre la simulación, para que lo vendido de enero al último mes cerrado más lo simulado sea igual al objetivo, y la mezcla de productos no cambia. El objetivo cuenta todos los plazos: lo que falta (objetivo − lo vendido este año en 1, 2 y 3 años) se reparte por plazos en la proporción de las ventas del año, o en la de los últimos 12 meses con todos los plazos visibles si el extracto aún no llega a ellos. Solo la parte de 1 año genera filas; la de 2 y 3 años se cuenta para que el total cuadre y vence después del horizonte. Lo enseña en la tabla ADQUISICIÓN HASTA EL OBJETIVO.
  3. Examina la captación: en cada uno de los 6 últimos meses cerrados rehace la caja como si fuera el mes en curso y compara lo simulado con lo vendido.
  4. Comprueba la caja.
  5. Cuenta las comprobaciones y para si alguna falla.
  6. Muestra los niveles, la comparación y el examen.
  7. Dibuja la adquisición mes a mes desde enero de hace 3 años, siempre. Un panel con el total y la suma de cada año (vendido + simulado frente al objetivo) y otro con una línea por plazo (1, 2 y 3 años). Real en línea continua, estimado en punteada, simulado en discontinua.
- **Comprobaciones (2, o 3 con objetivos):**
  1. La captación simulada está entre 0,5 y 2 veces lo vendido en los mismos meses un año antes (aviso).
  2. Con objetivos: cada región acaba el año a menos del 1 % de su objetivo (aviso). Avisa también de las regiones del objetivo que no están en los datos (un nombre que revisar) y de las que tienen hueco pero nada que escalar.
  3. El examen simula solo con meses anteriores a su origen.

## Bloque 6 · Time series (paso 6.1)

**Misión:** lo que el universo de retail a suscripción trae todavía este año, el año siguiente entero y la pipeline del año siguiente que genera, por regiones y con banda.

**Mecanismo:** el universo solo trae su resultado (lo que convirtió), así que se predice con su propia serie: el mismo mes del año anterior × el nivel de la región, valorado al valor por unidad de sus últimos meses; la regla del nivel y la ventana del valor por unidad son las que mejor simularon los meses cerrados (el examen compara 4 reglas × 3 ventanas sobre los mismos orígenes); el año siguiente repite este por defecto; lo proyectado este año vence el siguiente como pipeline (40 % de descuento, renueva al 100 %, a la tasa de su región); la simulación se examina en los meses cerrados y su error por horizonte es la banda de cada fila. Menos ciencia que los bloques 2 a 5, a propósito: una pieza que encaja con el resto y se mejora sola.

**Lee:** `time_series_rows`, `fine_table` (bloque 1). · **Entrega:** `time_series_universe`.

### 6.1 · Universo time series (`TimeSeriesUniverse`)

Simula, por regiones, el revenue de retail a suscripción que el extracto aún no tiene: lo que queda del año, el año siguiente entero y la pipeline del año siguiente que crea (la reentrada de lo proyectado), con una banda que sale del examen de la simulación en los meses cerrados.

- **Entra:** `time_series_rows` (1.2) y `fine_table` (1.4: la tasa de cada región en la pipeline normal, para la reentrada).
- **Sale:** `time_series_universe`: la caja, con sus filas región × mes (`ts_real`, `ts_proyectado`, `ts_simulado`, `ts_reentrada`, cada una con su banda), la regla del nivel y la ventana del valor por unidad elegidas y el examen de su simulación. Escribe `sff_ts_examen` (candidato — regla del nivel × ventana del valor por unidad — × horizonte: error del total simulado y candidato elegido); sus filas llegan al núcleo (8.3) y al total (7.3); el gráfico `ts_6_1_prediccion` sale en pantalla.
- **Depende de:** 1.2 (las filas time series) y 1.4 (la tasa de las regiones).
- **Acciones:**
  1. Las filas time series por región y mes.
  2. Este año: el nivel (con la regla elegida) y el valor por unidad de cada región, y los meses proyectados (por cuota del total donde una región no se puede proyectar).
  3. El año siguiente: los meses simulados (mismo mes de este año × `ts_next_year_level`, 1,0 por defecto) y la reentrada de lo proyectado (valor / (1 − 0,4) × tasa de la región).
  4. El examen de la simulación en los 6 últimos meses cerrados, con cada regla del nivel (`region`: el de la región; `credibilidad`: el de la región movido hacia el del total; `total`; `sin_nivel`: el mismo mes del año anterior tal cual), combinada con cada ventana del valor por unidad (1, 3 y 12 meses). Simula el candidato de menor error medio (`ts_level_rule` y ts\_auv\_window\_months fijan una parte) y su error por horizonte es la banda de cada fila. Desde el 26-oct, porque con el valor por unidad cayendo la media de 12 meses hacía saltar la proyección. En pantalla sale también el valor por unidad por región (último mes, 3 y 12 meses, hace un año) y si su cambio es precio dentro de las regiones o mezcla de regiones.
  5. Comprueba el universo.
  6. Escribe el examen de la simulación.
  7. Cuenta las comprobaciones y para si alguna falla.
  8. Muestra el universo, el examen de cada regla y su gráfico con la predicción (el mismo que dibuja la 1.2 con la historia).
- **Comprobaciones (5; sin filas time series no se evalúan):**
  1. Ninguna fila `ts_reentrada` sale de un mes `ts_real` (esas conversiones ya están en el extracto: contarían dos veces).
  2. El valor por unidad implícito de los meses proyectados está dentro de la historia de la región (aviso).
  3. Toda región con reentrada tiene la tasa de algún nivel de región (aviso si toma la global).
  4. El examen de la simulación lee solo meses anteriores a sus orígenes (con todas las reglas).
  5. `sff_ts_examen` se escribe y se relee.

## Bloque 7 · Examen y montaje (pasos 7.1 a 7.3)

**Misión:** un único forecast en dinero con todo lo que predicen los bloques 2 a 6, con una banda oficial medida en meses cerrados y un total que es la misma respuesta se sume como se sume.

**Mecanismo:** primero el examen: cada serie se predice en los meses de examen con solo lo que se sabía h meses antes, en las condiciones del forecast (la misma caja de la tasa, la misma regla de meses mínimos, los estados también predichos), frente a la baseline que haría cualquiera. Después el montaje: cada fila de pipeline (del extracto y del bloque 5) pide su tasa al bloque 2, su uplift al 3 y sus estados al 4, y se valora; se suman las filas del bloque 6. La banda oficial reúne todos los errores medidos: tasa (7.1) y precio (3.3) en cuadratura, más la captación (5.1) y el time series (6.1). Por último, el total por año y origen, comprobado contra el forecast.

**Lee:** los bloques 1 a 6. · **Entrega:** `series_exam`, `forecast` (filas, mes, año, maduración), `time_series`, `forecast_total`.

### 7.1 · Examen de cada forecast serie

Predice cada serie en los meses de examen con solo lo que se sabía h meses antes, de cuatro formas sobre las mismas filas: la baseline (celda mandatory, últimos 3 meses), la serie sola (raw), la escalera con la técnica por defecto y el framework. Añade el examen honesto: los estados del cliente también predichos con la caja 4.1. Su error del total por horizonte es la banda oficial del forecast.

- **Entra:** `rated_units` (1.7), `series_estimate` (2.3), `pool_series` (2.4), `backtest` (2.6), `ladder` (2.2), `fine_table` (1.4) y `rate_box` (2.7). Para el examen honesto construye la caja de los estados (la clase de la 4.1) en cada origen, con los meses anteriores a él.
- **Sale:** `series_exam`: cada predicción trazada, el examen por serie y método, el error del total por mes y el resumen por método y horizonte. Escribe `sff_series_exam_detail`, `sff_series_exam`, `sff_examen_cartera` y `sff_examen_cartera_resumen`.
- **Depende de:** 2.7 (la misma caja que usa el forecast). De la 4.1 usa el código, no su salida: en el grafo no la lee.
- **Acciones:**
  1. Fija el examen: calendario y forecast series con algo que vence en los meses de examen.
  2. Predice cada serie, mes y horizonte con los cuatro métodos, con el intervalo de raw, por defecto y framework.
  3. Calcula los errores por serie y del total por mes.
  4. Comprueba el examen.
  5. Escribe el examen.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra los métodos frente a frente.
- **Comprobaciones (8):**
  1. Ninguna predicción usó un mes posterior a su origen.
  2. Cada método predijo cada serie de cada mes de examen.
  3. Las renovaciones reales del examen son las de los meses de examen (nada contado dos veces).
  4. El resumen por serie suma el detalle (recuentos y unidades).
  5. a 8. Las cuatro tablas se escriben y se releen.

### 7.2 · Forecast

Monta el forecast en dinero. Cada fila futura (del extracto y de la proyectada que construye este paso y de la adquisición simulada de la caja 5.1) toma su tasa (caja 2.7), su uplift (3.1, con el veredicto de la 3.3) y sus estados al vencer (caja 4.1), y se valora: unidades × tasa, USD × tasa × uplift. Después suma por mes y año con la banda oficial, que reúne todos los errores medidos.

- **Entra:**
  - `fine_table` (1.4): las filas futuras;
  - `forecast_units` (1.5): las unidades de cada unit, para la banda;
  - `rate_box` (2.7): la tasa;
  - `uplift_cells`, `uplift_state_factors` y `price_increases` (3.1), con `uplift_verdict` (3.3): el uplift;
  - `state_maturation` (4.1): los estados al vencer;
  - `future_pipeline` (5.1): las filas simuladas de 1 año de la ventana y el examen de la adquisición (su error, en la banda);
  - `time_series_universe` (6.1): el universo time series y su banda;
  - `series_exam` (7.1): el error de la tasa por horizonte;
  - `pool_reference` (2.4) y `backtest` (2.6): la confianza de cada fila.
- **También declara** `series_estimate` (2.3), `pool_series` (2.4) y `rated_units` (1.7). Solo los usaría si no le llegara la caja 2.7; sobran en el grafo.
- **Sale:** `forecast`: cada fila futura con tasa, uplift, esperado (con los estados de hoy y al vencer), bandas y confianza (y, desde el 26-oct, el error del volumen de cada adquisición simulada, que llega al núcleo como s7\_2\_error\_captacion\_usd); los totales por mes y año; la maduración mes a mes. Escribe `sff_forecast_mes`, `sff_resumen_negocio`, sff\_curva\_maduracion y `sff_forecast_maduracion`.
- **Depende de:** 2.7, 3.1, 3.3, 4.1, 5.1, 6.1 y 7.1.
- **Acciones:**
  1. Toma las filas futuras del extracto.
  2. Pide la tasa de cada fila a la caja, con su técnica y la misma regla que el examen: una composición con menos de 3 meses cerrados no predice (sus filas toman su celda).
  3. Registra de dónde sale cada tasa (pool, celda o global) y su banda.
  4. Asigna el uplift (contrato o estadístico) y su banda.
  5. Calcula las renovaciones esperadas con los estados de hoy, fila a fila.
  6. Valora los estados al vencer de las filas del extracto (caja 4.1).
  7. Proyecta la renovación de 1 año de la pipeline real que vence en la ventana (mes en curso → diciembre): lo que el forecast espera que renueve vuelve a vencer el mismo mes del año siguiente (filas proyectadas, sin descuento y con las dims que cambia la renovación). Hasta el 27-oct-2026 lo hacía la caja 5.1. Las junta con las filas simuladas de la caja 5.1, las predice y las madura; asigna la confianza de cada fila.
  8. Suma por mes y año, por origen de la pipeline, y añade el universo time series de la caja 6.1 (sus ingresos simulados, la pipeline de su reentrada y su renovación esperada). La banda oficial, centrada en `total_usd`, reúne todo error medido en meses cerrados: √(tasa² + precio²) de la pipeline (la tasa del 7.1, y una renovación proyectada suma el error del mes en que se predijo su volumen; el precio del 3.3) + el volumen de la captación simulada (5.1) + el universo time series (6.1). Por año, los errores de sus meses sumados.
  9. Comprueba el forecast.
  10. Escribe los meses, el resumen y la maduración.
  11. Cuenta las comprobaciones y para si alguna falla.
  12. Muestra el forecast por mes, por año, la maduración y el detector de estados por encima de su proporción final.
- **Comprobaciones (11):**
  1. Toda fila futura tiene tasa, uplift y esperado.
  2. Toda tasa y toda banda están en \[0, 1\], y la banda contiene la tasa.
  3. No se predice ninguna fila cerrada; se predice toda fila futura del extracto.
  4. La pipeline de las filas futuras del extracto se conserva (Σ USD que vence).
  5. Los totales son la suma de las filas.
  6. Todo valor que toma una renovación proyectada al renovar existe en la historia, así que su serie de retención se encuentra (aviso; antes estaba en el 5.1).
  7. La maduración mueve unidades, no las crea: Σ al vencer = Σ que vence, en unidades y USD, y una fila conserva entre 0 y todas sus unidades.
  8. Ningún estado negativo está HOY 2 pp o más por encima de su proporción final en ningún mes futuro (aviso: el extracto lo pone antes de tiempo).
  9. a 11. `sff_forecast_mes`, `sff_resumen_negocio` y `sff_forecast_maduracion` (y sff\_curva\_maduracion) se escriben y se releen.

### 7.3 · Total por año y origen

Da el total de cada año separable por origen, con el universo time\_series dentro, y comprueba que es la misma respuesta que el total del 7.2.

- **Entra:** `time_series_universe` (6.1), `fine_table` (1.4) y `forecast` (7.2).
- **Sale:** `time_series` (las filas región × mes de la caja, para el núcleo) y `forecast_total` (año × origen: renovado real, esperado de la pipeline real, proyectada, simulada, `ts_real`, `ts_proyectado`, `ts_simulado` y `ts_reentrada`, y el TOTAL). Escribe `sff_forecast_total`.
- **Depende de:** 6.1 (el universo), 7.2 (el forecast y su total por año) y 1.4 (las renovaciones contabilizadas).
- **Acciones:**
  1. Resume las filas del universo time\_series por origen y año.
  2. Suma por año y origen.
  3. Comprueba el total.
  4. Escribe el total.
  5. Cuenta las comprobaciones y para si alguna falla.
  6. Muestra el total por año y origen.
- **Comprobaciones (3):**
  1. Los orígenes de cada año suman su total.
  2. El TOTAL de cada año es el `total_usd` del 7.2 por año: la misma respuesta.
  3. `sff_forecast_total` se escribe y se relee.

## Bloque 8 · Salida y control (pasos 8.1 a 8.6)

**Misión:** dejar el forecast donde lo leen negocio y el analista (núcleo y satélites en SQL, Power BI, informe), de forma que cada número se pueda seguir hasta el paso y la fila de la que sale. No cambia ningún número.

**Mecanismo:** una tabla ancha (el núcleo) con todas las filas y cada decisión como columna, con el prefijo del paso que la tomó; sus sumas SON las respuestas de negocio; satélites relacionados por ids; las tablas auxiliares de Power BI y las relaciones del modelo, comprobadas sobre los datos; una validación final de la cadena; el informe. Cada salida se comprueba contra el paso del que sale.

**Lee:** todo lo de los bloques 1 a 7. · **Entrega:** `sff_nucleo`, `sff_forecast_series`, satélites, tablas de Power BI (con `sff_relaciones_modelo`), validación e informe.

### 8.1 · Filas que el framework añade o borra

Pone una al lado de otra, mes a mes, todas las filas que el framework añade o borra: los huecos de la 1.7, lo que borra la 1.3 porque aún no se conoce y lo que crea la 7.2 para sustituirlo.

- **Entra:** `fine_table` (1.4), `rated_units` (1.7) y `forecast` (7.2).
- **Sale:** `new_rows`: una fila por mes × origen (filas, series, unidades, USD, USD esperado). No se escribe: su SQL sobre `sff_nucleo` lo reproduce.
- **Depende de:** 1.3 (lo borrado, en las columnas `s0_`), 1.7 (los huecos) y 7.2 (lo creado).
- **Acciones:**
  1. Toma los huecos de la 1.7.
  2. Toma lo que borró la 1.3.
  3. Toma lo que creó la 7.2.
  4. Construye la tabla mes × origen y los totales por origen.
  5. Comprueba que el extracto = lo que se queda + lo borrado.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra los totales por origen, los meses con actividad y la SQL que los reproduce.
- **Comprobaciones (1):**
  1. Lo que el extracto tenía por vencer = lo que sigue venciendo + la pipeline parcial borrada (unidades y USD).

### 8.2 · Auditoría

Construye los satélites del núcleo para el análisis en profundidad: cada id de cada etapa de la escalera, cada técnica de cada composición, cada referencia de credibilidad con sus miembros, la dinámica de cada serie y la tasa futura de cada técnica. Cada tabla se comprueba contra la que la origina.

- **Entra:** `ladder` (2.2), `series_rate` (1.7), `series_estimate` (2.3), `rated_units` (1.7), `pool_series` (2.4), `pool_reference` (2.4), `pool_dynamics` (2.5), `backtest` (2.6), `forecast` (7.2) y `rate_box` (2.7).
- **Sale:** `audit`, con seis tablas: `sff_composition`, `sff_composition_techniques`, `sff_credibility`, `sff_credibility_members`, `sff_series_dynamics` y `sff_composition_forecast_all`.
- **Depende de:** 2.2 a 2.7 (lo que audita) y 7.2 (para comprobar que la tasa usada es la de la técnica elegida).
- **Acciones:**
  1. `sff_composition`: cada id de cada etapa.
  2. `sff_composition_techniques`: cada técnica de cada composición.
  3. `sff_credibility` y sus miembros.
  4. `sff_series_dynamics`.
  5. `sff_composition_forecast_all`, con la historia de cada composición que da la caja 2.7.
  6. Escribe las tablas.
  7. Cuenta las comprobaciones y para si alguna falla.
- **Comprobaciones (12):**
  1. Cada etapa de la escalera es una partición: las unidades que vencen de sus ids suman las del extracto.
  2. Cada etapa contiene cada serie estimable una vez.
  3. Cada composición y tramo tiene exactamente una técnica elegida, la de la 2.6.
  4. Soporte, tasa y series de cada referencia, recalculados desde sus miembros, son los usados.
  5. Una fila de dinámica por forecast serie estimable.
  6. La técnica elegida da la tasa que usó la 7.2 (filas sin desplazamiento de credibilidad).
  7. a 12. Las seis tablas se escriben y se releen.

### 8.3 · El núcleo (`sff_nucleo`)

Pone toda la historia de cada fila en una tabla ancha, de izquierda a derecha en el orden de los pasos, con los huecos como filas explícitas. El prefijo de cada columna (`s1_4_`, `s7_2_`…) es el paso que la crea. Una leyenda dice cómo se agrega cada columna.

- **Entra:** `fine_table` (1.4), `forecast_units` (1.5), `rated_units` (1.7), `series_table` (1.6), `series_rate` (1.7), `series_estimate` (2.3), `pool_dynamics` (2.5), `backtest` (2.6), `forecast` (7.2), `time_series_rows` (1.2), `time_series` y `forecast_total` (7.3), `ladder` (2.2), `audit` (8.2) y `series_exam` (7.1).
- **Sale:** `core` (una fila por fila fina del extracto, hueco, fila de la proyectada o simulada, fila de predicción de los estados y fila time\_series), `core_legend` y `forecast_series` (la dimensión: una fila por serie). Escribe `sff_nucleo`, `sff_forecast_series` y `sff_nucleo_leyenda`.
- **Depende de:** casi todas; en especial 7.2 y 7.3, porque el total del forecast se comprueba como suma del núcleo.
- **Acciones:**
  1. Toma las filas del extracto con sus medidas `s1_1_` y `s1_3_` y sus ids.
  2. Añade las filas hueco de la 1.7.
  3. Añade los valores de la forecast unit (1.5).
  4. Añade las filas del forecast (7.2), el bloque final `forecast_*` y la dimensión de series.
  5. Construye la leyenda: cada columna, su paso, su nivel y cómo agregarla.
  6. Comprueba el núcleo contra el extracto y la leyenda.
  7. Escribe el núcleo y su leyenda, por tramos de 100.000 filas para no duplicar el núcleo en memoria.
  8. Cuenta las comprobaciones y para si alguna falla.
  9. Muestra una serie tal como queda en el núcleo y cómo leerla en Power BI.
- **Comprobaciones (10):**
  1. Todo registro original y toda fila sintética, una vez.
  2. El dinero cuadra con el extracto.
  3. Toda fila tiene los valores de cada bloque presente.
  4. La leyenda describe cada columna del núcleo.
  5. El núcleo sumado por año y origen = `sff_forecast_total`.
  6. La dimensión tiene una fila por forecast serie del núcleo.
  7. Las sumas del examen en la dimensión son las de la 7.1 (cada serie una vez).
  8. a 10. `sff_nucleo`, `sff_forecast_series` y `sff_nucleo_leyenda` se escriben y se releen.

### 8.4 · Power BI

Construye las tablas auxiliares del informe de Power BI, para que el informe solo relacione y sume, sin lógica dentro.

- **Entra:** `core` (8.3).
- **Sale:** `power_bi`, con cinco tablas: `sff_forecast_pipeline_source` (código, etiqueta, bloque y orden de cada origen), `sff_price_increase_monitor` (grupo de precio × mes cerrado), `sff_price_increase_calendar` (una fila por subida), `sff_parametros` (una fila por campo de la Config, con los pasos que lo leen) y `sff_relaciones_modelo` (una fila por relación del modelo de Power BI, comprobada sobre los datos).
- **Depende de:** 8.3 (el núcleo) y las tablas que escribió la ejecución (para las relaciones).
- **Acciones:**
  1. Construye la tabla de orígenes de la pipeline.
  2. Construye el monitor de subidas de precio: el uplift de los meses cerrados y sus escalones.
  3. Construye la tabla de parámetros de la ejecución.
  4. Comprueba las tablas contra el núcleo.
  5. Las escribe.
  6. Las relaciones del modelo de Power BI (`MODEL_RELATIONS`), comprobadas sobre las tablas de la ejecución.
  7. Cuenta las comprobaciones y para si alguna falla.
  8. Muestra las tablas.
- **Comprobaciones (12):**
  1. Todo `forecast_pipeline_source` del núcleo tiene su fila.
  2. Una fila por origen (la relación en Power BI es 1 → n).
  3. Todo mes cerrado de pipeline con renovaciones tiene su uplift en el monitor.
  4. Una subida solo se marca contra una referencia de al menos 6 meses anteriores.
  5. a 8. Las cuatro tablas se escriben y se releen.
  6. Toda relación del modelo une columnas de tablas que la ejecución escribió (aviso).
  7. La clave del lado 1 de cada relación es única (si no, Power BI no puede hacerla 1 → n).
  8. Toda fila del lado n encuentra su clave en el lado 1 (aviso: si no, queda en blanco en Power BI; las series que no suben la escalera no tienen composición; las filas agregadas de la maduración y las extendidas con una celda de uplift nunca vista no tienen fila en sff\_uplift\_celda, y su columna s3\_1\_origen\_uplift dice de dónde sale su uplift).
  9. `sff_relaciones_modelo` se escribe y se relee.

### 8.5 · Validación de la cadena

Comprueba, cuando todo ha corrido, que la cadena cuadra del extracto al forecast: el dinero se conserva, toda serie y toda fila futura están cubiertas, y el forecast es coherente con el pasado y con el examen.

- **Entra:** el extracto (`extract_summary`), `backtest` (2.6), `fine_table` (1.4), `forecast` (7.2), `forecast_units` (1.5), `pool_reference` (2.4), `series_exam` (7.1), `series_estimate` (2.3), `time_series_rows` (1.2) y `core` (8.3).
- **Sale:** `validation` (una fila por comprobación). Escribe `sff_validacion`.
- **Depende de:** 7.2, 7.1 y 8.3 (lo que valida).
- **Acciones:**
  1. Sigue el dinero a lo largo de la cadena.
  2. Comprueba la cobertura de las series y las filas del forecast.
  3. Comprueba la coherencia del forecast con el pasado y con el examen.
  4. Escribe la validación.
  5. Cuenta las comprobaciones y para si alguna falla.
- **Comprobaciones (7):**
  1. Σ USD que vence: extracto (sin time\_series ni la pipeline aún desconocida) = tabla fina = forecast units.
  2. El USD futuro que vence en el extracto = Σ USD que vence de las filas del extracto en el forecast.
  3. Toda serie con dinero por predecir tiene tasa y nivel de riesgo.
  4. Toda fila futura de una serie con composición (con al menos 3 meses cerrados) toma su tasa del pool.
  5. La tasa esperada del futuro está a ±15 pp de la del último año cerrado (aviso).
  6. El error del total en el examen está dentro de ±10 % (aviso).
  7. `sff_validacion` se escribe y se relee.

### 8.6 · Informe, veredicto y perfil

Cuenta con los números de la ejecución qué son los datos, qué hipótesis del framework se sostienen en ellos y lo bien que predice. El informe tiene un capítulo por bloque, precedido del veredicto, y abre con el resumen para negocio, que incluye la banda desglosada y lo que hay que validar. Escribe el veredicto de cada bloque y el perfil de la pipeline como tablas.

- **Entra:** prácticamente todo lo de los bloques 1 a 8, y los veredictos de los bloques 1 a 7 (`configuration.block_verdicts`) con el perfil del bloque 1 (`configuration.data_profile`).
- **Sale:** `report` y el fichero `informe_sff.md`, con los gráficos en `salida/graficos/`. Escribe `sff_veredicto_bloques` (bloque × pregunta) y `sff_perfil_datos` (cartera, mes, dimensión, concentración, soporte, historia). Lo último que enseña es la tabla final: el año por región, leída de sff\_nucleo con su SQL.
- **Depende de:** todas; corre la última.
- **Acciones:**
  1. Calcula el veredicto del bloque 8 y escribe los nueve capítulos: 0 el veredicto; 1 a 8, uno por bloque. Añade el resumen para negocio y los gráficos.
  2. Comprueba el informe.
  3. Escribe el fichero.
  4. Escribe el veredicto de cada bloque y el perfil de la pipeline.
  5. Lee de sff\_nucleo, con su SQL, el año en curso por tr\_regional\_level\_1 y tr\_regional\_level\_3: el USD renovado real hasta el último mes cerrado y el esperado hasta diciembre, con el TOTAL de cada región y del año.
  6. Cuenta las comprobaciones y para si alguna falla.
  7. Muestra las cifras del titular y el resumen para negocio.
  8. Muestra la tabla final, el año por región, y debajo el SQL que la da.
- **Comprobaciones (4):**
  1. El informe tiene sus 9 capítulos.
  2. `sff_veredicto_bloques` se escribe y se relee.
  3. `sff_perfil_datos` se escribe y se relee.
  4. El TOTAL del año por región es el TOTAL del año en sff\_forecast\_total (7.3).
