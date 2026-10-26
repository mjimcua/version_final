# Revisión del framework SFF con veredicto (25-oct-2026)

Revisión de los 8 bloques sobre la última ejecución del sintético parecido al real, con el universo time series:

- 385.154 filas;
- mes en curso 2026-10;
- examen de 2026-04 a 2026-09.

Cubre las comprobaciones, el resultado de cada acción y el análisis, y termina con un veredicto por bloque. Las cifras salen de esa ejecución. El veredicto de cada hipótesis queda ahora en la tabla `sff_veredicto_bloques`, y el perfil de la pipeline en `sff_perfil_datos`.

## Conclusión

1. **El framework está bien construido como cadena. Como modelo, en el sintético gana a 6 meses; a 1 mes no lo demuestra frente a la baseline, ni en unidades ni en dinero.**
   - **Cadena:** las 209 comprobaciones pasan (3 avisos explicados), el dinero se conserva del extracto al núcleo y las 15 relaciones de Power BI se pueden hacer.
   - **Frente a la baseline, en unidades renovadas del total del mes:**
     - a 6 meses gana: 1,4 % frente a 2,8 %, 6 de 6 meses, p = 0,03;
     - a 1 mes no es significativo: 1,0 % frente a 1,1 %, 4 de 6 meses, p = 0,69.
   - **Frente a la baseline, en el USD renovado del mes:**
     - a 1 mes empata: 1,2 % frente a 1,2 %, 3 de 6 meses, con 991 $ más de error medio al mes;
     - a 6 meses gana: 1,4 % frente a 2,3 %, 5 de 6 meses, 20 k$ menos de error al mes (test de signos p = 0,22; t pareado p = 0,07).
   - **Frente a la serie sola, en el total:** gana a los dos horizontes (5 de 6 con t pareado p = 0,01 a 1 mes; 6 de 6 a 6 meses).
2. **Tres piezas no demuestran su valor en el sintético y son las primeras candidatas a simplificar si tampoco lo demuestran con datos reales.**
   - **La selección de técnica, pool a pool.** En el examen 2.6, ponderado por el dinero:
     - a corto plazo la técnica elegida yerra más que la por defecto (4,35 pp frente a 4,20 pp, significativo: no se cumple);
     - a medio, apenas menos (4,74 frente a 4,91 pp, p = 0,25: no concluyente).
     - En el total del examen 7.1 sí mejora a la escalera con la técnica por defecto a 6 meses (1,4 % frente a 1,7 %, 5 de 6, t pareado p = 0,04), y no a 1 mes (2 de 6). Sin ponderar por dinero, la diferencia por pool es aún menor (−0,14 pp y +0,05 pp).
   - **El precio por estado del cliente:** el examen no lo elige.
   - **La regla del contrato:** su WAPE es del 21 %, frente al 7 % de la vía estadística.
3. **La escalera conoce mejor la tasa media, pero no predice mejor cada serie.**
   - El dinero con la tasa conocida a ±5 pp sube del 64 % al 78 %.
   - En el examen serie a serie, la serie sola yerra menos que el framework (WAPE 10,1 % frente a 10,7 % a 1 mes; 10,5 % frente a 11,1 % a 6 meses).
   - El framework gana en el total (1,0 % frente a 1,4 %), porque la serie sola arrastra más sesgo.
   - Es la hipótesis central del framework y la primera que hay que mirar con datos reales.
4. **El error del total del examen es casi todo sesgo**, no ruido ni falta de técnica.
   - La tasa de los meses de examen (66,3 %) está 2,5 pp por encima de la de los 12 meses anteriores (63,8 %).
   - El sesgo del framework es −0,9 % a 1 mes y −1,3 % a 6 meses, de un error del 1,0 % y del 1,4 %.
   - A 6 meses todo método que aprende del pasado se queda corto. A 1 mes no: la escalera con la técnica por defecto tiene un sesgo de solo −0,14 %.
   - Más técnicas no lo arreglan. Lo arregla detectar antes los cambios de nivel o corregirlos.
5. **El mayor ajuste de modelo es la maduración de los estados del cliente: −5,1 % en 2027 (−1,34 M$).**
   - Su curva acierta la cuota final de softcancel (error ≤ 0,09 pp, frente a 7,15 pp sin curva).
   - En el total del examen no mejora (1,5 % frente a 1,0 % a 6 meses con los estados vistos sin curva). La causa es que, sin curva, los sesgos se compensan:
     - el techo (la tasa con los estados reales) se queda corto: −0,87 % a 1 mes y −1,33 % a 6 meses;
     - el examen honesto (con los estados de la curva) queda casi igual: −0,96 % y −1,53 %, así que la curva añade poco error;
     - con los estados vistos sin curva, el sesgo es +0,25 % y +0,74 %: sobreestiman la renovación, y ese error tapa el sesgo negativo de la tasa.
   - Es decir, los estados vistos parecen mejores porque su error compensa el de la tasa, no porque la curva sobre.
   - dormant y not_installed no traen fecha: su maduración no tiene examen.
6. **El universo time series es el 4 % del dinero de 2027 y el 46 % de su banda.**
   - Se corrigió la regla del nivel (el examen elige ahora entre cuatro), pero su error sigue en el 42 % del total del mes, y en el 53 % a 1 mes.
   - 4 regiones no tienen historia propia suficiente y se proyectan por su cuota.
   - Mejorarlo es lo que más estrecharía la banda oficial.
7. **La captación Not-New se simula peor que "lo mismo que hace un año"** (20,4 % frente a 10,8 %), con un sesgo que crece con el horizonte (−11 % a −35 %). La simulación de Pure-New sí es buena (1,1 % frente a 7,0 %).
8. **Lo que funciona y no hay que tocar:**
   - el examen sin mirar el futuro;
   - la caja única de la tasa;
   - la banda de la técnica (91 % dentro de la del 90 %);
   - la etiqueta de confianza: el WAPE de las series en el examen es 3,6 % en alta, 6,7 % en media y 21,2 % en baja, así que la etiqueta está ordenada. Ojo: la confianza baja lleva el 44 % del USD esperado;
   - la conservación del dinero y la trazabilidad en Power BI.

La observación de partida (las ganancias frente al sintético no son significativas) queda confirmada, y ahora se mide en cada ejecución. Cada hipótesis tiene su fila en `sff_veredicto_bloques`, con su métrica, su referencia y un veredicto: se cumple, no se cumple, no concluyente o revisar.

## Qué cambia en las salidas

- **Veredicto al cerrar cada bloque.** En pantalla, una tabla con tipo, paso, pregunta, valor, referencia y veredicto, más una línea por cada hipótesis que no se cumple o hay que mirar. En total son 55 preguntas por ejecución (módulo `verdicts.py`); en el parecido al real: 16 se cumplen, 10 no se cumplen, 2 no son concluyentes, 11 hay que revisarlas y 16 son datos. De las 29 hipótesis, se cumplen 15.
- **Perfil de la pipeline al cerrar el bloque 1** (`sff_perfil_datos`, 6 secciones):
  - la cartera;
  - el volumen mes a mes;
  - cada valor de cada dimensión: peso en el dinero a predecir, tasa, crecimiento y cuota con soporte;
  - la concentración;
  - el soporte propio;
  - la historia propia.
- **El informe (8.6) pasa de 8 capítulos en el orden antiguo a 9:**
  - el veredicto, con la banda desglosada;
  - un capítulo por bloque, con capítulos nuevos para estados del cliente, pipeline futura y time series.
- **La banda por componentes** (tasa, precio, tasa y precio en cuadratura, captación y time series, por año) está en `sff_resumen_negocio`, en el resumen para negocio ("LA BANDA") y en el capítulo 0.
- **El resumen para negocio** añade "LO QUE HAY QUE VALIDAR": cuántas hipótesis se cumplen y cuáles no.
- **Pantalla más corta:**
  - 8.4 muestra solo los parámetros cambiados (31), no los 165;
  - 3.2 muestra 2 series de 24 meses, no 5 de 36;
  - las tablas de maduración que calcula el 7.2 salen con la etiqueta [7.2], no [4.1].
- **Nuevas columnas y tablas:**
  - `s3_1_origen_uplift`: de dónde sale el uplift de cada fila;
  - `sff_ts_examen`: el examen de las reglas del nivel time series;
  - los componentes de la banda en `sff_resumen_negocio`.

## Bloque a bloque

### B1 · Preparación (1.1–1.7)

**Comprobaciones.** 86 comprobaciones, con 0 fallos y 2 avisos:

- 1.2: 5 filas time series sin net_new;
- 1.6: 10 series con meses futuros a 0 unidades.

Cubren bien la estructura (roles y calendario), el dinero (nulos, negativos, renovado ≤ vence, isolated y sus tramos) y la coherencia de estados y maduración. Son la parte más sólida del framework.

Falta una comprobación de **cambio de volumen**: que la pipeline futura o la de un mes no se aleje de la del año anterior más de un umbral por dimensión. Hoy esa información está en el perfil, pero no avisa.

**Resultado de las acciones:**
- 14.676 forecast series, de las que 7.681 tienen dinero a predecir: 34,6 M$.
- La concentración es fuerte: **21 series hacen el 50 % del dinero y 181 el 80 %**.
- El 59 % del dinero está en 51 series con ≥ 271 unidades al mes, y el 91 % en series con ≥ 36 meses de historia.
- La pipeline de los próximos 12 meses es un 3,5 % menor que la de los 12 cerrados, y la de los 12 cerrados un 6,3 % menor que la del año anterior.
- El **59 % del dinero no conoce su descuento**.
- tr_term y tr_band no separan la tasa por encima del ruido: cada una queda en un solo grupo de nivel 1.

**Análisis.**
- **Concentración.** Con esta concentración, la precisión del total la deciden unas decenas de series grandes con soporte propio.
- **Alcance de la escalera.** La escalera no solo actúa sobre las series bajo el suelo. Mezcla la serie con su referencia por credibilidad en todo lo que no es tasa propia pura (A_propio): el 41 % del dinero.
- **Variación de la tasa.** La tasa mensual de la cartera varía ±2,1 pp, con un ruido binomial de ±0,17 pp. Ese movimiento es mezcla y nivel, no azar.

**Veredicto.** Se cumple: el dato está verificado y bien preparado. El perfil es lo primero que hay que leer con datos reales.

### B2 · Tasa (2.1–2.7)

**Comprobaciones.** 37, todas ok:
- las particiones de la escalera;
- los signos;
- tasas y z en [0, 1];
- el error de predicción ≥ el de estimación;
- el backtest sin mirar el futuro;
- la banda que cubre el ≥ 80 % (sale el 91 %).

Dos de ellas son débiles:
- **Banda de la técnica (2.6, check 5).** Exige el ≥ 80 % para una banda nominal del 90 %. El veredicto usa ya el mismo umbral (`min_exam_coverage`), para que la fila y la comprobación no se contradigan. Subiría ese umbral al 85 %.
- **Credibilidad (2.3).** No hay comprobación de que la k de credibilidad salga de los datos: **662 de 739 referencias usan la k por defecto (60)**, que son el 89 % del dinero de las series que se apoyan en su referencia. Ahora es una fila del veredicto.

**Resultado de las acciones:**
- **Ladder:** de 7.274 series estimables a 1.326 composiciones. 18.321 fusiones, el 65 % de las cuales baja el error de la serie.
- **Dynamics:** φ mediana de los pools de 2,25; el 93 % del dinero está en pools con φ > 1,5 y el 79 % con tendencia. La cartera no tiene efecto mes (p = 0,91).
- **Ranking de técnicas** en los meses de selección: theta (1,26 de error normalizado) por delante de ewma, ses y ma6. La por defecto, ma3, queda en 1,33.

**Análisis.**
- La tasa sí se mueve más allá del ruido, pero la ventaja de las técnicas en selección no pasa al examen pool a pool (ponderado por dinero):
  - corto: −0,15 pp (significativo, y peor);
  - medio: +0,17 pp (p = 0,25, no concluyente).
- En el total del examen 7.1 sí aparece a 6 meses (1,4 % frente a 1,7 %, t pareado p = 0,04). La ganancia está en el horizonte largo y en la suma, no en el pool.
- Es el resultado habitual cuando lo que mueve la tasa es un cambio de nivel que ninguna técnica anticipa.
- La k por defecto convierte la mezcla serie-referencia en un parámetro sin medir.

**Veredicto.**
- Se cumple que una serie sola no basta y que la escalera conoce mejor la tasa media.
- Pool a pool, que elegir técnica mejore a la por defecto no se cumple a corto plazo y no es concluyente a medio. En el total a 6 meses sí mejora.
- Con datos reales hay que mirar dos cosas:
  - el examen de la técnica: si sigue sin ganar, simplificar a una o dos técnicas;
  - la sensibilidad a k_cred (× 0,5 y × 2) en el examen 7.1.

### B3 · Precio (3.1–3.3)

**Comprobaciones.** 10, todas ok:
- toda celda tiene uplift, que está en [0,01, 3,0], con 5 recortados;
- la banda contiene el uplift;
- no hay escalón sin confirmar;
- la estimación se hace sin los meses de examen;
- las vías se comparan sobre las mismas filas.

El rango [0,01, 3,0] es muy ancho para un uplift (la mediana es 1,12). Un aviso por encima de 2 detectaría celdas raras sin parar la ejecución.

**Resultado de las acciones:**
- Uplift global 1,127 (ventana sin subidas). 16 subidas en juego y 1 con meses futuros dentro de su ciclo.
- La regla del contrato se realiza al 84 %: el uplift observado sobre el de la regla, ponderado por el USD renovado.
- En el examen, el WAPE donde se conoce el descuento es:
  - estadística: 7,2 %;
  - contrato: 21,4 % (sesgo +20,7 %);
  - contrato en las neutras: 19,8 %.
- El precio por estado no mejora (6,37 % frente a 6,36 %).
- El error del precio del total de un mes es del 1,1 %, sin sesgo.
- El 3.2 encuentra que la afirmación de que el precio se mantiene **no se cumple**: el 51 % del valor de las renovaciones isolated renueva un 10 % o más por encima.

**Análisis.**
- **La vía estadística funciona y el precio del total es preciso.**
- **Tres de las piezas sobran o están mal elegidas:**
  - **La regla del contrato.** Sobreestima un 20 %: los renovadores no llegan al MSRP del vencimiento.
  - **El precio por estado.** Sobra.
  - **La forma del uplift.** La fija la Config y no es la que mejor examina: "toda la historia" yerra 5,9 % frente al 6,4 % de "ventana sin subidas". Es una decisión que el examen podría tomar, como la vía, y por tipo de fila: "toda la historia" gana donde se conoce el descuento (5,4 % frente a 7,2 %) y pierde donde no se conoce (6,2 % frente a 5,7 %).
- **Una caída que no se vigila.** El uplift isolated cae un 6,2 % en 2026-09. El detector del 3.1 solo busca subidas, así que una bajada de precio pasa sin aviso.

**Veredicto.**
- Se cumple que el precio se predice bien, con un error del 1,1 % en el total.
- No se cumplen la regla del contrato ni el precio por estado.
- Recomendaciones:
  - elegir la forma del uplift por examen, por tipo de fila (descuento conocido o no);
  - que el detector avise también de las bajadas.

### B4 · Estados del cliente (4.1)

**Comprobaciones.** 2, ok: la caja aprende solo de meses cerrados y la curva sale de cohortes. Falta una comprobación de que la caja conserva unidades. Hoy la hace el 7.2 (check 6), que es correcto porque allí se aplica.

**Resultado de las acciones:**
- **Curva de softcancel:** al vencer la cuota es del 7,6 %. Con la curva, la cuota se yerra como mucho 0,09 pp; sin curva, con el estado visto entonces, hasta 7,15 pp.
- **Renovación por combinación de estados:** neutral 71 %, dormant 46 %, softcancel 31 % y not_installed 52 %. dormant+softcancel renueva un 7 %.

**Análisis.**
- La curva es buena para la cuota.
- El efecto en dinero, −5,1 % en 2027, es el ajuste más grande del forecast, y el examen honesto del 7.1 no lo confirma en el total. La causa no es la curva: con ella, el examen honesto queda casi en el techo (−1,53 % frente a −1,33 % a 6 meses). Es que, sin curva, el sesgo positivo de los estados vistos tapa el negativo de la tasa.
- dormant y not_installed (9 % y 3 % del USD que vence) no tienen fecha: su maduración es la proporción histórica, sin examen.

**Veredicto.**
- Se cumple que la curva predice la cuota de softcancel.
- Está sin validar el efecto en el total. Con datos reales:
  - examen honesto con sesgos comparados;
  - pedir la fecha de dormant al extracto.

### B5 · Pipeline futura (5.1)

**Comprobaciones.** 3, ok:
- el valor que toma una renovación proyectada existe en la historia;
- la simulación está entre 0,5× y 2× el año anterior (sale 0,91×);
- el examen se hace sin mirar el futuro.

El rango 0,5×–2× es ancho. La comprobación útil es la del examen, y ahora es una fila del veredicto.

**Resultado de las acciones:**
- 1.951 filas simuladas, 1,5 M$ que vencen.
- 28.478 filas proyectadas.
- Error del examen de la captación:

  | Valor | Simulación | Baseline (nivel 1) | Sesgo |
  |---|---|---|---|
  | Pure-New | 1,1 % | 7,0 % | — |
  | Not-New | 20,4 % | 10,8 % | de −11 % a −35 % |

**Análisis.** El nivel de los últimos 3 meses arrastra una caída temporal de Not-New a todo el horizonte.

**Veredicto.**
- Se cumple para Pure-New y no para Not-New.
- Recomendación: elegir el nivel de cada valor de captación por examen (nivel o nivel 1), igual que la regla del nivel del bloque 6.

### B6 · Time series (6.1)

**Comprobaciones.** 5, ok:
- ninguna reentrada sale de un mes real;
- el valor por unidad está dentro de la historia;
- la tasa de la región;
- el examen sin mirar el futuro;
- la tabla del examen.

**Resultado de las acciones:**
- Regla elegida: `sin_nivel`, con un error medio del 42 % (región 48 %, credibilidad 54 %, total 61 %); a 1 mes, el 53 %.
- 4 regiones se proyectan por su cuota, sin historia propia suficiente.
- 210 k$ proyectados en 2026; 743 k$ simulados y 224 k$ de reentrada en 2027.

**Análisis.**
- El universo del parecido al real es una muestra del 3 % de las filas, más ruidosa que una serie real de conversiones.
- Aun así, pesa el 46 % de la banda de 2027 con el 4 % del dinero.

**Veredicto.**
- El nivel ya no se dispara.
- Revisar con datos reales:
  - si el error sigue por encima del 25 %, el universo necesita información de las campañas;
  - si no, aceptar una banda ancha y decirlo en el informe (ya se dice).

### B7 · Examen y montaje (7.1–7.3)

**Comprobaciones.** 21, ok:
- el examen sin mirar el futuro;
- cada método predice cada serie;
- las renovaciones reales sin doble conteo;
- cada fila futura tiene tasa, uplift y esperado;
- las bandas en [0, 1];
- se conserva la pipeline;
- los totales son la suma de las filas;
- la maduración conserva unidades;
- el detector de estados;
- el total por año es la misma respuesta.

Son sólidas.

**Resultado de las acciones.** El examen del total del mes (mismas filas y meses):

| Comparación | h = 1 | h = 6 |
|---|---|---|
| framework vs baseline (unidades) | 1,0 % vs 1,1 % (4/6, p = 0,69) | 1,4 % vs 2,8 % (6/6, p = 0,03) |
| framework vs baseline (USD renovado) | 1,2 % vs 1,2 % (3/6, +991 $ de error al mes) | 1,4 % vs 2,3 % (5/6, p t = 0,07; −20 k$ de error al mes) |
| framework vs escalera con técnica por defecto | 1,0 % vs 0,9 % (2/6) | 1,4 % vs 1,7 % (5/6, p t = 0,04) |
| framework vs serie sola (total) | 1,0 % vs 1,4 % (5/6, p t = 0,01) | 1,4 % vs 1,8 % (6/6) |
| framework vs serie sola (WAPE serie a serie) | 10,7 % vs 10,1 % | 11,1 % vs 10,5 % |
| sesgo del framework | −0,9 % | −1,3 % |
| examen honesto vs estados vistos | 1,0 % vs 1,0 % | 1,5 % vs 1,0 % |
| sesgo: estados vistos / techo / honesto | +0,25 / −0,87 / −0,96 % | +0,74 / −1,33 / −1,53 % |

- La banda oficial cubre el 58 % de los meses de examen: es ±1 error típico, no un intervalo del 90 %.
- La maduración ajusta −1,5 % en 2026 y −5,1 % en 2027.
- La banda de 2027 es ±945 k$: tasa y precio 51 %, captación 4 %, time series 46 %.
- La confianza está ordenada: alta 3,6 %, media 6,7 %, baja 21,2 % de WAPE. La baja lleva el 44 % del USD esperado.

**Análisis.**
- La ganancia del framework está en el horizonte largo y en el total de unidades, no en la serie ni a un mes.
- En dinero la ventaja a 1 mes desaparece: el precio añade su propio error y la baseline no lo hace peor.
- El error que queda es el cambio de nivel del examen (+2,5 pp).

**Veredicto.**
- En unidades, se cumple a 6 meses y no es concluyente a 1 mes.
- En USD renovado, se cumple a 6 meses (por el t pareado) y no a 1 mes, donde empata.
- La banda está bien construida, pero hay que comunicarla como error típico. El informe y el resumen ya lo dicen.

### B8 · Salida y control (8.1–8.6)

**Comprobaciones.** 45, todas ok salvo un aviso esperado: las relaciones con filas huérfanas, explicadas por `s3_1_origen_uplift` y por las series sin composición.

Dos de la validación 8.5 son demasiado holgadas:
- **Tasa futura (check 5).** ±15 pp frente a la del último año. Compara la tasa en USD con uplift (77,3 % frente a 78,6 %). Con esa diferencia de 1,3 pp, un salto de 5 pp ya sería noticia: bajaría a ±5 pp o lo ligaría al ruido de la cartera.
- **Error del examen (check 6).** ±10 %, cuando el observado está entre el 1 y el 3 %. Lo ligaría a 3 × el error medio del examen.

**Resultado de las acciones:**
- Núcleo de 629.254 filas × 95 columnas (37 s de escritura) y 43 tablas escritas y releídas.
- En la base de pruebas quedan 23 tablas de una versión anterior que ya nadie escribe. El framework no las borra: habría que limpiarlas antes de conectar Power BI, o que el 8.5 avise de tablas `sff_` que la ejecución no escribió.
- 15 relaciones del modelo.
- El informe con su veredicto.

**Veredicto.** Se cumple: todo número se puede seguir hasta su paso y su fila.

## Plan de validación con datos reales

En orden, con la regla de decisión de cada punto (cada uno es una fila de `sff_veredicto_bloques`):

1. **Leer el perfil** (`sff_perfil_datos`): concentración, soporte propio, historia, descuento conocido, estados y crecimiento por dimensión. Decide dónde importa cada bloque. Con la concentración del sintético (181 series = 80 %), lo que pesa son las series grandes.
2. **Mirar el sesgo del total** en el examen y el cambio de nivel de la tasa en los meses de examen. Si el error es sobre todo sesgo, la prioridad es detectar o corregir cambios de nivel, no añadir técnicas.
3. **El framework frente a la baseline** a 1 y 6 meses, en unidades y en USD renovado, con el test de signos y el t pareado. Si no gana a ningún horizonte, la complejidad no se justifica. Si gana en unidades pero no en dinero, el problema está en el precio, no en la tasa.
4. **La escalera serie a serie** (WAPE frente a la serie sola). Si la serie sola sigue ganando, revisar el suelo de soporte (30), own_rate_floor (271) y la k de credibilidad, empezando por su sensibilidad.
5. **La técnica elegida frente a la por defecto** (examen 2.6 pool a pool y cascada 7.1 en el total). Si no mejora en ninguno de los dos, quedarse con una técnica o con la mejor del ranking en todos los pools.
6. **La maduración de los estados** con el examen honesto, comparando sesgos. Pedir la fecha de dormant.
7. **El precio:** la vía, la forma del uplift y el precio por estado. Elegir la forma por examen y por tipo de fila.
8. **La captación por valor de captación.** Elegir el nivel por examen.
9. **El universo time series:** su error y su peso en la banda.

## Comprobaciones que endurecería o añadiría

| Dónde | Hoy | Propuesta |
|---|---|---|
| 2.6 check 5 | banda ≥ 80 % (nominal 90 %) | ≥ 85 % |
| 3.1 check 2 | uplift en [0,01, 3,0] | aviso por encima de 2,0 |
| 5.1 check 2 | simulada entre 0,5× y 2× | dejarla y usar el veredicto del examen |
| 8.5 check 5 | tasa futura ±15 pp | ±5 pp o 3 × la desviación mensual de la cartera |
| 8.5 check 6 | error del examen ±10 % | 3 × el error medio del examen |
| B1 (nueva) | — | aviso de cambio de volumen por dimensión frente al año anterior |
| B2 (nueva) | — | sensibilidad del examen a k_cred (× 0,5, × 2) |
| 3.1 detector | solo subidas de precio | avisar también de bajadas (el uplift isolated cayó un 6,2 % en 2026-09) |
| 8.5 (nueva) | — | aviso de tablas `sff_` en la base que la ejecución no escribió |

## Límites de esta revisión

- **Hipótesis por construcción.** En el sintético las hipótesis están construidas: los patrones son los que el generador puso. El veredicto vale para el método de contraste, no para los datos reales.
- **Seis meses de examen.** Seis meses dan poca potencia. El test de signos solo es significativo con 6 de 6 (p = 0,031; 5 de 6 da p = 0,22). El t pareado sí puede serlo con 5 de 6 si las diferencias son claras (frente a la serie sola a 1 mes, p = 0,01). Por eso el veredicto mira los dos.
- **Examen optimista.** La estructura (escalera, técnica, regla del nivel) se elige con los mismos meses cerrados que la examinan, así que el examen es algo optimista. Lo dicen el informe y el resumen.
