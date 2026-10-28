# PLAN · SFF separado por etapas (28-oct-2026)

**Estado: plan acordado, para ejecutar en los próximos días.** El resultado de este mes se saca con el framework actual. La reestructuración empieza después, pieza a pieza.

## Conclusiones

- **Cada etapa produce algo que se persiste y que la siguiente lee.** Las etapas son el dataset raw validado, el análisis, los supuestos aprendidos, el núcleo de predicción, la capa de negocio y el seguimiento. Ninguna etapa recalcula lo que hace otra.
- **Una predicción nueva (un corte de datos nuevo) lo recalcula todo.** Dentro de un mismo corte, lo ya calculado se reutiliza: los escenarios del mes solo vuelven a ejecutar el núcleo.
- **El núcleo recibe tres clases de input y ninguna es "lo que el núcleo cree":**
  1. supuestos aprendidos de los datos;
  2. referencias externas (el forecast de retención, el objetivo de adquisición), en modo *comparar* o *forzar*;
  3. **palancas**: ajustes acordados con los retention managers (una campaña sobre los cancel de una región, por ejemplo), que no son dato real y no tienen por qué serlo.
- **Baseline = el núcleo sin palancas. Escenario = el mismo núcleo con palancas. Referencia = la ejecución entregada a negocio, congelada.** El seguimiento mide la realidad frente a las tres. La pregunta que importa es «¿la campaña acordada ganó a la baseline en su ámbito?».
- **La separación también es de gobierno.** El forecast es un número; cómo se construye es una filosofía, y cómo se cambia es la palanca. Las piezas son independientes, así que cada una se puede compartir o no por separado. Lo que no se entrega es la coordinación: qué supuestos se usan, qué palancas se acuerdan y cómo se lee el seguimiento. Esa coordinación es el trabajo consultivo con los retention managers.

## Las etapas

| # | Etapa | Lee | Produce (persistido) | Sale de lo que hay hoy |
|---|---|---|---|---|
| 0 | **Datos** | SQL de origen | `raw` validado por corte, con sus tests (nulos, huecos, cuadres, columnas de mes cerrado ≤ su total) | pasos 1.1–1.4 (dbt opcional para la SQL y los tests) |
| 1 | **Análisis** | raw | informe de patrones: pipeline real, ritmo de renovación, estabilidad, cambios de régimen, composición, adquisición por plazo, transiciones, subidas de precio | capítulos 1, 7 y 8 del informe, perfil de datos (8.6) |
| 2 | **Aprendizaje** | raw | **supuestos versionados**, cada uno con su corte, la versión de su método y una validación | 4.1 curvas de estados · `term_transitions` · detector de subidas (bloque 3, hoy escrito en 8.4) · niveles del 1.2 · reglas de 5.1 (adquisición) y 6.1 (time series y retail to subscription) |
| 3 | **Núcleo** | raw + supuestos + referencias + palancas | forecast por `run_id`: tasa, revalorización, maduración aplicada, ensamblado, banda; el examen | bloques 2, 3 (estimación), 7 y la aplicación de las cajas de 4, 5 y 6 |
| 4 | **Negocio** | salidas del núcleo | resumen, Power BI, comparación con las referencias, respuestas a preguntas concretas | bloque 8 (8.3–8.4) e informe para negocio |
| 5 | **Seguimiento** | ejecuciones registradas + raw nuevo | realidad frente a referencia, baseline y escenario, por pipeline, resultado y patrones; deriva de los supuestos; resultado de cada palanca | nuevo |

Las cajas del 25-oct (`RateBox`, `StateMaturation`, `FuturePipeline`, `TimeSeriesUniverse`) son las costuras naturales. Lo que está dentro de una caja y se *aprende* pasa a la etapa 2. Lo que se *aplica* se queda en el núcleo.

**Lo que no sale del núcleo:** la predicción de la tasa (escalera, técnica por composición, backtest). Es el núcleo, y sacarla lo vaciaría. La revalorización por celda está en la frontera; se decide al llegar a ella.

## Los inputs del núcleo

### Supuestos aprendidos (etapa 2)

Un fichero o tabla por supuesto, con un contrato común:

| Campo | Para qué |
|---|---|
| `supuesto`, `version_metodo` | qué es y con qué código se aprendió |
| `corte_datos` | último mes cerrado del raw del que se aprendió |
| `aprendido_a_fecha` | normalmente = corte; en el examen, la fecha de cada origen |
| grano (dims) + valores | p. ej. nivel × grupo × plazo de origen → parte a 1/2/3 años |
| `con_soporte`, `nivel` | de dónde sale el valor de cada grupo |

Supuestos de la primera ola: curvas de maduración de los estados, transiciones de plazo, calendario de subidas, niveles de las dimensiones, reglas de adquisición y de retail to subscription.

### Referencias externas

- El forecast de retención (`forecast_retencion_2027.csv`) y el objetivo de adquisición, cada uno con su modo:
  - `comparar`: se informa la diferencia;
  - `forzar`: el núcleo reparte el total de la referencia con su propia forma, como hace hoy la adquisición con su objetivo.

### Palancas (escenarios)

Un fichero editable (CSV o YAML) con una fila por ajuste acordado:

| Campo | Ejemplo |
|---|---|
| `id`, `descripcion` | `P-2027-01` · campaña de recuperación de cancel en DACH |
| `acordado_con`, `fecha_acuerdo`, `estado` | retention manager DACH · 2026-11-10 · propuesta / acordada / descartada |
| `ambito` (dims) | región = DACH, estado = cancel |
| `periodo` | 2027-03 a 2027-08 |
| `tipo` | `tasa_suma_pp` · `tasa_multiplica` · `uplift_multiplica` · `transicion_fija_parte` · `adquisicion_suma_usd` · `curva_sustituye` |
| `valor` | +3 pp |

Un escenario es un corte, un juego de supuestos, unas referencias con su modo y un conjunto de palancas. El núcleo aplica las palancas después de leer los supuestos y antes de ensamblar. Cada palanca deja una columna de su efecto, para que el resultado se pueda descomponer en baseline más el efecto de cada palanca.

## Registro de ejecuciones

Una tabla `sff_ejecuciones` con una fila por ejecución:
- `run_id`, `corte_datos`, `version_codigo`;
- `tipo`: baseline / escenario / referencia entregada;
- la huella de los supuestos usados, las palancas y los modos de las referencias;
- fecha y nota.

Las salidas del núcleo se guardan por `run_id`, así que un escenario nunca sobrescribe la baseline. **La referencia entregada se congela:** no se borra y no se recalcula.

## Reglas de diseño

1. **Aprender a fecha X.** La lógica de cada supuesto es una función `aprender(raw, a_fecha)`. La etapa 2 la llama a fecha del corte. El examen del núcleo la llama a fecha de cada origen, para no usar información del futuro. Es el mismo código usado de dos formas.
2. **Las palancas no entran en el examen.** El examen mide el método, no la campaña.
3. **Compatibilidad al arrancar.** El núcleo comprueba que el corte de cada supuesto coincide con el del raw. Fuera de un escenario declarado, la diferencia es un aviso fuerte. También comprueba que el supuesto cubre los grupos, los estados y los plazos del raw.
4. **Recalcular por corte, reutilizar dentro del corte.** Las etapas 0–2 se ejecutan una vez por corte; los escenarios solo ejecutan 3 y 4.
5. **Sin orquestador.** Un script por etapa, un `ejecutar_todo` y el registro de ejecuciones. Todo persistido en SQL Server (esquema por etapa), salvo los ficheros que se editan a mano: palancas y referencias.
6. **Sustitución gradual, con test de igualdad.** El framework actual sigue funcionando. Cada pieza que se extrae tiene un test que compara con lo de antes: mismos números con el mismo corte y sin palancas.
7. **dbt, solo si encaja.** Puede llevar la SQL del raw y sus tests. Por lo que sé, el adaptador de SQL Server no ejecuta modelos Python de dbt, así que el resto sigue en Python. Hay que comprobarlo antes de comprometerse.

## Orden de ejecución

| Hito | Contenido | Hecho cuando |
|---|---|---|
| H1 · Diseño | contratos de supuestos, palancas y registro; carpetas; esquemas SQL | documento revisado |
| H2 · Etapa 0 | raw validado y persistido por corte; los pasos 1.1–1.4 leen de él | el framework actual corre sobre el raw persistido con los mismos números |
| H3 · Primer supuesto: curvas de maduración | `aprender(raw, a_fecha)`; la caja `StateMaturation` lee el supuesto; examen a fecha de origen | test de igualdad bien; plantilla para el resto |
| H4 · Transiciones y calendario de subidas | misma plantilla | test de igualdad bien |
| H5 · Registro de ejecuciones y referencia congelada | `sff_ejecuciones`; salidas por `run_id`; congelar la entregada | se puede comparar la realidad del mes con la referencia |
| H6 · Palancas | fichero de palancas; aplicación en el núcleo; columnas de efecto; descomposición baseline + palancas | un escenario «campaña cancel DACH» corre sin recalcular 0–2 |
| H7 · Seguimiento | realidad frente a referencia, baseline y escenario; resultado por palanca; deriva de los supuestos | informe mensual de seguimiento |
| H8 · Limpieza | sacar del núcleo lo que queda de análisis (veredictos, perfil, informe) a las etapas 1 y 4 | el núcleo solo predice y examina |

H2 y H3 son los que más tiempo ahorran. H6 y H7 son los que dan el valor de negocio: acordar palancas y medir si ganan a la baseline.

## Pendiente antes de empezar

- Sacar el resultado de este mes con el framework actual. Antes, poner `TERM_TRANSITION_COLUMNS_IN_EXTRACT = True` cuando el extracto traiga las seis columnas.
- Revisar con datos reales:
  - la bajada de pipeline de Europa (7.2);
  - lo que falta en 7.3b;
  - las subidas detectadas (8.9).
- Mejoras ya anotadas, que encajan en el plan:
  - identificar la pipeline de retail to subscription (etapa 0);
  - revisar el nivel de retail to subscription (etapa 2).
