# CAMBIO · El forecast contado: una SQL y sus gráficos por pregunta (26-oct-2026)

## En una frase

`informe_forecast.py` cuenta el forecast pregunta a pregunta (26 preguntas desde el capítulo 6) sobre las tablas que una ejecución ya escribió, sin volver a ejecutar el framework. El orden es el de la historia:

1. el dato real, que se puede triangular con otras áreas;
2. lo que se simula para construir;
3. la pipeline construida;
4. el resultado;
5. cómo desgranarlo.

Cada pregunta deja su SQL de SQL Server (para SSMS o Power BI), su CSV y sus gráficos. Las preguntas y su numeración están en el documento «SFF · Informe del forecast contado: preguntas, consultas y gráficos (plan)».

## Cómo se usa

```python
from informe_forecast import report
answers = report(configuration)                    # las 24 preguntas
answers = report(configuration, ["3.1", "4.3"])    # algunas
answers["3.1"].table, answers["3.1"].sql, answers["3.1"].charts
report(configuration, acquisition_plan_units=180000)   # 2.4 con la previsión del equipo de adquisición
```

Desde la consola: `python informe_forecast.py` (producción), `python informe_forecast.py parecido 2026-10` o `python informe_forecast.py parecido 2026-10 3.1 4.3`.

Lo escribe en `salida/informe/`; con un sintético en SQLite, en `salida/informe_<base>/`, para que un informe no pise al otro.

## Las preguntas

| Capítulo | Preguntas |
|---|---|
| 1 · El dato real | 1.1 pipeline real por año y lo renovado · 1.2 pipeline real mes a mes, un año por línea · 1.3 la pipeline real de los meses que vienen (raw, usada, borrada) · 1.4 la tasa real con su banda binomial · 1.5 / 1.5b la adquisición real por región y por producto · 1.6 el retail to subscription real · 1.7 los estados del cliente hoy |
| 2 · Lo que simulo | 2.1 las curvas de maduración y su examen · 2.2 cuánto madurarán los estados de lo que vence · 2.3 lo que renovará hasta diciembre y vuelve como pipeline · 2.4 la adquisición simulada frente a la real y la previsión · 2.5 el retail to subscription simulado |
| 3 · Lo construido | 3.1 la pipeline del año que viene por sus cuatro orígenes · 3.2 sin doble conteo: lo borrado frente a lo que lo sustituye |
| 4 · El resultado | 4.1 cómo acaba el año en curso · 4.2 el año en curso mes a mes · 4.3 cómo acaba el año que viene · 4.4 el año que viene mes a mes |
| 5 · Desgranarlo | 5.1 / 5.1b la cadena pipeline → tasa → uplift → dinero por región y subregión, con sus bandas · 5.2 por producto · 5.3 ¿ha sido honesta la banda? · 5.4 lo que cuestan los estados y el potencial |
| 6 · Las bandas reales | 6.1 el error real del examen a cada nivel frente a la banda de cada regla (`sff_bandas_reales`) · 6.2 qué regla de suma acierta a cada nivel (`sff_bandas_reales_resumen`, y el factor k y la correlación entre composiciones en `6_2_factor_k.csv`) |

## La banda que se puede desagregar

Las filas de una composición, que comparten tasa, fallan juntas, y las composiciones distintas fallan por separado. El error del volumen de la adquisición simulada mueve todas las adquisiciones a la vez.

**Semibanda** = √(Σ por composición (Σ desviaciones de sus filas)² + (Σ error de adquisición)²)

La banda de una región se calcula en la región: las filas de una composición repartidas en dos subregiones se suman primero. Nunca es la suma de las bandas de sus subregiones.

En el parecido al real, 2027 da 24,8 M$ con banda de 23,5 a 25,4 M$. La banda lineal (todas las filas fallan a la vez, que es lo que suma hoy Power BI) va de 15,4 a 30,9 M$. 5.3 dice si la banda por composición ha sido honesta: la parte del examen que cayó dentro, por región, frente a la cobertura nominal de z.

La tasa lleva dos bandas:
- **binomial:** z √(Σ n p (1 − p)) / Σ n, el ruido mínimo del volumen;
- **del framework:** las desviaciones en unidades sumadas por composición.

En 2027, la tasa del 67,2 % tiene una banda binomial de ±0,08 pp y una del framework de ±1,42 pp. Lo que pesa es el error del modelo, no el volumen.

## Lo que añade al framework (necesita una ejecución nueva)

- **`sff_curva_maduracion`**, que escribe el paso 7.2 con `curve_table` del bloque 4: por estado con fechas y k (meses antes de vencer) da f(k), F(k), p(k), el tope, si la curva se usa y el examen por antelación. La lee la pregunta 2.1.
- **`s7_2_error_captacion_usd`** en el núcleo: el error del volumen de cada fila de adquisición simulada en dólares (examen del paso 5.1), y 0 en las demás. Antes solo estaba en el total del mes.

Con las tablas de una ejecución anterior funciona todo menos 2.1, que lo dice y sigue. La banda se queda sin el error de la adquisición hasta la ejecución nueva.

## Comprobado

`test_informe_forecast.py` tiene 12 comprobaciones:

- las 24 preguntas responden sobre una ejecución y cada una deja su SQL, su CSV y sus gráficos;
- el año en curso del informe es el de la tabla final del 8.6 y el siguiente es el del núcleo;
- las subregiones suman su región y las regiones suman el TOTAL;
- la banda por composición del SQL es la misma calculada en pandas y queda entre las filas independientes y la lineal;
- la pipeline del año que viene es la suma de sus cuatro orígenes;
- lo que sustituye a lo borrado no es menor en ningún mes.

En el parecido al real, 3.2 sale bien. En octubre de 2027, el extracto ya traía 0,9 M$ y lo proyectado es 2,1 M$: no hay doble conteo y se proyecta el mes entero.

## 6 · Las bandas reales: aprender de los datos cómo se suman (añadido el 26-oct-2026)

Cómo sumar las bandas es delicado, así que la regla se decide con los datos reales. El examen del paso 7.1 (`sff_series_exam_detail`, método framework) predijo cada serie en los meses de examen, a 1 y a 6 meses, con su banda, y sabe lo que renovó.

**6.1 suma el examen a cinco niveles:** serie, composición, subregión, región y total. Para cada grupo, mes (y los meses sumados) y horizonte da el error real y la semibanda de cuatro reglas:

| Regla | Semibanda | Qué supone |
|---|---|---|
| Lineal | Σ desviaciones | todas las series fallan a la vez |
| Composición | √(Σ por composición (Σ desviaciones)²) | cada composición falla a la vez; las composiciones, por separado |
| Cuadratura | √(Σ desviaciones²) | todas las series por separado |
| Binomial | z √(Σ n p (1 − p)) | solo el ruido del volumen |

Se escribe en la base como `sff_bandas_reales`, para mirarla en SQL o Power BI.

**6.2 dice qué regla acierta a cada nivel** (`sff_bandas_reales_resumen`):
- la cobertura real (la parte de los errores dentro de la banda; z = 1,645 promete el 90 %);
- el cociente error/banda (1 = honesta; por encima, la banda se queda corta);
- el error medio y la parte de sesgo (error medio / error cuadrático medio). Si es alta, el problema es un desplazamiento, no la banda.

Además, en `6_2_factor_k.csv`:
- el factor k que le falta a cada regla;
- la correlación entre composiciones que implican los errores, ρ = (z² · media(error²) − media(composición²)) / (media(lineal²) − media(composición²));
- la regla aprendida √(composición² + ρ (lineal² − composición²)), con un ρ común a los niveles agregados. Su cobertura se mide sobre los mismos datos de los que sale ρ: dice si un solo ρ vale para todos los niveles, no que vaya a acertar en el futuro.

**En los datos parecidos al real (a 1 mes; cobertura y factor k que le falta a cada regla):**

| Nivel | Regla de composición | Regla lineal | Regla aprendida (ρ ≈ 0,06) |
|---|---|---|---|
| Serie | 93 % | 93 % | 93 % |
| Subregión | 82 %, k ≈ 1,6 | 98 %, k ≈ 0,4 | 88 % |
| Región | 53 %, k ≈ 3,6 | 100 %, k ≈ 0,3 | 83 % |
| Total | 50 %, k ≈ 3,0 | 100 %, k ≈ 0,14 | 100 % |

- A nivel de serie y composición, la banda es honesta.
- Al sumar, la regla de composición se queda corta y la lineal es varias veces demasiado ancha.
- Una correlación pequeña entre composiciones lo explica casi todo, pero la correlación de cada nivel no es la misma: 0,35 en subregión, 0,12 en región y 0,02 en total.
- Russia++ tiene un sesgo de +547 unidades al mes (lo predicho se queda corto todos los meses): eso no lo arregla ninguna regla de banda.

Son datos sintéticos: **la decisión se toma con la ejecución real.** Con 6 meses de examen, el total solo tiene 6 observaciones por horizonte, así que su cobertura es ruidosa. Las subregiones (180) y las regiones (30) son las que deciden.

**Test:** en `test_informe_forecast.py`, bloque E, 5 comprobaciones:
- el error del total es la suma del de sus series;
- lineal ≥ composición ≥ cuadratura a cada nivel;
- cada nivel tiene sus meses y los meses sumados;
- las dos tablas están en la base;
- salen el factor k y la regla aprendida.

## 7 · Lo privado de tu máquina, fuera de main.py: `entorno_local.py` (añadido el 26-oct-2026)

El servidor, la base, el driver, el schema y la query del extracto ya no están en `main.py`. Están en `entorno_local.py`, junto a `main.py`. Ese fichero solo existe en tu máquina y no viaja en el zip, así que puedes sobrescribir `main.py` con cada versión sin volver a rellenar esas líneas.

La primera vez:
1. copia `entorno_local_ejemplo.py` como `entorno_local.py`;
2. rellena `SQL_SERVER`, `SQL_DATABASE`, `SQL_DRIVER`, `SQL_SCHEMA` y `RAW_EXTRACT_QUERY`.

Opcionales:
- `SQL_USER` / `SQL_PASSWORD`: sin ellos, autenticación de Windows. La contraseña, mejor desde una variable de entorno.
- `TABLE_PREFIX`.
- `CURRENT_MONTH`: así el mes en curso tampoco depende de `main.py`.

Lo leen todos los que usan `production_configuration()`: `python main.py`, `ejecutar_informe.py`, `informe_forecast.py` y el explorador. Los sintéticos no lo necesitan.

Si falta el fichero, o tiene campos sin rellenar, `production_configuration()` se para y dice cuál.

## 8 · La pipeline auditada por componentes: capítulo 7 (añadido el 28-oct-2026)

**El porqué.** En la ejecución real, la pipeline del año que viene sale más pequeña que la de este año. Para saber si es lo que dice el pasado o si una componente rompe la línea, cada componente se dibuja mes a mes desde el primer mes del extracto hasta diciembre del año que viene, con el pasado real como referencia. Una componente que sube o baja de golpe justo donde acaba el dato real es la que hay que auditar. No vuelve a ejecutar el framework: lee las tablas de la última ejecución.

**Las componentes de la pipeline** (lo que vence en un mes, en USD):

| Componente | De dónde sale |
|---|---|
| renovaciones de 1 año (extracto) | filas del extracto, plazo de 1 año, que no son adquisición |
| licencias de 2 y 3 años (extracto) | filas del extracto con plazo de varios años |
| adquisiciones de 1 año (extracto) | filas del extracto de 1 año con `net_new` de adquisición |
| renovaciones proyectadas (7.2) | lo que se espera que renueve de la pipeline de 1 año que vence del mes en curso a diciembre |
| adquisición simulada (5.1) | la adquisición de 1 año de la ventana, hasta el objetivo |
| reentrada del retail to subscription (6.1) | la pipeline que crea el universo time series |
| borrada por el 1.3 (fuera del total) | la pipeline de 1 año a medio construir que el calendario borra y las dos anteriores sustituyen |

La pipeline del pasado que viene de retail to subscription todavía no se distingue en el extracto: va dentro de las renovaciones. Está anotada como mejora pendiente (quizá una columna nueva).

**Las cuatro preguntas.** Las preguntas están en español y los gráficos de este capítulo en inglés. Cada gráfico lleva debajo su anexo con los números de lo que dibuja; el anexo también se escribe como `<número>_anexo_*.csv` y se imprime en pantalla.

- **7.1 · ¿De qué se compone la pipeline mes a mes, del pasado real al año que viene?**
  - Las componentes apiladas mes a mes, con la banda gris del forecast.
  - Debajo, el total frente al mismo mes del año anterior: azul lo real, naranja lo previsto.
  - El anexo da la pipeline por año y componente, con la variación de los dos últimos años. El subtítulo compara la tendencia real con el año que viene («Real trend 2023→2026: −5,0 %/year · 2027 vs 2026: −2,7 %»).
  - Un gráfico más por región, cada una en su escala.
- **7.2 · ¿Cuánta pipeline tiene cada región este año y el que viene, y cuánto es simulado?**
  - Barras del año anterior, el actual y el siguiente por región, con las componentes apiladas.
  - Anexo por región: el total de cada año y su variación; lo construido por el framework (proyectadas + adquisición simulada + retail to subscription) en dólares y en %; la adquisición (extracto + simulada) y el retail to subscription.
  - Otro anexo igual por subregión.
- **7.3 · ¿Lo renovado vuelve como pipeline 12 meses después?**
  - La pipeline de renovación de 1 año de cada mes frente a lo renovado de 1 año 12 meses antes (real en los meses cerrados, esperado en el futuro), y su cociente.
  - Las proyectadas vuelven exactamente a lo que renuevan (cociente 1). Si el cociente real es mayor que 1, hay pipeline que la proyección no construye; si es menor, construye de más.
  - En el parecido al real, el cociente real es 0,956 (de 0,939 a 0,975).
- **7.4 · ¿Es estable el uplift de cada componente, con y sin subidas de precio?**
  - El uplift mes a mes de cada componente (USD renovado / USD que vencía de lo que renovó): real en los meses cerrados, el que aplica el forecast en el futuro.
  - Dos paneles: con todas las filas y sin las filas que caen dentro del ciclo de una subida de precio. Una fila está dentro de una subida cuando su grupo de precio (`price_group_dims`) y su mes caen en el ciclo (`price_cycle_months`) de una subida de `sff_price_increase_calendar`.
  - El anexo da el uplift por año de cada componente, con y sin subidas, y la parte de la base que cae dentro de una subida.

**Lo que ya enseña el parecido al real** (son datos sintéticos; lo que vale es la ejecución real):

- La pipeline de 2027 (32,1 M$) baja un 2,7 % frente a 2026, menos de lo que baja la tendencia real (−5 %/año). No hay ruptura.
- El 22 % de 2027 lo construye el framework.
- Las renovaciones proyectadas renuevan en 2027 con un uplift de 1,033, frente al 1,11 del resto de renovaciones de 1 año. Vencen con descuento 0, así que toman el uplift estadístico de la celda de descuento 0 (1,005 en las celdas propias, 1,13 donde caen a la celda mandatory). Esto no cambia la pipeline de 2027, pero sí lo que renueva en octubre–diciembre de 2027. Hay que mirarlo con los datos reales.

**Cómo se ejecuta:**
- `python ejecutar_informe.py` con `PREGUNTAS = ["7.1", "7.2", "7.3", "7.4"]`, o
- `report(configuracion, ["7.1", "7.2", "7.3", "7.4"])` desde Jupyter.

**Test:** en `test_informe_forecast.py`, bloque F, 4 comprobaciones:
- las componentes suman el total de cada mes y cada año es la pipeline del núcleo;
- 7.2 coincide con 7.1;
- en los meses que construye, lo proyectado es lo renovado 12 meses antes;
- el uplift del total está entre los de sus componentes.

## 9 · Entender el año que viene frente al equipo de retención: capítulo 8 (añadido el 28-oct-2026)

**El porqué.** El forecast de 2027 del equipo de retención (133.851 miles de USD) parece razonable, y nuestra línea base sale bastante más baja. La pipeline de 2026 está bien construida, así que el capítulo busca dónde está la diferencia y qué palanca la explica. No toca el framework: lee las tablas de la última ejecución.

**El retail to subscription está dentro.** Nuestro 2027 de 8.1 es el total de `sff_resumen_negocio` con todos los conceptos:
- las renovaciones de la pipeline del extracto, de las renovaciones proyectadas y de la adquisición simulada;
- el retail to subscription: su ingreso simulado (`ts_simulado`) y la renovación de su reentrada (`ts_reentrada`).

El test lo comprueba: los conceptos suman el total, y el total es el de `sff_resumen_negocio`. El gráfico 8.1b compara ese total con la referencia por región del equipo, con los conceptos apilados (la referencia no viene desglosada), y lo anexa por concepto.

**La referencia.** `forecast_retencion_2027.csv`, junto a `main.py`: una fila por subregión (`tr_regional_level_3`), en miles de USD, con la región del equipo.
- La línea LATAM del equipo (21.824) incluía Canadá; el fichero solo tiene las subregiones.
- Se compara con nuestras renovaciones más el retail to subscription: se toma la referencia como el ingreso total del año. Si el equipo de retención no cuenta el retail to subscription, la columna Δ R2S dice cuánto hay que restar.

**El dinero de cada fila**, igual en todas las preguntas:
- renovado = pipeline × tasa en USD × uplift;
- tasa en USD = USD que vencía de lo que renueva / USD que vence;
- uplift = USD renovado / USD que vencía de lo que renueva;
- real en los meses cerrados, previsto en el futuro.

**Las siete preguntas** (en español; los gráficos, en inglés, cada uno con su anexo debajo, en CSV y en pantalla):

| | Pregunta | Qué enseña |
|---|---|---|
| 8.1 | ¿Dónde están las mayores diferencias con el forecast del equipo de retención? | El puente de nuestro 2026 a nuestro 2027, un factor cada vez (pipeline, tasa, uplift, R2S), hasta la referencia. El hueco de cada subregión, ordenado de mayor a menor falta. Anexos por región del equipo y por subregión con la tasa y el uplift que haría falta para llegar a la referencia con nuestra pipeline |
| 8.2 | ¿Son parecidas las tasas de renovación de un año a otro? | Una línea por año (2023–2027) mes a mes, en unidades y en USD, en total y por región |
| 8.3 | ¿Se está acabando el efecto de las subidas de precio en la revalorización? | El uplift de cada año con todas las filas y sin las que caen en el ciclo de una subida (`sff_price_increase_calendar`). El anexo da la parte de la base dentro de una subida y el dinero que añaden las subidas cada año (lo que renovaron las filas de dentro menos lo que renovarían al uplift sin subidas), por región y subregión |
| 8.4 | ¿Cómo evolucionan las combinaciones de referencia de precio? | Subregión × producto × plazo × banda, la forma en que se hacen las subidas: uplift, USD que vence por unidad y USD pagado por unidad renovada, mes a mes, con los meses dentro de una subida en naranja. Sigue las de `REFERENCE_PRICE_CELLS` (de partida DACH · premium · 1 year · 10, buscando el texto sin distinguir mayúsculas) más las 8 mayores del año en curso |
| 8.5 | ¿Es distinta la composición de la pipeline del año que viene? | Para cada dimensión (tipo de renovación, producto, plazo, banda, partner, estados, `net_new` y componente de la pipeline): la parte de la pipeline y el rendimiento (renovado / pipeline) de cada valor en los dos años. El cambio del dinero se parte en volumen, mezcla (la composición nueva a los rendimientos de este año) y rendimiento; las tres partes suman el cambio |
| 8.6 | ¿Maduran bien los estados del cliente? | Para dormant, softcancel y no instalado, la parte de las unidades que vencen en ese estado: final en los meses cerrados, hoy y al vencer en el futuro. Dice qué estados tienen curva de maduración con fecha (en producción, solo softcancel; los otros van a su parte histórica). El anexo da la tasa del estado, la neutra y las unidades que cuesta un punto más de estado |
| 8.7 | ¿Se queda corto el horizonte de más de un año? | La tasa prevista de cada mes futuro según los meses de antelación (h hasta 15–16), frente a la última tasa real del mismo mes. Las barras de la diferencia muestran si la técnica deriva con el horizonte. Anexo: el examen de la cartera por horizonte (sesgo y error) |

**Lo que ya enseña el parecido al real** (sintético: la escala no se compara con la referencia, y 6 subregiones de la referencia no existen en él):
- de 2026 a 2027 el uplift pasa de 1,148 a 1,114 (−720 k$), la tasa sube (+469 k$) y la pipeline baja (−651 k$);
- Europa pierde uplift (1,169 → 1,122; DACH 1,159 → 1,106) y Russia++ también (1,150 → 1,103). Es justo el patrón de «se acaba el efecto de las subidas» que hay que confirmar con los datos reales en 8.3 y 8.4;
- en 8.6, dormant y no instalado, sin curva con fecha, van a su parte histórica. No instalado queda algo por encima del pasado (3,4 % frente a 3,2 %).

**Cómo se ejecuta:**
- `python ejecutar_informe.py` con `PREGUNTAS = ["8.1", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7"]`, o
- `report(configuracion, ["8.1", …], retention_reference_file=…)` para otra referencia.

**Test:** en `test_informe_forecast.py`, bloque G, 4 comprobaciones:
- renovado = pipeline × tasa × uplift, y el puente suma;
- volumen + mezcla + rendimiento es el cambio de cada dimensión;
- las tasas están entre 0 y 1;
- cada mes futuro tiene su horizonte.

## 10 · Qué región tiene la composición más favorable (8.8) y un informe más rápido (añadido el 28-oct-2026)

**8.8 · ¿Qué región tiene la composición más favorable?** Dos regiones con la misma pipeline no traen el mismo dinero si una tiene más de lo que renueva a más valor: un producto, un plazo, una banda o la autorrenovación; menos estados del cliente; o más pipeline dentro de una subida de precio. Cada región se mide a los rendimientos de toda la cartera (rendimiento = renovado / pipeline, la tasa y el uplift juntos):

- **índice de composición** = Σ su pipeline de cada segmento × el rendimiento de ese segmento en la cartera / su pipeline / el rendimiento de la cartera. 1 es la composición de la cartera; por encima, una composición que renueva más.
- **índice de una dimensión**: lo mismo con esa dimensión sola (producto, plazo, banda, tipo de renovación, adquisición, subida de precio, dormant, softcancel, no instalado). Dice qué parte de la composición ayuda y cuál resta.
- **real** = el rendimiento de la región / el de la cartera. **Desempeño** = real / composición: cómo renueva la región frente a lo que promete su composición.
- La pipeline es la de la fecha de vencimiento, así que los estados ya están madurados.

Gráficos:
- un mapa de calor de 2027 por subregión × dimensión, más la composición completa, el real y el desempeño;
- las regiones en 2026 y 2027.

El anexo da, por región, la composición completa con su parte de tasa y de uplift, el real, la parte de la pipeline dentro de una subida y la dimensión que más ayuda y la que más resta.

Test: ponderados por su pipeline, la composición y el rendimiento real de las regiones son los de la cartera.

**El informe tardaba por las subidas de precio, no por los CSV.** Saber si una fila cae dentro del ciclo de una subida se hacía con un EXISTS fila a fila contra el calendario. En el parecido al real, 7.4 tardaba 168 s y 8.3, 72 s. Ahora los meses de cada ciclo se expanden una vez (una fila por grupo de precio y mes) y se cruzan por igualdad: 7.4 baja a 10 s y 8.3 a 6 s, con los mismos números. Los CSV no llegan a un segundo en total. El informe completo pasa de más de 6 minutos a unos 2 minutos en el parecido al real. Cada pregunta dice en pantalla los segundos que tarda, para ver cuál pesa con los datos reales; las 6.1 y 6.2 son las que más leen.

## 11 · Lo que le falta a la proyección de renovaciones (7.3b) y etiquetas de 7.2 (añadido el 28-oct-2026)

**7.3b · ¿Qué le falta a las renovaciones proyectadas si se mantiene el cociente real?** Las renovaciones proyectadas vuelven exactamente a lo que renuevan (cociente 1). La pipeline real del extracto vuelve a su cociente real: por encima de 1, el extracto crea pipeline que la proyección no crea; por ejemplo, una licencia de varios años que renueva como de 1 año, o un cliente que vuelve.

Por región y subregión, la tabla da:
- las renovaciones proyectadas del año que viene;
- el cociente real de los mismos meses un año antes y el de los últimos 12 meses;
- el cociente contando también las renovaciones de varios años de 12 meses antes. Si ese se acerca a 1, lo que sobra viene de licencias de varios años que renuevan a 1 año;
- la pipeline que daría el cociente real, menos la reentrada del retail to subscription que el framework ya construye aparte (el cociente real la lleva dentro, porque esa pipeline va en las renovaciones del extracto), y lo que falta.

7.3 dibuja además la línea de las renovaciones de 1 año más las de varios años de 12 meses antes, y su cociente.

**7.2, etiquetas más claras:**
- la adquisición es la de 1 año que **vence** en el año, vendida el año anterior. La de 2027 es la vendida en 2026 a 1 año; las de 2 y 3 años de los 25 M$ vencen en 2028–2029;
- el retail to subscription de 7.2 es solo la pipeline de su reentrada: la reentrada de lo convertido de octubre a diciembre de 2026. Su ingreso de 2027, simulado para el año entero, no tiene pipeline y está en 4.3 y 8.1b.

## 12 · 8.9 (subidas de precio detectadas) y 8.10 (transiciones de plazo) (añadido el 28-oct-2026)

- **8.9 · ¿Dónde se han detectado subidas de precio y desde cuándo?** El detector del paso 8.4 para cada grupo de precio con alguna subida: la revalorización de las renovaciones aisladas frente a la media de sus 12 meses anteriores, con los 12 meses de cada subida en naranja, su inicio y su escalón. El anexo trae todas las subidas de `sff_price_increase_calendar` con sus fechas, para comparar con las subidas de tarifa que se hicieron.
- **8.10 · ¿A qué plazo renuevan las licencias?** Con las seis columnas de las transiciones de plazo (`CAMBIO_transiciones_de_plazo.md`): por plazo de origen, la parte de cada destino mes a mes, su revalorización y su USD por año. Sin las columnas, la pregunta dice que no se puede responder.
