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
