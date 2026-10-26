# Cambio · el universo time_series como pieza del forecast · 25-oct-2026

## Qué es

Una caja, `TimeSeriesUniverse` (`time_series_universe.py`), que simula por regiones el universo de retail a suscripción y entrega filas que se suman al forecast, a la pipeline y a las bandas. Se construye en el paso 18, junto a la pipeline futura, porque es lo mismo: lo que el extracto aún no puede tener. Está aislada en su módulo para poder mejorarla sin tocar el resto.

## Las filas que da (región × mes, al nivel regional más fino)

| Origen | Meses | Regla | Cuenta como |
|---|---|---|---|
| `ts_real` | cerrados del año en curso | lo que convirtió | revenue del año |
| `ts_proyectado` | del mes en curso a diciembre | mismo mes del año anterior × nivel de la región (últimos 3 meses frente al año anterior) × valor medio de 12 meses | revenue del año |
| `ts_simulado` (nuevo) | los 12 meses del año siguiente | mismo mes de este año (real o proyectado) × `ts_next_year_level` | revenue del año siguiente |
| `ts_reentrada` | lo proyectado, un año después | valor / (1 − 0,4) × tasa de la región en la pipeline normal | pipeline y renovaciones del año siguiente |

- Las conversiones de los meses cerrados ya están en el extracto como pipeline del año siguiente, así que no reentran.
- Las del año siguiente vencen en el otro: quedan fuera del horizonte.
- Una región sin el mismo mes del año anterior o sin nivel toma su cuota del total (aviso en pantalla).

## La banda: un examen propio

- En cada uno de los últimos `ts_exam_origins` (6) meses cerrados se simula la serie con los meses anteriores solamente.
- Se compara el revenue total del mes con lo que convirtió. El error por horizonte es la media banda de cada fila: error(h) × revenue.
- Más allá del último h examinado se mantiene el último error. Es una cota inferior.
- A más de 12 meses el mes parte de otro simulado, así que se suma en cuadratura el error de ese mes.
- La reentrada toma el error del mes del que viene. El error de la tasa de la región no entra en la banda.

## Cómo entra en el resto

| Dónde | Qué |
|---|---|
| Paso 18 | construye la caja; 4 comprobaciones nuevas (7 en total) |
| Paso 20, por mes | `ingresos_ts_usd`, `vence_ts_reentrada_usd`, `esperado_ts_reentrada`, `error_ts_usd`, `usd_vence_total`, `total_usd` |
| Paso 20, por año | `ingresos_ts_real_usd`, `ingresos_ts_usd`, `vence_ts_reentrada_usd`, `esperado_ts_reentrada_usd`; `total_usd` los incluye |
| Banda oficial | `error_examen_usd` = error del examen × esperado + error del examen time_series, centrada en `total_usd`; la lineal y la de cuadratura también lo incluyen |
| Paso 21 | solo el total por año y origen; comprobación nueva: TOTAL de cada año = `total_usd` del paso 20 |
| Núcleo (24) | filas `ts_simulado`; las filas time_series llevan banda (`forecast_renewed_USD_low/high`) |
| Power BI (25) | fuente nueva "Retail to subscription (next year)" |
| Informe (27) | titular TOTAL por año con el universo; examen de la simulación; tabla por año y origen |

`esperado_usd` sigue siendo solo la renovación del universo normal (la comprobación 5 del paso 20 no cambia).

## Parámetros nuevos

- `ts_next_year_level = 1.0`: el año siguiente repite este mes a mes. `None` mantiene el nivel de la región un año más. En el parecido al real eso multiplicaba 2027 por 2,3, porque una campaña de julio entraba en la ventana del nivel. Por eso no es el valor por defecto.
- `ts_exam_origins = 6`.
- Corregido además: el nivel ya no lee como 0 vendido un mes anterior al inicio del extracto.

## Comprobación

- **Tests:** 27 baterías, 472 comprobaciones, todas bien. `test_step_21.py` reescrito (26 comprobaciones):
  - reglas a mano para el año siguiente (la base real y la proyectada);
  - examen a mano, exacto y con un mes que se desvía;
  - las cuatro filas;
  - la suma en el forecast por mes y por año, y la banda;
  - TOTAL = `total_usd`;
  - `ts_next_year_level` a 1,0 y a `None`.
- **Parecido al real** con un universo time_series inyectado (3 % de las filas, 30 regiones):
  - el universo normal da exactamente lo mismo que antes;
  - 2026 da lo mismo que antes en el time_series;
  - checks: 7/7 en el 18, 10/10 en el 20, 3/3 en el 21, 10/10 en el 24 y 7/7 en el 26.

| | 2026 | 2027 |
|---|---|---|
| TOTAL antes (sin time_series en `total_usd`) | $25,70M | $24,80M |
| TOTAL ahora | $26,50M | $25,90M |
| de ello, time_series | $0,80M (real 0,54 + proyectado 0,26) | $1,10M (simulado 0,80 + renovación de la reentrada 0,31) |
| error del examen, antes | ±$82k | ±$372k |
| error del examen, ahora | ±$277k | ±$1,04M |

- **El examen de la simulación es malo en estos datos:** 104 % de error a h = 1 y entre el 28 % y el 54 % a h de 2 a 6. La banda lo refleja.
- Los datos son una muestra al azar de renovaciones, no conversiones reales. Con el extracto real es lo primero que hay que mirar.

## Los gráficos (añadidos el mismo día)

`time_series_charts.py` dibuja el mismo gráfico dos veces, de cuatro paneles: ingresos al mes (total y regiones del nivel más grueso), unidades, valor por unidad y el nivel (últimos 3 meses frente al año anterior, el número que multiplica la proyección).

- `ts_01_historia`, en la etapa 01, antes de predecir nada.
- `ts_18_prediccion`, en la 18: lo simulado discontinuo, con la banda del examen, y la renovación de la reentrada punteada.

Se guardan en `<output_folder>/graficos`, se muestran en el notebook (`Config.show_figure`) y el informe los enlaza. Un gráfico que falla no para la ejecución. En el parecido al real el nivel osciló entre 0,40 y 2,25 en el último año: la regla de 3 meses es inestable en esa serie.

La cota baja de la banda ya no baja de 0 cuando el error del examen pasa del 100 %.

## Lo que queda por mejorar (la pieza está aislada para eso)

- ~~Un nivel más estable que el de 3 meses~~ hecho el 25-oct-2026 (CAMBIO_bloques_y_revision.md): el examen elige entre cuatro reglas del nivel; en el parecido al real gana `sin_nivel`.
- El reparto por cuota usa el nivel total, que se infla cuando aparece una región nueva.
- Incluir el error de la tasa de la región en la banda de la reentrada.
