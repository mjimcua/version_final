# CAMBIO · Memoria con el extracto real y maduración por región (26-oct-2026)

## En una frase

La ejecución real se quedó sin memoria (`MemoryError`, 1.128.657 filas). El framework guardaba varias copias completas del extracto durante toda la ejecución y escribía el núcleo de golpe; ahora guarda una y escribe por tramos. Además, el paso 4.1 enseña la maduración de cada estado por región. Ningún número del forecast cambia: en el parecido al real, 2026 da 26.447.337 $ y 2027 da 25.766.206 $, como antes.

## 1 · La memoria

### Dónde se iba

Medido con el parecido al real ×3 (1,12 M filas, 1,2 GB el extracto) en una máquina de 7 GB:

| Dónde | Qué pasaba | Coste |
|---|---|---|
| Bloque 1 | guardaba cinco copias del tamaño del extracto (el leído, el validado, el de niveles, el del calendario y la tabla fina), y las cinco seguían vivas hasta el final | 4,5 GB al cerrar el bloque 1 |
| 3.1 y 3.3 | los ids del padre de cada celda de uplift se calculaban sobre una copia entera de la tabla fina, una vez en 3.1 y tres en 3.3 | +2 GB de pico en 3.3 |
| Bloque 3 | `uplift_basis` (las filas de renovadores) salía del bloque y vivía toda la ejecución, aunque solo la lee el 3.2 | memoria fija hasta el final |
| 4.1 | la curva y la caja copiaban la historia entera para añadirle dos claves | copias pasajeras |
| 8.3 | el núcleo se escribía en una sola llamada a `to_sql`, que convierte cada valor en un objeto Python antes de enviarlo: con 1,3 M filas × 95 columnas, varios GB encima del núcleo | la máquina de pruebas se quedó sin memoria aquí |

### Qué cambia

- **Bloque 1:**
  - cada copia se libera en cuanto el paso siguiente tiene la suya, y solo queda la tabla fina;
  - del extracto se guarda `extract_summary`, con filas, columnas, unidades y USD que vencen. Es lo único que leían de él los pasos 8.5 y 8.6 y la huella de `persist_block`;
  - ya no se dan `raw_extract`, `validated_extract`, `leveled_raw` ni `calendared_raw`. Para ver una copia intermedia, se llama a su paso: `validate_raw`, `validate_values_and_build_levels`, `apply_calendar`.
- **3.1 y 3.3:** los ids se calculan sobre una fila por celda, no sobre la tabla entera. Es el mismo resultado: los ids salen de las dimensiones que definen la celda.
- **Bloque 3:** `uplift_basis` se queda dentro del bloque y se libera al acabar el 3.2.
- **4.1:** la curva y la caja leen solo las columnas que usan.
- **Escritura en SQL** (`write_table` y la persistencia del bloque 1): se escribe por tramos de 100.000 filas (`SQL_WRITE_SLICE_ROWS`, en config.py). El primer tramo crea la tabla y los demás se añaden. Las columnas de texto vacías en el primer tramo toman su tipo de la columna entera, para que la tabla no se cree con un tipo equivocado.

### Resultado

| | Antes | Ahora |
|---|---|---|
| Pico del bloque 1 (×3) | 4,75 GB | 2,72 GB |
| Memoria al cerrar el bloque 1 (×3) | 4,51 GB | 2,48 GB |

El pico de la ejecución completa, medido en ×2, está en la nota de entrega.

### Segunda ronda: el error de la ejecución real en el 3.2 (`ArrowMemoryError`)

La ejecución real paró en el 3.2 (`show_series_with_increases`, la línea `money = (closed[closed["_grupo"].isin(...`) con `ArrowMemoryError: malloc of size 219929728 failed`.

| Dónde | Qué pasaba | Ahora |
|---|---|---|
| 3.2, las series con subida | copiaba todas las filas cerradas con todas sus columnas para añadirles el grupo de precio, y después las filtraba y las volvía a copiar | toma solo las filas de los grupos con subida y solo las 7 columnas que suma (serie, mes, unidades y USD que vencen y renuevan) |
| 3.2, lo que vence en el ciclo de una subida | copiaba todas las filas futuras con todas sus columnas | solo las dimensiones del grupo de precio, el mes y el USD que vence |
| 3.1 y 3.3, las filas de renovadores | copiaba las filas cerradas enteras, las filtraba y las volvía a copiar para añadir los ids del padre; tres veces en el bloque | elige las filas primero y las copia una vez, sin las 13 columnas de maduración, que el bloque 3 no lee |

Con el parecido al real, el forecast y el veredicto del uplift salen idénticos a la ejecución anterior.

## 2 · La maduración por región (paso 4.1, acción 3)

Para cada estado negativo y cada región del primer nivel (`tr_regional_level_1`), se ve qué parte del estado final ya es visible k meses antes de vencer. Sale una tabla k × región en pantalla y un gráfico, `graficos/04_1_maduracion_por_region.png`, con un panel por estado y una línea por región. La cartera va en gris.

- **Estado con fechas (softcancel): la curva del PASADO.** Es la media de las cohortes cerradas de las que aprende la caja (los últimos `maturation_history_months`), ponderada por unidades: Σ V_k / Σ M por región. A k = 0 vale el 100 %.
- **Estado sin fechas (dormant, not_installed): el pasado no se ve,** porque el extracto guarda solo el último estado. Lo que se ve es la pipeline HOY: de las unidades que vencen dentro de k meses, la cuota que ya está en el estado, sobre la cuota final de la región en los meses cerrados. En el gráfico va con línea discontinua. El forecast toma para estos estados la cuota final histórica.
- **Regiones:** como mucho `maturation_regions_shown` (8, en la Config), las mayores por unidades que vencen.
- **Resultado:** queda en `box.maturation_by_region`, una tabla larga con estado, vista, región, k, `visible_del_final`, `cuota_final` y unidades.

## Comprobaciones

- `test_block_1`: el resumen del extracto vuelve igual al reanudar, y el bloque no guarda ninguna copia del extracto.
- `test_blocks_3_to_5_and_8_5`: el 4.1 enseña la maduración por región, y un caso hecho a mano (dos regiones, dormant) comprueba cada número.
- `test_term_column` y `compare_with_real.py` leen la tabla fina en lugar del raw del calendario.
- Los textos del gráfico que no se dibuja ya no hablan solo del universo time series.

## 3 · La tabla final: el año por región (paso 8.6, acciones 5 y 8)

Al final de la ejecución sale el USD renovado del año en curso por `tr_regional_level_1` y `tr_regional_level_3`, en dos columnas y su suma:

- `usd_real_hasta_hoy`: lo renovado en los meses cerrados (`forecast_status = 'actual'`);
- `usd_esperado_hasta_fin_de_ano`: lo esperado desde el mes en curso hasta diciembre (`forecast_status = 'forecast'`);
- `usd_total_ano`.

Lleva una fila por región, el TOTAL de cada región del primer nivel y el TOTAL del año.

- **Sale de una consulta sobre `sff_nucleo`,** no de pandas: la tabla en pantalla es lo que da el SQL, y el SQL se imprime debajo para llevarlo a SQL Server o Power BI (`year_by_region_query`, en `report_queries.py`). Es el primer caso de la regla "lo que se enseña sale de una consulta".
- **Comprobación 4 del 8.6:** el TOTAL de la tabla es el TOTAL del año en `sff_forecast_total` (paso 7.3). En el parecido al real, 2026 = 18.994.045 $ reales + 7.453.292 $ esperados = 26.447.337 $.
- **Los niveles de región** son el primero y el último de `ts_region_columns`.
- **Tests:** `test_report_queries` compara cada subtotal con el núcleo sumado en pandas; `test_2_5_8_6_dynamics_report` comprueba que cada región suma su subtotal y el TOTAL, y que real + esperado = el año.
