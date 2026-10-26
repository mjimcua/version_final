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

## 4 · El valor por unidad del universo time series (paso 6.1)

En la ejecución real el valor por unidad llevaba 20 meses cayendo, de 30 $ a unos 17 $ en 2026-09. La proyección lo tomaba de la media de los últimos 12 meses (unos 22,5 $), así que el dinero saltaba de unos 500 k$ a unos 650-700 k$ en octubre sin que subieran las unidades, y 2027 heredaba ese valor.

- **La ventana del valor por unidad la elige el examen**, junto con la regla del nivel. Los candidatos son cada regla × cada ventana de `ts_auv_window_candidates` (1, 3 y 12 meses), y simula el de menor error medio del total del mes. `ts_auv_window_months` fija una ventana.
- **`sff_ts_examen`** pasa a ser candidato × horizonte, con las columnas `candidato`, `regla_nivel` y `ventana_valor_medio`.
- **En pantalla, el 6.1 enseña el valor por unidad por región del primer nivel** en cuatro ventanas: último mes, últimos 3, últimos 12 y los mismos 3 de hace un año. Añade el cambio anual y una línea que separa el cambio del total en **precio dentro de las regiones** y **mezcla de regiones** (1 + total = (1 + precio) × (1 + mezcla)).
- **Una región que no vendió nada en una ventana corta** toma el valor por unidad del total, en vez de quedarse sin valor.
- **En el parecido al real** el examen elige `sin_nivel` con 3 meses (error medio 40,7 %, frente a 41,6 % con 12). El forecast cambia poco: 2026 da 26.475.182 $ y 2027 da 25.843.241 $. Con la caída del dato real, la ventana corta debería ganar con más claridad.
- **Test:** con un valor por unidad que cae un 2 % al mes, el examen elige 1 mes, y el error crece con la ventana (1 < 3 < 12).

## 5 · La consola: solo avisos, todo en el fichero

Con el extracto real, el navegador se quedó sin memoria mostrando la salida: unas 2.000 líneas y un centenar de tablas.

- **Nuevo `console_log_level`** en la Config. Con `"WARNING"` y un `log_file`, la consola (el notebook) solo enseña avisos, fallos y la tabla final; el fichero guarda todas las líneas y todas las tablas.
- **La Config de producción de `main.py`** ya lo trae: `log_file = salida/sff_ejecucion.log` y `console_log_level = "WARNING"`. Con `console_log_level=None` se vuelve a ver todo.
- **Test:** en `test_1_1_validate_raw`, la consola queda vacía con un paso limpio y el fichero tiene cada línea y cada tabla.

## 6 · Una curva de maduración que suspende su examen no se usa (paso 4.1)

La ejecución real dio 126,6 M$ en 2026, unos 5,5 M$ por debajo de lo esperado (124,2 M$ el 19-oct + 7,4 M$ de time series ≈ 132 M$), y una banda de ±6,5 M$. Las tres consultas sobre las tablas de la ejecución lo explican:

- **La maduración resta 6,4 M$ en 2026 (−17,6 %) y 12,8 M$ en 2027.** Mes a mes, octubre-diciembre: −1,52, −2,72 y −2,16 M$. La tasa en unidades cae de 0,64-0,66 (real) a 0,56-0,58 (previsto). El 19-oct la maduración restaba 0,84 M$ en todo 2026.
- **El examen ya lo decía:**
  - la curva de **dormant** falla la cuota final por 70,30 pp, lo mismo que sin curva;
  - el examen honesto (con los estados predichos) yerra un 23,0 % a 1 mes y un 11,9 % a 6, frente a 1,0 % y 0,9 % con los estados vistos (gana 0 de 6 meses).
- **La banda oficial usa el error del examen honesto,** así que ese 23 % es casi toda la banda: tasa 6,2 M$ de 6,5 M$; precio 0,18 M$; time series 0,31 M$.

**El cambio:**
- una curva solo se usa si su examen falla la cuota final por menos que los estados vistos entonces y por menos de `maturation_curve_max_error_pp` (5 pp, nuevo en la Config);
- si no, el estado toma su cuota histórica, como uno sin fechas, en el forecast (la caja de 4.1) y en el examen honesto (7.1); queda en `rejected_maturation_dims`;
- `maturation_dims_in_use` dice qué curvas se usan;
- 4.1 tiene una comprobación más (aviso), una línea `DECISIÓN` en pantalla, y la fila del veredicto del B4 dice que la curva no se usa.

En el parecido al real, la curva de softcancel pasa (se usa) y el forecast no cambia. Con los datos reales, dormant dejará de usar su curva: la maduración de 2026 debería volver a algo como el −0,8 M$ del 19-oct, y la banda al 1-2 % del examen.

**Queda por entender por qué las fechas de dormant no encajan con la curva:** la curva supone que un estado, una vez puesto, no se quita, y que `dormant_month_k` son las unidades que entraron en el estado k meses antes de vencer. Un dormant que se activa de nuevo antes de vencer rompe esa idea.


## 7 · El gráfico que explica la curva de maduración (paso 4.1)

Para contar por qué el forecast es este y no otro: «los estados del cliente maduran de esta forma; suponemos que las cohortes futuras maduran igual; eso proyecta X, y la curva acierta con este error según la antelación».

**Dónde está:** `salida/graficos/04_1_curva_de_maduracion_explicada.png`. Se ve siempre en el notebook, también con la consola en silencio (`show_figure(..., always=True)`), como la tabla final.

**Una fila por estado con fechas (softcancel, dormant) y tres paneles:**

1. **¿Cuándo se pone el estado?** Es una distribución de probabilidad.
   - Barras: f(k), la parte del estado final que se puso k meses antes de vencer (12+ = 12 o más). Suman 100 %.
   - Línea: F(h) = Σ f(k ≥ h), la parte del final ya visible h meses antes.
   - Marca: el mes en que ya se ve la mitad.
   - El título da el tope π: la parte de lo que vence que acaba en el estado.
2. **La pipeline de hoy, por mes de vencimiento.**
   - Barra oscura: lo que ya está en el estado hoy.
   - Barra clara: lo que añade la curva hasta vencer, (1 − visto) × p(h).
   - Línea discontinua: el tope de los últimos 12 meses cerrados.
3. **¿Cuánto acierta según la antelación?** El error medio de la cuota final en el examen (meses cerrados), a 1, 3, 6 y 12 meses:
   - con la curva;
   - sin curva (el estado visto entonces);
   - junto al umbral para usarla (5 pp).

Si la curva suspende su examen, la fila lo dice en naranja: «LA CURVA NO SE USA».

**Las fórmulas, para describirla de forma analítica.** Sobre las cohortes de los 12 meses cerrados, ponderadas por unidades (U = unidades que vencen, M = unidades en el estado al vencer, V_h = las que ya lo estaban h meses antes):

- π = ΣM / ΣU;
- f(k) = Σ entradas en k / ΣM;
- F(h) = ΣV_h / ΣM;
- p(h) = Σ(M − V_h) / Σ(U − V_h);
- previsión: M̂ = V + (U − V) · p(h).

**Los números salen del paso 4.1:** `maturation_explained(box, fine_table, state, configuration)` en `block_4_customer_states.py`. El gráfico (`draw_maturation_explained` en `graficos.py`) solo dibuja.

En el parecido al real (softcancel):

| Medida | Valor |
|---|---|
| Tope π | 7,6 % |
| La mitad del estado final ya se ve | 5 meses antes |
| Se pone 10 meses antes | 35 % |
| Se pone el mismo mes | 28 % |
| Error de la curva | 0,03-0,09 pp |
| Error sin curva | 2,15 pp a 1 mes → 7,15 pp a 12 meses |

**Tests:**
- `test_synthetic_real` comprueba que:
  - f suma 1 y F baja desde 1;
  - lo visto + lo añadido no pasa de 100 %;
  - la curva gana al estado visto en cada antelación;
  - el PNG existe;
  - una curva rechazada sale como «no se usa».
- `test_1_1` comprueba que, con la consola en silencio, solo se anuncian los gráficos `always`.
