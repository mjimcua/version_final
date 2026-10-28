# CAMBIO · Las renovaciones se proyectan según el plazo al que renuevan (28-oct-2026)

## En una frase

Una licencia no siempre renueva a su mismo plazo, y eso cambia cuándo vuelve a vencer y con qué valor. El paso 7.2 reparte ahora la renovación de cada licencia de la ventana (del mes en curso a diciembre) por la proporción de cada plazo de destino aprendida en los meses cerrados. Necesita seis columnas nuevas en el extracto.

## Por qué

| Transición | Dinero el año en que renueva | Pipeline del año siguiente |
|---|---|---|
| 1 → 1 | a su revalorización | vuelve a los 12 meses |
| 1 → 2 o 3 | más dinero (se pagan 2 o 3 años) | **hueco**: vence a los 24 o 36 meses |
| 2 o 3 → 1 | a su revalorización, que puede ser negativa | **vuelve a los 12 meses**; antes no se construía |
| 2 → 2, 3 → 3 | a su revalorización | vence a los 24 o 36 meses |

Hasta ahora toda la renovación de 1 año se trataba como 1→1, y las licencias de varios años no se proyectaban. Se construía de más lo que pasa de 1 a 2 años, valorado además al precio de 2 años, y no se construía lo que pasa de 2 a 1. El cociente de la 7.3b mide el neto de las dos cosas, y un cociente cerca de 1 puede esconder dos efectos grandes que se compensan.

## Las columnas del extracto

Son medidas de mes cerrado, como las de las renovaciones aisladas: vacía en un mes cerrado = 0, se borran desde el mes en curso y llegan al núcleo solo en los meses cerrados.

| Columna del extracto | Campo de la Config | En el núcleo |
|---|---|---|
| `total_renewed_units_to_1y` | `renewed_units_to_1y_col` | `forecast_renewed_units_to_1y` |
| `total_renewed_units_to_2y` | `renewed_units_to_2y_col` | `forecast_renewed_units_to_2y` |
| `total_renewed_units_to_3y` | `renewed_units_to_3y_col` | `forecast_renewed_units_to_3y` |
| `total_renewed_usd_to_1y` | `renewed_usd_to_1y_col` | `forecast_renewed_USD_to_1y` |
| `total_renewed_usd_to_2y` | `renewed_usd_to_2y_col` | `forecast_renewed_USD_to_2y` |
| `total_renewed_usd_to_3y` | `renewed_usd_to_3y_col` | `forecast_renewed_USD_to_3y` |

Se declaran las seis o ninguna; la Config se para si falta alguna o si no hay `term_column`. El plazo de origen es el de la fila (`tr_term`).

**En `main.py`:** `TERM_TRANSITION_COLUMNS_IN_EXTRACT = True`, activado desde el extracto del 28-oct-2026, que ya las trae. Con un extracto sin ellas se vuelve a poner `False` y todo funciona como antes. Ese extracto trae también `dormant_month_0` … `dormant_month_12`, las columnas de maduración de dormant. Por eso dormant pasa a ser un estado con fecha: `maturation_dims=["softcancel", "dormant"]`. Su curva tiene que aprobar el examen del paso 4.1; si no, dormant toma su proporción histórica, como hasta ahora.

## La regla (el supuesto del forecast)

- **Proporción constante:** la suma de los últimos `term_transition_window_months` (12) meses cerrados, por plazo de origen.
- **Nivel por soporte** (`term_transition_levels`; en producción, `TERM_TRANSITION_LEVELS` de `main.py`):
  1. subregión × producto;
  2. región × producto;
  3. producto;
  4. cartera.

  Cada grupo toma el primer nivel con `term_transition_min_renewed_units` (300) renovaciones en esos 12 meses; la cartera siempre vale. Un plazo de origen sin nada aprendido renueva a su mismo plazo.
- **En unidades y en USD por separado:**
  - pipeline proyectada en unidades = unidades esperadas × la parte de las unidades que renueva a 1 año;
  - en USD = USD esperado × la parte del USD que renueva a 1 año.

  Así la pipeline de una 2→1 lleva su propio valor, y la de una 1→1 no arrastra el precio de las que pasan a 2 años.
- Lo que renueva a 2 o 3 años no vuelve dentro del horizonte (el año que viene). La fila proyectada toma el plazo de 1 año y su nivel generado (`tr_term_level_1`).

## Lo que cambia en la salida

- **Paso 7.2:**
  - una acción dice de qué plazos vienen las renovaciones proyectadas (USD por plazo de origen);
  - una comprobación (aviso): el plazo de destino se conoce para el 99 % (`term_transition_known_share_min`) o más de las unidades renovadas de los meses cerrados. Por debajo, hay renovaciones a otro plazo o columnas sin rellenar;
  - en pantalla, la tabla TRANSICIONES DE PLAZO: la proporción de la cartera por plazo de origen, los grupos con soporte de cada nivel y el SUPUESTO escrito.
- **Tabla nueva `sff_transiciones_plazo`:** una fila por nivel × grupo × plazo de origen, con las renovaciones, la parte de cada destino en unidades y en USD, y si tiene soporte.
- **Paso 1.2:** comprueba que cada columna no supera su total (las unidades renovadas o el USD renovado), como las demás medidas de mes cerrado.
- **Informe, 8.10 · ¿A qué plazo renuevan las licencias?** Por plazo de origen, la parte de cada destino mes a mes, su revalorización (USD renovado / USD que vencía de esas unidades) y su USD por año.

## También en el informe: 8.9 · ¿Dónde se han detectado subidas de precio y desde cuándo?

- Por grupo de precio con alguna subida (los 12 con más dinero renovado dentro de ella): la revalorización que lee el detector (renovaciones aisladas), la de todas las renovaciones y la media de los 12 meses anteriores, con los 12 meses de cada subida en naranja y su inicio y escalón.
- Anexo con todas las subidas del calendario: desde, hasta, escalón, revalorización antes y durante, y USD renovado dentro.

Sirve para comparar con las subidas de tarifa que se hicieron. La detección es por `price_group_dims` (región × producto), no por plazo ni banda.

## Comprobado

`test_term_column.py`, bloque C: un extracto donde el producto A (1 año) renueva el 80 % de sus unidades a 1 año (70 % del USD) y el resto a 2, y el B (3 años) el 40 % a 1 año (25 % del USD) y el resto a 3.
- La pipeline proyectada es lo esperado × esas partes, en unidades y en USD.
- Las licencias de 3 años de la ventana se proyectan.
- La tabla tiene las partes aprendidas, y el supuesto y la comprobación salen en pantalla.

Sin las columnas, el bloque A sigue comprobando que solo se proyecta la renovación de 1 año.

El sintético parecido al real trae las seis columnas con «cada licencia renueva a su plazo», así que su forecast no cambia.
