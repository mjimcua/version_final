# CAMBIO · La adquisición simulada completa el objetivo del equipo de adquisición (27-oct-2026)

## En una frase

La adquisición simulada salía más alta de lo que la evolución justifica: en los últimos meses el nivel sube y la tendencia es a la baja. Ahora el paso 5.1 completa el año de cada subregión hasta el forecast del equipo de adquisición. Lo vendido de enero al último mes cerrado más lo simulado hasta diciembre es igual a su objetivo, y la mezcla de productos de la simulación no cambia.

## Los objetivos

`objetivos_adquisicion_2026.csv`: una fila por subregión (`tr_regional_level_3`), en miles de USD, con la región del equipo como referencia. La línea LATAM del equipo (5.861) incluía Canadá. Las subregiones suman 24.999 (el total del equipo, 25.000, redondeado).

| Región del equipo | Miles de USD |
|---|---|
| Europe | 8.130 |
| Russia++ | 6.294 |
| LATAM (sin Canadá) | 5.311 |
| Canada | 549 |
| META | 1.820 |
| APAC | 2.895 |

En `main.py` (producción):
- `acquisition_targets_file = objetivos_adquisicion_2026.csv`;
- `acquisition_target_column = "tr_regional_level_3"`;
- `acquisition_target_unit_usd = 1000`.

Para otro año o un objetivo nuevo, se cambia el CSV.

## Cómo se completa (corregido el 27-oct-2026: todos los plazos)

El objetivo cuenta todos los plazos, así que el cálculo se hace con todos los plazos y solo la parte de 1 año entra en el horizonte. Para cada subregión:

1. **Lo vendido este año**, de enero al último mes cerrado, en 1, 2 y 3 años. Una licencia de t años vendida en el mes m es la fila de plazo t que vence en m + 12 t.
2. **La proporción por plazo**, la de las ventas de este año. Si el extracto todavía no llega al vencimiento de los plazos largos vendidos este año (un 3 años vendido en septiembre de 2026 vence en septiembre de 2029), se usa la de los últimos 12 meses en que el extracto ya muestra todos los plazos. En ese caso lo vendido este año en 2 y 3 años se estima desde lo de 1 año: vendido de 1 año / su parte. Se avisa en pantalla.
3. **Lo que falta** = objetivo − lo vendido en todos los plazos.
4. **Lo que se simula hasta diciembre** = lo que falta, repartido por plazos con esa proporción.
   - **1 año:** es la única parte que vence en el horizonte. La simulación de 1 año de la subregión se multiplica por un único factor hasta esa cantidad, así que su mezcla de productos, valores de adquisición y meses no cambia: Black Friday sigue siendo el pico que fue.
   - **2 y 3 años:** se cuentan en la tabla para que el total cuadre, pero vencen en 2028 y 2029 y no generan filas.

Resultado: vendido + simulado, en todos los plazos, es igual al objetivo, y la parte de 1 año es algo menos que el objetivo (los 25.000 son de todos los plazos).

Con `acquisition_target_all_terms = False` el objetivo se toma como de 1 año solo.

**Casos que se avisan:**

| Caso | Qué se hace |
|---|---|
| Subregión del objetivo que no está en los datos | Se avisa (un nombre que revisar) |
| Subregión de los datos sin objetivo | Se queda con la simulación del método |
| Ya ha vendido su objetivo | No se simula más |
| Hay hueco pero nada que escalar (no vendió un año antes) | Se avisa |

**En pantalla:** la tabla ADQUISICIÓN HASTA EL OBJETIVO tiene una fila por subregión con:
- su objetivo;
- lo vendido, en total y por plazo;
- la proporción de cada plazo;
- lo que falta;
- lo que simulaba el método en 1 año y el factor;
- lo simulado por plazo, el final y la diferencia.

Añade una línea VALORACIÓN con lo simulado en todos los plazos y en 1 año.

**Comprobación nueva del 5.1 (aviso):** cada subregión acaba el año a menos del 1 % de su objetivo, y ninguna queda sin datos o sin forma.

## Lo que hay que mirar en la ejecución real

- **Los nombres.** En los datos parecidos al real (nombres inventados) no se encuentran Caucasus, Israel, NOLA, NWC Africa, South East Asia ni Stan. Si en el extracto real salen en la lista de «región_sin_datos», hay que poner en el CSV el nombre que usa el extracto, o la región del nivel en que esté.
- **La medida.** Lo vendido se mide con el USD de las filas de adquisición del extracto (el valor del contrato que vence). Si los miles de USD del equipo son otra cosa (facturación con o sin impuestos, otra divisa), el factor lo arrastra. El factor de la tabla lo enseña: algo muy lejos de 1 (×10) en muchas subregiones indica medidas distintas.
- **El examen de la captación** sigue examinando el método, no el objetivo: el objetivo no se puede examinar con el pasado.
- **La comprobación del 5.1 sobre lo simulado frente al año anterior** (0,5 a 2 veces) puede avisar si el objetivo pide mucho más o mucho menos que el año anterior. Es el aviso que interesa leer.

## El paso 5.1 es ahora solo adquisición, con su gráfico (27-oct-2026)

**Qué cambia:**

- **El bloque 5 se llama «Adquisición».** El paso 5.1 lee lo vendido en 1, 2 y 3 años, simula la ventana, la completa hasta el objetivo, examina el método y dibuja la adquisición. Todo lo de adquisición está ahí.
- **Las renovaciones proyectadas pasan al paso 7.2.** Son la pipeline real que vence entre el mes en curso y diciembre: renueva según el forecast y vuelve a vencer el mismo mes del año siguiente. No es adquisición y necesita el forecast, así que se construye donde se calcula. Su comprobación (todo valor que toma al renovar existe en la historia) también pasa al 7.2. El 5.1 tiene 2 comprobaciones (3 con objetivos) y el 7.2, 11.
- **Gráfico nuevo, siempre en pantalla: `05_1_adquisicion_mes_a_mes.png`.**
  - Panel de arriba: la adquisición total por mes de venta desde enero de hace 3 años, con la suma de cada año. En el año en curso pone vendido + simulado frente al objetivo.
  - Panel de abajo: una línea por plazo (1, 2 y 3 años).
  - Real en línea continua, estimado en punteada (vendido, pero su plazo vence después del extracto) y simulado en discontinua.
- **El informe:** el capítulo 5 se llama «B5 · La adquisición» y trae la adquisición por año y plazo, la tabla hasta el objetivo y lo que el 5.1 no hace.
- **Config:** `acquisition_chart_history_years = 3`. Las etiquetas de `renewal_reentry_discount`, `renewed_acquisition_value` y `dims_after_renewal` pasan a [7.2].
- **Diagrama:** se llama «D1 · Mapa de bloques» y va por la v2 (27-oct-2026), en el documento de los 8 bloques.

**Qué hay que mirar con datos reales.** En el gráfico, la línea de 2026 tiene que seguir la de años anteriores. Si lo vendido hasta septiembre ronda el 75 % del objetivo, Oct–Dic saldrá en el nivel de siempre. En los datos parecidos al real, el sintético vende unos 0,5 M$ al mes frente a un objetivo de 25 M$, así que Oct–Dic salta a 6–7 M$ al mes: es la escala del sintético, no el método. Un salto así con datos reales querría decir que lo vendido y el objetivo no se miden igual. El factor de la tabla lo enseña por subregión.

## Pendiente

Revisar si el retail to subscription tiene el mismo problema (un nivel por encima de lo que la tendencia justifica).

## Comprobado

`test_blocks_3_to_5_and_8_5`, bloque B2, con tablas finas hechas a mano.

Con plazos, EU ha vendido $16.000, mitad en 1 año y mitad en 2. Con un objetivo de $40.000 se simulan $12.000 en 1 año (las únicas filas) y $12.000 en 2 años.

Sin plazos:
- EU vendió $14.400 hasta agosto; con un objetivo de $30.000 se simulan $15.600 (factor 2,17);
- la mezcla de productos se mantiene (A vende el doble que B, como un año antes);
- una región sin objetivo se queda como estaba y una región del objetivo que no está en los datos se avisa.
