# Cambio · bloque 1 (preparación) y vocabulario en la Config · 25-oct-2026

## Qué cambia

- **`vocabulario.py` ya no existe.** Sus 113 etiquetas están en `config.py`, en la sección THE VOCABULARY, al principio del fichero. Son constantes del módulo y no campos de la Config: si una cambiara entre ejecuciones se romperían las tablas y Power BI. Todos los módulos las importan de `config`.
- **Los pasos 00 a 06 están en un solo módulo, `block_1_preparation.py`.** Cada paso conserva su número, sus acciones, sus comprobaciones y su texto en pantalla (`[00]` … `[06]`). En el orquestador son un solo paso, `preparation` (etiqueta `B1`), y la tabla de tiempos sigue dando cada paso por separado.
- **`PreparationBlock`** guarda los resultados del bloque como atributos, en memoria:
  - `raw_extract`, `validated_extract`, `leveled_raw`, `time_series_rows`, `calendared_raw`;
  - `fine_table`, `forecast_units`, `series_table`, `rated_units`, `series_rate`.
- **Persistir y reanudar son opcionales.** Si no se llaman, todo sigue en memoria como antes.

## Uso en el notebook

```python
from block_1_preparation import PreparationBlock
import main

configuration = main.production_configuration()
block = PreparationBlock(configuration).run()        # pasos 00 a 06, en memoria
block.persist_block()                                 # opcional: a SQL (sff_bloque1_*)
results = main.run(configuration, preparation=block)  # el resto, desde el paso 07

# otro día, con el mismo extracto y el mismo mes:
block = PreparationBlock(configuration).resume_block_from_sql()
results = main.run(configuration, preparation=block)
```

## Persistir y reanudar

- **`persist_block()`** escribe lo que leen los pasos posteriores (seis tablas `sff_bloque1_*`) más dos tablas pequeñas:
  - el manifiesto, con cada columna y su tipo;
  - la huella: filas y USD del extracto, mes en curso y el valor de cada campo de la Config que lee el bloque.
  - Sin motor SQL, escribe CSV en `output_folder`.
- **`resume_block_from_sql()`** lee las tablas y devuelve a cada columna su tipo y a cada tabla su índice; el índice de la tabla fina es el id de la fila que usa el forecast. Vuelve a leer el extracto, porque los pasos 26 y 27 lo necesitan.
- **Se para si algo ha cambiado desde que se persistió:** el extracto, el mes en curso o un campo de la Config que lee el bloque. Dice cuál y pide volver a ejecutar el bloque.

## Comprobación

- **Tests:** 27 baterías, 455 comprobaciones, todas bien. `test_block_1.py` es nueva (11 comprobaciones):
  - una tabla va a SQL y vuelve igual, con sus tipos y un índice con huecos;
  - el bloque reanudado es idéntico al ejecutado, en SQL y en CSV;
  - el forecast y el examen desde un bloque reanudado son iguales a los de una ejecución completa;
  - se rechaza reanudar con otro mes o con otra Config.
- **Sintético parecido al real:**
  - la ejecución completa da exactamente los mismos números que antes del cambio (forecast por mes y por año, y examen);
  - la ejecución desde el bloque reanudado, también.

## Tiempos (sintético parecido al real, SQLite)

| | Segundos |
|---|---|
| Bloque 1 ejecutado (lectura incluida) | 22 |
| `persist_block()` | 35 |
| `resume_block_from_sql()` (lectura del extracto incluida) | 31 |

Con SQLite y este tamaño, reanudar tarda más que volver a ejecutar el bloque. Persistir sirve para tener el estado guardado e inspeccionable, no para ganar tiempo. Si se quiere velocidad, el sitio para el punto de guardado son ficheros parquet, no SQL.
