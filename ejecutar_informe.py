"""
ejecutar_informe.py — El forecast contado, sobre las tablas sff_* de la última ejecución real. Sin parámetros:

    python ejecutar_informe.py

Usa la Config de producción de main.py, que toma el servidor, la base, el schema y el prefijo de entorno_local.py (lo
privado de tu máquina; la primera vez, cópialo de entorno_local_ejemplo.py). No vuelve a ejecutar el framework: solo
consulta las tablas que ya están en SQL Server y deja, por cada pregunta, su SQL, su CSV y sus gráficos en
salida/informe/. Las preguntas 6.1 y 6.2 escriben además sff_bandas_reales y sff_bandas_reales_resumen en ese schema.
"""

# ─── lo que se puede cambiar ─────────────────────────────────────────────────────
PREGUNTAS = []                 # [] = todas; o algunas, por ejemplo ["7.1", "7.2", "7.3", "7.4"] (la pipeline por componentes)
                               # o ["8.1", …, "8.10"] (entender 2027; 8.9 las subidas de precio detectadas, 8.10 las transiciones de plazo)
                               # o ["9.1"] (monográfico: cuánto cambia 2027 por la adquisición de 2026)

# ─── imports ─────────────────────────────────────────────────────────────────────
import os

import main
from informe_forecast import report


def ejecutar():
    print(f"[informe] main.py leído de: {os.path.abspath(main.__file__)}")
    configuracion = main.production_configuration()          # servidor, base y schema: entorno_local.py
    print(f"[informe] tablas {configuracion.sql_schema}.{configuracion.table_prefix}* · "
          f"mes en curso {configuracion.calendar_boundaries()['current']} · "
          f"preguntas: {', '.join(PREGUNTAS) if PREGUNTAS else 'todas'}")
    respuestas = report(configuracion, PREGUNTAS or None)
    fallidas = {numero: respuesta.note for numero, respuesta in respuestas.items() if respuesta.note.startswith("falló")}
    if fallidas:
        print("[informe] preguntas que fallaron (el resto sí se ha generado):")
        for numero, motivo in fallidas.items():
            print(f"    {numero}: {motivo}")
    return respuestas


if __name__ == "__main__":
    ejecutar()
