"""
entorno_local_ejemplo.py — La plantilla de lo privado de tu máquina.

La primera vez: copia este fichero como entorno_local.py, en la misma carpeta que main.py, y rellénalo. A partir de
ahí, main.py, ejecutar_informe.py, el explorador y cualquier script que use production_configuration() leen de
entorno_local.py, y puedes sobrescribir main.py con cada versión nueva sin volver a tocar estas líneas.

entorno_local.py no viaja en el zip del proyecto: no lo compartas ni lo subas a ningún sitio. Lo único obligatorio es
el primer bloque; el resto es opcional (déjalo en None para usar el valor de main.py).
"""

# ─── obligatorio ─────────────────────────────────────────────────────────────────
SQL_SERVER = "..."                                   # el servidor, p. ej. "MISERVIDOR" o "MISERVIDOR\\INSTANCIA"
SQL_DATABASE = "..."                                 # la base de datos
SQL_DRIVER = "ODBC Driver 17 for SQL Server"         # el driver ODBC instalado en tu máquina
SQL_SCHEMA = "dbo"                                   # el schema donde se escriben y se leen las tablas sff_*
RAW_EXTRACT_QUERY = "SELECT * FROM ..."              # la query del extracto que lee el paso 1.1

# ─── opcional ────────────────────────────────────────────────────────────────────
SQL_USER = None                 # None = autenticación de Windows (trusted_connection); si no, el usuario de SQL
SQL_PASSWORD = None             # solo con SQL_USER. Mejor no escribirla aquí: léela de una variable de entorno,
                                # p. ej. import os; SQL_PASSWORD = os.environ.get("SFF_SQL_PASSWORD")
TABLE_PREFIX = None             # None = "sff_"
CURRENT_MONTH = None            # None = el mes en curso de main.py; p. ej. "01/11/2026" para que no dependa de main.py
