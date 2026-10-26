# Estado y plan · 20-oct-2026

## Dónde estamos

- **El forecast acierta el total.** En el examen, el error del total es del 0,5 % a 1 mes y del 1,4 % a 6 meses, mejor que la hoja de cálculo y que la serie sola. Septiembre se predijo en $9,29M y ya hay $9,35M renovados.
- **Pero el examen es más fácil que la realidad.** Conoce las marcas FINALES de los meses que examina, y softcancel es casi el propio resultado: es el cobro que falla al renovar. El examen mide la precisión que tendríamos si acertáramos la mezcla de marcas; el error de predecir esa mezcla no lo mide nadie.
- **Las marcas mueven mucho el número:**
  - neutral renueva al 71 %, dormant al 45 % y softcancel al 28 %;
  - el reparto de marcas resta $5,5M en 2027 (−4,5 %);
  - en el sintético, sin el reparto el forecast se pasaría un +8 % a 12 meses.
- **Ya hecho y validado:**
  - el uplift con ventana sin subidas (sesgo plano del −1 %);
  - la regla del contrato: el +13 % mezcla poblaciones; ahora se examina sobre las neutras (ver más abajo);
  - la adquisición simulada corregida (de $71M a ~$7M que vencen);
  - el nivel de bandas cerrado: no hace falta más granularidad;
  - el sintético parecido al real, que conoce la verdad futura.

## Retos, por orden de impacto

1. **Mes en curso mal (2026-09).** Hay que repetir con 01/10/2026.
2. **La maduración de las marcas no se predice ni se examina.** Es la pieza nueva: se predice la tasa, el uplift… y también la mezcla de marcas. (Implementada para softcancel; ver el estado al final.)
3. **La banda del total es demasiado estrecha.** Enseña ±0,2 % (cuadratura), cuando el error real es del 1–5 % por mes. Tiene que salir del examen honesto.
4. **Dormant: corregido en el SQL (20-oct).**
   - Antes, sin información la licencia se marcaba como dormant, lo que inflaba el mes en curso (dormant hoy 9,6 % frente a 8,9 % final en septiembre).
   - Ahora, sin información cuenta como normal.
   - Queda un mes de retraso, porque el dato se calcula una vez al mes: el primer mes de una adquisición infravalora el dormant. Es un retraso constante, así que la curva de maduración, aprendida con los mismos datos, lo absorbe.
   - En la próxima ejecución hay que comprobar: la proporción y la tasa de dormant, el ajuste del reparto de 2027 y el detector de marcas.
5. **La tasa subió unos 4 pp en 2026 y las técnicas van por detrás.** El examen da −1,3 % a 1 mes y −3 % a 6 meses. Hay que vigilarlo antes de añadir tendencia.
6. **Elegir técnica por serie no aporta en el examen actual** (−0,11 pp frente al retador). Pero ese examen está condicionado a las marcas finales: series más limpias de lo que serán, en las que cualquier técnica funciona razonablemente bien. No se simplifica hasta medirlo de nuevo (elegida frente a retador) con el examen honesto.
7. **Datos pendientes:**
   - descuento desconocido en el 59 % del USD;
   - universo time_series vacío;
   - subidas de Brasil (+32 %) y México (+39 %) sospechosas;
   - Not-New a la mitad.
8. **Sintético con composición cambiante en el tiempo.** Aplazado: no es productivo ahora.

## Lo que se implementa ahora: la maduración de las marcas

### La convención del extracto (softcancel primero, dormant después)
- **El Config declara** qué marcas traen su maduración: `maturation_dims=["softcancel"]` y `maturation_months=12`.
- **Columnas por marca declarada:** `<marca>_month_0` … `<marca>_month_12` (13 columnas). `<marca>_month_k` son las unidades de la fila cuya marca se puso k meses antes del mes de vencimiento:
  - k = 0: el propio mes de vencimiento o después (el intento de cobro);
  - k = 12: 12 meses o más antes.
- **Filas con la marca a 1:** las 13 columnas suman `total_tr_units` de la fila.
- **Filas con la marca a 0:** nulos o ceros.
- **Filas de combinaciones** (dormant+softcancel): cada marca rellena sus propias columnas.
- **Si una marca declarada no trae sus columnas, el paso 00 falla.**

### Las piezas
1. **Paso 00 · validación.** Comprueba:
   - que están todas las columnas;
   - que son numéricas y ≥ 0;
   - que suman las unidades en las filas marcadas;
   - que van vacías en las no marcadas;
   - que no hay fuga de futuro: en un mes futuro a m meses vista, solo puede haber unidades en k ≥ m (una marca no puede tener fecha posterior a la foto).
2. **Pasos 03/04.** Las columnas viajan como medidas (se suman), no como dimensiones: no parten series.
3. **Curva de maduración** (módulo nuevo):
   - por marca y por grupo con soporte, qué parte de la marca final ya se veía a k meses;
   - la probabilidad de que una unidad sin marca a k meses la tenga al vencer;
   - los grupos se juntan hasta tener soporte, con credibilidad frente a la cartera, como en la escalera;
   - la curva se examina en los meses cerrados: se corta a k meses y se compara con el final real.
4. **Paso 17.** El reparto de las marcas con maduración usa la curva (lo visible hoy + la migración esperada de lo neutral) en vez de la proporción histórica a secas. Las marcas sin columnas siguen como ahora.
5. **Paso 19 · examen honesto.** Las marcas se reconstruyen tal como se veían a h meses y se aplica la curva aprendida con cohortes anteriores. Queda junto al examen actual, que pasa a llamarse "condicionado a las marcas finales". La diferencia entre los dos es lo que vale predecir bien las marcas.
6. **La banda del total** sale del error del examen honesto por horizonte.
7. **Sintético.** Solo lo necesario para probar: añadir `softcancel_month_k` a partir de la curva que ya simula.

## La regla del contrato (20-oct)

- **El descuento del extracto se mide contra el MSRP del contrato al vencer**, no al comprar. Una subida entre la compra y el vencimiento ya va dentro de d, así que la regla 1/(1−d) ya la incluye.
- **Lo que dije del +13 % era demasiado amplio.** Mezcla a todos los que renuevan: ofertas de retención, cambios de producto, softcancel. En las renovaciones típicas (isolated, 0–10 %, donde está el dinero) lo realizado es 1,07 frente a 1,04 de la regla, un 103 %. Con d contra el MSRP al vencer, ese +3 % ya no se explica por las subidas: queda por mirar qué lo da (¿cambios de banda o de plazo dentro de "mismo producto"?, ¿el mes exacto de la subida frente a la foto del MSRP?).
- **Decisión (Miguel Angel):** la regla es determinista y explicable; donde la renovación es neutra (sin softcancel, dormant ni not_installed) y hay descuento, se usa la regla. En el resto, el estadístico.
- **Implementado:**
  - vía nueva `contrato_neutras` en el examen del uplift (paso 16) y en el forecast (paso 17);
  - el examen elige entre estadística, contrato y contrato_neutras por WAPE en las filas con descuento conocido, y enseña el error de cada vía en las neutras y en las marcadas, y con y sin subida en ciclo (si el estadístico solo se pasa con subida, la está contando dos veces);
  - en el forecast, la regla se aplica a las filas neutras de hoy; las unidades que todavía tomarán una marca salen de ellas en el reparto de marcas, a la tasa en USD de su combinación (la madurez que decías);
  - sin ajuste configurable ("quitarle un punto"): si hace falta, el examen lo enseñará por tramo, y un ajuste aprendido de la realización es mejor que un número a mano.
- **El sintético no puede decidirlo.** No hace diferencias de precio entre neutras y marcadas, y realiza el 84 % de la regla por construcción, así que allí gana el estadístico (WAPE 7 % frente a 20 %). Lo decide el próximo run real. Lo que mirar:
  - en el paso 15, `realizacion_neutras` por tramo (cerca de 1 = la regla vale);
  - en el paso 16, la línea del veredicto y las filas "conocido · neutras".
- **El sintético ya mide el descuento como el extracto**: contra el MSRP al vencer (`DISCOUNT_AGAINST_MSRP_AT_EXPIRY`).

## Hecho el 20-oct (noche): limpieza, auditoría, baseline, resumen y gráficos

- **Limpieza.** Fuera los pasos 05 (lookups) y 07 (cota de soporte), las claves hash, la columna `universo` y 23 tablas SQL que nadie leía. El paso AUD ya no repite el examen del 19. En el sintético parecido, el mismo forecast bit a bit en 257 s en vez de 291; en el real, sumando `sff_fact_fine`, más de 4 minutos menos.
- **Auditoría** (`auditoria/AUDITORIA_CODIGO_2026-10-20.md`). Once errores que cambiaban números, ya arreglados; los dos de más peso son la elección de técnica, que se comparaba en meses distintos, y el k de credibilidad, que estaba sesgado. La maduración contaba dos veces las marcas del mes en curso.
- **"Retador" pasa a ser "técnica por defecto"** en todo el código y la salida.
- **La baseline** (celda mandatory, últimos 3 meses) es la referencia del examen. El paso 19 lleva la cascada baseline → escalera con la técnica por defecto → framework, en USD, con la ganancia y su consistencia. En el sintético parecido:
  - a 1 mes el framework no gana a la baseline (+$991 de error, mejor en el 50 % de los meses);
  - a 6 meses gana $20k al mes, mejor en el 83 %.
- **El resumen para negocio** sale en pantalla al final y abre el informe. Los **gráficos** están en `salida/graficos/` y enlazados en el informe.

## Pendientes nuevos

- **La banda oficial del total desde el examen.** La banda fila a fila sumada linealmente no es un intervalo del total: en 2027 sale $15,8M–$30M frente a un error típico de ±$353k según el examen. El resumen y los gráficos ya usan el examen; falta que el paso 17 la publique como banda oficial.
- **Examinar más lejos de 6 meses** (`backtest_horizons`), porque 2027 llega a h = 15.
- **Examen fuera de muestra de verdad.** Escalera, k, técnica elegida y bandas se estiman con todos los meses cerrados. O se re-estiman sin los meses de examen o se mantiene declarado.


- **Stay subscribe como time-varying.** Columnas `stay_subscribe_month_k`, como softcancel: quién aceptó la oferta y cuándo. Renuevan con otra tasa y otro uplift, y el softcancel que acepta tiene otro perfil que el que no. Con poco volumen, pero hay que separarlo para que no contamine ni la regla ni el softcancel.
- **Ciclo de subida de 24/36 meses** para licencias de 2 y 3 años. Hoy el detector asume 12 meses para todas.
- **Baseline como LA referencia.**
  - Definición: celda mandatory, tasa de los últimos 3 meses a T−h, sin escalera ni credibilidad ni marcas, mismo cálculo (`baseline_months=3`).
  - Las ganancias se cuentan frente a la baseline, en USD por mes y horizonte, y su consistencia en el tiempo.
  - La cascada: baseline → + escalera (técnica por defecto) → + elección de técnica → + extras → + marcas con maduración.
  - El retador pasa a ser "técnica por defecto", interno.

## Estado de la maduración (implementado 20-oct)

- **Paso 01:** valida las columnas `<marca>_month_k`, incluida la fuga de futuro.
- **`maturation.py`:** la curva por grupo, celda y cartera, y su examen.
- **Paso 17:** reparte con la curva.
- **Paso 19:** examen honesto junto al techo (marcas finales).
- **Resultado en el sintético:**
  - el error de la curva es ≤ 0,09 pp, frente a 2–7 pp con las marcas de hoy;
  - el examen honesto del total da un error del 1,0 % a h1 y del 1,5 % a h6, frente al techo del 1,0 % y el 1,4 %.
