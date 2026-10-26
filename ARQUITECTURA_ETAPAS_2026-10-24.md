# Arquitectura por etapas · 24-oct-2026

Pregunta: ¿cada etapa tiene una misión clara, recibe lo que necesita y ofrece algo a las siguientes? El criterio de Miguel Angel: cada etapa mejora los datos, mejora el conocimiento de los patrones o mejora la predicción. Además, cada predicción debe vivir en una sola caja, con una interfaz clara, para poder mejorarla por dentro sin tocar nada más.

## Conclusión

- **Lo que ya está bien:**
  - el orquestador es un grafo explícito: cada etapa declara qué lee y qué escribe;
  - cada módulo arranca con su misión, sus acciones, sus checks y su salida;
  - las etapas de datos (00–08) están limpias.
- **Lo que no está bien:** las cuatro predicciones del forecast no siguen el mismo patrón. Solo dos tienen caja propia:

| Predicción | Aprende | Se examina | Se aplica | ¿Una sola caja? |
|---|---|---|---|---|
| **Marcas** (maduración) | MD | MD (curva) · 19 (examen honesto, la misma caja) | 17 la llama | **Sí, desde hoy** |
| **Precio** (uplift) | 15 (celdas, factor de marca) | 16 (veredictos) | 17 | Casi: la estimación sí; 15 mezcla estimación con mucho diagnóstico |
| **Tasa** | 10–12 (escalera, composición) · 14 (técnica) | 14 (por pool) · 19 (por serie y total) · AUD | 17 | **No:** la predicción a un horizonte se calcula en tres sitios (17, 19 y AUD), y el repuesto de una serie sin composición vive en 17 |
| **Pipeline futura** (horizonte extendido) | 17 | nadie | 17 | **No:** está dentro del paso 17 y no tiene examen |

- **Paso 17 hace cuatro cosas:**
  - predice la tasa de cada fila a su horizonte;
  - inventa la pipeline futura (proyectada y simulada);
  - valora (unidades × tasa × uplift);
  - suma con bandas.
- La propuesta es que 17 sea solo **el montaje**: llamar a las cuatro cajas, multiplicar y sumar.

## El mapa de hoy

| Etapa | Misión | Tipo | Lee | Ofrece a |
|---|---|---|---|---|
| 00 | el extracto encaja con la Config | datos | extracto | 01 |
| 01 | separa time series, valida valores, crea niveles `_level_1` | datos + patrones | 00 | 02, 20 |
| 02 | rol de cada mes; borra lo que aún no se sabe | datos | 01 | 03 |
| 03 | ids de cada fila (serie, unidad, celda de uplift) | datos | 02 | casi todas |
| 04 | suma las filas finas de cada unidad | datos | 03 | 06, 08, 17 |
| 06 | ruta de cada serie (estimable, solo historia, solo futuro) | datos | 04 | 08–10 |
| 08 | rellena huecos; tasa, soporte y signo de cada serie | datos | 04, 06 | 09–13, 17, 19 |
| 09 | qué dimensión separa la tasa; orden de colapso | patrones | 08 | 10 |
| 10–11 | escalera y credibilidad | predicción (tasa) | 08, 09 | 12, 17, 19 |
| 12 | serie mensual de cada composición | predicción (tasa) | 10, 11 | 13, 14, 17, 19 |
| 13 | dinámica: φ, tendencia, estacionalidad | patrones (no decide) | 12 | AUD, NU, IN |
| 14 | elige técnica por composición y horizonte; bandas | predicción (tasa) + examen | 12 | 17, 19 |
| 15 | uplift de cada celda; factor de marca; diagnósticos de precio | predicción (precio) + patrones | 03 | 16, 17 |
| 16 | examen del uplift: vía con descuento, factor de marca | examen (precio) | 03 | 17 |
| **MD** | **caja de las marcas: cuántas unidades llevarán marca al vencer** | **predicción (marcas)** | 03 | 17, 19 |
| 17 | tasa a cada horizonte; pipeline futura; valoración; totales | predicción + montaje | 03–16, MD | 20, 21, AUD, NU |
| 19 | examen por serie y del total frente a la baseline; examen honesto | examen (tasa, marcas) | 08–14, MD | NU, 18, IN |
| 20 | time series y total por año y origen | predicción (otro universo) | 01, 17 | NU, IN |
| 21 | filas añadidas y borradas | informe | 03, 08, 17 | IN |
| AUD | satélites del núcleo (técnicas, referencias) | salida | 10–17 | NU, IN |
| NU | el núcleo: toda la historia de cada fila | salida | casi todas | 22, 18, IN |
| 22 | tablas auxiliares de Power BI | salida | NU | — |
| 18 | la cadena cuadra de punta a punta | control | casi todas | IN |
| IN | informe y resumen para negocio | salida | todas | — |

## Lo que propongo, por orden de valor

1. **Paso 19 antes del 17, y la banda oficial del total desde el examen.**
   - Hoy 19 corre después de 17 sin depender de él.
   - Si va antes, 17 publica como banda oficial la del examen en vez de la de cuadratura (±0,1 % en el run real, cuando el examen dice ±1,0–1,2 %).
   - Es el pendiente más visible para negocio. Coste bajo: cambiar el orden y que 17 lea el resumen del examen.

2. **La caja de la tasa (`RateBox`), como MD.**
   - Hoy la predicción de una composición a un horizonte se calcula en 17 (`pool_predictions`), en 19 (`predict_month`) y en AUD (`composition_forecast_all`). Las tres pasan por `prediction.py`, pero cada una arma su historia y su repuesto.
   - Una caja con `aprender(meses hasta el origen)` y `predecir(serie, h)` la usarían las tres. El repuesto que respeta la marca (arreglado hoy en 17) viviría ahí.
   - Coste medio-alto: toca 17, 19 y AUD. Lo que se gana: lo que se examina es exactamente lo que se usa, y AUD deja de recalcular (hoy es el 16 % del tiempo).

3. **La caja de la pipeline futura (etapa PX).**
   - La proyección de renovaciones y la simulación de captación salen de 17 a su propia etapa, con su interfaz.
   - Se le añade un examen: simular la captación de meses ya cerrados y compararla con lo que se vendió. Hoy no tiene ninguno, y en el run real decide $4,1M de 2027, con la captación Not-New a 0,45×.
   - Coste medio.

4. **El paso 15 en dos:**
   - la estimación (celdas, factor de marca, subidas en juego), que es lo que leen 16 y 17;
   - los diagnósticos de precio para negocio (revalorización, regla del contrato, histogramas, subidas), que solo lee IN.
   - Son unas 600 de sus 1.000 líneas. Coste bajo, sin cambio de números.

5. **Cosmético:** la validación lleva el número 18 pero corre la penúltima. Se puede renombrar a una etiqueta de control (por ejemplo `VA`). No lo haría sin tu visto bueno: cambia nombres de tablas y de tests.

## Hecho hoy

- **Etapa MD (`step_md_maturation.py`).** Toda la maduración está en una caja:
  - `MarkMaturation(fine_table, origen, config).mature(unidades)` devuelve las unidades de cada grupo, mes y combinación de marcas al vencer;
  - 17 solo valora, y 19 examina con la misma caja;
  - con `distribute_marks=False` la caja es la identidad y el informe muestra maduración 0.
- **Dato y predicción separados.**
  - El extracto no se toca: `s00_`/`s02_` y `s17_vencen_*` = hoy.
  - La predicción vive en columnas propias:
    - `s17_vencen_*_al_vencer` y `s17_maduracion_unidades`;
    - `s17_esperado_*_hoy` (con las marcas de hoy) y `s17_esperado_*` (el forecast).
  - Las unidades que llegan a una serie con marca son filas de predicción (`origen_fila = maduracion`): sin dato, salen de la neutra a su precio medio y renuevan a la tasa y el uplift de la serie a la que llegan.
  - En el núcleo: `forecast_to_renew_*_at_due` y `forecast_maturation_units`.
- **Repuesto de tasa que respeta la marca.** Una serie sin composición toma la tasa de su celda mandatory **con su combinación de marcas**, después la de su combinación en la cartera y por último la global. Antes, una serie softcancel sin historia tomaba la tasa de su celda (casi toda neutra).
- **Precio por marca.** El paso 15 mide el factor de cada combinación, el 16 lo examina (segundo veredicto) y el 17 lo aplica si gana.

## Comprobación en el sintético parecido al real

- **El forecast apenas cambia, como debe:** 2027 pasa de $24,79M a $24,76M (−0,1 %). Por meses, entre −0,6 % (oct–dic 2027, por el repuesto que respeta la marca) y +0,1 %.
- **Examen:**
  - la baseline y el framework, idénticos;
  - el examen honesto a 6 meses, 1,5 % (antes 1,4 %): ahora usa la misma caja que el forecast.
- **Coste:** 283 s frente a 257 s (+10 %).
  - La etapa MD tarda 22 s, pero 17 baja de 33 s a 20 s.
  - El examen sube 11 s (una caja por origen).
  - El núcleo sube 12 s, porque las filas de predicción de la maduración son muchas: 160.535 frente a 66.336 filas futuras del extracto (una por serie de destino, mes y celda de uplift).
  - En el real (279.000 filas futuras) podrían ser unas 600.000 filas más en `sff_nucleo`.
  - Si pesa, se pueden agrupar por serie de destino y mes con el uplift ponderado por USD: es exacto, porque la tasa solo depende de la serie y el horizonte.
