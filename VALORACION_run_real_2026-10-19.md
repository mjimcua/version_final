# Valoración de la ejecución real del 19-oct-2026

Fuente: salida de pantalla de la ejecución completa, con el mes en curso todavía en 2026-09. Por eso la ventana del uplift es 2025-09..2026-08 y el examen 2026-03..2026-08.

## Conclusión

1. **El total aguanta.**
   - En el examen, el framework falla el total de renovaciones un 0,5 % a 1 mes y un 1,4 % a 6 meses. La mejor hoja de cálculo falla un 1,7 % y un 4,3 %; la serie sola, un 1,2 % y un 2,2 %.
   - Septiembre sirve de prueba fuera de muestra gratis: por estar mal el mes en curso, se borró lo ya renovado ($9,35M) y se predijo $9,29M (−0,6 %; septiembre aún puede crecer).
2. **Tres cosas mueven el número más que cualquier elección de modelo:**
   - **El reparto de marcas:** −$5,5M en 2027 (−4,5 %), apoyado en la proporción histórica de dormant, que es la que va a cambiar con el arreglo del SQL.
   - **La tasa subió unos 4 pp en 2026 y las técnicas van por detrás:** el examen de los pools infravalora el total en todos los meses, un 1,3 % a 1 mes y un 3,0 % a 6 meses.
   - **La banda del total no es creíble:** ±$256k (0,2 %) para 2027, cuando el examen dice entre el 1 % y el 3 %.
3. **Varios puntos de decisión no aportan nada medible.**
   - La técnica elegida por id predice peor que el retador (−0,11 pp).
   - La agrupación de bandas da igual para el total.
   - Las extras se anulan primero.

   No son un riesgo, pero son complejidad sin retorno.

## Punto por punto

### A · Calendario (paso 02)
- **Lo que dice la ejecución:** mes en curso 2026-09.
  - Se borraron como resultado adelantado $9,35M de septiembre (229k unidades) y $4,55M de octubre.
  - Se borraron también $15,5M de pipeline parcial de 2027-09..12.
- **Valoración:** repetir con `current_month="01/10/2026"`. Septiembre entra en la ventana, el examen pasa a 2026-04..09 y se confirman (o no) los tres escalones pendientes.
- **Riesgo:** medio. Mientras tanto, septiembre se compara gratis: el forecast acertó a −0,6 %.

### B · Validación del extracto (paso 01)
- **Comprobaciones:** 49 ok y 1 WARN. Son 195 filas que renuevan sin valor de vencimiento de los renovadores; es despreciable.
- **Descuento desconocido:** 40 % de las filas, 59 % del USD que vence ($441M de $769M).
  - La regla del contrato solo podría usarse en el 41 %. Da igual, porque el veredicto del paso 16 es la vía estadística.
  - Las celdas de uplift sin tramo son gruesas. Es un tema del SQL.
- **Isolated coherente:** ratio medio de 1,069 en la historia; las seis bandas suman la base.
- **Universo time_series vacío:** 0 filas con `flag_time_series`.
  - La pregunta del retail a suscripción y su reentrada no se contesta en esta ejecución.
  - Si ese negocio existe, falta en el total.

### C · Niveles generados (paso 01)
- **tr_term:** 1 año renueva al 0,65 (12,8M unidades); 2+3 años al 0,54 (1,8M). Quedan separados, y es correcto.
- **tr_band:** un solo grupo.
  - Las bandas 1+2 son el 57 % de las unidades y renuevan al 0,62; las 3+4+5 son el 41 % y renuevan al 0,655. Hay 3 pp de diferencia.
  - Esa diferencia solo actúa en las series que se juntan en la pasada "without tr_band". Esa pasada lleva el dinero con suelo del 91 % al 98 %, es decir, afecta a un 7 % del dinero.
  - El sesgo es de ±1,5 pp como mucho sobre ese 7 %, y con signos opuestos, así que queda por debajo de 0,1 pp en el total.
- **Valoración:** se mantiene la regla del suelo. Cerrado.

### D · Dimensiones (paso 09)
- **tr_renewal_type domina:** eta² individual 0,32, contribución única 0,12; colapsa la última.
- **net_new y prev_OperationGroup tienen contribución única 0,00 cada una.** La razón es que se duplican entre ellas: al quitar una queda la otra, así que esa medida no ve el bloque.
  - Junto con renewal_type explican 0,58 (+0,26).
  - Se anulan antes que cualquier obligatoria (pasadas 2–3: dinero con suelo del 85 % al 90 %).
- **Riesgo:** bajo-medio. Falta medir la pérdida de quitar las dos a la vez.

### E · Escalera y nivel de riesgo (pasos 10–11)
- **Escalera:** de 21.540 a 6.069 grupos.
  - El dinero con suelo pasa del 82 % al 98 %, y el dinero con tasa propia del 46 % al 49 %.
  - Mejoran el error 23.698 de 40.681 fusiones.
- **Dinero por nivel de riesgo:**

  | Nivel | Dinero | Error |
  |---|---|---|
  | A_propio | 45 % | 2,05 pp |
  | A3_propio_reforzado | 35 % | 5,74 pp |
  | B_prestado | 8 % | 18,2 pp |
  | C_lejano | 9 % | 21,7 pp |
  | D_sin_historia | 2 % | — |
  | S_signo_bajo_suelo | 1 % | 31 pp |

- **USD con la tasa conocida a ±5 pp:** pasa del 50 % al 70 %.
- **Valoración:** cumple. El 17 % en B/C tiene unos 20 pp de error por serie y mes: sirve para el total, no para informar serie a serie.

### F · Dinámica (paso 13)
- **φ mediana 1,94:** las series se mueven unas 1,4 veces su ruido binomial.
- **Sin estacionalidad:** p = 0,971, amplitud 3,1 pp; solo noviembre se separa (−2,1 pp, con 3 observaciones).
- **Tendencia de cartera:** −0,3 pp/año con p = 0,35 sobre toda la historia, pero las tasas por año dicen otra cosa:
  - 2023: 0,67; 2024: 0,62; 2025: 0,63; 2026: 0,67.
  - En 1 año: 0,64 en 2025 y 0,69 en 2026.

  El test sobre toda la historia (con el pico de 2023) no ve la subida de 2026.
- **Tendencia en pools:** 2.156 pools tienen tendencia, el 69 % del dinero, y se compensan entre sí.

### G · Backtest de la tasa (paso 14): el punto más importante
- **Las técnicas están en el suelo del ruido.**
  - Las mejores a corto (holt_damped 1,36, ses 1,39, ma3 1,41, ma6 1,43, ewma 1,49) están todas en |err_norm| ≈ 1,4 ≈ √φ, y separadas 0,07 entre sí.
  - Elegir técnica por id es elegir ruido.
- **La técnica elegida frente al retador (T3_ma3):**
  - a corto, −0,11 pp (peor); a medio-largo, +0,04 pp;
  - error medio de la tasa por pool: 3,8 pp frente a 3,5 pp.
- **Holt-Winters y level_seasonal** son las peores de media, y la cartera no tiene estacionalidad. Aun así se eligen para 427 ids ($8,3M) a corto y 384 ($7,8M) a medio-largo.
- **El total del examen siempre sale negativo:**
  - a 1 mes: −1,5, −1,5, −0,6, −1,8, −1,4 y −1,0 % (−1,3 % en conjunto);
  - a 6 meses: −2,3, −2,8, −2,8, −4,1, −3,4 y −2,7 % (−3,0 %).

  Crece con el horizonte. Es el retraso frente a una tasa que sube unos 0,3 pp al mes, coherente con +4 pp en un año.
- **Bandas:** el 95 % y el 97 % caen dentro, con un 90 % nominal; son algo anchas.
- **Valoración:**
  - Dejar el retador como técnica por defecto, y cambiar solo si otra lo mejora por un margen en la selección (FVA).
  - Para 2027 (horizontes de 4 a 16 meses): si la subida sigue, el forecast se queda corto un 3 % o más; si ya paró, no.
  - Hay que mirar la tasa de cartera mes a mes de 2025-09 a 2026-09 antes de decidir si se añade tendencia.
- **Conciliar:** el paso 19 (todas las series, con crédito) da un sesgo del total del −1 % a 6 meses, frente al −3 % de los pools juzgados del paso 14.

### H · Uplift (pasos 15–16)
- **Uplift de la cartera:** 1,13 con toda la historia, 1,16 con la ventana y 1,10 con la ventana sin subidas (la que usa el forecast).
- **El cambio del 18-oct queda validado.** En el examen, la ventana sin subidas tiene un sesgo plano mes a mes (de −1 % a 0 %). Las otras dos derivan:
  - toda la historia, de +1 % a +4 %;
  - la ventana, de 0 % a +3 %.
- **Regla del contrato:** se realiza entre el 98 % (descuento 0–10 %) y el 66 % (80–90 %) de 1/(1−d).
  - Vía contrato: sesgo +13 %, WAPE 0,17. Vía estadística: −3 % y 0,09.
  - El veredicto es la vía estadística, y es correcto.
- **Subidas:**
  - Hay 43 en juego, casi todas con el ciclo terminado antes de 2026-09: solo se quitan de la ventana.
  - Siguen en el futuro tres pequeñas: Australia Standalone (+12 %), Canada Standalone y Turkey FL, con unos $0,25M que vencen.
  - Hay 3 escalones sin confirmar desde 2026-07 ($274k), que septiembre dirá.
- **Sospechosas:** Brasil NP (+32 %) y México NP (+39 %), con uplift previo de 0,88.
  - Parecen el fin de una campaña de descuento o un cambio de producto, no una tarifa.
  - Sus ciclos ya terminaron, así que solo afectan a la ventana ($7,1M renovados). Riesgo bajo.
- **Isolated:** media 1,096, desviación 0,285; el 46 % del valor renueva un 10 % o más por encima. El precio sube aunque no haya evento de retención.
- **Valoración:** es la pieza mejor validada.

### I · Marcas (paso 17)
- **Hoy las marcas de los meses futuros se quedan cortas:**
  - dormant: del 3 al 9 % hoy, frente al 9–11,5 % final;
  - softcancel: del 3 al 7 % hoy, frente al 7,5–9 % final.
- **Reparto:**
  - 2026: −$0,84M (−1,8 %);
  - 2027: −$5,52M (−4,5 %); el diagnóstico C da −$7,59M.
  - El detector no ve ninguna marca por encima de su proporción final.
- **Tasas por combinación:** neutral 71,4 %, dormant 44,5 %, softcancel 27,9 %, not_installed 50,1 %.
- **Riesgo alto.** Es el mayor ajuste individual y descansa en la proporción histórica de dormant, que es la inflada. Hay que repetir la ejecución tras el arreglo del SQL y comparar el ajuste de 2027.

### J · Adquisición simulada (paso 17)
- **Ya está corregida:**
  - Pure-New: nivel 0,97; simulado $6,0M frente a $6,2M un año antes.
  - Not-New: nivel 0,54; simulado $0,95M frente a $1,75M.
  - Renueva $5,18M en 2027 (el 4,5 % del año).
- **Pregunta de negocio:** ¿se han reducido a la mitad de verdad las ventas Not-New en jun–ago de 2026, o es el extracto?

### K · Horizonte extendido de 2027-09 a 2027-12
- Se borran $15,5M de pipeline parcial.
- Se crean $39,1M de pipeline proyectada (que renueva $30,4M, al 77,7 %) y $6,9M de simulada (que renueva $5,2M).
- El 31 % de lo renovado en 2027 ($35,5M de $116,1M) sale de pipeline modelada, no del extracto.

### L · Totales
- **2026:** $124,2M ($79,3M real + $44,9M esperado), sobre una pipeline de $162,3M.
- **2027:** $116,1M ($80,6M de pipeline conocida + $30,4M proyectada + $5,2M simulada), sobre una pipeline de $155,2M.
- **2027 frente a 2026: −6,5 %.**
  - La pipeline baja un −4,4 %.
  - El reparto de marcas pasa de −1,8 % a −4,5 %.
  - Frente a 2025 ($112,9M), 2027 sube un +2,9 %; 2026 sube un +10 %.

### M · Bandas
- **Banda del total en cuadratura:** ±$103k en 2026 y ±$256k en 2027 (0,1–0,2 %).
  - Supone errores independientes entre 25.000 series.
  - El examen dice que el error del total es del 0,5–1,4 % (paso 19) o del 1,3–3 % (paso 14): los errores de las series se mueven juntos.
- **Banda lineal:** 2027 entre $70,8M y $145,5M. No sirve.
- **Propuesta:** banda empírica del total, el error del total en el examen por horizonte (p. ej. p90 de |error|), aplicada al esperado de cada mes. Para 2027 hacen falta horizontes del examen de más de 6 meses.
- **Serie a serie:** el framework deja dentro del intervalo el 96–97 % (nominal 90 %, algo ancho); la serie sola, el 83–87 %.

### N · Examen serie a serie (paso 19)
- **WAPE:** framework 11,1 % frente a 10,1 % de la serie sola. Ponderado por volumen, la serie sola es mejor.
- **Error frente a ruido:** framework 1,16 frente a 1,84. Serie a serie, el framework es mejor.
- **Lectura:** el framework ayuda a las series pequeñas y empeora un poco las grandes. Es el mismo coste que la elección de técnica.
- **Propuesta:** FVA por tamaño.

### O · La serie de ejemplo y el soporte
- **El ejemplo es una serie que se ha extinguido.** "La serie con más dinero" del núcleo es DigitalRiver DACH KIS MD:
  - n_propio 4.444 (la mediana histórica);
  - en sus últimos 6 meses cerrados vencen entre 1 y 5 unidades.
- **Lectura:** es una migración de partner (las series NEXWAY nuevas tienen 22 meses).
- **Dos arreglos:**
  - elegir el ejemplo por dinero por predecir;
  - medir el soporte con meses recientes, no con la mediana de toda la historia.

### P · Rendimiento
- 45,5 minutos en total. La auditoría se lleva el 26 % (701 s), el núcleo el 16 % y la escritura de sff_fact_fine 216 s.

## Riesgos, por orden

1. **La banda del total que se enseña** (±0,2 %) no es la incertidumbre real (1–3 %).
2. **El reparto de marcas sobre un dormant histórico inflado:** −$5,5M en 2027.
3. **La subida de la tasa en 2026 y el retraso de las técnicas:** −3 % a 6 meses en el examen; 2027 puede salir corto.
4. **El mes en curso equivocado.**
5. **El 31 % de lo renovado en 2027 sale de pipeline modelada:** depende de la tasa de la proyectada y del nivel de adquisición.
6. **El descuento desconocido en el 59 % del USD.**
7. **Las migraciones de partner** cortan la identidad de las series; el soporte se mide con la mediana histórica.
8. **La elección de técnica por id** se ajusta al ruido (pequeño).
9. **El universo time_series vacío.**

## Acciones propuestas

1. Repetir con `current_month="01/10/2026"`.
2. Implementar la banda empírica del total por horizonte, con más horizontes en el examen.
3. Arreglar dormant en el SQL, repetir y comparar el ajuste de marcas de 2027.
4. Mirar la tasa de cartera mes a mes de 2025-09 a 2026-09; decidir sobre la tendencia.
5. FVA: el retador por defecto, y otra técnica solo con margen.
6. Elegir el ejemplo del núcleo por dinero por predecir; medir el soporte con meses recientes.

## Lo que dice el sintético parecido al real (añadido el mismo día)

`synthetic_real.py` reproduce la forma de esta ejecución y conoce la verdad futura. Con el mes en curso en 2026-09:

| Indicador | Real | Sintético |
|---|---|---|
| Tasa en unidades / en USD | 0,640 / 0,729 | 0,640 / 0,752 |
| Neutral / dormant / softcancel / not_installed | 0,71 / 0,45 / 0,28 / 0,50 | 0,70 / 0,45 / 0,29 / 0,50 |
| Descuento desconocido (USD) | 59 % | 60 % |
| Dinero con suelo, antes → después de la escalera | 82 % → 98 % | 82 % → 96 % |
| φ mediana | 1,94 | 2,28 |
| Mejor \|err_norm\| a corto | 1,36 | 1,31 |
| Error del total de los pools, h1 / h6 | −1,3 % / −3,0 % | −0,7 % / −1,8 % |
| Error del total del framework (paso 19), h1 / h6 | 0,5 % / 1,4 % | 0,6 % / 1,9 % |
| Uplift: historia / ventana / sin subidas | 1,13 / 1,16 / 1,10 | 1,14 / 1,18 / 1,13 |
| Sesgo de la vía contrato | +13 % | +12 % |
| Nivel de adquisición Pure-New / Not-New | 0,97 / 0,54 | 0,91 / 0,51 |
| Reparto de marcas, año siguiente | −4,5 % | −5,2 % |

**El forecast frente a la verdad (2026-09..2027-08):**
- **Total:** −0,1 %.
- **Cada mes:** de −4,8 % a +3,2 %, cuando la banda en cuadratura dice ±0,65 %. La banda es unas cinco veces más estrecha de lo que falla.
- **Sin el reparto de marcas:** +2,9 % en el total y +8,4 % a doce meses.
- **Sin partir por marcas:** +0,5 % en el total, con el mismo error mes a mes. En el sintético las marcas no hacen que prediga mejor; el reparto solo corrige lo que el partir por marcas estropea en los meses futuros.
