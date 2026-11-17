# Validación del análisis spellF1 — actual vs recalibrado

**Fecha:** 28 oct 2026 · **Alcance:** los 5 ficheros de `handoff-plugin-data/datos/`.
**Método:** sólo lectura; todos los números se recalcularon de forma independiente desde los CSV/JSON
(script: `verificar.py`, en el scratchpad de la sesión). No se tocó nada de `data/`.

## Veredicto: **compartir con salvedades, y corregir dos afirmaciones antes de enviarlo**

> **29-10, más tarde:** la afirmación de que el descenso "no transfiere nada" **es falsa**;
> corriendo el procedimiento sobre los 35 splits gana +1,7 pp fuera de su ajuste. Ver
> [VALIDACION-CRUZADA.md](VALIDACION-CRUZADA.md), que sustituye a la sección de corrección
> de abajo en todo lo que se refiere al sobreajuste.
>
> **29-10:** el hallazgo 1 se midió de nuevo con las tres semillas de degradación y **se
> retira** su recomendación de cambiar de config. Leer *Corrección del 29-10* antes del
> hallazgo 1. El resto del informe se mantiene.

Las cuentas del paquete son correctas: los agregados del JSON reproducen exactamente la media de los
CSV por partido, las filas de la matriz de confusión suman 100 %, los `n` cuadran entre tablas
(1.002.644 fotogramas, 1.945 posesiones reales) y los centros recalibrados son en efecto el punto
medio de las medianas por clase. No hay errores de cálculo.

---

## Corrección del 29-10-2026 — el hallazgo 1 no sobrevive a su propia comprobación

Al intentar llevar `solo_centros` a `config.py` se midió de nuevo el arnés IDSSE, esta vez con las
**tres semillas de degradación** que hay en `out/cache` en lugar de una sola. Dos cosas cambian:

**a) `solo_centros` no es "config.py sin el ajuste extra".** El script que generó `recalibracion.json`
lo define como `merged(**centres)` sobre el `DEFAULT` **de ese momento**, es decir el `config.py`
anterior al 28-10: ventanas de suavizado ±50, `switch_turnover` 28 y los centros de la ventana
pública, con sólo los tres centros nuevos encima. Ese fichero **no se conserva** (el repo no tiene
ningún commit), así que `solo_centros` no se puede reproducir exactamente. Reconstruido desde la
procedencia documentada en `config.py` queda a −0,3 pp en ajuste y −0,02 pp en validación de los
valores publicados — cerca, pero no igual.

**b) la ventaja de −0,22 pp está dentro del ruido entre semillas.** Delta de spellF1
(`recalibrado` − `solo_centros`), midiendo las dos configs con el mismo código y la misma semilla:

| semilla | ajuste (4) | validación (3) |
|---|---|---|
| 0 | +1,91 pp | +0,60 pp |
| 1 | +0,59 pp | −0,44 pp |
| 2 | +2,13 pp | −0,61 pp |
| **media** | **+1,54 pp** | **−0,15 pp** |

El signo en validación cambia con la semilla y el recorrido es de 1,2 pp: siete veces el −0,22 pp
sobre el que se apoyaba la recomendación. El sesgo de la reconstrucción es un desplazamiento
constante y no afecta a ese recorrido.

**Lo que sí queda demostrado, y más firme que antes:** en ajuste la recalibración gana en las tres
semillas (+0,6 a +2,1 pp) y en validación no gana en ninguna (media −0,15 pp). El descenso por
coordenadas compra puntos donde se ajustó y **no transfiere nada**. El hallazgo 2 se mantiene; lo que
cae es el corolario de que `solo_centros` sea mejor. Las dos son indistinguibles fuera del ajuste.

**Qué hacer con esto:** no cambiar la config de entrega. No hay evidencia para preferir ninguna de
las dos, y `config.py` tiene la procedencia de cada valor escrita al lado, que es lo que exige
RULES.md. Lo que hay que corregir es el **relato**: la recalibración vale +1,5 pp donde se ajustó y
0 pp fuera.

### De dónde sale la ganancia, parámetro a parámetro

Revirtiendo **un solo valor** de `config.py` al que tenía antes del 28-10 (semilla 0, el resto igual):

| se revierte | Δ ajuste | Δ validación |
|---|---|---|
| `control_center` 0,008 → 0,096 | −0,06 pp | **−1,55 pp** |
| `switch_turnover` 40 → 28 | −0,37 pp | **−1,27 pp** |
| `shape_smooth` 200 → 50 | −0,69 pp | −0,63 pp |
| `ball_speed_center` 3,14 → 4,10 | −0,52 pp | +0,06 pp |
| `ball_speed_smooth` 6 → 50 | −1,10 pp | +0,33 pp |
| `activity_smooth` 12 → 50 | −0,27 pp | **+1,14 pp** |
| `shape_center` 0,976 → 0,975 | 0,00 pp | 0,00 pp |

Se separa limpio en dos grupos. Lo que **transfiere** son los dos valores con justificación
mecánica: el centro de control (simétrico entre equipos por construcción) y el coste de giro de
pelota (que `config.py` ya documenta como una meseta por encima de 40, no un pico). Lo que **no
transfiere** es `activity_smooth` = 12: cuesta 1,1 pp en validación y sólo compra 0,27 pp en ajuste
— es ruido elegido por rejilla. `ball_speed_smooth` = 6 va en la misma dirección, más suave.
`shape_center` es indistinguible: el redondeo a tres decimales no mueve ni un fotograma.

Aviso de método: esa última tabla está medida **sobre el conjunto de validación**. Sirve para
diagnosticar de dónde viene el sobreajuste; usarla para elegir `activity_smooth` = 50 sería cometer
otra vez el mismo error, con el último conjunto limpio que queda. Si se quiere decidir eso bien,
hace falta volver a partir los 7 partidos y re-derivar, no leer esta columna.

Medido con `handoff-plugin-data/semillas.py` y `reeval.py`; números en
`datos/robustez_semillas.json`. Se re-ejecutó sólo el arnés sobre `out/cache` (partidos externos
degradados): no se tocó el ATD ni la ventana pública.

---

Lo que no se sostiene es la lectura: **"0,3767 → 0,3927 y no se sobreajustó" es demasiado optimista, y
la configuración que el paquete recomienda no es la que mejor generaliza.**

---

## Hallazgos

### 1. [ALTA] `solo_centros` gana a `recalibrado` en los partidos de validación

Está en `recalibracion.json`, en el bloque `scores`, y el paquete no lo menciona. spellF1:

| config | ajuste (4) | validación (3) | los 7 |
|---|---|---|---|
| `original_pre_sesion` | 0,3110 | 0,3272 | 0,3179 |
| `actual_config_py` | 0,3714 | 0,3837 | 0,3767 |
| `solo_centros` | 0,3747 | **0,3979** | 0,3847 |
| `recalibrado` | **0,3905** | 0,3957 | 0,3927 |

Leído por columnas: re-centrar y nada más gana +1,42 pp en validación; añadir el resto de la
recalibración encima gana +1,58 pp **en ajuste** y pierde **−0,22 pp en validación**. Es la firma
clásica de sobreajuste: los parámetros extra compran puntos donde se ajustaron y no los transfieren.

La brecha ajuste−validación lo confirma: −0,0232 con `solo_centros` → −0,0052 con `recalibrado`. El
conjunto de ajuste "alcanza" al de validación, que es lo que pasa cuando se ajusta sobre él.

**Consecuencia práctica** — *corregida el 29-10, ver la sección de corrección*: la ventaja de
−0,22 pp es una sola semilla de degradación y cambia de signo con las otras dos. **No hay evidencia
para cambiar de config.** Lo que sí hay que decir es que la ventaja de `recalibrado` sobre
`solo_centros` no está demostrada fuera del conjunto de ajuste: ahí son indistinguibles.

### 2. [ALTA] "No se sobreajustó" no es lo que dicen los datos

La mejora es +1,91 pp en ajuste y +1,19 pp en validación: transfiere el 62 %, no el 100 %. Y uno de
los tres partidos de validación **empeora** (J03WPY, −0,0035).

Deltas por partido (recalibrado − actual):

| partido | grupo | actual | recalibrado | Δ |
|---|---|---|---|---|
| J03WMX | ajuste | 0,3556 | 0,3606 | +0,0050 |
| J03WN1 | ajuste | 0,3227 | 0,3508 | +0,0281 |
| J03WOH | ajuste | 0,4515 | 0,4570 | +0,0055 |
| J03WOY | ajuste | 0,3556 | 0,3936 | +0,0380 |
| J03WPY | validación | 0,3824 | 0,3789 | **−0,0035** |
| J03WQQ | validación | 0,4210 | 0,4357 | +0,0147 |
| J03WR9 | validación | 0,3478 | 0,3724 | +0,0246 |

Los dos saltos grandes (+0,028 y +0,038) caen los dos en ajuste.

Significación con n=7: t emparejado p=0,029, Wilcoxon p=0,031, pero test de signos p=0,125 (6/7).
IC 95 % del delta medio: **[+0,002, +0,030]** — el intervalo roza el cero. Con los 3 de validación
solos (2/3 positivos, p=1,0) no se puede afirmar nada estadísticamente.

**Redacción honesta:** "+1,2 pp en 3 partidos no usados para ajustar, con 1 de los 3 en contra; la
muestra no permite descartar que parte del salto sea ruido."

### 3. [ALTA] El conjunto de validación es más fácil que el de ajuste, así que 0,3957 no es una predicción

En **las cuatro** configs, validación puntúa por encima de ajuste (entre +0,5 y +2,3 pp). Es una
propiedad de esos 3 partidos, no del modelo. Citar 0,3957 como "lo que esperamos en la competición"
hereda ese sesgo favorable. A eso se suma que los 7 partidos son Bundesliga, con otro proveedor de
tracking y otra cámara que el partido de la competición: **ninguno de estos números es una
predicción para la entrega**, sólo evidencia de que el margen sobre el mínimo 0,2433 es amplio
(el peor partido, 0,3227, lo supera en +33 %).

### 4. [MEDIA] Sí hay sesgo de agregación en las categorías con pocos fotogramas

`error_por_racha_sin_balon`, que es la tabla donde más "mejora" se ve:

| categoría | n | peso | error actual | recalibrado | Δ pp | fotogramas salvados |
|---|---|---|---|---|---|---|
| 1-25 fr | 512 | 0,05 % | 11,52 % | 4,10 % | −7,42 | **38** |
| 26-125 fr | 854 | 0,09 % | 20,73 % | 19,56 % | −1,17 | 10 |
| 126-500 fr | 131.727 | 13,1 % | 26,58 % | 26,23 % | −0,35 | 461 |
| >500 fr | 276.897 | 27,6 % | 27,69 % | 27,39 % | −0,30 | 831 |
| balón visible | 592.654 | 59,1 % | 15,27 % | 14,27 % | −1,00 | 5.927 |

El bin de 1-25 fr es el 0,05 % de los datos: su −7,42 pp son **38 fotogramas**, ruido puro. Dibujado
como cinco barras del mismo tamaño, el gráfico sugiere una mejora media de −2,05 pp; ponderando por
fotogramas la mejora real es **−0,72 pp**. Casi todo el efecto (82 %) viene del bin "balón visible".

→ En el dashboard, los dos bins pequeños van etiquetados con su `n` y marcados como no fiables.

Mismo aviso en sentido contrario: en `error_por_distancia_a_transicion`, el bin más grande (>250 fr,
291.549 fotogramas, 29 % del total) **empeora** 7,22 % → 7,48 %. No está en el resumen del paquete.

### 5. [MEDIA] La ganancia es casi toda precisión, y el recall tiene techo

Descomponiendo spellF1 desde la tabla de recall por duración y las posesiones predichas:

| config | aciertos | recall | precisión | F1 reconstruido |
|---|---|---|---|---|
| actual | ~644 / 1.945 | 0,331 | 0,414 | 0,368 |
| recalibrado | ~660 / 1.945 | 0,339 | 0,458 | 0,390 |

(reconstruye 0,368 y 0,390 frente a los 0,3767 y 0,3927 oficiales — la diferencia es que el oficial
promedia F1 por partido en vez de agrupar; coherente.)

El recall sube +0,8 pp y la precisión +4,4 pp: la recalibración no acierta más posesiones, emite
menos (1.555 → 1.441 predichas, frente a 1.945 reales) y más largas (duración media 14,1 s → 15,2 s
contra 11,8 s reales). Mejora la métrica alejándose de la estructura real del partido.

**El techo que eso impone:** con recall 0,339, incluso con precisión perfecta spellF1 ≈ **0,507**. Las
posesiones de menos de 5 s son 764 de 1.945 (**39 %**) y se recuperan al 9 %. Mientras eso no cambie,
seguir afinando umbrales rinde cada vez menos. Y el mecanismo es visible en la propia config:
`min_duration_frames: 12` (0,48 s) y `switch_turnover: 40` (1,6 s) hacen estructuralmente imposible
emitir las posesiones cortas. Ahí está la palanca grande, no en los centros.

### 6. [MEDIA] La recalibración no reduce la confusión H/A: la reequilibra hacia H

| | actual | recalibrado |
|---|---|---|
| real H → pred H | 66,3 % | 72,8 % |
| real A → pred A | 73,2 % | **67,9 %** |
| H↔A sobre el total | 11,76 % | 10,90 % |

Gana 6,5 pp en H y pierde 5,4 pp en A; neto −8.630 fotogramas (−0,9 pp). La causa es directa: los
centros `control` y `shape` son el punto medio de las medianas **H y A medidas en los 4 partidos de
ajuste** (verificado: 0,008298 y 0,975781 son exactamente esos puntos medios). Esa asimetría H/A
queda horneada en la config. Si en el partido de la competición el reparto H/A es distinto, el sesgo
va en contra. El paquete no incluye la dispersión de esas medianas, así que no se puede saber si las
clases están bien separadas o solapan — con 20 % de confusión H/A, solapan mucho.

### 7. [BAJA] Observaciones informativas, no accionables

- **Media de medias:** los agregados promedian spellF1 por partido sin ponderar. Ponderando por
  posesiones reales la diferencia es ≤ 0,25 pp en todos los casos. Comprobado y sin impacto.
- **`n_frames_o_posesiones` mezcla dos significados:** en tres tablas es el tamaño de la categoría
  (idéntico entre configs, lo que hace justa la comparación); en la matriz de confusión es el conteo
  de la celda, que sí varía. Las filas suman igual (292.078 / 261.027 / 449.539), así que es
  coherente, pero conviene aclararlo en la leyenda.
- **La ventana de 5 min no es representativa:** 19,6 % de D frente al 44,8 % de los partidos
  Bundesliga. Son 5 min de juego seguido tras un saque. El acuerdo por fotograma ahí es 81,0 % y el
  modelo infra-predice D (13,0 % vs 19,6 % real). No extrapolar de esa ventana.
- **Las partes del partido completo duran 53 min cada una** (3.178 s y 3.175 s; 105,9 min en total,
  frente a 95,5 min de media por partido en el set externo). Probablemente incluye relleno antes del
  saque o del descanso, que se etiquetaría como D e infla el 44,8 % de D. Merece una comprobación de
  si la ventana de entrega cubre esos minutos.

---

## Comprobaciones hechas, y su resultado

| comprobación | resultado |
|---|---|
| Agregados del JSON = media de los CSV | ✅ exacto en los 6 casos (tune/hold/all × 2 configs) |
| Filas de la matriz de confusión suman 100 % | ✅ (99,99–100,00 por redondeo) |
| Totales de fotogramas coherentes entre tablas | ✅ 1.002.644 en las tres tablas de fotogramas |
| Posesiones de la tabla de recall = suma de `posesiones_reales` | ✅ 1.945 |
| `n` de categoría idéntico entre configs (comparación justa) | ✅ en las 3 tablas donde debe serlo |
| Centros = punto medio de las medianas declaradas | ✅ control, shape y ball_speed |
| spellF1 reconstruido desde precisión/recall | ✅ 0,368 / 0,390 vs 0,377 / 0,393 (coherente) |
| Ventana pública: 7.500 filas a 25 fps = 300 s | ✅ |
| Todos los partidos superan el mínimo 0,2433 | ✅ el peor es 0,3227 |
| Media simple vs ponderada | ✅ diferencia ≤ 0,25 pp, inmaterial |

---

## Qué cambiar antes de compartirlo

1. **Sustituir "no se sobreajustó"** por el desglose del hallazgo 2 (+1,91 pp ajuste / +1,19 pp
   validación, 1 de 3 en contra, IC [+0,002, +0,030]).
2. **Añadir la fila `solo_centros`** a la tabla de resultados y decir cuál se lleva a la entrega.
   Es la decisión de producto que sale de esta validación.
3. **Dejar de citar 0,3957 como expectativa** para el partido de la competición: validación es más
   fácil que ajuste en las cuatro configs, y además es otra liga y otro tracking.
4. **Poner el `n` en todos los gráficos de diagnóstico** y marcar los bins de 512 y 854 fotogramas.
5. **Mencionar el techo de 0,507** y el 39 % de posesiones cortas como el límite de seguir ajustando.

## Lo que no se pudo evaluar con este paquete

- **No hay ground truth del partido de la competición** (es privado), así que no hay forma de medir
  ahí: todo lo anterior es transferencia desde Bundesliga.
- **`ventana_publica_5min.csv` y `partido_completo_1hz.csv` sólo traen `modelo_actual`.** No hay
  columna recalibrada, así que las vistas temporales no se pueden comparar entre configs.
- **No hay dispersión** (IQR/desviación) de las medianas por clase, sólo el punto medio: no se puede
  juzgar la separabilidad H/A.
- **No hay intervalos por partido** (una sola evaluación por partido y config), así que no se puede
  separar varianza del método de varianza entre partidos.
- No se re-ejecutó el modelo ni se verificó el código del scorer: se auditaron sus salidas.
