# Validación cruzada del procedimiento de recalibración — 35 splits

**Fecha:** 29 oct 2026 · **Qué se midió:** el *procedimiento* de `training/recalibrate.py`, no una
configuración concreta. **Cómo:** el mismo descenso por coordenadas, desde el mismo punto de
partida, sobre **los 35 splits** posibles de 4 partidos de ajuste / 3 de validación con los 7
partidos del IDSSE. Cada descenso se mide en los 3 partidos que ese descenso no vio, y además en
las semillas de degradación 1 y 2, que ninguno vio.

> **Añadido el 31-10:** la rejilla se ampliaba por consejo de este informe. Ya está hecho, y el
> resultado es que **no sirve**: gana +0,62 pp en el ajuste y −0,13 pp fuera de él. Ver la primera
> sección; matiza el hallazgo de "la rejilla se queda corta" sin invalidar el resto.

Scripts: `splits.py` (los 35 descensos) y la comprobación de consenso. Datos:
`datos/splits_cruzados.json`, `datos/consenso_splits.json`. Sólo se leyó `out/cache` (partidos
externos degradados): no se tocó el ATD ni la ventana pública.

---

## 31-10: ampliar la rejilla no sirve

**Veredicto:** la rejilla sí se quedaba corta, y arreglarlo **no mejora nada fuera del ajuste**.
Toda la capacidad de búsqueda extra se va en sobreajuste. No hay que ampliar la rejilla en
`training/recalibrate.py`.

Se ampliaron **siete** parámetros, no los cinco de la sección siguiente: al revisar la tabla de
estabilidad aparecieron dos más que también se clavaban en un borde — `min_duration_frames` (50 es
su tope y es la moda, 17/35) y `ball_speed_smooth`, que se clava en el **mínimo** de su lista
(6 en 19/35). Cada lista se prolongó con su mismo paso geométrico (~×1,4) hasta dejar por dentro el
valor que se elegía: 64 → 82 valores. Nada más cambió — mismo punto de partida, los mismos 14
parámetros barridos, las mismas 2 pasadas y la misma función `descenso()`, que ahora recibe la
rejilla como argumento. Antes de lanzar se comprobó que con la rejilla vieja reproduce el split
original al sexto decimal y elige los mismos 14 valores, así que la parametrización no movió nada.

Los dos brazos corren sobre los mismos 35 splits desde el mismo punto de partida, así que la
comparación es **pareada** split a split:

| eje | rejilla original | ampliada | diferencia pareada | IC 95 % | gana en |
|---|---|---|---|---|---|
| en su propio ajuste (4) | +3,27 pp | +3,89 pp | **+0,62 pp** | [+0,34, +0,90] | 26/35 |
| **en su propia validación (3)** | +1,71 pp | +1,58 pp | **−0,13 pp** | [−0,52, +0,26] | 14/35 |
| semilla 1, los 7 | +1,93 pp | +1,91 pp | −0,02 pp | [−0,34, +0,30] | 18/35 |
| semilla 2, los 7 | +1,74 pp | +1,84 pp | +0,10 pp | [−0,18, +0,38] | 20/35 |
| transferencia | 52 % | **41 %** | | | |

Es un resultado limpio y es negativo. En el ajuste la mejora es real y significativa: +0,62 pp con
el intervalo entero por encima de cero. En los tres partidos que el descenso no vio, el intervalo
cruza el cero y la mejora se da en **14 de 35** splits, o sea menos de la mitad: indistinguible de
una moneda. Las dos semillas de degradación dicen lo mismo. La transferencia baja del 52 % al 41 %,
que es exactamente la firma de ampliar el espacio de búsqueda sin más datos: se ajusta mejor a los
cuatro partidos de turno y no mejor al problema.

El diagnóstico de la sección siguiente era correcto —**ningún** parámetro se clava ya en un borde,
luego el óptimo sobre el ajuste estaba de verdad fuera de la rejilla vieja. Lo que no se sostiene
es la implicación: ese óptimo no generaliza. Y la región recién alcanzable es visiblemente ruido.
`switch_from_dead` se reparte ahora entre **siete** valores distintos (14:12, 8:6, 22:5, 31:4,
60:4, 4:2, 43:2) y `switch_to_dead` entre 63:12 y 123:8, un rango de casi el doble. Donde antes
había un borde que parecía señal, hay una meseta plana.

En el split original, el de `config.py`, la rejilla ampliada llega más arriba en el ajuste
(+3,14 pp frente a +2,72) y **más abajo** en la validación (+2,07 pp frente a +2,82). El mismo
patrón, en el único split que se había mirado antes.

### Qué se lleva de aquí sobre `config.py`

Con la rejilla ampliada, los splits que eligen el valor que hay hoy en `config.py` se desploman:
`shape_smooth` 2/35, `activity_smooth` 3/35, `min_duration_frames` 5/35, `switch_turnover` 6/35,
`switch_to_dead` 6/35, `ball_speed_weight` 6/35. **Eso no es un argumento para cambiarlos.** Si el
brazo ampliado no gana fuera de su ajuste, lo que elige no es mejor; es sólo distinto. Los únicos
valores que aguantan en los dos brazos son `control_weight` (34/35), `shape_weight` (32/35),
`control_smooth` (31/35) y `activity_weight` (27/35).

Lo de `activity_smooth` = 12 sigue en pie como lo que era: 1/35 con la rejilla vieja, 3/35 con la
ampliada, y la moda nunca es 12. Esa conclusión no dependía de la rejilla.

### Lo que esta comparación no puede decir

*Dónde* ampliar se decidió mirando los 35 splits del día anterior. Esa decisión usó sólo los
valores que el descenso eligió en los partidos de **ajuste** de cada split, nunca puntuaciones de
validación — pero como cada partido entra en el ajuste de algún split, hay una filtración débil.
Dado que el resultado es negativo, la filtración juega a favor de la rejilla ampliada y aun así
pierde, luego el veredicto no está en riesgo por ese lado.

El coste real fueron **5 h 48 min** de CPU, no la hora que estimé: los parámetros nuevos incluyen
ventanas de suavizado grandes, que multiplican las entradas de la caché de `Match._smoothed` y
encarecen cada evaluación. Si alguien repite esto con otra rejilla, contar con ese orden de
magnitud.

Scripts: `splits.py ampliada` (el brazo nuevo) y `compara_rejillas.py` (la tabla pareada de
arriba). Datos: `datos/splits_cruzados_ampliada.json`.

---

## 31-10: `activity_smooth` vuelve a 50 (aplicado)

Único cambio en `config.py` que ha salido de todo esto. Medido con las tres semillas de
degradación, sin tocar nada más:

| | ajuste (4) | validación (3) | los 7 |
|---|---|---|---|
| semilla 0 | −0,27 pp | **+1,14 pp** | +0,33 pp |
| semilla 1 | +0,59 pp | **+1,40 pp** | +0,94 pp |
| semilla 2 | +0,05 pp | **+0,81 pp** | +0,37 pp |
| media | +0,12 pp | **+1,11 pp** | +0,55 pp |

Las tres semillas del mismo signo en validación, y mejora en 5 de los 7 partidos. En el
ajuste el cambio es neutro (+0,12 pp), no la pérdida que sugería el barrido documentado en
`config.py` — ese barrido daba al 12 una ventaja de 0,0014 sobre el 50, con un recorrido
total de 0,14 pp entre 6 y 50. No distinguía nada.

Tres medidas independientes eligen 50 y ninguna elige 12: el reparto de los 35 splits
(12 sale 1/35, y 3/35 con la rejilla ampliada; la moda es 50), la ablación de arriba, y la
planicie del propio barrido.

**Procedencia, que importa aquí más de lo normal.** 50 coincide con el valor que había antes
del 28-10, y aquel salía de la ventana pública, que `RULES.md` prohíbe como conjunto de
ajuste. La coincidencia es solo eso: el 50 de ahora se justifica con los partidos del IDSSE
y con nada más. Si la medición hubiera dado 25, iría 25. Queda escrito así en `config.py`
para que nadie lea el cambio como una vuelta al valor viejo.

Script: `activity_smooth.py`. Datos: `datos/activity_smooth.json`.

### Comprobado contra el partido real

El cambio se aplicó a `config.py` y se regeneró la entrega con los assets del ATD. Los 55
tests pasan y el contrato del CSV es válido. Dos lecturas más, ninguna de las cuales decidió
nada — la decisión ya estaba tomada con el IDSSE:

- **Tasa de balón parado:** 45,8 % → **44,9 %** de los 158.857 frames puntuados, contra la
  media de 44,9 % de los siete partidos del IDSSE. `activity_center_ms` = 1,26 se fijó por
  percentil sobre la señal *suavizada*, así que está acoplado a `activity_smooth`: cambiar
  la ventana mueve la tasa. La movió hacia el objetivo. Si se vuelve a tocar el suavizado,
  hay que volver a medir esta tasa.
- **Ventana pública de 5 minutos:** spellF1 0,4875 → **0,4903**; macroF1 0,8108 → 0,8009.
  Es una lectura, no un criterio: `RULES.md` la prohíbe como conjunto de ajuste. Se mira
  para comprobar que el cambio no rompe nada ahí, y no lo rompe.

### Un número mal citado en `config.py`

El docstring decía *"ajuste 0.3905, validacion 0.3957"*. El 0,3905 era correcto; el 0,3957
no lo producía el fichero. Salía del volcado de `recalibrate.py`, que guarda los tres
centros de clase sin redondear; con los centros redondeados como están en `config.py` la
validación da **0,3968**. Son 0,001 de suerte en el redondeo, no un error de medida, pero el
número citado tiene que ser el que produce el fichero. Corregido, y ahora dice lo que da el
fichero tal cual está: ajuste 0,3878, validación 0,4081, los siete 0,3965.

## Corrección de la corrección

El 29-10 por la mañana escribí, en `VALIDACION.md`, que el descenso por coordenadas **"no
transfiere nada"**. Es falso, y el error es mío de método: lo deduje comparando dos
configuraciones fijas (`recalibrado` y `solo_centros`) en tres semillas. Eso mide la diferencia
entre dos puntos de llegada, que es ruidosa; no mide lo que gana el procedimiento. Corriéndolo 35
veces:

| | media | rango | mejora en |
|---|---|---|---|
| en su propio ajuste (4 partidos) | **+3,27 pp** | +1,49 a +4,58 | 35/35 |
| **en su propia validación (3 partidos)** | **+1,71 pp** | −0,58 a +3,51 | **32/35** |
| en la semilla 1, los 7 partidos | +1,93 pp | +0,78 a +3,15 | 35/35 |
| en la semilla 2, los 7 partidos | +1,74 pp | +0,20 a +2,66 | 35/35 |

IC 95 % de la media en validación: **[+1,37, +2,05] pp**. La recalibración gana algo real fuera de
donde se ajustó. La transferencia es del **52 %**: de cada dos puntos que compra en ajuste, uno se
queda en el camino. Eso es sobreajuste parcial, que es lo normal y lo que había que decir desde el
principio, no "no transfiere nada".

Lo que sigue siendo cierto del informe anterior: el 0,3927 del documento técnico mezcla ajuste y
validación y no es una predicción, y el split original salió afortunado — su delta en validación
(+2,82 pp) está en el **percentil 83** de los 35.

---

## El hallazgo que importa: la rejilla se queda corta

> **Medido el 31-10:** cierto, se queda corta — pero ampliarla no mejora la validación. El título
> de esta sección promete más de lo que acabó valiendo. Ver "ampliar la rejilla no sirve" arriba.

En cada descenso se registró qué valor eligió cada hiperparámetro. Cinco de ellos eligen **el
máximo de su propia rejilla** en la mayoría de los splits, lo que significa que la búsqueda se
está topando con el borde y el óptimo está más allá:

| parámetro | elige el tope | en | `config.py` tiene |
|---|---|---|---|
| `switch_turnover` | 40 | 30/35 | 40 ✅ |
| `ball_speed_weight` | 1,2 | 28/35 | **0,6** |
| `switch_to_dead` | 45 | 28/35 | **34** |
| `switch_from_dead` | 22 | 25/35 | **14** |
| `shape_smooth` | 200 | 22/35 | 200 ✅ |

`config.py` documenta `switch_to_dead` y `switch_from_dead` como *"Confirmado"* porque en el split
original el barrido no los movió. En 28 y 25 de los 35 splits sí los mueve, y al tope. Los tres
valores en negrita son candidatos a estar mal por construcción del barrido, no por mala medida.

## Estabilidad por parámetro

Cuántos de los 35 splits eligen el valor que hay hoy en `config.py`:

| parámetro | coincide | reparto de lo elegido |
|---|---|---|
| `switch_turnover` 40 | **30/35** | 40:30 · 28:4 · 20:1 |
| `control_smooth` 50 | **28/35** | 50:28 · 100:7 |
| `control_weight` 6,0 | **27/35** | 6:27 · 9:4 · 13:3 · 4:1 |
| `shape_smooth` 200 | 22/35 | 200:22 · 100:6 · 25:5 · 12:2 |
| `shape_weight` 1,6 | 22/35 | 1,6:22 · 1,0:12 · 0,5:1 |
| `activity_weight` 1,0 | 22/35 | 1,0:22 · 0,5:13 |
| `ball_speed_smooth` 6 | 19/35 | 6:19 · 12:11 · 25:3 · 50:2 |
| `min_duration_frames` 12 | 12/35 | 50:17 · 12:12 · 0:4 · 25:2 |
| `activity_center_ms` 1,26 | 12/35 | 1,18:14 · 1,26:12 · 1,10:7 · 1,35:2 |
| `ball_unseen_logit` 0,5 | 11/35 | 0,0:12 · 0,25:12 · 0,5:11 |
| `switch_from_dead` 14 | 8/35 | 22:25 · 14:8 · 8:2 |
| `ball_speed_weight` 0,6 | 7/35 | 1,2:28 · 0,6:7 |
| `switch_to_dead` 34 | 6/35 | 45:28 · 34:6 · 26:1 |
| **`activity_smooth` 12** | **1/35** | 50:14 · 6:8 · 100:8 · 25:4 · **12:1** |

Tres grupos:

**Confirmados de verdad.** `switch_turnover` 40, `control_smooth` 50 y `control_weight` 6,0 los
elige el descenso en ~80 % de los splits. Son los tres que la ablación del informe anterior ya
señalaba como los que transfieren, y los dos primeros tienen además justificación mecánica escrita
en `config.py`. Nada que tocar.

**`activity_smooth` = 12 no tiene ningún respaldo: 1 de 35.** Lo modal es 50, el valor que tenía
antes del 28-10. Y es coherente con todo lo demás: el propio barrido documentado en `config.py`
(6:.3754 12:.3761 25:.3755 50:.3747) tiene un recorrido de 0,14 pp, o sea nada, y la ablación
medía +1,14 pp en validación al devolverlo a 50. Tres señales independientes apuntando al mismo
sitio: **ese 12 es ruido de rejilla**.

**Indiferentes.** `ball_unseen_logit` se reparte 12/12/11 entre 0, 0,25 y 0,5: el parámetro no
hace nada, exactamente lo que el propio `config.py` razona al explicar que sumarlo cuenta la
evidencia dos veces. `activity_center_ms` se reparte entre 1,10/1,18/1,26, pero aquí hay que
quedarse con 1,26 **de todas formas**, porque no sale de un barrido sino de igualar la tasa de
balón parado a la de partidos completos, que es un argumento que no depende de esta medición.

## ¿Y si se toman los valores modales?

La "config de consenso" (el valor más votado de cada parámetro entre los 35 splits) puntúa, en las
dos semillas de degradación que el descenso no vio, sobre los 7 partidos:

| | partida | `config.py` | consenso |
|---|---|---|---|
| semilla 1 | 0,3778 | 0,3793 | **0,3935** (+1,42 pp) |
| semilla 2 | 0,3996 | 0,4092 | **0,4171** (+0,79 pp) |
| semilla 0 *(vista)* | 0,3797 | 0,3932 | 0,4018 |

Gana en las dos semillas no vistas. **Pero no es un resultado limpio y no debe entregarse así:**
los valores modales se eligieron mirando los 7 partidos, así que las semillas 1 y 2 controlan el
ruido de degradación pero **no** la elección de partidos — son los mismos 7. Como estimación de lo
que rendiría en el partido de la competición no vale.

## Qué hacer

1. ~~**Ampliar la rejilla**~~ **HECHO el 31-10, y la respuesta es no.** Se ampliaron siete
   parámetros (64 → 82 valores): +0,62 pp en el ajuste, −0,13 pp en la validación, 14/35. No
   ampliar la rejilla en `recalibrate.py`, y no cambiar `switch_to_dead`, `switch_from_dead` ni
   `ball_speed_weight` a lo que elige el brazo ampliado. Detalle en la primera sección. Lo que
   queda abierto no es la rejilla: es que con 7 partidos de ajuste no hay señal para distinguir
   esos valores, y eso no lo arregla buscar más.
2. **`activity_smooth`:** 1/35 es suficiente para dejar de defender el 12. Pero el valor de
   repuesto hay que justificarlo con la validación cruzada completa, no con esta columna.
3. **En el documento técnico**, sustituir "no se sobreajustó" por el número que sí se puede
   defender: *la recalibración gana +1,7 pp (IC 95 % [+1,4, +2,1]) en partidos que no se usaron
   para ajustarla, sobre 35 particiones; en los partidos de ajuste gana +3,3 pp, así que
   aproximadamente la mitad de la ganancia es sobreajuste.*

## Cobertura no evaluada

Los 7 partidos son todos Bundesliga con el mismo tracking, degradados para parecerse al ATD; nada
de esto mide el salto a la liga, la cámara y el tracker del partido de la competición. Las semillas
1 y 2 varían el ruido de degradación, no los partidos. No se ha tocado el ATD ni la ventana
pública, y no se ha re-derivado ningún centro de clase: los tres siguen siendo los medianos del
IDSSE sobre los 4 partidos de ajuste del split original.
