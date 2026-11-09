# Estado del proyecto — 8 de noviembre de 2026

Documento para el equipo. Este repositorio tiene dos partes:

- **La entrega** (`src/`, `tests/`, `vendor/`, `scripts/`): el código que genera el CSV que se
  puntúa. Hoy es privado; se hará público el día de la entrega. Léase `README.md`.
- **El trabajo** (`trabajo/`): análisis, documentos, scripts de medición y tablas de
  resultados. Es el material con el que se tomaron las decisiones. Léase `trabajo/README.md`.

Todo lo que se afirma aquí está medido y el fichero que lo respalda se nombra al lado. Si algo
no está medido, se dice.

---

## 1. En una pantalla

| | |
|---|---|
| Qué se entrega el **17-11** | un CSV de 199.319 filas (H/A/D por frame), este repositorio en público y un documento técnico corto |
| Métrica de ranking | **spellF1** (posesiones acertadas por solape temporal). Desempate: macroF1 por frame |
| Mínimo oficial a superar | **0,2433** (`baseline_smooth50`) |
| **Dónde estamos, medido** | **0,4073** de media en 7 partidos de Bundesliga degradados para parecerse a nuestros datos, dejando cada partido fuera del entrenamiento del detector (0,3965 sin el detector de fronteras). El Smart Tagging (8-11) suma **+0,5 a +0,9 pp estimados** con un tagging simulado: el IDSSE no tiene uno real, así que esa parte no es medible del todo (§2) |
| En los 5 minutos públicos | 0,4148 (0,4444 sin el tagging, 0,4903 sin el detector) — **lectura, no criterio**: 17 posesiones, una fallada mueve 6 puntos |
| La entrega de hoy | `out/submission.csv`, generada el 8-11 con `config.py` tal cual está (sha256 `e7938350…`); pasa el validador oficial; 103 tests en verde. Con `smart_tagging_team_weight = 0` sale la del 7-11 (`5d97eccc…`), y además con `boundary_discount_weight = 0` la del 31-10 (`73c361d5…`) |
| Qué configuración lleva | la de `config.py`; cada valor tiene escrito al lado de dónde sale. **Ninguno viene ya de los 5 minutos públicos** (prohibido desde el 22-10) |

Si todo se parase hoy, hay algo válido que entregar y está 15 puntos por encima del mínimo.

---

## 2. Qué está hecho

### El programa (`src/mtah/`)

- Lectura del tracking táctico (`io/atd.py`), homografía (`io/homography.py`) y Smart Tagging
  (`io/smart_tagging.py`). El tracking y, desde el 8-11, el equipo de las fases del Smart
  Tagging están conectados al pipeline; la homografía tiene el lector escrito y no se usa.
- Features geométricas por frame (`features/geometry.py`): control del balón por equipo,
  actividad de los jugadores, forma de equipo, velocidad del balón.
- Modelo en dos mitades (`models/emissions.py`): *vivo/parado* sale de la cinemática de los
  jugadores y no necesita el balón; *local/visitante* sale de la geometría balón-jugador. Se
  separan porque el balón falta en el **41 %** de los frames puntuados y la primera mitad
  sigue funcionando sin él.
- Decodificación Viterbi offline (`decode/viterbi.py`) con costes de cambio y duración mínima.
  Gana al voto por mayoría del baseline en las dos métricas a la vez (tabla en `README.md`).
- Nunca se escribe `X` en un frame puntuado: en spellF1 da igual y en macroF1 se paga.
- CLI: `python -m mtah status | build | validate | evaluate`.
- Scorer oficial vendorizado (`vendor/evaluation`, MIT) con un test de regresión que reproduce
  los números publicados de los cuatro baselines oficiales.

### El arnés de medición (`src/mtah/training/`, fuera de la ruta de reproducción)

Es la pieza que convirtió "no podemos medir" en "podemos medir en siete partidos completos":

- `idsse.py`: descarga y adapta el dataset IDSSE/Bundesliga (Bassek et al., *Sci Data* 2025,
  CC-BY 4.0): 7 partidos con posesión y estado del balón etiquetados por frame a 25 Hz.
- `convention.py`: lleva su convenio de posesión (cambia en cada toque) al del reto (cambia
  solo con posesión controlada).
- `degrade.py`: estropea el tracking limpio hasta que se parece al nuestro (balón ausente el
  41 %, identidades mezcladas, ruido). Tres semillas (`out/cache/recal_*_s{0,1,2}.npz`).
- `harness.py`: puntúa el pipeline sobre un partido completo con etiquetas reales.
- `recalibrate.py`: re-deriva los hiperparámetros sobre esos partidos (medianas por clase +
  descenso por coordenadas en 4 partidos de ajuste, comprobación en 3 de validación).

### Las decisiones medidas (cronología)

| Fecha | Qué | Resultado | Dónde |
|---|---|---|---|
| 22-23 oct | Las reglas cambian: los 5 min públicos no son conjunto de ajuste; nada de etiquetado manual | 16 valores de `config.py` quedaban inadmisibles | `trabajo/analisis-y-propuesta.md` §1 |
| 28 oct | Recalibración completa sobre IDSSE, sin ventana pública | cambiaron 7 valores; +1,2 pp en los 3 partidos de validación | `trabajo/handoff-plugin-data/datos/recalibracion.json` |
| 29 oct | Auditoría del análisis | "no se sobreajustó" **no se sostenía**; la validación es más fácil que el ajuste | `trabajo/handoff-plugin-data/VALIDACION.md` |
| 29-30 oct | Validación cruzada: el procedimiento entero, sobre las **35 particiones** posibles de 4 ajuste / 3 validación | **+1,71 pp** fuera del ajuste (IC 95 % [+1,37, +2,05], 32/35) y +3,27 pp dentro → transfiere, y **la mitad de la ganancia es sobreajuste** | `VALIDACION-CRUZADA.md`, `datos/splits_cruzados.json` |
| 31 oct | Rejilla ampliada (64 → 82 valores), mismas 35 particiones, comparación por pares | +0,62 pp en ajuste, **−0,13 pp en validación** (14/35): descartada | `datos/splits_cruzados_ampliada.json`, `compara_rejillas.py` |
| 31 oct | `activity_smooth` 12 → 50 | +1,11 pp en validación, positivo en las 3 semillas, neutro en ajuste; tasa de balón parado 44,9 % como el IDSSE | `datos/activity_smooth.json` |
| 31 oct | Tres números mal citados en `config.py` corregidos; convención de redondeo de los centros | el fichero produce exactamente lo que dice | `config.py` docstring, commits `0e1560e`, `d805776` |
| 2 nov | Datos del dashboard regenerados con el `config.py` de hoy | reproducen los del 28-10 en todo lo que no depende de la config | `regenera_datos.py` |
| 7 nov | Portado lo útil de `DeteccionPosesionBalon` (ver abajo). **La entrega no cambia**: CSV idéntico byte a byte antes y después (sha256 `73c361d5…`) | 73 tests en verde (55 + 18) | `io/atd.py`, `features/tagging.py`, `decode/viterbi.py`, `training/weak_labels.py` |
| 7 nov | **Detector de fronteras** entrenado en el IDSSE, medido con leave-one-match-out anidado y tres brazos (ver abajo) | **+1,17 pp** spellF1 y +1,12 pp macroF1 sobre el modelo anterior (21 ejecuciones, 16 mejoran); **+1,07 pp sobre encarecer los cambios sin detector, positivo en los 7 partidos** | `training/boundaries.py`, `datos/fronteras_lomo_anidado.json` |
| 7 nov | **El detector entra en la entrega.** Entrenado con los 7 partidos; peso 0,95 y escala 2,0 elegidos con el mismo procedimiento, fuera de muestra | reexportar da el mismo modelo byte a byte; la entrega cambia en el 3,8 % de los frames puntuados (369 → 333 segmentos) | `models/boundary_detector.{joblib,json}`, `config.py` |
| 7 nov | Curva de visibilidad del balón (Paso cero de `plan-detector-balon.md`) | **no es plana**: de 59 % a 70 / 80 / 90 % de balón visible da +2,2 / +3,5 / +6,3 pp recuperando primero lo fácil, en los 7 partidos | `training/visibility.py`, `datos/curva_visibilidad.json` |

**Qué entró el 7-11 desde `DeteccionPosesionBalon`** (la otra línea, hecha sin acceso a este repo):

- **Unidades de velocidad, ahora con un fallo explícito.** `atd.load()` compara el canal de velocidad con la derivada de la
  posición y falla si el cociente sale de [0,8, 1,25]. Medido en BAR-ATH: **0,992**, así que es m/s aunque el XML diga kph.
  Si alguien divide por 3,6 fiándose del metadato, el pipeline falla en vez de predecir `D` en casi todo.
- **El Smart Tagging se lee por código.** Los `DEFENDING IN *` nombran al equipo **sin** balón (la entrega acuerda con el
  equipo tagueado un 16-26 % en esas fases, contra un 84 % en `BUILD UP`/`PROGRESSION`) y son espejo de las fases de
  posesión. `features/tagging.py::team_logit` los sumaba como si fueran del equipo con balón; no estaba conectado, así
  que no afectó a ninguna entrega, pero lo habría hecho al conectarlo.
- **Primera medición de "de quién es" sobre el partido del reto** (`python -m mtah.training.weak_labels`), contra las
  fases del Smart Tagging (63.682 frames puntuados, fases de ~18 s, no ground truth):

  | | acuerdo | clase mayoritaria en los mismos frames |
  |---|---|---|
  | emisión por frame | **0,765** | 0,659 |
  | emisión, balón visible | 0,755 | 0,674 |
  | emisión, balón no visible | 0,812 | 0,592 |
  | salida decodificada (frames vivos) | **0,791** | 0,676 |

  Y sube con la confianza de la emisión: 0,60 → 0,76 → 0,84 → 0,86 por cuartiles. La señal del arnés se sostiene en
  BAR-ATH. Es diagnóstico: **no se eligen pesos maximizando este acuerdo.**
- **El hueco para el detector de fronteras.** `viterbi.decode(..., switch_discount=...)` multiplica los costes de cambio
  de cada frame por `1 - descuento` donde un detector vea una transición. Fracción y no resta: los cambios cuestan muy
  distinto (robo 40, reinicio 14) y una resta acotada por el más barato apenas movería un robo. Con ceros reproduce el
  decodificador bit a bit (test); un descuento de 1 o más falla en vez de hacer gratis fragmentar.

**El detector de fronteras, medido el 7-11** (`python -m mtah.training.boundaries`, 8 min con la caché):

- **Qué es.** Gradient boosting (`HistGradientBoostingClassifier`) sobre 109 features. Para 13 señales por frame
  (control balón-jugador, distancia al más cercano por equipo, actividad, velocidad y dirección del balón, velocidad
  de cada centroide, forma, y los dos logits de la emisión actual): la media en una ventana *después* del frame y su
  diferencia con la de *antes*, a 0,24 / 0,48 / 1 / 2 s; más el tiempo desde y hasta la última vez que se vio el
  balón. Objetivo: estar a ±12 frames de un cambio de etiqueta (en el convenio del reto). Fuera de muestra: AUC
  0,90-0,92 y precisión media 0,55-0,61, con un 6 % de positivos.
- **Cómo se midió.** Para cada partido de fuera, el detector se entrena con los otros 6, y el peso del descuento y la
  escala de los costes de cambio se eligen sobre esos 6 con puntuaciones fuera de muestra. El de fuera no influye en
  nada. Las semillas 1 y 2 de degradación usan el modelo y la elección ya fijados. El control reproduce exactamente
  el 0,3965 de §3. Tres brazos con la misma elección anidada, porque encarecer los cambios podría mejorar por sí solo:

  | sobre 7 partidos × 3 semillas | media | ejecuciones que mejoran | partidos que mejoran (media de semillas) |
  |---|---|---|---|
  | solo encarecer los cambios | +0,10 pp | 12/21 | 4/7 |
  | solo el descuento del detector | +0,48 pp | 15/21 | 4/7 |
  | **los dos** | **+1,17 pp** | 16/21 | 5/7 |
  | **los dos frente a solo encarecer** | **+1,07 pp** | 16/21 | **7/7** |

  Semilla 0: 0,3965 → **0,4073**, y macroF1 sube a la vez (+1,12 pp). Encarecer por sí solo es ruido; el detector
  rinde cuando el decodificador corta *solo* donde él ve frontera. Los dos partidos que pierden frente al control
  (J03WOH, J03WOY) eligieron escala 2,0, el borde de la rejilla. No se amplía: el 31-10 ya se vio que ampliar
  rejillas sobreajusta.
- **Qué no dice.** Es Bundesliga degradada; en BAR-ATH no hay forma de medirlo (la ventana pública no es criterio).
  Lo que sí se pudo comprobar, el 7-11, está en el punto siguiente.
- **Las tres pruebas de que aporta** (`python -m mtah.training.evidence stress | tagging`; tablas en
  `trabajo/handoff-plugin-data/datos/evidencia_*.json`). Con huecos al azar el recorrido reproduce al bit el LOMO
  de arriba, y el script falla si no.
  1. **Estrés: el balón se pierde donde duele.** El mismo detector, peso y escala de cada pliegue (entrenados y
     elegidos con huecos al azar, como la entrega), pero el partido de fuera se degrada con los huecos centrados
     donde hay gente a < 5 m del balón (`crowded`, sin etiquetas: así describe DATA.md el ATD) o, peor caso, junto
     a los cambios de posesión reales (`transitions`). Misma longitud de racha y mismo 59 % visible; con las rachas
     reales no se puede esconder mucho más del 57 % de los cambios. Media de semillas por partido:

     | huecos | cambios sin balón | AUC | los dos − control | los dos − solo encarecer |
     |---|---|---|---|---|
     | al azar | 41 % | 0,91 | +1,17 pp [+0,21, +2,07], 5/7 | +1,07 pp [+0,51, +1,55], 7/7 (p 0,008) |
     | abarrotados | 49 % | 0,91 | +1,07 pp [−0,22, +2,20], 6/7 | +1,14 pp [+0,41, +1,86], 6/7 |
     | junto a los cambios | 57 % | 0,90 | +0,41 pp [−0,16, +0,96], 4/7 | +0,81 pp [+0,40, +1,20], 6/7 |

     IC 95 % bootstrap sobre los 7 partidos. **Con los huecos donde los pone el ATD la ganancia se mantiene**; en
     el peor caso se reduce a +0,4 pp y deja de ser distinguible de cero, pero no es negativa, y lo que aporta el
     detector sobre encarecer los cambios sigue claro. No hay motivo para retirarlo; si alguna vez se retira, es
     `boundary_discount_weight = 0`.
  2. **Mecanismo: arregla lo que el diagnóstico decía.** Mismo recorrido, media de 7 partidos × 3 semillas, huecos
     al azar (en las otras dos colocaciones el patrón es el mismo):

     | | control | solo encarecer | los dos |
     |---|---|---|---|
     | cambios: precisión / recall a ±0,5 s | 0,30 / 0,24 | 0,31 / 0,24 | **0,42 / 0,33** |
     | cambios: F1 a ±1 s | 0,35 | 0,35 | **0,42** |
     | error a ±12 frames de un cambio real | 48,6 % | 48,7 % | 46,9 % |
     | error en el resto | 17,4 % | 17,1 % | 16,3 % |
     | recall de posesiones < 2 s / 2-5 s / > 10 s | 1,8 / 13,3 / 59,6 % | 1,4 / 12,4 / 59,8 % | 3,1 / 14,0 / 59,5 % |

     La ganancia sale de **dónde** se corta (+0,10 de F1 de cambios con el mismo número de posesiones que solo
     encarecer), no de cortar más o menos. Las posesiones cortas casi no se recuperan: el techo sigue ahí.
  3. **BAR-ATH sin etiquetas: neutro.** Los saques (`SET PIECES`) y las transiciones (`ATTACKING / DEFENSIVE
     TRANSITION`) del Smart Tagging, que el pipeline no usa, como ventanas ±1 s. De los cortes que el detector
     añade (34) cae dentro el 44 %, y de los que quita (70) también el 44 %, frente a un 34 % al azar. Ventanas de
     transición con algún corte: 21 → 19 de 24. Ni a favor ni en contra: el tagging es demasiado grueso (ventanas
     de saque de 5 a 35 s) para distinguir este cambio. No se elige nada con esto.
- **En la entrega desde el 7-11.** `python -m mtah.training.boundaries export` lo entrena con los 7 partidos y elige
  peso y escala con el mismo procedimiento (cada partido puntuado por un detector que no lo vio): **peso 0,95, escala
  2,0**, spellF1 fuera de muestra 0,4111 frente a 0,4089 con escala 1,5. Los dos valores son el borde de su rejilla,
  igual que en los 2 pliegues que perdían; no se amplía (el 31-10 se vio que ampliar sobreajusta). El modelo va en el
  repo (`models/boundary_detector.joblib`, 420 KB) con su sha256, la versión de scikit-learn y la config de features
  con que se entrenó; si cualquiera de las tres no cuadra, `build` falla en vez de puntuar basura. Reexportar da el
  mismo fichero byte a byte.
- **Comprobaciones sin etiquetas en BAR-ATH**, que no eligen nada pero avisarían de un salto de dominio grosero: el
  score medio en los frames puntuados es 0,063 (0,061-0,064 en el IDSSE) y el 2,2 % pasa de 0,5 (2,7 % en el IDSSE).
  Cambia el 3,8 % de los frames puntuados y baja de 369 a 333 segmentos, la misma dirección que en el arnés. En la
  ventana pública baja de 0,4903 a 0,4444 (14 → 10 posesiones de 17) con macroF1 igual; es menos de una posesión y
  la ventana no es criterio, pero queda dicho.

**La curva de visibilidad, medida el 7-11** (`python -m mtah.training.visibility`, 1 min). Mismo modelo y mismas
degradaciones, devolviendo huecos de balón enteros. spellF1 medio de los 7 partidos:

| balón visible | huecos al azar | primero los fáciles (balón lejos de todos) |
|---|---|---|
| 59 % (como el ATD) | 0,3965 | 0,3965 |
| 70 % | 0,4255 | 0,4181 |
| 80 % | 0,4460 | 0,4310 |
| 90 % | 0,4747 | 0,4596 |
| 100 % | 0,4977 | 0,4977 |

Mejora en los 7 partidos en cada nivel. La columna que vale es la de la derecha: unos 2-3 pp por cada 10 puntos de
visibilidad. La pendiente es lo fiable; el nivel está inflado porque la degradación reparte los huecos al azar.

**El Smart Tagging como evidencia de equipo, desde el 8-11** (`python -m mtah.training.team_tags` y `... robust`;
tablas en `datos/smart_tagging_lomo.json` y `smart_tagging_robustez.json`).

- **Por qué.** En el IDSSE, el 60 % de los frames fallados son local por visitante (71 % con el balón visible). Las
  fases de Metrica con equipo (BUILD UP, PROGRESSION, CREATING CHANCES, LONG BALL y los DEFENDING IN * en espejo)
  dicen de quién es el balón en el 40 % de los frames puntuados, en tramos de ~20 s.
- **Cómo entra.** `features/tagging.team_tags` da +1/−1/0 por frame y `models/emissions.team_tag_logit` lo suma al
  logit de local con `smart_tagging_team_weight`. Va *después* del detector de fronteras, que lee `home_logit` y se
  entrenó sin tagging: sumarlo antes cambiaría sus entradas. Peso 0 = la entrega del 7-11 al byte.
- **El problema: el IDSSE no tiene Smart Tagging.** Se simula sobre sus etiquetas, copiando lo que se mide del real
  sin etiquetas: longitudes de tramo, cobertura, tramos separados (nunca un cambio directo de equipo) y equipo
  mayoritario del tramo. Lo que no se sabe es cuánto acierta. Lo único medible es cuánto coincide con nuestra
  emisión en BAR-ATH: **0,762**, y por cuartil de confianza 0,60 / 0,75 / 0,84 / 0,86. Un tagging simulado
  *perfecto* coincide solo 0,706, así que el real coincide más que uno perfecto: o copia parte de nuestros errores
  (Metrica etiqueta con el mismo tracking), o cae en los tramos fáciles, o nuestra emisión acierta más aquí.
- **Qué se midió.** LOMO anidado sobre el pipeline entregado (detector incluido), 7 partidos × 3 semillas:

  | escenario del tagging | ¿compatible con BAR-ATH? | peso 0,5 fijo | peso 1,0 fijo |
  |---|---|---|---|
  | perfecto | no (acuerdo 0,706) | +1,89 pp | +2,64 pp |
  | 10 % / 20 % de tramos invertidos | no | +1,24 / +0,67 | +1,56 / +0,48 |
  | copia nuestro equipo en la mitad de los tramos | solo el acuerdo medio | +0,47 | +0,41 |
  | cae donde nuestra emisión está segura | sí | +0,90 | +0,59 |
  | **mejor ajuste a los cuatro cuartiles** (fácil + 20 % de copia) | **sí, el mejor** | **+0,61** [−0,15, +1,51], 5/7 | +0,10 |
  | copia del todo nuestra emisión | no (daría 0,807) | −1,12 | −2,06 |

- **Decisión: 0,5.** Gana en todos los escenarios compatibles con lo medido y solo pierde en uno que el acuerdo
  descarta. Se eligió entre dos pesos fijos por robustez, mirando estos resultados: es una elección hecha sobre
  los partidos de evaluación y así hay que contarla. La elección automática por pliegue salía más ruidosa (+0,39
  en el mejor ajuste; −0,20 en el de copia, donde casi siempre elegía 0).
- **Qué no dice.** Es una simulación con un supuesto sobre los errores del tagging real, calibrado con un solo
  número por cuartil. Ninguno de los intervalos del escenario que mejor ajusta excluye el cero. En BAR-ATH cambia el
  2,1 % de los frames puntuados y baja de 334 a 316 segmentos. En la ventana pública baja de 0,4444 a 0,4148, menos
  de una posesión (10 de 17 en los dos casos).

### Los documentos (`trabajo/`)

- `resumen-simple.md` — el reto sin jerga. Empezar por aquí.
- `analisis-y-propuesta.md` — análisis del reto, propuesta, qué se probó y qué no funcionó.
- `datos-externos.md` — qué datasets externos existen, cuál usar (IDSSE) y cómo degradarlos.
- `handoff-plugin-data/VALIDACION.md` y `VALIDACION-CRUZADA.md` — las auditorías. La segunda
  corrige a la primera en lo del sobreajuste.
- `handoff-plugin-data/datos/` — todas las tablas; `handoff-plugin-data/*.py` los scripts que
  las producen.
- Hay además un dashboard interactivo (artefacto privado de claude.ai, "Posesión FC Medallo")
  alimentado con exactamente esas tablas. Pedidle el enlace a Brayan.

---

## 3. Los números, y cómo leerlos

Semilla 0 de degradación, `config.py` del 31-10 (`trabajo/handoff-plugin-data/datos/scores_por_partido.csv`):

| partido | grupo | antes del 28-10 | hoy |
|---|---|---|---|
| J03WMX | ajuste | 0,3434 | 0,3607 |
| J03WN1 | ajuste | 0,3254 | 0,3435 |
| J03WOH | ajuste | 0,4442 | 0,4678 |
| J03WOY | ajuste | 0,3598 | 0,3790 |
| J03WPY | validación | 0,3807 | 0,3889 |
| J03WQQ | validación | 0,4209 | 0,4584 |
| J03WR9 | validación | 0,3488 | 0,3771 |
| **media de los 7** | | 0,3748 | **0,3965** |

Cuatro avisos para no leerlos mal:

1. **Son partidos de otra liga con otro tracker, degradados a mano.** Miden diferencias
   *relativas* entre versiones del modelo; el nivel absoluto en el partido de la competición
   puede ser otro.
2. **El grupo de validación es más fácil que el de ajuste** en todas las configuraciones
   (0,41 frente a 0,39). El 0,41 no es una predicción de la nota.
3. **La columna "antes del 28-10" es una reconstrucción.** Ese `config.py` no se conservó (el
   repo no tuvo commits hasta el 31-10). La reconstrucción puntúa 0,3–0,7 pp por debajo de lo
   que se midió aquel día (`datos/robustez_semillas.json`, `_procedencia`).
4. **Una partición no distingue configuraciones.** Con las otras dos semillas, diferencias de
   menos de un punto cambian de signo. Lo único que distingue es la validación cruzada sobre
   las 35 particiones (§2).

Dónde falla el modelo hoy (`datos/diagnostico_errores.csv`, config de hoy):

- **Balón parado se detecta bien**: 92 % de los frames D acertados.
- **De qué equipo es, regular**: real H → predicho H 74 %, real A → A 69 %. El error H/A es
  ~20 % de los frames vivos, y la recalibración lo reequilibró hacia H, no lo eliminó.
- **Las posesiones cortas se pierden**: las de menos de 5 s son el **39 %** de todas y se
  recuperan ~8 % (3 % las de <2 s, 11 % las de 2-5 s). Con ese recall, aunque la precisión
  fuese perfecta spellF1 no pasaría de ≈0,50. **Es el techo real del modelo actual.**
- **El error se concentra junto a las transiciones**: a ±12 frames de un cambio real falla uno
  de cada dos frames; a más de 250 frames, el 7 %.
- **Cuando el balón lleva más de 5 s sin verse, el error sube al 27 %** (frente al 14 % con el
  balón visible). El balón no se pierde en oclusiones cortas: se pierde en bloques de minutos.

---

## 4. Qué falta

Por orden de lo que yo haría. Nada de esto está empezado salvo que se diga.

1. **El documento técnico final.** No existe como tal; el material está en `trabajo/`. La
   frase sobre el sobreajuste que va en él es la del §2 (29-30 oct), no "no se sobreajustó".
2. **Dejar el repositorio listo para hacerlo público.** Antes de publicar: comprobar otra vez
   que no entra nada de `data/` (hoy: 0 ficheros, auditado en cada push), citar el paper del
   IDSSE y acreditar a la DFL (**condición de la licencia CC-BY, no cortesía**), revisar
   `vendor/NOTICE.md`, y probar la reproducción en una máquina limpia (**no se ha hecho**).
3. **Un detector de balón propio sobre el vídeo** (la línea de visión). La curva de visibilidad
   (§2, 7-11) dice que merece la pena *si llega*: subir al 70 % de balón visible vale unos
   +2 pp, y al 80 % unos +3,5 pp. Pero el plan lo estima en 7,5 días, quedan 10 hasta la
   entrega, y su criterio de abandono (50 % de recall al 90 % de precisión a los tres días)
   sigue en pie. No se ha empezado.
4. **Los saques del Smart Tagging** como pista de reinicios. El equipo de las fases ya está
   conectado (§2, 8-11). Los 59 saques no: sus ventanas son de 5 a 35 s, y el 7-11 los cortes
   del detector caían en ellas igual que al azar, así que como pista de dónde cortar son
   demasiado gruesos. Lo que sí podrían dar es el equipo que saca (saques de puerta y
   córners; los de banda van con `N/A`).
5. **Mejorar la degradación** para que las rachas sin balón caigan donde caen de verdad
   (zonas abarrotadas, balones aéreos). Desde el 7-11 `degrade.apply(gap_weights=...)` las
   coloca por peso y se usa como prueba de estrés del detector (§2); el entrenamiento y las
   elecciones siguen con huecos al azar. Pasarlo todo a `crowded` es un cambio de arnés que
   movería todos los números, y no se ha hecho.
6. **Una emisión aprendida** sobre los partidos IDSSE degradados (fase C de
   `datos-externos.md`). Prueba rápida del 7-11 con las features del detector (medio partido
   para entrenar, medio para medir): acierto por frame 0,783, igual que el geométrico; mezclada
   con él, cambios del mismo tamaño que el ruido. No es un atajo; si se hace, con el mismo LOMO
   anidado.
7. **Validar en otro dominio** (SkillCorner, fase D) para saber si nos hemos ajustado al IDSSE.

Lo que **ya no** merece la pena: seguir afinando hiperparámetros del modelo actual. Con 7
partidos de ajuste no hay señal para distinguir más valores (lo demostró la rejilla ampliada:
5 h 48 min de cómputo para −0,13 pp), y el techo de 0,50 lo pone el recall de las posesiones
cortas, no los umbrales.

---

## 5. Problemas que vimos

Lo que salió mal, cómo se detectó y en qué quedó. Está aquí para que nadie se lo vuelva a tragar.

| Problema | Cómo se vio | Qué se hizo |
|---|---|---|
| 16 valores de `config.py` venían de los 5 minutos públicos, que desde el 22-10 no pueden usarse para ajustar | lectura de `RULES.md` tras el cambio | re-derivados el 28-10 desde IDSSE; `config.py` dice la procedencia de cada valor |
| "La recalibración no se sobreajustó" | auditoría del 29-10: ganaba +1,9 en ajuste y +1,2 en validación con 1 de 3 partidos en contra | validación cruzada sobre 35 particiones: +3,3 / +1,7; la mitad es sobreajuste. Frase corregida en todos los documentos |
| Una afirmación intermedia del 29-10 decía que el descenso "no transfiere nada" | era falsa; salió de mirar un solo split | retirada; la validación cruzada la sustituye |
| `config.py` citaba tres números que el fichero no producía (validación 0,3957; tasa D 45,3 %; "300 y 450 ya bajan") | al re-medir el 31-10 | corregidos; convención: los centros de clase se escriben redondeados (los dieciséis dígitos del script son ruido de semilla) |
| Siete hiperparámetros se clavaban en el borde de su rejilla | estabilidad por parámetro sobre los 35 splits | rejilla ampliada y medida: no mejora la validación. Diagnóstico correcto, implicación equivocada |
| `activity_smooth` = 12 solo lo elegía 1 de 35 particiones | mismo análisis | vuelto a 50, justificado con las tres semillas (+1,1 pp validación) |
| La configuración anterior al 28-10 no se conservó | el repo no tenía commits | se reconstruyó; la reconstrucción va 0,3–0,7 pp por debajo y así se etiqueta donde aparece |
| El conjunto de validación es más fácil que el de ajuste | en las cuatro configuraciones | no se cita el 0,41 como expectativa |
| Sesgo de agregación en dos categorías de 512 y 854 frames (0,14 % de los datos) | `VALIDACION.md` hallazgo 4 | se muestran con su `n` y marcadas; no se concluye nada de ellas |
| La recalibración mejora spellF1 emitiendo **menos** posesiones y más largas (~1.540 → ~1.390 predichas frente a 1.945 reales) | `VALIDACION.md` hallazgo 5 | se sabe y se dice: mejora la métrica alejándose de la estructura real; el techo de 0,50 viene de ahí |
| Estimé 50 min para la rejilla ampliada; tardó 348 | las ventanas grandes multiplican la caché por partido | documentado para la próxima vez |
| Un comando de ejemplo con un marcador (`USUARIO`) se ejecutó tal cual e invitó al repo privado a un usuario ajeno de GitHub | se vio en la salida | invitación revocada en el acto; verificado 0 pendientes y un solo colaborador. Nunca llegó a tener acceso (las invitaciones requieren aceptación) |

Pregunta abierta, contestada en parte el 8-11: el pipeline usa el tracking táctico y el equipo
de las fases del Smart Tagging. La homografía y el tracking del broadcast se leen pero no se
usan, por los motivos del `README.md` (no separan jugada de parada; otra línea de tiempo).

---

## 6. Lo que no vemos (cobertura no evaluada)

- **No hay ground truth del partido de la competición.** Todo lo medido es transferencia desde
  Bundesliga: otra liga, otra cámara, otro tracker. Nadie sabe cuánto se pierde en el salto.
- **Siete partidos son pocos.** Las semillas varían el ruido de degradación, no los partidos.
- **La degradación es un modelo nuestro del ATD.** Si el ATD real pierde el balón de otra forma,
  el arnés es optimista o pesimista y no sabemos cuál.
- **No se ha medido la dispersión de los centros de clase** (solo la mediana), así que no se
  puede juzgar cuán separables son H y A por esas señales.
- **No hay intervalos por partido**: una evaluación por partido y configuración.
- **No se ha probado la reproducción en una máquina limpia** (venv nuevo, datos recién
  descargados, `./scripts/reproduce.sh`). Hay que hacerlo antes de publicar.
- **No se ha ejecutado SAST, linting en CI ni pruebas de carga**; los tests son unitarios y de
  contrato.
- **No consta que se haya probado el lector con los datos de muestra de Metrica** (fase E de
  `datos-externos.md`).
- **La vía supervisada solo está medida en el arnés**: el detector de fronteras gana +1,1 pp
  en Bundesliga degradada y +0,4 a +1,1 pp con los huecos colocados de forma adversa; si
  transfiere a BAR-ATH no se puede comprobar (el Smart Tagging sale neutro).
- **El peso del Smart Tagging sale de un tagging simulado**, no del real: cuánto acierta el de
  Metrica no se puede medir sin etiquetas, solo acotar por su acuerdo con nuestra emisión.

---

## 7. Cómo revisarlo y reproducirlo

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # numpy 2.5.4 pide Python >= 3.12
pip install -r requirements-dev.txt
python -m pytest -q                      # 103 tests
```

Para regenerar la entrega hacen falta los datos del partido (no están en el repo, no pueden
estar: 5 GB y no redistribuibles). Descargarlos del Drive de la organización y:

```bash
source scripts/env.sh                    # MTAH_DATA_DIR, por defecto ~/Documents/Hackaton/data
python -m mtah status
./scripts/reproduce.sh FC_Medallo        # -> out/FC_Medallo.csv
```

Para rehacer las mediciones del arnés hace falta además `pip install -r requirements-train.txt`
(kloppy, para descargar el IDSSE). La primera llamada a `idsse.load` descarga cada partido (~2 min
cada uno) y lo deja en `out/cache/`; después `python -m mtah.training.boundaries` y
`python -m mtah.training.visibility` corren en minutos. Los
scripts de `trabajo/handoff-plugin-data/` se escribieron para correr desde la carpeta de
trabajo original (`~/Documents/Hackaton/`) con rutas absolutas; cada uno dice arriba qué hace
y qué fichero produce.

---

## 8. Reglas que no se pueden romper

- **Los datos del partido no se redistribuyen** (vídeo, tracking, homografía, Smart Tagging).
  `.gitignore` los excluye; no des-ignorarlos. Cada push se audita.
- **No se sube ningún extracto del partido a un LLM en la nube.** Salidas de nuestro modelo,
  el ground truth público y agregados sobre partidos públicos sí.
- **Los 5 minutos públicos no son conjunto de ajuste ni de selección.** Se miran, no se usan
  para elegir.
- **Sin etiquetado manual.**
- **El repo público tiene que regenerar el CSV entregado exactamente**, por eso
  `requirements.txt` solo lleva numpy y pandas pinneados y `training/` queda fuera de la ruta.
