# Datos externos para un modelo supervisado de posesión

**More than a Hack 2027 — Fase 1 · investigación previa a decidir el v2**
Fecha: 20 de octubre de 2026 · Entrega: 17 de noviembre de 2026 (28 días)

---

## 1. Veredicto

**Sí merece la pena, y hay un ganador claro: el dataset IDSSE/Bundesliga (Bassek et al.,
*Scientific Data* 2025).** Trae `BallPossession` y `BallStatus` **por frame a 25 Hz**, que es
literalmente la etiqueta `H`/`A`/`D` del reto, sobre 7 partidos completos (~665 min ≈ 1,0 M de
frames etiquetados, frente a los 7.500 del GT público: **×133**). Es CC-BY 4.0, descarga directa
desde figshare sin registro, y `kloppy` ya lo carga con una llamada.

Pero con una condición que decide el resultado: **entrenar solo las emisiones, no la
segmentación.** El convenio de posesión del IDSSE es más fino que el del reto (medí 5,20
posesiones/min frente a 3,40 en la ventana pública), y `spellF1` puntúa exactamente eso. La
arquitectura actual del repo (`models/emissions.py` + `decode/viterbi.py`) ya separa esas dos
mitades, así que encaja sin rediseño: **datos externos → emisiones; GT público de 5 min →
costes de transición.**

Secundarios, por orden de valor:

- **SkillCorner opendata** (MIT, 20 partidos): no tanto para entrenar como para **medir** cómo
  se degrada un tracking automático real. Es el único candidato que *ya* es visión por
  computador sobre broadcast, con huecos y extrapolación.
- **Metrica sample-data**: mismo proveedor y mismo formato EPTS que el reto, lo que lo hace
  ideal para validar el *parser*. Como datos de entrenamiento tiene una fuga grave (§5.2).
- **SoccerNet, StatsBomb, SoccerTrack v2**: descartados, con motivos en §4.

---

## 2. Alcance y método

Qué hice: verificación de fuentes primarias, sin tocar `mtah-possession/` ni `repo-mtah/` ni
los assets del partido (no se necesitan y no se usaron).

- Leí `RULES.md`, `DATA.md` y `EVALUATION.md` del repo oficial para fijar los requisitos.
- Para cada candidato descargué los **ficheros reales** y medí estadísticas, en lugar de
  fiarme de la documentación: un partido completo de SkillCorner (97 MB), el tracking y los
  eventos completos de Metrica Sample Game 3 (53 MB), y la cola del XML de posiciones de un
  partido del IDSSE (70 MB, suficiente para aislar el `FrameSet` del balón de una parte).
- Verifiqué el código fuente de `kloppy` para confirmar el mapeo de campos, no solo la
  documentación.
- Caractericé el convenio de etiqueta del GT público de 5 minutos para poder compararlo
  con el de cada fuente externa.

Todo lo que digo **«medido»** sale de esos ficheros. Lo que digo **«documentado»** sale del
paper, del README o de `DATA.md` y no lo he comprobado de primera mano.

### 2.1 Los dos requisitos de RULES.md

> 1. *«The organisers can access and inspect it … a public model, a commercial API, an open dataset.»*
> 2. *«Your technical document explains it.»*

Esto **no** exige licencia permisiva: exige accesibilidad. Un dataset con registro abierto que
los organizadores también pueden completar cumple; lo que queda fuera es «a private in-house
model or an unavailable internal dataset». Aun así, ordeno los candidatos penalizando la
fricción de acceso, porque cada paso extra es una pregunta que un juez puede no querer hacerse.

Nada de esto afecta a la otra prohibición relevante: **no subir los assets del partido a ningún
LLM en cloud**. Los datos externos son públicos y ajenos al partido, así que no entran en ese
supuesto; los del reto siguen sin salir de la máquina.

---

## 3. Tabla comparativa

| | **IDSSE / Bundesliga** | **SkillCorner opendata** | **Metrica sample-data** | **SoccerNet (GSR / BAS)** | **StatsBomb open** | **PFF FC WC2022** | **SoccerTrack v2** |
|---|---|---|---|---|---|---|---|
| **Licencia** | CC-BY 4.0 (+ autorización DFL) | **MIT** | Sin fichero de licencia; README pide uso responsable y atribución | NDA de SoccerNet, sin redistribución | Acuerdo de usuario, no comercial | No publicada | CC-BY 4.0 |
| **Acceso** | figshare, directo, sin registro | GitHub + Git LFS | GitHub, directo | HF *gated*, **aprobación manual** | GitHub, directo | Formulario | HF *gated* |
| **Inspeccionable por los organizadores** | Sí, trivial | Sí, trivial | Sí, trivial | Sí, pero firmando NDA | Sí | Solo rellenando el formulario | Sí, aceptando condiciones |
| **Tracking de 22 jugadores + balón** | Sí | Sí (broadcast) | Sí | Solo clips de 30 s | **No** | Sí | Sí (balón interpolado) |
| **Frecuencia** | **25 Hz** | 10 Hz | **25 Hz** | 25 Hz | — | n/d | n/d |
| **Etiqueta de posesión** | **`BallPossession` por frame** | **`possession.group` por frame** | Derivable de eventos a frame exacto | No | Derivable, pero sin tracking | n/d | No (solo eventos) |
| **Balón parado / vivo** | **`BallStatus` por frame** | Vía `phases_of_play` | Derivable (huecos entre bloques) | No | Parcial | n/d | No |
| **Identidad de jugador** | Sí (oficial) | Sí (~97%) | Sí | Sí | — | Sí | Sí |
| **Volumen etiquetado** | **~665 min** (7 partidos) | ~1.600 min útiles (20 partidos) | ~287 min (3 partidos) | 100 min sin etiqueta de posesión | — | ~6.100 min | 934 min sin etiqueta |
| **Esfuerzo de conversión** | **Bajo** (kloppy) | Medio (JSONL + remuestreo) | Bajo (EPTS) / Medio (CSV) | Alto e inútil | — | Medio | Alto |
| **Veredicto** | **Usar** | **Usar, para medir degradación** | Usar para validar el parser | Descartar | Descartar | Opcional | Descartar |

---

## 4. Ficha por dataset

### 4.1 IDSSE / Bundesliga — **la recomendación**

Bassek, M., Rein, R., Weber, H. et al. *An integrated dataset of spatiotemporal and event data
in elite soccer*. Sci Data 12, 195 (2025). DOI del artículo `10.1038/s41597-025-04505-y`;
datos en figshare `10.6084/m9.figshare.28196177`.

- **Contenido** (documentado): 7 partidos completos, Bundesliga 1ª y 2ª división 2022/23,
  con metadatos, event data oficial y position data de TRACAB a 25 Hz.
- **Licencia** (verificado en la API de figshare): `CC BY 4.0`, `is_public: true`,
  `is_embargoed: false`, 21 ficheros, 2,63 GB. Pide citar el paper y acreditar a la DFL.
- **Formato** (verificado descargando el XML): formato DFL. Un `FrameSet` por persona y por
  parte, con `<Frame N= T= X= Y= D= S= A= M=/>`. Coordenadas en **metros**, origen en el centro
  del campo, `PitchSize X="105.00" Y="68.00"`, `S` en m/s, intervalos de 0,040 s → **25 Hz
  exacto**, igual que el vídeo táctico del reto.
- **La etiqueta** (verificado): el `FrameSet` con `TeamId="BALL"` lleva, en cada frame,
  `BallPossession` (`1`=local, `2`=visitante) y `BallStatus` (`1`=vivo, `0`=parado). El mapeo
  al espacio de etiquetas del reto es directo y sin ambigüedad:

  ```
  BallStatus == 0                     -> D
  BallStatus == 1 and Possession == 1 -> H
  BallStatus == 1 and Possession == 2 -> A
  ```

- **Dos ventajas que no tiene ningún otro candidato** (ambas verificadas en el fichero):
  1. El balón trae **`Z` (altura)** en todos los frames. Esto permite identificar los balones
     aéreos, que es justo donde `DATA.md` dice que el ATD pierde el balón. Sin esta columna, el
     sesgo de la degradación hacia duelos aéreos (§5.1) habría que inventarlo.
  2. El balón tiene coordenadas **también cuando `BallStatus=0`**. Es decir, hay posición de
     balón durante las paradas. Metrica no la tiene, y eso es lo que lo descalifica como fuente
     principal (§5.2).
- **Carga** (verificado en el código fuente de kloppy, `kloppy/_providers/sportec.py` y
  `kloppy/infra/serializers/tracking/sportec/deserializer.py`): existe
  `sportec.load_open_tracking_data(match_id=...)` con los 7 `match_id` del IDSSE cableados y la
  URL de figshare construida; el deserializador mapea `BallStatus` → `ball_state`
  (`ALIVE`/`DEAD`) y `BallPossession` → `ball_owning_team`. El parámetro
  `coordinates="metrica"` devuelve coordenadas normalizadas 0–1 con origen arriba-izquierda
  sobre un campo de 105×68 (verificado en `MetricaCoordinateSystem`): **la convención exacta
  del reto**, sin conversión propia.

  ```python
  from kloppy import sportec
  ds = sportec.load_open_tracking_data(match_id="J03WPY", coordinates="metrica")
  ```

- **Medido** (2ª parte de `J03WMX`, 75.259 frames de balón = 50,2 min):

  | | |
  |---|---|
  | Reparto crudo | `H` 26,2% · `A` 24,5% · `D` 49,3% |
  | Posesiones | 261 → **5,20/min**, media 5,8 s, mediana 2,8 s |
  | Rachas de `D` | 62 → 1,24/min, media 23,9 s |

- **Limitación real**: 7 partidos de Bundesliga 2022/23, de los cuales 5 son del mismo equipo
  local (Fortuna Düsseldorf, 2ª división). La diversidad táctica es escasa y el estilo de
  LaLiga/Barcelona no está representado. Para aprender «qué aspecto tiene que el local tenga el
  balón» en términos geométricos eso importa poco; para cualquier cosa que dependa de la forma
  de equipo concreta, importa.

### 4.2 SkillCorner opendata — el espejo del dominio

- **Licencia** (verificado, fichero `LICENSE`): **MIT**. La más permisiva de todas.
- **Volumen** (verificado en el árbol del repo): **20 partidos**, no 10 como dice el README —
  está desactualizado. Todos con `tracking_extrapolated.jsonl`, `dynamic_events.csv`,
  `phases_of_play.csv` y `match.json`. A-League australiana 2024/25. El tracking va por Git LFS
  (~97 MB por partido).
- **Formato** (verificado): JSONL, una línea por frame, **10 fps**. Coordenadas en metros con
  origen en el centro del campo. Cada frame trae `ball_data` (con `is_detected`),
  `player_data` (lista con `x`, `y`, `player_id`, `is_detected`) y **`possession: {player_id,
  group}`** con `group ∈ {"home team", "away team", null}`.
- **Medido** (partido `1925299`, 61.282 frames con `period` = 102,1 min):

  | | |
  |---|---|
  | Balón con coordenadas | 78,2% |
  | Balón **detectado** (no extrapolado) | **63,0%** |
  | Frames sin ningún objeto trackeado | 21,8% (cortes y repeticiones del broadcast) |
  | `player_data` | o 22 objetos o 0 — nunca intermedio |
  | Objetos **detectados** por frame (en frames con tracking) | media 12,73 · p50 14 · p90 18 · máx 21 |
  | `possession.group` | `home` 36,5% · `away` 21,6% · `null` 41,9% |
  | De los `null`, fuera de una fase de juego | 94,3% |
  | Rachas sin balón detectado | n=214, media 10,6 s, p50 3,4 s, p90 34,2 s, máx 142,8 s |
  | Spells de equipo | n=335, media 10,6 s |

- **Para qué sirve de verdad.** Es el único candidato que **ya es tracking automático por
  visión**, con el balón perdiéndose por oclusión real. Su 63,0% de balón detectado está
  notablemente cerca del 56,9% del ATD del reto, y su distribución de rachas sin balón es una
  referencia empírica para el parámetro más importante de la degradación (§5.1). Úsalo para
  **estimar la forma de las degradaciones**, y como conjunto de validación en un dominio
  distinto.
- **Por qué no es la fuente principal**: 10 Hz contra 25 Hz. Subir a 25 Hz interpola
  suavidad que el ATD no tiene, y eso enseñaría al modelo a confiar en una continuidad falsa.
  El 21,8% de frames sin tracking son cortes de realización que el vídeo táctico no tiene.

### 4.3 Metrica sample-data — mismo formato, pero con una fuga

- **Licencia** (verificado): **no hay fichero `LICENSE`** (404). El README pide uso responsable
  y atribución. Descarga pública desde GitHub, así que cumple el requisito de inspección, pero
  conviene decirlo tal cual en el documento técnico en vez de llamarlo «open source».
- **Contenido** (verificado): 3 partidos. Sample Game 1 y 2 en CSV propio de Metrica; **Sample
  Game 3 en formato EPTS/FIFA, idéntico al del reto** (`<FrameRate>25</FrameRate>`,
  `FieldSize 105×68`, coordenadas 0–1). 143.761 frames = **95,8 min** por partido; ~287 min en
  total.
- **Eventos** (verificado, SG3): 3.620 eventos con `start.frame` y `end.frame` exactos —
  `CARRY` 1.395, `PASS` 1.121, `CHALLENGE` 335, `RECOVERY` 294, `BALL LOST` 288, `SET PIECE` 90
  (con subtipos `THROW IN`, `CORNER KICK`, `FREE KICK`, `KICK OFF`…), `BALL OUT` 64, `SHOT` 20.
  Cada uno con equipo. La derivación de posesión es mecánica y a precisión de frame.
- **El problema, medido**:

  | | |
  |---|---|
  | Objetos por frame | **22 en el 100% de los frames.** Sin huecos, sin fragmentación |
  | Balón presente, global | 67,78% |
  | Balón presente **dentro de bloque on-ball** | **100,0%** |
  | Balón presente **fuera de bloque on-ball** | **5,2%** |

  El patrón es **el inverso exacto del ATD**. En el reto el balón se pierde en el juego
  abarrotado; en Metrica el balón está siempre durante el juego y desaparece en las paradas,
  porque sencillamente dejan de trackearlo. Entrenar sobre esto sin corregirlo enseña al modelo
  que «hay balón ⇒ el juego está vivo», con una precisión casi perfecta en entrenamiento y
  catastrófica sobre el ATD. Y no se puede arreglar añadiendo ruido: **no se puede inventar una
  posición de balón donde no la hay.** Solo se puede diluir la correlación tirando balón
  también durante el juego, lo que deja el conjunto permanentemente sesgado en el eje que más
  importa.
- **Para qué sí**: como **test del parser**. SG3 entra por `io/atd.py` casi tal cual (22 slots
  en vez de 46), y es la única forma de verificar el lector de EPTS contra un fichero del mismo
  proveedor sin gastar la ventana pública.

### 4.4 Descartados

- **SoccerNet.** GSR son 200 clips de **30 s** (100 min) con coordenadas de campo, rol, equipo
  y dorsal, pero **sin etiqueta de posesión**. Y 30 segundos no contienen posesiones: la media
  del GT público es 14,2 s, así que cada clip daría una o dos, truncadas por ambos extremos —
  inservible para calibrar nada relacionado con `spellF1`. Ball Action Spotting etiqueta
  acciones sobre vídeo pero no trae tracking. Además el acceso es **NDA vía Hugging Face con
  aprobación manual**, lo que añade latencia y una restricción de redistribución, a cambio de
  nada que los otros no den mejor.
- **StatsBomb open data.** Es event data. Los *freeze frames* de 360 son instantáneas en los
  eventos, no tracking continuo, y la cobertura de 360 es parcial. No hay forma de construir la
  representación de 46 slots por frame. Sigue siendo útil como referencia del convenio de qué
  cuenta como una posesión, nada más.
- **SoccerTrack v2.** CC-BY 4.0 y 934 min, pero fútbol universitario, *gated* en HF, el track
  del balón es una **reconstrucción interpolada entre eventos** y no hay etiqueta de posesión.
- **PFF FC World Cup 2022.** 64 partidos con tracking y eventos, gratis mediante formulario, y
  `kloppy` tiene proveedor `PFF`. Es el mayor volumen disponible con diferencia. Lo dejo como
  **opcional**: no encontré los términos de licencia publicados, y el acceso por formulario
  significa que un organizador que quiera inspeccionarlo tiene que rellenarlo él. Si en algún
  momento hace falta más volumen, este es el sitio al que ir — pero antes hay que leer los
  términos.

---

## 5. Estrategia de domain randomization

El problema en una frase: **todo dato externo está corregido y tiene identidad; el ATD del reto
no.** Un modelo entrenado sobre datos limpios aprende a apoyarse en señales que en inferencia no
existen.

El principio que organiza todo lo que sigue: **cada parámetro de degradación se estima del ATD
real del partido, sin etiquetas.** Eso es legítimo —no toca el ground truth privado, solo la
distribución marginal de los datos de entrada, que están publicados— y es lo que convierte esto
en *domain adaptation* medida en vez de *domain randomization* a ojo. Las estadísticas del
apartado 4.2 (SkillCorner) sirven de comprobación de cordura: si un parámetro estimado sobre el
ATD cae lejos de lo que da un tracking automático real, probablemente está mal estimado.

> Nota de implementación: la degradación debe aplicarse **en tiempo de carga de cada época, con
> semilla distinta**, no una vez sobre el disco. Un corpus degradado y congelado es un corpus
> con un patrón de oclusión fijo que la red aprende a memorizar.

### 5.1 Pérdida del balón — la degradación que decide el resultado

**Objetivo** (documentado en `DATA.md`): balón ausente en el **43,1%** de todos los frames y el
**40,7%** de los puntuados.

**Lo que no hay que hacer:** tirar cada frame de forma independiente con p=0,43. Eso da rachas
geométricas de media 1,75 frames (0,07 s). Las oclusiones reales duran segundos. Medí las de
SkillCorner: media 10,6 s, mediana 3,4 s, p90 34,2 s, máx 142,8 s — una cola larguísima que una
Bernoulli i.i.d. no produce nunca. Un modelo entrenado contra huecos de 2 frames aprende a
interpolar; contra huecos de 300 frames tiene que aprender a sobrevivir sin balón, que es lo que
hace falta en el 41% de los frames puntuados.

**Modelo propuesto: cadena de Markov de dos estados con duración explícita, modulada por
contexto.**

1. **Marginal.** Fijar `P(ausente) = 0,407` sobre frames puntuados.
2. **Distribución de longitud de racha.** Estimar de `features.geometry.ball_gap_lengths`
   aplicada al ATD real (la función ya existe en el repo). Ajustar una log-normal o una binomial
   negativa a las longitudes de racha de ausencia **y** de presencia por separado, y muestrear de
   ahí en lugar de usar una tasa por frame. Dos distribuciones, cuatro parámetros.
   - *Estadística de origen:* rachas de `NaN` en `tracking.ball_xy` sobre los bloques puntuados.
3. **Sesgo por contexto.** `DATA.md` dice que el balón se pierde en «occlusions, crowded boxes
   and balls in the air». Hay que reproducir ese sesgo, porque si no el conjunto de frames sin
   balón queda compuesto de situaciones fáciles y el modelo infravalora lo difícil que es el caso
   real. Ajustar una regresión logística

   ```
   logit P(balón ausente en el frame t) = b0 + b1·densidad(t) + b2·altura_balón(t) + b3·dist_centro(t)
   ```

   donde `densidad(t)` es el número de jugadores en un radio de 5 m alrededor del balón
   (`features.geometry.count_within`, que ya está implementada).
   - *Estadística de origen:* esta regresión **se ajusta sobre el ATD real sin etiquetas**. Los
     jugadores se observan incluso cuando el balón falta, así que `densidad` es medible en los
     dos casos; la variable dependiente es simplemente `isnan(ball_xy)`. `b2` no se puede
     estimar del ATD (no hay altura), así que se estima del IDSSE: `P(ausente | Z > 2 m)` contra
     `P(ausente | Z ≤ 2 m)` usando el modelo ya ajustado, y se transfiere. **Esta es la razón
     concreta por la que el IDSSE gana: es el único que trae `Z`.**
4. **Falsos positivos de balón.** El ATD a veces coloca el balón donde no está. Introducir, con
   tasa pequeña, frames en los que la posición del balón salta a un punto incorrecto.
   - *Estadística de origen:* frecuencia de frames en el ATD real donde la velocidad implícita
     del balón entre frames consecutivos supera `constants.BALL_MAX_SPEED` (40 m/s). Ya está el
     umbral definido en el repo.

### 5.2 Corrección obligatoria si se usa Metrica

Antes de cualquier otra degradación, hay que **romper la correlación balón↔juego vivo**
(§4.3: 100% contra 5,2%). Opciones, en orden de preferencia:

1. **No usar Metrica para entrenar**, solo para validar el parser. Es lo que recomiendo.
2. Si se usa: descartar de entrenamiento todos los frames fuera de bloque on-ball. Se pierde el
   33% del partido y, con él, casi toda la clase `D` — lo que deja a Metrica aportando solo
   ejemplos de `H`/`A`, que es la mitad del problema que menos falta hace.
3. Lo que **no** vale: aplicar la degradación de §5.1 encima y dar por hecho que basta. Diluye
   la correlación pero no la elimina, y deja un sesgo residual que no se puede medir.

El IDSSE no tiene este problema: verificado que los frames con `BallStatus=0` llevan `X`, `Y`,
`Z` reales.

### 5.3 Destrucción de identidad y asignación a slots

**Objetivo**: 46 slots (24 local, 22 visitante) para 22 jugadores, sin identidad, que
«aparecen, desaparecen e intercambian identidad».

1. **Asignación.** Un slot no es un jugador: es un *objeto con vida*. Trocear la trayectoria de
   cada jugador en segmentos; asignar cada segmento a un slot libre del equipo correcto; al
   terminar un segmento el slot queda libre y puede reutilizarse más tarde para otro jugador.
2. **Distribución de longitud de segmento.** Es lo que fija cuánto dura la continuidad de un
   track antes de romperse.
   - *Estadística de origen:* para cada uno de los 46 slots del ATD real, contar los ciclos
     maximales de no-`NaN` y su duración. Eso da directamente la distribución de longitud de
     segmento y el patrón de ocupación de slots, sin ningún supuesto.
3. **Cambios de identidad.** Cuando dos jugadores del mismo equipo se cruzan a menos de `d`
   metros, intercambiar sus slots con probabilidad `p_swap`.
   - *Estadística de origen:* `p_swap` no se observa directamente sin verdad terreno, así que
     usar como proxy la **tasa de teletransportes**: frames donde la posición de un slot salta
     más de lo que permite una velocidad humana plausible (p. ej. > 12 m/s) entre frames
     consecutivos. Cada salto es un cambio de identidad o una readquisición; la tasa acota las
     dos juntas, que para este propósito es lo que importa. Cruzar con SkillCorner, que
     documenta ~97% de identidad correcta.
4. **Error de equipo.** El ATD fija el equipo por slot en el XML, así que un intercambio entre
   equipos es imposible en la representación — pero el tracker puede meter una detección del
   equipo equivocado en un slot. Modelarlo a tasa baja (del orden del 1–3%, anclado al 97% de
   SkillCorner), porque es el error que más daño hace: convierte directamente `H` en `A`.

### 5.4 Número de objetos trackeados

**Objetivo** (documentado): media de **20,5** objetos sobre frames puntuados, y
«occasionally exceeds 22 when the tracker duplicates someone».

1. Por frame, muestrear el número de objetos presentes del **histograma empírico** de slots
   no-`NaN` por frame puntuado del ATD real. No de un rango uniforme 20–23: la forma de la cola
   importa, y el histograma es gratis de calcular.
2. **Qué jugadores se caen no es uniforme.** Sesgar hacia (a) jugadores dentro de aglomeraciones
   y (b) jugadores lejos del balón o del centro de la acción. Un tracker pierde antes a quien
   está solapado con otro.
   - *Estadística de origen:* ajustar `P(slot ausente | nº de jugadores a < 2 m)` sobre el ATD
     real, con `count_within(radii=(2.0,))`, que ya está en `geometry.py`.
3. **Duplicados.** Con probabilidad pequeña, un jugador ocupa dos slots a la vez con un
   desplazamiento de ~0,5–1,5 m. Es lo que produce los frames con más de 22 objetos.
   - *Estadística de origen:* fracción de frames puntuados del ATD con más de 22 slots activos.

### 5.5 Ruido de posición y de velocidad

El ruido i.i.d. gaussiano es un error clásico aquí: es trivialmente filtrable con una media
móvil, así que el modelo aprende a suavizar y nunca aprende a desconfiar.

1. **Usar ruido temporalmente correlado** (AR(1)): `e_t = ρ·e_{t-1} + N(0, σ²)`, aplicado por
   track. Dos parámetros, `ρ` y `σ`.
2. **Hacer `σ` dependiente de la posición.** El error de una detección proyectada por homografía
   crece con la distancia a la cámara. Escalar `σ` con la distancia al centro del campo en el eje
   largo.
3. *Estadística de origen:* no hay posiciones verdaderas del partido, así que la estimación es
   **espectral**. Calcular la autocovarianza del desplazamiento entre frames (lags 1 y 2) de los
   tracks del ATD real y de los tracks limpios del IDSSE; ajustar `ρ` y `σ` para que la
   autocovarianza del IDSSE degradado iguale la del ATD. Comprobación cruzada: la distribución
   de la velocidad `s` (que el ATD da directamente, en m/s) debe coincidir en los dos conjuntos
   tras la degradación. Esa comparación es la única que no requiere ningún supuesto.

### 5.6 Armonización geométrica

| | IDSSE | Reto |
|---|---|---|
| Unidades | metros, origen en el centro | normalizado 0–1 |
| Campo | 105,0 × 68,0 | 105,3 × 68,0 (`constants.PITCH`) |
| Frecuencia | 25 Hz | 25 Hz |

- La conversión la hace `kloppy` con `coordinates="metrica"`. Conviene verificarla contra una
  conversión propia antes de fiarse.
- Los 0,3 m de diferencia de largo son un 0,3%: irrelevante salvo que se comparen distancias
  absolutas en metros entre conjuntos. Como `geometry.py` trabaja en metros, lo correcto es
  normalizar con el campo de cada fuente y volver a metros con el campo del reto.
- **Dirección de ataque.** Hay que alinear qué equipo ataca hacia dónde en cada parte, o todas
  las features de forma de equipo quedan reflejadas. El EPTS de Metrica trae
  `attack_direction_first_half`; el DFL también lo recoge. Es un error silencioso y caro:
  conviene un test que lo verifique sobre cada partido importado.

### 5.7 Qué **no** degradar

- **No** reproducir los cortes y repeticiones de SkillCorner (21,8% de frames sin tracking). El
  vídeo táctico del reto no los tiene; su homografía es válida en el 100% de los frames.
- **No** degradar el conjunto de validación con una semilla distinta de la que se usará para
  decidir. La comparación entre configuraciones tiene que ser sobre la misma degradación.
- **No** degradar la ventana pública de 5 minutos. Es el único dato real etiquetado que hay y se
  usa tal cual.

---

## 6. El convenio de etiqueta: dónde está el riesgo de verdad

Esto es lo que más me ha sorprendido de la investigación y lo que cambia la forma de usar los
datos externos.

Caractericé la ventana pública (frames 11.185–18.684) y la comparé con el IDSSE crudo:

| | GT público (5 min) | IDSSE crudo (50 min) |
|---|---|---|
| Posesiones por minuto | **3,40** | **5,20** |
| Duración media de posesión | 14,2 s | 5,8 s |
| Mediana | 11,8 s (`H`) / 7,4 s (`A`) | 2,8 s |
| Rachas de `D` por minuto | 0,80 | 1,24 |
| % de frames en `D` | 19,6% | 49,3% |

La secuencia real de la ventana pública es:
`A7 H8 A3 H42 A7 D6 H29 A3 H42 A20 H1 A16 H1 A4 D20 A36 H3 D18 A8 D14 H12` (segundos).

Hay posesiones de 42 segundos junto a otras de 1 segundo. El convenio del reto **no** parte una
posesión larga por cada contacto del rival, pero **sí** registra toques brevísimos. El
`BallPossession` de la DFL cambia en cada toque individual, y su `BallStatus=0` cubre cualquier
interrupción arbitral. Son convenios distintos.

Probé si un suavizado morfológico (duración mínima de posesión, duración mínima de `D`, puente
sobre spells cortos del rival) acerca uno al otro. Con `min_poss=3 s, min_dead=3 s, puente=3 s`
el IDSSE baja a 3,31 posesiones/min — muy cerca del 3,40 objetivo — pero la **duración media se
queda en 9,2 s frente a 14,2 s**. El suavizado arregla el recuento y no la forma.

**Dos advertencias honestas sobre esos números objetivo:**

1. **La ventana pública es una muestra sesgada.** Son los primeros 5 minutos tras el saque
   inicial, el tramo del partido con menos interrupciones. Su 19,6% de `D` es casi con certeza
   muy inferior al del partido completo; el 49,3% del IDSSE sobre una parte entera es más
   representativo de lo que es el fútbol. Calibrar el convenio contra esos 19,6% sería ajustar a
   una ventana no representativa.
2. **17 posesiones son 17 observaciones.** `EVALUATION.md` avisa de que en `spellF1` las
   diferencias por debajo de ~0,04 son ruido sobre el partido completo; sobre 5 minutos el ruido
   es mucho mayor. Estos cuatro números sirven para descartar configuraciones absurdas, no para
   afinar.

**Conclusión operativa, y es la recomendación central de este informe:**

> **No aprender la segmentación de los datos externos.** Entrenar con ellos únicamente el
> **modelo de emisión** — la pregunta «dado este frame, ¿qué pinta tiene que lo tenga el local,
> el visitante, o que esté parado?» —, que es local al frame y donde el convenio apenas influye.
> Dejar la segmentación donde ya está: en los **costes de transición del Viterbi**, ajustados
> contra los 5 minutos públicos y contra el conocimiento previo de cuánto dura una posesión.

Esto encaja exactamente con cómo está construido el repo. `models/emissions.py` produce una
matriz de costes `(n, 3)` y `decode/viterbi.py` la segmenta. Sustituir la primera por una versión
entrenada **no toca la segunda**. Y la factorización que ya documenta `emissions.py`
—`P(H) = P(vivo)·P(local|vivo)`— sigue siendo la correcta: `BallStatus` entrena la primera mitad
y `BallPossession` la segunda, cada una con su propio conjunto de frames válidos.

---

## 7. Esfuerzo

Estimaciones para una persona, asumiendo que el pipeline actual funciona y hay 28 días.

| Fase | Qué incluye | Esfuerzo | Valor |
|---|---|---|---|
| **A. Calibrar el v1 geométrico** | Cargar IDSSE con kloppy, adaptador a `Tracking`, extraer etiquetas, ajustar los ~6 escalares de `config.DEFAULT` (`tau`, `activity_center_ms`, `activity_scale_ms`, `ball_gap_weight`…) por búsqueda directa contra las etiquetas externas | **1–2 días** | **Alto.** Hoy esos valores son conjeturas. Ajustarlos es robusto al salto de dominio porque son pocos parámetros y de significado físico |
| **B. Módulo de degradación** | §5.1–5.6, más los scripts que estiman cada parámetro del ATD real | **3–4 días** | **Alto.** Sin esto, la fase C es contraproducente. Y los scripts de estimación son por sí solos una caracterización útil del ATD |
| **C. Emisión aprendida** | Regresión logística o *gradient boosting* sobre las features de `geometry.py`, entrenada sobre IDSSE degradado. Dos cabezas: vivo/parado y local/visitante | **2–3 días** | **Medio-alto.** Es donde está el margen: el 41% de frames puntuados sin balón necesita un modelo de forma y actividad de equipo que es difícil de diseñar a mano |
| **D. SkillCorner como validación** | Parser JSONL, conversión de coordenadas, etiquetas de `possession.group` + `phases_of_play` | **1–2 días** | **Medio.** Validar en un dominio distinto es la única defensa contra haberse ajustado al IDSSE |
| **E. Metrica como test del parser** | SG3 entra casi por `io/atd.py` | **0,5 días** | **Medio.** Barato y verifica el lector de EPTS sin gastar la ventana pública |
| **F. Modelo de secuencia** | Sustituir el Viterbi por algo aprendido | 5+ días | **Bajo / negativo.** Es exactamente donde el convenio de etiqueta no transfiere (§6). No lo haría con este calendario |

**Ruta recomendada: A → B → E → C → D.** Unos 8–11 días, dejando margen. El criterio de parada es
explícito: **medir cada fase contra los 5 minutos públicos y contra el suelo oficial de 0,2433.
Si C no supera a A, se entrega A.** La fase A tiene valor por sí sola y no depende de que el
resto salga bien.

**Riesgo principal:** que B se coma el calendario. La degradación tiene muchas piezas y cada una
invita a refinarse. Mitigación: implementar §5.1 (pérdida del balón) y §5.3 (identidad) primero,
que son las dos que de verdad separan los dominios, y tratar §5.4 y §5.5 como refinamientos
opcionales.

---

## 8. Cobertura: qué no he evaluado

Para que nadie lea este informe con más confianza de la que merece:

- **No he ejecutado nada del pipeline del equipo ni he tocado los datos del partido.** Todas las
  estadísticas del ATD que cito vienen de `DATA.md`, no de medirlas. Los parámetros de §5 están
  especificados pero **ninguno está estimado**: hace falta correr esos cálculos sobre el ATD
  real, y es posible que alguna estimación salga mal condicionada.
- **No he entrenado ni evaluado ningún modelo.** No hay ninguna evidencia empírica aquí de que la
  vía supervisada supere al v1 geométrico. El argumento es de headroom, no de resultado.
- **Las estadísticas del IDSSE salen de una parte de un partido** (75.259 frames del `J03WMX`),
  no de los siete. El reparto `H`/`A`/`D` y las tasas de posesión podrían variar entre partidos.
- **Las de SkillCorner salen de un partido de veinte.**
- **No he verificado los términos de PFF FC** ni descargado su dataset.
- **No he comprobado que `kloppy` descargue correctamente desde figshare** — leí el código que
  construye las URLs y verifiqué a mano que esas URLs existen y sirven el fichero, pero no
  ejecuté `load_open_tracking_data`.
- **La licencia de Metrica sample-data es ambigua**, no inexistente ni permisiva: no hay fichero
  de licencia. Si se usa, conviene decirlo así en el documento técnico y no adornarlo.

---

## 9. Fuentes

- Repo oficial del reto: <https://github.com/metrica-sports/more-than-a-hack-2027> (`RULES.md`, `DATA.md`, `EVALUATION.md`)
- Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Sci Data 12, 195 — <https://doi.org/10.1038/s41597-025-04505-y> · datos: <https://doi.org/10.6084/m9.figshare.28196177> · código: <https://github.com/spoho-datascience/idsse-data>
- SkillCorner open data — <https://github.com/SkillCorner/opendata> (licencia MIT)
- Metrica Sports sample data — <https://github.com/metrica-sports/sample-data>
- kloppy — <https://github.com/PySport/kloppy> (`kloppy/_providers/sportec.py`, `kloppy/infra/serializers/tracking/sportec/deserializer.py`)
- SoccerNet — <https://www.soccer-net.org/data> · GSR: <https://arxiv.org/abs/2404.11335>
- StatsBomb open data — <https://github.com/statsbomb/open-data>
- PFF FC 2022 World Cup dataset — <https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset>
- SoccerTrack v2 — <https://arxiv.org/abs/2508.01802> · <https://huggingface.co/datasets/atomscott/soccertrack-v2>
