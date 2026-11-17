# More than a Hack 2027 — análisis del reto y propuesta

**FC Medallo** · Día 1 de trabajo · 24 de octubre de 2026

**En corto:** el pipeline ya produce una entrega válida de 199.319 filas. Va por **spellF1
0,3902** medido sobre siete partidos completos, contra un suelo oficial de **0,2433**. El
hallazgo del día es que **los 5 minutos de ground truth que publican no representan al
partido** (88,2% de balón visible contra 59,3% del partido entero), y eso estaba
descalibrando el detector de balón parado. Lo demás de este documento es el reto explicado
de arriba abajo, qué propongo, y qué probé.

---

# PARTE 1 — Qué nos están pidiendo

## El problema, sin adornos

Un partido: **Barcelona – Athletic**, LaLiga 2024-25, jornada 2. Nos dan el vídeo y el
tracking automático, y hay que decir, **para cada uno de los 199.319 frames**, quién tiene
el balón.

Cuatro etiquetas:

| | |
|---|---|
| `H` | el Barça tiene el balón, con el juego vivo |
| `A` | el Athletic tiene el balón, con el juego vivo |
| `D` | balón parado, no está en juego |
| `X` | no contesto |

La entrega es un CSV de 199.319 filas. Ya está. Eso es lo que se puntúa.

De esos frames, **se puntúan 158.857** (unos 106 minutos): las dos partes. Lo de antes del
saque, el descanso y lo de después del pitido final no cuenta, así que ahí podemos poner
`X` tranquilamente.

## Lo que hace difícil el problema

Tres cosas, y conviene tenerlas claras porque condicionan todo:

**1. El balón no está en el 41% de los frames puntuados.** Y no es que no se vea en el
vídeo: es que el tracker automático de Metrica lo pierde. Se pierde justo donde más
importa — áreas abarrotadas, duelos aéreos, entradas.

**2. No hay identidad de jugador.** Hay **46 "slots" de tracking para 22 jugadores**. Los
slots aparecen, desaparecen y se intercambian la identidad según el tracker pierde y
recupera gente. O sea: no puedes decir "el jugador 7 tiene el balón", solo puedes razonar
de forma colectiva.

**3. Solo publican 5 minutos de respuestas correctas.** 7.500 frames, 17 posesiones. El
resto del ground truth es privado hasta que cierre la competición.

## Qué datos nos dan

| | **táctica** | **broadcast** |
|---|---|---|
| papel | **la referencia — la fila *i* de la entrega es el frame *i*** | opcional, **no sincronizado** |
| vídeo | 199.319 frames, 25 fps | su propia línea temporal, con cortes y repeticiones |
| tracking (ATD) | 199.319 registros | 164.488 |
| homografía | 100% utilizable | 65,8% |
| Smart Tagging | 638 instancias | — |
| balón visible | 56,9% | 50,5% |

- **ATD (Automatic Tracking Data)**: posición y velocidad de cada objeto por frame,
  coordenadas normalizadas 0–1, velocidad en m/s, `NaN` donde falta. Es automático y **sin
  corregir**.
- **Smart Tagging**: 638 fases tácticas detectadas automáticamente (Build Up, Progression,
  Set Pieces...), cada una con equipo. **No es ground truth**: lo genera Metrica del mismo
  vídeo que tenemos, sin revisión. Es una pista gratis, no una respuesta.
- **Homografía**: matriz 3×3 por frame para pasar de posición en el campo a píxel. Solo
  hace falta si vamos a leer el vídeo.

> Ojo con un detalle que tumba entregas: si vuestro fichero sale con ~164k filas,
> indexasteis el vídeo broadcast por error.

## Cómo nos puntúan

Dos métricas, y la que manda **no es la obvia**.

**`macroF1`** — F1 medio de las tres clases, frame a frame. Es la de toda la vida.
Solo sirve de desempate.

**`spellF1`** — **esta es la que rankea.** Trata cada posesión como **un objeto**, no como
un montón de frames. Empareja cada posesión que predices con una real por solapamiento
temporal, y promedia sobre cinco umbrales (0,3 a 0,7).

La diferencia importa muchísimo. Mirad los baselines que ellos mismos publican — es
**el mismo algoritmo cuatro veces**, cambiando solo la ventana de suavizado:

| baseline | macroF1 | spellF1 |
|---|---|---|
| `baseline_raw` | 0,6532 | 0,0486 |
| `baseline_smooth25` | **0,6752** | 0,1996 |
| `baseline_smooth50` | 0,6724 | **0,2433** |
| `baseline_smooth125` | 0,6356 | 0,2242 |

`macroF1` se mueve 0,04 y prefiere una versión. `spellF1` **se mueve por un factor de
cinco** y prefiere otra distinta. Un sistema puede acertar el 97% de los frames y quedar
fatal si reporta el triple de posesiones de las que hubo.

**El suelo oficial a batir es `baseline_smooth50`, spellF1 = 0,2433.**

Dos cosas más de la métrica:

- Un desfase de 1 frame sale **gratis**. Un segundo tarde te cuesta un tercio del resultado.
- Una constante ("todo `H`") saca exactamente **0**.

### Lo que hace la `X`

Esto es importante para diseñar:

- Para `macroF1`, una `X` cuenta como fallo.
- Para `spellF1`, **el scorer la rellena con tu etiqueta anterior** de esa parte.

O sea: callarse equivale a repetir lo último que dijiste. **Dentro de una posesión sale
gratis** — si pierdes el balón diez segundos, callarte es mejor que inventar y partir la
posesión en dos. **En un cambio de posesión sale carísimo**, porque arrastras la etiqueta
vieja al estado nuevo.

### No hay jurado

Esto cambió respecto a 2026. **No se puntúa nada más que el fichero.** Ni originalidad, ni
presentación, ni uso de LLMs. El repo y el documento técnico no puntúan: existen para que
el número signifique algo y se pueda reproducir.

## Las reglas de posesión (lo que de verdad piden)

Esto está en `CHALLENGE.md` y vale la pena leerlo entero, pero el resumen:

**Cambio de posesión** = un jugador del equipo contrario gana posesión **controlada**.

- Mientras el juego está vivo, **siempre hay un equipo en posesión**. No existe "disputado".
- En fases de balón suelto o disputado, **la posesión se queda con el último equipo que la
  tuvo**.
- Tras una parada siempre hay un poseedor claro. Lo difícil es detectar cuándo cambia después.

**Qué NO cuenta como cambio de posesión:**
- un simple toque sin control
- un desvío de un pase que no lo intercepta
- un duelo aéreo ganado sin control posterior

**Lo que ya está cerrado:**
- Tres estados y no más. No hay "disputado" ni "desconocido".
- **Una posesión es una racha máxima de un equipo con el balón vivo.** `H H H A A` son dos
  posesiones. `H H D D H H` son dos posesiones con una parada en medio, no una.
- Balón parado es una parada, **nunca** una posesión.
- **No preguntan de quién es el balón cuando está parado.** Si está parado, la respuesta
  es `D` y punto.

**Y el dato más útil de todos:** el ground truth **se deriva mecánicamente del tracking
corregido**. No es una persona viendo el vídeo. Un frame es `D` donde el tracking corregido
marca el balón muerto, y si no, toma el equipo que el tracking dice que lo tiene. **Nada se
suaviza ni se interpola después.**

O sea que la pregunta no es "cómo etiquetó el humano", es "qué entiende el tracking
corregido por muerto y por poseer". Y los 5 minutos públicos son donde se puede mirar.

## Qué se puede usar y qué no

**Se puede casi todo**: cualquier lenguaje, framework, modelo preentrenado, datos externos,
LLMs. Con dos condiciones: que los organizadores puedan acceder e inspeccionarlo, y que el
documento técnico lo explique.

Incluso **mejorar o sustituir el tracking que nos dan es explícitamente legítimo**. Nos lo
dan para que nadie tenga que montar detección y calibración antes de llegar a la pregunta
de verdad.

**Lo que NO se puede, y descalifica:**

1. **Etiquetar a mano, en cualquier cantidad y para cualquier uso, sobre este partido.**
   Ni para la entrega, ni para entrenar, ni para validar. Y **corregir a mano el tracking**
   de este partido (posiciones del balón, cambios de identidad, lo que sea) cuenta como
   etiquetar a mano. Sobre **otros partidos** sí: etiqueta lo que quieras, siempre que los
   organizadores puedan acceder y el documento lo explique.

   > Esto cambió el **22 y 23 de octubre**. Antes decía que etiquetar a mano para entrenar o
   > validar estaba permitido. Ya no. Quien lleve la parte de visión tiene que saberlo.

   Y hay dos reglas nuevas en la misma línea:

   - **Los 5 minutos públicos no son conjunto de entrenamiento ni de ajuste.** Son para
     comprobar que el pipeline es correcto. Lo dicen literalmente: *«That is a rule, not
     just advice»*. Eso incluye sacar de ahí medianas, pesos o constantes.
   - **Nada producido por humanos sobre este partido, de ninguna fuente**: ni eventos de
     Opta, StatsBomb, Wyscout o LaLiga, ni tracking corregido, ni anotaciones. Sobre otros
     partidos, sí.
2. **Usar el ground truth privado**, o cualquier cosa derivada de él, más allá de los 5
   minutos publicados.
3. **Entregar un fichero que el repo no pueda regenerar.** Mismo fichero, no parecido.

Y una que nos toca directamente: **los términos prohíben subir el vídeo, los datos "o
cualquier extracto" a un LLM en cloud.** Usarlos para escribir código no tiene problema;
pegarles filas del ATD sí.

## Calendario

| | |
|---|---|
| datos liberados | 20 de octubre |
| **entrega** | **17 de noviembre** |
| resultados | antes del 26 de noviembre |

Quedan **24 días**.

---

# PARTE 2 — Qué propongo nosotros

## La idea

Un modelo **geométrico, sin entrenamiento**, que lee solo el tracking. Tres piezas:
features geométricos → coste por frame y estado → un Viterbi que segmenta.

La decisión de diseño que lo sostiene todo es **partir la pregunta en dos**:

```
P(H) = P(vivo) · P(local | vivo)
P(A) = P(vivo) · P(visitante | vivo)
P(D) = P(muerto)
```

¿Por qué? Porque las dos mitades se apoyan en cosas distintas:

- **¿Está el balón en juego?** Se responde con la **cinemática de los jugadores**. Con el
  juego parado la gente anda; con el balón vivo corre. **No necesita ver el balón.**
- **¿De quién es?** Se responde con la **geometría balón-jugador**. Sí lo necesita.

Como el balón falta en el 41% de los frames puntuados, meter las dos preguntas en un solo
clasificador significa tirar a la basura la mitad que sí funciona en esos frames. Separadas,
cuando no hay balón la segunda se queda en 0,5 (el frame no opina) y el decodificador
arrastra el estado anterior — que es justo lo que `spellF1` premia.

Y el control del balón se calcula como **suma de `exp(−d/τ)` sobre todos los jugadores del
equipo**, no como "el más cercano gana". Con 46 slots que aparecen y desaparecen, una
decisión basada en un solo jugador salta de equipo en cuanto ese slot se pierde un frame.

## Por qué no un modelo supervisado de entrada

Porque hay **17 posesiones** de ground truth. No da para entrenar. Y como se ve abajo,
esos 5 minutos además no representan al partido.

---

# PARTE 3 — Cómo lo estoy llevando

## La regla

Todo lo que he hecho hoy sale de una sola decisión metodológica:

> **Ningún parámetro puede salir de los 5 minutos públicos.**
> Cada valor tiene que salir, o de **datos externos etiquetados** (otros partidos), o de una
> **medición sin etiquetas sobre el partido completo**.

Desde el 22 de octubre esto es regla, no criterio. Y hay que decirlo claro: **hoy el modelo
no la cumple del todo.** Yo re-derivé el umbral de balón parado de forma limpia, pero el
resto de constantes vienen de antes y salen de la ventana pública: las medianas por clase
de cada señal, los pesos elegidos por AUC, las ventanas de suavizado, los costes de
transición (*«12 de los 16 cambios son directos»*) y la duración mínima (*«6 de 17 duran
menos de 4 s»*). Son **dieciséis valores** en el fichero de configuración, repartidos en unas diez decisiones. Hay que volver a derivarlos todos desde los 7
partidos externos y desde estadísticas sin etiquetas — y es lo primero que hay que hacer,
porque el documento técnico tiene que poder afirmar que no queda nada de la ventana.

## El problema de fondo: no teníamos con qué medir

Hasta hoy, cualquier decisión nuestra se apoyaba en 17 posesiones. Eso es muy poco y además
engañoso. Así que dediqué buena parte del día a **construir un instrumento de medida**.

---

# PARTE 4 — Qué probé y qué salió

## Lo primero: buscar datos externos

Revisé los candidatos públicos de tracking de fútbol (Metrica sample data, SkillCorner,
SoccerNet, StatsBomb, PFF, SoccerTrack). El ganador claro es el **dataset IDSSE**:

- 7 partidos completos de Bundesliga, licencia **CC-BY 4.0**, descarga directa.
- Tracking a **25 Hz**, igual que nuestro vídeo.
- Y lo importante: cada frame trae **`BallPossession`** (1 local, 2 visitante) y
  **`BallStatus`** (1 vivo, 0 parado). O sea, **la etiqueta `H`/`A`/`D` literal, por frame**.
- Son ~1,0 millón de frames etiquetados contra los 7.500 que tenemos. **×133.**

> Trampa que casi me cuesta todo: la librería que lo carga entrega **la velocidad en km/h
> sin convertir**. Si no lo pillas, el umbral de "balón parado" da todos los frames por
> vivos y el modelo entrena contra basura sin fallar en ningún sitio. Lo detecté comparando
> con la velocidad derivada de la posición: ratio 3,590 ≈ 3,6.

## Hallazgo 1: los 5 minutos públicos no representan al partido

Este es el que sostiene todo lo demás. Medido sobre el ATD real:

| | ventana pública | partido completo |
|---|---|---|
| Balón visible | **88,2%** | **59,3%** |
| Frames en balón parado | **19,6%** | ~45% |

Son los primeros 5 minutos tras el saque: el tramo con menos interrupciones y mejor
seguimiento de cámara. **Cualquier estadística medida ahí y llevada al partido entero
aterriza en el sitio equivocado.**

## Hallazgo 2: eso estaba rompiendo el detector de balón parado

El umbral de "balón parado" estaba fijado como el cruce de las medianas de las dos clases
sobre la actividad de los jugadores. Las medianas estaban **bien medidas**. El problema es
que **un cruce de medianas depende de la mezcla de clases de la muestra**.

Trasplantado al partido completo, ese umbral caía en el **percentil 56** de la actividad
real, y el modelo acababa diciendo `D` en el **55,4%** de los frames.

Lo corregí usando como objetivo la tasa de balón parado medida en los 7 partidos completos
del IDSSE (**44,9%**) y la distribución real de actividad de nuestro partido, que se mide
**sin usar ninguna etiqueta**. Es el cambio que más ha dado del día.

## Hallazgo 3: el balón no se pierde en oclusiones, se pierde en bloques de minutos

```
150 bloques sin balón | mediana 10,7 s | máximo 3,7 minutos
solo el 9% de las rachas duran 5 frames o menos
```

La posesión media dura 14,2 s. **Un hueco típico se come casi una posesión entera.** No hay
que aguantar oclusiones de medio segundo: hay que aguantar cegueras de diez segundos.

## La POC: un conjunto de validación de 7 partidos completos

La idea: **coger un partido limpio del IDSSE y ensuciarlo hasta que sus estadísticas
coincidan con las del ATD**, pasarle nuestro pipeline y puntuar contra sus etiquetas reales.

Los parámetros de la degradación se miden sobre nuestro ATD **sin usar etiquetas**. Cómo
quedó la imitación:

| | ATD real | degradado |
|---|---|---|
| Balón visible | 59,31% | 59,39% |
| Objetos por frame | 20,49 | 20,46 |
| Racha sin balón, mediana | 268 frames | 275 |
| Racha sin balón, máximo | 5.537 frames | 5.537 |

También hubo que **alinear el convenio**: el dataset alemán parte la posesión en cada toque
(421 posesiones por partido, 7,7 s de media) y el reto las fusiona (17 posesiones, 14,2 s).
Sin corregir eso le pedíamos al modelo acertar 421 posesiones cuando está diseñado para
producir ~220.

## Resultados

```
spellF1 ventana pública de 5 min ....... 0.4727   (empezamos el día en 0.3200)
spellF1 7 partidos completos ........... 0.3902   (empezamos en 0.3156)
suelo oficial .......................... 0.2433
```

**El número bueno es el 0,3902**, no el 0,4727. La ventana pública es muy optimista.

Y una honestidad importante: **0,3902 tampoco es una predicción de nuestra puntuación**.
El proxy tiene sesgos en las dos direcciones y ninguno está cuantificado:

- *Optimista*: coloca las rachas sin balón al azar, mientras que el ATD las concentra en los
  frames difíciles. Tampoco imita los cambios de identidad entre slots.
- *Incierto*: la Bundesliga no es este partido.

Sirve para **comparar versiones entre sí**, que ya es infinitamente más de lo que teníamos.

## Lo que probé y NO funcionó

**Subir el peso de "balón no visible".** Su cociente de verosimilitudes medido es +1,79 y el
modelo usa +0,50. Parecía infraponderado. A tasa de balón parado igualada, **no mejora**, y
pasado 2,5 se hunde.

**Bajar el peso de la forma de equipo.** Su AUC real es 0,632 y no el 0,812 que decía la
ventana pública, así que lo bajé. Validado con 7 partidos × 3 semillas: **indistinguible**.
Revertido.

> **La lección, que me salió dos veces:** el modelo suma logits, y eso supone que las
> señales son independientes. **No lo son.** La visibilidad del balón va del 9,7% al 90,0%
> según la actividad de los jugadores, porque las causa lo mismo: el tracker en apuros.
> Un AUC o una verosimilitud marginal **no te dicen el peso correcto**.

---

# PARTE 5 — Dónde estamos y qué sigue

Hoy el pipeline produce un CSV válido de 199.319 filas que pasa el validador oficial, con
55 tests en verde. Si todo se parase ahora, hay algo que entregar.

Lo que yo haría a continuación, por orden:

0. ~~Re-derivar los dieciséis valores heredados de la ventana pública~~ — **hecho el 28-10**
   con `training/recalibrate.py`: centros de clase por medianas sobre 4 partidos externos
   degradados, el resto por barrido en esos 4 y comprobación en 3 reservados. Cambiaron 7
   valores; `config.py` documenta la procedencia de cada uno y **ninguno depende ya de la
   ventana pública.** Validado del 29 al 31-10 sobre las 35 particiones posibles: el
   procedimiento gana +1,7 pp (IC 95 % [+1,4, +2,1]) en partidos que no se usaron para
   ajustarlo y +3,3 pp en los de ajuste, así que la mitad de la ganancia es sobreajuste.
   Ampliar la rejilla no mejora la validación (−0,13 pp) y se descartó. El 31-10
   `activity_smooth` volvió a 50 (+1,1 pp en validación con las tres semillas). Con el
   fichero de hoy: ajuste 0,3878, validación 0,4081, los siete 0,3965.
1. **Quitarle el optimismo al conjunto de validación**, haciendo que las rachas sin balón se
   concentren donde se concentran de verdad (zonas abarrotadas, balones aéreos).
2. **Un detector de balón propio sobre el vídeo.** El 41% ausente no es que el balón no se
   vea: es que el tracker lo pierde. Recuperar parte de esos frames mejoraría la mitad del
   modelo que hoy está ciega. Esto conecta directo con la línea de visión por computador.
3. **Conectar el Smart Tagging** como pista para los reinicios.
4. **Una emisión aprendida** sobre los datos externos degradados — pero solo si el arnés dice
   que hay margen.

## Cómo encaja esto con las otras líneas

Quedamos en que cada uno mirase una alternativa y luego comparásemos. Por lo que he visto
hoy, **esto no es una competición entre las tres**: la de visión y la mía se alimentan.

El 41% de frames sin balón es el cuello de botella de todo mi modelo. Es donde la mitad que
decide "de quién es" se queda ciega y lo único que queda es arrastrar el estado anterior. Y
ese 41% **no es que el balón no se vea en el vídeo** — es que el tracker de Metrica lo
pierde. Un detector propio que recupere aunque sea una parte de esos frames **entra directo
en mi pipeline sin tocar el diseño**, y además cambiaría los parámetros del conjunto de
validación: habría que ensuciar menos.

Lo mismo con el Smart Tagging. Sus 59 saques marcan reinicios, que es justo donde el modelo
más se equivoca hoy.

Así que cuando comparemos, la pregunta útil igual no es *cuál de las tres es mejor* sino
*qué le da cada una a las otras*.

## Cosas que necesito que alguien mire con ojo crítico

Tres decisiones mías de hoy. Tienen criterio detrás, pero ninguna es una verdad:

1. Que la tasa de balón parado del dataset alemán (44,9%) sea comparable a lo que el reto
   llama `D`.
2. ~~Alinear el convenio apuntando a 14,2 s de duración media~~ — retirado: ese número salía
   de la ventana pública. La regla de fusión sale ahora de la definición de *posesión
   controlada* del reto (un contacto del rival de menos de unos segundos, sin control, no
   cambia la posesión), aplicada a los partidos externos. La duración media resultante es
   una medición, no un objetivo.
3. Que el pipeline esté usando **solo el ATD táctico**. La homografía, el Smart Tagging y el
   ATD del broadcast tienen el código de lectura escrito pero **no están conectados**.
   ¿Decisión o se quedó a medias?
