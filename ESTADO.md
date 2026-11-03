# Estado del proyecto — 3 de noviembre de 2026

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
| **Dónde estamos, medido** | **0,3965** de media en 7 partidos de Bundesliga degradados para parecerse a nuestros datos (ajuste 0,3878 / validación 0,4081) |
| En los 5 minutos públicos | 0,4903 — **lectura, no criterio**: 17 posesiones, una fallada mueve 6 puntos |
| La entrega de hoy | `out/submission.csv`, generada el 31-10 con `config.py` tal cual está; pasa el validador oficial; 55 tests en verde |
| Qué configuración lleva | la de `config.py`; cada valor tiene escrito al lado de dónde sale. **Ninguno viene ya de los 5 minutos públicos** (prohibido desde el 22-10) |

Si todo se parase hoy, hay algo válido que entregar y está 15 puntos por encima del mínimo.

---

## 2. Qué está hecho

### El programa (`src/mtah/`)

- Lectura del tracking táctico (`io/atd.py`), homografía (`io/homography.py`) y Smart Tagging
  (`io/smart_tagging.py`). **Solo el tracking está conectado al pipeline**; los otros dos
  tienen el lector escrito y no se usan (ver §4, pregunta abierta).
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
3. **Un detector de balón propio sobre el vídeo** (la línea de visión). El 41 % de frames sin
   balón es el cuello de botella de la mitad "de quién es". Cada frame recuperado entra en el
   pipeline sin tocar el diseño. No se ha empezado en este repositorio.
4. **Conectar el Smart Tagging** como pista de reinicios (sus 59 saques marcan justo donde más
   se falla). El lector existe (`io/smart_tagging.py`); la conexión no.
5. **Mejorar la degradación** para que las rachas sin balón caigan donde caen de verdad
   (zonas abarrotadas, balones aéreos). Hoy están repartidas con un modelo más simple.
6. **Una emisión aprendida** sobre los partidos IDSSE degradados (fase C de
   `datos-externos.md`): regresión logística o gradient boosting sobre las features actuales,
   dos cabezas. **Solo si el arnés dice que hay margen**, y midiendo contra el mínimo 0,2433:
   si no supera al modelo geométrico, se entrega el geométrico.
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

Pregunta abierta que alguien debería contestar: el pipeline usa **solo el tracking táctico**.
La homografía, el Smart Tagging y el tracking del broadcast se leen pero no se usan. ¿Es una
decisión o se quedó a medias? (`analisis-y-propuesta.md`, parte 5, pregunta 3.)

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
- **Nada de la vía supervisada está medido**: el argumento a su favor es de margen, no de
  resultado.

---

## 7. Cómo revisarlo y reproducirlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                      # 55 tests
```

Para regenerar la entrega hacen falta los datos del partido (no están en el repo, no pueden
estar: 5 GB y no redistribuibles). Descargarlos del Drive de la organización y:

```bash
source scripts/env.sh                    # MTAH_DATA_DIR, por defecto ~/Documents/Hackaton/data
python -m mtah status
./scripts/reproduce.sh FC_Medallo        # -> out/FC_Medallo.csv
```

Para rehacer las mediciones del arnés hace falta además `pip install -r requirements-train.txt`
(kloppy, para descargar el IDSSE) y la caché de partidos degradados en `out/cache/`. Los
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
