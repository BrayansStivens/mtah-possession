# Posesiones sin balón: qué se midió el 11 y 12 de noviembre

Carpeta de trabajo, fuera de `src/`. La entrega no cambia: `out/submission.csv`
sigue siendo `e793835071b54afb`. **Actualización del 12-11:** la idea 1 de §8 se integró en la
rama `equipo-sin-balon` (`features/team_signals.py`, `models/team_classifier.py`,
`training/team_classifier.py`, 17 tests). La medición integrada reproduce la fila de (a)
reducido de §4 al decimal: +16,01, +17,13 y +15,18 pp. Detalle en `ESTADO.md`, §2. Todo lo de abajo son medidas en los 7 partidos del IDSSE
degradados, fuera de muestra por partido. La ventana pública del reto (5 min, 17 posesiones)
solo se lee: una posesión mueve unos 6 puntos y no decide nada.

## 1. Lo que ya había se reproduce

| Comprobación | Resultado |
|---|---|
| Tests | 103 pasan |
| `scripts/reproduce.sh` | regenera la entrega byte a byte |
| Lectura pública de la entrega | spellF1 0,4148, macroF1 0,8003, 10 posesiones predichas de 17 |
| LOMO anidado del detector de Simon | +1,08, +0,87 y +1,56 pp por semilla, media +1,17 pp |

## 2. Dónde se pierde la nota (`diag.py`, semilla 0, huecos al azar)

| Tramo | Frames | Acierto por frame |
|---|---|---|
| Balón visto | 59,1 % | 87,5 % |
| Balón no visto | 40,9 % | 74,5 % |

| Posesiones reales recuperadas a IoU 0,5 | |
|---|---|
| Con 75 a 100 % de frames sin balón | 15,0 % |
| Con 25 a 50 % | 55,2 % |
| De menos de 3 s | 5,8 % |
| De más de 25 s | 69,0 % |

| Cambios reales encontrados a 1 s | Con balón | Sin balón |
|---|---|---|
| A muerto | 82,0 % | 36,9 % |
| Reinicio | 69,1 % | 28,4 % |
| Robo | 18,5 % | 4,1 % |

La entrega predice 179 posesiones de media contra 278 reales. El fallo dominante sin balón es
el equipo: 13,3 % de esos frames tienen el equipo cambiado, contra 8,2 % de muerto inventado
y 4,1 % de muerto perdido.

## 3. La forma de los equipos dice quién tiene el balón (`feats.py`, `modelos.py`, `auc.py`)

AUC fuera de muestra, semilla 0.

| Señal | AUC |
|---|---|
| Equipo, frames sin balón a 2 s: logit a mano de la entrega | 0,6996 |
| Equipo, mismos frames: clasificador (a), solo jugadores | 0,9536 |
| Muerto, frames sin balón: logit a mano | 0,9359 |
| Muerto, mismos frames: clasificador (b) | 0,9735 |

(a) usa 9 señales por equipo y sentido de ataque (centroide, largo, ancho, líneas de defensa
y ataque, marea, velocidad, jugadores en campo rival, sprints) en ventanas de 0,5, 1,5 y 4 s.
La diferencia de anchura sola llega a 0,90.

## 4. Dentro del decodificador, huecos al azar, tres semillas (`exp.py`, `semillas.py`)

Modelos entrenados con la semilla 0 de los otros 6 partidos. Pesos fijados por la selección
anidada de `exp.py`.

| Variante | Semilla 0 | Semilla 1 | Semilla 2 | Media | Partidos mejor |
|---|---|---|---|---|---|
| Entrega | 0,4111 | 0,4042 | 0,4290 | | |
| (a) todas las señales, peso 2,0 | +16,00 | +17,29 | +15,29 | +16,19 pp | 21 de 21 |
| (a) sin `opp`, `cen`, `len`, peso 2,0 | +16,01 | +17,13 | +15,18 | +16,11 pp | 21 de 21 |
| (b) muerto aprendido | +4,33 | +3,70 | +2,44 | +3,49 pp | 20 de 21 |
| (a) reducido + (b) | +20,86 | +21,74 | +19,91 | +20,84 pp | 21 de 21 |

Con (a) reducido + (b) el spellF1 medio pasa de 0,41 a 0,62 y el macroF1 de 0,79 a 0,89.

## 5. Huecos como los del partido real (`perfil.py`, `realista.py`, semilla 0)

El ATD pierde el balón sobre todo cuando los jugadores están parados: 93,9 % de frames sin
balón en el tramo de menos actividad y 9,8 % en el de más. El proxy lo escondía al azar,
cerca del 41 % en todos los tramos. `realista.py` recoloca los huecos según la actividad sin
tocar a los jugadores: 84,8 % y 19,4 % en esos mismos tramos, una versión suave de la curva
real. Con eso, el 68,3 % de los frames sin balón son de juego parado, contra 44,2 % al azar.

| Variante, pesos fijados de antemano | spellF1 | Diferencia | Partidos mejor |
|---|---|---|---|
| Entrega | 0,4511 | | |
| Entrega sin detector de fronteras | 0,4372 | −1,38 pp | 1 de 7 |
| Detector reentrenado con huecos realistas | 0,4514 | +0,03 pp | 4 de 7 |
| (a) todas, peso 2,0 | 0,5780 | +12,70 pp | 7 de 7 |
| (a) sin `opp`, `cen`, `len` | 0,5766 | +12,55 pp | 7 de 7 |
| (b) muerto aprendido | 0,4898 | +3,87 pp | 7 de 7 |
| (a) + (b) | 0,6273 | +17,62 pp | 7 de 7 |
| (a) + (b) reentrenados con huecos realistas | 0,6321 | +18,10 pp | 7 de 7 |

Mover `ball_unseen_logit` en cualquier sentido empeora: de −0,25 a −2,84 pp. La entrega ya
marca muerto en el 73,5 % de los frames sin balón y lo real es 68,3 %.

## 6. El partido real no se parece del todo al IDSSE (`deriva.py`, `deriva_senal.py`, `robusto.py`)

Sin etiquetas: un clasificador intenta distinguir el ATD del IDSSE degradado con las señales
de equipo. 0,5 es indistinguible. La referencia es un partido del IDSSE contra los demás.

| Señales de (a) | Columnas | AUC equipo IDSSE | Dominio, balón visto | Dominio, sin balón | Ganancia azar | Ganancia realista |
|---|---|---|---|---|---|---|
| Referencia entre partidos IDSSE | | | 0,53 | 0,59 | | |
| Las 9 | 87 | 0,9518 | 0,933 | 0,873 | +16,00 | +12,70 |
| Sin `opp` | 78 | 0,9512 | 0,923 | 0,843 | +16,51 | +13,04 |
| Sin `opp`, `cen`, `len` | 57 | 0,9506 | 0,724 | 0,659 | +16,01 | +12,55 |
| Anchura y líneas | 27 | 0,9180 | 0,701 | 0,659 | +7,98 | +5,35 |
| Solo anchura | 9 | 0,9059 | 0,602 | 0,516 | +5,18 | +2,67 |

La deriva está en `opp` (0,91 por sí sola), `cen` y `len` (0,70). Quitarlas no cuesta nada en
validación y deja el clasificador mucho más cerca de la distribución del partido real. El
subconjunto se eligió mirando la distribución sin etiquetas del partido del reto, no su nota.

Otra diferencia que el proxy no reproduce: en el ATD faltan jugadores justo cuando falta el
balón. 19,2 jugadores seguidos de media sin balón contra 21,4 con balón, y 14,3 % de frames
con menos de 16 contra 1,0 %. En el IDSSE degradado son 20,4 y 6,4 % en los dos casos.

## 7. Lectura en el partido real (`real.py`, `robusto.py`), no es criterio

| Candidato | spellF1 público | Posesiones de 17 | Posesiones en el partido | Frames iguales a la entrega |
|---|---|---|---|---|
| Entrega | 0,4148 | 10 | 202 | 100 % |
| (a) todas | 0,6143 | 11 | 185 | 89,7 % |
| (a) sin `opp`, `cen`, `len` | 0,6286 | 11 | 190 | 89,9 % |
| (a) anchura y líneas | 0,3333 | 7 | 190 | 90,3 % |
| (a) solo anchura | 0,3333 | 7 | 171 | 91,1 % |
| (b) muerto aprendido | 0,4000 | | | |

Los dos clasificadores débiles suben en IDSSE y bajan en la lectura pública: confirma que
17 posesiones no ordenan candidatos.

## 8. Ideas, por orden de valor esperado

1. **Adoptar (a) reducido, peso 2,0.** Es lo único con ganancia grande y estable: +16 pp en
   tres semillas, +12,6 pp con huecos realistas, 28 de 28 comparaciones a favor. Pide un
   artefacto nuevo entrenado con IDSSE, su test de reproducción y el crédito de licencia.
2. **Reajustar los costes de cambio con (a) puesto.** 40, 34 y 14 se ajustaron cuando el
   equipo sin balón era casi ruido. Solo se encuentra el 18,5 % de los robos con balón y con
   (a) se predicen 206 posesiones de 278. Experimento anidado, todavía sin correr.
3. **Quitar jugadores cuando falta el balón al degradar.** Cerraría la diferencia del punto 6
   y diría si (a) aguanta con 16 a 19 jugadores. Sin correr.
4. **Normalizar las señales por partido.** La deriva residual de (a) reducido es 0,72 y 0,66
   contra 0,53 y 0,59. Centrar cada señal con la media del propio partido no usa etiquetas.
   Sin correr.
5. **(b) muerto aprendido.** +3,5 pp solo y unos +4,7 pp encima de (a) en IDSSE, pero no
   mejora la lectura real y depende del contexto de balón, que es donde más difiere el proxy.
   Segundo paso, no primero.
6. **Saques del Smart Tagging como anclas de muerto** en tramos sin balón. Es la tarea de
   Simon y encaja con el punto 2.

Descartado con medida: mover `ball_unseen_logit`, reentrenar el detector con huecos
realistas, un clasificador solo de anchura.

## Pendiente de comprobar

- Huecos realistas solo en la semilla 0, y la curva es más suave que la real.
- (a) contra las fases del Smart Tagging en frames sin balón da AUC 0,61 a 0,70. Es un
  diagnóstico débil, las fases duran unos 18 s.
- Integrado en `src/` solo (a) reducido, en la rama `equipo-sin-balon`. El resto no.
