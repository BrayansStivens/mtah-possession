# Posesión por frame — More than a Hack 2027, Fase 1

Determinar, para cada uno de los **199.319 frames** del vídeo táctico de
FC Barcelona – Athletic Club (LaLiga 2024-25, J2), qué equipo tiene el balón:
`H` (Barcelona), `A` (Athletic), `D` (balón parado), `X` (sin respuesta).

Se puntúa con `spellF1` sobre un ground truth privado de partido completo.
Suelo oficial a batir: **`spellF1` 0.2433** (`baseline_smooth50`).

> **Para el equipo:** el estado del proyecto —qué está hecho, qué falta, qué problemas
> vimos y qué no hemos podido medir— está en [`ESTADO.md`](ESTADO.md). El análisis, los
> documentos y las tablas de resultados están en [`trabajo/`](trabajo/README.md). Este
> README describe solo el código de la entrega.

---

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # produccion: numpy, pandas, scikit-learn (pinneados)
pip install -r requirements-dev.txt      # + pytest, matplotlib
```

Python 3.12 o superior (lo pide `numpy==2.5.4`). La versión de scikit-learn tiene que ser
exactamente la pinneada: el detector de fronteras es un modelo guardado y se niega a
cargar con otra.

## Los datos

Los assets del partido **no están en este repositorio y no pueden estarlo**: pesan ~5 GB y
no son redistribuibles según los términos de la competición. Descárgalos del Drive de la
organización y descomprime manteniendo la estructura que traen:

```
<raiz>/tactical/BAR-ATH_tactical.mp4
<raiz>/tactical/BAR-ATH_tactical_atd.xml
<raiz>/tactical/BAR-ATH_tactical_atd.txt
<raiz>/tactical/BAR-ATH_tactical_homography.csv
<raiz>/tactical/BAR-ATH_tactical_smart_tagging.xml
<raiz>/broadcast/...
```

Apunta `MTAH_DATA_DIR` a `<raiz>`:

```bash
source scripts/env.sh                    # usa ~/Documents/Hackaton/data por defecto
python -m mtah status                    # dice que hay y que falta
```

> **No descargues la carpeta entera de Drive como ZIP.** Drive la parte en trozos de ~2 GB
> y es fácil acabar con una parte suelta. Descarga los ficheros uno a uno.

## Reproducir la entrega

```bash
source scripts/env.sh
./scripts/reproduce.sh <nombre_equipo>   # -> out/<nombre_equipo>.csv
```

Equivalente, paso a paso:

```bash
python -m mtah build --out out/equipo.csv    # tracking -> features -> decodificacion -> csv
python -m mtah validate out/equipo.csv       # el contrato, antes de subir
python -m mtah evaluate out/equipo.csv       # scorer oficial, ventana publica de 5 min
```

El parseo del tracking se cachea en `out/cache/`. Bórralo si cambias el bundle, o usa
`--no-cache`.

## Tests

```bash
python -m pytest -q
```

Cubren el contrato del fichero, el decodificador, el pipeline extremo a extremo sobre
tracking sintético, y una regresión que comprueba que el scorer vendorizado reproduce los
números publicados de los cuatro baselines oficiales.

---

## Arquitectura

```
src/mtah/
  constants.py          frames, bloques puntuados, campo, etiquetas
  paths.py              resolucion de assets + MTAH_DATA_DIR
  config.py             TODOS los hiperparametros, con su justificacion al lado
  io/atd.py             tracking (46 slots + balon) -> numpy, cacheado
  io/homography.py      matriz 3x3 por frame; campo <-> pixel
  io/smart_tagging.py   fases tacticas y saques -> intervalos en frames
  features/geometry.py  distancias, control, actividad, forma de equipo
  features/boundaries.py que cambia entre justo antes y justo despues de cada frame
  models/emissions.py   features -> coste por frame y estado
  models/boundary_detector.{py,joblib,json}  score de frontera, entrenado en el IDSSE
  decode/viterbi.py     coste -> segmentos. El nucleo del reto
  pipeline.py           el encadenado completo
  submission.py         construir, escribir y validar el CSV
  evaluate.py           scorer oficial vendorizado
vendor/                 scorer oficial (MIT) + ground truth publico + baselines
```

La descomposición es `P(H) = P(vivo) · P(local | vivo)` y `P(D) = P(muerto)`, porque las
dos mitades se apoyan en señales distintas: **vivo/muerto** sale de la cinemática de los
jugadores y no necesita el balón, mientras que **la propiedad** sale de la geometría
balón-jugador y sí lo necesita. Con el balón ausente en el 41% de los frames puntuados,
mezclarlas en un solo clasificador tira a la basura la mitad que sigue funcionando.

Encima, un **detector de fronteras**: gradient boosting que mira qué cambia entre una
ventana corta antes y otra después de cada frame, y devuelve un score de "aquí cambia la
posesión". El decodificador encarece todos los cambios (×2) y abarata los de cada frame en
proporción al score (×(1 − 0,95·score)): corta sobre todo donde el detector ve frontera. Se
entrenó en los 7 partidos del IDSSE degradados como el ATD; medido dejando cada partido
fuera, con el peso elegido sin verlo, da **+1,17 pp de `spellF1`** sobre el modelo sin él
(ver `ESTADO.md`). Se regenera con `python -m mtah.training.boundaries export`, que necesita
`requirements-train.txt` y descarga el IDSSE; la entrega solo carga el modelo guardado.

## Tres decisiones y por qué

Medidas con el scorer oficial, degradando el ground truth público de forma controlada.
Los números están en los docstrings de los módulos correspondientes.

**1. La decodificación es offline y centrada.** +1 frame de retraso es gratis
(`spellF1` 1.0000); +25 frames (1 s) baja a 0.7765 y +50 frames (2 s) a 0.5882. Cualquier
filtro causal es inviable.

**2. Viterbi en lugar de voto por mayoría.** Partiendo de la misma etiqueta cruda que el
baseline oficial, sobre la ventana pública:

| | spellF1 | macroF1 | posesiones |
|---|---|---|---|
| voto mayoría, `smooth50` | 0.3256 | 0.6207 | 26 |
| voto mayoría, `smooth125` | 0.4242 | 0.5812 | 16 |
| **Viterbi** `switch_cost=40` | **0.4000** | **0.6686** | 16 |

Gana a `smooth50` en las dos métricas a la vez. El motivo es estructural: una ventana
controla el número de segmentos indirectamente y para no fragmentar una posesión larga
tiene que borrar las cortas; Viterbi penaliza las transiciones directamente y puede hacer
las dos cosas. En la ventana pública, **6 de las 17 posesiones duran menos de 4 s** y la
más corta 0,76 s, así que borrar las cortas es caro: perder las de menos de 2 s cuesta
0.17 de `spellF1` y las de menos de 5 s, 0.44.

**3. Nunca escribimos `X` en un frame puntuado.** El scorer rellena las abstenciones hacia
delante con la última etiqueta del mismo tiempo *antes* de emparejar posesiones, así que en
`spellF1` una `X` es literalmente "repite mi respuesta anterior" — pero `macroF1`, el
desempate, la cobra como fallo:

```
con 20% de X                spellF1 0.3610   macroF1 0.5438
mismo pero relleno a mano   spellF1 0.3610   macroF1 0.6192
```

Idéntico en la métrica de ranking, 0.075 peor en el desempate. Está estrictamente dominada.

## Lo que NO usamos, y por qué

- **`reprojection_error` de la homografía como detector de balón parado.** Medido en este
  partido: mediana 0,231 en juego vs 0,395 en el descanso, p90 en juego 0,308. Separa
  "cámara apuntando al campo" de "cámara en el público", pero no separa jugada de parada.
- **El vídeo broadcast y su tracking.** Está en su propia línea de tiempo, con cortes y
  repeticiones, sin tabla de correspondencia, y un tercio de sus frames no tienen
  homografía usable. Mapearlo de vuelta cuesta más de lo que aporta.

## Avisos sobre la ventana pública

Son 5 minutos y 17 posesiones. Sirve para comprobar que el pipeline es **correcto**, nunca
para elegir entre dos versiones de él: `baseline_smooth125` gana ahí y pierde en el
partido completo. Una posesión fallada mueve `spellF1` unos 6 puntos en esa ventana.
Diferencias por debajo de ~0.04 en `spellF1` y ~0.01 en `macroF1` son ruido.

## Licencia

Código propio bajo MIT. `vendor/` es copia literal del repositorio oficial de la
competición (MIT, ver `vendor/NOTICE.md`). Los datos del partido no están cubiertos por
ninguna de las dos y no se redistribuyen.

`src/mtah/models/boundary_detector.joblib` es una obra derivada del dataset IDSSE —
Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer". *Sci Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y — publicado
bajo **CC-BY 4.0** con autorización de la **Deutsche Fußball Liga (DFL)**. Datos:
doi:10.6084/m9.figshare.28196177. El repositorio no contiene ningún dato del IDSSE, solo el
modelo entrenado sobre él.
