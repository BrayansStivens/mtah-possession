# FC Medallo — More than a Hack 2027 · material de trabajo

Esta carpeta es el **trabajo**, no la entrega: documentos, figuras, scripts de medicion y
las tablas de resultados con las que se tomaron las decisiones. El codigo que se entrega
esta en la raiz de este mismo repositorio (`src/`, `tests/`, `vendor/`). El resumen de
donde esta todo, que falta y que problemas se vieron esta en `../ESTADO.md`.

(Esta carpeta se copio el 3-11 desde el repositorio de trabajo `mtah-2027-analisis`, que
queda como historico.)
Los `.pdf` son la version del 28-10 de los `.md` del mismo nombre; el `.md` es el vigente.

## Aviso sobre los datos

Los terminos de la competicion prohiben redistribuir los assets del partido:
video, tracking ATD, homografia y Smart Tagging. **Nada de eso esta aqui** —
`.gitignore` excluye `data/` y `staging-*/`, y no hay que des-ignorarlos.

Lo que si hay, y es seguro compartir:

- salidas de nuestro propio modelo (etiquetas H/A/D),
- el ground truth **publico** de 5 minutos, que publican los organizadores,
- estadisticas agregadas medidas sobre 7 partidos publicos de la Bundesliga.

## El reto, en tres lineas

Etiquetar cada frame del partido como **H** (posesion local), **A** (visitante)
o **D** (balon parado). La metrica es **spellF1**, que compara posesiones
completas por solape temporal, no frames sueltos. El minimo oficial a batir es
**0,2433**. Entrega: **17 de noviembre de 2026**.

## Donde esta cada cosa

| Ruta | Que es |
|---|---|
| `resumen-simple.md` | El reto explicado sin jerga. Empieza por aqui. |
| `analisis-y-propuesta.md` | Analisis del reto y propuesta tecnica. |
| `datos-externos.md` | Investigacion de datos externos para un modelo supervisado. |
| `handoff-plugin-data/VALIDACION.md` | Validacion del modelo actual, hallazgo por hallazgo. |
| `handoff-plugin-data/VALIDACION-CRUZADA.md` | Las 35 particiones. **Corrige a `VALIDACION.md`** en todo lo relativo al sobreajuste. |
| `handoff-plugin-data/datos/` | Tablas de resultados en JSON y CSV. |
| `handoff-plugin-data/*.py` | Los scripts que producen esas tablas. |
| `handoff-plugin-data/regenera_datos.py` | Rehace los ficheros que alimentan el dashboard del equipo con el `config.py` de hoy (comprueba antes que reproduce los del 28-10). Correr tras cada cambio de `config.py`. |
| `figuras/` | Graficas para los documentos. |

## Estado de las mediciones (31-10-2026)

El dashboard del equipo (artefacto privado de claude.ai, "Posesion FC Medallo")
se regenero el 2-11 con estos mismos numeros; sus ficheros son los de
`handoff-plugin-data/datos/`.

Sobre los 7 partidos externos, con `config.py` tal cual esta hoy (semilla 0 de
degradacion):

| | spellF1 |
|---|---|
| ajuste (4 partidos) | 0,3878 |
| validacion (3 partidos) | 0,4081 |
| los 7 | **0,3965** |

Los numeros de antes (0,3905 / 0,3957 / 0,3927) eran de antes del 31-10, cuando
`activity_smooth` valia 12. El 0,3957 ademas nunca lo producia el fichero: ver la
seccion correspondiente de `VALIDACION-CRUZADA.md`.

El procedimiento de recalibracion, corrido sobre las 35 particiones posibles de
4 ajuste / 3 validacion:

- **+1,71 pp** de media en los partidos que no se usaron para ajustarlo
  (IC 95 % [+1,37, +2,05], mejora en 32/35),
- **+3,27 pp** en los partidos de ajuste (35/35).

O sea: transfiere, pero aproximadamente **la mitad de la ganancia es
sobreajuste**. Esa es la frase defendible; no vale decir "no se sobreajusto".

## La rejilla ampliada: probada y descartada (31-10)

Siete parametros se clavaban en un borde de su lista de valores, asi que se
amplio la rejilla (64 -> 82 valores) y se repitieron las 35 particiones con el
mismo descenso. Comparando por pares, split a split:

| | ajuste (4) | validacion (3) |
|---|---|---|
| rejilla original | +3,27 pp | +1,71 pp |
| ampliada | +3,89 pp | +1,58 pp |
| diferencia pareada | **+0,62 pp** [+0,34, +0,90] | **-0,13 pp** [-0,52, +0,26] |

Ajusta mejor los partidos que ve y **no** mejora los que no ve: gana en 14 de 35
particiones, menos de la mitad. La transferencia baja del 52 % al 41 %. Asi que
**no se amplia la rejilla** en `recalibrate.py`, y no se cambian los valores a lo
que elige el brazo ampliado. Detalle en `VALIDACION-CRUZADA.md`.

Ya ningun parametro se clava en un borde, luego el diagnostico era correcto: el
optimo sobre el ajuste estaba fuera de la rejilla vieja. Lo que no se sostiene es
que llegar a el sirva de algo.

## Lo siguiente

Con 7 partidos de ajuste no hay senal para distinguir buena parte de los
hiperparametros, y eso no lo arregla buscar mas: lo arreglan mas partidos. La via
que queda abierta es el modelo supervisado de `datos-externos.md`, no seguir
afinando este.

## Reproducir las mediciones

Los scripts de `handoff-plugin-data/` importan `mtah` desde el repo de entrega y
leen la cache de `mtah-possession/out/`, que no esta versionada. Hacen falta las
dos cosas en local:

```sh
git clone https://github.com/BrayansStivens/mtah-possession.git
```
