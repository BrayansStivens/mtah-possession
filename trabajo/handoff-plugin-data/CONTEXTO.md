# Traspaso a una sesión con el plugin Data — FC Medallo

Este paquete existe para usar el plugin **Data** de claude.ai (`/analyze`, `/create-viz`,
`/build-dashboard`, `/validate`) sobre los resultados de nuestro modelo. El plugin está activado
en la cuenta de claude.ai pero no en la sesión de Claude Code donde se hizo el trabajo.

## ⚠️ Lo primero: qué se puede subir y qué no

Las reglas de la competición prohíben subir a cualquier LLM en la nube **el vídeo, el tracking,
la homografía, el Smart Tagging o cualquier extracto** del partido. Por eso este paquete
**no contiene nada de eso** y en la sesión nueva **no hay que subir nada de la carpeta `data/`**.

Lo que sí hay aquí, y es seguro:
- **Salidas de nuestro modelo** (etiquetas H/A/D que produce nuestro programa).
- **El ground truth público** de 5 minutos, que los organizadores publican en GitHub.
- **Estadísticas agregadas** medidas sobre **otros partidos** (7 de la Bundesliga, públicos).

## Qué hay en `datos/`

| fichero | qué es |
|---|---|
| `scores_por_partido.csv` | spellF1 y macroF1 por partido externo, para la config **actual** (la anterior al 28-10, reconstruida) y la **recalibrada** (el `config.py` de hoy). Columna `grupo`: 4 partidos de *ajuste*, 3 de *validación* que no se tocaron al ajustar. Regenerado el 2-11 con `regenera_datos.py` |
| `diagnostico_errores.csv` | cuatro tablas (formato largo) para cada config: error según cuánto lleva el balón sin verse, error según distancia a la transición real, matriz de confusión, % de posesiones recuperadas por duración |
| `recalibracion.json` | la config recalibrada del 28-10 (histórica: `activity_smooth` volvió a 50 el 31-10), las medianas por clase medidas, y los scores de ese día |
| `ventana_publica_5min.csv` | 7.500 filas: ground truth público y nuestra etiqueta, por fotograma |
| `partido_completo_1hz.csv` | nuestra etiqueta a 1 muestra por segundo, las dos partes (sin ground truth: es privado) |

Y en `figuras/` las tres gráficas ya hechas a mano con matplotlib, por si sirven de referencia.

## Los números que importan

- Métrica de ranking: **spellF1** (posesiones acertadas por solapamiento temporal, no fotogramas).
- Mínimo oficial a superar: **0,2433**.
- Nuestro modelo sobre los 7 partidos externos, con el `config.py` de hoy (31-10): **0,3965**
  (ajuste 0,3878, validación 0,4081). La config anterior al 28-10 daba 0,3767 (validación 0,3837).
- ¿Sobreajuste? La frase defendible, medida sobre las 35 particiones posibles de 4 ajuste / 3
  validación: **la recalibración gana +1,7 pp (IC 95 % [+1,4, +2,1]) en partidos que no se usaron
  para ajustarla; en los de ajuste gana +3,3 pp, así que aproximadamente la mitad de la ganancia
  es sobreajuste.** No vale decir "no se sobreajustó". Detalle en `VALIDACION-CRUZADA.md`.
- Lo que más falla: decir de qué equipo es el balón (~20% de los frames vivos) y las posesiones
  de menos de 5 segundos (se recupera ~1 de cada 10).

---

## Prompt para pegar en la sesión nueva

```
Contexto: competimos en "More than a Hack 2027". Hay que etiquetar cada fotograma de un partido
de fútbol con H (balón del equipo local), A (visitante) o D (balón parado). Nos puntúan con
spellF1: cada posesión (tramo seguido de un equipo con el balón) cuenta una vez y se acierta si
se solapa bastante con la real. El mínimo a superar es 0,2433.

Te adjunto resultados de nuestro modelo evaluado sobre 7 partidos externos (Bundesliga), con dos
configuraciones: "actual" y "recalibrado". Cuatro partidos se usaron para ajustar y tres para
validar. Los ficheros están descritos en CONTEXTO.md. Son salidas de nuestro modelo y agregados
sobre otros partidos: no contienen datos del partido de la competición.

Quiero tres cosas, en este orden:

1. /validate — revisa el análisis antes de que lo compartamos con el equipo. En particular:
   ¿la comparación actual vs recalibrado es justa? ¿hay algún sesgo en cómo se agregan los
   errores (por ejemplo, las categorías con pocos frames)? ¿la mejora de 0,3767 a 0,3965 es
   creíble dado que 3 partidos de validación dan 0,3837 → 0,4081?

2. /build-dashboard — un dashboard HTML para el equipo con: (a) spellF1 por partido, actual vs
   recalibrado, marcando ajuste vs validación; (b) las cuatro tablas de diagnóstico como gráficas,
   con un selector de config; (c) la línea de tiempo de los 5 minutos públicos (ground truth
   arriba, modelo abajo, errores marcados); (d) el partido completo a 1 Hz como franja de colores.
   Colores: H azul, A rojo, D gris.

3. /create-viz — una figura limpia "antes/después" para el documento técnico: spellF1 en los 7
   partidos con las dos configs, separando ajuste y validación, con la línea del mínimo 0,2433.

No inventes datos que no estén en los ficheros. Si algo no se puede responder con lo adjunto, dilo.
```
