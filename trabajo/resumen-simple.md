# FC Medallo — el reto explicado fácil

28 de octubre de 2026 · quedan **20 días** (entrega: 17 de noviembre)

---

## 1. ¿Qué nos piden?

Nos dan **un partido de fútbol grabado** (Barcelona – Athletic) y los datos de dónde está cada
jugador y el balón en cada momento. El vídeo tiene **199.319 fotogramas** (25 por segundo).

Para **cada fotograma** tenemos que decir una de estas cuatro cosas:

| | |
|---|---|
| `H` | el balón lo tiene el **Barcelona** |
| `A` | el balón lo tiene el **Athletic** |
| `D` | el balón está **parado** (falta, fuera, gol, lesión…) |
| `X` | no lo sé |

Eso es todo. Entregamos un archivo con 199.319 líneas, una por fotograma.

## 2. ¿Cómo nos puntúan?

**No cuentan los fotogramas acertados. Cuentan las posesiones acertadas.**

Una posesión es un tramo seguido en que un equipo tiene el balón. Un partido tiene unas 250–300.
Para que una posesión cuente como acertada, la que decimos nosotros tiene que **solaparse bastante**
con la real: empezar y acabar más o menos donde empieza y acaba de verdad.

Por qué esto importa tanto: si una posesión real dura 40 segundos y nosotros la partimos en tres
trozos, **acertamos el 97% de los fotogramas pero fallamos la posesión**. La métrica se llama
`spellF1`, va de 0 a 1, y **el mínimo que hay que superar es 0,24**.

No hay jurado ni presentación. **Solo el número.**

## 3. ¿Por qué es difícil?

Tres cosas:

1. **El balón "desaparece" el 41% del tiempo.** No es que no se vea en el vídeo: es que el
   programa automático que lo sigue lo pierde (en las áreas, en los balones aéreos, en las
   entradas). Y cuando lo pierde, lo pierde **durante 10 segundos de media**, a veces minutos.
2. **No sabemos quién es cada jugador.** Los datos dicen "hay alguien del Barcelona aquí", pero
   no quién. Y a veces confunden a dos jugadores.
3. **Solo nos dan las respuestas correctas de 5 minutos** del partido. El resto es secreto hasta
   el final.

## 4. ¿Qué NO se puede hacer? (esto descalifica)

Las reglas cambiaron el 22 de octubre y son más estrictas de lo que pensábamos:

- ❌ **Etiquetar a mano nada de este partido.** Ni para la entrega, ni para entrenar, ni para
  comprobar. Ni corregir a mano el seguimiento del balón. **Sobre otros partidos sí se puede.**
- ❌ **Usar los 5 minutos de respuestas para ajustar el programa.** Solo valen para comprobar que
  no hemos metido la pata (que el archivo está bien hecho, por ejemplo).
- ❌ **Usar datos de este partido hechos por personas**: estadísticas de LaLiga, Opta, etc.
- ❌ **Subir el vídeo o los datos a ChatGPT, Claude, Copilot o similares.** Usarlos para escribir
  código sí; pasarles trozos de los datos, no.
- ❌ Entregar un archivo que nuestro código no pueda volver a generar igual.

## 5. ¿Qué vamos a hacer?

Un programa que mira **dónde están los jugadores y el balón** y decide con dos preguntas:

**Pregunta 1: ¿está el juego parado?**
Se responde mirando **cómo se mueven los jugadores**. Con el juego parado, la gente anda. Con el
balón en juego, corre. **No hace falta ver el balón** para esto.

**Pregunta 2: si está en juego, ¿de quién es el balón?**
Se responde mirando **qué equipo tiene más jugadores cerca del balón** y cómo están colocados los
dos equipos (el que ataca se estira, el que defiende se encoge).

Y encima, un **"suavizador"**: en vez de decidir fotograma a fotograma (que da saltos), decide por
tramos, porque una posesión no cambia cada 4 centésimas de segundo.

### ¿Por qué esto y no una inteligencia artificial entrenada?

Porque para entrenarla hacen falta **miles de ejemplos con la respuesta correcta**, y de este
partido solo tenemos 5 minutos, y además **está prohibido usarlos para eso**. Hay datos de otros
partidos (7 de la Bundesliga, públicos y legales) y los estamos usando — pero para **ajustar y
comprobar** nuestro programa, no para entrenar una red desde cero, porque esos datos son
"limpios" (el balón siempre se ve) y los nuestros no.

Separar las dos preguntas es la decisión clave: **cuando el balón desaparece, la pregunta 1 sigue
funcionando** y el programa se calla en la 2 en vez de inventar. Y callarse dentro de una
posesión **no penaliza**.

## 6. ¿Dónde estamos?

El programa **ya funciona de punta a punta** y produce un archivo válido.

| | puntuación |
|---|---|
| Mínimo a superar | 0,24 |
| **Nosotros, en 7 partidos de práctica** | **0,40** |
| Nosotros, en los 5 minutos públicos | 0,49 (pero esos 5 minutos son "fáciles", no fiarse) |

¿Qué son los "partidos de práctica"? Cogimos 7 partidos de la Bundesliga que sí tienen las
respuestas, les **estropeamos los datos a propósito** para que se parezcan a los nuestros (le
quitamos el balón el 41% del tiempo, mezclamos jugadores…) y así podemos **medir de verdad** en
partidos completos, no en 5 minutos.

**Lo que hemos aprendido midiendo:**
- Detectar el balón parado **ya lo hacemos bien** (92% de acierto).
- **Lo que fallamos es decir de qué equipo es** (unas 2 de cada 10 veces) y, sobre todo,
  **las posesiones cortas**: de las que duran menos de 5 segundos, que son 4 de cada 10,
  solo pillamos 1 de cada 10.
- Sospechamos por qué: el "suavizador" mira tramos de 4 segundos, así que una posesión de 3
  segundos se le escapa. Y ese ajuste, junto con otros 15, **venía de los 5 minutos públicos**,
  cosa que ahora las reglas prohíben. **Ya están rehechos** (28 de octubre) usando solo los
  partidos de práctica: cambiaron 7 valores. Después (29-31 de octubre) comprobamos el
  procedimiento de ajuste de todas las formas posibles de repartir los 7 partidos entre
  "para ajustar" y "para comprobar": gana unos 1,7 puntos en partidos que no vio al ajustar
  y unos 3,3 en los que sí vio, o sea que **la mitad de la mejora es real y la otra mitad es
  aprenderse los partidos de ajuste**. Lo importante: **ya no queda nada de los 5 minutos en
  el programa.**

## 7. ¿Qué vamos a entregar?

Tres cosas, por un formulario, antes del 17 de noviembre:

1. **El archivo** de 199.319 líneas con las etiquetas. Es lo único que se puntúa.
2. **El código**, en un repositorio público, que genere ese archivo exacto.
3. **Un documento de pocas páginas** explicando qué hicimos, por qué, qué datos usamos, qué
   descartamos y dónde falla.

## 8. ¿Qué falta? (por orden)

1. ~~Rehacer los 16 ajustes que venían de los 5 minutos públicos~~ — **hecho el 28 de octubre.**
2. **Un detector de balón propio** sobre el vídeo, para recuperar parte del 41% que se pierde.
   Esto es la línea de visión por computador y **se enchufa directo** a nuestro programa:
   cuanto más balón recupere, mejor responde la pregunta 2. *(la parte de visión)*
3. **Mejorar los partidos de práctica** para que se parezcan aún más a los nuestros.
4. **Escribir el documento final** y dejar el repositorio listo. *(2 días, al final)*

## Cómo encaja el trabajo de cada uno

No competimos entre líneas: **se alimentan**. El detector de balón de visión da de comer al
programa de posesión. Y los partidos de práctica sirven para medir cualquier cosa que haga
cualquiera — es la única forma que tenemos de saber si algo mejora de verdad.
