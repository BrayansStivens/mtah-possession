"""Smart Tagging convertido a features por frame.

Es la informacion gratis mas grande del bundle y la mas facil de usar mal. Lo que es:
638 fases tacticas detectadas automaticamente, 602 con tag de equipo, y 59 saques, cada
uno de los cuales marca un **reinicio**, es decir el final de una racha de balon parado.

Lo que NO es, y DATA.md lo dice explicitamente:

- **No es ground truth.** Metrica lo genera del mismo video y tracking que tenemos, sin
  revision manual. Tiene falsos positivos y negativos.
- **`Team` no siempre es un equipo.** 36 instancias van con `N/A`, y entre ellas estan
  *los 26 saques de banda*, justo donde saber el lado valdria mas.
- **Los intervalos son fases, no posesiones.** Un Build Up puede cubrir una posesion,
  quedarse corto o abarcar dos. Copiar un intervalo como etiqueta importa sus errores.

De ahi que todo lo de aqui entre al modelo como *evidencia con peso medido*, nunca como
respuesta. Los tags de saque, por ejemplo, cubren una ventana de unos 5 s alrededor del
reinicio y no el instante: verificado contra los dos saques iniciales, cuyos tags
(11097-11222 y 113660-113785) contienen los frames reales 11185 y 113742.

**El `Team` de una fase no siempre es el equipo con el balon.** Depende del codigo, y
leerlo mal invierte la evidencia. Medido el 07-11 sobre BAR-ATH, fraccion de frames vivos
de cada fase en que la entrega actual predice el equipo tagueado (sin etiquetas: es
nuestro modelo contra el tag, solo para leer la semantica del codigo):

    BUILD UP .84   PROGRESSION .84   CREATING CHANCES .74   LONG BALL .66
    DEFENDING IN ATTACKING THIRD .16   ... MIDDLE THIRD .16   ... DEFENSIVE THIRD .26

Los `DEFENDING IN *` nombran al equipo SIN balon. Y son espejo de los otros: cubren los
mismos intervalos que BUILD UP / PROGRESSION / CREATING CHANCES con el equipo contrario
(mismos recuentos de frames, al frame). Los demas codigos con equipo no entran: BALL IN
FINAL THIRD / BALL IN THE BOX / PLAYERS IN THE BOX nombran al atacante pero acuerdan solo
.65-.72, y ATTACKING / DEFENSIVE TRANSITION acuerdan .55 / .45 -- su intervalo cabalga el
cambio de posesion, asi que no dicen de quien es el balon (si quiza donde cambia).
La lectura por codigo viene de DeteccionPosesionBalon/tactical_tagging.py.
"""

import numpy as np

from ..constants import N_FRAMES

# Codigos cuyo `Team` es el equipo con el balon.
CODES_NAMING_TEAM_IN_POSSESSION = frozenset({
    "build up", "progression", "creating chances", "long ball"})
# Codigos cuyo `Team` es el equipo SIN balon: el voto va al rival.
CODES_NAMING_TEAM_OUT_OF_POSSESSION = frozenset({
    "defending in attacking third", "defending in middle third",
    "defending in defensive third"})


def possession_team(phase):
    """'H', 'A' o None: el equipo con el balon segun la fase, leyendo su codigo.

    None si la fase no tiene equipo o si su codigo no dice quien tiene el balon.
    """
    tagged = phase.team_label
    if tagged is None:
        return None
    code = phase.code.strip().lower()
    if code in CODES_NAMING_TEAM_IN_POSSESSION:
        return tagged
    if code in CODES_NAMING_TEAM_OUT_OF_POSSESSION:
        return "A" if tagged == "H" else "H"
    return None


def team_votes(phases, n_frames=N_FRAMES):
    """(home_votes, away_votes), cada uno (n,) int16: fases que dan el balon a cada equipo.

    Como los DEFENDING son espejo de las fases de posesion, un frame cubierto por un par
    espejo cuenta dos votos para el mismo equipo. Es la misma evidencia dos veces: los
    recuentos dicen cuantas fases hablan, no cuanta informacion independiente hay.
    """
    home_votes = np.zeros(n_frames, dtype=np.int16)
    away_votes = np.zeros(n_frames, dtype=np.int16)
    for phase in phases:
        team = possession_team(phase)
        if team is None:
            continue
        start = max(0, phase.start_frame)
        stop = min(n_frames, phase.end_frame)
        if stop > start:
            (home_votes if team == "H" else away_votes)[start:stop] += 1
    return home_votes, away_votes


def team_logit(phases, n_frames=N_FRAMES):
    """(n,) votos al local menos votos al visitante, leyendo cada codigo (ver arriba).

    Se suma en lugar de tomar la ultima porque las fases se solapan. Que coincidan fases
    de equipos distintos es una contradiccion y se cancela.
    """
    home_votes, away_votes = team_votes(phases, n_frames)
    return (home_votes - away_votes).astype(np.float32)


def weak_labels(phases, n_frames=N_FRAMES):
    """(n,) 'H', 'A' o '' por frame, y (n,) bool de conflicto.

    '' donde ninguna fase dice de quien es el balon y tambien donde se contradicen (el
    conflicto se marca aparte). No es ground truth: son fases de ~18 s generadas por
    Metrica del mismo video. Solo valen como diagnostico del modelo en este partido.
    """
    home_votes, away_votes = team_votes(phases, n_frames)
    conflict = (home_votes > 0) & (away_votes > 0)
    labels = np.full(n_frames, "", dtype="<U1")
    labels[(home_votes > 0) & ~conflict] = "H"
    labels[(away_votes > 0) & ~conflict] = "A"
    return labels, conflict


def set_piece_window(phases, n_frames=N_FRAMES, pad=0):
    """(n,) True dentro de un intervalo de saque, opcionalmente ensanchado `pad` frames.

    Donde esto es True hay un reinicio cerca, asi que hay una transicion de balon parado a
    balon vivo cerca. Es una pista sobre *donde* ocurre una transicion, no sobre que
    etiqueta poner.
    """
    out = np.zeros(n_frames, dtype=bool)
    for phase in phases:
        if not phase.is_set_piece:
            continue
        out[max(0, phase.start_frame - pad):min(n_frames, phase.end_frame + pad)] = True
    return out


def restart_team_logit(phases, n_frames=N_FRAMES):
    """(n,) equipo que reinicia, solo dentro de ventanas de saque con tag de equipo.

    Separado de team_logit porque en un reinicio el tag es mucho mas fiable: el equipo que
    saca un saque de puerta o un corner tiene el balon por definicion. Los saques de banda
    quedan fuera porque van todos con `N/A`.
    """
    out = np.zeros(n_frames, dtype=np.float32)
    for phase in phases:
        if not phase.is_set_piece:
            continue
        label = phase.team_label
        if label is None:
            continue
        start = max(0, phase.start_frame)
        stop = min(n_frames, phase.end_frame)
        out[start:stop] += 1.0 if label == "H" else -1.0
    return out


def build(phases, n_frames=N_FRAMES, set_piece_pad=0):
    return {
        "tag_team": team_logit(phases, n_frames),
        "tag_set_piece": set_piece_window(phases, n_frames, set_piece_pad),
        "tag_restart_team": restart_team_logit(phases, n_frames),
    }
