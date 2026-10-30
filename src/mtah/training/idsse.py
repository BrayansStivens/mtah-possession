"""Dataset IDSSE: siete partidos de Bundesliga con posesion etiquetada por frame.

Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer". Sci Data 12, 195 (2025).  doi:10.1038/s41597-025-04505-y
Datos en figshare doi:10.6084/m9.figshare.28196177, licencia **CC-BY 4.0** con
autorizacion de la DFL. Si esto acaba en la entrega, hay que citar el paper y acreditar a
la DFL -- es una condicion de la licencia, no una cortesia.

**Por que este dataset y no otro.** Trae `BallPossession` (1 local, 2 visitante) y
`BallStatus` (1 vivo, 0 parado) en cada frame del track del balon, a 25 Hz, que es
exactamente la etiqueta H/A/D del reto y exactamente la frecuencia del video tactico. Son
~1,0 M de frames etiquetados contra los 7.500 del ground truth publico.

Tres trampas, las tres verificadas midiendo y no leyendo documentacion:

1. **La velocidad viene en km/h, no en m/s.** kloppy pasa el campo `S` del DFL sin
   convertir. Medido sobre J03WPY: el p99 del fichero es 22.88 y el p99 de la velocidad
   derivada de la posicion es 6.37, ratio **3.590** -- o sea 3.6 clavado. Sin dividir, la
   actividad media sale 6.53 en vez de 1.81 y `activity_center_ms = 1.75` da todo por vivo.
   Es el fallo silencioso mas caro que tiene este import.

2. **Los slots no son los del reto.** Aqui hay un slot por jugador que pisa el campo
   (~15-16 por equipo contando suplentes) y la identidad es correcta. El ATD tiene 46 slots
   sin identidad que se fragmentan y se intercambian. Convertir esto en algo parecido al ATD
   es trabajo del modulo de degradacion, no de aqui: este modulo entrega el dato limpio.

3. **El campo mide 105 x 68 y el del reto 105.3 x 68.** Las coordenadas salen normalizadas
   0-1 en los dos casos, asi que pasarlas por `features.geometry` las escala con el
   `PITCH` del reto y cada distancia en x queda un 0,3% larga. Por debajo del ruido, pero
   conviene saberlo antes de comparar distancias absolutas entre los dos conjuntos.

El balon tiene coordenadas en **todos** los frames, tambien en los que estan parados --
verificado: 0 frames sin posicion de balon en J03WPY. Eso es lo que descarta a Metrica
sample-data como fuente principal, donde el balon solo existe mientras el juego esta vivo.
"""

import os

import numpy as np

from ..io.atd import HOME_CODE, AWAY_CODE, Tracking
from ..paths import cache_dir

# Los siete partidos publicados. Los ids son los que acepta kloppy.
MATCHES = {
    "J03WMX": "1. FC Koln vs FC Bayern Munchen",
    "J03WN1": "VfL Bochum 1848 vs Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Dusseldorf vs 1. FC Nurnberg",
    "J03WOH": "Fortuna Dusseldorf vs SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Dusseldorf vs FC St. Pauli",
    "J03WOY": "Fortuna Dusseldorf vs F.C. Hansa Rostock",
    "J03WR9": "Fortuna Dusseldorf vs 1. FC Kaiserslautern",
}

# Ver la trampa 1 del docstring. Medido, no supuesto.
KMH_TO_MS = 1.0 / 3.6

# Tolerancia del chequeo de unidades: un sprint de elite ronda los 10 m/s, asi que
# cualquier maximo por encima de esto significa que el fichero no esta en m/s.
MAX_PLAUSIBLE_SPEED_MS = 12.0


def _allocate_slots(frames, home_team):
    """(slot_de_jugador, team_of_slot). Un slot por jugador, los locales primero.

    Los locales van antes que los visitantes para que `team_of_slot` quede en bloques,
    igual que en el XML del ATD, y `slots_of()` devuelva indices contiguos.
    """
    home_ids, away_ids = [], []
    seen = set()
    for frame in frames:
        for player in frame.players_data:
            if player.player_id in seen:
                continue
            seen.add(player.player_id)
            (home_ids if player.team == home_team else away_ids).append(player.player_id)

    order = home_ids + away_ids
    slot_of = {pid: i for i, pid in enumerate(order)}
    team_of_slot = np.array([HOME_CODE] * len(home_ids) + [AWAY_CODE] * len(away_ids),
                            dtype=np.int8)
    return slot_of, team_of_slot


def _to_arrays(frames, home_team, slot_of, n_slots):
    """Recorre los frames una sola vez y rellena los arrays del Tracking + las etiquetas."""
    from kloppy.domain import BallState

    n = len(frames)
    players_xy = np.full((n, n_slots, 2), np.nan, dtype=np.float32)
    players_speed = np.full((n, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    labels = np.empty(n, dtype="<U1")

    for i, frame in enumerate(frames):
        for player, data in frame.players_data.items():
            j = slot_of[player.player_id]
            if data.coordinates is not None:
                players_xy[i, j] = (data.coordinates.x, data.coordinates.y)
            if data.speed is not None:
                players_speed[i, j] = data.speed * KMH_TO_MS

        ball = frame.ball_coordinates
        if ball is not None and ball.x is not None:
            ball_xy[i] = (ball.x, ball.y)
        if frame.ball_speed is not None:
            ball_speed[i] = frame.ball_speed * KMH_TO_MS

        # BallStatus=0 es parada arbitral, y manda sobre la posesion: el DFL sigue
        # diciendo quien tenia el balon cuando se paro el juego, pero para el reto ese
        # frame es D. Ese "quien lo tenia" es senal extra que aqui se descarta a proposito,
        # porque la etiqueta del reto no la contempla.
        if frame.ball_state == BallState.DEAD:
            labels[i] = "D"
        elif frame.ball_owning_team is None:
            labels[i] = "X"
        else:
            labels[i] = "H" if frame.ball_owning_team == home_team else "A"

    return players_xy, players_speed, ball_xy, ball_speed, labels


def _check_speed_units(players_speed):
    """Falla si la velocidad no parece estar en m/s despues de convertir.

    Existe porque la conversion depende de que kloppy siga entregando km/h. El dia que lo
    arreglen aguas arriba, esto salta en vez de envenenar silenciosamente la calibracion.
    """
    finite = players_speed[np.isfinite(players_speed)]
    if finite.size == 0:
        raise ValueError("ninguna velocidad de jugador es finita")
    top = float(np.percentile(finite, 99.9))
    if top > MAX_PLAUSIBLE_SPEED_MS:
        raise ValueError(
            "velocidad de jugador implausible tras convertir: p99.9 = {:.2f} m/s.\n"
            "Probablemente kloppy ya devuelve m/s y la division por 3.6 sobra. "
            "Comprueba KMH_TO_MS.".format(top))


def load(match_id="J03WPY", use_cache=True):
    """(Tracking, labels) de un partido del IDSSE.

    `labels` es un array de 'H'/'A'/'D' de la misma longitud que el tracking, con la misma
    convencion que `decode.viterbi.STATES`. La primera llamada descarga ~370 MB de figshare
    y tarda minuto y medio; despues sale de la cache en `out/cache`.
    """
    if match_id not in MATCHES:
        raise KeyError("partido desconocido {!r}; hay {}".format(
            match_id, sorted(MATCHES)))

    cache = os.path.join(cache_dir(), "idsse_{}.npz".format(match_id))
    if use_cache and os.path.exists(cache):
        blob = np.load(cache, allow_pickle=False)
        tracking = Tracking(
            blob["ball_xy"], blob["ball_speed"], blob["players_xy"],
            blob["players_speed"], blob["team_of_slot"],
            str(blob["home_name"]), str(blob["away_name"]))
        return tracking, blob["labels"]

    from kloppy import sportec

    # coordinates="metrica" devuelve 0-1 con origen arriba-izquierda sobre 105x68, que es
    # la convencion del reto. Verificado contra MetricaCoordinateSystem de kloppy.
    dataset = sportec.load_open_tracking_data(match_id=match_id, coordinates="metrica")
    home_team, away_team = dataset.metadata.teams

    slot_of, team_of_slot = _allocate_slots(dataset.frames, home_team)
    arrays = _to_arrays(dataset.frames, home_team, slot_of, len(team_of_slot))
    players_xy, players_speed, ball_xy, ball_speed, labels = arrays
    _check_speed_units(players_speed)

    tracking = Tracking(
        ball_xy=ball_xy, ball_speed=ball_speed,
        players_xy=players_xy, players_speed=players_speed,
        team_of_slot=team_of_slot,
        home_name=home_team.name, away_name=away_team.name)

    if use_cache:
        np.savez_compressed(
            cache, ball_xy=ball_xy, ball_speed=ball_speed, players_xy=players_xy,
            players_speed=players_speed, team_of_slot=team_of_slot, labels=labels,
            home_name=home_team.name, away_name=away_team.name)
    return tracking, labels
