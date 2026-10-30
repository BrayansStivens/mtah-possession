"""Features geometricos por frame, en metros y vectorizados sobre todo el partido.

Todo se calcula en metros, no en coordenadas normalizadas: el campo mide 105.3 x 68 m y
comparar distancias sin escalar los dos ejes estira cada distancia en el eje corto y
elige el jugador equivocado cerca de las bandas.

Dos familias, a proposito:

- **Dependientes del balon** (distancia al mas cercano, control, presion). Son las mas
  informativas y las menos fiables: el balon falta en el 43% de los frames y
  desproporcionadamente en los que importan -- areas abarrotadas, duelos aereos, entradas.
- **Independientes del balon** (actividad de los jugadores, forma de equipo, centroides).
  Son las que tienen que sostener el 41% de frames puntuados sin balon, y la razon por la
  que este modulo calcula forma de equipo aunque no haya balon que ver.
"""

import numpy as np

from ..constants import PITCH

HOME_CODE, AWAY_CODE = 0, 1
SCALE = np.asarray(PITCH, dtype=np.float32)

# Minimos por debajo de los cuales la forma de equipo no significa nada. Un equipo del que
# se ven menos de 3 jugadores no tiene "forma", y una dispersion en x por debajo de 1 m no
# ocurre en un partido real: es el tracker viendo a dos personas juntas.
MIN_TRACKED_FOR_SHAPE = 3
MIN_SPREAD_M = 1.0


def to_metres(xy):
    """Coordenadas normalizadas 0-1 -> metros, escalando cada eje."""
    return np.asarray(xy, dtype=np.float32) * SCALE


def ball_distances(tracking):
    """(n, slots) distancia en metros de cada slot al balon. NaN donde falte alguno."""
    ball = to_metres(tracking.ball_xy)[:, None, :]          # (n, 1, 2)
    players = to_metres(tracking.players_xy)                # (n, slots, 2)
    delta = players - ball
    return np.sqrt(np.einsum("nsc,nsc->ns", delta, delta))


def nearest_per_team(distances, team_of_slot):
    """(n, 2) distancia al jugador mas cercano de cada equipo. NaN si el equipo no tiene a nadie."""
    out = np.full((distances.shape[0], 2), np.nan, dtype=np.float32)
    for team in (HOME_CODE, AWAY_CODE):
        slots = np.flatnonzero(team_of_slot == team)
        subset = distances[:, slots]
        empty = np.isnan(subset).all(axis=1)
        with np.errstate(invalid="ignore"):
            nearest = np.nanmin(np.where(np.isnan(subset), np.inf, subset), axis=1)
        nearest[empty] = np.nan
        out[:, team] = nearest
    return out


def control_scores(distances, team_of_slot, tau=3.0):
    """(n, 2) influencia de cada equipo sobre el balon: suma de exp(-d / tau).

    Un kernel exponencial en lugar de "el mas cercano gana" porque el mas cercano es una
    decision de un solo jugador sobre un tracking con huecos y cambios de identidad: si
    ese slot se pierde un frame, la etiqueta salta de equipo. Sumar sobre todos los
    jugadores hace la senal colectiva, que es lo unico que este dataset soporta -- no hay
    identificadores de jugador, solo 46 slots que aparecen y desaparecen.

    `tau` en metros fija cuanto cuenta estar cerca. 3 m es un radio de control razonable.
    """
    weights = np.exp(-distances / float(tau))
    weights = np.where(np.isnan(distances), 0.0, weights)
    out = np.zeros((distances.shape[0], 2), dtype=np.float32)
    for team in (HOME_CODE, AWAY_CODE):
        out[:, team] = weights[:, np.flatnonzero(team_of_slot == team)].sum(axis=1)
    return out


def count_within(distances, team_of_slot, radius):
    """(n, 2) jugadores de cada equipo a menos de `radius` metros del balon."""
    close = np.where(np.isnan(distances), False, distances < float(radius))
    out = np.zeros((distances.shape[0], 2), dtype=np.int16)
    for team in (HOME_CODE, AWAY_CODE):
        out[:, team] = close[:, np.flatnonzero(team_of_slot == team)].sum(axis=1)
    return out


def player_activity(tracking):
    """(n,) velocidad media de los jugadores visibles, en m/s.

    El mejor candidato a detector de balon parado que hay en estos datos y no depende del
    balon: con el juego detenido la gente anda o esta parada, y con el balon vivo corre.
    Se usa la media y no la suma para que no dependa de cuantos slots ve el tracker en ese
    frame -- cuenta entre 20 y 23 y eso varia por si solo.
    """
    speed = tracking.players_speed
    with np.errstate(invalid="ignore"):
        return np.nanmean(speed, axis=1).astype(np.float32)


def team_shape(tracking):
    """Dict de features de forma de equipo, todos independientes del balon.

    Por que importan: en el 41% de frames puntuados sin balon son lo unico que queda. El
    equipo que defiende se comprime y retrocede; el que ataca se estira y sube. La
    asimetria entre las dos formas lleva informacion de quien tiene el balon incluso
    cuando el balon no se ve.
    """
    xy = to_metres(tracking.players_xy)
    out = {}
    for team, name in ((HOME_CODE, "home"), (AWAY_CODE, "away")):
        slots = np.flatnonzero(tracking.team_of_slot == team)
        subset = xy[:, slots, :]
        with np.errstate(invalid="ignore"):
            out[name + "_centroid_x"] = np.nanmean(subset[:, :, 0], axis=1).astype(np.float32)
            out[name + "_centroid_y"] = np.nanmean(subset[:, :, 1], axis=1).astype(np.float32)
            out[name + "_spread_x"] = np.nanstd(subset[:, :, 0], axis=1).astype(np.float32)
            out[name + "_spread_y"] = np.nanstd(subset[:, :, 1], axis=1).astype(np.float32)
            out[name + "_n_tracked"] = np.sum(~np.isnan(subset[:, :, 0]), axis=1).astype(np.int16)
    out["centroid_gap_x"] = out["home_centroid_x"] - out["away_centroid_x"]

    # NaN, no un numero, cuando el ratio no significa nada. Si el tracker ve menos de dos
    # jugadores de un equipo en ese frame, la desviacion tipica es 0 y el ratio colapsa a
    # 0 o a infinito, dando una opinion fortisima e infundada sobre quien tiene el balon.
    # Marcarlo indefinido hace que el sumando de forma se neutralice en lugar de mentir.
    degenerate = ((out["home_n_tracked"] < MIN_TRACKED_FOR_SHAPE)
                  | (out["away_n_tracked"] < MIN_TRACKED_FOR_SHAPE)
                  | (out["home_spread_x"] < MIN_SPREAD_M)
                  | (out["away_spread_x"] < MIN_SPREAD_M))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = out["home_spread_x"] / out["away_spread_x"]
    out["spread_ratio_x"] = np.where(degenerate, np.nan, ratio).astype(np.float32)
    out["shape_degenerate"] = degenerate
    return out


def ball_out_of_pitch(tracking, margin=0.01):
    """(n,) True donde el balon trackeado cae fuera del campo -> el balon esta muerto.

    Senal dura y barata, pero de cobertura limitada: cuando el balon sale, el tracker
    suele perderlo en lugar de seguirlo fuera, asi que esto marca menos frames de los que
    deberia. No sustituye a un detector, lo ancla.
    """
    xy = tracking.ball_xy
    outside = (xy[:, 0] < -margin) | (xy[:, 0] > 1 + margin) | \
              (xy[:, 1] < -margin) | (xy[:, 1] > 1 + margin)
    return np.where(np.isnan(xy[:, 0]), False, outside)


def ball_gap_lengths(tracking):
    """(n,) longitud de la racha de ausencia del balon a la que pertenece cada frame.

    Un hueco de 3 frames es una oclusion y se interpola; uno de 400 es que el balon lleva
    15 s sin verse y cualquier posicion inferida ahi es ficcion. Distinguirlos permite
    confiar en el relleno solo donde se puede.
    """
    seen = tracking.ball_seen
    out = np.zeros(len(seen), dtype=np.int32)
    start = None
    for i, visible in enumerate(seen):
        if not visible and start is None:
            start = i
        elif visible and start is not None:
            out[start:i] = i - start
            start = None
    if start is not None:
        out[start:] = len(seen) - start
    return out


def build(tracking, tau=3.0, radii=(2.0, 5.0)):
    """Todos los features en un dict de arrays de longitud n_frames."""
    distances = ball_distances(tracking)
    features = {
        "ball_seen": tracking.ball_seen,
        "ball_speed": tracking.ball_speed,
        "ball_x": tracking.ball_xy[:, 0],
        "ball_y": tracking.ball_xy[:, 1],
        "ball_out": ball_out_of_pitch(tracking),
        "ball_gap": ball_gap_lengths(tracking),
        "activity": player_activity(tracking),
    }
    nearest = nearest_per_team(distances, tracking.team_of_slot)
    control = control_scores(distances, tracking.team_of_slot, tau)
    features["nearest_home"], features["nearest_away"] = nearest[:, 0], nearest[:, 1]
    features["control_home"], features["control_away"] = control[:, 0], control[:, 1]
    for radius in radii:
        counts = count_within(distances, tracking.team_of_slot, radius)
        features["n_home_{:g}m".format(radius)] = counts[:, 0]
        features["n_away_{:g}m".format(radius)] = counts[:, 1]
    features.update(team_shape(tracking))
    return features
