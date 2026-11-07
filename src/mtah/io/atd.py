"""Lectura del Automatic Tracking Data (export FIFA/EPTS de Metrica).

Dos ficheros: un XML de metadatos (equipos y a que equipo pertenece cada track slot) y
un TXT crudo con una linea por frame:

    <frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>
                 jugadores (46 slots)       balon

Las coordenadas estan normalizadas 0-1 sobre el campo y la velocidad va en m/s. Las
observaciones ausentes son NaN, tanto en jugadores como en balon.

Dos cosas que es facil equivocar y caro detectar:

1. `frame_id` cuenta desde 1 y la submission desde 0  ->  video_frame = frame_id - 1.
2. El campo no es cuadrado (105.3 x 68 m). Comparar distancias sin escalar los dos ejes
   elige el jugador equivocado cerca de las bandas.

El parseo del TXT cuesta decenas de segundos, asi que se cachea en out/cache/*.npz.
Borra la cache si cambias el bundle.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import FPS, GAME, N_FRAMES, N_TRACK_SLOTS, PITCH
from ..paths import cache_dir, require

HOME_CODE, AWAY_CODE = 0, 1
_TRANSLATE = str.maketrans({":": ",", ";": ","})

# Banda aceptable para (velocidad del fichero / velocidad derivada de la posicion). En m/s
# sale ~1.0 (medido 0.992 sobre BAR-ATH); en km/h saldria ~3.6 y en km/h dividido por 3.6
# dos veces, ~0.28. La banda es ancha a proposito: separa unidades, no mide precision.
SPEED_RATIO_ACCEPTED = (0.8, 1.25)
# Por debajo de esta velocidad derivada el cociente lo domina el jitter del tracking.
SPEED_RATIO_MIN_DERIVED_MS = 0.5


class Tracking:
    """Tracking de un partido, alineado al frame de video (indice 0..N_FRAMES-1).

    Atributos (todos float32, NaN donde no hay observacion):
        ball_xy        (n, 2)      normalizado 0-1
        ball_speed     (n,)        m/s
        players_xy     (n, slots, 2)
        players_speed  (n, slots)
        team_of_slot   (slots,)    int8, 0 = local, 1 = visitante
        ball_seen      (n,)        bool, True donde el tracker ve el balon
    """

    __slots__ = ("ball_xy", "ball_speed", "players_xy", "players_speed",
                 "team_of_slot", "home_name", "away_name")

    def __init__(self, ball_xy, ball_speed, players_xy, players_speed,
                 team_of_slot, home_name, away_name):
        self.ball_xy = ball_xy
        self.ball_speed = ball_speed
        self.players_xy = players_xy
        self.players_speed = players_speed
        self.team_of_slot = team_of_slot
        self.home_name = home_name
        self.away_name = away_name

    @property
    def n_frames(self):
        return len(self.ball_xy)

    @property
    def ball_seen(self):
        return ~np.isnan(self.ball_xy[:, 0])

    def xy_metres(self, xy):
        """Convierte coordenadas normalizadas a metros, escalando cada eje."""
        return xy * np.asarray(PITCH, dtype=np.float32)

    def slots_of(self, team):
        """Indices de los slots de un equipo (0 local, 1 visitante)."""
        return np.flatnonzero(self.team_of_slot == team)

    def summary(self):
        seen = self.ball_seen
        n_tracked = np.sum(~np.isnan(self.players_xy[:, :, 0]), axis=1)
        return {
            "n_frames": int(self.n_frames),
            "slots": int(self.players_xy.shape[1]),
            "home": self.home_name,
            "away": self.away_name,
            "ball_seen_pct": float(100.0 * seen.mean()),
            "mean_tracked_objects": float(n_tracked.mean()),
            "max_tracked_objects": int(n_tracked.max()),
        }


def read_metadata(meta_path):
    """(team_of_slot, home_name, away_name) del XML de metadatos.

    El primer <Team> del fichero es el local: es la convencion FIFA/EPTS y la que sigue
    kloppy. Para este partido eso es Barcelona, que DATA.md confirma por separado.
    """
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: esperaba 2 equipos, hay {}".format(meta_path, len(teams)))

    home_id, home_name = teams[0]
    _, away_name = teams[1]
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError("{}: ningun <Player>".format(meta_path))

    team_of_slot = np.array(
        [HOME_CODE if tid == home_id else AWAY_CODE for tid in slots], dtype=np.int8)
    return team_of_slot, home_name, away_name


def speed_channel_ratio(tracking, min_derived_speed_ms=SPEED_RATIO_MIN_DERIVED_MS):
    """Mediana de (velocidad de jugador del fichero / velocidad derivada de la posicion).

    La derivada se calcula entre frames consecutivos en metros, asi que es NaN en cualquier
    hueco y no cruza ninguna ausencia. Los cambios de identidad entre slots producen saltos
    absurdos en la derivada, pero son pocos y la mediana no los ve.
    """
    xy_metres = tracking.players_xy * np.asarray(PITCH, dtype=np.float32)
    step = xy_metres[1:] - xy_metres[:-1]
    derived_speed = np.sqrt((step ** 2).sum(axis=-1)) * float(FPS)
    declared_speed = tracking.players_speed[1:]
    comparable = (np.isfinite(derived_speed) & np.isfinite(declared_speed)
                  & (derived_speed > min_derived_speed_ms))
    if not comparable.any():
        raise ValueError("ningun frame permite comparar la velocidad del fichero con la "
                         "derivada de la posicion")
    return float(np.median(declared_speed[comparable] / derived_speed[comparable]))


def check_speed_channel_units(tracking):
    """Falla si la velocidad de jugador del fichero no esta en m/s. Devuelve el cociente.

    `atd.xml` declara `<Unit>kph</Unit>` y es falso: medido contra la posicion el canal
    esta en m/s. El siguiente que lea el metadato lo "arreglara" dividiendo por 3.6, y como
    la actividad de los jugadores -- el detector de balon parado -- sale de este canal, el
    pipeline seguiria corriendo y prediciendo D en casi todo sin fallar en ningun sitio.
    Esto lo convierte en un error explicito. (Portado de DeteccionPosesionBalon.)
    """
    ratio = speed_channel_ratio(tracking)
    low, high = SPEED_RATIO_ACCEPTED
    if not low <= ratio <= high:
        raise ValueError(
            "la velocidad de jugador del fichero mide {:.3f} veces la derivada de la "
            "posicion; en m/s daria ~1.0 (km/h daria ~3.6). No conviertas unidades "
            "fiandote del metadato: vuelve a medirlas.".format(ratio))
    return ratio


def _parse_raw(raw_path, n_slots, n_frames):
    """Parsea el TXT crudo a una matriz (n_frames, 1 + 3*n_slots + 3).

    Linea a linea en lugar de cargando el fichero entero: son ~200 MB y hacer replace
    sobre todo el texto triplica la memoria sin ganar velocidad apreciable.
    """
    n_cols = 1 + 3 * n_slots + 3
    out = np.full((n_frames, n_cols), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            values = line.translate(_TRANSLATE).split(",")
            if len(values) != n_cols:
                raise ValueError(
                    "{}:{}: esperaba {} campos ({} slots), hay {}. "
                    "Comprueba que el fichero corresponde a este XML de metadatos."
                    .format(os.path.basename(raw_path), lineno, n_cols, n_slots, len(values)))
            row = np.array(values, dtype=np.float32)
            frame = int(row[0]) - 1          # frame_id cuenta desde 1
            if not 0 <= frame < n_frames:
                raise ValueError("{}:{}: frame_id {} fuera de rango"
                                 .format(os.path.basename(raw_path), lineno, int(row[0])))
            out[frame] = row
            seen[frame] = True

    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError("{}: faltan {} frames de {}".format(
            os.path.basename(raw_path), missing, n_frames))
    return out


def load(source="tactical", game=GAME, n_frames=None, use_cache=True):
    """Carga el tracking. Cachea el parseo en out/cache para las siguientes veces."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], source=source, game=game)
    team_of_slot, home_name, away_name = read_metadata(meta_path)
    n_slots = len(team_of_slot)
    if source == "tactical" and n_slots != N_TRACK_SLOTS:
        # No es un error: solo avisa de que el bundle no es el esperado.
        print("aviso: {} slots en el XML, esperaba {}".format(n_slots, N_TRACK_SLOTS))
    if n_frames is None:
        n_frames = N_FRAMES if source == "tactical" else 164_488

    cache = os.path.join(cache_dir(), "{}_{}_atd.npz".format(game, source))
    stamp = os.path.getmtime(raw_path)
    if use_cache and os.path.exists(cache):
        blob = np.load(cache)
        if float(blob["stamp"]) == stamp and int(blob["n_slots"]) == n_slots:
            tracking = Tracking(blob["ball_xy"], blob["ball_speed"], blob["players_xy"],
                                blob["players_speed"], team_of_slot, home_name, away_name)
            check_speed_channel_units(tracking)
            return tracking

    matrix = _parse_raw(raw_path, n_slots, n_frames)
    players = matrix[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = matrix[:, 1 + 3 * n_slots:]

    tracking = Tracking(
        ball_xy=np.ascontiguousarray(ball[:, :2]),
        ball_speed=np.ascontiguousarray(ball[:, 2]),
        players_xy=np.ascontiguousarray(players[:, :, :2]),
        players_speed=np.ascontiguousarray(players[:, :, 2]),
        team_of_slot=team_of_slot, home_name=home_name, away_name=away_name)
    check_speed_channel_units(tracking)

    if use_cache:
        np.savez_compressed(
            cache, ball_xy=tracking.ball_xy, ball_speed=tracking.ball_speed,
            players_xy=tracking.players_xy, players_speed=tracking.players_speed,
            stamp=np.float64(stamp), n_slots=np.int32(n_slots))
    return tracking
