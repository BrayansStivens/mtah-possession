"""Degradar tracking limpio hasta que se parezca al ATD automatico del reto.

**Para que existe.** Dos usos, y el segundo importa mas que el primero:

1. Entrenar. Un modelo ajustado sobre tracking corregido aprende a apoyarse en senales que
   en inferencia no existen -- balon siempre visible, identidad estable, 22 objetos.
2. **Medir.** El ground truth publico son 5 minutos y 17 posesiones, y EVALUATION.md dice
   que no sirve para elegir entre dos versiones del modelo. Degradando un partido del
   IDSSE, que si tiene etiqueta por frame de partido completo, se obtiene un spellF1 sobre
   un partido entero. Siete veces. Eso convierte "no podemos medir" en "podemos medir en
   siete partidos proxy", y ahora mismo ese es el riesgo mas grande que tiene el proyecto.

**El principio.** Ningun parametro de aqui es inventado: todos salen de `estimate()`, que
los mide sobre el ATD real del partido **sin usar ninguna etiqueta**. Eso es legitimo --
no toca el ground truth privado, solo la distribucion marginal de los datos de entrada,
que estan publicados -- y es lo que separa esto de un domain randomization a ojo.

**Se aplica en carga, con semilla distinta cada vez.** Un corpus degradado y congelado
tiene un patron de oclusion fijo que un modelo aprende a memorizar.

Lo que esta version NO hace, y conviene tener presente al leer un numero que salga de aqui:

- **Las rachas sin balon se colocan al azar.** En el ATD real se concentran en areas
  abarrotadas y duelos aereos (DATA.md), o sea en los frames dificiles. Al repartirlas
  uniformemente, el proxy queda **optimista**: pierde el balon en momentos mas faciles de
  lo que lo pierde el tracker de verdad.
- **No modela intercambios de identidad entre slots**, solo fragmentacion. Un slot que
  cambia de persona a mitad de un tramo es un error que aqui no ocurre.
- El ruido de posicion es AR(1) homogeneo, sin depender de la distancia a camara.
"""

import numpy as np

from ..io.atd import HOME_CODE, AWAY_CODE, Tracking

# Reparto de slots del ATD tactico, de DATA.md: 46 = 24 local + 22 visitante.
N_SLOTS_HOME, N_SLOTS_AWAY = 24, 22


class Spec:
    """Parametros de degradacion medidos sobre un tracking real. Todo son muestras
    empiricas, no distribuciones ajustadas: remuestrear de la realidad no exige elegir
    una familia parametrica ni acertar con ella."""

    __slots__ = ("ball_absent_runs", "ball_present_runs", "slot_runs",
                 "n_tracked_values", "n_tracked_probs", "step_sigma_m")

    def __init__(self, ball_absent_runs, ball_present_runs, slot_runs,
                 n_tracked_values, n_tracked_probs, step_sigma_m):
        self.ball_absent_runs = ball_absent_runs
        self.ball_present_runs = ball_present_runs
        self.slot_runs = slot_runs
        self.n_tracked_values = n_tracked_values
        self.n_tracked_probs = n_tracked_probs
        self.step_sigma_m = step_sigma_m

    def summary(self):
        a, p = self.ball_absent_runs, self.ball_present_runs
        return {
            "rachas_sin_balon": len(a),
            "media_sin_balon_fr": float(a.mean()) if len(a) else 0.0,
            "max_sin_balon_fr": int(a.max()) if len(a) else 0,
            "fraccion_sin_balon": float(a.sum() / (a.sum() + p.sum())) if len(a) else 0.0,
            "tramos_de_slot": len(self.slot_runs),
            "media_tramo_fr": float(self.slot_runs.mean()) if len(self.slot_runs) else 0.0,
            "objetos_media": float((self.n_tracked_values * self.n_tracked_probs).sum()),
            "step_sigma_m": float(self.step_sigma_m),
        }


def _runs(mask):
    """Longitudes de las rachas de True en un array booleano."""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return np.array(out, dtype=np.int32)


def estimate(tracking, blocks):
    """Mide los parametros sobre un tracking real. No usa etiquetas en ningun sitio."""
    n = tracking.n_frames
    graded = np.zeros(n, dtype=bool)
    for a, b in blocks:
        graded[a:b] = True

    seen = tracking.ball_seen[graded]
    visible = ~np.isnan(tracking.players_xy[:, :, 0])

    # Tramos continuos de cada slot: cuanto dura un track antes de romperse.
    slot_runs = [r for s in range(visible.shape[1]) for r in _runs(visible[graded, s])]

    n_tracked = visible[graded].sum(axis=1)
    values, counts = np.unique(n_tracked, return_counts=True)

    # Ruido de posicion: desviacion tipica del paso entre frames consecutivos, que en un
    # tracking limpio es movimiento real y en uno ruidoso es movimiento mas jitter. Se usa
    # como cota superior del ruido a inyectar, no como su valor exacto.
    step = np.diff(tracking.players_xy[:, :, 0], axis=0)
    step = step[np.isfinite(step)]
    step_sigma = float(np.std(step)) if step.size else 0.0

    return Spec(
        ball_absent_runs=_runs(~seen),
        ball_present_runs=_runs(seen),
        slot_runs=np.array(slot_runs, dtype=np.int32),
        n_tracked_values=values.astype(np.int32),
        n_tracked_probs=(counts / counts.sum()).astype(np.float64),
        step_sigma_m=step_sigma,
    )


def _alternating_mask(n, present_runs, absent_runs, rng, start_present=True):
    """Mascara de n frames alternando rachas remuestreadas de las dos distribuciones.

    Remuestrear longitudes enteras es la diferencia entre un hueco de 2 frames y uno de
    300. Una Bernoulli i.i.d. con la misma tasa marginal da rachas de media 1.7 frames; el
    ATD real tiene mediana 268 (10.7 s) y maximo 5537 (3.7 min).
    """
    target_absent = absent_runs.sum() / float(absent_runs.sum() + present_runs.sum())

    def draw(start):
        mask = np.empty(n, dtype=bool)
        i, present = 0, start
        while i < n:
            pool = present_runs if present else absent_runs
            length = int(rng.choice(pool)) if len(pool) else 1
            mask[i:i + length] = present
            i += length
            present = not present
        return mask

    # Las rachas son de cola muy larga (max 5537 frames) y solo caben ~140 en un partido,
    # asi que una sola tirada se desvia facil 5-7 puntos de la tasa objetivo. Se redibuja
    # hasta acercarse, en lugar de recortar las rachas -- que es justo lo que no se puede
    # tocar, porque la longitud de racha es el parametro que importa.
    best, best_err = None, float("inf")
    for _ in range(40):
        mask = draw(start_present)
        err = abs((~mask).mean() - target_absent)
        if err < best_err:
            best, best_err = mask, err
        if err <= 0.01:
            break
        start_present = bool(rng.integers(2))
    return best


def _reassign_slots(players_xy, players_speed, team_of_slot, spec, rng):
    """Trocea cada jugador y reparte los tramos entre los 46 slots sin identidad.

    Un slot no es una persona: es un objeto con vida. Al terminar un tramo el slot queda
    libre y mas tarde puede recibir a otro jugador, que es exactamente lo que hace el
    tracker cuando pierde a alguien y readquiere a otro.

    El reparto se hace **en orden temporal global**, no jugador a jugador. Hacerlo por
    jugador parece equivalente y no lo es: el primero en procesarse ocupa slots hasta el
    final del partido y los siguientes se quedan sin sitio desde el frame 0. La primera
    version de esto tenia ese fallo y daba 6 objetos por frame en lugar de 20,5.
    """
    n = players_xy.shape[0]
    n_out = N_SLOTS_HOME + N_SLOTS_AWAY
    out_xy = np.full((n, n_out, 2), np.nan, dtype=np.float32)
    out_sp = np.full((n, n_out), np.nan, dtype=np.float32)
    target_team = np.array([HOME_CODE] * N_SLOTS_HOME + [AWAY_CODE] * N_SLOTS_AWAY,
                           dtype=np.int8)

    for team in (HOME_CODE, AWAY_CODE):
        src = np.flatnonzero(team_of_slot == team)
        dst = np.flatnonzero(target_team == team)

        # 1) Todos los tramos de todos los jugadores del equipo, troceados.
        pieces = []
        for s in src:
            present = np.isfinite(players_xy[:, s, 0])
            i = 0
            while i < n:
                if not present[i]:
                    i += 1
                    continue
                end = i
                while end < n and present[end]:
                    end += 1
                while i < end:
                    length = int(rng.choice(spec.slot_runs)) if len(spec.slot_runs) else end - i
                    stop = min(i + max(length, 1), end)
                    pieces.append((i, stop, s))
                    i = stop
        # 2) En orden de inicio, que es lo que hace correcto el reparto.
        pieces.sort()

        busy = np.zeros(len(dst), dtype=np.int64)
        for begin, stop, s in pieces:
            free = np.flatnonzero(busy <= begin)
            if not len(free):
                continue            # sin slot libre el tramo se pierde, como en el ATD
            k = int(free[rng.integers(len(free))])
            slot = dst[k]
            out_xy[begin:stop, slot] = players_xy[begin:stop, s]
            out_sp[begin:stop, slot] = players_speed[begin:stop, s]
            busy[k] = stop
    return out_xy, out_sp, target_team


def _thin_to_histogram(players_xy, players_speed, spec, rng):
    """Deja caer objetos por frame hasta igualar el histograma de objetos del ATD real.

    No se puede inventar un objeto donde no lo hay, asi que esto solo quita. Por eso se
    aplica despues de la fragmentacion, que ya ha quitado por su cuenta.
    """
    n = players_xy.shape[0]
    target = rng.choice(spec.n_tracked_values, size=n, p=spec.n_tracked_probs)
    for i in range(n):
        live = np.flatnonzero(np.isfinite(players_xy[i, :, 0]))
        excess = len(live) - int(target[i])
        if excess > 0:
            drop = rng.choice(live, size=excess, replace=False)
            players_xy[i, drop] = np.nan
            players_speed[i, drop] = np.nan
    return players_xy, players_speed


def apply(tracking, spec, rng=None, position_noise=True, rho=0.9):
    """Devuelve un Tracking nuevo, degradado. No modifica el original."""
    rng = rng or np.random.default_rng()
    n = tracking.n_frames

    xy, sp, team_of_slot = _reassign_slots(
        tracking.players_xy, tracking.players_speed, tracking.team_of_slot, spec, rng)
    xy, sp = _thin_to_histogram(xy, sp, spec, rng)

    if position_noise and spec.step_sigma_m > 0:
        # AR(1) y no ruido blanco: el blanco se filtra con una media movil y el modelo
        # aprende a suavizar en vez de a desconfiar.
        sigma = 0.25 * spec.step_sigma_m
        noise = np.zeros((n, xy.shape[1], 2), dtype=np.float32)
        step = rng.normal(0.0, sigma, size=noise.shape).astype(np.float32)
        noise[0] = step[0]
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + step[i]
        xy = xy + noise

    ball_xy = tracking.ball_xy.copy()
    ball_speed = tracking.ball_speed.copy()
    seen = _alternating_mask(n, spec.ball_present_runs, spec.ball_absent_runs, rng,
                             start_present=bool(rng.integers(2)))
    ball_xy[~seen] = np.nan
    ball_speed[~seen] = np.nan

    return Tracking(ball_xy=ball_xy, ball_speed=ball_speed, players_xy=xy,
                    players_speed=sp, team_of_slot=team_of_slot,
                    home_name=tracking.home_name, away_name=tracking.away_name)
