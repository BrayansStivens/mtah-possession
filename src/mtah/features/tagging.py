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
"""

import numpy as np

from ..constants import N_FRAMES


def team_logit(phases, n_frames=N_FRAMES):
    """(n,) +1 por cada fase tagueada como local que cubre el frame, -1 por cada visitante.

    Se suma en lugar de tomar la ultima porque las fases se solapan: un mismo frame puede
    estar dentro de un Build Up y de un Ball in Final Third a la vez. Que coincidan dos
    fases del mismo equipo es mas evidencia que una sola, y que coincidan de equipos
    distintos es una contradiccion que debe cancelarse.
    """
    out = np.zeros(n_frames, dtype=np.float32)
    for phase in phases:
        label = phase.team_label
        if label is None:                      # el N/A explicito no es un equipo
            continue
        start = max(0, phase.start_frame)
        stop = min(n_frames, phase.end_frame)
        if stop > start:
            out[start:stop] += 1.0 if label == "H" else -1.0
    return out


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
