"""Constantes del partido y del contrato de entrega.

Todo lo que aqui aparece esta documentado publicamente en el repo de los organizadores
(DATA.md, FAQ.md) o se deriva del propio video. Nada viene del ground truth privado.
"""

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = 199_319          # frames del video tactico == filas de la submission
N_TRACK_SLOTS = 46          # 24 Barcelona + 22 Athletic, NO son jugadores
PITCH = (105.3, 68.0)       # metros, medido sobre el tracking corregido (games.py)

HOME, AWAY, DEAD, ABSTAIN = "H", "A", "D", "X"
LABELS = (HOME, AWAY, DEAD, ABSTAIN)

HOME_TEAM, AWAY_TEAM = "Barcelona", "Atletic Club"   # el primer Team del XML es el local

# Bloques puntuados, en frames de video. Semiabiertos [inicio, fin).
# Publicados en DATA.md: son fronteras, no etiquetas.
GRADED_BLOCKS = ((11_185, 90_652), (110_808, 190_198))
N_GRADED = sum(b - a for a, b in GRADED_BLOCKS)       # 158_857

# El segundo tiempo se puntua desde 110_808 pero no se saca hasta 113_742: esos 2_934
# frames son balon parado. Dato publicado en el FAQ del repo oficial.
KICKOFF_2ND = 113_742

# Velocidad maxima plausible del balon en m/s. Se usa para descartar saltos imposibles
# del tracking al rellenar huecos.
BALL_MAX_SPEED = 40.0


def is_graded(frame):
    """True si el frame cuenta para la puntuacion."""
    return any(a <= frame < b for a, b in GRADED_BLOCKS)


def graded_mask(n_frames=N_FRAMES):
    """Mascara booleana sobre todos los frames, True donde se puntua."""
    import numpy as np
    mask = np.zeros(n_frames, dtype=bool)
    for a, b in GRADED_BLOCKS:
        mask[a:b] = True
    return mask
