"""Smart Tagging: anotacion tactica *automatica* del video tactico.

XML de SportsCode, 638 instancias en este partido. Cada una es un intervalo con un `code`
y un conjunto de etiquetas agrupadas (`Team`, `Half`, `Type`, ...).

Base de tiempos verificada por los organizadores: `<start>` y `<end>` son segundos desde el
primer frame del video tactico, asi que `round(s * 25)` es un frame de video y no hay nada
mas que corregir.

Tres avisos que cuestan dinero si se ignoran:

- **No es ground truth.** Metrica lo genera automaticamente del mismo video y tracking que
  tienes tu, sin revision manual. Falsos positivos y negativos incluidos.
- **`Team` no siempre es un equipo.** 36 instancias van con `N/A`, entre ellas *todos* los
  saques de banda, justo donde saber el lado valdria mas.
- **Los intervalos son fases, no posesiones.** Un Build Up puede abarcar una posesion,
  quedarse corto o cubrir dos. Copiar un intervalo como etiqueta importa sus errores.

Los codigos van en mayusculas (`SET PIECES`) y los valores de etiqueta en Title Case
(`Kick Off (Start)`), asi que todo se compara sin distinguir mayusculas.
"""

import xml.etree.ElementTree as ET

from ..constants import FPS, GAME, HOME_TEAM, AWAY_TEAM
from ..paths import require

SET_PIECES = "set pieces"
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick",
                 "kick off (after goal)", "kick off (start)")


class Phase:
    """Una instancia del tagging, ya convertida a frames de video [start, end)."""

    __slots__ = ("code", "start_frame", "end_frame", "start_s", "end_s", "tags")

    def __init__(self, code, start_s, end_s, tags, fps=FPS):
        self.code = code or ""
        self.start_s = start_s
        self.end_s = end_s
        self.start_frame = int(round(start_s * fps))
        self.end_frame = int(round(end_s * fps))
        self.tags = tags

    def tag(self, group, default=None):
        """Valor de un grupo de etiquetas, comparando el nombre sin distinguir mayusculas."""
        for key, value in self.tags.items():
            if (key or "").strip().lower() == group.strip().lower():
                return value
        return default

    @property
    def team_label(self):
        """'H', 'A' o None. None incluye el `N/A` explicito, que no es un equipo."""
        value = (self.tag("Team") or "").strip().lower()
        if value == HOME_TEAM.lower():
            return "H"
        if value == AWAY_TEAM.lower():
            return "A"
        return None

    @property
    def is_set_piece(self):
        return self.code.strip().lower() == SET_PIECES

    @property
    def restart_type(self):
        value = (self.tag("Type") or "").strip().lower()
        return value if value in RESTART_TYPES else None

    def __repr__(self):
        return "Phase({!r}, {}-{}, team={})".format(
            self.code, self.start_frame, self.end_frame, self.team_label)


def load(source="tactical", game=GAME, fps=FPS):
    """Lista de Phase en el orden del fichero."""
    path = require(["smart_tagging"], source=source, game=game)[0]
    phases = []
    for instance in ET.parse(path).iter("instance"):
        start = instance.findtext("start")
        end = instance.findtext("end")
        if start is None or end is None:
            continue
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        phases.append(Phase(instance.findtext("code"), float(start), float(end), tags, fps))
    return phases


def set_pieces(phases):
    """Solo los saques: cada uno marca un reinicio, es decir el final de un balon parado."""
    return [p for p in phases if p.is_set_piece]


def code_counts(phases):
    counts = {}
    for phase in phases:
        counts[phase.code] = counts.get(phase.code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
