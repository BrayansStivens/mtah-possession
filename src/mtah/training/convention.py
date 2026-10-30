"""Llevar las etiquetas del IDSSE al convenio de posesion del reto.

**El problema.** El `BallPossession` del DFL cambia en cada toque individual: J03WPY tiene
421 posesiones de 7,7 s de media. El reto no: CHALLENGE.md define el cambio de posesion
como el momento en que un rival gana posesion **controlada**, y dice que un simple toque sin
control, un desvio que no intercepta o un duelo aereo sin control posterior **no** son
cambios de posesion. Son dos convenios distintos para el mismo fenomeno.

Mientras no se corrija, el spellF1 del arnes esta hundido por una razon estructural que no
tiene que ver con la calidad del modelo: se le pide acertar 421 posesiones cuando esta
disenado para producir ~220.

**De donde sale la regla de fusion -- y de donde NO.** Sale de la definicion del reto,
aplicada a los partidos externos. CHALLENGE.md propone como proxy de posesion controlada
que *«a player keeps the ball for a few seconds»*. Aqui eso se lee como: un tramo del rival
mas corto que `MIN_CONTROLLED_S`, encajado dentro de una posesion ajena, es un contacto sin
control y se absorbe. Un tramo igual o mas largo es un cambio real y se respeta.

Una version anterior de este modulo buscaba el umbral que diera 14,2 s de duracion media,
que es la media de las 17 posesiones de la ventana publica. Se retiro: desde el 22 de
octubre las reglas dicen que esa ventana no es conjunto de ajuste, y aunque fuera solo
para interpretar el convenio, 17 posesiones son poca base para una constante. Ahora la
duracion media resultante es una **medicion** que se reporta, no un objetivo que se busca.

**Que se ajusta y que no.** Solo se fusionan posesiones de equipo separadas por contactos
breves del rival. **El balon parado no se toca**: la tasa de `D` del IDSSE ya es comparable
a la que se espera del partido completo, y ahi no hay discrepancia que arreglar.

**Lo que esta lectura deja fuera, a proposito.** CHALLENGE.md da un segundo proxy, *«a team
completes more than two consecutive passes»*, que necesitaria el event data del DFL
(kloppy lo carga con `sportec.load_open_event_data`). No esta implementado. Un rival que
en menos de `MIN_CONTROLLED_S` completa tres pases es un caso raro, pero existe, y esta
version lo absorberia mal.
"""

import numpy as np

# "A few seconds" de CHALLENGE.md, leido como 3. Es una lectura del texto de la regla, no
# una medida sobre ningun partido: si alguien tiene mejor criterio para el numero, este es
# el unico sitio que hay que tocar.
MIN_CONTROLLED_S = 3.0


def spells(labels):
    """[(etiqueta, inicio, fin_exclusivo)] de rachas maximas."""
    labels = np.asarray(labels, dtype="<U1")
    if not len(labels):
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    return [(labels[bounds[i]], int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1)]


def _absorb_uncontrolled(sp, max_len):
    """Absorbe tramos de equipo mas cortos que `max_len` flanqueados por el MISMO rival.

    Solo cuando los dos vecinos son el mismo equipo: eso es un contacto breve dentro de una
    posesion ajena. Un toque breve entre dos equipos distintos, o pegado a un balon parado,
    no se toca: ahi no hay posesion ajena que lo envuelva.
    """
    changed = True
    while changed:
        changed = False
        out, i = [], 0
        while i < len(sp):
            lab, a, b = sp[i]
            if (0 < i < len(sp) - 1 and lab in "HA" and (b - a) < max_len
                    and sp[i - 1][0] == sp[i + 1][0] and sp[i - 1][0] in "HA"):
                prev = out[-1]
                out[-1] = (prev[0], prev[1], sp[i + 1][2])
                i += 2
                changed = True
                continue
            out.append((lab, a, b))
            i += 1
        sp = out
    return sp


def _mean_possession_s(sp, fps):
    lengths = [b - a for lab, a, b in sp if lab in "HA"]
    return (float(np.mean(lengths)) / fps) if lengths else 0.0


def harmonise(labels, fps=25, min_controlled_s=MIN_CONTROLLED_S):
    """Etiquetas con el convenio del reto. Devuelve (etiquetas, info) para poder auditarlo."""
    labels = np.asarray(labels, dtype="<U1")
    base = spells(labels)
    final = _absorb_uncontrolled(base, min_controlled_s * fps)

    out = labels.copy()
    for lab, a, b in final:
        out[a:b] = lab

    info = {
        "min_controlled_s": float(min_controlled_s),
        "n_poss_antes": sum(1 for l, _, _ in base if l in "HA"),
        "n_poss_despues": sum(1 for l, _, _ in final if l in "HA"),
        "dur_media_antes_s": _mean_possession_s(base, fps),
        "dur_media_despues_s": _mean_possession_s(final, fps),
        "frames_cambiados": int((out != labels).sum()),
        "pct_frames_cambiados": 100.0 * float((out != labels).mean()),
    }
    return out, info
