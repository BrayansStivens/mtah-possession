"""Decodificacion temporal: de puntuaciones por frame a segmentos de posesion.

Por que este modulo es el nucleo del reto
-----------------------------------------
El ranking es `spellF1`: cada posesion se empareja 1 a 1 con la verdad por IoU temporal.
Eso castiga dos cosas a la vez y de forma simetrica -- inventar posesiones y perderlas.

Un voto por mayoria en ventana (lo que hacen los baselines oficiales) controla el numero
de segmentos *indirectamente*, via el tamano de ventana, y para no fragmentar una posesion
larga tiene que borrar las cortas. Son dos objetivos distintos y una sola ventana no puede
separarlos: por eso `baseline_smooth125` gana en numero de posesiones y pierde en macroF1,
y `baseline_smooth25` lo contrario.

Viterbi penaliza el numero de transiciones *directamente*, asi que puede mantener una
posesion de 1 s y a la vez no trocear una de 40 s. Medido con el scorer oficial sobre la
ventana publica, partiendo de la MISMA etiqueta cruda que el baseline:

    voto mayoria: baseline_smooth50      spellF1 0.3256   macroF1 0.6207   26 poss
    voto mayoria: baseline_smooth125     spellF1 0.4242   macroF1 0.5812   16 poss
    Viterbi switch_cost=40               spellF1 0.4000   macroF1 0.6686   16 poss

Gana a smooth50 en las dos metricas a la vez y supera el macroF1 de los cuatro baselines.

AVISO: 40 es el valor que gana en los 5 minutos publicos, y ese es exactamente el error
que EVALUATION.md demuestra con smooth125. El valor para el partido completo hay que
elegirlo por un criterio que no sea ajustar contra esa ventana -- distribucion de
duraciones, validacion cruzada en datos externos, o bootstrap.

Latencia
--------
Medido degradando el ground truth publico: +1 frame de retraso es gratis (spellF1 1.0000),
+25 frames (1 s) baja a 0.7765 y +50 frames (2 s) a 0.5882. Cualquier filtro causal
(media movil hacia atras, EMA, decodificacion online) es inviable. Viterbi es offline y
centrado por construccion: usa la secuencia entera, y por eso no introduce retraso.
"""

import numpy as np

from ..constants import GRADED_BLOCKS

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2
N_STATES = 3


def uniform_switch_costs(switch_cost):
    """Matriz de coste de transicion (3, 3) con el mismo precio para todo cambio."""
    matrix = np.full((N_STATES, N_STATES), float(switch_cost))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def switch_costs(turnover, to_dead, from_dead):
    """Matriz de transicion asimetrica, que es como se comporta el futbol de verdad.

    En los 5 minutos publicos, 12 de los 16 cambios de posesion son *directos* (H->A sin
    balon parado en medio) y solo 4 pasan por una parada. Es decir: el problema dominante
    son los robos en juego, no los saques, y cobrar lo mismo por H->A que por H->D->A
    sesga el decodificador hacia inventar paradas.

        turnover    H <-> A directo
        to_dead     H|A -> D   (el balon se muere)
        from_dead   D -> H|A   (reinicio)
    """
    matrix = np.zeros((N_STATES, N_STATES))
    matrix[H, A] = matrix[A, H] = float(turnover)
    matrix[H, D] = matrix[A, D] = float(to_dead)
    matrix[D, H] = matrix[D, A] = float(from_dead)
    return matrix


def costs_from_hard_labels(labels, miss_cost=1.0, unknown_cost=0.0):
    """Etiquetas duras por frame -> matriz de costes (n, 3), para decodificar un baseline.

    Coste 0 para el estado observado y `miss_cost` para los otros dos. Donde la
    observacion es desconocida (`X` o cualquier cosa fuera de H/A/D) el coste es
    `unknown_cost` para los tres, es decir: el frame no opina y deja que manden los vecinos.
    """
    labels = np.asarray(labels)
    cost = np.full((len(labels), N_STATES), float(miss_cost))
    unknown = np.ones(len(labels), dtype=bool)
    for code, name in enumerate(STATES):
        hit = labels == name
        cost[hit, code] = 0.0
        unknown &= ~hit
    cost[unknown] = float(unknown_cost)
    return cost


def check_switch_discount(switch_discount, n_frames):
    """Valida un descuento por frame, en fraccion, sobre el coste de cambiar de estado.

    `switch_discount[t]` multiplica los seis costes de cambio entre los frames t-1 y t por
    `1 - switch_discount[t]`; `switch_discount[0]` no se usa. Es la entrada del detector de
    fronteras (training/boundaries.py): donde ve una transicion, cambiar sale mas barato.

    Fraccion y no resta, porque los cambios cuestan muy distinto (un robo 40, un reinicio
    14): una resta tendria que quedar por debajo de 14 y apenas moveria un robo. Tiene que
    ser < 1: con 1 el cambio sale gratis y el decodificador fragmentaria ahi sin que
    ninguna emision lo pida.
    """
    discount = np.asarray(switch_discount, dtype=np.float64)
    if discount.shape != (n_frames,):
        raise ValueError("switch_discount debe tener forma ({},), tiene {}".format(
            n_frames, discount.shape))
    if not np.isfinite(discount).all() or (discount < 0).any():
        raise ValueError("switch_discount debe ser finito y no negativo")
    if discount.size and discount.max() >= 1.0:
        raise ValueError(
            "switch_discount llega a {:.3f}: con 1 o mas un cambio sale gratis y hace "
            "gratis fragmentar. Baja el peso del detector.".format(discount.max()))
    return discount


def _viterbi_block(cost, transition, switch_discount=None):
    """Viterbi sobre un bloque contiguo. cost (n, 3) coste, menor es mejor.

    Bucle en Python puro a proposito: con 3 estados son 9 sumas y 3 minimos por frame, y
    la sobrecarga por llamada de numpy sobre arrays de 3 elementos es mayor que el calculo.

    `switch_discount` (n,) escala los seis costes de cambio al entrar en cada frame (ver
    check_switch_discount). Sin el, o con ceros, el resultado es identico bit a bit al de
    antes de existir: multiplicar por 1.0 no cambia ningun float.
    """
    n = len(cost)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    emit = cost.tolist()
    keep = ([1.0] * n if switch_discount is None
            else (1.0 - np.asarray(switch_discount, float)).tolist())
    t00, t01, t02 = transition[0]
    t10, t11, t12 = transition[1]
    t20, t21, t22 = transition[2]

    d0, d1, d2 = emit[0]
    back0, back1, back2 = [], [], []

    for t in range(1, n):
        e0, e1, e2 = emit[t]
        k = keep[t]

        c0, c1, c2 = d0 + t00, d1 + t10 * k, d2 + t20 * k
        if c0 <= c1 and c0 <= c2:
            n0, b0 = c0, 0
        elif c1 <= c2:
            n0, b0 = c1, 1
        else:
            n0, b0 = c2, 2

        c0, c1, c2 = d0 + t01 * k, d1 + t11, d2 + t21 * k
        if c0 <= c1 and c0 <= c2:
            n1, b1 = c0, 0
        elif c1 <= c2:
            n1, b1 = c1, 1
        else:
            n1, b1 = c2, 2

        c0, c1, c2 = d0 + t02 * k, d1 + t12 * k, d2 + t22
        if c0 <= c1 and c0 <= c2:
            n2, b2 = c0, 0
        elif c1 <= c2:
            n2, b2 = c1, 1
        else:
            n2, b2 = c2, 2

        d0, d1, d2 = n0 + e0, n1 + e1, n2 + e2
        back0.append(b0)
        back1.append(b1)
        back2.append(b2)

    out = np.empty(n, dtype=np.int8)
    state = 0 if (d0 <= d1 and d0 <= d2) else (1 if d1 <= d2 else 2)
    out[n - 1] = state
    backs = (back0, back1, back2)
    for t in range(n - 1, 0, -1):
        state = backs[state][t - 1]
        out[t - 1] = state
    return out


def decode(cost, transition, blocks=GRADED_BLOCKS, switch_discount=None):
    """Viterbi por bloque puntuado. Devuelve codigos int8 (0=H, 1=A, 2=D).

    Nunca decodifica atravesando la frontera entre bloques: el agujero del descanso son
    20.156 frames y una transicion fantasma ahi es una posesion inventada.
    Los frames fuera de los bloques quedan como D, que no se puntua.

    `switch_discount` es opcional y por frame (ver check_switch_discount). El primer frame
    de cada bloque no es una transicion, asi que su descuento no se usa.
    """
    transition = np.asarray(transition, dtype=np.float64)
    if switch_discount is not None:
        switch_discount = check_switch_discount(switch_discount, len(cost))
    out = np.full(len(cost), D, dtype=np.int8)
    for start, stop in blocks:
        block_discount = None if switch_discount is None else switch_discount[start:stop]
        out[start:stop] = _viterbi_block(cost[start:stop], transition, block_discount)
    return out


def enforce_min_duration(codes, min_frames, blocks=GRADED_BLOCKS):
    """Absorbe segmentos mas cortos que `min_frames` en el vecino mas barato.

    Cuidado con pasarse: en los 5 minutos publicos, 6 de las 17 posesiones duran menos de
    4 s y la mas corta 0,76 s (19 frames). Medido sobre el ground truth, borrar las
    posesiones de menos de 2 s cuesta 0.17 de spellF1 y las de menos de 5 s cuesta 0.44.
    Un minimo por encima de ~0,5 s empieza a destruir posesiones reales.
    """
    if min_frames <= 1:
        return codes.copy()

    out = codes.copy()
    for start, stop in blocks:
        window = out[start:stop]
        changed = True
        while changed:
            changed = False
            breaks = np.flatnonzero(np.diff(window)) + 1
            bounds = np.concatenate(([0], breaks, [len(window)]))
            for i in range(len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                if b - a >= min_frames:
                    continue
                before = window[a - 1] if a > 0 else None
                after = window[b] if b < len(window) else None
                if before is None and after is None:
                    continue
                if before is None:
                    window[a:b] = after
                elif after is None:
                    window[a:b] = before
                else:
                    # Al vecino mas largo, para no encadenar absorciones en cascada.
                    len_before = a - int(bounds[i - 1]) if i > 0 else 0
                    len_after = int(bounds[i + 2]) - b if i + 2 < len(bounds) else 0
                    window[a:b] = before if len_before >= len_after else after
                changed = True
                break
        out[start:stop] = window
    return out


def to_labels(codes):
    """Codigos int -> array de letras 'H'/'A'/'D'."""
    return np.array(STATES, dtype="<U1")[np.asarray(codes, dtype=np.int64)]


def segment_table(codes, blocks=GRADED_BLOCKS, fps=25):
    """[(start, stop, label, duracion_s)] por bloque. Para inspeccionar a ojo."""
    rows = []
    for start, stop in blocks:
        window = codes[start:stop]
        breaks = np.flatnonzero(np.diff(window)) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            a, b = start + int(bounds[i]), start + int(bounds[i + 1])
            rows.append((a, b, STATES[int(codes[a])], (b - a) / float(fps)))
    return rows
