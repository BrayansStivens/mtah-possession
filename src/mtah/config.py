"""Todos los hiperparametros en un sitio, con la procedencia de cada numero al lado.

Un dict de Python y no un YAML a proposito: una dependencia menos, y el entregable exige
que este repo reproduzca el CSV exacto.

**Como NO elegir estos numeros.** EVALUATION.md demuestra que ajustar un solo
hiperparametro contra los 5 minutos publicos elige el valor equivocado para el partido
completo (`baseline_smooth125` gana ahi y pierde en el partido). Asi que:

**Desde el 22-10-2026 esto es regla, no consejo**: RULES.md dice que los 5 minutos
publicos "are not training data and not a tuning set". Hasta el 28-10 dieciseis valores de
aqui salian de esa ventana (medianas por clase, pesos por AUC, ventanas, costes de
transicion, duracion minima). Se re-derivaron ese dia con `training/recalibrate.py`, y
**ningun valor de este fichero depende ya de la ventana publica**. Las fuentes son dos:

- **Estadisticas sin etiquetas del ATD real** (IQR globales sobre los 158.857 frames
  puntuados, histograma de objetos, rachas sin balon). No usan ninguna etiqueta.
- **Los siete partidos del IDSSE** (Bundesliga, CC-BY 4.0, ver `training/idsse.py`),
  degradados para parecerse al ATD (`training/degrade.py`) y con el convenio de posesion
  del reto (`training/convention.py`). Los centros de clase son medianas medidas ahi. El
  resto se eligio o se confirmo por busqueda por coordenadas maximizando spellF1 medio en
  4 partidos de AJUSTE (J03WMX, J03WN1, J03WOH, J03WOY) y se comprobo en 3 de VALIDACION
  (J03WPY, J03WQQ, J03WR9) que no se habian tocado: ajuste 0.3905, validacion 0.3957.
  Con dos semillas de degradacion nuevas la mejora frente a la version anterior fue
  +0.004 y +0.009 de spellF1: pequena, pero del mismo signo en las tres.

Cada valor lleva al lado de donde sale. RULES.md exige que el documento tecnico pueda
explicar cualquier dato externo que entre en el metodo, y SUBMISSION.md que liste cada
dataset etiquetado: fuente, quien lo etiqueto y cuanto. Aqui esta la respuesta.
"""

DEFAULT = {
    # ---------------------------------------------------------------- features
    "control_tau_m": 3.0,          # radio de control del kernel exponencial, en metros
    "count_radii_m": (2.0, 5.0),

    # Anchos de ventana de suavizado centrado, en frames (25 fps). Hasta el 28-10 eran
    # +-50 los cuatro, elegidos por AUC en la ventana publica. Re-derivados en el arnes
    # IDSSE (barrido 6/12/25/50/100, mas 200 para la forma):
    #   actividad  +-12  (6:.3754 12:.3761 25:.3755 50:.3747 100:.3657). La senal de balon
    #              parado quiere ventana corta: +-2 s se comia las posesiones cortas.
    #   vel. balon +-6   (6:.3871 12:.3848 25:.3838 50:.3770). Cuanto mas corta mejor;
    #              2 y 4 dan +0.001, ruido.
    #   control    +-50  (6:.3304 12:.3419 25:.3608 50:.3761 100:.3735). Al reves: con el
    #              balon ausente en rachas de 10 s de mediana, el nanmean necesita ventana
    #              ancha para puentear los huecos. Confirmado, no cambia.
    #   forma      +-200 (12:.3769 ... 100:.3816 200:.3838; 300:.3871 450:.3867 ya bajan).
    #              Es una senal lenta y debil (AUC 0.63): solo sirve como prior de 8 s.
    "activity_smooth": 12,
    "ball_speed_smooth": 6,
    "control_smooth": 50,
    "shape_smooth": 200,

    # ------------------------------------------------- detector de balon muerto
    # 1.75 -> 1.26 el 2026-10-24. Fallo estructural, no de medida.
    #
    # El 1.75 salia del cruce de medianas por clase de la ventana publica (vivo 2.296,
    # muerto 1.204) y esas dos medianas son correctas. El problema es que un cruce de
    # medianas depende de la MEZCLA DE CLASES de la muestra, y la ventana publica tiene
    # 19.6% de balon parado mientras que un partido completo ronda el 45%. Trasplantar ese
    # cruce a otra mezcla lo coloca en el percentil equivocado.
    #
    # Medido: 1.75 cae en el **percentil 56.3** de la actividad real del partido, y el
    # modelo acababa prediciendo D en el 55.4% de los frames puntuados. Cuadra exacto.
    #
    # La tasa objetivo sale de los 7 partidos del IDSSE (training/idsse.py), que son
    # partidos completos y por tanto tienen la mezcla de clases correcta:
    #     41.5  51.7  44.3  47.6  40.7  46.5  41.9   ->  media 44.9%, rango 40.7-51.7
    # El valor 1.26 es el que hace que el pipeline completo prediga D en el 45.3% de los
    # frames puntuados, que es la tasa esperada.
    #
    # Los dos ingredientes son independientes de la ventana publica: la tasa viene de
    # datos externos y la distribucion de actividad se mide sobre los 158.857 frames
    # puntuados sin usar ninguna etiqueta. No se ha elegido por maximizar spellF1 ahi.
    # (Para que conste: en la ventana publica spellF1 sube de 0.3385 a 0.4824 y macroF1
    # de 0.7181 a 0.8021. La mejora es monotona en todo el barrido 1.75 -> 1.26, no un
    # pico, y de un tamano muy superior a la banda de ruido de ~0.04 de EVALUATION.md.)
    #
    # La escala sigue saliendo del IQR global (1.037 - 2.305), que no depende de la mezcla.
    # Barrido en el arnes 1.10/1.18/1.26/1.35 -> 1.26 (.3872/.3845/.3905/.3815): el valor
    # que sale de igualar el prior de D es tambien el mejor sobre partidos externos.
    "activity_center_ms": 1.26,
    "activity_scale_ms": 0.45,
    "activity_weight": 1.0,        # arnes IDSSE: 0.5/1.0/1.5/2.5 -> .3793/.3905/.3753/.3599

    # Cruce de medianas por clase sobre los 4 partidos de ajuste del IDSSE, degradados:
    # vivo 6.279 m/s, muerto 0.000 -> 3.14. (Antes 4.10, de la ventana publica.) El muerto
    # da 0 porque el balon parado de verdad no se mueve; en la ventana publica salia 1.84
    # porque ahi "muerto" incluia el balon rodando hacia el saque. La escala sigue siendo
    # el IQR global del ATD (0.0 - 6.8), sin etiquetas.
    "ball_speed_center_ms": 3.14,
    "ball_speed_scale_ms": 2.00,
    "ball_speed_weight": 0.6,      # arnes IDSSE: 0/0.3/0.6/1.2 -> .3524/.3752/.3905/.3899

    # No ver el balon es evidencia debil de parada. (La version anterior lo justificaba con
    # tasas de la ventana publica, 93.5% vivos / 66.4% muertos, que ademas no pueden valer
    # para el partido completo: con ellas la visibilidad global seria 0.81 y es 0.59.)
    # Valor confirmado en el arnes IDSSE: 0/0.25/0.5/1.0 -> .3820/.3884/.3905/.3744.
    # Probado 2026-10-24: subirlo al cociente de verosimilitudes implicito (+1.79, medido
    # sobre el ATD real) NO mejora. A tasa de D igualada por biseccion, spellF1 cae de
    # 0.4824 a 0.4686 y a partir de 2.5 se hunde. Motivo: ese +1.79 es un cociente
    # MARGINAL y aqui los logits se suman, lo que supone independencia condicional. La
    # visibilidad del balon va del 9.7% al 90.0% segun la actividad -- las dos senales las
    # causa lo mismo, el tracker en apuros. Subirlo es contar la evidencia dos veces.
    "ball_unseen_logit": 0.5,      # AUC 0.7700
    # Balon trackeado fuera del campo: evidencia dura pero de cobertura baja, porque
    # cuando el balon sale el tracker suele perderlo en lugar de seguirlo fuera.
    "ball_out_logit": 2.5,

    # ------------------------------------------------------ propiedad del equipo
    # Cruce de medianas de la diferencia de control sobre los 4 partidos de ajuste del
    # IDSSE, degradados: H 0.212, A -0.196 -> 0.008. (Antes 0.096 de la ventana publica,
    # H 0.356 / A -0.164: con solo 9 posesiones visitantes el lado A salia corto.) Que el
    # cruce quede en ~0 es lo esperable de una senal simetrica entre equipos.
    # El IQR global del ATD es estrecho (-0.041 a 0.173), asi que la pendiente tiene que
    # ser alta o la senal queda plana.
    "control_center": 0.008,
    "control_weight": 6.0,         # arnes IDSSE: 2/4/6/9/13 -> .3559/.3861/.3905/.3871/.3755

    # Forma de equipo: la unica senal de equipo que NO necesita el balon, y por eso
    # sostiene el 41% de frames puntuados sin el. Cruce de medianas sobre los 4 partidos
    # de ajuste del IDSSE, degradados: H 1.046, A 0.905 -> 0.976. (La ventana publica
    # daba 0.975 con H 1.195 / A 0.756: mismo centro, clases mucho mas separadas de lo
    # que son en realidad -- el AUC real es 0.63, no 0.81.) IQR global 0.742 - 1.652.
    "shape_center": 0.976,

    # shape_weight: se queda en 1.6, pero el AUC documentado estaba mal.
    #
    # 2026-10-24. El 0.8124 de la version anterior se midio sobre la ventana publica (17
    # posesiones) y es ruido: medido sobre los SIETE partidos del IDSSE (~670 min, ver
    # training/idsse.py) el AUC real es 0.632, rango 0.586-0.670, ningun partido por
    # encima de 0.68.
    #
    # Con ese dato baje el peso a 0.6 reescalando por d-prime. Al validarlo sobre los 7
    # partidos degradados (training/harness.py), 3 semillas cada uno, resulta que **da
    # igual**:
    #     1.6 -> spellF1 0.2880 +- 0.0067   macroF1 0.7638 +- 0.0032
    #     0.6 -> spellF1 0.2892 +- 0.0056   macroF1 0.7616 +- 0.0011
    # La diferencia (+0.0013 / -0.0022) es mucho menor que la dispersion entre semillas.
    # Son indistinguibles, asi que se mantiene 1.6, que es el valor con procedencia en el
    # metodo documentado de este fichero.
    #
    # La leccion, que vale para TODOS los pesos de aqui: esto es una suma de logits, o sea
    # supone independencia condicional, y las features no son independientes. El AUC
    # marginal de la forma de equipo es bajo porque se promedia sobre todos los frames,
    # pero su valor esta concentrado en el 41% de frames sin balon, donde es la unica
    # senal de equipo que queda. Un estadistico marginal no determina el peso correcto.
    # El mismo error se cometio con ball_unseen_logit (ver abajo).
    #
    # Correccion del 28-10: lo de arriba se midio con shape_smooth = 50. Con la ventana
    # re-derivada (+-200) la senal SI importa: 0/0.5/1.0/1.6/3.0 -> .3629/.3739/.3866/
    # .3905/.3829. Una senal de AUC 0.63 frame a frame no vale nada por frames, pero
    # promediada sobre 8 s es un prior de equipo util justo donde no hay balon.
    "shape_weight": 1.6,           # arnes IDSSE, con shape_smooth 200

    "prob_floor": 0.02,            # suelo de probabilidad, para que ningun frame sea infinito

    # ------------------------------------------------------------ decodificador
    # Costes de transicion asimetricos, en nats. La asimetria (salir de D barato, entrar
    # caro) se mantiene; los valores salen del arnes IDSSE, no de la ventana publica:
    #   turnover   8/14/20/28/40 -> .3673/.3854/.3852/.3871/.3905; 55/75/100 dan .3918,
    #              una meseta: por encima de 40 el valor exacto no importa. Antes 28.
    #   to_dead    10/18/26/34/45 -> .3728/.3828/.3860/.3905/.3896. Confirmado.
    #   from_dead  4/8/14/22 -> .3845/.3882/.3905/.3887. Confirmado.
    "switch_turnover": 40.0,
    "switch_to_dead": 34.0,
    "switch_from_dead": 14.0,

    # Duracion minima de segmento. 12 frames = 0,5 s. En el arnes IDSSE 0, 6 y 12 dan
    # exactamente lo mismo (.3871) y 25 / 50 empeoran (.3865 / .3857): con los costes de
    # transicion de arriba el decodificador ya no produce segmentos de menos de 0,5 s, asi
    # que este valor es un cinturon de seguridad, no una palanca. Se deja en 12.
    "min_duration_frames": 12,
}


def merged(**overrides):
    """Copia de DEFAULT con sustituciones, fallando si el nombre no existe."""
    unknown = sorted(set(overrides) - set(DEFAULT))
    if unknown:
        raise KeyError("hiperparametros desconocidos: {}".format(unknown))
    config = dict(DEFAULT)
    config.update(overrides)
    return config
