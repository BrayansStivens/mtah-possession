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
  (J03WPY, J03WQQ, J03WR9) que no se habian tocado. Frente a la version anterior la
  mejora fue de +0.004 y +0.009 de spellF1 en dos semillas de degradacion nuevas:
  pequena, pero del mismo signo en las tres.

  Con este fichero **tal cual esta**, semilla 0 de degradacion: ajuste 0.3878,
  validacion 0.4081, los siete 0.3965. (Hasta el 31-10 aqui se leia "ajuste 0.3905,
  validacion 0.3957". El 0.3905 era correcto; el 0.3957 no salia de este fichero sino
  del volcado de `recalibrate.py`, que guarda los tres centros de clase sin redondear.
  Redondeados como estan abajo dan 0.3968. Son 0.001 de suerte en el redondeo, no un
  error de medida, pero el numero citado tiene que ser el que produce el fichero.)

**Validacion del procedimiento, 29 al 31-10.** Se midio el descenso por coordenadas
entero, no una configuracion: el mismo procedimiento sobre los 35 splits posibles de 4
partidos de ajuste / 3 de validacion. Gana **+1.7 pp** (IC 95 % [+1.4, +2.1]) en los
partidos que cada descenso no uso para ajustarse y +3.3 pp en los que si, luego
aproximadamente la mitad de la ganancia es sobreajuste. No vale decir que no se
sobreajusto. Siete hiperparametros se clavaban en un borde de su lista de valores, asi
que se amplio la rejilla (64 -> 82 valores) y se repitieron los 35 splits: +0.62 pp en el
ajuste y **-0.13 pp** en validacion, mejorando en 14/35. Por eso la rejilla de
`training/recalibrate.py` se queda como esta. El detalle no vive en este repo, que es el
de entrega, sino en el material de trabajo del equipo (`VALIDACION-CRUZADA.md`).

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
    #   actividad  +-50  (6:.3754 12:.3761 25:.3755 50:.3747 100:.3657). **12 -> 50 el
    #              31-10.** Ese barrido es plano: entre 6 y 50 hay 0.14 pp y el 12 ganaba
    #              al 50 por 0.0014, que no distingue nada. Tres medidas independientes
    #              eligen 50 y ninguna elige 12: (a) en los 35 splits el descenso elige 12
    #              en 1/35 -- 3/35 con la rejilla ampliada -- y la moda es 50; (b) poner 50
    #              y no tocar nada mas da +1.11 pp de media en los 3 partidos de validacion,
    #              con las tres semillas de degradacion del mismo signo (+1.14 / +1.40 /
    #              +0.81) y mejorando en 5 de los 7 partidos; (c) en el ajuste el cambio es
    #              neutro, +0.12 pp de media, no la perdida que sugeria el barrido.
    #              AVISO DE PROCEDENCIA: 50 coincide con el valor que habia antes del 28-10,
    #              que salia de la ventana publica y por tanto era inadmisible. La
    #              coincidencia es solo eso. El 50 de ahora se justifica con los partidos
    #              del IDSSE, en (a), (b) y (c), sin mirar la ventana publica; si la medida
    #              hubiera dado otro numero, iria otro numero.
    #   vel. balon +-6   (6:.3871 12:.3848 25:.3838 50:.3770). Cuanto mas corta mejor;
    #              2 y 4 dan +0.001, ruido. Con la rejilla ampliada hacia abajo la moda baja
    #              a 3 (15/35) y 1 (12/35), pero ese brazo no mejora en validacion: se queda.
    #   control    +-50  (6:.3304 12:.3419 25:.3608 50:.3761 100:.3735). Al reves: con el
    #              balon ausente en rachas de 10 s de mediana, el nanmean necesita ventana
    #              ancha para puentear los huecos. Confirmado, no cambia: 31/35 splits lo
    #              eligen, el mas estable de los cuatro.
    #   forma      +-200 (12:.3769 ... 100:.3816 200:.3838; 300:.3871 450:.3867).
    #              CORRECCION del 31-10: aqui se leia que 300 y 450 "ya bajan" y es falso,
    #              esos dos numeros son mas altos que el de 200. La rejilla ampliada tambien
    #              los prefiere (300 en 16/35, 450 en 11/35). Se queda en 200 igualmente,
    #              porque ese brazo gana +0.62 pp en el ajuste y -0.13 pp en los partidos
    #              que no vio: lo que elige no generaliza. Es una senal lenta y debil
    #              (AUC 0.63): solo sirve como prior de 8 s.
    "activity_smooth": 50,
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
    # El valor 1.26 es el que hace que el pipeline completo prediga D en la tasa esperada.
    # Medido el 31-10 sobre los 158.857 frames puntuados del ATD: **44.9 %**, contra la
    # media de 44.9 % de los siete partidos del IDSSE. Clavado.
    #
    # Aqui se leia "45.3 %" y era otro numero que el fichero ya no producia: con
    # activity_smooth en 12, como estaba hasta el 31-10, la tasa medida es 45.8 %. El
    # suavizado de la actividad y este centro estan acoplados -- el centro se fija por
    # percentil sobre la senal suavizada -- asi que cambiar la ventana mueve la tasa. La
    # movio a favor: de 45.8 % a 44.9 %, acercandola al objetivo. Si se vuelve a tocar
    # activity_smooth, hay que volver a medir esta tasa.
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

    # CONVENCION PARA LOS TRES CENTROS DE CLASE (este, control_center y shape_center):
    # se escriben redondeados a dos o tres cifras significativas, y si se re-derivan con
    # training/recalibrate.py se redondean igual antes de pegarlos aqui. El script los
    # vuelca con dieciseis digitos, pero son la mediana de una senal float32 sobre partidos
    # degradados con una semilla: a partir de la tercera cifra cambian con la semilla, asi
    # que escribirlos enteros aparenta una precision que la medida no tiene. Comprobado el
    # 31-10 que redondeados y sin redondear dan el mismo spellF1 a cuatro decimales en
    # ajuste, validacion y los siete partidos. (Hasta ese dia el docstring citaba un numero
    # de validacion que solo salia con los centros sin redondear; por eso esta nota.)
    #
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

    # ------------------------------------------------------------ detector de fronteras
    # models/boundary_detector.py: un score por frame de "aqui cambia la posesion". El
    # decodificador multiplica los costes de cambio de cada frame por (1 - peso * score), y
    # todos los costes de cambio por la escala. Medido con leave-one-match-out anidado en
    # los 7 partidos IDSSE x 3 semillas (ESTADO.md, 7-11): +1,17 pp spellF1 sobre el modelo
    # sin detector, y +1,07 pp sobre subir solo la escala (positivo en los 7 partidos). Los
    # dos valores los elige `python -m mtah.training.boundaries export` sobre los 7 con
    # puntuaciones fuera de muestra; se copian de models/boundary_detector.json.
    # Elegidos el 7-11: peso 0,95 y escala 2,0 (spellF1 fuera de muestra 0,4111 frente a
    # 0,4089 con escala 1,5). Los dos son el borde de su rejilla, como en los 2 pliegues de 7
    # que perdian en la medicion; no se amplia: el 31-10 se vio que ampliar sobreajusta.
    # Peso 0 = el modelo del 31-10 exactamente, sin cargar el detector.
    "boundary_discount_weight": 0.95,
    "boundary_switch_scale": 2.0,

    # ------------------------------------------------------------ Smart Tagging
    # features/tagging.team_tags: +1/-1 donde una fase de Metrica dice que equipo tiene el
    # balon. Se suma al logit de local despues del detector (models/emissions.py). El IDSSE
    # no tiene tagging, asi que el peso se elige con un tagging simulado sobre sus
    # etiquetas (training/team_tags.py). 0 = sin tagging, el modelo de antes exactamente.
    # Elegido el 8-11 entre dos pesos fijos (0,5 y 1,0) por robustez, no por maximo: con 0,5
    # el LOMO de los 7 partidos x 3 semillas gana en todos los escenarios compatibles con el
    # acuerdo medido en BAR-ATH (+0,47 a +0,90 pp; +0,61 en el que mejor ajusta los cuatro
    # cuartiles) y solo pierde si el tagging copiara del todo nuestra emision (-1,12), que
    # el acuerdo medido descarta (daria 0,807 frente a 0,762). 1,0 gana mas con tags buenos
    # y cae a +0,10 en el mejor ajuste. La eleccion automatica por pliegue era mas ruidosa
    # (+0,39). Tablas en trabajo/handoff-plugin-data/datos/smart_tagging_robustez.json.
    "smart_tagging_team_weight": 0.5,
}


def merged(**overrides):
    """Copia de DEFAULT con sustituciones, fallando si el nombre no existe."""
    unknown = sorted(set(overrides) - set(DEFAULT))
    if unknown:
        raise KeyError("hiperparametros desconocidos: {}".format(unknown))
    config = dict(DEFAULT)
    config.update(overrides)
    return config
