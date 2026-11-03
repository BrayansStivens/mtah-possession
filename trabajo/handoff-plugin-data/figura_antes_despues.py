"""Figura antes/despues de spellF1 para el documento tecnico."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

D = "/Users/bultoesal/Documents/Hackaton/handoff-plugin-data/datos/"
OUT = "/Users/bultoesal/Documents/Hackaton/figuras/"
MINIMO = 0.2433

AZUL, GRIS, ROJO, TENUE, TINTA = "#2f6db5", "#8b909a", "#c0564e", "#dcdfe4", "#2b2f36"
X_VAL, X_PP, X_MAX = 0.525, 0.645, 0.66   # columnas de texto, fuera de la zona de datos

sc = pd.read_csv(D + "scores_por_partido.csv")
agg = json.load(open(D + "agregados_por_config.json"))
media = {(a["config"], a["grupo"]): a["spellF1"] for a in agg}

piv = sc.pivot(index="partido", columns="config", values="spellF1")
grupo = sc.drop_duplicates("partido").set_index("partido").grupo

# De arriba abajo: ajuste (4 partidos + media), hueco, validacion (3 + media).
filas = []
for g, etiqueta in [("ajuste", "Ajuste"), ("validacion", "Validación")]:
    ids = sorted([p for p in piv.index if grupo[p] == g],
                 key=lambda p: -piv.loc[p, "recalibrado"])
    ini = len(filas)
    for p in ids:
        filas.append({"y": len(filas), "nom": p, "a": piv.loc[p, "actual"],
                      "r": piv.loc[p, "recalibrado"], "media": False})
    filas.append({"y": len(filas), "nom": f"media ({len(ids)})",
                  "a": media[("actual", g)], "r": media[("recalibrado", g)], "media": True})
    filas[-1]["banda"] = (etiqueta, ini, len(filas) - 1)
    if g == "ajuste":
        filas.append(None)   # hueco entre bloques
        len_hueco = len(filas) - 1

filas_d = [f for f in filas if f]
alto_total = len(filas)

fig, ax = plt.subplots(figsize=(9.6, 5.8))

for f in filas_d:
    y, a, r = f["y"], f["a"], f["r"]
    sube = r >= a
    col = AZUL if sube else ROJO
    ancho = 3.0 if f["media"] else 2.2
    ms = 9 if f["media"] else 7.5
    ax.plot([a, r], [y, y], color=col, lw=ancho, solid_capstyle="round", zorder=3)
    ax.plot(a, y, "o", ms=ms, mfc="white", mec=GRIS, mew=1.8, zorder=4)
    ax.plot(r, y, "o", ms=ms, color=col, zorder=4)
    peso = "bold" if f["media"] else "normal"
    ax.annotate(f"{r:.4f}".replace(".", ","), (X_VAL, y), va="center", ha="left",
                fontsize=9.5, color=col, fontweight=peso)
    ax.annotate(f"{(r - a) * 100:+.2f} pp".replace(".", ","), (X_PP, y), va="center",
                ha="right", fontsize=9, color=GRIS if sube else ROJO, fontweight=peso)

# Bandas de grupo: etiqueta girada a la izquierda y corchete.
for f in filas_d:
    if "banda" not in f:
        continue
    etiqueta, y0, y1 = f["banda"]
    centro = (y0 + y1) / 2
    ax.annotate(etiqueta, xy=(-0.165, centro), xycoords=("axes fraction", "data"),
                rotation=90, va="center", ha="center", fontsize=10.5, color=TINTA)
    ax.annotate("", xy=(-0.125, y0 - 0.34), xytext=(-0.125, y1 + 0.34),
                xycoords=("axes fraction", "data"),
                arrowprops=dict(arrowstyle="-", color=TENUE, lw=1.4))

ax.axhline(len_hueco, color=TENUE, lw=1, zorder=1)
ax.axvline(X_VAL - 0.016, color=TENUE, lw=1, zorder=1)
ax.axvline(MINIMO, color=GRIS, lw=1.1, ls=(0, (4, 3)), zorder=2)
ax.annotate(f"mínimo oficial {MINIMO:.4f}".replace(".", ","), (MINIMO, 1.5),
            xytext=(-6, 0), textcoords="offset points", fontsize=9, color=GRIS,
            va="center", ha="center", rotation=90)

ax.set_yticks([f["y"] for f in filas_d], [f["nom"] for f in filas_d], fontsize=10)
for t, f in zip(ax.get_yticklabels(), filas_d):
    t.set_fontweight("bold" if f["media"] else "normal")
    t.set_color(TINTA if f["media"] else "#4a4f57")

ax.set_xlim(0, X_MAX)
ax.set_ylim(alto_total - 0.4, -0.6)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5],
              ["0", "0,1", "0,2", "0,3", "0,4", "0,5"], fontsize=9.5)
ax.set_xlabel("spellF1", fontsize=10.5, loc="left")
ax.set_title("La recalibración sube el spellF1 en 6 de los 7 partidos externos,\n"
             "pero gana 1,9 pp donde se ajustó y sólo 1,2 pp donde no",
             fontsize=13, fontweight="bold", loc="left", linespacing=1.45, pad=16)
ax.grid(axis="x", color=TENUE, lw=0.8)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(TENUE)
ax.spines["bottom"].set_bounds(0, 0.5)
ax.tick_params(length=0)

ax.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=7.5, mfc="white", mec=GRIS, mew=1.8,
           label="configuración actual"),
    Line2D([], [], marker="o", ls="", ms=7.5, color=AZUL, label="recalibrada, mejora"),
    Line2D([], [], marker="o", ls="", ms=7.5, color=ROJO, label="recalibrada, empeora"),
], loc="upper left", bbox_to_anchor=(0.0, 0.30), frameon=False, fontsize=9.5,
   handletextpad=0.5, borderaxespad=0, labelspacing=0.5)

fig.text(0.016, 0.018,
         "7 partidos de Bundesliga. Las medias promedian el spellF1 de cada partido sin ponderar; "
         "los 3 de validación no se usaron al ajustar.\nEl mínimo oficial lo superan los 7 partidos "
         "con ambas configuraciones.",
         fontsize=8.2, color=GRIS, linespacing=1.5)
fig.subplots_adjust(left=0.175, right=0.995, top=0.825, bottom=0.165)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}04_antes_despues_spellf1.{ext}", dpi=220)
print("guardado:", OUT + "04_antes_despues_spellf1.{png,pdf}")
for f in filas_d:
    print(f"  {f['nom']:14s} {f['a']:.4f} -> {f['r']:.4f}  {(f['r']-f['a'])*100:+.2f} pp")
