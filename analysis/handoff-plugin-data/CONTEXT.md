# Handoff to a session with the Data plugin — FC Medallo

This package exists to use claude.ai's **Data** plugin (`/analyze`, `/create-viz`,
`/build-dashboard`, `/validate`) on our model's results. The plugin is enabled
in the claude.ai account but not in the Claude Code session where the work was done.

## ⚠️ First things first: what can be uploaded and what cannot

The competition rules forbid uploading **the video, the tracking, the homography, the Smart
Tagging or any excerpt** of the match to any cloud LLM. That is why this package
**contains none of that** and in the new session **nothing from the `data/` folder
should be uploaded**.

What is here, and is safe:
- **Outputs of our model** (H/A/D labels that our program produces).
- **The public ground truth** of 5 minutes, which the organizers publish on GitHub.
- **Aggregate statistics** measured on **other matches** (7 from the Bundesliga, public).

## What's in `data/`

| file | what it is |
|---|---|
| `scores_por_partido.csv` | spellF1 and macroF1 per external match, for the **current** config (the one before Oct 28, reconstructed) and the **recalibrated** one (today's `config.py`). Column `grupo`: 4 *fit* matches, 3 *validation* matches that were not touched when tuning. Regenerated on Nov 2 with `regenera_datos.py` |
| `diagnostico_errores.csv` | four tables (long format) for each config: error by how long the ball has gone unseen, error by distance to the real transition, confusion matrix, % of possessions recovered by duration |
| `recalibracion.json` | the recalibrated config from Oct 28 (historical: `activity_smooth` went back to 50 on Oct 31), the measured per-class medians, and that day's scores |
| `ventana_publica_5min.csv` | 7,500 rows: public ground truth and our label, per frame |
| `partido_completo_1hz.csv` | our label at 1 sample per second, both halves (no ground truth: it's private) |
| `clasificador_equipo.json` | (Nov 15) the 21 leave-one-match-out runs of the team classifier: spellF1 without and with it, per match and seed |
| `clasificador_equipo_resumen.json` | (Nov 15) its summary per seed and on average, plus the measurement with ball gaps like those of the real match |
| `lectura_publica.json` | (Nov 15) spellF1 of every preserved submission on the 5 public minutes. A reading, never a criterion |

Since Nov 15, `scores_por_partido.csv`, `diagnostico_errores.csv` and `agregados_por_config.json`
carry two more configurations, `detector` and `equipo` (today's), measured with models that did
not see the match being scored (`regenera_equipo.py`). The two time views are the
submission that `main` generates (sha256 `a43c52fb…`).

And in `figures/` the three charts already made by hand with matplotlib, in case they are useful
as a reference.

## The numbers that matter

- Ranking metric: **spellF1** (possessions correctly identified by temporal overlap, not frames).
- Official minimum to beat: **0.2433**.
- Our model over the 7 external matches, with today's `config.py` (Oct 31): **0.3965**
  (fit 0.3878, validation 0.4081). The config prior to Oct 28 gave 0.3767 (validation 0.3837).
- Overfitting? The defensible sentence, measured over the 35 possible splits of 4 fit / 3
  validation: **recalibration gains +1.7 pp (95% CI [+1.4, +2.1]) on matches that were not used
  to tune it; on the fit matches it gains +3.3 pp, so roughly half of the gain
  is overfitting.** It is not valid to say "it did not overfit". Details in `CROSS-VALIDATION.md`.
- What fails most: saying which team the ball belongs to (~20% of live frames) and possessions
  shorter than 5 seconds (about 1 in 10 is recovered).

---

## Prompt to paste into the new session

```
Context: we are competing in "More than a Hack 2027". We have to label every frame of a football
match as H (ball with the home team), A (away) or D (dead ball). We are scored with
spellF1: each possession (a continuous stretch of one team having the ball) counts once and is
a hit if it overlaps enough with the real one. The minimum to beat is 0.2433.

I'm attaching the results of our model evaluated on 7 external matches (Bundesliga), with two
configurations: "current" and "recalibrated". Four matches were used to tune and three to
validate. The files are described in CONTEXT.md. They are outputs of our model and aggregates
over other matches: they contain no data from the competition match.

I want three things, in this order:

1. /validate — review the analysis before we share it with the team. In particular:
   is the comparison current vs recalibrated fair? Is there any bias in how the
   errors are aggregated (for example, the categories with few frames)? Is the improvement from
   0.3767 to 0.3965 believable given that the 3 validation matches go 0.3837 → 0.4081?

2. /build-dashboard — an HTML dashboard for the team with: (a) spellF1 per match, current vs
   recalibrated, marking fit vs validation; (b) the four diagnostic tables as charts,
   with a config selector; (c) the timeline of the 5 public minutes (ground truth
   on top, model below, errors marked); (d) the full match at 1 Hz as a color strip.
   Colors: H blue, A red, D gray.

3. /create-viz — a clean "before/after" figure for the technical document: spellF1 over the 7
   matches with the two configs, separating fit and validation, with the 0.2433 minimum line.

Don't invent data that isn't in the files. If something can't be answered with what's attached, say so.
```
