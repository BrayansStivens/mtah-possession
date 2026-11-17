# FC Medallo — More than a Hack 2027 · working material

This folder is the **work**, not the submission: documents, figures, measurement scripts and
the results tables on which the decisions were based. The code that is submitted
is at the root of this same repository (`src/`, `tests/`, `vendor/`). The summary of
where everything is, what's left and what problems we ran into is in `../docs/STATUS.md`.

(This folder was copied on Nov 3 from the working repository `mtah-2027-analisis`, which
remains as a historical record.)
The `.pdf` files are the Oct 28 versions of the `.md` files of the same name, kept in `../docs/archive-es/`; the `.md` is the current one.

## Data notice

The competition terms forbid redistributing the match assets:
video, ATD tracking, homography and Smart Tagging. **None of that is here** —
`.gitignore` excludes `data/` and `staging-*/`, and they must not be un-ignored.

What is here, and is safe to share:

- outputs of our own model (H/A/D labels),
- the **public** 5-minute ground truth, published by the organizers,
- aggregate statistics measured on 7 public Bundesliga matches.

## The challenge, in three lines

Label every frame of the match as **H** (home possession), **A** (away)
or **D** (dead ball). The metric is **spellF1**, which compares whole possessions
by temporal overlap, not individual frames. The official minimum to beat is
**0.2433**. Submission: **November 17, 2026**.

## Where everything is

| Path | What it is |
|---|---|
| `../docs/plain-summary.md` | The challenge explained without jargon. Start here. |
| `../docs/analysis-and-proposal.md` | Analysis of the challenge and technical proposal. |
| `../docs/external-data.md` | Research on external data for a supervised model. |
| `handoff-plugin-data/VALIDATION.md` | Validation of the current model, finding by finding. |
| `handoff-plugin-data/CROSS-VALIDATION.md` | The 35 splits. **Corrects `VALIDATION.md`** on everything related to overfitting. |
| `handoff-plugin-data/datos/` | Results tables in JSON and CSV. |
| `handoff-plugin-data/*.py` | The scripts that produce those tables. |
| `handoff-plugin-data/regenera_datos.py` | Regenerates the files that feed the team dashboard with today's `config.py` (it first checks that it reproduces the Oct 28 ones). Run after every change to `config.py`. |
| `handoff-plugin-data/regenera_equipo.py` | Adds to those files the two configurations that carry models trained on the IDSSE (boundary detector and team classifier), each match scored with models that did not see it. It first checks that it reproduces `datos/equipo_sin_balon_lomo.json`. Run after `regenera_datos.py`. |
| `handoff-plugin-data/datos/equipo_sin_balon_jugadores.json` | Nov 16 stress test: the team classifier with missing players where the ball is missing, as in the ATD. It comes from `python -m mtah.training.missing_players` (a copy of `out/missing_players.json`). Summary in `STATUS.md` §2. |
| `sin-balon-1111/SUMMARY.md` | Diagnosis of the frames without the ball (Nov 11 and 12) and the ideas that came out of it; its scripts and aggregate tables are alongside. The team classifier comes from here (branch `equipo-sin-balon`, in `main` since Nov 15). |
| `../docs/figures/` | Charts for the documents. |

## Measurement status (Oct 31, 2026)

The team dashboard (private claude.ai artifact, "Posesion FC Medallo")
was regenerated on Nov 2 with these same numbers and on Nov 15 with those of the
boundary detector and the team classifier (`regenera_equipo.py`); its files are those in
`handoff-plugin-data/datos/`.

On the 7 external matches, with `config.py` exactly as it is today (degradation
seed 0):

| | spellF1 |
|---|---|
| fit set (4 matches) | 0.3878 |
| validation set (3 matches) | 0.4081 |
| all 7 | **0.3965** |

The earlier numbers (0.3905 / 0.3957 / 0.3927) were from before Oct 31, when
`activity_smooth` was 12. Besides, the file never actually produced the 0.3957: see the
corresponding section of `CROSS-VALIDATION.md`.

The recalibration procedure, run over the 35 possible splits of
4 fit / 3 validation:

- **+1.71 pp** on average on the matches that were not used to tune it
  (95% CI [+1.37, +2.05], improvement in 32/35),
- **+3.27 pp** on the fit matches (35/35).

That is: it transfers, but roughly **half the gain is
overfitting**. That is the defensible sentence; it's not valid to say "it didn't overfit".

## The expanded grid: tried and discarded (Oct 31)

Seven parameters were pinned to an edge of their value list, so the grid was
widened (64 -> 82 values) and the 35 splits were repeated with the
same descent. Comparing in pairs, split by split:

| | fit set (4) | validation set (3) |
|---|---|---|
| original grid | +3.27 pp | +1.71 pp |
| expanded | +3.89 pp | +1.58 pp |
| paired difference | **+0.62 pp** [+0.34, +0.90] | **-0.13 pp** [-0.52, +0.26] |

It fits the matches it sees better and does **not** improve those it doesn't see: it wins in 14 of 35
splits, fewer than half. Transfer drops from 52% to 41%. So
**the grid is not widened** in `recalibrate.py`, and the values are not changed to
what the expanded arm picks. Details in `CROSS-VALIDATION.md`.

No parameter is pinned to an edge any more, so the diagnosis was correct: the
optimum on the fit set was outside the old grid. What doesn't hold up is
that reaching it serves any purpose.

## Next

With 7 fit matches there is no signal to tell apart a good part of the
hyperparameters, and searching more won't fix that: more matches will. The route
that remains open is the supervised model in `external-data.md`, not continuing to
fine-tune this one.

## Reproducing the measurements

The scripts in `handoff-plugin-data/` import `mtah` from the submission repo and
read the cache in `mtah-possession/out/`, which is not versioned. Both
are needed locally:

```sh
git clone https://github.com/BrayansStivens/mtah-possession.git
```
