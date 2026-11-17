# Per-frame possession — More than a Hack 2027, Phase 1

*Labeling every frame of a football match with who has the ball: `H`, `A`, `D` or `X`.*

Determine, for each of the **199,319 frames** of the tactical video of
FC Barcelona – Athletic Club (LaLiga 2024-25, matchday 2), which team has the ball:
`H` (Barcelona), `A` (Athletic), `D` (dead ball), `X` (no answer).

It is scored with `spellF1` against a private full-match ground truth.
Official floor to beat: **`spellF1` 0.2433** (`baseline_smooth50`).

> **For the team:** the project status —what is done, what's left, the problems we ran into
> and what we haven't been able to measure— is in [`STATUS.md`](docs/STATUS.md). The rest of the documents are in
> [`docs/`](docs/), and the measurement scripts and results tables in
> [`analysis/`](analysis/README.md). This README describes only the submission code.

---

## Contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [The data](#the-data)
- [Reproducing the submission](#reproducing-the-submission)
- [Tests](#tests)
- [Architecture](#architecture)
- [Three decisions and why](#three-decisions-and-why)
- [What we do NOT use, and why](#what-we-do-not-use-and-why)
- [Caveats about the public window](#caveats-about-the-public-window)
- [License](#license)

## Quickstart

With the match assets already downloaded (see [The data](#the-data)):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # production deps + pytest, matplotlib
source scripts/env.sh                    # sets MTAH_DATA_DIR (see "The data")
python -m mtah status                    # tells you what is there and what is missing
./scripts/reproduce.sh <team_name>       # -> out/<team_name>.csv
```

## Repository layout

```
mtah-possession/
  src/mtah/           the package: IO, features, models, decoder, CSV (module map in "Architecture")
  tests/              pytest suite
  docs/               project documents: STATUS.md (what is done, what's left, problems we ran into),
                      plain-summary.md, analysis-and-proposal.md, external-data.md, figures/ and
                      archive-es/ (stale Spanish PDFs of the Oct 28 versions)
  analysis/           measurement scripts and results tables (see analysis/README.md)
    handoff-plugin-data/   validation documents, the scripts and the results tables (in data/)
    no-ball-1111/          diagnosis of the frames without the ball (Nov 11-12) and its scripts
  vendor/             official scorer (MIT) + public ground truth + baselines (see vendor/NOTICE.md)
  scripts/            env.sh (data location), reproduce.sh (regenerates the submitted file)
  requirements*.txt   pinned dependencies (production, dev, training)
  pyproject.toml      package metadata
  out/                generated CSVs and the tracking cache (git-ignored)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # production: numpy, pandas, scikit-learn (pinned)
pip install -r requirements-dev.txt      # + pytest, matplotlib
```

Python 3.12 or higher (required by `numpy==2.5.4`). The scikit-learn version has to be
exactly the pinned one: the boundary detector and the team classifier are saved models and
refuse to load with any other.

## The data

The match assets **are not in this repository and cannot be**: they weigh ~5 GB and are
not redistributable under the competition's terms. Download them from the organizers' Drive
and unzip them keeping the structure they come with:

```
<root>/tactical/BAR-ATH_tactical.mp4
<root>/tactical/BAR-ATH_tactical_atd.xml
<root>/tactical/BAR-ATH_tactical_atd.txt
<root>/tactical/BAR-ATH_tactical_homography.csv
<root>/tactical/BAR-ATH_tactical_smart_tagging.xml
<root>/broadcast/...
```

Point `MTAH_DATA_DIR` at `<root>`:

```bash
source scripts/env.sh                    # uses ~/Documents/Hackaton/data by default
python -m mtah status                    # tells you what is there and what is missing
```

> **Don't download the whole Drive folder as a ZIP.** Drive splits it into ~2 GB chunks
> and it is easy to end up with a loose part. Download the files one by one.

## Reproducing the submission

```bash
source scripts/env.sh
./scripts/reproduce.sh <team_name>       # -> out/<team_name>.csv
```

It works on Linux, macOS and Windows (Git Bash). It requires Python ≥ 3.12 and stops if it
is not; `PYTHON=path/to/python ./scripts/reproduce.sh ...` picks the interpreter (on Windows,
the venv's: `.venv/Scripts/python.exe`). Checked on Nov 16 from a fresh clone with a newly
created venv on Windows 11 and on Ubuntu 24.04: both give the same file byte for byte
(sha256 in `STATUS.md`).

Equivalent, step by step:

```bash
python -m mtah build --out out/team.csv      # tracking -> features -> decoding -> csv
python -m mtah validate out/team.csv         # the contract, before uploading
python -m mtah evaluate out/team.csv         # official scorer, 5-min public window
```

The tracking parse is cached in `out/cache/`. Delete it if you change the bundle, or use
`--no-cache`.

## Tests

```bash
python -m pytest -q
```

They cover the file contract, the decoder, the end-to-end pipeline on synthetic tracking,
and a regression that checks that the vendored scorer reproduces the published numbers of
the four official baselines.

---

## Architecture

```
src/mtah/
  constants.py          frames, scored blocks, pitch, labels
  paths.py              asset resolution + MTAH_DATA_DIR
  config.py             ALL hyperparameters, with their justification alongside
  io/atd.py             tracking (46 slots + ball) -> numpy, cached
  io/homography.py      3x3 matrix per frame; pitch <-> pixel
  io/smart_tagging.py   tactical phases and restarts -> frame intervals
  features/geometry.py  distances, control, activity, team shape
  features/boundaries.py what changes between just before and just after each frame
  features/team_signals.py shape and movement of each team along its attacking axis, without the ball
  models/emissions.py   features -> cost per frame and state
  models/boundary_detector.{py,joblib,json}  boundary score, trained on the IDSSE
  models/team_classifier.{py,joblib,json}    who has possession without looking at the ball, ditto
  decode/viterbi.py     cost -> segments. The core of the challenge
  pipeline.py           the full chain
  submission.py         build, write and validate the CSV
  evaluate.py           vendored official scorer
vendor/                 official scorer (MIT) + public ground truth + baselines
```

The decomposition is `P(H) = P(live) · P(home | live)` and `P(D) = P(dead)`, because the
two halves rely on different signals: **live/dead** comes from player kinematics and does
not need the ball, while **ownership** comes from ball-player geometry and does need it.
With the ball absent in 41% of the scored frames, mixing them into a single classifier
throws away the half that keeps working.

On top of that, a **boundary detector**: gradient boosting that looks at what changes
between a short window before and another one after each frame, and returns a "possession
changes here" score. The decoder makes every change more expensive (×2) and makes each
frame's changes cheaper in proportion to the score (×(1 − 0.95·score)): it cuts mostly where
the detector sees a boundary. It was trained on the 7 IDSSE matches degraded like the ATD;
measured leaving each match out, with the weight chosen without seeing it, it gives
**+1.17 pp of `spellF1`** over the model without it (see `STATUS.md`). It is regenerated
with `python -m mtah.training.boundaries export`, which needs `requirements-train.txt` and
downloads the IDSSE; the submission only loads the saved model.

And Metrica's **Smart Tagging** as team evidence: the phases that say who has the ball
(BUILD UP, PROGRESSION, ...; the DEFENDING IN * ones name the opponent) add ±0.5 to the home
logit where they exist, 40% of the frames. The IDSSE has no tagging, so the weight was chosen
with a simulated one over its labels, calibrated by its agreement with our emission in this
match (`training/team_tags.py`, see `STATUS.md`): +0.5 to +0.9 pp estimated.

Since Nov 12, and in `main` since Nov 15, a **team classifier that works without the ball**:
gradient boosting that reads who owns the possession from the shape and movement of the two
teams (width, defensive and attacking lines, speed toward the opponent's goal, sprints), each
measured along its own attacking direction, and adds its logit ×2 to the home logit. It
exists because of the 41% of frames without the ball, where the ball-player geometry has
nothing to read. Trained on the 7 degraded IDSSE matches; measured leaving each match out,
with the weight chosen without seeing it, it gives **+16.1 pp of `spellF1`** over three
seeds. It is an optimistic bound: the degraded IDSSE keeps the players when the ball is
missing and this match does not (see `STATUS.md`). With `team_classifier_weight = 0` it is
not loaded and the Nov 8 submission comes out byte for byte. It is regenerated with
`python -m mtah.training.team_classifier export`.

## Three decisions and why

Measured with the official scorer, degrading the public ground truth in a controlled way.
The numbers are in the docstrings of the corresponding modules.

**1. Decoding is offline and centered.** +1 frame of delay is free (`spellF1` 1.0000);
+25 frames (1 s) drops it to 0.7765 and +50 frames (2 s) to 0.5882. Any causal filter is
not viable.

**2. Viterbi instead of majority vote.** Starting from the same raw label as the official
baseline, on the public window:

| | spellF1 | macroF1 | possessions |
|---|---|---|---|
| majority vote, `smooth50` | 0.3256 | 0.6207 | 26 |
| majority vote, `smooth125` | 0.4242 | 0.5812 | 16 |
| **Viterbi** `switch_cost=40` | **0.4000** | **0.6686** | 16 |

It beats `smooth50` on both metrics at once. The reason is structural: a window controls
the number of segments indirectly, and to avoid fragmenting a long possession it has to
erase the short ones; Viterbi penalizes transitions directly and can do both. In the public
window, **6 of the 17 possessions last less than 4 s** and the shortest 0.76 s, so erasing
the short ones is expensive: losing those under 2 s costs 0.17 of `spellF1` and those under
5 s, 0.44.

**3. We never write `X` on a scored frame.** The scorer forward-fills abstentions with the
last label of the same half *before* matching possessions, so in `spellF1` an `X` is
literally "repeat my previous answer" — but `macroF1`, the tiebreaker, counts it as a miss:

```
with 20% of X               spellF1 0.3610   macroF1 0.5438
same but filled in by hand  spellF1 0.3610   macroF1 0.6192
```

Identical on the ranking metric, 0.075 worse on the tiebreaker. It is strictly dominated.

## What we do NOT use, and why

- **The homography's `reprojection_error` as a dead-ball detector.** Measured on this
  match: median 0.231 in play vs 0.395 at halftime, p90 in play 0.308. It separates
  "camera pointing at the pitch" from "camera on the crowd", but it does not separate live
  play from a stoppage.
- **The broadcast video and its tracking.** It is on its own timeline, with cuts and
  replays, without a correspondence table, and a third of its frames have no usable
  homography. Mapping it back costs more than it brings.

## Caveats about the public window

It is 5 minutes and 17 possessions. It is good for checking that the pipeline is
**correct**, never for choosing between two versions of it: `baseline_smooth125` wins there
and loses on the full match. One missed possession moves `spellF1` about 6 points in that
window. Differences below ~0.04 in `spellF1` and ~0.01 in `macroF1` are noise.

## License

Our own code is under MIT. `vendor/` is a verbatim copy of the competition's official
repository (MIT, see `vendor/NOTICE.md`). The match data is not covered by either and is
not redistributed.

`src/mtah/models/boundary_detector.joblib` and `src/mtah/models/team_classifier.joblib` are
derivative works of the IDSSE dataset —
Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of spatiotemporal and event
data in elite soccer". *Sci Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y — published
under **CC-BY 4.0** with the authorization of the **Deutsche Fußball Liga (DFL)**. Data:
doi:10.6084/m9.figshare.28196177. The repository contains no IDSSE data, only the two
models trained on it.
