"""Loading of labeled external data, for training or calibrating the model.

**This package is not on the reproduction path.** RULES.md requires that running the repo
against the match data reproduces the submitted CSV, not approximately but as the very same
file, and that is why `requirements.txt` only has numpy and pandas pinned. The training
dependencies (kloppy) live in `requirements-train.txt` and are imported only from here.
`python -m mtah build` touches nothing in this package.
"""
