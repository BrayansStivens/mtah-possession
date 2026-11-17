# Third-party code

`evaluation/` and `games.py` are a verbatim copy of the competition's official repository:

    https://github.com/metrica-sports/more-than-a-hack-2027

MIT license, included in `LICENSE.metrica`. Copyright Metrica Sports and FC Barcelona
Innovation Hub.

It is vendored unmodified so that this repository can be scored locally with **the same
scorer** that decides the competition, without needing the official repo cloned alongside.
Do not edit it: if something needs to change, the bug is in our code.

`evaluation/gt/` contains the **public** ground truth (5 minutes), which is distributed with that
repository. The match assets (video, tracking, homography) are NOT here and cannot be.
