# Verified results: phase3-calibration

A Phase 3B chronological calibration experiment. `folds.csv` holds every fitted group in every fold with the fit that produced it and the hash of the artifact it came from; `paired-deviance.csv` is the primary forecast estimand; `policy.csv` compares each candidate against identity's replay of the same strategy; `decision-changes.csv` and `advice-changes.csv` count assignment and static hold disagreements. Individual picks are stored as deterministic `picks.csv.gz`.

The generated [report](CALIBRATION.md) states facts, methods and provenance. It selects no candidate: the promotion rule is in the resolved configuration here, and applying it is a separate, dated authoring step. Production constants are unchanged and nothing in this directory deploys a calibrated model.

Published with `uv run python experiments/publish.py phase3-calibration` after `uv run python experiments/verify.py phase3-calibration` and a benchmark resume check. Fold artifacts, out-of-fold surfaces and the frozen database remain in the ignored `data/experiments/phase3-calibration` directory. Matching source implementation: `bce82abeea3dfb43b8c3319a8a742d07939bcbff`.
