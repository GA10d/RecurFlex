# Nine-subject chronological experiment

This is the experiment source snapshot for the nine-subject Miller fingerflex evaluation. `protocol.json` preserves the original per-record indices and hashes. `main_source/` is the frozen BCI implementation used by this experiment.

- `prepare_local.py`: constructs chronological splits and archives containing development labels only.
- `prepare_server.py`: prepares features and training-only statistics from the archives.
- `worker.py`: validates, refits, and predicts one subject.
- `run.py`: orchestrates the nine jobs and freezes predictions before scoring.
- `score_local.py`: scores frozen outputs against separately retained local test labels.

The original pipeline separates preparation, training, and local scoring. It expects the Miller library metadata, source recordings, prepared archives/manifests, and collected server outputs in the original relative directory layout. These generated inputs, original datasets, and nine-subject weights are not included. Server login and transfer utilities are excluded. This source snapshot is not a one-command standalone reproduction package.

The first two thirds are development data and the last third is the test, with a 2 s gap. Inner validation selects training duration; each final model is freshly initialized with seed 42. See [saved results](../../../result/miller_fingerflex_9subjects_2of3/test_results.json) and the [paper](../../../overleaf/main.tex).
