# Full-test trajectory figures

The [repository README](../../../README.md) displays S1/S2/S3 figures from the complete official 200-second test recordings. Blue denotes recorded glove labels; orange denotes predictions. Ring is an auxiliary output excluded from the official four-finger metric.

Scores use every original 1,000 Hz test sample. The figure display directly takes every tenth sample, with no smoothing, time shift, or amplitude calibration.

To redraw from saved predictions and separately obtained original test labels:

```bash
python plot_prediction_vs_label.py --labels-dir /path/to/BCICIV_4_mat
```

The included CSV lists per-finger scores. Historical figure provenance retains the original source hashes and filenames; README images are identical copies with simpler filenames under `assets/`.
