"""Read official test labels only after all predictions have been saved."""
import csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from config import FINGER_NAMES
from metrics import score
from utils import load_json, save_json, sha256


def evaluate(manifest_path, labels_dir, output_dir):
    manifest = load_json(manifest_path)
    rows = []
    for item in manifest["predictions"]:
        if sha256(item["prediction"]) != item["prediction_sha256"]:
            raise ValueError("Prediction changed after generation.")
    for item in manifest["predictions"]:
        subject = item["subject"]
        path = Path(labels_dir) / f"sub{subject}_testlabels.mat"
        labels = loadmat(path, variable_names=["test_dg"])
        if "test_dg" not in labels:
            raise ValueError(f"Expected variable test_dg in {path}")
        prediction = np.load(item["prediction"])["prediction_1000hz"]
        if prediction.shape != (200000, 5) or labels["test_dg"].shape != prediction.shape:
            raise ValueError("Official scoring requires all 200000 original samples and five outputs.")
        rows.append(dict(subject=subject, seed=42, label_sha256=sha256(path), **score(labels["test_dg"], prediction)))
    summary = dict(model=manifest.get("model"), features=manifest.get("features"), seed=42,
                   subjects=rows, means={k:float(np.mean([r[k] for r in rows]))
                   for k in ("r4", "r5", "mae4", "mse4")},
                   prediction_manifest_sha256=sha256(manifest_path),
                   metric="Full original 1000 Hz, all 200000 samples, four fingers excluding ring")
    output_dir = Path(output_dir)
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "subject_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["subject", "seed", *FINGER_NAMES, "r4", "r5", "mae4", "mse4", "samples"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r[k] for k in ("subject", "seed", "r4", "r5", "mae4", "mse4", "samples")},
                             **dict(zip(FINGER_NAMES, r["r_fingers"]))})
    return summary
