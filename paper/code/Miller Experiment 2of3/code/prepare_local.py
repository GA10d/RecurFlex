"""Freeze nine chronological splits and prepare uploads without test labels."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PAPER = EXPERIMENT.parent.parent
MAIN = PAPER / "code/Main Experiment"
LIBRARY = PAPER / "data/Miller_ECoG_Library"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if (HERE / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; preserve the existing experiment.")
    source = HERE / "main_source"
    source.mkdir(parents=True, exist_ok=True)
    names = ("config.py", "data.py", "model.py", "inference.py", "train.py", "metrics.py", "utils.py")
    for name in names:
        shutil.copy2(MAIN / name, source / name)
    upload = EXPERIMENT / "upload"
    labels = EXPERIMENT / "local_test_labels"
    upload.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    records = []
    summary = json.loads((LIBRARY / "metadata/library_summary.json").read_text(encoding="utf-8"))
    for row in summary["fingerflex_subjects"]:
        subject = row["subject"]
        path = LIBRARY / row["path"]
        values = loadmat(path, variable_names=["data", "flex"])
        ecog, flex = values["data"], values["flex"]
        n, channels = ecog.shape
        assert flex.shape == (n, 5) and n % 40 == 0 and channels >= 16
        assert np.isfinite(ecog).all() and np.isfinite(flex).all()
        test_start = (2 * n // 3) // 40 * 40
        development_end = test_start - 2000
        validation_start = int(development_end * .8) // 40 * 40
        training_end = validation_start - 2000
        splits = dict(train_inner=[0, training_end], val_inner=[validation_start, development_end],
                      train_full=[0, development_end], test=[test_start, n],
                      outer_gap=[development_end, test_start], inner_gap=[training_end, validation_start])
        assert all((last-first)%40 == 0 for first,last in splits.values())
        assert np.ptp(flex[:training_end], axis=0).min() > 0
        target = upload / f"{subject}.npz"
        # Only development labels are uploaded; official test truth remains local.
        np.savez_compressed(target, data=ecog, development_flex=flex[:development_end])
        np.save(labels / f"{subject}.npy", flex[test_start:])
        records.append(dict(subject=subject, original_source=str(path), original_source_sha256=sha256(path),
            source_samples=n, channels=channels, splits=splits,
            upload_file=target.name, upload_bytes=target.stat().st_size, upload_sha256=sha256(target),
            local_test_labels=str(labels / f"{subject}.npy"), local_test_labels_sha256=sha256(labels / f"{subject}.npy")))
        print(dict(subject=subject, splits=splits, upload_bytes=target.stat().st_size), flush=True)
    protocol = dict(experiment="RecurFlex on Miller fingerflex: chronological two-thirds development, one-third test",
        subjects=[r["subject"] for r in records], records=records, seed=42, features=44, electrodes=16,
        parameters=6808198, train_window=512, inference_window=2048, inference_stride=2048,
        batch=64, windows_per_epoch=8192, max_validation_epochs=100, patience=25,
        full_budget_rule="clip(round(best_inner_epoch * development_samples / inner_training_samples), 8, 160)",
        optimizer="AdamW", lr=8.42e-5, weight_decay=1e-4, dropout=.1,
        loss="0.5*MSE + 0.5*(1-temporal cosine)",
        outer_split="last one-third test; first two-thirds development minus 2000-sample gap; boundary floored to 40 samples",
        inner_split="last 20% of development validation; preceding inner training with 2000-sample gap",
        split_alignment_samples=40, split_guard_samples=2000,
        channel_selection="same-time training-only maximum absolute correlation; refit on development for final model",
        preprocessing="unchanged Main Experiment raw_features applied separately to each split",
        evaluation="all original 1000Hz test samples; r5 all fingers for literature comparison; r4 retained; validation selection uses r4; equal subject mean",
        test_labels_uploaded=False, workers=2, split_fraction_numerator=2, split_fraction_denominator=3,
        prior_experiment="../Miller Experiment", initialization="fresh seed-42 training_model; no old checkpoints loaded",
        remote_root="/root/autodl-tmp/bci_miller_fingerflex_2of3_20261008",
        python="/root/autodl-tmp/bci_backbones_20261008/venv/bin/python",
        source_sha256={name:sha256(source/name) for name in names})
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (EXPERIMENT / "result").mkdir(exist_ok=True)
    print(dict(protocol_frozen=True, subjects=9, upload_bytes=sum(r["upload_bytes"] for r in records)), flush=True)


if __name__ == "__main__":
    main()
