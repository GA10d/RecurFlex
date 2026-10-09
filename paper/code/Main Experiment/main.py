"""CLI for preparation, training, prediction and official evaluation."""
import argparse
from pathlib import Path

from config import CACHE_DIR, DATA_DIR, HERE
from data import cached_features
from evaluate import evaluate
from inference import infer_subject
from train import fit
from utils import device_name, save_json, setup, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "predict", "test", "all"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--weights-dir", type=Path, default=HERE / "artifacts" / "pretrained")
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs" / "main")
    parser.add_argument("--phase", choices=("full", "validation"), default="full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Each subject may appear only once.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("Epoch count must be positive.")
    if args.stage == "all" and args.phase != "full":
        parser.error("The all stage uses the full labeled training record.")
    setup()
    device = device_name(args.device)
    if args.stage == "prepare":
        for subject in args.subjects:
            for segment in ("train_full", "train_inner", "val_inner", "test"):
                features = cached_features(args.data_dir, args.cache_dir, subject, segment)
                print(f"S{subject} {segment}: {features.shape}", flush=True)
        return
    weights_dir = args.weights_dir
    if args.stage in ("train", "all"):
        weights_dir = args.run_dir / "weights"
        for subject in args.subjects:
            fit(subject, args.data_dir, args.cache_dir, weights_dir, device, args.phase, args.epochs)
        if args.stage == "train":
            return
    predictions = []
    for subject in args.subjects:
        item = infer_subject(subject, args.data_dir, args.cache_dir, weights_dir, args.run_dir / "predictions", device)
        predictions.append(item)
        print(f"S{subject} prediction saved: {item['shape_1000hz']}", flush=True)
    manifest = args.run_dir / "prediction_manifest.json"
    save_json(manifest, dict(seed=42, features=44, model="RecurFlex high Morlet + signed delta/theta/alpha/beta", predictions=predictions,
              source_sha256={p.name:sha256(p) for p in HERE.glob("*.py")}))
    if args.stage in ("test", "all"):
        summary = evaluate(manifest, args.labels_dir or args.data_dir, args.run_dir / "results")
        print(summary["means"], flush=True)


if __name__ == "__main__":
    main()
