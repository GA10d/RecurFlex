"""Shared fixed inputs and feature masks for the three paired training arms."""
import hashlib
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "main_source"))
from config import Config
from data import fit_statistics, training_labels, transform, normalize_targets, inverse_targets
from inference import predict, restore_1000hz, coverage
from model import RecurFlex
from train import objective
from utils import load_json, save_json, setup, sha256


def protocol():
    return load_json(HERE / "protocol.json")


def feature_mask(x, group):
    active = protocol()["groups"][group]["active_features"]
    excluded = sorted(set(range(43)) - set(active))
    result = np.array(x, dtype=np.float32, copy=True, order="C")
    result[:, excluded, :] = 0.
    return result


def tensor_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def prepare_inputs():
    p = protocol()
    for name, digest in p["main_source_sha256"].items():
        if sha256(HERE / "main_source" / name) != digest:
            raise ValueError(f"Main source changed: {name}")
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)
    metadata = dict(subjects=[], protocol_sha256=sha256(HERE / "protocol.json"))
    for subject in p["subjects"]:
        source = Path(p["data_source"]) / f"sub{subject}_comp.mat"
        if sha256(source) != p["dataset_sha256"][str(subject)]:
            raise ValueError(f"Dataset differs from local official file: S{subject}")
        feature_files = {}
        for segment, samples in (("train_full", 40000), ("test", 20000)):
            folder = Path(p["feature_source"]) / f"s{subject}_{segment}"
            m = load_json(folder / "metadata.json")
            assert m["source_sha256"] == p["dataset_sha256"][str(subject)]
            assert m["n_cycles"] == 7 and m["mne"] == "1.8.0"
            assert m["signed_bands"] == [[0.5, 4], [4, 13], [13, 30]]
            assert m["indices"] == [0, samples * 10]
            assert np.allclose(m["frequencies"], np.logspace(np.log10(40), np.log10(300), 40), rtol=1e-12)
            path = folder / "power.npy"
            f = np.load(path, mmap_mode="r")
            assert f.shape[1:] == (43, samples) and f.dtype == np.float32 and np.isfinite(f).all()
            feature_files[segment] = path
        training = np.load(feature_files["train_full"], mmap_mode="r")
        truth = training_labels(p["data_source"], subject, "train_full")
        stats = fit_statistics(training, truth, 16)
        folder = inputs / f"s{subject}"
        folder.mkdir(exist_ok=True)
        save_json(folder / "scalers.json", stats)
        np.save(folder / "x_train.npy", transform(training, stats))
        np.save(folder / "y_train.npy", normalize_targets(truth, stats))
        test = np.load(feature_files["test"], mmap_mode="r")
        np.save(folder / "x_test.npy", transform(test, stats))
        metadata["subjects"].append(dict(subject=subject, electrodes=stats["selected_electrodes"],
                    original_cache_sha256={k:sha256(v) for k,v in feature_files.items()},
                    prepared_sha256={f.name:sha256(f) for f in folder.iterdir() if f.is_file()}))
        print(f"Prepared fixed train/test inputs for S{subject}", flush=True)
    save_json(HERE / "input_manifest.json", metadata)
    return metadata
