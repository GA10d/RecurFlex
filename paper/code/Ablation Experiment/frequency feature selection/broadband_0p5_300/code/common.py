"""Fixed shared model and training-only broadband feature statistics."""
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".plotcache"))
sys.path.insert(0, str(HERE / "main_source"))
from config import Config
from data import training_labels, normalize_targets, inverse_targets
from inference import predict, restore_1000hz, coverage
from model import RecurFlex
from train import objective
from utils import load_json, save_json, setup, sha256


def protocol():
    return load_json(HERE / "protocol.json")


def feature_mask(x, group):
    if group not in protocol()["groups"]:
        raise ValueError(group)
    result = np.array(x, dtype=np.float32, copy=True, order="C")
    if result.shape[:2] != (16, 43):
        raise ValueError("Expected 16 electrodes and 43 model slots.")
    result[:, 40:, :] = 0.
    return result


def tensor_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def fit_statistics(features, truth, reference, group):
    if features.shape[:2] != (16, 40) or truth.shape != (features.shape[-1], 5):
        raise ValueError("Unexpected feature or target dimensions.")
    n = features.shape[-1]
    matrix = features[..., 225:-25].transpose(2, 0, 1).reshape(n-250, -1)
    scaler = RobustScaler(quantile_range=(0.1, 0.9), unit_variance=True).fit(matrix)
    values = truth[225:-25]
    assert np.array_equal(values.min(0), reference["target_min"])
    assert np.array_equal(values.max(0)-values.min(0), reference["target_range"])
    return dict(feature_center=scaler.center_.tolist(), feature_scale=scaler.scale_.tolist(),
                target_min=reference["target_min"], target_range=reference["target_range"],
                selected_electrodes=reference["selected_electrodes"], electrodes=16, frequencies=40,
                model_slots=43, input_gain=protocol()["groups"][group]["input_gain"],
                scaler="RobustScaler quantile_range=(0.1,0.9) unit_variance=True; TRAIN only")


def transform(features, stats):
    c, f, n = features.shape
    if (c, f) != (16, 40):
        raise ValueError("Expected the fixed 16-electrode, 40-feature representation.")
    matrix = features.transpose(2, 0, 1).reshape(n, -1).astype(np.float32, copy=True)
    matrix -= np.asarray(stats["feature_center"], dtype=np.float32)
    matrix /= np.asarray(stats["feature_scale"])
    values = matrix.reshape(n, c, f).transpose(1, 2, 0)
    result = np.zeros((c, 43, n), dtype=np.float32)
    result[:, :40] = values * stats["input_gain"]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite scaled features.")
    return result


def prepare_inputs():
    from features import precompute
    p = protocol()
    for name, digest in p["main_source_sha256"].items():
        assert sha256(HERE / "main_source" / name) == digest, name
    manifest = dict(protocol_sha256=sha256(HERE/"protocol.json"), subjects=[])
    for subject in p["subjects"]:
        reference = p["reference"][str(subject)]
        train = precompute(subject, "train_full")
        test = precompute(subject, "test")
        truth = training_labels(p["data_source"], subject, "train_full")
        info = dict(subject=subject, selected_electrodes=reference["selected_electrodes"], groups={})
        for group in p["groups"]:
            folder = HERE / "inputs" / f"s{subject}" / group
            folder.mkdir(parents=True, exist_ok=True)
            f = np.load(train[group], mmap_mode="r")
            stats = fit_statistics(f, truth, reference, group)
            save_json(folder / "scalers.json", stats)
            np.save(folder / "x_train.npy", transform(f, stats))
            np.save(folder / "y_train.npy", normalize_targets(truth, stats))
            np.save(folder / "x_test.npy", transform(np.load(test[group], mmap_mode="r"), stats))
            info["groups"][group] = dict(
                raw_feature_sha256={"train":sha256(train[group]), "test":sha256(test[group])},
                prepared_sha256={f.name:sha256(f) for f in folder.iterdir() if f.is_file()})
        manifest["subjects"].append(info)
        save_json(HERE / "input_manifest.json", manifest)
        print(f"Prepared broadband inputs for S{subject}", flush=True)
    return manifest
