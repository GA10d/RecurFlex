"""Morlet power, signed bands, training-only statistics and matched windows."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import RobustScaler

from config import SCORED_FINGERS
from utils import load_json, save_json, sha256

FREQUENCIES = np.logspace(np.log10(40), np.log10(300), 40)
SIGNED_BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30))
SEGMENTS = {"train_full": ("train_data", 0, 400000),
            "train_inner": ("train_data", 0, 319000),
            "val_inner": ("train_data", 321000, 400000),
            "test": ("test_data", 0, 200000)}


def raw_features(raw):
    import mne
    raw = np.asarray(raw)
    if raw.ndim != 2 or len(raw) < 3000 or len(raw) % 10 or not np.isfinite(raw).all():
        raise ValueError("Expected finite 1000 Hz ECoG with at least 3000 points and length divisible by 10.")
    x = raw.T.astype(np.float64)
    scale = x.std(1, keepdims=True)
    if (scale <= 0).any():
        raise ValueError("Constant ECoG electrode.")
    x = (x - x.mean(1, keepdims=True)) / scale
    x -= np.median(x, axis=0, keepdims=True)
    filtered = mne.filter.filter_data(x, 1000., 40., 300., verbose=False)
    filtered = mne.filter.notch_filter(filtered, 1000., np.arange(60, 500, 60), verbose=False)
    result = np.empty((len(x), 44, len(raw) // 10), dtype=np.float32)
    for first in range(0, len(x), 4):
        last = min(first + 4, len(x))
        result[first:last, :40] = mne.time_frequency.tfr_array_morlet(
            filtered[None, first:last], 1000., FREQUENCIES, n_cycles=7,
            zero_mean=False, use_fft=True, output="power", decim=10, n_jobs=1, verbose=False)[0]
    for index, band in enumerate(SIGNED_BANDS):
        sos = butter(4, band, btype="bandpass", fs=1000, output="sos")
        result[:, 40 + index] = sosfiltfilt(sos, x, axis=-1)[:, ::10]
    return result


def labels100(y):
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 5 or len(y) % 40 or not np.isfinite(y).all():
        raise ValueError("Expected original (N, 5) glove labels, with N divisible by 40.")
    support = y[::40]
    support = np.concatenate([support, support[-1:]], axis=0)
    return interp1d(np.arange(len(support)) * 4, support, axis=0, kind="cubic")(
        np.arange((len(support) - 1) * 4))


def training_labels(data_dir, subject, segment):
    if segment not in ("train_full", "train_inner", "val_inner"):
        raise ValueError("Training labels are available only for labeled training segments.")
    y = loadmat(Path(data_dir) / f"sub{subject}_comp.mat", variable_names=["train_dg"])["train_dg"]
    _, first, last = SEGMENTS[segment]
    return labels100(y[first:last])


def cached_features(data_dir, cache_dir, subject, segment):
    import mne
    if segment not in SEGMENTS:
        raise ValueError(segment)
    source = Path(data_dir) / f"sub{subject}_comp.mat"
    folder = Path(cache_dir) / f"s{subject}_{segment}"
    metadata = dict(source_sha256=sha256(source), segment=segment, mne=mne.__version__,
                    frequencies=FREQUENCIES.tolist(), n_cycles=7, signed_bands=[list(b) for b in SIGNED_BANDS],
                    features=44, recipe="high_morlet40_signed4_v1")
    path = folder / "features.npy"
    if path.exists() and (folder / "metadata.json").exists():
        if load_json(folder / "metadata.json") != metadata:
            raise ValueError(f"Cache signature changed: {folder}. Use a different cache directory.")
        return np.load(path, mmap_mode="r")
    key, first, last = SEGMENTS[segment]
    raw = loadmat(source, variable_names=[key])[key]
    if len(raw) != (200000 if key == "test_data" else 400000):
        raise ValueError(f"Unexpected official record length: {source} / {key}")
    features = raw_features(raw[first:last])
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "features.partial.npy"
    np.save(tmp, features)
    tmp.replace(path)
    save_json(folder / "metadata.json", metadata)
    return np.load(path, mmap_mode="r")


def fit_statistics(features, truth, channel_count=16, selected_electrodes=None):
    c, f, n = features.shape
    if f != 44 or truth.shape != (n, 5) or n <= 800 or c < channel_count:
        raise ValueError("Invalid training feature/target dimensions.")
    values = truth[225:-25]
    target_range = values.max(0) - values.min(0)
    if (target_range <= 0).any():
        raise ValueError("Constant finger target in training data.")
    selection = "fixed training-derived full-record electrode set"
    scores = None
    if selected_electrodes is None:
        # A separate validation fit selects electrodes using only its inner training split.
        a = features[..., 225:-25:5].transpose(2, 0, 1).reshape(-1, c * f).astype(np.float64)
        b = truth[225:-25:5].astype(np.float64)
        a -= a.mean(0)
        b -= b.mean(0)
        corr = (a.T @ b) / np.maximum(np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)[None]), 1e-12)
        scores = np.max(np.abs(corr.reshape(c, f, 5)[..., SCORED_FINGERS]), axis=(1, 2))
        selected_electrodes = np.sort(np.argsort(scores)[-channel_count:]).tolist()
        selection = "same-time correlation on supplied training split only"
    chosen = np.asarray(selected_electrodes, dtype=int)
    if len(chosen) != channel_count or len(set(chosen)) != channel_count or (chosen < 0).any() or (chosen >= c).any():
        raise ValueError("Invalid selected electrode indices.")
    matrix = features[chosen, :, 225:-25].transpose(2, 0, 1).reshape(n - 250, -1)
    scaler = RobustScaler(quantile_range=(0.1, 0.9), unit_variance=True).fit(matrix)
    stats = dict(feature_center=scaler.center_.tolist(), feature_scale=scaler.scale_.tolist(),
                 feature_gains=[1.] * 40 + [200.] * 4,
                 target_min=values.min(0).tolist(), target_range=target_range.tolist(),
                 selected_electrodes=chosen.tolist(), electrodes=channel_count, frequencies=f,
                 model_slots=44, slot_map=list(range(44)), electrode_selection=selection)
    if scores is not None:
        stats["electrode_training_scores"] = scores.tolist()
    return stats


def transform(features, stats):
    c, f, n = features.shape
    if f != 44 or stats["frequencies"] != 44 or stats.get("model_slots", 44) != 44:
        raise ValueError("The main experiment requires all 44 feature slots.")
    # The supplied scalers were fitted after selecting 16 raw electrodes.
    # Reference subtraction still uses every raw electrode in raw_features().
    if c != stats["electrodes"]:
        features = features[stats["selected_electrodes"]]
        c = features.shape[0]
    if len(stats["feature_center"]) != c * f or stats.get("slot_map", list(range(44))) != list(range(44)):
        raise ValueError("Electrode count/order or feature count differs from fitted statistics.")
    matrix = features.transpose(2, 0, 1).reshape(n, -1).astype(np.float32, copy=True)
    matrix -= np.asarray(stats["feature_center"], dtype=np.float32)
    matrix /= np.asarray(stats["feature_scale"])
    x = matrix.reshape(n, c, f).transpose(1, 2, 0)
    x[:, 40:] *= 200.
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite scaled input.")
    return np.ascontiguousarray(x)


def normalize_targets(truth, stats):
    return ((truth - np.array(stats["target_min"])) / np.array(stats["target_range"])).astype(np.float32)


def inverse_targets(prediction, stats):
    return prediction * np.array(stats["target_range"]) + np.array(stats["target_min"])


def sample_batch(x, y, batch_size, generator, window=512):
    import torch
    if len(x) != len(y):
        raise ValueError("Features and labels must share the same time axis.")
    high = len(x) - window - 25 + 1
    if high <= 225:
        raise ValueError("Training record is too short for interior windows.")
    starts = torch.randint(225, high, (batch_size,), generator=generator, device=x.device)
    ix = starts[:, None] + torch.arange(window, device=x.device)[None]
    return x[ix].permute(0, 2, 3, 1).contiguous(), y[ix].transpose(1, 2).contiguous()
