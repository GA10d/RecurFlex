"""Single-checkpoint full-block continuous reconstruction."""
from pathlib import Path
import numpy as np
import torch

from config import Config
from data import cached_features, inverse_targets, transform
from model import RecurFlex
from utils import load_json, save_json, sha256


def coverage(n, window=2048):
    if n < 1 or window != 2048:
        raise ValueError("A nonempty record and a 2048-point window are required.")
    blocks = (n + window - 1) // window
    return dict(record_length=n, input_window=window, stride=window, blocks=blocks,
                right_padding=blocks * window - n,
                output_ranges=[[i * window, min((i + 1) * window, n)] for i in range(blocks)])


@torch.inference_mode()
def predict(model, x, device="cpu", window=2048):
    model.eval()
    plan = coverage(x.shape[-1], window)
    padded = np.pad(x, ((0, 0), (0, 0), (0, plan["right_padding"])), mode="reflect")
    parts = []
    for first in range(0, plan["blocks"] * window, window):
        batch = torch.from_numpy(padded[..., first:first + window].copy())[None].to(device)
        p = model(batch)
        if p.shape != (1, 5, window):
            raise ValueError("Unexpected model output shape.")
        parts.append(p[0].float().cpu().numpy().T)
    result = np.concatenate(parts)[:plan["record_length"]]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite model output.")
    return result


def restore_1000hz(prediction, samples):
    if prediction.shape != (samples // 10, 5) or samples % 10:
        raise ValueError("Prediction length does not match the original record.")
    return np.column_stack([
        np.interp(np.arange(samples), np.arange(len(prediction)) * 10, prediction[:, f])
        for f in range(5)])


def load_model(folder, device):
    folder = Path(folder)
    config = load_json(folder / "config.json")
    cfg = Config(**config["settings"])
    stats = load_json(folder / "scalers.json")
    if (stats["electrodes"], stats["frequencies"]) != (cfg.channel_count, cfg.features):
        raise ValueError("Scaler dimensions differ from the model configuration.")
    model = RecurFlex(stats["electrodes"], stats["frequencies"], cfg.dropout).to(device)
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    if saved["seed"] != 42 or saved["subject"] != config["subject"]:
        raise ValueError("Checkpoint seed or subject differs from the experiment configuration.")
    return model.eval(), stats, cfg


def infer_subject(subject, data_dir, cache_dir, weights_dir, output_dir, device):
    folder = Path(weights_dir) / f"s{subject}"
    config = load_json(folder / "config.json")
    if config["subject"] != subject:
        raise ValueError("Checkpoint subject differs from the requested subject.")
    features = cached_features(data_dir, cache_dir, subject, "test")
    model, stats, cfg = load_model(folder, device)
    pred = inverse_targets(predict(model, transform(features, stats), device, cfg.inference_window), stats)
    full = restore_1000hz(pred, features.shape[-1] * 10)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"s{subject}_prediction.npz"
    np.savez_compressed(path, prediction_100hz=pred, prediction_1000hz=full)
    save_json(output_dir / f"s{subject}_coverage.json", coverage(len(pred)))
    return dict(subject=subject, prediction=str(path.resolve()), prediction_sha256=sha256(path),
                model_sha256=sha256(folder / "model.pt"), scalers_sha256=sha256(folder / "scalers.json"),
                shape_100hz=list(pred.shape), shape_1000hz=list(full.shape), seed=42)
