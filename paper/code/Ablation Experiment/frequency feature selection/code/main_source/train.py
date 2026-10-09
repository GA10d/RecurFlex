"""Seed-42 matched-time training with ordinary weights."""
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from config import Config, FULL_EPOCHS, HERE
from data import cached_features, fit_statistics, inverse_targets, normalize_targets, sample_batch, training_labels, transform
from inference import predict
from metrics import score
from model import RecurFlex
from utils import save_json, setup, sha256


def objective(prediction, truth):
    return 0.5 * F.mse_loss(prediction, truth) + 0.5 * (
        1 - F.cosine_similarity(prediction, truth, dim=-1).mean())


def fit(subject, data_dir, cache_dir, output_dir, device, phase="full", epochs=None, cfg=None):
    cfg = cfg or Config()
    setup()
    segment = "train_full" if phase == "full" else "train_inner"
    folder = Path(output_dir) / f"s{subject}"
    if (folder / "model.pt").exists():
        raise FileExistsError(f"Checkpoint already exists: {folder}. Choose a new output directory.")
    features = cached_features(data_dir, cache_dir, subject, segment)
    truth = training_labels(data_dir, subject, segment)
    stats = fit_statistics(features, truth, cfg.channel_count)
    inputs = transform(features, stats)
    x = torch.from_numpy(inputs.transpose(2, 0, 1).copy()).to(device)
    y = torch.from_numpy(normalize_targets(truth, stats)).to(device)
    model = RecurFlex(cfg.channel_count, cfg.features, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    count = epochs or (FULL_EPOCHS[subject] if phase == "full" else cfg.max_epochs)
    validation = None
    if phase == "validation":
        validation = (transform(cached_features(data_dir, cache_dir, subject, "val_inner"), stats),
                      training_labels(data_dir, subject, "val_inner"))
    config = dict(subject=subject, phase=phase, epochs=count, settings=cfg.to_dict())
    save_json(folder / "config.json", config)
    save_json(folder / "scalers.json", stats)
    save_json(folder / "environment.json", dict(torch=torch.__version__, device=device,
              gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None,
              parameters=sum(p.numel() for p in model.parameters()), seed=cfg.seed,
              source_sha256={p.name:sha256(p) for p in HERE.glob("*.py")}))
    history = []
    best = -float("inf")
    best_epoch = 0
    best_state = None
    start = time.monotonic()
    for epoch in range(1, count + 1):
        model.train()
        total = 0.
        steps = (cfg.windows_per_epoch + cfg.batch - 1) // cfg.batch
        for _ in range(steps):
            bx, by = sample_batch(x, y, cfg.batch, generator, cfg.window)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(bx), by)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        row = dict(epoch=epoch, loss=total / steps, elapsed=time.monotonic() - start)
        if validation is not None:
            vx, vy = validation
            prediction = inverse_targets(predict(model, vx, device), stats)
            validation_score = score(vy[225:-25], prediction[225:-25])["r4"]
            row["validation_r4"] = validation_score
            if validation_score > best + 1e-4:
                best = validation_score
                best_epoch = epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
        save_json(folder / "history.json", history)
        print(f"S{subject} {phase}: {row}", flush=True)
        if validation is not None and epoch - best_epoch >= cfg.patience:
            break
    state = best_state if validation is not None else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    saved_epoch = best_epoch if validation is not None else len(history)
    torch.save(dict(state_dict=state, subject=subject, seed=42, epoch=saved_epoch), folder / "model.pt")
    fit_result = dict(subject=subject, phase=phase, seed=42, epochs=len(history), selected_epoch=saved_epoch,
                      elapsed=time.monotonic() - start, model_sha256=sha256(folder / "model.pt"))
    if validation is not None:
        fit_result.update(validation_r4=best, suggested_full_epochs=max(8, min(160, round(best_epoch * 400 / 319))))
    save_json(folder / "fit.json", fit_result)
    return fit_result
