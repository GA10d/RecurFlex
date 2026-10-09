"""Fresh fixed-budget seed-42 fitting of one feature arm and subject."""
import argparse
import hashlib
import time
import numpy as np
import torch
from common import HERE, Config, RecurFlex, feature_mask, tensor_hash, protocol, setup, save_json, sha256
from common import objective, predict, inverse_targets, restore_1000hz, coverage, load_json


def run(group, subject):
    p = protocol()
    cfg = Config()
    setup()
    folder = HERE / "fits" / f"s{subject}_{group}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists():
        result = load_json(folder / "fit.json")
        assert result["model_sha256"] == sha256(folder / "model.pt")
        return
    source = HERE / "inputs" / f"s{subject}"
    inputs = feature_mask(np.load(source / "x_train.npy"), group)
    x = torch.from_numpy(inputs.transpose(2, 0, 1).copy()).cuda()
    y = torch.from_numpy(np.load(source / "y_train.npy")).cuda()
    model = RecurFlex(16, 43, cfg.dropout).cuda()
    initial_sha = tensor_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device="cuda").manual_seed(42)
    offsets = torch.arange(512, device="cuda")
    high = len(x) - 512 - 25 + 1
    epochs = p["training_epochs"][str(subject)]
    config = dict(group=group, subject=subject, seed=42, epochs=epochs, settings=cfg.to_dict(),
                  active_features=p["groups"][group]["active_features"], initial_state_sha256=initial_sha,
                  parameters=sum(v.numel() for v in model.parameters()),
                  scalers_sha256=sha256(source / "scalers.json"), sampling_bounds=[225, high])
    save_json(folder / "config.json", config)
    history = []
    start = time.monotonic()
    sampling_digest = hashlib.sha256()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for _ in range(cfg.windows_per_epoch // cfg.batch):
            starts = torch.randint(225, high, (cfg.batch,), generator=generator, device="cuda")
            sampling_digest.update(starts.cpu().numpy().tobytes())
            ix = starts[:, None] + offsets[None]
            bx = x[ix].permute(0, 2, 3, 1).contiguous()
            by = y[ix].transpose(1, 2).contiguous()
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(bx), by)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        row = dict(group=group, subject=subject, epoch=epoch, epochs=epochs,
                   loss=total / 128, elapsed=time.monotonic() - start)
        history.append(row)
        save_json(folder / "history.json", history)
        save_json(folder / "progress.json", row)
        print(row, flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    group=group, subject=subject, seed=42, epoch=epochs), folder / "model.pt")
    result = dict(group=group, subject=subject, seed=42, epochs=epochs, elapsed=time.monotonic() - start,
                  initial_state_sha256=initial_sha, sampling_sha256=sampling_digest.hexdigest(),
                  model_sha256=sha256(folder / "model.pt"), parameters=config["parameters"])
    save_json(folder / "fit.json", result)


@torch.inference_mode()
def infer(group, subject):
    setup()
    folder = HERE / "fits" / f"s{subject}_{group}"
    model = RecurFlex().cuda()
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    assert (checkpoint["group"], checkpoint["subject"], checkpoint["seed"]) == (group, subject, 42)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    source = HERE / "inputs" / f"s{subject}"
    x = feature_mask(np.load(source / "x_test.npy"), group)
    stats = load_json(source / "scalers.json")
    native = inverse_targets(predict(model, x, "cuda"), stats)
    full = restore_1000hz(native, 200000)
    out = HERE / "predictions"
    out.mkdir(exist_ok=True)
    path = out / f"s{subject}_{group}.npz"
    np.savez_compressed(path, prediction_100hz=native, prediction_1000hz=full)
    save_json(out / f"s{subject}_{group}_coverage.json", coverage(len(native)))
    return dict(group=group, subject=subject, seed=42, file=str(path.relative_to(HERE)),
                sha256=sha256(path), model_sha256=sha256(folder / "model.pt"),
                shape_100hz=list(native.shape), shape_1000hz=list(full.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("morlet", "low_frequency", "hybrid"), required=True)
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    run(args.group, args.subject)
