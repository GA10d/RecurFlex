"""Verify gradients, timestamp pairing, full-block coverage and supplied weights."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat

from config import Config, DATA_DIR, FULL_ELECTRODES, HERE
from data import SIGNED_BANDS, fit_statistics, labels100, normalize_targets, raw_features, sample_batch, transform
from inference import coverage, load_model, predict, restore_1000hz
from model import RecurFlex, training_model
from train import objective
from utils import device_name, load_json, save_json, setup, sha256


class ClockModel(torch.nn.Module):
    def forward(self, x):
        return x[:, 0, :5]


def verify(device, real_data=False, data_dir=DATA_DIR):
    setup()
    cfg = Config()
    model = RecurFlex().to(device)
    assert sum(p.numel() for p in model.parameters()) == 6808198
    assert SIGNED_BANDS == ((0.5, 4), (4, 8), (8, 13), (13, 30))
    bx = torch.randn(2, 16, 44, 512, device=device)
    by = torch.rand(2, 5, 512, device=device)
    out = model(bx)
    assert out.shape == by.shape
    loss = objective(out, by)
    loss.backward()
    for name in ("network.reduction.conv.weight", "gru.weight_ih_l0", "gru.weight_ih_l0_reverse",
                 "context_output.weight", "context_gain", "network.output.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    model.eval()
    with torch.inference_mode():
        assert model(torch.randn(1, 16, 44, 2048, device=device)).shape == (1, 5, 2048)
    clock = torch.arange(1200, dtype=torch.float32, device=device)
    x = clock[:, None, None].expand(-1, 16, 44)
    y = clock[:, None].expand(-1, 5)
    a, b = sample_batch(x, y, 8, torch.Generator(device=device).manual_seed(42))
    assert torch.equal(a[:, 0, 0], b[:, 0])
    for length in (512, 2048, 5000, 20000):
        features = np.broadcast_to(np.arange(length, dtype=np.float32), (16, 44, length))
        result = predict(ClockModel().to(device), features, device)
        assert np.array_equal(result, np.broadcast_to(np.arange(length)[:, None], (length, 5)))
        plan = coverage(length)
        assert sum(last - first for first, last in plan["output_ranges"]) == length
        assert plan["output_ranges"][0][0] == 0 and plan["output_ranges"][-1][1] == length
    native = np.broadcast_to(np.arange(20000)[:, None], (20000, 5))
    restored = restore_1000hz(native, 200000)
    assert restored.shape == (200000, 5) and np.array_equal(restored[::10], native)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(18, 44, 1200)).astype(np.float32)
    truth = rng.normal(size=(1200, 5))
    features[3, 0] = truth[:, 0]
    stats = fit_statistics(features, truth)
    assert 3 in stats["selected_electrodes"] and len(stats["selected_electrodes"]) == 16
    fitted = dict(stats)
    assert transform(features * 100, stats).shape == (16, 44, 1200)
    assert stats == fitted
    assert len(stats["feature_center"]) == 16 * 44
    assert np.array_equal(transform(features, stats), transform(features[stats["selected_electrodes"]], stats))
    # Verify the actual training initialization recipe, including the beta-slot remap.
    setup()
    reference = RecurFlex(16, 43).state_dict()
    initialized = training_model().state_dict()
    stem = "network.reduction.conv.weight"
    assert all(torch.equal(v, initialized[k]) for k, v in reference.items() if k != stem)
    old = reference[stem].reshape(64, 16, 43, 3)
    new = initialized[stem].reshape(64, 16, 44, 3)
    assert torch.equal(old[:, :, :42], new[:, :, :42])
    assert torch.equal(old[:, :, 42], new[:, :, 43])
    provenance = load_json(HERE / "artifacts/pretrained/provenance.json")
    for item in provenance["subjects"]:
        folder = HERE / "artifacts/pretrained" / f"s{item['subject']}"
        assert sha256(folder / "model.pt") == item["checkpoint_sha256"]
        assert sha256(folder / "scalers.json") == item["scalers_sha256"]
        loaded, supplied_stats, _ = load_model(folder, device)
        assert supplied_stats["selected_electrodes"] == FULL_ELECTRODES[item["subject"]]
        assert len(supplied_stats["feature_center"]) == 16 * 44
        with torch.inference_mode():
            assert loaded(torch.randn(1, 16, 44, 512, device=device)).shape == (1, 5, 512)
        del loaded
    report = dict(passed=True, seed=42, parameters=6808198, features=44, device=device,
                  training_output=[2, 5, 512], inference_output=[1, 5, 2048],
                  gradient_branches_checked=6, same_timestamp_pairing=True,
                  full_block_coverage_lengths=[512, 2048, 5000, 20000],
                  single_checkpoint_subjects=[1, 2, 3], training_only_statistics=True,
                  signed_bands=[list(b) for b in SIGNED_BANDS], shared_random_initialization_verified=True,
                  selection_before_scaling_verified=True,
                  real_training_minibatch=False)
    if real_data:
        raw = loadmat(Path(data_dir) / "sub1_comp.mat", variable_names=["train_data", "train_dg"])
        f = raw_features(raw["train_data"][:20000])
        supplied = load_json(HERE / "artifacts/pretrained/s1/scalers.json")
        tx = torch.from_numpy(transform(f, supplied).transpose(2, 0, 1).copy()).to(device)
        ty = torch.from_numpy(normalize_targets(labels100(raw["train_dg"][:20000]), supplied)).to(device)
        model = training_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        a, b = sample_batch(tx, ty, 2, torch.Generator(device=device).manual_seed(42))
        optimizer.zero_grad(set_to_none=True)
        real_loss = objective(model(a), b)
        real_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(real_loss) and torch.isfinite(norm)
        optimizer.step()
        report.update(real_training_minibatch=True, real_training_feature_shape=list(f.shape),
                      real_training_loss=float(real_loss.detach()))
    save_json(HERE / "verification.json", report)
    print(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    verify(device_name(args.device), args.real_data, args.data_dir)
