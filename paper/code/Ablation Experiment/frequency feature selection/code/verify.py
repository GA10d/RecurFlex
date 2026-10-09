"""Check that ablation arms only differ in active input features."""
import numpy as np
import torch
from common import HERE, RecurFlex, protocol, feature_mask, tensor_hash, objective, setup, save_json


def verify():
    p = protocol()
    results = []
    baseline = np.random.default_rng(42).normal(size=(16, 43, 2048)).astype(np.float32)
    for group, definition in p["groups"].items():
        setup()
        model = RecurFlex().cuda()
        initial_sha = tensor_hash(model.state_dict())
        x = feature_mask(baseline, group)
        active = definition["active_features"]
        excluded = sorted(set(range(43)) - set(active))
        assert np.array_equal(x[:, active], baseline[:, active])
        assert not np.count_nonzero(x[:, excluded])
        bx = torch.from_numpy(x[..., :512].copy())[None].cuda()
        y = torch.rand(1, 5, 512, device="cuda")
        prediction = model(bx)
        assert prediction.shape == y.shape
        loss = objective(prediction, y)
        loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        gradient = model.network.reduction.conv.weight.grad.reshape(64, 16, 43, 3)
        assert not torch.count_nonzero(gradient[:, :, excluded])
        assert torch.count_nonzero(gradient[:, :, active])
        model.eval()
        with torch.inference_mode():
            assert model(torch.from_numpy(x)[None].cuda()).shape == (1, 5, 2048)
        results.append(dict(group=group, active_features=len(active), initial_state_sha256=initial_sha,
                            parameters=sum(v.numel() for v in model.parameters()), loss=float(loss.detach())))
        del model
    assert len({r["initial_state_sha256"] for r in results}) == 1
    assert {r["parameters"] for r in results} == {6805126}
    result = dict(passed=True, groups=results, identical_initialization=True,
                  masked_features_and_gradients_zero=True, train_window=512, inference_window=2048)
    save_json(HERE / "verification.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    verify()
