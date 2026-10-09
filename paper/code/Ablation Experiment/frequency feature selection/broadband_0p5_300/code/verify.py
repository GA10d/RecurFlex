"""Validate full-band representations, decimation, masks and model shapes."""
import argparse
import numpy as np
import torch
from common import HERE, protocol, feature_mask, tensor_hash, objective, setup, RecurFlex, save_json
import mne
from features import morlet_power, signed_filterbank


def verify(device="cuda"):
    p = protocol()
    frequencies = np.asarray(p["feature_recipe"]["frequencies"])
    edges = np.asarray(p["feature_recipe"]["band_edges"])
    assert len(frequencies)==40 and len(edges)==41 and np.all(np.diff(edges)>0)
    assert np.allclose(frequencies,np.geomspace(.5,300,40),rtol=1e-12)
    assert np.allclose(edges[1:-1],np.sqrt(frequencies[:-1]*frequencies[1:]),rtol=1e-12)
    t = np.arange(30000)/1000.
    wave = (np.sin(2*np.pi*20*t)+np.sin(2*np.pi*70*t))[None]
    power = morlet_power(wave, frequencies)
    signed = signed_filterbank(wave, edges)
    assert power.shape==signed.shape==(1,40,3000)
    assert np.isfinite(power).all() and np.isfinite(signed).all() and (power>=0).all()
    band = np.searchsorted(edges,70)-1
    sampled = signed[0,band,1000:3000]
    assert sampled.min()<0 and sampled.max()>0
    peaks = np.fft.rfftfreq(len(sampled),.01)
    alias = peaks[np.argmax(np.abs(np.fft.rfft(sampled-sampled.mean())))]
    assert abs(alias-30.)<.1
    support = [len(w) for w in mne.time_frequency.morlet(1000.,frequencies,n_cycles=7,zero_mean=False)]
    rows = []
    baseline = np.random.default_rng(42).normal(size=(16,43,2048)).astype(np.float32)
    for group in p["groups"]:
        setup()
        model = RecurFlex().to(device)
        digest = tensor_hash(model.state_dict())
        if device=="cuda":
            assert digest==p["reference"]["1"]["initial_state_sha256"]
        x = feature_mask(baseline,group)
        assert np.array_equal(x[:,:40],baseline[:,:40]) and not np.count_nonzero(x[:,40:])
        bx = torch.from_numpy(x[...,:512].copy())[None].to(device)
        y = torch.rand(1,5,512,device=device)
        prediction = model(bx)
        assert prediction.shape==y.shape
        loss = objective(prediction,y)
        loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        gradient = model.network.reduction.conv.weight.grad.reshape(64,16,43,3)
        assert not torch.count_nonzero(gradient[:,:,40:]) and torch.count_nonzero(gradient[:,:,:40])
        model.eval()
        with torch.inference_mode():
            assert model(torch.from_numpy(x)[None].to(device)).shape==(1,5,2048)
        rows.append(dict(group=group,initial_state_sha256=digest,parameters=sum(v.numel() for v in model.parameters())))
    assert len({r["initial_state_sha256"] for r in rows})==1
    assert {r["parameters"] for r in rows}=={6805126}
    result = dict(passed=True,groups=rows,device=device,morlet_nonnegative=True,signed_waveform_sign_preserved=True,
                  direct_decimation_70hz_alias=alias,morlet_support_samples=support,
                  identical_initialization=True,masked_features_and_gradients_zero=True,train_window=512,inference_window=2048)
    save_json(HERE/("verification.json" if device=="cuda" else "local_verification.json"),result)
    print(result,flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cuda",choices=("cpu","cuda"))
    verify(parser.parse_args().device)
