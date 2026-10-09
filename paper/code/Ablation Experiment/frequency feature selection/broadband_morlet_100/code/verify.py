"""Check paired initialization, masking, gradients, windows and shared Morlet centers."""
import argparse
import numpy as np
import torch
import mne
from common import HERE,STEM,protocol,feature_mask,tensor_hash,objective,make_model,save_json
from features import morlet_power


def verify(device='cuda'):
    p=protocol()
    frequencies={n:np.asarray(p['feature_recipe'][f'frequencies{n}']) for n in (40,100)}
    for n,f in frequencies.items():
        assert len(f)==n and np.allclose(f,np.geomspace(.5,300,n),rtol=1e-12)
    assert np.allclose(frequencies[40][[0,13,26,39]],frequencies[100][[0,33,66,99]],rtol=1e-12)
    t=np.arange(30000)/1000.
    wave=(np.sin(2*np.pi*20*t)+np.sin(2*np.pi*70*t))[None]
    powers={n:morlet_power(wave,f) for n,f in frequencies.items()}
    assert np.allclose(powers[40][:,[0,13,26,39]],powers[100][:,[0,33,66,99]],rtol=1e-5,atol=1e-7)
    support=[len(w) for w in mne.time_frequency.morlet(1000.,frequencies[100],n_cycles=7,zero_mean=False)]
    assert support[0]==22281
    baseline=np.random.default_rng(42).normal(size=(16,100,2048)).astype(np.float32)
    rows=[]
    for group in p['groups']:
        model=make_model(device)
        digest=tensor_hash(model.state_dict())
        x=feature_mask(baseline,group)
        n=p['groups'][group]['frequency_count']
        assert np.array_equal(x[:,:n],baseline[:,:n]) and not np.count_nonzero(x[:,n:])
        bx=torch.from_numpy(x[...,:512].copy())[None].to(device)
        prediction=model(bx)
        truth=torch.rand(1,5,512,device=device)
        assert prediction.shape==truth.shape
        loss=objective(prediction,truth)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        gradient=model.network.reduction.conv.weight.grad.reshape(64,16,100,3)
        assert not torch.count_nonzero(gradient[:,:,n:]) and torch.count_nonzero(gradient[:,:,:n])
        model.eval()
        with torch.inference_mode():
            assert model(torch.from_numpy(x)[None].to(device)).shape==(1,5,2048)
        rows.append(dict(group=group,initial_state_sha256=digest,
                    nonstem_initial_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if k!=STEM}),
                    parameters=sum(v.numel() for v in model.parameters())))
        del model
    assert len({r['initial_state_sha256'] for r in rows})==len({r['nonstem_initial_sha256'] for r in rows})==1
    assert {r['parameters'] for r in rows}=={p['parameters']}
    result=dict(passed=True,device=device,groups=rows,shared_centers_match=True,
                morlet_power_shape={str(n):list(v.shape) for n,v in powers.items()},
                morlet_support_samples=support,identical_initialization=True,
                original_nonstem_initialization_preserved=True,masked_features_and_gradients_zero=True,
                train_window=512,inference_window=2048)
    save_json(HERE/('verification.json' if device=='cuda' else 'local_verification.json'),result)
    print({k:v for k,v in result.items() if k!='morlet_support_samples'},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    verify(parser.parse_args().device)
