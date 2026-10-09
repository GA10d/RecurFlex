"""Validate initialization embedding, signed split bands and complete window shapes."""
import argparse
import numpy as np
import torch
from common import HERE,STEM,protocol,make_model,feature_mask,tensor_hash,objective,save_json
from features import morlet_power,signed_bands

def verify(device='cuda'):
    p=protocol();f=np.asarray(p['feature_recipe']['linear_frequencies'])
    assert len(f)==40 and np.allclose(np.diff(f),(300-.5)/39,rtol=1e-12)
    assert int((f>=40).sum())==34
    t=np.arange(30000)/1000.;x=sum(np.sin(2*np.pi*h*t) for h in (2,6,10,20))[None]
    power=morlet_power(x,f);low=signed_bands(x,p['feature_recipe']['signed_bands_4'])
    assert power.shape==(1,40,3000) and low.shape==(1,4,3000)
    assert (power>=0).all() and np.all(low.min(axis=-1)<0) and np.all(low.max(axis=-1)>0)
    coefficients=np.array([[abs(np.sum(low[0,i,500:-500]*np.exp(-2j*np.pi*h*t[::10][500:-500]))) for h in (2,6,10,20)] for i in range(4)])
    assert np.array_equal(coefficients.argmax(1),np.arange(4))
    original=make_model('linear_morlet','cpu').eval();control=make_model('hybrid_3_control','cpu').eval()
    sample=torch.from_numpy(np.random.default_rng(42).normal(size=(1,16,43,512)).astype(np.float32))
    embedded=torch.zeros(1,16,44,512);embedded[:,:,:42]=sample[:,:,:42];embedded[:,:,43]=sample[:,:,42]
    with torch.inference_mode():
        a=original(sample);b=control(embedded)
    assert torch.allclose(a,b,rtol=1e-4,atol=1e-5)
    max_difference=float((a-b).abs().max());del original,control
    rows=[]
    for group,settings in p['groups'].items():
        model=make_model(group,device);digest=tensor_hash(model.state_dict())
        slots=settings['model_slots'];raw=np.random.default_rng(42).normal(size=(16,slots,2048)).astype(np.float32)
        features=feature_mask(raw,group);active=settings['active_features'];inactive=sorted(set(range(slots))-set(active))
        assert np.array_equal(features[:,active],raw[:,active]) and not np.count_nonzero(features[:,inactive])
        bx=torch.from_numpy(features[...,:512].copy())[None].to(device)
        output=model(bx);target=torch.rand(1,5,512,device=device);assert output.shape==target.shape
        loss=objective(output,target);assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        gradient=model.network.reduction.conv.weight.grad.reshape(64,16,slots,3)
        assert not torch.count_nonzero(gradient[:,:,inactive]) and torch.count_nonzero(gradient[:,:,active])
        model.eval()
        with torch.inference_mode():assert model(torch.from_numpy(features)[None].to(device)).shape==(1,5,2048)
        count=sum(v.numel() for v in model.parameters());assert count==settings['parameters']
        rows.append(dict(group=group,initial_state_sha256=digest,nonstem_initial_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if k!=STEM}),parameters=count))
        del model
    assert len({r['initial_state_sha256'] for r in rows if r['group']!='linear_morlet'})==1
    assert len({r['nonstem_initial_sha256'] for r in rows})==1
    result=dict(passed=True,device=device,groups=rows,linear_spacing_hz=float(np.diff(f)[0]),
        linear_high_centers=34,signed_band_sign_and_selectivity_verified=True,
        original_control_initial_function_matches=True,original_control_max_abs_difference=max_difference,
        original_nonstem_initialization_preserved=True,masked_features_and_gradients_zero=True,
        train_window=512,inference_window=2048)
    save_json(HERE/('verification.json' if device=='cuda' else 'local_verification.json'),result)
    print(result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    verify(parser.parse_args().device)
