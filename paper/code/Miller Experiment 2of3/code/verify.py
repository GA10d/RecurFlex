"""Check split boundaries, label access, input dimensions and a real model gradient."""
import numpy as np
import torch
from common import HERE, protocol, load_json, save_json, sha256, setup, training_model, objective, coverage
from worker import tensor_hash


def verify_subject(subject):
    p=protocol();r=next(r for r in p['records'] if r['subject']==subject)
    splits=r['splits']
    assert splits['val_inner'][0]-splits['train_inner'][1]==2000
    assert splits['test'][0]-splits['train_full'][1]==2000
    assert splits['val_inner'][1]==splits['train_full'][1]
    assert splits['test'][1]==r['source_samples']
    assert all(v%40==0 for a,b in splits.values() for v in (a,b))
    with np.load(HERE/'raw'/r['upload_file']) as z:
        assert set(z.files)=={'data','development_flex'}
        assert z['development_flex'].shape==(splits['train_full'][1],5)
    selections={}
    for phase,segment in [('validation','train_inner'),('full','train_full')]:
        folder=HERE/'inputs'/subject/phase
        manifest=load_json(folder/'manifest.json')
        for name,digest in manifest['files_sha256'].items():assert sha256(folder/name)==digest
        stats=load_json(folder/'scalers.json')
        assert stats['electrode_selection']=='same-time correlation on supplied training split only'
        assert len(stats['selected_electrodes'])==16 and len(set(stats['selected_electrodes']))==16
        assert min(stats['selected_electrodes'])>=0 and max(stats['selected_electrodes'])<r['channels']
        assert stats['feature_gains']==[1.]*40+[200.]*4
        selections[phase]=stats['selected_electrodes']
        x=np.load(folder/'x_train.npy',mmap_mode='r');y=np.load(folder/'y_train.npy',mmap_mode='r')
        n=(splits[segment][1]-splits[segment][0])//10
        assert x.shape==(16,44,n) and y.shape==(n,5) and x.dtype==y.dtype==np.float32
        assert np.isfinite(x).all() and np.isfinite(y).all()
        if phase=='full':
            assert not list(folder.glob('*truth*')) and not (folder/'y_test.npy').exists()
            test=np.load(folder/'x_test.npy',mmap_mode='r')
            assert test.shape==(16,44,(splits['test'][1]-splits['test'][0])//10)
            assert np.isfinite(test).all()
    return dict(subject=subject,selected_electrodes=selections,test_labels_present=False)


def model_check(subject):
    setup();model=training_model().cuda()
    initial=tensor_hash(model.state_dict())
    folder=HERE/'inputs'/subject/'validation'
    x=np.load(folder/'x_train.npy',mmap_mode='r')
    y=np.load(folder/'y_train.npy',mmap_mode='r')
    bx=torch.tensor(x[...,225:737].copy()[None],device='cuda')
    by=torch.tensor(y[225:737].T.copy()[None],device='cuda')
    model.train();pred=model(bx);assert pred.shape==(1,5,512)
    loss=objective(pred,by);assert torch.isfinite(loss)
    loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    model.eval()
    with torch.inference_mode():assert model(torch.zeros(1,16,44,2048,device='cuda')).shape==(1,5,2048)
    return dict(parameters=sum(v.numel() for v in model.parameters()),loss=float(loss.detach()),initial_state_sha256=initial,
                train_shape=[1,16,44,512],output_shape=[1,5,512],inference_shape=[1,16,44,2048])


def main():
    p=protocol()
    for name,digest in p['source_sha256'].items():assert sha256(HERE/'main_source'/name)==digest
    rows=[verify_subject(s) for s in p['subjects']]
    result=dict(passed=True,subjects=rows,model=model_check(p['subjects'][0]),test_labels_present=False)
    save_json(HERE/'verification.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
