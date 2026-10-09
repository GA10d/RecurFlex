"""Fixed training protocol, band layouts and embedded original initialization."""
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.plotcache'))
sys.path.insert(0,str(HERE/'main_source'))
from config import Config
from data import training_labels,normalize_targets,inverse_targets
from inference import predict,restore_1000hz,coverage
from model import RecurFlex
from train import objective
from utils import load_json,save_json,setup,sha256

STEM='network.reduction.conv.weight'

def protocol():return load_json(HERE/'protocol.json')

def tensor_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def make_model(group,device='cuda'):
    p=protocol();setup();reference=RecurFlex(16,43,.1);original=reference.state_dict()
    if device=='cuda':assert tensor_hash(original)==p['reference']['1']['initial_state_sha256']
    if p['groups'][group]['model_slots']==43:return reference.to(device)
    setup();model=RecurFlex(16,44,.1);state=model.state_dict()
    for name in state:
        if name!=STEM:state[name].copy_(original[name])
    old=original[STEM].reshape(64,16,43,3)
    new=state[STEM].reshape(64,16,44,3)
    new[:,:,:42].copy_(old[:,:,:42]);new[:,:,43].copy_(old[:,:,42])
    # Only slot42 (the new alpha band) retains the new random initialization.
    assert tensor_hash({k:v for k,v in state.items() if k!=STEM})==tensor_hash({k:v for k,v in original.items() if k!=STEM})
    return model.to(device)

def feature_mask(x,group):
    settings=protocol()['groups'][group];slots=settings['model_slots']
    if x.shape[:2]!=(16,slots):raise ValueError('Unexpected electrode/feature dimensions')
    result=np.array(x,dtype=np.float32,copy=True,order='C')
    result[:,sorted(set(range(slots))-set(settings['active_features']))]=0.
    return result

def fit_statistics(features,truth,reference,group):
    c,f,n=features.shape;settings=protocol()['groups'][group]
    assert c==16 and f==settings['raw_features'] and truth.shape==(n,5)
    scaler=RobustScaler(quantile_range=(.1,.9),unit_variance=True).fit(features[...,225:-25].transpose(2,0,1).reshape(n-250,-1))
    values=truth[225:-25]
    assert np.array_equal(values.min(0),reference['target_min'])
    assert np.array_equal(values.max(0)-values.min(0),reference['target_range'])
    gains=np.ones(f);gains[settings['signed_start']:]=200.
    return dict(feature_center=scaler.center_.tolist(),feature_scale=scaler.scale_.tolist(),feature_gains=gains.tolist(),
        selected_electrodes=reference['selected_electrodes'],electrodes=16,frequencies=f,
        model_slots=settings['model_slots'],slot_map=settings['slot_map'],
        target_min=reference['target_min'],target_range=reference['target_range'])

def transform(features,stats):
    c,f,n=features.shape
    matrix=features.transpose(2,0,1).reshape(n,-1).astype(np.float32,copy=True)
    matrix-=np.asarray(stats['feature_center'],dtype=np.float32)
    matrix/=np.asarray(stats['feature_scale'])
    raw=matrix.reshape(n,c,f).transpose(1,2,0)
    # Match the original float32 signed-band scaling, after RobustScaler.
    raw[:,40:]*=200.
    result=np.zeros((c,stats['model_slots'],n),dtype=np.float32)
    result[:,stats['slot_map']]=raw
    assert np.isfinite(result).all()
    return result

def prepare_inputs():
    from features import precompute
    p=protocol();manifest=dict(protocol_sha256=sha256(HERE/'protocol.json'),subjects=[])
    for name,digest in p['main_source_sha256'].items():assert sha256(HERE/'main_source'/name)==digest
    for s in p['subjects']:
        train=precompute(s,'train_full');test=precompute(s,'test')
        truth=training_labels(p['data_source'],s,'train_full');record=dict(subject=s,groups={})
        for group in p['groups']:
            folder=HERE/'inputs'/f's{s}'/group;folder.mkdir(parents=True,exist_ok=True)
            features=np.load(train[group],mmap_mode='r');stats=fit_statistics(features,truth,p['reference'][str(s)],group)
            save_json(folder/'scalers.json',stats)
            np.save(folder/'x_train.npy',transform(features,stats));np.save(folder/'y_train.npy',normalize_targets(truth,stats))
            np.save(folder/'x_test.npy',transform(np.load(test[group],mmap_mode='r'),stats))
            record['groups'][group]=dict(raw_feature_sha256={'train':sha256(train[group]),'test':sha256(test[group])},
                prepared_sha256={f.name:sha256(f) for f in folder.iterdir() if f.is_file()})
        manifest['subjects'].append(record);save_json(HERE/'input_manifest.json',manifest)
        print(f'Prepared three inputs for S{s}',flush=True)
    return manifest
