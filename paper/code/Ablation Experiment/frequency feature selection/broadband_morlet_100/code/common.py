"""Paired 40-versus-100 frequency inputs on the same 100-slot RecurFlex model."""
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(HERE/".plotcache"))
sys.path.insert(0,str(HERE/"main_source"))
from config import Config
from data import training_labels,normalize_targets,inverse_targets
from inference import predict,restore_1000hz,coverage
from model import RecurFlex
from train import objective
from utils import load_json,save_json,setup,sha256

STEM="network.reduction.conv.weight"


def protocol():
    return load_json(HERE/"protocol.json")


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model(device="cuda"):
    # Preserve every common backbone tensor from the original seed-42 initialization.
    setup()
    reference=RecurFlex(16,43,.1)
    original=reference.state_dict()
    if device=="cuda":
        assert tensor_hash(original)==protocol()["reference"]["1"]["initial_state_sha256"]
    setup()
    model=RecurFlex(16,100,.1)
    state=model.state_dict()
    for name in state:
        if name!=STEM:
            state[name].copy_(original[name])
    assert tensor_hash({k:v for k,v in state.items() if k!=STEM})==tensor_hash({k:v for k,v in original.items() if k!=STEM})
    return model.to(device)


def feature_mask(x,group):
    count=protocol()["groups"][group]["frequency_count"]
    if x.shape[:2]!=(16,100):
        raise ValueError("Expected 16 electrodes and 100 input slots.")
    result=np.array(x,dtype=np.float32,copy=True,order="C")
    result[:,count:]=0.
    return result


def fit_statistics(features,truth,reference):
    c,f,n=features.shape
    if c!=16 or f not in (40,100) or truth.shape!=(n,5):
        raise ValueError("Invalid feature or target dimensions.")
    scaler=RobustScaler(quantile_range=(.1,.9),unit_variance=True).fit(features[...,225:-25].transpose(2,0,1).reshape(n-250,-1))
    values=truth[225:-25]
    assert np.array_equal(values.min(0),reference["target_min"])
    assert np.array_equal(values.max(0)-values.min(0),reference["target_range"])
    return dict(feature_center=scaler.center_.tolist(),feature_scale=scaler.scale_.tolist(),
                target_min=reference["target_min"],target_range=reference["target_range"],
                selected_electrodes=reference["selected_electrodes"],electrodes=16,frequencies=f,
                model_slots=100,input_gain=1,scaler="TRAIN RobustScaler (.1,.9) percentiles; unit_variance=True")


def transform(features,stats):
    c,f,n=features.shape
    matrix=features.transpose(2,0,1).reshape(n,-1).astype(np.float32,copy=True)
    matrix-=np.asarray(stats["feature_center"],dtype=np.float32)
    matrix/=np.asarray(stats["feature_scale"])
    result=np.zeros((c,100,n),dtype=np.float32)
    result[:,:f]=matrix.reshape(n,c,f).transpose(1,2,0)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite inputs.")
    return result


def prepare_inputs():
    from features import precompute
    p=protocol()
    for name,digest in p["main_source_sha256"].items():
        assert sha256(HERE/"main_source"/name)==digest
    manifest=dict(protocol_sha256=sha256(HERE/"protocol.json"),subjects=[])
    for subject in p["subjects"]:
        train=precompute(subject,"train_full")
        test=precompute(subject,"test")
        truth=training_labels(p["data_source"],subject,"train_full")
        info=dict(subject=subject,groups={})
        for group in p["groups"]:
            folder=HERE/"inputs"/f"s{subject}"/group
            folder.mkdir(parents=True,exist_ok=True)
            f=np.load(train[group],mmap_mode="r")
            stats=fit_statistics(f,truth,p["reference"][str(subject)])
            save_json(folder/"scalers.json",stats)
            np.save(folder/"x_train.npy",transform(f,stats))
            np.save(folder/"y_train.npy",normalize_targets(truth,stats))
            np.save(folder/"x_test.npy",transform(np.load(test[group],mmap_mode="r"),stats))
            info["groups"][group]=dict(raw_feature_sha256={"train":sha256(train[group]),"test":sha256(test[group])},
                    prepared_sha256={v.name:sha256(v) for v in folder.iterdir() if v.is_file()})
        manifest["subjects"].append(info)
        save_json(HERE/"input_manifest.json",manifest)
        print(f"Prepared paired 40/100 inputs for S{subject}",flush=True)
    return manifest
