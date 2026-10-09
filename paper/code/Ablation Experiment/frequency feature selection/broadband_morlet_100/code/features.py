"""Native-rate broadband Morlet power at 100 logarithmic frequency centers."""
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from common import HERE,protocol,load_json,save_json,sha256


def preprocess(raw,selected):
    import mne
    x=np.asarray(raw).T.astype(np.float64)
    scale=x.std(1,keepdims=True)
    if not np.isfinite(x).all() or (scale<=0).any():
        raise ValueError("Invalid ECoG data.")
    x=(x-x.mean(1,keepdims=True))/scale
    x-=np.median(x,axis=0,keepdims=True)
    x=np.ascontiguousarray(x[selected])
    x=mne.filter.filter_data(x,1000.,.5,300.,verbose=False)
    return mne.filter.notch_filter(x,1000.,np.arange(60,500,60),verbose=False)


def morlet_power(filtered,frequencies):
    import mne
    result=np.empty((len(filtered),len(frequencies),filtered.shape[-1]//10),dtype=np.float32)
    for first in range(0,len(filtered),2):
        last=min(first+2,len(filtered))
        result[first:last]=mne.time_frequency.tfr_array_morlet(
            filtered[None,first:last],1000.,frequencies,n_cycles=7,zero_mean=False,
            use_fft=True,output="power",decim=10,n_jobs=1,verbose=False)[0]
    if not np.isfinite(result).all() or (result<0).any():
        raise FloatingPointError("Invalid Morlet power.")
    return result


def precompute(subject,segment):
    import mne
    p=protocol()
    source=Path(p["data_source"])/f"sub{subject}_comp.mat"
    assert sha256(source)==p["dataset_sha256"][str(subject)]
    samples=400000 if segment=="train_full" else 200000
    key="train_data" if segment=="train_full" else "test_data"
    old=Path(p["reference_feature_source"])/f"s{subject}_{segment}"/"morlet_broadband.npy"
    assert sha256(old)==p["reference"][str(subject)]["raw40_feature_sha256"][segment]
    f40=np.load(old,mmap_mode="r")
    assert f40.shape==(16,40,samples//10) and f40.dtype==np.float32
    selected=p["reference"][str(subject)]["selected_electrodes"]
    signature=dict(source_sha256=sha256(source),segment=segment,indices=[0,samples],selected_electrodes=selected,
                   mne=mne.__version__,recipe=p["feature_recipe"],feature_source_sha256=sha256(HERE/"features.py"),
                   reference40_sha256=sha256(old))
    folder=HERE/"features"/f"s{subject}_{segment}"
    path=folder/"morlet_100.npy"
    if (folder/"metadata.json").exists():
        stored=load_json(folder/"metadata.json")
        assert stored["signature"]==signature
        assert np.load(path,mmap_mode="r").shape==(16,100,samples//10)
        return dict(morlet_40=old,morlet_100=path)
    save_json(HERE/"preprocessing_progress.json",dict(subject=subject,segment=segment,step="morlet_100"))
    start=time.monotonic()
    raw=loadmat(source,variable_names=[key])[key]
    assert len(raw)==samples
    filtered=preprocess(raw,selected)
    del raw
    features=morlet_power(filtered,p["feature_recipe"]["frequencies100"])
    shared40=f40[:,[0,13,26,39]]
    shared100=features[:,[0,33,66,99]]
    assert np.allclose(shared40,shared100,rtol=1e-5,atol=1e-7)
    folder.mkdir(parents=True,exist_ok=True)
    np.save(path,features)
    save_json(folder/"metadata.json",dict(signature=signature,shared_centers_match=True,
              shared_centers_max_abs_difference=float(np.max(np.abs(shared40-shared100)))))
    print(dict(subject=subject,segment=segment,shape=list(features.shape),shared_centers_match=True,elapsed=time.monotonic()-start),flush=True)
    return dict(morlet_40=old,morlet_100=path)
