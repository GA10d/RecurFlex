"""Linear broadband Morlet and four signed bands; keep original high-power cache."""
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt
from common import HERE,protocol,load_json,save_json,sha256

def referenced_ecog(raw,selected):
    x=np.asarray(raw).T.astype(np.float64);scale=x.std(1,keepdims=True)
    assert np.isfinite(x).all() and (scale>0).all()
    x=(x-x.mean(1,keepdims=True))/scale;x-=np.median(x,axis=0,keepdims=True)
    return np.ascontiguousarray(x[selected])

def morlet_power(filtered,frequencies):
    import mne
    result=np.empty((len(filtered),len(frequencies),filtered.shape[-1]//10),dtype=np.float32)
    for first in range(0,len(filtered),4):
        last=min(first+4,len(filtered))
        result[first:last]=mne.time_frequency.tfr_array_morlet(filtered[None,first:last],1000.,frequencies,
            n_cycles=7,zero_mean=False,use_fft=True,output='power',decim=10,n_jobs=1,verbose=False)[0]
    assert np.isfinite(result).all() and (result>=0).all()
    return result

def signed_bands(x,bands):
    result=np.empty((len(x),len(bands),x.shape[-1]//10),dtype=np.float32)
    for i,band in enumerate(bands):result[:,i]=sosfiltfilt(butter(4,band,btype='bandpass',fs=1000,output='sos'),x,axis=-1)[:,::10]
    assert np.isfinite(result).all()
    return result

def precompute(subject,segment):
    import mne
    p=protocol();r=p['reference'][str(subject)]
    source=Path(p['data_source'])/f'sub{subject}_comp.mat';assert sha256(source)==p['dataset_sha256'][str(subject)]
    oldpath=Path(p['feature_source'])/f's{subject}_{segment}'/'power.npy'
    assert sha256(oldpath)==r['original_cache_sha256'][segment]
    old=np.load(oldpath,mmap_mode='r')[r['selected_electrodes']]
    samples=400000 if segment=='train_full' else 200000;key='train_data' if segment=='train_full' else 'test_data'
    assert old.shape==(16,43,samples//10) and old.dtype==np.float32
    signature=dict(source_sha256=sha256(source),reference_cache_sha256=sha256(oldpath),segment=segment,indices=[0,samples],
        selected_electrodes=r['selected_electrodes'],mne=mne.__version__,recipe=p['feature_recipe'],feature_source_sha256=sha256(HERE/'features.py'))
    folder=HERE/'features'/f's{subject}_{segment}';paths={g:folder/f'{g}.npy' for g in p['groups']}
    if (folder/'metadata.json').exists():
        metadata=load_json(folder/'metadata.json');assert metadata['signature']==signature
        for g,path in paths.items():assert np.load(path,mmap_mode='r').shape==(16,p['groups'][g]['raw_features'],samples//10)
        return paths
    save_json(HERE/'preprocessing_progress.json',dict(subject=subject,segment=segment,step='linear_morlet_and_split_bands'))
    start=time.monotonic();raw=loadmat(source,variable_names=[key])[key];assert len(raw)==samples
    x=referenced_ecog(raw,r['selected_electrodes']);del raw
    low=signed_bands(x,p['feature_recipe']['signed_bands_4'])
    assert np.allclose(low[:,0],old[:,40],rtol=1e-6,atol=1e-7)
    assert np.allclose(low[:,3],old[:,42],rtol=1e-6,atol=1e-7)
    filtered=mne.filter.filter_data(x,1000.,.5,300.,verbose=False)
    filtered=mne.filter.notch_filter(filtered,1000.,np.arange(60,500,60),verbose=False)
    linear=morlet_power(filtered,p['feature_recipe']['linear_frequencies'])
    broadpath=Path(p['broadband_reference_source'])/f's{subject}_{segment}'/'morlet_broadband.npy'
    assert sha256(broadpath)==r['broad40_cache_sha256'][segment]
    broad=np.load(broadpath,mmap_mode='r')
    assert np.allclose(linear[:,[0,39]],broad[:,[0,39]],rtol=1e-5,atol=1e-7)
    folder.mkdir(parents=True,exist_ok=True)
    np.save(paths['linear_morlet'],linear);np.save(paths['hybrid_3_control'],old)
    np.save(paths['hybrid_4'],np.concatenate([old[:,:40],low],axis=1))
    save_json(folder/'metadata.json',dict(signature=signature,unchanged_high_power=True,unchanged_delta_beta_match=True,
        linear_shared_endpoints_match=True,elapsed=time.monotonic()-start))
    print(dict(subject=subject,segment=segment,unchanged_delta_beta_match=True,linear_shared_endpoints_match=True,elapsed=time.monotonic()-start),flush=True)
    return paths
