"""Broadband Morlet power and signed logarithmic filterbank, native 1000 Hz."""
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from common import HERE, protocol, load_json, save_json, sha256


def preprocess(raw, selected):
    import mne
    x = np.asarray(raw).T.astype(np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid ECoG data.")
    scale = x.std(1, keepdims=True)
    if (scale <= 0).any():
        raise ValueError("Constant electrode.")
    x = (x-x.mean(1, keepdims=True))/scale
    x -= np.median(x, axis=0, keepdims=True)
    # Referencing uses all electrodes before the fixed subset is extracted.
    x = np.ascontiguousarray(x[selected])
    x = mne.filter.filter_data(x, 1000., 0.5, 300., verbose=False)
    return mne.filter.notch_filter(x, 1000., np.arange(60, 500, 60), verbose=False)


def morlet_power(filtered, frequencies):
    import mne
    result = np.empty((len(filtered), 40, filtered.shape[-1]//10), dtype=np.float32)
    for first in range(0, len(filtered), 4):
        last = min(first+4, len(filtered))
        result[first:last] = mne.time_frequency.tfr_array_morlet(
            filtered[None, first:last], 1000., frequencies, n_cycles=7, zero_mean=False,
            use_fft=True, output="power", decim=10, n_jobs=1, verbose=False)[0]
    if not np.isfinite(result).all() or (result<0).any():
        raise FloatingPointError("Invalid Morlet power.")
    return result


def signed_filterbank(filtered, edges):
    result = np.empty((len(filtered), 40, filtered.shape[-1]//10), dtype=np.float32)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sos = butter(4, [lo, hi], btype="bandpass", fs=1000., output="sos")
        # Keep the sign. This is exactly sample selection, with no anti-alias filter.
        result[:, i] = sosfiltfilt(sos, filtered, axis=-1)[:, ::10]
    if not np.isfinite(result).all():
        raise FloatingPointError("Invalid signed-band features.")
    return result


def precompute(subject, segment):
    import mne
    p = protocol()
    source = Path(p["data_source"])/f"sub{subject}_comp.mat"
    assert sha256(source)==p["dataset_sha256"][str(subject)]
    selected = p["reference"][str(subject)]["selected_electrodes"]
    samples = 400000 if segment=="train_full" else 200000
    key = "train_data" if segment=="train_full" else "test_data"
    signature = dict(source_sha256=sha256(source), segment=segment, indices=[0,samples],
                     selected_electrodes=selected, mne=mne.__version__,
                     recipe=p["feature_recipe"], feature_source_sha256=sha256(HERE/"features.py"))
    folder = HERE/"features"/f"s{subject}_{segment}"
    paths = {g:folder/f"{g}.npy" for g in p["groups"]}
    if (folder/"metadata.json").exists():
        assert load_json(folder/"metadata.json")==signature
        for path in paths.values():
            f = np.load(path,mmap_mode="r")
            assert f.shape==(16,40,samples//10) and f.dtype==np.float32 and np.isfinite(f).all()
        return paths
    raw = loadmat(source, variable_names=[key])[key]
    assert len(raw)==samples
    start = time.monotonic()
    save_json(HERE/"preprocessing_progress.json",dict(subject=subject,segment=segment,step="broadband_filter"))
    filtered = preprocess(raw, selected)
    del raw
    folder.mkdir(parents=True,exist_ok=True)
    for group in p["groups"]:
        save_json(HERE/"preprocessing_progress.json",dict(subject=subject,segment=segment,step=group))
        if group=="morlet_broadband":
            features = morlet_power(filtered, p["feature_recipe"]["frequencies"])
        elif group=="signed_logbank":
            features = signed_filterbank(filtered, p["feature_recipe"]["band_edges"])
        else:
            raise ValueError(group)
        np.save(paths[group],features)
        print(dict(subject=subject,segment=segment,group=group,shape=list(features.shape),
                   min=float(features.min()),max=float(features.max()),elapsed=time.monotonic()-start),flush=True)
        del features
    save_json(folder/"metadata.json",signature)
    return paths
