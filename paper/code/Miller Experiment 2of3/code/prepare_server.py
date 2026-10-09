"""Create train/development/validation/test inputs independently for each subject."""
import time
import numpy as np
from common import HERE, protocol, record, raw_features, labels100, fit_statistics, transform, normalize_targets, save_json, load_json, sha256


def prepare(subject):
    r = record(subject)
    archive = HERE / "raw" / r["upload_file"]
    assert sha256(archive) == r["upload_sha256"]
    with np.load(archive) as package:
        assert set(package.files) == {"data", "development_flex"}
        raw = package["data"]
        flex = package["development_flex"]
    assert raw.shape == (r["source_samples"], r["channels"])
    assert flex.shape == (r["splits"]["train_full"][1], 5)
    features = {}
    for segment in ("train_inner", "val_inner", "train_full", "test"):
        first, last = r["splits"][segment]
        folder = HERE / "features" / subject / segment
        metadata = dict(source_sha256=r["upload_sha256"], original_source_sha256=r["original_source_sha256"],
             segment=segment, indices=[first,last], features=44, channels=r["channels"],
             main_data_source_sha256=protocol()["source_sha256"]["data.py"])
        path = folder / "features.npy"
        if path.exists() and (folder/"metadata.json").exists():
            assert load_json(folder/"metadata.json") == metadata
        else:
            save_json(HERE/"preprocessing_progress.json", dict(subject=subject, segment=segment, samples=last-first))
            start = time.monotonic()
            values = raw_features(raw[first:last])
            assert values.shape == (r["channels"],44,(last-first)//10)
            folder.mkdir(parents=True,exist_ok=True)
            np.save(path, values)
            save_json(folder/"metadata.json", metadata)
            print(dict(subject=subject,segment=segment,shape=list(values.shape),elapsed=time.monotonic()-start),flush=True)
            del values
        features[segment] = np.load(path,mmap_mode="r")
    del raw
    for phase, segment in (("validation","train_inner"),("full","train_full")):
        folder = HERE / "inputs" / subject / phase
        if (folder/"manifest.json").exists():
            saved=load_json(folder/"manifest.json")
            for name,digest in saved["files_sha256"].items():assert sha256(folder/name)==digest
            continue
        first,last=r["splits"][segment]
        truth=labels100(flex[first:last])
        stats=fit_statistics(features[segment],truth,16)
        folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/"scalers.json",stats)
        np.save(folder/"x_train.npy",transform(features[segment],stats))
        np.save(folder/"y_train.npy",normalize_targets(truth,stats))
        if phase=="validation":
            a,b=r["splits"]["val_inner"]
            np.save(folder/"x_validation.npy",transform(features["val_inner"],stats))
            np.save(folder/"validation_truth.npy",labels100(flex[a:b]))
        else:
            np.save(folder/"x_test.npy",transform(features["test"],stats))
        save_json(folder/"manifest.json",dict(subject=subject,phase=phase,
            selected_electrodes=stats["selected_electrodes"],
            files_sha256={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!="manifest.json"}))
    print(f"Prepared {subject}: labels confined to development segment",flush=True)


def main():
    p=protocol()
    for name,digest in p["source_sha256"].items():assert sha256(HERE/"main_source"/name)==digest
    for subject in p["subjects"]:prepare(subject)
    save_json(HERE/"input_manifest.json",dict(subjects=p["subjects"],features=44,test_labels_present=False,
         inputs=[load_json(HERE/"inputs"/s/phase/"manifest.json") for s in p["subjects"] for phase in ("validation","full")]))


if __name__ == "__main__":
    main()
