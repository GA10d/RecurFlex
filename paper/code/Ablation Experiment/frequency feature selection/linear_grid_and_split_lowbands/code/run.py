"""Two workers train nine fixed-budget models before frozen test prediction."""
import datetime
import os
import subprocess
import sys
import time
import traceback
from common import HERE, protocol, prepare_inputs, load_json, save_json, sha256


def run():
    p = protocol()
    save_json(HERE / "status.json", dict(stage="preprocessing", completed=[], active=[], pending=9))
    prepare_inputs()
    subprocess.run([sys.executable, str(HERE / "verify.py")], check=True)
    jobs = [(g,s) for s in p["subjects"] for g in p["groups"]]
    pending = list(jobs)
    active = {}
    done = []
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1")
    while pending or active:
        for key, (child, stream) in list(active.items()):
            code = child.poll()
            if code is not None:
                stream.close()
                del active[key]
                if code:
                    raise RuntimeError(f"Training failed: {key}; see corresponding log")
                group, subject = key
                fit = load_json(HERE / "fits" / f"s{subject}_{group}" / "fit.json")
                assert fit["model_sha256"] == sha256(HERE / "fits" / f"s{subject}_{group}" / "model.pt")
                done.append(fit)
        while pending and len(active) < p["workers"]:
            group, subject = pending.pop(0)
            stream = (logs / f"s{subject}_{group}.log").open("a")
            child = subprocess.Popen([sys.executable,"-u",str(HERE/"worker.py"),"--group",group,"--subject",str(subject)],
                                     cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            active[(group,subject)] = (child,stream)
        save_json(HERE / "status.json", dict(stage="training", completed=done,
                  active=[dict(group=g,subject=s,pid=child.pid) for (g,s),(child,_) in active.items()], pending=len(pending)))
        if pending or active:
            time.sleep(3)
    fits = [load_json(HERE/"fits"/f"s{s}_{g}"/"fit.json") for g,s in jobs]
    assert len({v["initial_state_sha256"] for v in fits if v["group"]!="linear_morlet"}) == 1
    assert len({v["nonstem_initial_sha256"] for v in fits}) == 1
    for subject in p["subjects"]:
        assert len({v["sampling_sha256"] for v in fits if v["subject"]==subject}) == 1
    save_json(HERE/"frozen_models_before_prediction.json",dict(fits=fits,
              protocol_sha256=sha256(HERE/"protocol.json"),
              source_sha256={str(f.relative_to(HERE)):sha256(f) for f in HERE.rglob("*.py") if "__pycache__" not in str(f)},
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    save_json(HERE/"status.json",dict(stage="prediction",completed=done,active=[],pending=0))
    from worker import infer
    predictions = []
    for group,subject in jobs:
        predictions.append(infer(group,subject))
    save_json(HERE/"frozen_predictions.json",dict(predictions=predictions,
              model_manifest_sha256=sha256(HERE/"frozen_models_before_prediction.json"),
              protocol_sha256=sha256(HERE/"protocol.json"),
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    save_json(HERE/"status.json",dict(stage="awaiting_local_scoring",completed=fits,models=9,predictions=9))
    print("All nine training runs and complete test predictions finished.",flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        save_json(HERE/"status.json",dict(stage="failed",error=traceback.format_exc()))
        raise
