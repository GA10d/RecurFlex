"""Run all nine validation/refit jobs and freeze every prediction before scoring."""
import datetime
import os
import subprocess
import sys
import time
import traceback
from common import HERE, protocol, save_json, load_json, sha256


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p=protocol();active={};completed=[];handles=[]
    try:
        save_json(HERE/'status.json',dict(stage='preprocessing',utc=utc(),completed=[]))
        subprocess.run([sys.executable,'-u',str(HERE/'prepare_server.py')],cwd=HERE,check=True)
        save_json(HERE/'status.json',dict(stage='verification',utc=utc(),completed=[]))
        subprocess.run([sys.executable,'-u',str(HERE/'verify.py')],cwd=HERE,check=True)
        queue=list(p['subjects']);(HERE/'logs').mkdir(exist_ok=True)
        while queue or active:
            for subject,child in list(active.items()):
                code=child.poll()
                if code is not None:
                    if code:raise RuntimeError(f'{subject} worker exited with {code}; see logs/{subject}.log')
                    completed.append(subject);del active[subject]
            while queue and len(active)<p['workers']:
                subject=queue.pop(0)
                handle=(HERE/'logs'/f'{subject}.log').open('a');handles.append(handle)
                active[subject]=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),'--subject',subject],
                     cwd=HERE,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            save_json(HERE/'status.json',dict(stage='training',utc=utc(),completed=completed,
                       active={s:c.pid for s,c in active.items()},queued=queue))
            if active:time.sleep(10)
        fits=[load_json(HERE/'fits'/s/'full'/'fit.json') for s in p['subjects']]
        assert len(fits)==9 and all(r['seed']==42 and r['phase']=='full' for r in fits)
        assert len(set(r['initial_state_sha256'] for r in fits))==1
        for r in fits:assert sha256(HERE/'fits'/r['subject']/'full/model.pt')==r['model_sha256']
        frozen=dict(utc=utc(),protocol_sha256=sha256(HERE/'protocol.json'),
                    input_manifest_sha256=sha256(HERE/'input_manifest.json'),
                    upload_manifest_sha256=sha256(HERE/'upload_manifest.json'),fits=fits,test_labels_present=False)
        save_json(HERE/'frozen_models_before_prediction.json',frozen)
        predictions=[]
        for subject in p['subjects']:
            save_json(HERE/'status.json',dict(stage='inference',utc=utc(),completed=completed,predicting=subject))
            command="from worker import infer;from common import save_json,HERE;import sys;save_json(HERE/'predictions'/(sys.argv[1]+'_manifest.json'),infer(sys.argv[1]))"
            subprocess.run([sys.executable,'-u','-c',command,subject],cwd=HERE,check=True)
            predictions.append(load_json(HERE/'predictions'/f'{subject}_manifest.json'))
        save_json(HERE/'frozen_predictions_before_scoring.json',dict(utc=utc(),predictions=predictions,
            frozen_models_sha256=sha256(HERE/'frozen_models_before_prediction.json'),
            protocol_sha256=sha256(HERE/'protocol.json'),test_labels_present=False))
        save_json(HERE/'status.json',dict(stage='awaiting_local_scoring',utc=utc(),completed=completed,predictions=9))
    except BaseException:
        for child in active.values():
            if child.poll() is None:child.terminate()
        save_json(HERE/'status.json',dict(stage='failed',utc=utc(),completed=completed,error=traceback.format_exc()))
        raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':main()
