"""Create the paired density protocol and adapt the preceding fixed-budget runner."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ABL=HERE.parent.parent
OLD=ABL/'broadband_0p5_300'/'code'

def write(name,text):
    (HERE/name).write_text(text,encoding='utf-8')

def main():
    p=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    p['experiment']='RecurFlex paired 40 versus 100 broadband Morlet centers'
    p['remote_root']='/root/autodl-tmp/bci_paper_morlet100_20261008'
    p['reference_feature_source']='/root/autodl-tmp/bci_paper_broadband_features_20261008/features'
    p['initialization']='Both arms: identical seed42 16x100 model. Every non-stem tensor copied from original seed42 16x43 model; new 100-slot stem shared across arms.'
    p['input_implementation']='100 model slots in both arms; 40-center arm uses first40 slots and zeros remaining60; 100-center arm uses all100'
    p['parameters']=6980230
    p['groups']={g:dict(frequency_count=n,active_features=list(range(n)),input_gain=1) for g,n in [('morlet_40',40),('morlet_100',100)]}
    p['feature_recipe']={k:v for k,v in p['feature_recipe'].items() if k not in ('frequencies','band_edges') and not k.startswith('signed_')}
    p['feature_recipe'].update(frequencies40=np.logspace(np.log10(.5),np.log10(300),40).tolist(),frequencies100=np.logspace(np.log10(.5),np.log10(300),100).tolist())
    old_manifest=json.loads((OLD.parent/'result/server_results/input_manifest.json').read_text(encoding='utf-8'))
    for subject in old_manifest['subjects']:
        hashes=subject['groups']['morlet_broadband']['raw_feature_sha256']
        p['reference'][str(subject['subject'])]['raw40_feature_sha256']={'train_full':hashes['train'],'test':hashes['test']}
    p['previous_broadband_summary_sha256']=hashlib.sha256((OLD.parent/'result/summary.json').read_bytes()).hexdigest()
    write('protocol.json',json.dumps(p,indent=2,ensure_ascii=False)+'\n')
    worker=(OLD/'worker.py').read_text(encoding='utf-8')
    worker=worker.replace('coverage, load_json','coverage, load_json, make_model, STEM')
    worker=worker.replace('cfg = Config()','cfg = Config(features=100)')
    worker=worker.replace('model = RecurFlex(16, 43, cfg.dropout).cuda()','model = make_model("cuda")')
    worker=worker.replace('    assert initial_sha == p["reference"][str(subject)]["initial_state_sha256"]','    nonstem_sha = tensor_hash({k:v for k,v in model.state_dict().items() if k!=STEM})\n    assert sum(v.numel() for v in model.parameters())==p["parameters"]')
    worker=worker.replace('initial_state_sha256=initial_sha,','initial_state_sha256=initial_sha, nonstem_initial_sha256=nonstem_sha,')
    worker=worker.replace('model = RecurFlex().cuda()','model = RecurFlex(16,100,.1).cuda()')
    worker=worker.replace('("morlet_broadband", "signed_logbank")','("morlet_40", "morlet_100")')
    assert '16, 43' not in worker
    write('worker.py',worker)
    write('run.py',(OLD/'run.py').read_text(encoding='utf-8'))
    remote=(OLD/'remote.py').read_text(encoding='utf-8').replace('bci_paper_broadband_features_20261008','bci_paper_morlet100_20261008').replace('model=RecurFlex().cuda()','model=RecurFlex(16,100,.1).cuda()')
    write('remote.py',remote)
    write('execute_results.py',(OLD/'execute_results.py').read_text(encoding='utf-8'))
    print(json.dumps({'groups':list(p['groups']),'parameters':p['parameters'],'centers_40_to_300':{str(n):int(np.sum(np.asarray(p['feature_recipe'][f'frequencies{n}'])>=40)) for n in (40,100)}},indent=2))

if __name__=='__main__':main()
