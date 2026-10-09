"""Pin two requested feature recipes and a paired low-band control."""
import json
import hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ABL=HERE.parent.parent

def main():
    p=json.loads((ABL/'broadband_0p5_300/code/protocol.json').read_text(encoding='utf-8'))
    p.update(experiment='Linear broadband Morlet and split signed low-frequency bands',
        remote_root='/root/autodl-tmp/bci_paper_linear_split_20261008',
        feature_source='/root/autodl-tmp/bci_backbones_20261008/cache',
        broadband_reference_source='/root/autodl-tmp/bci_paper_broadband_features_20261008/features',models=9)
    p['groups']={
        'linear_morlet':dict(model_slots=43,raw_features=40,slot_map=list(range(40)),signed_start=40,parameters=6805126),
        'hybrid_3_control':dict(model_slots=44,raw_features=43,slot_map=list(range(42))+[43],signed_start=40,parameters=6808198),
        'hybrid_4':dict(model_slots=44,raw_features=44,slot_map=list(range(44)),signed_start=40,parameters=6808198)}
    for settings in p['groups'].values():settings['active_features']=settings['slot_map']
    p['feature_recipe']=dict(sampling_rate=1000,linear_broadband=[.5,300],
        linear_frequencies=np.linspace(.5,300,40).tolist(),high_broadband=[40,300],
        high_frequencies=np.logspace(np.log10(40),np.log10(300),40).tolist(),
        notches=list(range(60,500,60)),n_cycles=7,zero_mean=False,decimation=10,
        signed_bands_3=[[.5,4],[4,13],[13,30]],signed_bands_4=[[.5,4],[4,8],[8,13],[13,30]],
        signed_filter_order=4,signed_filter='Butterworth sosfiltfilt on standardized median-referenced ECoG before high-band filtering',
        signed_gain=200,morlet_gain=1)
    p['initialization']='Linear arm: original seed42 43-slot model. Both hybrid arms: identical 44-slot model with all common tensors and stem weights embedded from original seed42 43-slot model. Extra alpha input weight retains seed42 random initialization; control alpha slot zero.'
    p['input_implementation']='Linear: 40 active slots in43. Hybrid44 canonical slots 40=delta,41=theta or theta+alpha,42=alpha or zero,43=beta.'
    p['scalers']='TRAIN RobustScaler (.1,.9) percentiles, unit_variance=True; signed features multiplied by200; targets shared with preceding ablation'
    p.pop('low_frequency_gain',None)
    original=json.loads((ABL/'result/server_results/input_manifest.json').read_text(encoding='utf-8'))
    for s in original['subjects']:p['reference'][str(s['subject'])]['original_cache_sha256']=s['original_cache_sha256']
    broad=json.loads((ABL/'broadband_0p5_300/result/server_results/input_manifest.json').read_text(encoding='utf-8'))
    for s in broad['subjects']:
        h=s['groups']['morlet_broadband']['raw_feature_sha256']
        p['reference'][str(s['subject'])]['broad40_cache_sha256']={'train_full':h['train'],'test':h['test']}
    paths={'original':ABL/'result/summary.json','broad40':ABL/'broadband_0p5_300/result/summary.json','broad100':ABL/'broadband_morlet_100/result/summary.json'}
    p['reference_summary_sha256']={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in paths.items()}
    p.pop('previous_summary_sha256',None)
    (HERE/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    old=ABL/'broadband_morlet_100/code'
    worker=(old/'worker.py').read_text(encoding='utf-8')
    worker=worker.replace('cfg = Config(features=100)','cfg = Config(features=p["groups"][group]["model_slots"])')
    worker=worker.replace('make_model("cuda")','make_model(group,"cuda")')
    worker=worker.replace('p["parameters"]','p["groups"][group]["parameters"]')
    worker=worker.replace('RecurFlex(16,100,.1)','RecurFlex(16,protocol()["groups"][group]["model_slots"],.1)')
    worker=worker.replace('("morlet_40", "morlet_100")','("linear_morlet", "hybrid_3_control", "hybrid_4")')
    (HERE/'worker.py').write_text(worker,encoding='utf-8')
    runner=(old/'run.py').read_text(encoding='utf-8').replace('six','nine').replace('pending=6','pending=9').replace('models=6,predictions=6','models=9,predictions=9')
    runner=runner.replace('assert len({v["initial_state_sha256"] for v in fits}) == 1','assert len({v["initial_state_sha256"] for v in fits if v["group"]!="linear_morlet"}) == 1\n    assert len({v["nonstem_initial_sha256"] for v in fits}) == 1')
    (HERE/'run.py').write_text(runner,encoding='utf-8')
    remote=(old/'remote.py').read_text(encoding='utf-8').replace('bci_paper_morlet100_20261008','bci_paper_linear_split_20261008').replace("'fresh_fits':6","'fresh_fits':9").replace('RecurFlex(16,100,.1)','RecurFlex(16,p["groups"][g]["model_slots"],.1)')
    (HERE/'remote.py').write_text(remote,encoding='utf-8')
    print({'groups':list(p['groups']),'models':p['models'],'linear_step_hz':p['feature_recipe']['linear_frequencies'][1]-.5})

if __name__=='__main__':main()
