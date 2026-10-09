"""Score the frozen paired frequency-density experiment on all original test points."""
import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr

HERE=Path(__file__).resolve().parent
ABL=HERE.parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'result/.plotcache'))
sys.path.insert(0,str(HERE/'main_source'))
from metrics import score
from utils import load_json,save_json,sha256

GROUPS=['morlet_40','morlet_100']
LABELS={'morlet_40':'Broadband Morlet, 40 centers','morlet_100':'Broadband Morlet, 100 centers'}

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def check_artifacts(server,p):
    assert sha256(server/'protocol.json')==sha256(HERE/'protocol.json')
    frozen=load_json(server/'frozen_models_before_prediction.json')
    predictions=load_json(server/'frozen_predictions.json')
    assert predictions['model_manifest_sha256']==sha256(server/'frozen_models_before_prediction.json')
    assert predictions['protocol_sha256']==sha256(HERE/'protocol.json')
    for manifest in (load_json(server/'upload_manifest.json'),frozen['source_sha256']):
        for name,digest in manifest.items():assert sha256(server/name)==digest,name
    expected={(g,s) for g in GROUPS for s in (1,2,3)}
    assert len(predictions['predictions'])==len(frozen['fits'])==6
    assert {(v['group'],v['subject']) for v in predictions['predictions']}==expected
    assert {(v['group'],v['subject']) for v in frozen['fits']}==expected
    verification=load_json(server/'verification.json')
    assert verification['passed'] and verification['original_nonstem_initialization_preserved']
    assert len({v['initial_state_sha256'] for v in frozen['fits']})==1
    assert len({v['nonstem_initial_sha256'] for v in frozen['fits']})==1
    assert {v['initial_state_sha256'] for v in frozen['fits']}=={v['initial_state_sha256'] for v in verification['groups']}
    for fit in frozen['fits']:
        s=fit['subject']
        assert fit['sampling_sha256']==p['reference'][str(s)]['sampling_sha256']
        assert fit['epochs']==p['training_epochs'][str(s)] and fit['seed']==42
        assert fit['parameters']==p['parameters']==6980230
        assert sha256(server/'fits'/f"s{s}_{fit['group']}"/'model.pt')==fit['model_sha256']
    inputs=load_json(server/'input_manifest.json')
    for subject in inputs['subjects']:
        s=subject['subject'];ref=p['reference'][str(s)]['raw40_feature_sha256']
        assert subject['groups']['morlet_40']['raw_feature_sha256']=={'train':ref['train_full'],'test':ref['test']}
    for s in (1,2,3):
        for segment in ('train_full','test'):
            metadata=load_json(server/'features'/f's{s}_{segment}'/'metadata.json')
            assert metadata['shared_centers_match']
            assert metadata['signature']['recipe']==p['feature_recipe']
    for v in predictions['predictions']:
        g,s=v['group'],v['subject'];folder=server/'fits'/f's{s}_{g}'
        assert sha256(server/v['file'])==v['sha256'] and sha256(folder/'model.pt')==v['model_sha256']
        config=load_json(folder/'config.json');stats_path=server/'inputs'/f's{s}'/g/'scalers.json'
        assert config['scalers_sha256']==sha256(stats_path) and config['settings']['features']==100
        stats=load_json(stats_path)
        for name in ('selected_electrodes','target_min','target_range'):assert stats[name]==p['reference'][str(s)][name]
        assert stats['input_gain']==1 and stats['model_slots']==100 and stats['frequencies']==p['groups'][g]['frequency_count']
        if g=='morlet_40':
            previous_stats=load_json(ABL/'broadband_0p5_300/result/server_results/inputs'/f's{s}'/'morlet_broadband/scalers.json')
            for name in ('feature_center','feature_scale'):assert stats[name]==previous_stats[name]
        history=load_json(folder/'history.json')
        assert len(history)==p['training_epochs'][str(s)] and history[-1]['epoch']==len(history)
        coverage=load_json(server/'predictions'/f's{s}_{g}_coverage.json')
        assert coverage['record_length']==20000 and coverage['stride']==coverage['input_window']==2048
        assert coverage['output_ranges']==[[i,min(i+2048,20000)] for i in range(0,20000,2048)]
    return frozen,predictions

def run(result,labels_dir):
    result=Path(result);server=result/'server_results';p=load_json(HERE/'protocol.json')
    frozen,predictions=check_artifacts(server,p)
    original_path=ABL/'result/summary.json';broad_path=ABL/'broadband_0p5_300/result/summary.json'
    assert sha256(original_path)==p['previous_summary_sha256']
    assert sha256(broad_path)==p['previous_broadband_summary_sha256']
    original=load_json(original_path);broad=load_json(broad_path)
    # All new model and prediction hashes above are verified before test labels are read.
    truth={};label_sha={}
    for s in (1,2,3):
        path=Path(labels_dir)/f'sub{s}_testlabels.mat'
        truth[s]=loadmat(path,variable_names=['test_dg'])['test_dg'];assert truth[s].shape==(200000,5)
        label_sha[str(s)]=sha256(path);assert label_sha[str(s)]==original['labels_sha256'][str(s)]==broad['labels_sha256'][str(s)]
    fits={(v['group'],v['subject']):v for v in frozen['fits']};rows=[]
    for item in predictions['predictions']:
        g,s=item['group'],item['subject'];arrays=np.load(server/item['file'])
        native,full=arrays['prediction_100hz'],arrays['prediction_1000hz']
        assert native.shape==(20000,5) and full.shape==(200000,5) and np.array_equal(full[::10],native)
        m=score(truth[s],full)
        assert np.allclose([pearsonr(truth[s][:,i],full[:,i]).statistic for i in range(5)],m['r_fingers'],rtol=0,atol=1e-12)
        fit=fits[g,s]
        rows.append(dict(group=g,subject=s,seed=42,frequency_count=p['groups'][g]['frequency_count'],model_slots=100,
            **{f'r_{f}':m['r_fingers'][i] for i,f in enumerate(('thumb','index','middle','ring','little'))},
            **{k:m[k] for k in ('r4','r5','mae4','mse4','samples')},epochs=fit['epochs'],parameters=fit['parameters'],training_seconds=fit['elapsed']))
    rows.sort(key=lambda r:(GROUPS.index(r['group']),r['subject']))
    means={g:{k:float(np.mean([r[k] for r in rows if r['group']==g])) for k in ('r4','r5','mae4','mse4')} for g in GROUPS}
    contrasts=[dict(subject=s,delta_r4=next(r['r4'] for r in rows if (r['group'],r['subject'])==('morlet_100',s))-next(r['r4'] for r in rows if (r['group'],r['subject'])==('morlet_40',s))) for s in (1,2,3)]
    delta=means['morlet_100']['r4']-means['morlet_40']['r4'];contrasts.append(dict(subject='mean',delta_r4=delta))
    comparison=[dict(group=g,label=label,source='previous 43-slot model',**original['means'][g]) for g,label in [('morlet','Morlet 40-300 Hz, 40 centers'),('hybrid','Morlet 40-300 Hz + signed low bands')]]
    comparison.append(dict(group='broadband_old40',label='Broadband 40 centers, original model',source='previous 43-slot model',**broad['means']['morlet_broadband']))
    comparison.extend(dict(group=g,label=LABELS[g],source='new paired 100-slot model',**means[g]) for g in GROUPS)
    summary=dict(subjects=rows,means=means,delta_100_minus_40=delta,comparison=comparison,seed=42,fresh_full_training_models=6,
        protocol_sha256=sha256(HERE/'protocol.json'),labels_sha256=label_sha,
        predictions_manifest_sha256=sha256(server/'frozen_predictions.json'),
        primary_metric='All 200000 original test points per subject; four fingers excluding ring; equal-subject mean',
        previous_summary_sha256=p['previous_summary_sha256'],previous_broadband_summary_sha256=p['previous_broadband_summary_sha256'])
    diagnostic_path=result/'train_diagnostics.json'
    if diagnostic_path.exists():
        diagnostics=load_json(diagnostic_path)['training_diagnostics']
        assert len(diagnostics)==6 and {(r['group'],r['subject']) for r in diagnostics}=={(g,s) for g in GROUPS for s in (1,2,3)}
        assert all(r['samples']==40000 and r['sampling_rate']==100 and r['split']=='TRAIN' for r in diagnostics)
        summary['full_train_record_r4']={g:float(np.mean([r['r4'] for r in diagnostics if r['group']==g])) for g in GROUPS}
        summary['train_diagnostics_sha256']=sha256(diagnostic_path)
        write_csv(result/'train_test_gap.csv',[dict(group=r['group'],subject=r['subject'],train_r4=r['r4'],test_r4=next(v['r4'] for v in rows if (v['group'],v['subject'])==(r['group'],r['subject']))) for r in diagnostics])
    write_csv(result/'subject_metrics.csv',rows);write_csv(result/'mean_metrics.csv',[dict(group=g,**means[g]) for g in GROUPS])
    write_csv(result/'contrasts.csv',contrasts);write_csv(result/'all_feature_comparison.csv',comparison)
    save_json(result/'summary.json',summary)
    from plot_report import create_figures,write_report
    create_figures(result,summary,p);write_report(result,summary,p)
    save_json(result/'verification.json',dict(passed=True,models=6,full_training_complete=True,full_test_records=6,
        samples_per_record=200000,paired_initialization_equal=True,original_nonstem_initialization_preserved=True,
        sample_windows_match_previous=True,weights_and_predictions_and_source_hashes_checked=True,
        shared_morlet_centers_match_previous=True,reference_40_scaling_exactly_preserved=True,
        independent_scipy_pearson_agrees=True,full_output_coverage_checked=True,means=means))
    print({'means':means,'delta_100_minus_40':delta},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result-dir',type=Path,default=HERE.parent/'result')
    parser.add_argument('--labels-dir',type=Path,default=HERE.parents[4]/'data/BCICIV_4_mat')
    args=parser.parse_args();run(args.result_dir,args.labels_dir)
