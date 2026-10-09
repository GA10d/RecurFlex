"""Full-record scoring after all nine model predictions have been frozen."""
import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr

HERE=Path(__file__).resolve().parent;ABL=HERE.parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'result/.plotcache'))
sys.path.insert(0,str(HERE/'main_source'))
from metrics import score
from utils import load_json,save_json,sha256

GROUPS=['linear_morlet','hybrid_3_control','hybrid_4']
LABELS={'linear_morlet':'Linear Morlet, 40 centers','hybrid_3_control':'Morlet + 3 signed bands (control)','hybrid_4':'Morlet + 4 signed bands'}

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def check_artifacts(server,p):
    frozen=load_json(server/'frozen_models_before_prediction.json');predictions=load_json(server/'frozen_predictions.json')
    assert sha256(server/'protocol.json')==predictions['protocol_sha256']==sha256(HERE/'protocol.json')
    assert predictions['model_manifest_sha256']==sha256(server/'frozen_models_before_prediction.json')
    for manifest in (load_json(server/'upload_manifest.json'),frozen['source_sha256']):
        for name,digest in manifest.items():assert sha256(server/name)==digest,name
    expected={(g,s) for g in GROUPS for s in (1,2,3)}
    assert len(frozen['fits'])==len(predictions['predictions'])==9
    assert {(v['group'],v['subject']) for v in frozen['fits']}=={(v['group'],v['subject']) for v in predictions['predictions']}==expected
    verification=load_json(server/'verification.json');assert verification['passed'] and verification['original_control_initial_function_matches']
    init={v['group']:v for v in verification['groups']}
    assert len({v['initial_state_sha256'] for v in frozen['fits'] if v['group']!='linear_morlet'})==1
    assert len({v['nonstem_initial_sha256'] for v in frozen['fits']})==1
    original_manifest=load_json(ABL/'result/server_results/input_manifest.json')
    target_hash={str(v['subject']):v['prepared_sha256']['y_train.npy'] for v in original_manifest['subjects']}
    manifest=load_json(server/'input_manifest.json')
    for subject in manifest['subjects']:
        for g in GROUPS:assert subject['groups'][g]['prepared_sha256']['y_train.npy']==target_hash[str(subject['subject'])]
    for s in (1,2,3):
        for segment in ('train_full','test'):
            metadata=load_json(server/'features'/f's{s}_{segment}'/'metadata.json')
            assert metadata['signature']['recipe']==p['feature_recipe']
            assert metadata['signature']['reference_cache_sha256']==p['reference'][str(s)]['original_cache_sha256'][segment]
            assert all(metadata[k] for k in ('unchanged_high_power','unchanged_delta_beta_match','linear_shared_endpoints_match'))
    for fit in frozen['fits']:
        g,s=fit['group'],fit['subject'];settings=p['groups'][g]
        assert fit['initial_state_sha256']==init[g]['initial_state_sha256']
        assert fit['sampling_sha256']==p['reference'][str(s)]['sampling_sha256']
        assert fit['epochs']==p['training_epochs'][str(s)] and fit['seed']==42
        assert fit['parameters']==settings['parameters']
        assert sha256(server/'fits'/f's{s}_{g}'/'model.pt')==fit['model_sha256']
        if g=='linear_morlet':assert fit['initial_state_sha256']==p['reference'][str(s)]['initial_state_sha256']
    for v in predictions['predictions']:
        g,s=v['group'],v['subject'];settings=p['groups'][g];folder=server/'fits'/f's{s}_{g}'
        assert sha256(server/v['file'])==v['sha256'] and sha256(folder/'model.pt')==v['model_sha256']
        stats_path=server/'inputs'/f's{s}'/g/'scalers.json';config=load_json(folder/'config.json');stats=load_json(stats_path)
        assert config['scalers_sha256']==sha256(stats_path) and config['settings']['features']==stats['model_slots']==settings['model_slots']
        for name in ('selected_electrodes','target_min','target_range'):assert stats[name]==p['reference'][str(s)][name]
        assert stats['slot_map']==settings['slot_map'] and stats['frequencies']==settings['raw_features']
        assert stats['feature_gains']==[1.]*40+[200.]*(settings['raw_features']-40)
        oldpath=ABL/'result/server_results/inputs'/f's{s}'/'scalers.json'
        assert sha256(oldpath)==p['reference'][str(s)]['previous_scalers_sha256']
        old=load_json(oldpath)
        if g=='hybrid_3_control':
            for name in ('feature_center','feature_scale'):
                expected_values=np.asarray(old[name]).reshape(-1,43)[stats['selected_electrodes']].reshape(-1)
                assert np.array_equal(stats[name],expected_values)
        if g=='hybrid_4':
            for name in ('feature_center','feature_scale'):
                expected_values=np.asarray(old[name]).reshape(-1,43)[stats['selected_electrodes']][:,list(range(41))+[42]]
                actual=np.asarray(stats[name]).reshape(16,44)[:,list(range(41))+[43]]
                assert np.allclose(actual,expected_values,rtol=1e-6,atol=1e-8)
        history=load_json(folder/'history.json');assert len(history)==p['training_epochs'][str(s)] and history[-1]['epoch']==len(history)
        coverage=load_json(server/'predictions'/f's{s}_{g}_coverage.json')
        assert coverage['record_length']==20000 and coverage['stride']==coverage['input_window']==2048
        assert coverage['output_ranges']==[[i,min(i+2048,20000)] for i in range(0,20000,2048)]
    return frozen,predictions

def run(result,labels_dir):
    result=Path(result);server=result/'server_results';p=load_json(HERE/'protocol.json')
    frozen,predictions=check_artifacts(server,p)
    paths={'original':ABL/'result/summary.json','broad40':ABL/'broadband_0p5_300/result/summary.json','broad100':ABL/'broadband_morlet_100/result/summary.json'}
    references={}
    for name,path in paths.items():assert sha256(path)==p['reference_summary_sha256'][name];references[name]=load_json(path)
    # All source, weights and prediction hashes above are checked before reading test labels.
    truth={};label_sha={}
    for s in (1,2,3):
        path=Path(labels_dir)/f'sub{s}_testlabels.mat';truth[s]=loadmat(path,variable_names=['test_dg'])['test_dg']
        assert truth[s].shape==(200000,5);label_sha[str(s)]=sha256(path)
        assert all(label_sha[str(s)]==r['labels_sha256'][str(s)] for r in references.values())
    fits={(v['group'],v['subject']):v for v in frozen['fits']};rows=[]
    for item in predictions['predictions']:
        g,s=item['group'],item['subject'];a=np.load(server/item['file']);native,full=a['prediction_100hz'],a['prediction_1000hz']
        assert native.shape==(20000,5) and full.shape==(200000,5) and np.array_equal(full[::10],native)
        m=score(truth[s],full);assert np.allclose([pearsonr(truth[s][:,i],full[:,i]).statistic for i in range(5)],m['r_fingers'],rtol=0,atol=1e-12)
        fit=fits[g,s]
        rows.append(dict(group=g,subject=s,seed=42,features=p['groups'][g]['raw_features'],model_slots=p['groups'][g]['model_slots'],
            **{f'r_{f}':m['r_fingers'][i] for i,f in enumerate(('thumb','index','middle','ring','little'))},
            **{k:m[k] for k in ('r4','r5','mae4','mse4','samples')},epochs=fit['epochs'],parameters=fit['parameters'],training_seconds=fit['elapsed']))
    rows.sort(key=lambda r:(GROUPS.index(r['group']),r['subject']))
    means={g:{k:float(np.mean([r[k] for r in rows if r['group']==g])) for k in ('r4','r5','mae4','mse4')} for g in GROUPS}
    lookup={(r['group'],r['subject']):r for r in rows}
    contrasts=[dict(comparison='four_minus_three_lowbands',subject=s,delta_r4=lookup['hybrid_4',s]['r4']-lookup['hybrid_3_control',s]['r4']) for s in (1,2,3)]
    contrasts.append(dict(comparison='four_minus_three_lowbands',subject='mean',delta_r4=means['hybrid_4']['r4']-means['hybrid_3_control']['r4']))
    for g,label,ref in [('morlet','High Morlet only, log40',references['original']),('hybrid','High Morlet + 3 signed bands, original',references['original']),('morlet_broadband','Broadband Morlet, log40',references['broad40']),('morlet_100','Broadband Morlet, log100',references['broad100'])]:
        contrasts.append(dict(comparison='linear_minus_'+g,subject='mean',delta_r4=means['linear_morlet']['r4']-ref['means'][g]['r4']))
    comparison=[dict(group=g,label=label,source='previous completed experiment',**ref['means'][g]) for g,label,ref in [('morlet','High Morlet only, log40',references['original']),('hybrid','High Morlet + 3 signed bands, original',references['original']),('morlet_broadband','Broadband Morlet, log40',references['broad40']),('morlet_100','Broadband Morlet, log100',references['broad100'])]]
    comparison.extend(dict(group=g,label=LABELS[g],source='new full training',**means[g]) for g in GROUPS)
    summary=dict(subjects=rows,means=means,comparison=comparison,contrasts=contrasts,best_new_group=max(GROUPS,key=lambda g:means[g]['r4']),seed=42,fresh_full_training_models=9,
        labels_sha256=label_sha,protocol_sha256=sha256(HERE/'protocol.json'),predictions_manifest_sha256=sha256(server/'frozen_predictions.json'),
        reference_summary_sha256=p['reference_summary_sha256'],primary_metric='All 200000 original test points per subject; four fingers excluding ring; equal-subject mean')
    if (result/'train_diagnostics.json').exists():
        diagnostics=load_json(result/'train_diagnostics.json')['training_diagnostics']
        assert len(diagnostics)==9 and {(r['group'],r['subject']) for r in diagnostics}=={(g,s) for g in GROUPS for s in (1,2,3)}
        assert all(r['split']=='TRAIN' and r['samples']==40000 and r['sampling_rate']==100 for r in diagnostics)
        summary['full_train_record_r4']={g:float(np.mean([r['r4'] for r in diagnostics if r['group']==g])) for g in GROUPS}
        write_csv(result/'train_test_gap.csv',[dict(group=r['group'],subject=r['subject'],train_r4=r['r4'],test_r4=lookup[r['group'],r['subject']]['r4']) for r in diagnostics])
    write_csv(result/'subject_metrics.csv',rows);write_csv(result/'mean_metrics.csv',[dict(group=g,**means[g]) for g in GROUPS])
    write_csv(result/'contrasts.csv',contrasts);write_csv(result/'all_feature_comparison.csv',comparison);save_json(result/'summary.json',summary)
    from plot_report import create_figures,write_report
    create_figures(result,summary,p);write_report(result,summary,p)
    save_json(result/'verification.json',dict(passed=True,models=9,full_training_complete=True,full_test_records=9,samples_per_record=200000,
        paired_hybrid_initialization_equal=True,original_nonstem_initialization_preserved=True,sample_windows_match_previous=True,
        unchanged_high_power_and_delta_beta_verified=True,reference_three_band_scaling_preserved=True,
        weights_and_predictions_and_source_hashes_checked=True,independent_scipy_pearson_agrees=True,full_output_coverage_checked=True,means=means))
    print({'means':means,'contrasts':contrasts},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result-dir',type=Path,default=HERE.parent/'result')
    parser.add_argument('--labels-dir',type=Path,default=HERE.parents[4]/'data/BCICIV_4_mat')
    args=parser.parse_args();run(args.result_dir,args.labels_dir)
