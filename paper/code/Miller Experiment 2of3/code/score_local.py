"""Score all nine frozen predictions locally, after all models have finished."""
import datetime
import hashlib
import shutil
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr
from common import HERE, protocol, load_json, save_json, sha256, score, coverage, restore_1000hz, RecurFlex
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT=HERE.parent/'result'
SERVER=RESULT/'server_results'
NAMES=['Thumb','Index','Middle','Ring','Little']


def evaluate():
    p=protocol()
    assert sha256(SERVER/'protocol.json')==sha256(HERE/'protocol.json')
    frozen=load_json(SERVER/'frozen_predictions_before_scoring.json')
    assert frozen['protocol_sha256']==sha256(HERE/'protocol.json')
    assert frozen['frozen_models_sha256']==sha256(SERVER/'frozen_models_before_prediction.json')
    models=load_json(SERVER/'frozen_models_before_prediction.json')
    assert models['input_manifest_sha256']==sha256(SERVER/'input_manifest.json')
    assert models['upload_manifest_sha256']==sha256(SERVER/'upload_manifest.json')
    assert load_json(SERVER/'upload_manifest.json')==load_json(RESULT/'upload_manifest.json')
    assert not p['test_labels_uploaded'] and not frozen['test_labels_present'] and not models['test_labels_present']
    assert len(frozen['predictions'])==len(models['fits'])==9
    assert [r['subject'] for r in frozen['predictions']]==p['subjects']
    assert len(set(r['initial_state_sha256'] for r in models['fits']))==1
    for name,digest in p['source_sha256'].items():
        assert sha256(HERE/'main_source'/name)==sha256(SERVER/'main_source'/name)==digest
        assert sha256(HERE.parents[1]/'Main Experiment'/name)==digest
    for name,digest in load_json(SERVER/'upload_manifest.json').items():
        if name.endswith('.py') or name=='protocol.json':assert sha256(SERVER/name)==digest
    assert load_json(SERVER/'verification.json')['passed']
    assert load_json(SERVER/'status.json')['stage']=='awaiting_local_scoring'
    rows=[]
    remote_proof=load_json(SERVER/'artifact_verification.json') if (SERVER/'artifact_verification.json').exists() else None
    if remote_proof:assert not remote_proof['test_labels_present'] and len(remote_proof['models'])==9
    for r,pred,fit in zip(p['records'],frozen['predictions'],models['fits']):
        subject=r['subject'];folder=SERVER/'fits'/subject/'full'
        assert pred['subject']==fit['subject']==subject and pred['seed']==fit['seed']==42
        assert pred['model_sha256']==fit['model_sha256']
        assert sha256(folder/'scalers.json')==pred['scalers_sha256']
        proof=next(x for x in remote_proof['models'] if x['subject']==subject) if remote_proof else None
        if (folder/'model.pt').exists():
            assert sha256(folder/'model.pt')==pred['model_sha256']
            checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            assert checkpoint['seed']==42 and checkpoint['subject']==subject
            model=RecurFlex();model.load_state_dict(checkpoint['state_dict'],strict=True)
            assert sum(v.numel() for v in model.parameters())==6808198
            checkpoint_epoch=checkpoint['epoch'];del model,checkpoint
        else:
            assert proof and proof['model_sha256']==pred['model_sha256']
            assert proof['seed']==42 and proof['parameters']==6808198 and proof['state_shape_verified']
            checkpoint_epoch=proof['epoch']
        validation=load_json(SERVER/'fits'/subject/'validation/fit.json')
        assert fit['selected_epoch']==validation['suggested_full_epochs']
        assert checkpoint_epoch==fit['selected_epoch']
        for phase in ['validation','full']:
            phasefolder=SERVER/'fits'/subject/phase
            config=load_json(phasefolder/'config.json')['settings']
            assert (config['seed'],config['window'],config['inference_window'],config['features'])==(42,512,2048,44)
            assert len(load_json(phasefolder/'history.json'))==load_json(phasefolder/'fit.json')['epochs']
            stats=load_json(phasefolder/'scalers.json')
            assert stats['electrode_selection']=='same-time correlation on supplied training split only'
            assert stats['frequencies']==44 and stats['electrodes']==16
            assert stats['feature_gains']==[1.]*40+[200.]*4
        n=r['splits']['test'][1]-r['splits']['test'][0]
        path=SERVER/pred['file']
        if path.exists():
            assert sha256(path)==pred['prediction_sha256']
            with np.load(path) as z:native=z['prediction_100hz'];prediction=z['prediction_1000hz']
        else:
            assert proof and proof['original_npz_sha256']==pred['prediction_sha256']
            path=SERVER/'predictions'/f'{subject}_native.npz'
            assert sha256(path)==proof['native_package_sha256']
            with np.load(path) as z:native=z['prediction_100hz']
            assert hashlib.sha256(native.tobytes()).hexdigest()==proof['native_array_sha256']
            prediction=restore_1000hz(native,n)
            assert hashlib.sha256(prediction.tobytes()).hexdigest()==proof['full_array_sha256']
        assert prediction.shape==(n,5) and native.shape==(n//10,5)
        assert pred['test_raw_indices']==r['splits']['test']
        assert np.array_equal(restore_1000hz(native,n),prediction)
        assert load_json(SERVER/'predictions'/f'{subject}_coverage.json')==coverage(n//10)
        truthpath=Path(r['local_test_labels']);assert sha256(truthpath)==r['local_test_labels_sha256']
        truth=np.load(truthpath)
        metrics=score(truth,prediction)
        independent=np.array([pearsonr(truth[:,i],prediction[:,i]).statistic for i in range(5)])
        assert np.isfinite(independent).all() and np.allclose(metrics['r_fingers'],independent,rtol=0,atol=1e-10)
        rows.append(dict(subject=subject,**metrics,test_seconds=n/1000,
                    validation_best_epoch=validation['selected_epoch'],validation_r4=validation['validation_r4'],
                    full_epochs=fit['selected_epoch'],raw_channels=r['channels'],
                    selected_electrodes=load_json(folder/'scalers.json')['selected_electrodes'],
                    model_sha256=pred['model_sha256'],prediction_sha256=pred['prediction_sha256']))
    means={key:float(np.mean([r[key] for r in rows])) for key in ['r4','r5','mae4','mse4']}
    means['r_fingers']=np.mean([r['r_fingers'] for r in rows],axis=0).tolist()
    value=dict(experiment=p['experiment'],scoring_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               primary_metric='r5: all five fingers; equal nine-subject mean',
               auxiliary_metric='r4: thumb, index, middle, little; equal nine-subject mean',
               split='within subject, chronological last one-third test, 2 s guard, inner validation and refit',
               seed=42,features=44,electrodes=16,train_window=512,inference_window=2048,
               subjects=rows,average=means,
               r4_subject_sd=float(np.std([r['r4'] for r in rows],ddof=1)),
               r5_subject_sd=float(np.std([r['r5'] for r in rows],ddof=1)),
               samples=sum(r['samples'] for r in rows),
               frozen_predictions_sha256=sha256(SERVER/'frozen_predictions_before_scoring.json'),
               all_nine_scored=True,independent_pearson_check=True)
    save_json(RESULT/'test_results.json',value)
    return value


def figures(value):
    output=RESULT/'figures';output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    rows=value['subjects'];subjects=[r['subject'] for r in rows];x=np.arange(9)
    fig,ax=plt.subplots(figsize=(10,4.3))
    ax.bar(x-.18,[r['r4'] for r in rows],width=.36,label='4 fingers (excluding ring)',color='#287F8E')
    ax.bar(x+.18,[r['r5'] for r in rows],width=.36,label='All 5 fingers',color='#DB8F39')
    ax.axhline(value['average']['r4'],ls='--',lw=1,color='#287F8E',label=f"Mean r4 = {value['average']['r4']:.4f}")
    ax.axhline(value['average']['r5'],ls=':',lw=1,color='#DB8F39',label=f"Mean r5 = {value['average']['r5']:.4f}")
    ax.set(xticks=x,xticklabels=subjects,ylabel='Test Pearson r',xlabel='Miller fingerflex subject',ylim=(min(0,min(min(r['r4'],r['r5']) for r in rows)-.05),1))
    ax.set_title('Chronological 2/3 development, 1/3 test')
    ax.legend(fontsize=9,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.10));fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'subject_test_correlations.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));matrix=np.array([r['r_fingers'] for r in rows])
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set(xticks=np.arange(5),xticklabels=NAMES,yticks=x,yticklabels=subjects,xlabel='Finger',ylabel='Subject')
    ax.set_title('Chronological 2/3 development, 1/3 test')
    for i in range(9):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(im,ax=ax,label='Test Pearson r');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'finger_test_correlations.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,8),sharey=True)
    for row,ax in zip(rows,axes.flat):
        history=load_json(SERVER/'fits'/row['subject']/'validation/history.json')
        ax.plot([h['epoch'] for h in history],[h['validation_r4'] for h in history],color='#287F8E')
        ax.axvline(row['validation_best_epoch'],color='#DB8F39',ls='--',lw=1)
        ax.set(title=f"{row['subject']} | selected {row['validation_best_epoch']}",xlabel='Inner training epoch',ylabel='Validation r4')
    fig.tight_layout();fig.savefig(output/'validation_selection.png',dpi=180);plt.close(fig)


def report(value):
    rows=value['subjects'];mean=value['average']
    text=f"""# Miller fingerflex 九人前 2/3 训练测试

九名被试已全部训练、完成最后 1/3 时间段的测试。九人等权的五指平均 **r5 = {mean['r5']:.6f}**；排除无名指的四指平均 **r4 = {mean['r4']:.6f}**。r4 延续原 BCI 实验的四指口径（拇指、食指、中指、小指）；r5 给出这个数据集的完整五指口径。

| 被试 | 测试 r4 | 测试 r5 | 内层最佳轮数 | 最终训练轮数 | 测试时长 / s |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for r in rows:text+=f"| {r['subject']} | {r['r4']:.6f} | {r['r5']:.6f} | {r['validation_best_epoch']} | {r['full_epochs']} | {r['test_seconds']:.2f} |\n"
    text+=f"""
## 实验设置

模型为 Main Experiment 原样快照：40 个对数间隔的 40–300 Hz Morlet 功率特征，加 0.5–4、4–8、8–13、13–30 Hz 四路带符号波形。输入采用训练段选出的 16 个电极，主干 RecurFlex 有 6,808,198 个参数。训练使用 512 个 100 Hz 时间点，推理使用连续、互不重叠的 2048 点块。种子固定 42，单模型；输出直接还原为原始 1000 Hz 时间轴评分。

每人前 2/3 为开发段、最后约 1/3 原始时间点为测试，测试前留 2 s 隔离。前段开发数据中再划分后 20% 做验证，并留 2 s 隔离。最多验证训练 100 轮，耐心值 25；按验证 r4 选最佳轮数，再按开发段／内层训练段长度比折算轮数，从头在整个开发段训练最终模型。电极选择、特征 RobustScaler 和目标量程只由对应训练段拟合；原始 ECoG 的通道均值／标准差按 Main Experiment 的现有实现，在每个独立片段内计算。九人全部沿用同一架构、优化器和超参数。

本次上传包只包含开发段标签，当前训练和推理流程不读取测试标签；新划分的测试标签单独保存在本地。旧 80/20 实验使用过更长的训练段，本次全部从种子 42 随机初始化重新训练，没有加载旧权重。服务器先保存九个最终权重及其哈希，再完成九人的全部预测；本地取得冻结的预测后才评分。所有原始测试时间点均计分，没有挑选片段。每个被试先计算逐指 Pearson r，再计算九人的等权均值；没有把九人的信号拼起来算一次相关系数。45 个逐指相关系数均通过 scipy.stats.pearsonr 独立复核。

这是九名被试各自重新训练后的**被试内时间划分评估**，不是直接迁移已有三个 BCI 权重的跨被试测试，也不是这个九人子集的官方划分。原始数据包 `BCI_Competion4_dataset4_data_fingerflexions/readme.txt` 明确说明 BCI IV Dataset 4 来源于本库的 fingerflex 数据，因此这里是九人扩展评估，不能称为完全独立的数据来源。它与 BCI Competition IV 的成绩采用不同划分，应分别列出。

## 结果文件

- [逐被试、逐指完整结果](test_results.json)
- [被试平均相关图](figures/subject_test_correlations.png)
- [逐指相关热图](figures/finger_test_correlations.png)
- [内层验证选轮数](figures/validation_selection.png)
- [服务器原始清单、已核验的模型信息、预测和日志](server_results/)
- [训练前协议](../code/protocol.json)
- [评分前冻结的模型](server_results/frozen_models_before_prediction.json)
- [评分前冻结的预测](server_results/frozen_predictions_before_scoring.json)

平均逐指 r（拇指／食指／中指／无名指／小指）：{', '.join(f'{x:.6f}' for x in mean['r_fingers'])}。

总测试样本数：{value['samples']:,}，原始采样率 1000 Hz。r4 的被试间标准差为 {value['r4_subject_sd']:.6f}，r5 为 {value['r5_subject_sd']:.6f}。

全部 18 个验证／最终权重保存在服务器 `/root/autodl-tmp/bci_miller_fingerflex_2of3_20261008/fits/`。本地保存了九人原始 100 Hz 预测；按同一个 Main Experiment 函数还原的 1000 Hz 预测逐字节 SHA-256 与服务器结果一致。完整权重可用 `remote.py fetch` 另行获取。
"""
    (RESULT/'README.md').write_text(text,encoding='utf-8')
    paper_result=HERE.parents[2]/'result'/'miller_fingerflex_9subjects_2of3'
    paper_result.mkdir(parents=True,exist_ok=True)
    shutil.copy2(RESULT/'test_results.json',paper_result/'test_results.json')
    for path in (RESULT/'figures').iterdir():shutil.copy2(path,paper_result/path.name)
    summary=text.replace('(figures/','(').replace('(server_results/)', '(../../code/Miller%20Experiment%202of3/result/server_results/)')
    summary=summary.replace('(../code/protocol.json)', '(../../code/Miller%20Experiment%202of3/code/protocol.json)')
    summary=summary.replace('(server_results/', '(../../code/Miller%20Experiment%202of3/result/server_results/')
    (paper_result/'README.md').write_text(summary,encoding='utf-8')


if __name__=='__main__':
    value=evaluate();figures(value);report(value)
    print(value['average'],flush=True)
