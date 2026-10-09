"""Static publication figures and evidence-linked prose for the paired experiment."""
import numpy as np
from score_and_report import GROUPS,LABELS,write_csv,load_json

COLORS={'morlet_40':'#648BB1','morlet_100':'#DB8645'}

def create_figures(result,summary,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(result/f'{name}.{ext}',dpi=350,bbox_inches='tight')
        plt.close(fig)
    rows,means=summary['subjects'],summary['means'];lookup={(r['group'],r['subject']):r for r in rows}
    fig,axs=plt.subplots(1,3,figsize=(12.5,4.1),gridspec_kw={'width_ratios':[1.1,.85,1.25]})
    for i,g in enumerate(GROUPS):
        values=[lookup[g,s]['r4'] for s in (1,2,3)]
        bars=axs[0].bar(np.arange(3)+(i-.5)*.32,values,.3,color=COLORS[g],label=f"{p['groups'][g]['frequency_count']} centers")
        axs[0].bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=8)
    axs[0].set_xticks(range(3),['S1','S2','S3']);axs[0].set_ylabel('Four-finger Pearson r');axs[0].legend(frameon=False,fontsize=9)
    axs[0].set_title('(a) Complete test recordings',loc='left',fontweight='bold')
    bars=axs[1].bar(range(2),[means[g]['r4'] for g in GROUPS],width=.55,color=[COLORS[g] for g in GROUPS])
    axs[1].bar_label(bars,labels=[f"{means[g]['r4']:.4f}" for g in GROUPS],padding=4)
    axs[1].set_xticks(range(2),['40 centers','100 centers']);axs[1].set_ylabel('Equal-subject mean r')
    axs[1].set_title('(b) Paired mean',loc='left',fontweight='bold')
    for ax in axs[:2]:
        ax.set_ylim(min(0.,min(r['r4'] for r in rows)-.05),min(1.,max(r['r4'] for r in rows)+.15));ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fingers=['thumb','index','middle','ring','little']
    matrix=np.array([[np.mean([lookup[g,s][f'r_{f}'] for s in (1,2,3)]) for f in fingers] for g in GROUPS])
    im=axs[2].imshow(matrix,cmap='YlGnBu',vmin=min(0.,matrix.min()),vmax=1.,aspect='auto')
    axs[2].set_xticks(range(5),['Thumb','Index','Middle','Ring*','Little'],rotation=30,ha='right');axs[2].set_yticks(range(2),['40 centers','100 centers'])
    for i in range(2):
        for j in range(5):axs[2].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black')
    axs[2].set_title('(c) Mean by finger',loc='left',fontweight='bold');fig.colorbar(im,ax=axs[2],fraction=.04,pad=.03)
    fig.text(.5,.025,'0.5-300 Hz | Same 100-slot model and initialization | Seed 42 | *Ring excluded from primary metric',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'density_comparison')
    comparison=summary['comparison'];fig,ax=plt.subplots(figsize=(9.5,4.2))
    colors=['#A3AEC0','#B8AA92','#C4CDD7',COLORS['morlet_40'],COLORS['morlet_100']]
    bars=ax.barh(range(len(comparison)),[r['r4'] for r in comparison],color=colors,height=.6)
    ax.bar_label(bars,labels=[f"{r['r4']:.4f}" for r in comparison],padding=5)
    ax.set_yticks(range(len(comparison)),[r['label'] for r in comparison]);ax.invert_yaxis()
    ax.set_xlim(0,min(1.,max(r['r4'] for r in comparison)+.1));ax.set_xlabel('Mean four-finger Pearson r');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Frequency density and preceding feature representations',fontweight='bold')
    fig.text(.5,.02,'First 3: previous 43-slot model | Last 2: new paired 100-slot model | Complete test recordings',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1));save(fig,'all_feature_comparison')
    fig,axs=plt.subplots(1,3,figsize=(12,3.4))
    for ax,s in zip(axs,(1,2,3)):
        for g in GROUPS:
            history=load_json(result/'server_results/fits'/f's{s}_{g}'/'history.json')
            ax.plot([v['epoch'] for v in history],[v['loss'] for v in history],color=COLORS[g],label=f"{p['groups'][g]['frequency_count']} centers",lw=1.5)
        ax.set_title(f'S{s}');ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    axs[0].set_ylabel('Training loss (MSE + cosine)');axs[-1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'training_curves')
    grids=[(np.geomspace(40,300,40),'Original high-frequency grid', '#A3AEC0'),
           (np.asarray(p['feature_recipe']['frequencies40']),'Broadband, 40 centers',COLORS['morlet_40']),
           (np.asarray(p['feature_recipe']['frequencies100']),'Broadband, 100 centers',COLORS['morlet_100'])]
    write_csv(result/'frequency_grid.csv',[dict(grid=label,index=i+1,center_hz=float(f),temporal_envelope_fwhm_seconds=float(7*np.sqrt(2*np.log(2))/(np.pi*f))) for frequencies,label,_ in grids for i,f in enumerate(frequencies)])
    fig,ax=plt.subplots(figsize=(10.4,3.4));ax.set_xscale('log');ax.axvspan(.5,40,color='#F0F0F0')
    for y,(f,label,color) in enumerate(grids):
        ax.scatter(f,np.full(len(f),y),s=17 if len(f)==40 else 12,color=color,zorder=3)
        ax.text(np.sqrt(.5*40),y+.17,f'{np.sum(f<40)} centers below 40 Hz',ha='center',fontsize=9,color='#555555')
        ax.text(np.sqrt(40*300),y+.17,f'{np.sum(f>=40)} centers at or above 40 Hz',ha='center',fontsize=9,color='#555555')
    ax.set_yticks(range(3),[v[1] for v in grids]);ax.set_ylim(-.5,2.65);ax.set_xlim(.43,345);ax.axvline(40,ls=':',lw=1,color='#666666')
    ticks=[.5,1,2,4,8,16,30,40,70,150,300];ax.set_xticks(ticks,[str(v) for v in ticks]);ax.grid(axis='x',alpha=.12)
    ax.set_xlabel('Frequency (Hz), logarithmic axis');ax.set_title('Spectral sampling density with fixed wavelet cycles',fontweight='bold')
    fig.tight_layout();save(fig,'frequency_design')

def write_report(result,summary,p):
    means=summary['means'];rows=summary['subjects'];delta=summary['delta_100_minus_40']
    table='\n'.join(f"| {p['groups'][g]['frequency_count']} 点 | {means[g]['r4']:.6f} | {means[g]['r5']:.6f} | {means[g]['mae4']:.6f} | {means[g]['mse4']:.6f} |" for g in GROUPS)
    subjects='\n'.join(f'| S{s} | '+' | '.join(f"{next(r for r in rows if (r['group'],r['subject'])==(g,s))['r4']:.6f}" for g in GROUPS)+' |' for s in (1,2,3))
    comparisons='\n'.join(f"| {r['label']} | {r['r4']:.6f} | {r['source']} |" for r in summary['comparison'])
    outcome='提高' if delta>0 else '降低'
    text=f'''# 0.5–300 Hz Morlet：40 与 100 个频率中心

六个模型已完成服务器全量训练与完整测试预测。固定宽频带条件下，100 点相对同容量 40 点对照的四指平均 r {outcome} **{abs(delta):.6f}**。

## 完整测试结果

每个模型评分完整 200 秒、200000 个原始 1000 Hz 测试点。四指指标排除无名指；五指为辅助。先逐被试、逐手指计算 Pearson r，再等权平均。

| 频率中心数 | 四指 r | 五指 r | 四指 MAE | 四指 MSE |
|---|---:|---:|---:|---:|
{table}

| 被试 | 40 点，同容量对照 | 100 点 |
|---|---:|---:|
{subjects}

![成对测试结果](density_comparison.png)

## 实验控制

两组共用 `(B,16,100,T)` 输入和 6,980,230 参数：40 点组使用前 40 槽，其余 60 槽为零；100 点组全部有效。相同 seed 42 初始化、目标缩放、16 个训练选定电极、每轮窗口采样序列及训练预算。除输入卷积需要适配 100 槽外，所有主干张量的初始值保留原模型 seed 42 的值。两组输入层本身也完全同初始化。零槽相应梯度经过校验。

S1/S2/S3 分别从头训练 112/69/104 轮，每轮 8192 个窗口，batch 64。训练窗口 512 点、100 Hz；推理窗口 2048 点，步长 2048，输出整窗。AdamW lr 8.42e-5、weight_decay 1e-4、dropout 0.1，损失为 0.5 MSE + 0.5 时间轴 cosine distance。使用最终普通权重和同时间标签。

两组保持原始 1000 Hz ECoG 逐通道标准化、全部电极中位数参考、0.5–300 Hz MNE 带通和 60 Hz 谐波陷波。Morlet 使用 7 cycles、zero_mean=False、FFT 卷积后取功率、每十点取一点到 100 Hz。40 点组直接复用上一轮宽频缓存并校验哈希；100 点新特征与旧特征四个相同频率中心逐点核对一致。特征先在完整记录上提取，再切训练窗口。

频率网格分别为 `logspace(log10(0.5),log10(300),40/100)`。两组各自在训练内部 225:-25 的时间范围拟合 RobustScaler，quantile_range=(0.1,0.9) 表示百分位数 0.1 与 0.9，unit_variance=True；增益均为 1。标签采用相同训练 MinMax，原 25 Hz 标签经三次插值到 100 Hz。预测逆变换后线性插值回 1000 Hz 用于完整测试评分。

## 频率数量改变了什么

40 点宽频网格中，27 个中心低于 40 Hz、13 个在 40–300 Hz；100 点网格中对应为 68 与 32。相邻中心的比值从约 1.178 降为 1.067，恢复了部分高频采样密度，但仍比原 40–300 Hz 的 40 点网格稀疏。若希望全范围均匀对数网格至少有 40 个中心落在 40–300 Hz，需至少 125 个中心；本次没有额外训练 125 点模型。

![频率中心设计](frequency_design.png)

固定 7 cycles 时，0.5 Hz 的小波振幅包络半高宽约 5.25 秒，完整有限支持为 22.281 秒；100 个中心不会缩短此时间跨度。增加频率数同样不能恢复功率运算丢失的符号和相位。因此本次检验的是宽频带内加密采样的效果，不能单独确定上一轮下降的全部原因。

时间尺度的参数定义见 [MNE 1.8 Morlet 文档](https://mne.tools/1.8/generated/mne.time_frequency.morlet.html)。

## 与已有结果的关系

| 特征表示 | 四指平均 r | 模型来源 |
|---|---:|---|
{comparisons}

前三行来自此前已完成的 43 槽模型，6,805,126 参数；后两行来自这次相同 100 槽模型。主比较应使用本次 40 与 100 点的配对差值；原 40 点宽频分数可作历史参考，因为输入层尺寸和初始化不同。

![与原特征方案比较](all_feature_comparison.png)

## 论文结果段落

We examined whether increasing spectral sampling density could improve broadband Morlet decoding. Seven-cycle Morlet power spanning 0.5–300 Hz was sampled at either 40 or 100 logarithmically spaced frequency centers, with 13 and 32 centers, respectively, within 40–300 Hz. Both conditions used the same 100-slot RecurFlex model, seed-42 initialization, electrodes, training-window sequence, and training budget; unused slots in the 40-center condition were zero-filled. On complete test recordings, mean four-finger Pearson correlations were {means['morlet_40']['r4']:.4f} with 40 centers and {means['morlet_100']['r4']:.4f} with 100 centers, a paired mean difference of {delta:+.4f}. The comparison assesses increased density under the fixed broadband representation; wavelet cycle counts and the power representation were unchanged.

![训练曲线](training_curves.png)

六份权重与预测均先冻结，再读取本地测试标签。完整窗口覆盖、源文件与权重哈希、相同频率中心及独立 SciPy Pearson 复算全部通过。该实验固定单个种子；三个被试的结果为描述性比较，不作为统计显著性检验。
'''
    if 'full_train_record_r4' in summary:
        train_table='\n'.join(f"| {p['groups'][g]['frequency_count']} 点 | {summary['full_train_record_r4'][g]:.6f} | {means[g]['r4']:.6f} |" for g in GROUPS)
        text+=f'''\n## 冻结模型的训练记录诊断

相同 2048 点窗口推理完整 400 秒训练记录，无追加训练或权重修改。训练分数计算原生 100 Hz 的 40000 点，测试分数仍计算原始 1000 Hz 的全部 200000 点。

| 频率数 | 训练四指 r | 测试四指 r |
|---|---:|---:|
{train_table}

两组均能拟合训练记录，但 100 点没有改善测试表现。按本次固定 seed 42 与训练预算，单纯加密宽频 Morlet 中心未恢复原高频功率方案的测试效果。当前主实验继续采用 40–300 Hz Morlet 功率与三路有符号低频组合；低频时间跨度、功率表示、频段配额等因素的独立贡献仍不能由这次加密实验单独确定。
'''
    (result/'report.md').write_text(text,encoding='utf-8')
