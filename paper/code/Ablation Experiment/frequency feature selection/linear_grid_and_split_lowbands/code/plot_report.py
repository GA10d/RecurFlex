"""Publication figures for linear Morlet sampling and low-band separation."""
import numpy as np
from score_and_report import GROUPS,LABELS,write_csv,load_json

COLORS={'linear_morlet':'#6595A9','hybrid_3_control':'#8B91A0','hybrid_4':'#D78145'}
SHORT={'linear_morlet':'Linear Morlet','hybrid_3_control':'3 signed bands','hybrid_4':'4 signed bands'}

def create_figures(result,summary,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(result/f'{name}.{ext}',dpi=350,bbox_inches='tight')
        plt.close(fig)
    rows,means=summary['subjects'],summary['means'];lookup={(r['group'],r['subject']):r for r in rows}
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),gridspec_kw={'width_ratios':[1.25,.95,1.4]})
    for i,g in enumerate(GROUPS):
        values=[lookup[g,s]['r4'] for s in (1,2,3)]
        bars=axs[0].bar(np.arange(3)+(i-1)*.25,values,.235,color=COLORS[g],label=SHORT[g])
        axs[0].bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=7)
    axs[0].set_xticks(range(3),['S1','S2','S3']);axs[0].set_ylabel('Four-finger Pearson r')
    axs[0].set_title('(a) Complete test recordings',loc='left',fontweight='bold')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.35,.055),ncol=3,frameon=False,fontsize=9)
    bars=axs[1].bar(range(3),[means[g]['r4'] for g in GROUPS],width=.6,color=[COLORS[g] for g in GROUPS])
    axs[1].bar_label(bars,labels=[f"{means[g]['r4']:.4f}" for g in GROUPS],padding=4,fontsize=9)
    axs[1].set_xticks(range(3),['Linear','Hybrid-3','Hybrid-4']);axs[1].set_ylabel('Equal-subject mean r')
    axs[1].set_title('(b) Mean test correlation',loc='left',fontweight='bold')
    for ax in axs[:2]:
        ax.set_ylim(min(0.,min(r['r4'] for r in rows)-.05),min(1.,max(r['r4'] for r in rows)+.15));ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fingers=['thumb','index','middle','ring','little']
    matrix=np.array([[np.mean([lookup[g,s][f'r_{f}'] for s in (1,2,3)]) for f in fingers] for g in GROUPS])
    im=axs[2].imshow(matrix,cmap='YlGnBu',vmin=min(0.,matrix.min()),vmax=1,aspect='auto')
    axs[2].set_xticks(range(5),['Thumb','Index','Middle','Ring*','Little'],rotation=30,ha='right');axs[2].set_yticks(range(3),['Linear','Hybrid-3','Hybrid-4'])
    for i in range(3):
        for j in range(5):axs[2].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black')
    axs[2].set_title('(c) Mean by finger',loc='left',fontweight='bold');fig.colorbar(im,ax=axs[2],fraction=.04,pad=.03)
    fig.text(.5,.015,'Seed 42 | 512-point training / 2048-point inference | Same 44-slot model for Hybrid-3/4 | *Ring auxiliary',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.14,1,1));save(fig,'feature_comparison')
    comparison=summary['comparison'];fig,ax=plt.subplots(figsize=(10,5.1))
    bars=ax.barh(range(len(comparison)),[r['r4'] for r in comparison],color=['#A8B5C8','#B8AA92','#C6CBD4','#C6CBD4',*[COLORS[g] for g in GROUPS]],height=.62)
    ax.bar_label(bars,labels=[f"{r['r4']:.4f}" for r in comparison],padding=5)
    ax.set_yticks(range(len(comparison)),[r['label'] for r in comparison]);ax.invert_yaxis();ax.set_xlim(0,min(1.,max(r['r4'] for r in comparison)+.1))
    ax.set_xlabel('Mean four-finger Pearson r');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Frequency sampling and signed low-band representations',fontweight='bold')
    fig.text(.5,.02,'First 4: preceding experiments | Last 3: new full training | Log100 uses the previous 100-slot model',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1));save(fig,'all_feature_comparison')
    fig,axs=plt.subplots(1,3,figsize=(12,3.4))
    for ax,s in zip(axs,(1,2,3)):
        for g in GROUPS:
            history=load_json(result/'server_results/fits'/f's{s}_{g}'/'history.json')
            ax.plot([v['epoch'] for v in history],[v['loss'] for v in history],color=COLORS[g],label=SHORT[g],lw=1.5)
        ax.set_title(f'S{s}');ax.set_xlabel('Epoch');ax.grid(alpha=.15)
    axs[0].set_ylabel('Training loss (MSE + cosine)');axs[-1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'training_curves')
    fig,axs=plt.subplots(1,2,figsize=(12.5,3.6),gridspec_kw={'width_ratios':[1.4,1]})
    grids=[(np.asarray(p['feature_recipe']['high_frequencies']),'40-300 Hz, log40','#A8B5C8'),(np.geomspace(.5,300,40),'0.5-300 Hz, log40','#B2B8C3'),(np.asarray(p['feature_recipe']['linear_frequencies']),'0.5-300 Hz, linear40',COLORS['linear_morlet'])]
    write_csv(result/'frequency_grid.csv',[dict(grid=label,index=i+1,center_hz=float(f)) for fs,label,_ in grids for i,f in enumerate(fs)])
    ax=axs[0];ax.axvspan(.5,40,color='#F0F0F0');ax.axvline(40,color='#666666',lw=1,ls=':')
    for i,(fs,label,color) in enumerate(grids):ax.scatter(fs,np.full(len(fs),i),s=15,color=color);ax.text(170,i+.18,f'{np.sum(fs>=40-1e-10)} centers at or above 40 Hz',ha='center',fontsize=9,color='#555555')
    ax.set_yticks(range(3),[v[1] for v in grids]);ax.set_xlim(-4,305);ax.set_ylim(-.45,2.65);ax.set_xticks([.5,40,100,200,300],['0.5','40','100','200','300']);ax.set_xlabel('Frequency (Hz), linear axis')
    ax.set_title('(a) Forty frequency centers',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.12)
    ax=axs[1];colors=['#90AAA6','#B1A985','#B78E8E','#9C9BB8']
    for y,bands in enumerate((p['feature_recipe']['signed_bands_3'],p['feature_recipe']['signed_bands_4'])):
        for j,(lo,hi) in enumerate(bands):
            color=colors[j if y==1 else (0 if j==0 else 1 if j==1 else 3)]
            ax.broken_barh([(lo,hi-lo)],(y-.2,.4),facecolors=color,edgecolors='white',lw=1.5)
            ax.text((lo+hi)/2,y,f'{lo:g}-{hi:g}',ha='center',va='center',fontsize=9)
    ax.set_yticks([0,1],['3 bands','4 bands']);ax.set_ylim(-.5,1.5);ax.set_xlim(.2,30.3);ax.set_xticks([.5,4,8,13,30],['0.5','4','8','13','30']);ax.set_xlabel('Frequency (Hz)')
    ax.set_title('(b) Signed low-frequency bands',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.12)
    fig.tight_layout();save(fig,'feature_design')

def write_report(result,summary,p):
    means=summary['means'];rows=summary['subjects'];lookup={(r['group'],r['subject']):r for r in rows}
    split=means['hybrid_4']['r4']-means['hybrid_3_control']['r4']
    broad40=next(r['r4'] for r in summary['comparison'] if r['group']=='morlet_broadband')
    high40=next(r['r4'] for r in summary['comparison'] if r['group']=='morlet')
    original=next(r['r4'] for r in summary['comparison'] if r['group']=='hybrid')
    linear=means['linear_morlet']['r4']
    split_subjects=[lookup['hybrid_4',s]['r4']-lookup['hybrid_3_control',s]['r4'] for s in (1,2,3)]
    table='\n'.join(f"| {LABELS[g]} | {means[g]['r4']:.6f} | {means[g]['r5']:.6f} | {means[g]['mae4']:.6f} | {means[g]['mse4']:.6f} |" for g in GROUPS)
    subject_table='\n'.join(f'| S{s} | '+' | '.join(f"{lookup[g,s]['r4']:.6f}" for g in GROUPS)+' |' for s in (1,2,3))
    comparison_table='\n'.join(f"| {r['label']} | {r['r4']:.6f} | {r['source']} |" for r in summary['comparison'])
    text=f'''# 等间隔宽频 Morlet 与四路有符号低频消融

三组九个模型均已完成服务器全量训练和完整测试。第一项比较仅使用 0.5–300 Hz 的 40 个等间隔 Morlet 功率；第二项保留原 40–300 Hz 对数 Morlet，将原 4–13 Hz 有符号低频拆成 4–8 与 8–13 Hz。另从头训练相同 44 槽输入的三路低频对照。

## 完整测试结果

每个模型评分完整 200 秒、200000 个原始 1000 Hz 时间点。四指排除无名指，逐被试、逐手指计算相关性后等权平均；五指为辅助。

| 特征表示 | 四指 r | 五指 r | 四指 MAE | 四指 MSE |
|---|---:|---:|---:|---:|
{table}

| 被试 | 等间隔 Morlet | 三路低频对照 | 四路低频 |
|---|---:|---:|---:|
{subject_table}

四路低频相对相同 44 槽三路对照的四指平均 r 差值为 **{split:+.6f}**；相对此前原混合模型 {original:.6f} 的差值为 **{means['hybrid_4']['r4']-original:+.6f}**。等间隔宽频 Morlet 相对上一轮相同 43 槽的宽频对数40点 {broad40:.6f} 的差值为 **{linear-broad40:+.6f}**，相对原高频对数40点 {high40:.6f} 的差值为 **{linear-high40:+.6f}**。

四路低频相对配对三路对照在 S1/S2/S3 的差值分别为 {split_subjects[0]:+.6f}、{split_subjects[1]:+.6f}、{split_subjects[2]:+.6f}。平均收益主要来自 S1，S3 有下降。按本次固定 seed42 设置，四路低频取得新组中最高的平均 r，平均 MAE 与 MSE 也低于三路对照。

同为40个宽频中心、相同43槽主干，仅改变对数与等间隔网格便产生明显的测试差值。结合此前100点加密未改善的结果，本组证据支持优先研究频率点在高低频之间的分配。这里的比较针对已定义的完整特征表示。

![完整测试结果](feature_comparison.png)

## 两项改变的具体实现

等间隔网格为 `linspace(0.5,300,40)`，中心间距约 7.67949 Hz，6 个中心低于 40 Hz、34 个在 40–300 Hz。原 0.5–300 Hz 对数40点有 27 个低频、13 个高频中心。线性与对数宽频比较保持相同预滤波配方、16 个电极、43 槽模型、完整 seed42 初始化和窗口采样序列。线性输入的最后三槽置零，不加有符号低频。特征为标准化、中位数参考、0.5–300 Hz 带通、60 Hz 谐波陷波后，以 7 cycles、zero_mean=False 计算 Morlet 功率，再每十点取一点。

四路低频表示仍使用原 40–300 Hz 的40个对数 Morlet 功率缓存，直接复用并核对 SHA256。新增的四路频带为 `[0.5,4]`、`[4,8]`、`[8,13]`、`[13,30]` Hz，在高频带通之前的标准化、中位数参考信号上，用四阶 Butterworth 与 sosfiltfilt，保留正负幅度后每十点取一点。Delta 和 Beta 与原缓存逐点核对，只有 Theta、Alpha 替代原合并 4–13 Hz 频带。

![频率与低频设计](feature_design.png)

各组在训练内部 225:-25 时间范围拟合 RobustScaler，quantile_range=(0.1,0.9) 为百分位数 0.1 与 0.9，unit_variance=True。Morlet 输入增益1，有符号低频缩放后增益200。三路对照的缩放参数与原模型固定电极上的参数一致，四路组的既有高频、Delta、Beta 缩放也核对一致。

两种混合模型均为 44 槽、6,808,198 参数。三路对照按规范顺序放置 Delta、原4–13、零槽、Beta；四路组放置 Delta、Theta、Alpha、Beta。输入层原有权重按上述顺序嵌入；新增 Alpha 槽保留 seed42 的随机初值，两组完整初始化一致。所有非输入层张量保留原模型初值。三路对照初始化输出与原43槽模型在 CPU 核对一致，浮点误差见服务器验证记录。

## 固定训练与测试流程

各组复用相同的 16 个原训练选定电极、训练目标 MinMax、seed42 和采样序列。S1/S2/S3 分别训练 112/69/104 轮，每轮8192窗口，batch64。100 Hz 时间轴上训练512点、推理2048点，步长2048，输出整窗，同时间标签配对。AdamW lr8.42e-5、weight_decay1e-4、dropout0.1，损失0.5MSE+0.5时间轴cosine distance。采用最终普通权重。特征先在完整记录计算，再切训练窗口；原25Hz标签经三次插值到100Hz，预测逆变换后线性插值回1000Hz。

九份模型与预测在读取本地测试标签前冻结。源码、权重、预测哈希、相同采样序列、标签哈希和完整输出覆盖均核对通过，另以 SciPy Pearson 独立复算所有逐指 r 一致。实验固定单种子，三个被试的配对结果为描述性比较。

## 与已有结果比较

| 特征表示 | 四指平均 r | 来源 |
|---|---:|---|
{comparison_table}

前四行来自已完成实验；其中对数100点使用先前100槽模型，只作历史参考。等间隔与宽频对数40点则均为原43槽模型，可直接比较网格分配。两种混合输入的主要比较使用本次相同44槽模型的配对差值。

![所有方案比较](all_feature_comparison.png)

## 论文英文结果段落

We evaluated two refinements to the frequency representation. First, broadband Morlet power was sampled at 40 linearly spaced centers spanning 0.5–300 Hz, allocating 34 centers within 40–300 Hz. With the original 43-slot backbone and seed-42 initialization, this representation achieved a mean four-finger test correlation of {linear:.4f}, compared with {broad40:.4f} for the preceding logarithmic broadband representation and {high40:.4f} for logarithmic 40–300 Hz Morlet power. Second, we retained the high-frequency Morlet features and separated the signed 4–13 Hz band into 4–8 Hz and 8–13 Hz bands. Matched 44-slot models using three or four signed low-frequency bands achieved mean correlations of {means['hybrid_3_control']['r4']:.4f} and {means['hybrid_4']['r4']:.4f}, respectively, yielding a paired mean difference of {split:+.4f}. All models used the same training-selected electrodes, target normalization, window sequence, and training budgets, and were evaluated on the complete test recordings.

![训练曲线](training_curves.png)
'''
    if 'full_train_record_r4' in summary:
        table='\n'.join(f"| {LABELS[g]} | {summary['full_train_record_r4'][g]:.6f} | {means[g]['r4']:.6f} |" for g in GROUPS)
        text+=f'''\n## 冻结模型的完整训练记录诊断

使用同样2048点推理完整400秒训练记录，不追加训练或修改权重。训练r计算原生100Hz的40000点，测试r仍为原始1000Hz全部200000点。

| 特征表示 | 训练四指 r | 测试四指 r |
|---|---:|---:|
{table}
'''
    (result/'report.md').write_text(text,encoding='utf-8')
