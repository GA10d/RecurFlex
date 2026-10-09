"""Score six frozen predictions on complete local test records and plot comparisons."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / "result/.plotcache"))
sys.path.insert(0,str(HERE/"main_source"))
from metrics import score
from utils import load_json, save_json, sha256

LABELS = {"morlet_broadband":"Broadband Morlet", "signed_logbank":"Signed log filterbank"}
COLORS = {"morlet_broadband":"#497DB5", "signed_logbank":"#9271AD"}
GROUPS = list(LABELS)


def write_csv(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_artifacts(server, p):
    assert sha256(server/"protocol.json")==sha256(HERE/"protocol.json")
    frozen=load_json(server/"frozen_models_before_prediction.json")
    predictions=load_json(server/"frozen_predictions.json")
    assert predictions["model_manifest_sha256"]==sha256(server/"frozen_models_before_prediction.json")
    assert predictions["protocol_sha256"]==sha256(HERE/"protocol.json")
    for name,digest in load_json(server/"upload_manifest.json").items():
        assert sha256(server/name)==digest,name
    for name,digest in frozen["source_sha256"].items():
        assert sha256(server/name)==digest,name
    expected={(g,s) for g in GROUPS for s in (1,2,3)}
    assert len(predictions["predictions"])==len(frozen["fits"])==6
    assert {(v["group"],v["subject"]) for v in predictions["predictions"]}==expected
    assert {(v["group"],v["subject"]) for v in frozen["fits"]}==expected
    for fit in frozen["fits"]:
        s=fit["subject"]
        assert fit["initial_state_sha256"]==p["reference"][str(s)]["initial_state_sha256"]
        assert fit["sampling_sha256"]==p["reference"][str(s)]["sampling_sha256"]
        assert fit["epochs"]==p["training_epochs"][str(s)] and fit["seed"]==42
        assert fit["parameters"]==6805126
    for v in predictions["predictions"]:
        group,subject=v["group"],v["subject"]
        folder=server/"fits"/f"s{subject}_{group}"
        assert sha256(server/v["file"])==v["sha256"]
        assert sha256(folder/"model.pt")==v["model_sha256"]
        config=load_json(folder/"config.json")
        path=server/"inputs"/f"s{subject}"/group/"scalers.json"
        assert config["scalers_sha256"]==sha256(path)
        stats=load_json(path)
        for name in ("selected_electrodes","target_min","target_range"):
            assert stats[name]==p["reference"][str(subject)][name]
        assert stats["input_gain"]==p["groups"][group]["input_gain"]
        history=load_json(folder/"history.json")
        assert len(history)==p["training_epochs"][str(subject)] and history[-1]["epoch"]==len(history)
        coverage=load_json(server/"predictions"/f"s{subject}_{group}_coverage.json")
        assert coverage["record_length"]==20000 and coverage["stride"]==coverage["input_window"]==2048
        assert coverage["output_ranges"]==[[i,min(i+2048,20000)] for i in range(0,20000,2048)]
    return frozen,predictions


def figures(result, rows, means, comparison, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
    lookup={(r["group"],r["subject"]):r for r in rows}
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),gridspec_kw={"width_ratios":[1.2,.9,1.4]})
    for i,g in enumerate(GROUPS):
        values=[lookup[g,s]["r4"] for s in (1,2,3)]
        bars=axs[0].bar(np.arange(3)+(i-.5)*.32,values,.31,color=COLORS[g],label=LABELS[g])
        axs[0].bar_label(bars,labels=[f"{v:.3f}" for v in values],fontsize=8,padding=3)
    axs[0].set_xticks(range(3),["S1","S2","S3"])
    axs[0].set_ylabel("Four-finger Pearson r")
    axs[0].set_title("(a) Complete test recordings",loc="left",fontweight="bold")
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.28,.065),ncol=2,frameon=False)
    bars=axs[1].bar(range(2),[means[g]["r4"] for g in GROUPS],color=[COLORS[g] for g in GROUPS],width=.55)
    axs[1].bar_label(bars,labels=[f"{means[g]['r4']:.4f}" for g in GROUPS],padding=4)
    axs[1].set_xticks(range(2),["Morlet power","Signed bands"])
    axs[1].set_ylabel("Equal-subject mean r")
    axs[1].set_title("(b) Four-finger mean",loc="left",fontweight="bold")
    lower=min(0.,min(r["r4"] for r in rows)-.05)
    upper=min(1.,max(r["r4"] for r in rows)+.12)
    for ax in axs[:2]:
        ax.set_ylim(lower,upper)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fingers=("thumb","index","middle","ring","little")
    matrix=np.array([[np.mean([lookup[g,s][f"r_{f}"] for s in (1,2,3)]) for f in fingers] for g in GROUPS])
    im=axs[2].imshow(matrix,cmap="YlGnBu",vmin=min(0.,float(matrix.min())),vmax=1,aspect="auto")
    axs[2].set_yticks(range(2),["Morlet power","Signed bands"])
    axs[2].set_xticks(range(5),["Thumb","Index","Middle","Ring*","Little"],rotation=30,ha="right")
    for i in range(2):
        for j in range(5):
            axs[2].text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center",color="white" if matrix[i,j]>.65 else "black")
    axs[2].set_title("(c) Mean correlation by finger",loc="left",fontweight="bold")
    fig.colorbar(im,ax=axs[2],fraction=.045,pad=.03,label="Pearson r")
    fig.text(.5,.025,"0.5-300 Hz | 40 logarithmic features | Seed 42 | 512-point training / 2048-point inference | *Ring is auxiliary",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.14,1,1))
    for extension in ("png","svg","pdf"):
        fig.savefig(result/f"broadband_comparison.{extension}",dpi=400,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,4.1))
    colors=["#A3AEC0","#B5A8C5","#DD8746",COLORS[GROUPS[0]],COLORS[GROUPS[1]]]
    bars=ax.barh(range(len(comparison)),[r["r4"] for r in comparison],color=colors,height=.6)
    ax.bar_label(bars,labels=[f"{r['r4']:.4f}" for r in comparison],padding=5)
    ax.set_yticks(range(len(comparison)),[r["label"] for r in comparison])
    ax.invert_yaxis()
    ax.set_xlabel("Mean four-finger Pearson r")
    ax.set_xlim(0,min(1.,max(r["r4"] for r in comparison)+.1))
    ax.grid(axis="x",alpha=.15)
    ax.set_axisbelow(True)
    ax.set_title("Feature representations on the fixed RecurFlex backbone",fontweight="bold")
    fig.text(.5,.015,"First three rows: preceding completed ablation | Last two rows: new full-band training",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    for extension in ("png","svg","pdf"):
        fig.savefig(result/f"all_feature_comparison.{extension}",dpi=350,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,s in zip(axs,(1,2,3)):
        for g in GROUPS:
            history=load_json(result/"server_results/fits"/f"s{s}_{g}"/"history.json")
            ax.plot([v["epoch"] for v in history],[v["loss"] for v in history],color=COLORS[g],label=LABELS[g],lw=1.6)
        ax.set_title(f"S{s}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.15)
    axs[0].set_ylabel("Training loss (MSE + cosine)")
    axs[-1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(result/"training_curves.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    frequencies=np.asarray(p["feature_recipe"]["frequencies"])
    edges=np.asarray(p["feature_recipe"]["band_edges"])
    design=[dict(index=i+1,center_hz=float(f),lower_hz=float(edges[i]),upper_hz=float(edges[i+1]),
                 center_alias_hz_at_100hz=float(abs((f+50)%100-50))) for i,f in enumerate(frequencies)]
    write_csv(result/"frequency_grid.csv",design)
    fig,ax=plt.subplots(figsize=(10,3.4))
    ax.set_xscale("log")
    ax.axvspan(.5,40,color="#EEEEEE",zorder=0)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ax.broken_barh([(lo,hi-lo)],(1-.18,.36),facecolors=COLORS["signed_logbank"],edgecolors="white",linewidth=.6,alpha=.8)
    ax.scatter(frequencies,np.full(40,2),color=COLORS["morlet_broadband"],s=20,zorder=3)
    ax.scatter(frequencies,np.ones(40),color="#453458",s=10,zorder=3)
    old_grid=np.logspace(np.log10(40),np.log10(300),40)
    ax.scatter(old_grid,np.zeros(40),color="#A3AEC0",s=20,zorder=3)
    ax.axvline(40,color="#666666",ls=":",lw=1)
    ax.set_xlim(.43,345)
    ax.set_ylim(-.55,2.7)
    ax.set_yticks([0,1,2],["Previous Morlet (40 points)","Signed log bands (40 bands)","Broadband Morlet (40 points)"])
    ticks=[.5,1,2,4,8,16,30,40,70,150,300]
    ax.set_xticks(ticks,[str(v) for v in ticks])
    ax.set_xlabel("Frequency (Hz), logarithmic axis")
    ax.set_title("Spectral coverage with a fixed 40-feature budget",fontweight="bold")
    ax.text(np.sqrt(.5*40),2.4,"27 broadband centers below 40 Hz",ha="center",fontsize=9)
    ax.text(np.sqrt(40*300),2.4,"13 centers at or above 40 Hz",ha="center",fontsize=9)
    ax.grid(axis="x",alpha=.12)
    fig.tight_layout()
    for extension in ("png","svg","pdf"):
        fig.savefig(result/f"frequency_design.{extension}",dpi=350,bbox_inches="tight")
    plt.close(fig)


def run(result, labels_dir):
    result=Path(result)
    server=result/"server_results"
    p=load_json(HERE/"protocol.json")
    frozen,predictions=check_artifacts(server,p)
    previous_dir=HERE.parents[1]/"result"
    assert sha256(previous_dir/"summary.json")==p["previous_summary_sha256"]
    previous=load_json(previous_dir/"summary.json")
    for subject in (1,2,3):
        assert sha256(previous_dir/"server_results/inputs"/f"s{subject}"/"scalers.json")==p["reference"][str(subject)]["previous_scalers_sha256"]
    # Predictions and weights are frozen and verified before any new test-label access.
    truth={}
    label_sha={}
    for s in (1,2,3):
        path=Path(labels_dir)/f"sub{s}_testlabels.mat"
        truth[s]=loadmat(path,variable_names=["test_dg"])["test_dg"]
        assert truth[s].shape==(200000,5)
        label_sha[str(s)]=sha256(path)
        assert label_sha[str(s)]==previous["labels_sha256"][str(s)]
    fits={(v["group"],v["subject"]):v for v in frozen["fits"]}
    rows=[]
    for item in predictions["predictions"]:
        group,subject=item["group"],item["subject"]
        arrays=np.load(server/item["file"])
        native,full=arrays["prediction_100hz"],arrays["prediction_1000hz"]
        assert native.shape==(20000,5) and full.shape==(200000,5) and np.array_equal(full[::10],native)
        m=score(truth[subject],full)
        independent=[pearsonr(truth[subject][:,i],full[:,i]).statistic for i in range(5)]
        assert np.allclose(independent,m["r_fingers"],rtol=0,atol=1e-12)
        fit=fits[group,subject]
        rows.append(dict(group=group,subject=subject,seed=42,active_features=40,input_gain=p["groups"][group]["input_gain"],
                    **{f"r_{f}":m["r_fingers"][i] for i,f in enumerate(("thumb","index","middle","ring","little"))},
                    r4=m["r4"],r5=m["r5"],mae4=m["mae4"],mse4=m["mse4"],samples=m["samples"],
                    epochs=fit["epochs"],parameters=fit["parameters"],training_seconds=fit["elapsed"]))
    rows.sort(key=lambda r:(GROUPS.index(r["group"]),r["subject"]))
    means={g:{k:float(np.mean([r[k] for r in rows if r["group"]==g])) for k in ("r4","r5","mae4","mse4")} for g in GROUPS}
    comparisons=[]
    for g,label in (("morlet","Morlet 40-300 Hz (previous)"),("low_frequency","Three signed low bands (previous)"),("hybrid","Morlet + three low bands (previous)")):
        comparisons.append(dict(group=g,label=label,source="preceding completed ablation",**previous["means"][g]))
    comparisons.extend(dict(group=g,label=LABELS[g]+" 0.5-300 Hz",source="new full training",**means[g]) for g in GROUPS)
    contrasts=[]
    for s in (1,2,3):
        a=next(r for r in rows if r["group"]==GROUPS[0] and r["subject"]==s)
        b=next(r for r in rows if r["group"]==GROUPS[1] and r["subject"]==s)
        contrasts.append(dict(first=GROUPS[0],second=GROUPS[1],subject=s,delta_r4=a["r4"]-b["r4"],delta_r5=a["r5"]-b["r5"]))
    contrasts.append(dict(first=GROUPS[0],second=GROUPS[1],subject="mean",delta_r4=means[GROUPS[0]]["r4"]-means[GROUPS[1]]["r4"],delta_r5=means[GROUPS[0]]["r5"]-means[GROUPS[1]]["r5"]))
    write_csv(result/"subject_metrics.csv",rows)
    write_csv(result/"mean_metrics.csv",[dict(group=g,input_gain=p["groups"][g]["input_gain"],**means[g]) for g in GROUPS])
    write_csv(result/"contrasts.csv",contrasts)
    write_csv(result/"all_feature_comparison.csv",comparisons)
    best=max(GROUPS,key=lambda g:means[g]["r4"])
    save_json(result/"summary.json",dict(subjects=rows,means=means,seed=42,fresh_full_training_models=6,
              best_new_group=best,comparison= comparisons,protocol_sha256=sha256(HERE/"protocol.json"),labels_sha256=label_sha,
              previous_summary_sha256=p["previous_summary_sha256"],predictions_manifest_sha256=sha256(server/"frozen_predictions.json"),
              primary_metric="Complete 200000 original points per subject; four fingers excluding ring; equal-subject mean"))
    figures(result,rows,means,comparisons,p)
    table="\n".join(f"| {LABELS[g]} | {means[g]['r4']:.6f} | {means[g]['r5']:.6f} | {means[g]['mae4']:.6f} | {means[g]['mse4']:.6f} |" for g in GROUPS)
    subject_table="\n".join(f"| S{s} | "+" | ".join(f"{next(r for r in rows if r['group']==g and r['subject']==s)['r4']:.6f}" for g in GROUPS)+" |" for s in (1,2,3))
    comparison_table="\n".join(f"| {r['label']} | {r['r4']:.6f} | {r['r5']:.6f} |" for r in comparisons)
    delta=means[GROUPS[0]]["r4"]-means[GROUPS[1]]["r4"]
    broad_gain=means[GROUPS[0]]["r4"]-previous["means"]["morlet"]["r4"]
    signed_gain=means[GROUPS[1]]["r4"]-previous["means"]["low_frequency"]["r4"]
    hybrid_gaps={g:means[g]["r4"]-previous["means"]["hybrid"]["r4"] for g in GROUPS}
    support=load_json(server/"verification.json")["morlet_support_samples"]
    report=f"""# RecurFlex 全频段 Morlet 与有符号对数频带比较

两组各三个被试均已在服务器完成从头全量训练和完整测试预测。仅使用 seed 42，S1/S2/S3 分别训练 112/69/104 轮；模型保持当前 RecurFlex 主干。

## 完整测试结果

每个模型评分被试完整 200 秒、200000 个 1000 Hz 测试点。四指 r 排除无名指，五指为辅助，三个被试等权平均。

| 新特征表示 | 四指 r | 五指 r | 四指 MAE | 四指 MSE |
|---|---:|---:|---:|---:|
{table}

| 被试 | 全频段 Morlet | 有符号对数频带 |
|---|---:|---:|
{subject_table}

这两组中四指平均 r 最高的是 **{LABELS[best]}**。全频段 Morlet 相对有符号对数频带的平均差值为 **{delta:+.6f}**。

![两组完整测试比较](broadband_comparison.png)

## 与上一轮消融比较

| 特征表示 | 四指平均 r | 五指平均 r |
|---|---:|---:|
{comparison_table}

前三行来自上一轮已完成的全量训练，本次没有重训；后两行是本次六个新模型。训练电极集合、标签缩放、初始化张量、窗口采样序列和训练轮数已核对一致。

全频段 Morlet 相对原 40–300 Hz Morlet 的四指平均 r 差值为 **{broad_gain:+.6f}**；40 路有符号表示相对原三路低频表示的差值为 **{signed_gain:+.6f}**。两组相对原混合主模型的差值分别为 **{hybrid_gaps[GROUPS[0]]:+.6f}** 和 **{hybrid_gaps[GROUPS[1]]:+.6f}**。这些是固定 seed 42、固定训练配置下的比较。

![五种特征表示比较](all_feature_comparison.png)

## 特征定义与参数

两组先对原始 1000 Hz 记录逐通道标准化，用全部电极做中位数参考，再对既定 16 个训练电极做 0.5–300 Hz 带通与 60 Hz 谐波陷波。40 个中心为 `logspace(log10(0.5),log10(300),40)`，其中 27 个低于 40 Hz、13 个位于 40–300 Hz。精确频率和频带边界见 `../code/protocol.json`。

![频率点与频带设计](frequency_design.png)

Morlet 为 `n_cycles=7`、`zero_mean=False` 的功率特征，以 1000 Hz 完整记录计算后按每十点取一点保留 100 Hz 时间轴。0.5 Hz 小波时域支持为 {support[0]} 点，约 {support[0]/1000:.3f} 秒；特征先在完整记录上计算，随后切训练窗口。

有符号组按相邻中心的几何中点划出 40 个频带，两端为 0.5 和 300 Hz。每路使用四阶 Butterworth 与 `sosfiltfilt`，保留正负幅度，再直接 `[:,::10]`。每组各自用训练内部 225:-25 时间范围拟合 RobustScaler，`quantile_range=(0.1,0.9)`、`unit_variance=True`；有符号表示沿用原低频分支的缩放后 ×200 增益，Morlet 增益为 1。

高于 50 Hz 的有符号波形抽取到 100 Hz 会发生频率折叠；合成 70 Hz 波形的校验确认其抽取后峰值为 30 Hz。这项结果不能直接推导为“高频有符号信号没有信息”，它评估的是当前直接抽取方案。Morlet 功率与有符号波形还采用不同带宽定义；本比较针对上述完整特征提取配方。

两组输入均为 `(B,16,43,T)`，前 40 槽有效，最后 3 槽为零，不加入原三路低频。共用 6,805,126 参数、seed 42 初始化、训练窗口随机序列、标签 MinMax 与固定训练预算。训练 512 点、推理 2048 点全部输出，按同时间标签配对。没有 EMA、多种子集成、幅度校准或额外输出处理，预测只逆变换并线性插值到原始 1000 Hz 后评分。

## 论文英文结果段落

We compared two broadband representations spanning 0.5–300 Hz: Morlet power at 40 logarithmically spaced frequencies and signed amplitudes from a 40-band logarithmic Butterworth filterbank. Both used the same broadband filtering and harmonic notch filtering, training-selected electrodes, seed-42 model initialization, sampled windows, and training budgets. The filterbank retained waveform polarity and directly subsampled every tenth sample, following the signed low-frequency branch; its training-scaled inputs used the same gain of 200. The Morlet representation retained seven-cycle power features with a gain of one. On complete test recordings, mean four-finger Pearson correlations were {means[GROUPS[0]]['r4']:.4f} for broadband Morlet power and {means[GROUPS[1]]['r4']:.4f} for signed logarithmic bands. The corresponding correlations in the preceding matched-backbone experiments were {previous['means']['morlet']['r4']:.4f} for 40–300 Hz Morlet power, {previous['means']['low_frequency']['r4']:.4f} for three signed low-frequency bands, and {previous['means']['hybrid']['r4']:.4f} for their combination.

The full-band filterbank uses direct 100-Hz subsampling, which aliases signed components above 50 Hz. Its performance therefore characterizes this specific representation rather than the information available in the original high-frequency waveforms. Expanding the Morlet range also reallocates the fixed 40 frequency samples, leaving 13 centers above 40 Hz, and thus changes both spectral coverage and sampling density.

![训练曲线](training_curves.png)

六份权重及预测在读取本地测试标签前均已冻结并核对 SHA256，窗口覆盖完整，评分另用 SciPy Pearson 独立复算一致。原有实验和主模型目录未被覆盖。
"""
    (result/"report.md").write_text(report,encoding="utf-8")
    diagnostic_path=result/"train_diagnostics.json"
    if diagnostic_path.exists():
        train_rows=load_json(diagnostic_path)["training_diagnostics"]
        assert len(train_rows)==6
        assert {(r["group"],r["subject"]) for r in train_rows}=={(g,s) for g in GROUPS for s in (1,2,3)}
        assert all(r["split"]=="TRAIN" and r["samples"]==40000 and r["sampling_rate"]==100 and r["inference_window"]==2048 for r in train_rows)
        train_means={g:float(np.mean([r["r4"] for r in train_rows if r["group"]==g])) for g in GROUPS}
        gaps=[dict(group=r["group"],subject=r["subject"],train_r4=r["r4"],
                   test_r4=next(v["r4"] for v in rows if (v["group"],v["subject"])==(r["group"],r["subject"]))) for r in train_rows]
        write_csv(result/"train_test_gap.csv",gaps)
        training_table="\n".join(f"| {LABELS[g]} | {train_means[g]:.6f} | {means[g]['r4']:.6f} |" for g in GROUPS)
        report += f"""

## 完整训练记录诊断

冻结权重后的附加检查使用相同 2048 点推理，在各被试完整 400 秒训练记录上生成预测，没有追加训练或修改任何权重。训练 r 评分完整 40000 个 100 Hz 原生点，测试 r 仍是前文全部 200000 个 1000 Hz 官方测试点。

| 新特征表示 | 完整训练记录四指 r | 完整测试记录四指 r |
|---|---:|---:|
{training_table}

两组在训练记录上均达到约 0.996 的相关性，而测试分数明显下降，呈现较大的训练与测试泛化差距。按本次固定设置，主实验继续采用 40–300 Hz Morlet 功率与三路有符号低频的组合。范围扩展、有符号高频抽取和频带变窄的贡献仍需分别控制实验才能归因。

An additional evaluation of the frozen models on the complete training recordings yielded mean four-finger correlations of {train_means[GROUPS[0]]:.4f} and {train_means[GROUPS[1]]:.4f} for the broadband Morlet and signed filterbank representations, respectively. The high training correlations alongside the substantially lower test correlations indicate a pronounced generalization gap under the fixed training configuration.
"""
        (result/"report.md").write_text(report,encoding="utf-8")
        summary=load_json(result/"summary.json")
        summary["full_train_record_r4"]=train_means
        summary["train_diagnostics_sha256"]=sha256(diagnostic_path)
        save_json(result/"summary.json",summary)
    save_json(result/"verification.json",dict(passed=True,models=6,full_training_complete=True,
              full_test_records=6,samples_per_record=200000,matched_initialization_and_sample_windows_to_previous=True,
              weights_and_predictions_and_source_hashes_checked=True,independent_scipy_pearson_agrees=True,
              full_output_coverage_checked=True,best_new_group=best,means=means))
    print(json.dumps(dict(best_new_group=best,means=means,delta_morlet_minus_signed=delta),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--result-dir",type=Path,default=HERE.parent/"result")
    parser.add_argument("--labels-dir",type=Path,default=HERE.parents[4]/"data/BCICIV_4_mat")
    args=parser.parse_args()
    run(args.result_dir,args.labels_dir)
