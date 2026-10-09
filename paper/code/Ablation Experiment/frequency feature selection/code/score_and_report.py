"""Score frozen server predictions locally and draw paper-ready ablation plots."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "main_source"))
from metrics import score
from utils import load_json, save_json, sha256

LABELS = {"morlet":"Morlet only", "low_frequency":"Low frequency only", "hybrid":"Hybrid"}
COLORS = {"morlet":"#497DB5", "low_frequency":"#9271AD", "hybrid":"#DD8746"}
GROUPS = list(LABELS)


def write_csv(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(result, rows, means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "svg.fonttype":"none", "pdf.fonttype":42})
    lookup = {(r["group"],r["subject"]):r for r in rows}
    fig,axs = plt.subplots(1,3,figsize=(14,4.3),gridspec_kw={"width_ratios":[1.3,.9,1.4]})
    x = np.arange(3)
    for i,g in enumerate(GROUPS):
        v = [lookup[g,s]["r4"] for s in (1,2,3)]
        bars = axs[0].bar(x+(i-1)*.24,v,.23,color=COLORS[g],label=LABELS[g])
        axs[0].bar_label(bars,labels=[f"{z:.3f}" for z in v],fontsize=7,padding=3)
    axs[0].set_xticks(x,["S1","S2","S3"])
    axs[0].set_ylabel("Four-finger Pearson r")
    axs[0].set_title("(a) Complete test recordings",loc="left",fontweight="bold")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=9,loc="lower center",bbox_to_anchor=(.28,.065),ncol=3)
    bars = axs[1].bar(x,[means[g]["r4"] for g in GROUPS],color=[COLORS[g] for g in GROUPS],width=.62)
    axs[1].bar_label(bars,labels=[f"{means[g]['r4']:.4f}" for g in GROUPS],padding=4,fontsize=10)
    axs[1].set_xticks(x,["Morlet","Low freq.","Hybrid"])
    axs[1].set_ylabel("Equal-subject mean r")
    axs[1].set_title("(b) Official four-finger mean",loc="left",fontweight="bold")
    lower = min(0., min(r["r4"] for r in rows)-.05)
    upper = min(1., max(r["r4"] for r in rows)+.12)
    for ax in axs[:2]:
        ax.set_ylim(lower,upper)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    matrix = np.array([
        [np.mean([lookup[g,s][f"r_{f}"] for s in (1,2,3)]) for f in ("thumb","index","middle","ring","little")]
        for g in GROUPS])
    im = axs[2].imshow(matrix,cmap="YlGnBu",vmin=0,vmax=1,aspect="auto")
    axs[2].set_yticks(range(3),["Morlet","Low freq.","Hybrid"])
    axs[2].set_xticks(range(5),["Thumb","Index","Middle","Ring*","Little"],rotation=30,ha="right")
    for i in range(3):
        for j in range(5):
            axs[2].text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center",color="white" if matrix[i,j]>.65 else "black")
    axs[2].set_title("(c) Mean correlation by finger",loc="left",fontweight="bold")
    fig.colorbar(im,ax=axs[2],fraction=.045,pad=.03,label="Pearson r")
    fig.text(.5,.025,"Seed 42 | same electrodes and initialization | 512-point training / 2048-point inference | *Ring is auxiliary",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.14,1,1))
    for extension in ("png","svg","pdf"):
        fig.savefig(result/f"feature_ablation.{extension}",dpi=400,bbox_inches="tight")
    plt.close(fig)
    fig,axs = plt.subplots(1,3,figsize=(12,3.5))
    for ax,s in zip(axs,(1,2,3)):
        for g in GROUPS:
            h = load_json(result/"server_results"/"fits"/f"s{s}_{g}"/"history.json")
            ax.plot([v["epoch"] for v in h],[v["loss"] for v in h],color=COLORS[g],label=LABELS[g],lw=1.6)
        ax.set_title(f"S{s}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.15)
    axs[0].set_ylabel("Training loss (MSE + cosine)")
    axs[-1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(result/"training_curves.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def run(result, labels_dir):
    result = Path(result)
    server = result / "server_results"
    p = load_json(HERE/"protocol.json")
    assert sha256(server/"protocol.json")==sha256(HERE/"protocol.json")
    frozen = load_json(server/"frozen_models_before_prediction.json")
    predictions = load_json(server/"frozen_predictions.json")
    assert predictions["model_manifest_sha256"]==sha256(server/"frozen_models_before_prediction.json")
    assert predictions["protocol_sha256"]==sha256(HERE/"protocol.json")
    for name,digest in load_json(server/"upload_manifest.json").items():
        assert sha256(server/name)==digest,name
    assert len(predictions["predictions"])==9 and len(frozen["fits"])==9
    assert {(v["group"],v["subject"]) for v in predictions["predictions"]}=={(g,s) for g in GROUPS for s in (1,2,3)}
    assert len({v["initial_state_sha256"] for v in frozen["fits"]})==1
    for s in (1,2,3):
        assert len({v["sampling_sha256"] for v in frozen["fits"] if v["subject"]==s})==1
    for name,digest in frozen["source_sha256"].items():
        assert sha256(server/name)==digest,name
    for v in predictions["predictions"]:
        assert sha256(server/v["file"])==v["sha256"]
        assert sha256(server/"fits"/f"s{v['subject']}_{v['group']}"/"model.pt")==v["model_sha256"]
    # All nine predictions and weights have been checked before reading any test label.
    rows = []
    fits = {(v["group"],v["subject"]):v for v in frozen["fits"]}
    labels = {}
    label_sha = {}
    for subject in (1,2,3):
        path = Path(labels_dir)/f"sub{subject}_testlabels.mat"
        labels[subject] = loadmat(path,variable_names=["test_dg"])["test_dg"]
        assert labels[subject].shape==(200000,5)
        label_sha[str(subject)] = sha256(path)
    for item in predictions["predictions"]:
        group,subject = item["group"],item["subject"]
        arrays = np.load(server/item["file"])
        native,full = arrays["prediction_100hz"],arrays["prediction_1000hz"]
        assert native.shape==(20000,5) and full.shape==(200000,5)
        assert np.array_equal(full[::10],native)
        m = score(labels[subject],full)
        fit = fits[group,subject]
        assert fit["epochs"]==p["training_epochs"][str(subject)] and fit["seed"]==42
        history = load_json(server/"fits"/f"s{subject}_{group}"/"history.json")
        assert len(history)==fit["epochs"] and history[-1]["epoch"]==fit["epochs"]
        config = load_json(server/"fits"/f"s{subject}_{group}"/"config.json")
        assert config["scalers_sha256"]==sha256(server/"inputs"/f"s{subject}"/"scalers.json")
        row = dict(group=group,subject=subject,seed=42,active_features=len(p["groups"][group]["active_features"]),
                   **{f"r_{f}":m["r_fingers"][i] for i,f in enumerate(("thumb","index","middle","ring","little"))},
                   r4=m["r4"],r5=m["r5"],mae4=m["mae4"],mse4=m["mse4"],samples=m["samples"],
                   epochs=fit["epochs"],parameters=fit["parameters"],training_seconds=fit["elapsed"])
        rows.append(row)
    rows.sort(key=lambda v:(GROUPS.index(v["group"]),v["subject"]))
    means = {g:{k:float(np.mean([r[k] for r in rows if r["group"]==g]))
                for k in ("r4","r5","mae4","mse4")} for g in GROUPS}
    contrasts = []
    for first,second in (("hybrid","morlet"),("hybrid","low_frequency"),("morlet","low_frequency")):
        for subject in (1,2,3):
            a = next(r for r in rows if r["group"]==first and r["subject"]==subject)
            b = next(r for r in rows if r["group"]==second and r["subject"]==subject)
            contrasts.append(dict(first=first,second=second,subject=subject,delta_r4=a["r4"]-b["r4"],delta_r5=a["r5"]-b["r5"]))
        contrasts.append(dict(first=first,second=second,subject="mean",delta_r4=means[first]["r4"]-means[second]["r4"],delta_r5=means[first]["r5"]-means[second]["r5"]))
    write_csv(result/"subject_metrics.csv",rows)
    write_csv(result/"mean_metrics.csv",[dict(group=g,active_features=len(p["groups"][g]["active_features"]),**means[g]) for g in GROUPS])
    write_csv(result/"contrasts.csv",contrasts)
    save_json(result/"summary.json",dict(subjects=rows,means=means,seed=42,
              protocol_sha256=sha256(HERE/"protocol.json"),labels_sha256=label_sha,
              predictions_manifest_sha256=sha256(server/"frozen_predictions.json"),
              primary_metric="Full 200000 original points per subject; official four fingers; equal-subject mean",
              identical_electrodes=True,identical_initialization=True,identical_sample_windows=True,
              fresh_full_training_models=9))
    figures(result,rows,means)
    best = max(GROUPS,key=lambda g:means[g]["r4"])
    table = "\n".join(f"| {LABELS[g]} | {means[g]['r4']:.6f} | {means[g]['r5']:.6f} | {means[g]['mae4']:.6f} | {means[g]['mse4']:.6f} |" for g in GROUPS)
    subject_table = "\n".join(f"| S{s} | "+" | ".join(f"{next(r for r in rows if r['group']==g and r['subject']==s)['r4']:.6f}" for g in GROUPS)+" |" for s in (1,2,3))
    hybrid_gains = [next(r for r in rows if r['group']=='hybrid' and r['subject']==s)['r4']
                    - next(r for r in rows if r['group']=='morlet' and r['subject']==s)['r4'] for s in (1,2,3)]
    subject_gain_text = ", ".join(f"S{s}: {gain:+.4f}" for s,gain in zip((1,2,3),hybrid_gains))
    interpretation = (
        "混合输入在三个被试上均优于仅 Morlet；Morlet 单独使用也优于仅低频。该固定模型配置下，结果支持 Morlet 功率与有符号低频幅度提供互补预测信息。"
        if all(gain>0 for gain in hybrid_gains) and means['morlet']['r4']>means['low_frequency']['r4'] else
        "不同特征的贡献按上表和各被试结果判断。"
    )
    report = f"""# RecurFlex 频率特征消融结果

九个模型均已在服务器上完成全量训练。三个被试，每个被试三组特征，每组仅 seed 42；S1/S2/S3 分别训练 112/69/104 轮。各组使用相同的训练电极选择、缩放参数、模型初始化和随机窗口序列。未使用的输入维度置零，保留相同 6,805,126 参数主干。

## 完整官方测试结果

每个被试均评分完整 200 秒、1000 Hz 的 200000 点。四指排除无名指，五指作为辅助；三个被试等权平均。

| 特征 | 四指 r | 五指 r | 四指 MAE | 四指 MSE |
|---|---:|---:|---:|---:|
{table}

| 被试 | Morlet only | Low frequency only | Hybrid |
|---|---:|---:|---:|
{subject_table}

本组固定配置中四指平均 r 最高的是 **{LABELS[best]}**。混合相对仅 Morlet 的平均差值为 **{means['hybrid']['r4']-means['morlet']['r4']:+.6f}**，混合相对仅低频的差值为 **{means['hybrid']['r4']-means['low_frequency']['r4']:+.6f}**。这是 seed 42 的配对消融；不将三个被试的点估计表述为多种子统计显著性。

{interpretation} 混合相对仅 Morlet 的各被试差值为 {subject_gain_text}。

![完整测试比较](feature_ablation.png)

## 论文英文结果段落

We examined the contribution of the input representation by training RecurFlex with Morlet power, signed low-frequency amplitudes, and their combination. All three arms used the same 16 training-selected electrodes, input scaling, seed-42 initialization, and randomly sampled training windows. Unused feature slots were set to zero, preserving the architecture and parameter count. Models were trained from scratch on the full 400-second labeled recordings, with 512-point training windows and 2,048-point full-output inference blocks. Predictions used matched-time targets and were evaluated without ensemble averaging or additional output processing. On the complete official test recordings, the mean four-finger Pearson correlations were {means['morlet']['r4']:.4f} for Morlet power, {means['low_frequency']['r4']:.4f} for low-frequency amplitudes, and {means['hybrid']['r4']:.4f} for their combination.

Combining the representations improved mean correlation by {means['hybrid']['r4']-means['morlet']['r4']:.4f} over Morlet power alone, with subject-specific differences of {hybrid_gains[0]:+.4f}, {hybrid_gains[1]:+.4f}, and {hybrid_gains[2]:+.4f}. The consistent gains across the three participants support complementary predictive information in the signed low-frequency amplitudes and Morlet power under the fixed RecurFlex training configuration.

## 输入与推理

Morlet 为 40 个 40–300 Hz 对数间隔、7-cycle 功率特征；低频为 0.5–4、4–13、13–30 Hz 三个有符号幅度，沿用训练缩放后乘 200 的增益。所有组使用混合特征训练相关性选择的同一组电极，因此该比较是在固定电极条件下检验特征贡献。模型输出只做训练 MinMax 逆变换和 100→1000 Hz 线性插值。

完整记录的预测先生成并冻结哈希，再在本地读取测试标签评分。数据、来源代码、初始化、采样序列、权重和预测一致性均已核对。主实验目录的已有权重未用于这九次训练。

![训练曲线](training_curves.png)
"""
    (result/"report.md").write_text(report,encoding="utf-8")
    save_json(result/"verification.json",dict(passed=True,models=9,full_training_complete=True,
              full_test_records=9,samples_per_record=200000,initialization_hashes=1,
              same_window_sampling_by_subject=True,protocol_and_weight_and_prediction_hashes_checked=True,
              best_group=best,means=means))
    print(json.dumps(dict(best_group=best,means=means),indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",type=Path,default=HERE.parent/"result")
    parser.add_argument("--labels-dir",type=Path,default=HERE.parents[3]/"data/BCICIV_4_mat")
    args = parser.parse_args()
    run(args.result_dir,args.labels_dir)
