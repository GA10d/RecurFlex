# RecurFlex 频率特征消融

比较三个被试的三种输入，每组只使用 seed 42，九个模型均从头训练。保持当前主实验的同时间标签配对、512 点训练、2048 点全部输出推理、普通最终权重和固定训练预算。运行状态与完成后的完整测试结果保存在 `result`。

## 已完成：服务器全量训练与完整测试

2026-10-08，九次训练和九份完整测试预测均已完成，下载文件、权重与预测哈希核对通过。每个被试评分全部 200000 点，下面为三个被试等权平均；四指排除无名指。

| 输入 | 四指 r | 五指 r |
|---|---:|---:|
| 仅 Morlet | 0.716050 | 0.727287 |
| 仅低频 | 0.557363 | 0.576347 |
| 混合 | **0.758373** | **0.771415** |

混合相对仅 Morlet 的四指平均 r 提升 0.042323，S1/S2/S3 均有提升。混合组是本次重新训练的模型，所有组使用同一套训练电极选择和固定训练预算。结果支持保留混合输入。

查看 [详细结果与论文英文段落](result/report.md)、[论文对比图](result/feature_ablation.png) 或 [结果 notebook](code/results.ipynb)。矢量图保存在 `result/feature_ablation.svg` 和 `result/feature_ablation.pdf`，九份权重、预测和训练记录位于 `result/server_results`。

## 扩展到 0.5 至 300 Hz 的两组对照

后续完成六次新的全量训练：0.5–300 Hz 的 40 点对数 Morlet 功率，四指平均 r 为 **0.398289**；相同中心网格的 40 路有符号对数频带波形，四指平均 r 为 **0.355140**。两组均每 10 点抽取 1 点，沿用 seed42、512/2048 点窗口和同一主干、电极、初始化及训练预算。

两个新配方均未超过原混合表示的 0.758373。代码、六份新权重和完整结果独立保存在 [扩展实验目录](broadband_0p5_300/README.md)，查看 [五种表示比较图](broadband_0p5_300/result/all_feature_comparison.png) 与 [详细报告](broadband_0p5_300/result/report.md)。

## 宽频 Morlet 从 40 点增加到 100 点

又完成六次从头全量训练与完整测试。保留 0.5–300 Hz，40 与 100 点两组均采用相同 100 槽模型、初始化和训练窗口序列。配对 40 点对照四指平均 r 为 **0.371024**，100 点为 **0.336524**，差值 **−0.034500**，三个被试均下降。两组完整训练记录 r 分别为 0.996238 与 0.995619。

100 点虽将 40–300 Hz 内的中心数从 13 增至 32，但本次未改善测试泛化。原宽频 40 点的 0.398289 来自之前的 43 槽模型，只作历史参考。实验代码、权重、预测、图表及论文英文段落在 [100 点消融目录](broadband_morlet_100/README.md)，查看 [成对测试图](broadband_morlet_100/result/density_comparison.png) 和 [结果 notebook](broadband_morlet_100/code/results.ipynb)。

## 等间隔宽频 Morlet 与四路有符号低频

后续又完成九次全量训练及完整测试：仅0.5–300Hz的40个等间隔Morlet中心，四指平均r为 **0.683921**；原高频Morlet＋四路有符号低频为 **0.774094**，同44槽模型的三路低频对照为 **0.762424**。四路频带是0.5–4、4–8、8–13、13–30Hz。

四路方案相对配对三路对照平均提高0.011670，相对此前原主模型0.758373提高0.015720。S1/S2提高，S3下降，平均收益主要来自S1。等间隔宽频网格将高频中心数从13增至34，相对宽频对数40点提高0.285632，但仍低于原仅高频Morlet的0.716050。

查看 [本次实验目录](linear_grid_and_split_lowbands/README.md)、[正式报告与论文段落](linear_grid_and_split_lowbands/result/report.md) 和 [结果notebook](linear_grid_and_split_lowbands/code/results.ipynb)。九份权重、完整预测及四组科研图表均已保存。四路有符号低频是这组已完成特征实验中平均测试r最高的输入表示。

| 组别 | 实际输入特征 |
|---|---|
| `morlet` | 40 个 40–300 Hz 对数间隔 Morlet 功率特征，n_cycles=7 |
| `low_frequency` | 0.5–4、4–13、13–30 Hz 三个有符号低频幅度 |
| `hybrid` | 上述 40＋3 项混合 |

## 配对控制

每个被试在完整 400 秒训练记录上，用当前主实验的同时间相关性规则从混合表示选择 16 个电极。输入缩放、标签 MinMax 和电极集合只拟合训练数据，三组共用。低频缩放后增益固定为 200；RobustScaler 的 quantile_range=(0.1,0.9)、unit_variance=True 保持原配置。

三组模型输入都保留 `(B,16,43,T)`，在缩放后将未使用的特征槽严格置零。三组的编码器、瓶颈双向 GRU、解码器、参数量（6,805,126）和初始化完全相同，训练窗口序列也相同。未使用槽的输入卷积梯度为零。该实现控制了改变输入宽度造成的参数量及初始权重变化；比较针对固定训练电极集合，不包含为各频率表示独立优化电极的影响。

每个被试三组都分别训练相同轮数：S1 112，S2 69，S3 104。batch=64，每轮8192个随机窗口；AdamW lr=8.42e-5、weight_decay=1e-4；dropout=0.1；损失为0.5 MSE＋0.5时间轴cosine；梯度范数裁剪为1。没有EMA、早停、额外输出处理、幅度校准或多种子集成。固定预算来自当前主实验配置，没有根据本次测试分数调整。

模型输出经训练 MinMax 逆变换后，线性插值到1000 Hz评分。2048点窗口全部输出，步长2048，只在末尾反射填充后裁到记录长度。每组每个被试评分完整200秒的200000个原始点。主指标为四指r（排除无名指），五指r作为辅助，三个被试等权平均。

## 文件组织

```text
code/
  main_source/              当前主实验核心源码的原样快照
  protocol.json            训练前固定的三组实验方案及数据和源码哈希
  common.py                共用训练统计、电极集合和特征掩码
  worker.py                一个被试一组特征的完整训练及预测
  run.py                   两个训练进程调度九次拟合，冻结权重后生成预测
  verify.py                三组初始化、掩码、梯度、512/2048尺寸验证
  remote.py                密码仅经不回显输入，独立服务器目录操作
  score_and_report.py      本地完整测试评分、对比表及科研图表
  results.ipynb            已执行的结果表、图像和一致性记录
  execute_results.py       评分后重新执行结果 notebook 并保存输出
result/
  server_results/          下载的权重、训练曲线、配置、预测、日志及冻结清单
  summary.json             全部正式分数
  subject_metrics.csv      各组各被试各指分数及训练信息
  mean_metrics.csv         三组平均四指/五指r、MAE、MSE
  contrasts.csv            各组间配对差值
  feature_ablation.*       正式完整测试对比图，PNG/SVG/PDF
  training_curves.png      九次训练的学习曲线
  report.md                结果、协议和可用于论文的英文段落
  verification.json        完成状态、覆盖与一致性检查
```

服务器目录为 `/root/autodl-tmp/bci_paper_frequency_features_20261008`，读取已存在的原始数据及 Morlet＋低频缓存，不修改其他实验目录。数据 SHA256 与本地官方文件匹配。测试标签保留在本地：九份预测生成并冻结后才读取标签评分。当前公开测试分数此前已可见，因此该消融不称为新的盲测。

## 运行

训练使用服务器已有的 `/root/autodl-tmp/bci_backbones_20261008/venv/bin/python` 环境。核心依赖与主实验相同：PyTorch2.5.1、MNE1.8.0、NumPy2.1.2、SciPy1.16.3、scikit-learn1.7.2。SSH客户端另需Paramiko；本地评分需Matplotlib。

在 `code` 目录执行远程操作，密码在提示时输入，不写入代码或配置：

```bash
python remote.py setup
python remote.py launch
python remote.py monitor
python remote.py fetch
```

`setup` 和 `launch` 拒绝覆盖已有实验或重复启动，并在启动前核对GPU空闲。训练失败时查看 `result/server_status.json` 和服务器日志，保留原记录。

下载后在本地读取官方测试标签评分：

```bash
python score_and_report.py
python execute_results.py
```

执行结果 notebook 另需 `IPython` 和 `nbformat`。`execute_results.py` 在本进程执行结果读取单元，不启动独立 Jupyter 内核。

标签默认读取本项目 `paper/data/BCICIV_4_mat`，可用 `--labels-dir` 修改。所有测试样本均保留；报告真实单种子结果，不混入此前多种子集成分数。当前混合组也为这次新训练模型，不复用主实验目录已有的历史权重。
