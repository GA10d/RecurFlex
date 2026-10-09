# RecurFlex 0.5 至 300 Hz 特征比较

在同一主模型上比较 40 个全频段 Morlet 功率特征与 40 路有符号对数频带波形。两组均先对原始 1000 Hz ECoG 按通道标准化、中位数参考，再做 0.5 至 300 Hz 带通和 60 Hz 谐波陷波。

## 已完成的全量训练与完整测试

2026-10-08，六个新模型均完成全量训练，完整预测已冻结、下载和核对哈希。每个被试评分 200000 个原始测试点，三个被试等权平均，四指排除无名指。

| 特征表示 | 四指平均 r | 五指平均 r |
|---|---:|---:|
| 全频段 Morlet 功率，40 点 | 0.398289 | 0.422182 |
| 有符号对数频带波形，40 路 | 0.355140 | 0.388331 |
| 原 Morlet＋三路低频混合，上一轮参考 | **0.758373** | **0.771415** |

全频段 Morlet 比有符号对数频带的四指平均 r 高 0.043149，两个新配方均低于原混合表示。完整训练记录上的附加推理相关性均约 0.996，显示明显的训练与测试泛化差距，未追加训练或更改权重。

查看 [详细报告与论文英文段落](result/report.md)、[新两组测试图](result/broadband_comparison.png)、[五种表示比较](result/all_feature_comparison.png)、[频率网格图](result/frequency_design.png) 和 [结果 notebook](code/results.ipynb)。图表均另存 SVG/PDF 矢量版本。

40 个频率中心为 `logspace(log10(0.5), log10(300), 40)`。Morlet 保持 `n_cycles=7`、`zero_mean=False`，先计算功率再每 10 点抽取 1 点。带通组以相邻频率中心的几何中点划分边界，两端裁到 0.5 和 300 Hz，使用与主实验低频分支相同的四阶 Butterworth、`sosfiltfilt` 和有符号幅度，每 10 点取 1 点。

两组各自用完整训练记录拟合 RobustScaler，`quantile_range=(0.1,0.9)`、`unit_variance=True`。Morlet 缩放后增益为 1，有符号组沿用主实验低频增益 200，应用到全部 40 路。没有取绝对值、平方或额外抗混叠滤波。超过 50 Hz 的有符号波形抽取到 100 Hz 后会发生频率折叠；这组实验检验的是用户指定的直接抽取方案，与功率特征的时间包络不同。

## 固定主模型与训练协议

复用上一轮消融在训练数据上选定的每个被试 16 个电极和标签 MinMax。模型输入保持 `(B,16,43,T)`，40 路新特征放在前 40 槽，最后 3 槽严格为零，两组均无原始三路低频分支。主干及 6,805,126 参数、seed 42 初始化、随机训练窗口序列保持一致。

每组三个被试分别从头训练 112、69、104 轮，每轮 8192 个 512 点窗口，batch 64。AdamW lr 8.42e-5、weight_decay 1e-4、dropout 0.1，损失为 0.5 MSE 加 0.5 时间轴 cosine distance。按同时间标签配对，使用最终普通权重、2048 点全输出推理，无多种子集成及额外输出处理。

模型和预测先在服务器冻结，再在本地评分每个被试完整 200 秒、200000 个 1000 Hz 测试点，三个被试等权平均，四指指标排除无名指，五指为辅助。前一轮三组结果作为同模型参考列单独标识。

## 文件与运行

`code/main_source` 保留当前主模型原样快照；`protocol.json` 固定频率网格、频带边界、数据哈希和全部训练设置；`features.py` 实现特征提取；`common.py` 实现训练缩放和输入组织。

服务器目录独立为 `/root/autodl-tmp/bci_paper_broadband_features_20261008`，原数据读取服务器已有文件。新特征缓存、权重和预测保存在该独立目录，不覆盖上一轮实验。

在 `code` 目录运行 `remote.py setup`、`remote.py launch`、`remote.py monitor`、`remote.py fetch`。SSH 密码仅在不回显提示时输入。下载后运行 `score_and_report.py`，完整分数、曲线、图表和论文英文段落保存在 `result`。

`score_and_report.py` 使用主实验的 Python 环境，依赖 NumPy、SciPy、Matplotlib 和 PyTorch。测试标签默认读取项目 `paper/data/BCICIV_4_mat`。评分会核对上一轮相同测试标签的哈希，以及相同电极、目标缩放、初始化和窗口采样记录。

```text
code/
  features.py              全频段预处理、Morlet功率和有符号对数频带
  common.py                训练缩放、固定电极和43槽输入组织
  worker.py                一个特征组、一个被试的从头训练与推理
  run.py                   两进程执行六次训练，冻结权重和预测
  verify.py                特征符号、抽取、梯度和模型尺寸校验
  remote.py                指定服务器的独立实验操作
  score_and_report.py      完整测试评分与论文图表
  results.ipynb            测试表格与图像
  execute_results.py       评分后执行并保存notebook输出
result/
  server_results/          六份新权重、预测、配置、日志和冻结清单
  summary.json             两组正式完整测试分数
  subject_metrics.csv      每个被试、每个手指的分数及训练预算
  mean_metrics.csv         三个被试等权平均
  all_feature_comparison.csv  上一轮三组及本次两组对照
  contrasts.csv            新两组之间的配对差值
  broadband_comparison.*   新两组完整测试图，PNG/SVG/PDF
  all_feature_comparison.* 五种表示的比较图，PNG/SVG/PDF
  frequency_design.*       频率网格与频带边界图，PNG/SVG/PDF
  frequency_grid.csv       精确40个频率中心与上下界
  training_curves.png      六次训练曲线
  report.md                结果和论文英文段落
  verification.json        完整性与一致性校验
```

评分完成后运行 `python execute_results.py` 将表格和图像保存进 notebook；这一步另需 IPython 和 nbformat，不启动独立 Jupyter 内核。
