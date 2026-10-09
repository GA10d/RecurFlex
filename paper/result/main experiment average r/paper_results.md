# RecurFlex 高频 Morlet＋四路低频主实验

每电极使用 40 个 40–300 Hz 对数间隔 Morlet 功率，以及 0.5–4、4–8、8–13、13–30 Hz 四路有符号波形，共 44 个特征。主干为五级时间卷积编码解码器与瓶颈残差双向 GRU，参数量 6,808,198；输入 `(B,16,44,T)`，输出 `(B,5,T)`。

三个被试各一个 seed 42 模型，已经在服务器上完成全量训练：S1 112 轮、S2 69 轮、S3 104 轮，每轮 8192 个窗口。训练窗口为 512 个 100 Hz 时间点，输入和目标使用相同时间；推理使用连续 2048 点窗口，保留全部输出。附带权重与服务器完成的 `hybrid_4` 权重逐张量、逐文件一致。此次用整理后的 Main Experiment 从原始 ECoG 重新提取特征、执行三个被试的完整推理与评分。

## 完整官方测试结果

每名被试评分全部 200 秒，即原始 1000 Hz 的 200000 个点。四指 r 对拇指、食指、中指、小指平均，再对三个被试等权平均。五指 r 为辅助指标；MAE、MSE 使用原始手套标签单位。

| 被试 | 四指 r | 五指 r | 四指 MAE | 四指 MSE |
|---|---:|---:|---:|---:|
| S1 | 0.838068 | 0.845696 | 0.368927 | 0.352761 |
| S2 | 0.696298 | 0.704356 | 0.521532 | 0.604666 |
| S3 | 0.787917 | 0.800588 | 0.397740 | 0.401945 |
| 平均 | **0.774094** | **0.783547** | **0.429400** | **0.453124** |

## 可用于论文的结果段落

RecurFlex combines 40 log-spaced Morlet power features between 40 and 300 Hz with four signed low-frequency bands (0.5–4, 4–8, 8–13, and 13–30 Hz). We trained one seed-42 model per subject on the complete 400-second training recording using 512-sample windows at 100 Hz and same-time targets. Inference used consecutive 2,048-sample blocks with all outputs retained. Evaluation on every original test sample yielded a mean four-finger Pearson correlation of 0.7741, a mean five-finger correlation of 0.7835, and mean four-finger MAE and MSE of 0.4294 and 0.4531. Outputs were restored to glove units and linearly interpolated to 1,000 Hz.

## 实现与来源

高频支路在通道标准化、中位数参考后进行 40–300 Hz FIR 带通与 60 Hz 谐波陷波，再计算 7-cycle Morlet 功率，每 10 点取 1 点。四路低频从高频带通之前的参考信号计算，使用四阶零相位 Butterworth 并保留正负号。选定电极的 44 项特征分别用训练内部 `225:-25` 样本拟合 RobustScaler，分位数为 0.1%/0.9%，低频增益 200。全量训练沿用已冻结、由训练数据得到的 16 电极集合，44 项缩放统计已重新拟合；内部验证模式独立在训练段选择电极。

训练从随机初始化开始。为了保持频带拆分实验的共享参数初值一致，44 槽输入保留原 43 槽随机模型的共享张量，原 beta 输入权重映射到槽 43，新增 alpha 槽 42 使用 seed 42 的随机初值。`model.training_model()` 实现相同规则。

当前数值对应已经完成全量训练的四路低频模型，经新代码完整重测；本次迁移没有再执行一遍 112/69/104 轮训练。真实全记录训练、单次优化更新与保存加载检查保存在 `training_smoke.json`；预测与来源实验的数值差异保存在 `reproduction_check.json`。旧三路低频主实验保存在 `code/Main Experiment/archive/legacy_hybrid3_20261008/`。
