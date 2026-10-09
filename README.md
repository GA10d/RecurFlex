# RecurFlex

### Multiband Recurrent Decoding of Finger Trajectories from ECoG

**Zhewen Guo** · Columbia University · [zg2567@columbia.edu](mailto:zg2567@columbia.edu)  
**Hongxun Peng** · Beihang University · [huxley329@buaa.edu.cn](mailto:huxley329@buaa.edu.cn)

RecurFlex reconstructs continuous finger trajectories from electrocorticography (ECoG). It combines high-frequency Morlet power with signed low-frequency voltage features, a temporal convolutional encoder–decoder, and a residual bidirectional GRU at the bottleneck.

[Manuscript source](paper/overleaf/main.tex) · [BCI implementation](paper/code/Main%20Experiment) · [BCI results](paper/result/main%20experiment%20average%20r/summary.json) · [Nine-subject results](paper/result/miller_fingerflex_9subjects_2of3/test_results.json) · [Feature ablations](paper/code/Ablation%20Experiment/frequency%20feature%20selection)

## Model overview

![RecurFlex architecture: multiband features, recurrent encoder–decoder, and trajectory output](assets/architecture.png)

- **Complementary features:** 40 log-spaced Morlet power bands at 40–300 Hz, plus signed delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), and beta (13–30 Hz) voltage.
- **Temporal representation:** 16 training-selected electrodes × 44 features at 100 Hz; five encoder stages, symmetric skip connections, and a residual BiGRU bottleneck.
- **Continuous output:** five finger trajectories; 512-sample training windows and consecutive 2,048-sample inference blocks. Outputs are restored to glove units and interpolated to the original 1,000 Hz grid.

The model has **6,808,198 parameters**. Zero-phase filtering and bidirectional recurrence make this an **offline reconstruction** method.

## Results

### BCI Competition IV Dataset 4

Each subject uses the complete 400-second training recording and the complete 200-second official test recording. Pearson correlations are computed over every original test timestamp, averaged first across fingers and then equally across subjects.

| Subject | Official four-finger r | Five-finger r | Four-finger MAE | Four-finger MSE |
|:--|--:|--:|--:|--:|
| S1 | 0.838068 | 0.845696 | 0.368927 | 0.352761 |
| S2 | 0.696298 | 0.704356 | 0.521532 | 0.604666 |
| S3 | 0.787917 | 0.800588 | 0.397740 | 0.401945 |
| **Mean** | **0.774094** | **0.783547** | **0.429400** | **0.453124** |

The official four-finger metric includes the thumb, index, middle, and little fingers. The ring finger is predicted and shown as an auxiliary output. MAE and MSE use the original glove units.

### Prediction vs. recorded labels

**Blue: recorded label. Orange: RecurFlex prediction.** Every panel shows the full 0–200 s test recording, with no display smoothing, time shift, or amplitude calibration. Display samples are taken directly at 100 Hz; scores use all 200,000 original 1,000 Hz samples per subject.

#### Subject 1 · four-finger r = 0.8381

![Subject 1: complete test prediction versus recorded finger flexion](assets/s1_prediction_vs_label.png)

#### Subject 2 · four-finger r = 0.6963

![Subject 2: complete test prediction versus recorded finger flexion](assets/s2_prediction_vs_label.png)

#### Subject 3 · four-finger r = 0.7879

![Subject 3: complete test prediction versus recorded finger flexion](assets/s3_prediction_vs_label.png)

### Miller fingerflex: nine-subject extension

| Evaluation | Subjects | Split | Four-finger r | Five-finger r |
|:--|--:|:--|--:|--:|
| Miller fingerflex | 9 | First 2/3 development; final 1/3 test, with a 2 s gap | 0.637346 | **0.643181** |

![Subject-level correlations in the nine-subject chronological evaluation](assets/miller_subject_correlations.png)

All nine models were initialized and trained separately with seed 42. Electrode selection, feature scaling, and target ranges use the corresponding training segment. The primary metric for this extension is the five-finger mean.

**Interpretation.** BCI Dataset 4 originates from the Miller fingerflex collection, so these evaluations have overlapping source cohorts. The nine-subject experiment uses a custom chronological split, not an official competition split. Published baselines differ in their splits, target delays, smoothing, and scoring grids; this release does not establish common-protocol superiority over all published methods. Earlier method exploration inspected the public BCI test, and the reported results are single-seed results. See the [manuscript discussion](paper/overleaf/sections/04_results.tex).

## Reproduce the BCI results

Use Python 3.11/3.12. The repository includes the three trained BCI checkpoints and scaling statistics. Obtain the original dataset and test labels separately and place them in:

```text
paper/data/BCICIV_4_mat/
├── sub1_comp.mat
├── sub2_comp.mat
├── sub3_comp.mat
├── sub1_testlabels.mat
├── sub2_testlabels.mat
└── sub3_testlabels.mat
```

The competition files contain `train_data`, `train_dg`, and `test_data`; label files contain `test_dg`. Raw recordings and test labels are not distributed in this repository.

```bash
cd "paper/code/Main Experiment"
python -m pip install -r requirements.txt
python main.py --stage test --device cuda
```

Use `--device cpu` for CPU execution, or `--data-dir /path/to/data --labels-dir /path/to/labels` for external data locations. Predictions are saved before evaluation reads test labels. Outputs appear in `runs/main/predictions/` and `runs/main/results/`.

```bash
# Train fresh models with the published subject-specific budgets, then evaluate.
python main.py --stage all --run-dir runs/retrained42 --device cuda

# Run the implementation's model/checkpoint verification.
python verify.py --device cpu
```

The included checkpoints were trained for 112, 69, and 104 epochs for S1, S2, and S3. The reported scores were obtained by rerunning full inference with those completed checkpoints, rather than by repeating full training during packaging. Platform-dependent numerical differences are possible.

For the nine-subject experiment, [the source and frozen protocol](paper/code/Miller%20Experiment%202of3/code) document the original preparation, training, and scoring pipeline. It requires separately obtained Miller recordings and experiment input preparation; the BCI quick-start command does not run this experiment. Nine-subject checkpoints are not included.

## Repository contents

```text
assets/                       Architecture and README result figures
paper/code/Main Experiment/   BCI preprocessing, model, training, inference, checkpoints
paper/code/Miller Experiment 2of3/code/
                              Nine-subject experiment source and frozen protocol
paper/result/                 Saved scores, prediction files, and plotting source
paper/overleaf/               IEEE-format LaTeX manuscript source
```

The paper source can be uploaded to Overleaf with `main.tex` as the entry point and pdfLaTeX as the compiler.

## Citation

Until a DOI or arXiv identifier is available, cite the repository:

```bibtex
@misc{guo_recurflex_2026,
  author = {Guo, Zhewen and Peng, Hongxun},
  title = {RecurFlex: Multiband Recurrent Decoding of Finger Trajectories from ECoG},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/GA10d/RecurFlex}
}
```

## Contact

Questions about the manuscript and experiments: [Zhewen Guo](mailto:zg2567@columbia.edu) and [Hongxun Peng](mailto:huxley329@buaa.edu.cn).
