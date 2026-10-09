# BCI main experiment

This directory contains the self-contained 44-feature RecurFlex implementation and the three completed seed-42 checkpoints in `artifacts/pretrained/`.

See the [repository README](../../../README.md) for the model overview, data layout, scores, and commands. The default input location is `../../data/BCICIV_4_mat/`.

```bash
python -m pip install -r requirements.txt
python main.py --stage test --device cuda
python main.py --stage all --run-dir runs/retrained42 --device cuda
python verify.py --device cpu
```

Use `--device cpu` when CUDA is unavailable. Training uses fixed budgets of 112/69/104 epochs, 512-sample windows, and 8192 windows per epoch. Inference uses consecutive 2048-sample blocks and evaluates all original test timestamps.

`results/` holds the reported full-test metrics and original verification evidence. `artifacts/pretrained/provenance.json` records checkpoint hashes and original source locations; those historical paths are provenance, not runtime dependencies.

Raw recordings, labels, feature caches, and training runs are obtained or generated separately. The historical export script depended on other experiments and is omitted; the CLI writes predictions and scores directly to the selected run directory.
