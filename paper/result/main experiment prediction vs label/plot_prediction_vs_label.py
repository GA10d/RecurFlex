"""Draw saved seed-42 test predictions against the original glove labels."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.io import loadmat

FINGERS = ["Thumb", "Index", "Middle", "Ring (auxiliary)", "Little"]
FOUR = [0, 1, 2, 4]
COLORS = {"label": "#264E70", "prediction": "#D76A35"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def correlations(label, prediction):
    a = label - label.mean(0)
    b = prediction - prediction.mean(0)
    return (a * b).sum(0) / np.maximum(np.sqrt((a * a).sum(0) * (b * b).sum(0)), 1e-12)


def load_records(prediction_dir, labels_dir):
    manifest = read_json(prediction_dir / "prediction_manifest.json")
    summary = read_json(prediction_dir / "summary.json")
    scores = {r["subject"]: r for r in summary["subjects"]}
    records = {}
    provenance = {}
    for item in manifest["predictions"]:
        subject = item["subject"]
        prediction_path = prediction_dir / item["prediction"]
        label_path = labels_dir / f"sub{subject}_testlabels.mat"
        assert sha256(prediction_path) == item["prediction_sha256"], prediction_path
        assert sha256(label_path) == scores[subject]["label_sha256"], label_path
        with np.load(prediction_path) as arrays:
            p = arrays["prediction_1000hz"].astype(np.float64)
        y = loadmat(label_path, variable_names=["test_dg"])["test_dg"].astype(np.float64)
        assert p.shape == y.shape == (200000, 5)
        assert np.isfinite(p).all() and np.isfinite(y).all()
        r = correlations(y, p)
        assert np.allclose(r, scores[subject]["r_fingers"], rtol=0, atol=1e-10)
        assert abs(r[FOUR].mean() - scores[subject]["r4"]) < 1e-10
        records[subject] = dict(label=y, prediction=p, r=r,
                                r4=float(r[FOUR].mean()), r5=float(r.mean()))
        provenance[f"S{subject}"] = dict(prediction_path=str(prediction_path.resolve()),
             prediction_sha256=sha256(prediction_path), label_path=str(label_path.resolve()),
             label_sha256=sha256(label_path), r_fingers=r.tolist(), r4=float(r[FOUR].mean()),
             r5=float(r.mean()), scoring_samples=200000)
    assert sorted(records) == [1, 2, 3]
    return records, provenance


def legend(fig, y):
    handles = [Line2D([0], [0], color=COLORS["label"], lw=1.6, label="Recorded label"),
               Line2D([0], [0], color=COLORS["prediction"], lw=1.3, label="RecurFlex prediction")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=2,
               frameon=False, handlelength=3, columnspacing=3)


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#B5BEC7")
        ax.spines[name].set_linewidth(0.65)
    ax.grid(axis="y", color="#E5E9ED", lw=0.6)
    ax.tick_params(colors="#3F4D5A", length=3, width=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))


def draw(ax, record, finger, indices, start, end):
    t = indices / 1000.
    ax.plot(t, record["label"][indices, finger], color=COLORS["label"], lw=1.05,
            alpha=0.98, zorder=2)
    ax.plot(t, record["prediction"][indices, finger], color=COLORS["prediction"],
            lw=0.8, alpha=0.9, zorder=3)
    ax.set_xlim(start, end)
    ax.text(0.985, 0.93, f"Full-test r = {record['r'][finger]:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#263746", bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=2))
    axis_style(ax)


def save(fig, stem, output):
    paths = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=350, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot(records, output, start, end, display_hz):
    stride = 1000 // display_hz
    indices = np.arange(round(start * 1000), round(end * 1000), stride)
    if len(indices) < 2:
        raise ValueError("The displayed interval must contain at least two points.")
    interval = "full_test_200s" if (start, end) == (0, 200) else f"{start:g}_{end:g}s"
    footer = (f"Display: {display_hz} Hz, direct sampling. Scores: all 200,000 original points at 1,000 Hz. "
              "Ring is excluded from R4. No smoothing or amplitude calibration.")
    outputs = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "svg.fonttype": "none", "savefig.dpi": 350})
    fig, axes = plt.subplots(5, 3, figsize=(15.5, 11.2), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.075, top=0.845,
                        hspace=0.22, wspace=0.1)
    mean_r4 = np.mean([r["r4"] for r in records.values()])
    fig.suptitle("RecurFlex prediction vs. recorded finger flexion", fontsize=17, y=0.975, weight="bold")
    fig.text(0.5, 0.944, f"High Morlet + 4 signed low bands  |  Seed 42  |  {start:g}-{end:g} s  |  Mean R4 = {mean_r4:.6f}",
             ha="center", color="#536170", fontsize=11)
    legend(fig, 0.919)
    for column, subject in enumerate(sorted(records)):
        record = records[subject]
        for finger in range(5):
            ax = axes[finger, column]
            draw(ax, record, finger, indices, start, end)
            if finger == 0:
                ax.set_title(f"S{subject}  |  R4 = {record['r4']:.4f}", pad=12, weight="bold")
            if column == 0:
                ax.set_ylabel(FINGERS[finger] + "\nFlexion (glove units)", labelpad=9)
            if finger == 4:
                ax.set_xlabel("Time (s)")
                ax.set_xticks(np.linspace(start, end, 5))
    for finger in range(5):
        values = np.concatenate([r[key][:, finger] for r in records.values()
                                 for key in ("label", "prediction")])
        span = max(float(values.max() - values.min()), 0.1)
        axes[finger, 0].set_ylim(values.min() - .06 * span, values.max() + .18 * span)
    fig.text(0.5, 0.018, footer, ha="center", fontsize=8.5, color="#61707D")
    outputs.extend(save(fig, f"all_subjects_prediction_vs_label_{interval}", output))
    for subject, record in sorted(records.items()):
        fig, axes = plt.subplots(5, 1, figsize=(15.5, 10.2), sharex=True)
        fig.subplots_adjust(left=.075, right=.99, bottom=.085, top=.855, hspace=.28)
        fig.suptitle(f"Subject {subject}: prediction vs. recorded finger flexion", fontsize=17, y=.975, weight="bold")
        fig.text(.5, .943, f"High Morlet + 4 signed low bands  |  Seed 42  |  {start:g}-{end:g} s  |  R4 = {record['r4']:.6f}  |  R5 = {record['r5']:.6f}",
                 ha="center", color="#536170", fontsize=11)
        legend(fig, .92)
        for finger, ax in enumerate(axes):
            draw(ax, record, finger, indices, start, end)
            ax.set_ylabel(FINGERS[finger] + "\nFlexion (glove units)", labelpad=10)
            values = np.concatenate([record[key][:, finger] for key in ("label", "prediction")])
            span = max(float(values.max() - values.min()), .1)
            ax.set_ylim(values.min() - .06 * span, values.max() + .18 * span)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xticks(np.linspace(start, end, 11))
        fig.text(.5, .022, footer, ha="center", fontsize=8.5, color="#61707D")
        outputs.extend(save(fig, f"s{subject}_prediction_vs_label_{interval}", output))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=HERE.parent / "main experiment average r")
    parser.add_argument("--labels-dir", type=Path, default=HERE.parent.parent / "data/BCICIV_4_mat")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=200)
    parser.add_argument("--display-hz", type=int, choices=(10, 25, 50, 100, 200, 500, 1000), default=100)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 200:
        parser.error("Require 0 <= start < end <= 200 seconds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, inputs = load_records(args.prediction_dir, args.labels_dir)
    outputs = plot(records, args.output_dir, args.start, args.end, args.display_hz)
    tag = "full_test_200s" if (args.start, args.end) == (0, 200) else f"{args.start:g}_{args.end:g}s"
    csv_path = args.output_dir / f"finger_correlations_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "thumb", "index", "middle", "ring", "little", "r4", "r5"])
        for subject, record in sorted(records.items()):
            writer.writerow([subject, *record["r"], record["r4"], record["r5"]])
    outputs.append(csv_path)
    provenance = dict(seed=42, features=44, feature_design="40 high Morlet power + 4 signed low bands",
                      displayed_interval_seconds=[args.start, args.end], display_hz=args.display_hz,
                      display_operation="Every kth original sample; no averaging or smoothing",
                      scoring_hz=1000, scoring_samples_per_subject=200000,
                      all_15_correlations_recomputed_and_verified=True, inputs=inputs,
                      outputs_sha256={p.name:sha256(p) for p in outputs},
                      script_sha256=sha256(__file__), visual_qa="Pending image inspection")
    save_json(args.output_dir / f"figure_provenance_{tag}.json", provenance)
    print(json.dumps({"saved_figures":len(outputs)-1, "directory":str(args.output_dir),
                      "r4":float(np.mean([r['r4'] for r in records.values()]))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
