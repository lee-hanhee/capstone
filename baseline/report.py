"""Generate presentation figures and ten 20-second slide scripts from a real run."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

# Use Matplotlib's bundled fonts; macOS system font discovery can be very slow.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from baseline.data import CLASSES


def generate(data_dir, run_dir, output):
    result = json.loads((run_dir / "metrics.json").read_text())
    if result["status"] != "trained":
        raise ValueError("Presentation results require a real pretrained-model run")
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    with (run_dir / "test_predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    colors = ["#2463a6", "#e58b2a", "#278568"]

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(3)
    for i, split in enumerate(["train", "val", "test"]):
        counts = manifest["counts"][split]
        vals = [counts["positive_frames"][c] for c in CLASSES]
        bars = ax.bar(x + (i-1)*0.24, vals, width=0.24,
                      label=f"{split}: {counts['frames']} frames", color=colors[i])
        ax.bar_label(bars, padding=3)
    ax.set(xticks=x, xticklabels=CLASSES, ylabel="Positive source frames",
           title="One observation per unmarked source frame")
    ax.set_ylim(0, max(manifest["counts"]["train"]["positive_frames"].values()) * 1.2)
    ax.legend()
    fig.tight_layout()
    save(fig, "dataset_balance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = [r["epoch"] for r in history]
    for name, color in [("train_loss", colors[0]), ("val_loss", colors[1])]:
        axes[0].plot(epochs, [r[name] for r in history], marker="o", label=name, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Training and validation loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_macro_f1"] for r in history], marker="o", color=colors[2])
    axes[1].axvline(result["best_epoch"], color="black", linestyle=":", label="Selected checkpoint")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro F1", ylim=(0, 1), title="Checkpoint selection")
    axes[1].legend()
    for ax in axes:
        ax.axvline(3.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_xticks(epochs)
    fig.suptitle("Epochs 1–3: train heads; epochs 4–10: fine-tune all layers")
    fig.tight_layout()
    save(fig, "learning_curves")

    test = result["test"]
    fig, ax = plt.subplots(figsize=(11, 5))
    names = ["precision", "recall", "f1", "accuracy"]
    for i, name in enumerate(names):
        vals = [test["per_class"][c][name] for c in CLASSES]
        bars = ax.bar(x + (i-1.5)*0.19, vals, width=0.19, label=name.title())
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set(xticks=x, xticklabels=CLASSES, ylim=(0, 1.18), ylabel="Score",
           title=f"Held-out video test: {test['frames']} frames, threshold 0.5")
    ax.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    save(fig, "class_metrics")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, label in zip(axes, CLASSES):
        cm = np.array(test["per_class"][label]["confusion_matrix_tn_fp_fn_tp"]).reshape(2, 2)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22,
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Absent", "Present"],
               yticklabels=["Absent", "Present"], xlabel="Predicted", ylabel="Actual", title=label.title())
    fig.suptitle("Test confusion matrices (raw frame counts)")
    fig.tight_layout()
    save(fig, "confusion_matrices")

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = {"Always absent": result["baselines"]["always_absent"],
               "Training majority": result["baselines"]["train_majority"], "MobileNetV3": test}
    for i, name in enumerate(["label_accuracy", "exact_match_accuracy", "macro_f1"]):
        bars = ax.bar(np.arange(3)+(i-1)*0.24, [m[name] for m in methods.values()],
                      width=0.24, label=name.replace("_", " "), color=colors[i])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set(xticks=np.arange(3), xticklabels=list(methods), ylim=(0, 1.2), ylabel="Score",
           title="Constant predictions can score well on imbalanced labels")
    ax.legend(ncol=3, loc="upper center", fontsize=11)
    fig.tight_layout()
    save(fig, "baseline_comparison")

    # Deterministic illustrations: first exact success and first error per class.
    selected, used = [], set()
    for error in [False, True]:
        for c in CLASSES:
            for r in predictions:
                y = np.array([int(r[f"true_{k}"]) for k in CLASSES])
                p = np.array([float(r[f"prob_{k}"]) for k in CLASSES]) >= 0.5
                match = ((y != p).any() if error else (y == p).all() and int(r[f"true_{c}"]) == 1)
                if error:
                    match = y[CLASSES.index(c)] != p[CLASSES.index(c)]
                if match and r["image"] not in used:
                    selected.append((r, error, c))
                    used.add(r["image"])
                    break
            else:
                selected.append((None, error, c))
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (row, error, label) in zip(axes.flat, selected):
        if row is None:
            ax.text(0.5, 0.5, f"No matching {'error' if error else 'exact success'}\nfor {label}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)
            continue
        with Image.open(data_dir / row["image"]) as im:
            ax.imshow(im)
        actual = ", ".join(c for c in CLASSES if int(row[f"true_{c}"])) or "none"
        predicted = ", ".join(c for c in CLASSES if float(row[f"prob_{c}"]) >= 0.5) or "none"
        ax.set_title(f"{'Error' if error else 'Correct'}: {Path(row['source_video']).stem}, f{row['frame_index']}\n"
                     f"Actual: {actual}\nPredicted: {predicted}", fontsize=10,
                     color="#b03636" if error else "#227756")
    fig.suptitle("Test examples: first matching cases, not a representative sample")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.85,
                        hspace=0.55, wspace=0.16)
    save(fig, "test_examples")

    gpu, cpu = result["latency_training_device"], result["latency_server_cpu"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(["Modal T4 GPU", "Modal CPU (1 thread)"],
                  [gpu["mean_ms"], cpu["mean_ms"]], color=colors[:2])
    ax.bar_label(bars, fmt="%.2f ms", padding=6)
    ax.set(ylabel="Mean latency (ms)", title="Warm batch-one model forward pass + sigmoid")
    ax.set_ylim(0, max(gpu["mean_ms"], cpu["mean_ms"])*1.3)
    fig.text(0.5, 0.01, "100 repeats after 20 warmups. iOS and request-to-speech latency unmeasured.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "server_latency")

    for name in ["config.json", "metrics.json", "history.json", "test_predictions.csv"]:
        shutil.copy2(run_dir / name, output / name)
    summary = {k: v for k, v in manifest.items() if k != "rows"}
    (data_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def pct(x):
        return f"{100*x:.1f}%"

    table = "| Class | Positive / negative | Accuracy | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    for c in CLASSES:
        m = test["per_class"][c]
        table += f"| {c} | {m['positives']} / {m['negatives']} | " + " | ".join(pct(m[k]) for k in ["accuracy", "precision", "recall", "f1"]) + " |\n"
    baseline_table = "| Method | Label accuracy | Exact match | Macro F1 |\n| --- | ---: | ---: | ---: |\n"
    for name, m in methods.items():
        baseline_table += f"| {name} | {pct(m['label_accuracy'])} | {pct(m['exact_match_accuracy'])} | {m['macro_f1']:.3f} |\n"
    best_class = max(CLASSES, key=lambda c: test["per_class"][c]["f1"])
    worst_class = min(CLASSES, key=lambda c: test["per_class"][c]["f1"])
    restaurant = test["per_class"]["restaurant"]
    tn, fp, fn, tp = restaurant["confusion_matrix_tn_fp_fn_tp"]
    restaurant_note = (
        f"The restaurant head produced {tp} true positives, {fn} false negatives, and "
        f"{fp} false positives at the confirmed 0.5 threshold. Its precision, recall, "
        f"and F1 are {restaurant['precision']:.3f}, {restaurant['recall']:.3f}, and "
        f"{restaurant['f1']:.3f}. Its accuracy is {pct(restaurant['accuracy'])}, versus "
        f"{pct(result['baselines']['always_absent']['per_class']['restaurant']['accuracy'])} "
        "for always absent, illustrating why accuracy alone can mislead. "
        "The threshold has not been changed after viewing test results. "
        f"Restaurant ROC AUC is {restaurant['roc_auc']:.3f}; ranking quality does not "
        "establish useful fixed-threshold predictions. Future threshold selection "
        "must use validation data and a fresh independent evaluation."
    )
    if tp == 0:
        restaurant_note += " This baseline failed to retrieve any restaurant-positive test frame."
    usage_note = "Actual billing must be checked in Modal; the estimate is not an invoice."
    usage_path = output / "modal_usage.json"
    if usage_path.exists():
        from decimal import Decimal
        usage = json.loads(usage_path.read_text())
        amount = sum(Decimal(r["cost"]) for r in usage["reported_rows"])
        usage_note = (f"Modal's usage report currently attributes ${amount:.8f} to this app, "
                      "within the $10 budget. This is reported usage at check time, not a final "
                      "invoice; costs may lag or be attributed separately to image builds. "
                      "See [usage record](modal_usage.json).")
    report = f"""# Measured baseline results

One confirmed experiment; no hyperparameter search or test-driven checkpoint selection.
Best validation checkpoint: epoch {result['best_epoch']} of {result['epochs_completed']}.
Validation macro F1: {result['validation']['macro_f1']:.3f}; test macro F1: {test['macro_f1']:.3f}.

## Held-out test performance

{table}
{baseline_table}
Label accuracy averages the three binary decisions per image. Exact match requires
all three decisions to be correct. Macro F1 gives equal weight to the three sign
classes. These are presence metrics; they do not measure text reading, arrow
association, sign localization, or correct spoken responses.

No-relevant-sign inference is the conjunction of all three absent predictions.
Its test precision is {pct(test['irrelevant']['precision'])}, recall {pct(test['irrelevant']['recall'])},
F1 {test['irrelevant']['f1']:.3f}, and accuracy {pct(test['irrelevant']['accuracy'])}.

## Architecture and method

ImageNet-pretrained MobileNetV3-Small from Hugging Face, one shared feature extractor,
three independent linear binary heads, sigmoid outputs, fixed 0.5 thresholds.
Measured model parameters: {result['parameters']:,}; checkpoint: {result['checkpoint_bytes']/1024**2:.2f} MiB.
The checkpoint includes weights, configuration, and validation metrics.
Removing the original 1,000-way classifier explains the smaller parameter count
than the roughly 2.5-million-parameter pretrained model card.

MobileNetV3 targets mobile CPU efficiency; sharing features avoids running three
backbones when answering all three questions. Transfer learning provides a starting
representation with limited data. Independent binary heads allow co-occurring
classes. Weighted BCE addresses positive-class imbalance using training counts only.
These are design rationales, not measured comparisons against other architectures.

224x224 RGB inputs, bilinear aspect-ratio-preserving padding, ImageNet normalization,
no augmentation. Batch 32, seed 42, AdamW, weight decay 0.0001. Three frozen-backbone
epochs at head learning rate 0.001, then seven all-layer fine-tuning epochs at 0.0001.
Frozen-backbone batch normalization stays in evaluation mode. Checkpoint selection
uses validation macro F1 only; test was evaluated after selection. Full settings,
training-only class weights, software versions, and Hugging Face revision are in
[config.json](config.json).

## Data and evaluation limits

1,692 unmarked source frames are retained; 208 all-three-positive frames are excluded
with user approval because every available copy contains a red box. This removes
54.5% of original restaurant-positive frames and all triple-positive observations.
The resulting test distribution does not represent the complete original dataset.
Original instance boxes are retained in `data/baseline/manifest.json` for later
detection and in the unchanged source ground-truth files.

13/3/3 videos produce 1,209 training, 249 validation, and 234 test frames. A seeded
class-coverage check chooses the first split with positives and negatives in every
class, with at least two positive videos per class per partition. The second seeded
assignment qualifies. Test videos are IMG_8457, IMG_8460, and IMG_8466. The split
uses annotation labels only and is fixed before training. There are only three test
videos, and frames within a video are correlated; scores are descriptive rather
than evidence of broad generalization. Distinct videos may show the same physical
signs. No unseen-airport or unseen-sign claim is supported.

## What worked and what remains weak

The complete data-to-training-to-inference pipeline runs on unmarked images, retains
multi-label observations, and keeps source videos separate. On this test split,
{best_class} has the highest class F1 and {worst_class} the lowest. See
[confusion matrices](figures/confusion_matrices.png) for the actual false positives
and false negatives and [test examples](figures/test_examples.png) for deterministic
illustrations. The examples are selected for illustration and do not estimate error
frequencies. Do not attribute errors to small signs or blur without reviewing
the original full-resolution frames; those are hypotheses for a later analysis.

{restaurant_note}

Validation macro F1 ({result['validation']['macro_f1']:.3f}) versus test macro F1
({test['macro_f1']:.3f}) shows a gap. Training loss falls
while validation loss rises after the frozen phase, consistent with overfitting
or distribution differences. These diagnostics motivate broader independent data;
they do not establish a single cause or a measured fix.

The first unconstrained split lacked restaurant-positive test frames; class coverage
checks resolved that evaluation problem. Training on drawn boxes was rejected
before model fitting because of label leakage risk. Neither issue is an unsuccessful
architecture experiment. Only MobileNetV3-Small was trained; no claim that it beats
other neural architectures or that fine-tuning caused an improvement is supported.

## Latency, cost, and proposal objectives

T4 mean model latency: {gpu['mean_ms']:.2f} ms, median {gpu['median_ms']:.2f} ms,
p95 {gpu['p95_ms']:.2f} ms. Modal server CPU (one thread): mean {cpu['mean_ms']:.2f} ms,
median {cpu['median_ms']:.2f} ms, p95 {cpu['p95_ms']:.2f} ms. Measurements use batch one,
20 warmups, 100 repeats, and synchronized GPU timing; forward pass plus sigmoid
only. CPU model is float32. Excludes decode, preprocessing, camera, network,
speech and cold start. The saved device and platform information is in config.json.

No iPhone is selected, so **iOS latency is unmeasured**. No accuracy or latency pass
threshold is defined. These measurements establish a starting point rather than
showing that a product requirement has been met. Presence evaluation is one component
of the proposal's percentage of correct readings per task. Model latency is one
component of its request-to-response latency objective.

Training wall time inside the function: {result['wall_seconds']:.1f} seconds. The
single T4 function has a 1,800-second timeout, two CPU cores, 4 GiB RAM, no retries,
and no persistent volume or deployed endpoint. At the listed rates checked for
this experiment, that function's maximum compute estimate is about $0.36
(`1800 * (0.000164 + 2*0.0000131 + 4*0.00000222)`). Image build/startup and other
account usage are outside that function-only estimate. {usage_note}
The timeout is not a hard account spending cap. No further paid experiments are
started automatically.

Offline CPU prediction is provided locally; iOS model conversion, app integration,
quantization, energy use, and actual per-query costs are not measured. On-device
inference could avoid per-query cloud fees after integration, but that is a proposed
deployment property, not a demonstrated iOS result.

## Sources and next experiments

- [MobileNetV3 paper](https://arxiv.org/abs/1905.02244): mobile-oriented architecture rationale.
- [Hugging Face model card](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k): pretrained model and provenance.
- [Modal pricing](https://modal.com/pricing): cost estimate inputs, checked November 3, 2026.
- Project requirements: all pages of the three PDFs in `docs/` were reviewed.

Next: recover the 208 clean originals, collect independent capture sessions, inspect
full-resolution errors, evaluate a higher input resolution if small targets are
missed, and benchmark a converted model on a named iPhone. Instance detection, OCR,
direction association and speech integration remain future components. Any next
experiment needs a separate documented configuration and untouched evaluation data.
"""
    (output / "baseline_results.md").write_text(report)

    slides = [
        ("3.1", "A small model answers three presence questions", "",
         "Camera image → padded 224×224 input → shared MobileNetV3-Small → three binary heads.",
         "Our first baseline answers three questions about each camera frame: is there a gate sign, a restaurant sign, or a washroom sign? One shared image encoder feeds three independent binary heads. Several answers can be positive together. This stage identifies sign categories; later components will read their content."),
        ("3.2", "Choose a mobile-oriented starting point", "",
         f"{result['parameters']/1e6:.2f}M parameters after replacing the ImageNet classifier.",
         f"We chose MobileNetV3-Small because its architecture was designed around mobile CPU efficiency. ImageNet pretrained weights give us an initial visual representation without training from scratch. Our three-head version has {result['parameters']/1e6:.2f} million parameters. This is a practical starting point for iOS evaluation, although phone performance has not yet been measured."),
        ("3.3", "Remove annotation shortcuts", "dataset_balance",
         "1,692 unmarked frames retained; 208 marked-only frames excluded.",
         "The positive images contain red annotation boxes, so feeding those pixels into training could reveal the answer. We recovered unmarked copies from other class folders and merged labels by source frame. We excluded 208 marked-only frames. That removes every triple-positive example and over half the restaurant-positive frames, limiting the evaluation."),
        ("3.4", "Keep related frames together", "dataset_balance",
         "13 training / 3 validation / 3 test videos; seed 42.",
         "We split by video rather than by image, keeping adjacent frames and duplicate sign instances together. Thirteen videos train the model, three select its checkpoint, and three provide the final test. Each partition contains every class. Different videos may still show the same physical signs, so this does not test unseen-airport generalization."),
        ("3.5", "Start with transfer learning, then fine-tune", "learning_curves",
         f"3 frozen epochs + 7 fine-tuning epochs; selected epoch {result['best_epoch']}.",
         "For three epochs, we trained only the new heads. We then fine-tuned all layers for seven epochs at a lower learning rate. Training-count-weighted binary cross-entropy addresses class imbalance. Validation macro F1 selects the checkpoint. Only this architecture was attempted; the learning curves document its behavior without establishing superiority over alternatives."),
        ("4.1", "Measure each class separately", "class_metrics",
         f"Test macro F1: {test['macro_f1']:.3f} on {test['frames']} frames.",
         f"On the held-out test videos, macro F1 is {test['macro_f1']:.2f}. We also report precision, recall, and accuracy separately for each sign category. {best_class.title()} has the highest F1 and {worst_class} the lowest. These are whole-image presence results. They do not yet evaluate reading text, assigning arrows, or speaking a correct response."),
        ("4.2", "Compare with constant predictions", "baseline_comparison",
         f"Model label accuracy: {pct(test['label_accuracy'])}; exact match: {pct(test['exact_match_accuracy'])}.",
         "Our simple comparisons always predict no sign, or predict each class's training-set majority label. These baselines show why raw accuracy can look high when positive signs are uncommon. We report macro F1 and exact match alongside accuracy. Exact match requires all three predictions to be correct for the same frame."),
        ("4.3", "Inspect false alarms and missed signs", "confusion_matrices",
         "Use the test-example gallery as a companion visual.",
         f"The restaurant head has {tp} true positives, {fn} missed positives, and {fp} false alarms at our fixed threshold. Its F1 is {restaurant['f1']:.2f}, while accuracy is {pct(restaurant['accuracy'])} because many labels are negative. This result needs improvement. We keep the threshold unchanged and will investigate with validation data and new independent captures."),
        ("4.4", "Establish server latency measurements", "server_latency",
         f"T4 mean: {gpu['mean_ms']:.2f} ms; server CPU mean: {cpu['mean_ms']:.2f} ms.",
         f"A warm model call averages {gpu['mean_ms']:.1f} milliseconds on the Modal T4 and {cpu['mean_ms']:.1f} milliseconds on a single server CPU thread. These timings include the model and sigmoid only. They exclude camera capture, image preparation, and speech. We have no iPhone measurement yet, so these are reference timings, not mobile latency claims."),
        ("4.5", "Connect the baseline to the final objectives", "",
         "Next: clean originals, independent evaluation, and a named iPhone benchmark.",
         "This baseline establishes measured sign-presence accuracy and model execution time, supporting two parts of our proposal. It does not establish full task accuracy or request-to-speech latency. Next we need clean originals, independent capture sessions, and an iPhone benchmark. Localization, text reading, direction association, and speech remain necessary for the complete assistant."),
    ]
    lines = ["# Progress on Models and Preliminary Results: ten slides", "",
             "Each slide auto-advances after 20 seconds: 3 minutes 20 seconds total.",
             "Scripts are approximately 40–60 words; rehearse and trim for your speaking pace.",
             "Use one headline and one visual on each slide. Technical backup: [measured report](baseline_results.md).", ""]
    for number, title, figure, headline, script in slides:
        lines += [f"## Slide {number}: {title}", "", f"**On slide:** {headline}", ""]
        if figure:
            lines += [f"![{title}](figures/{figure}.png)", ""]
        elif number == "3.1":
            lines += ["```mermaid", "flowchart LR", "  A[Camera frame] --> B[224 x 224 padded RGB]",
                      "  B --> C[Shared MobileNetV3-Small]", "  C --> D[Gate sigmoid]",
                      "  C --> E[Restaurant sigmoid]", "  C --> F[Washroom sigmoid]", "```", ""]
        lines += [f"**20-second script:** {script}", ""]
    lines += ["## Backup and attribution", "", "- [Measured results, settings, limitations, and sources](baseline_results.md)",
              "- [Test examples](figures/test_examples.png): deterministic illustrative successes/errors.",
              "- Model source: [Hugging Face](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k).",
              "- Architecture source: [MobileNetV3 paper](https://arxiv.org/abs/1905.02244).",
              "- All charts are generated from this experiment. No iOS measurements or numerical pass targets exist yet.", ""]
    (output / "progress_slides.md").write_text("\n".join(lines))
    print(f"Saved measured report, ten slide scripts, and PNG/SVG figures to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/baseline"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    generate(args.data_dir, args.run_dir, args.output)
