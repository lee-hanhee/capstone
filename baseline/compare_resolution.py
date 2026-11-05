"""Write a small four-case resolution/threshold report from saved experiments."""

import csv
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline.data import CLASSES


def generate():
    output = Path("reports/resolution_thresholds")
    output.mkdir(parents=True, exist_ok=True)
    roots = [Path("artifacts/v3_224_thresholds"), Path("artifacts/v3_320")]
    runs = [json.loads((root / "metrics.json").read_text()) for root in roots]
    configs = [json.loads((root / "config.json").read_text()) for root in roots]
    assert configs[0]["counts"] == configs[1]["counts"]
    assert configs[0]["annotation_sha256"] == configs[1]["annotation_sha256"]
    test_keys = []
    for root in roots:
        with (root / "test_predictions.csv").open() as f:
            test_keys.append([(r["source_video"], r["frame_index"], *[r[f"true_{c}"] for c in CLASSES])
                              for r in csv.DictReader(f)])
    assert test_keys[0] == test_keys[1], "Test frames/labels differ"
    cases = [(f"{size}px / {policy}", r[key]) for size, r in zip([224, 320], runs)
             for policy, key in [("0.5", "test"), ("selected", "test_selected_thresholds")]]
    summary = "| Input / threshold | Gate F1 | Restaurant F1 | Washroom F1 | Macro F1 | Exact match |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    for name, m in cases:
        summary += f"| {name} | " + " | ".join(f"{m['per_class'][c]['f1']:.3f}" for c in CLASSES)
        summary += f" | {m['macro_f1']:.3f} | {m['exact_match_accuracy']*100:.1f}% |\n"
    detail = "| Input / threshold / class | Precision | Recall | F1 |\n| --- | ---: | ---: | ---: |\n"
    for name, m in cases:
        for c in CLASSES:
            p = m["per_class"][c]
            detail += f"| {name} / {c} | " + " | ".join(f"{p[k]:.3f}" for k in ["precision", "recall", "f1"]) + " |\n"
    threshold_table = "| Input | Gate threshold | Restaurant threshold | Washroom threshold |\n| --- | ---: | ---: | ---: |\n"
    for size, r in zip([224, 320], runs):
        threshold_table += f"| {size}px | " + " | ".join(f"{r['selected_thresholds'][c]:.6f}" for c in CLASSES) + " |\n"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(4)
    for i, (name, m) in enumerate(cases):
        values = [m["per_class"][c]["f1"] for c in CLASSES] + [m["macro_f1"]]
        bars = ax.bar(x+(i-1.5)*0.2, values, width=0.2, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=["Gate", "Restaurant", "Washroom", "Macro F1"],
           ylim=(0, 1.2), ylabel="Test F1", title="Same 234 test frames; thresholds selected on validation only")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for name, root in zip(["v3_224_thresholds", "v3_320"], roots):
        dest = output / name
        dest.mkdir(exist_ok=True)
        for file in ["config.yaml", "config.json", "metrics.json", "thresholds.json",
                     "validation_predictions.csv", "test_predictions.csv"]:
            shutil.copy2(root / file, dest / file)
        if (root / "history.json").exists():
            shutil.copy2(root / "history.json", dest / "history.json")
    latency = "| Input | T4 mean | Server CPU mean (1 thread) |\n| --- | ---: | ---: |\n"
    for size, r in zip([224, 320], runs):
        latency += f"| {size}px | {r['latency_training_device']['mean_ms']:.2f} ms | {r['latency_server_cpu']['mean_ms']:.2f} ms |\n"
    direction = "improved" if runs[1]["test"]["macro_f1"] > runs[0]["test"]["macro_f1"] else "did not improve"
    outcome = (f"Increasing resolution {direction} fixed-threshold test macro F1 in this run "
               f"({runs[0]['test']['macro_f1']:.3f} at 224px versus {runs[1]['test']['macro_f1']:.3f} at 320px). "
               f"Validation-selected thresholds give {runs[0]['test_selected_thresholds']['macro_f1']:.3f} "
               f"and {runs[1]['test_selected_thresholds']['macro_f1']:.3f}, respectively. "
               "This is a single-seed outcome, not evidence that higher resolution can never help.")
    text = f"""# Higher resolution and validation-selected thresholds

Historical backup. The current presentation uses the [clean-image 224px model](../clean_224/results.md).
Its dataset and split differ, so scores are not a controlled comparison.

Presentation: [ten claim-and-evidence slides](../progress_slides.md), with local
SVGs under `artifacts/svg/` and matching PNGs under `artifacts/png/`.
Regenerate with `.venv/bin/python scripts/build_progress_figures.py`.

Two checkpoints, four evaluation cases. The 224px checkpoint is the original
baseline (epoch {runs[0]['best_epoch']}); it was not retrained. The 320px model was
trained for ten epochs and selected epoch {runs[1]['best_epoch']} by validation
macro F1 at fixed 0.5 thresholds. Historical baseline reports remain unchanged.

**Outcome:** {outcome}

{summary}
![Comparison](comparison.png)

## Validation-only threshold selection

{threshold_table}
Restaurant test F1 is {runs[0]['test_selected_thresholds']['per_class']['restaurant']['f1']:.3f}
at 224px and {runs[1]['test_selected_thresholds']['per_class']['restaurant']['f1']:.3f}
at 320px with selected thresholds. Resolution and threshold changes have not
resolved restaurant recognition on these held-out videos.

Each threshold maximizes its class's F1 over the unique validation scores of the
selected checkpoint. Equal maxima prefer the threshold nearest 0.5, then the larger
threshold. The rule is `score >= threshold`. Checkpoint selection happens first at
fixed 0.5; thresholds are then selected using validation predictions only, before
test inference. Thresholds are not learned from test labels or changed after test
inspection. The validation F1 after threshold fitting is an optimization result,
not an independent performance estimate.

Validation and test predictions include frame identity, all three labels, and all
three sigmoid scores. See the CSVs in [224px results](v3_224_thresholds/) and
[320px results](v3_320/). The prediction command reads selected thresholds and input
size directly from each new checkpoint. Scores from weighted BCE are not calibrated
probabilities; a selected threshold need not be near 0.5.

## Per-class precision and recall

{detail}
## Controlled settings

Architecture: pretrained MobileNetV3-Small with three binary heads. Same pinned
Hugging Face revision, seed 42, batch 32, AdamW, weight decay 0.0001, no augmentation,
ImageNet normalization, and training-only positive class weights. Three frozen
epochs at head LR 0.001, seven all-layer epochs at LR 0.0001. The new training run
changes image resolution only; both threshold policies are evaluated from the same
checkpoint per resolution.

320px inputs were generated directly from original unmarked PNGs using the same
bilinear aspect-ratio-preserving padding, rather than enlarging the 224px files.
Frame identities, labels, boxes, exclusions and video splits were checked for exact
agreement: 1,209 train / 249 validation / 234 test, grouped into 13/3/3 videos.
The 208 marked-only frames remain excluded, removing all triple-positive examples
and 54.5% of original restaurant-positive frames. This experiment does not fix that
selection bias. Only three correlated test videos and one seed are evaluated.

Experiment YAMLs are tracked under `experiments/`; the original run's YAML is a
historical transcription of its saved config.json, not a claim it originally ran
from YAML. New Modal calls read their YAML settings and save a copy beside outputs.
Runtime normalization, class weights, versions and selected thresholds are saved in
config.json; thresholds.json describes the selection rule. Large image/annotation
manifests and weights stay under gitignored data/ and artifacts/.

## Latency

{latency}
The 224px timings are reused from the original run; threshold-only evaluation did
not benchmark a new model. The 320px timings were measured in the new T4 run.
Both use warm batch-one forward pass plus sigmoid, 20 warmups and 100 repeats.
CPU host hardware may differ. Image decode, preprocessing, camera, speech, network,
cold start and threshold application are excluded. iOS latency remains unmeasured.

These comparisons establish descriptive resolution and threshold effects on the
existing test set, which has already been inspected. Continue tuning on validation
data and reserve new independent capture sessions for a final assessment.

## Reproduce

```sh
.venv/bin/python -m baseline.data --size 320 --output data/v3_320
.venv/bin/modal run modal_train.py --config experiments/v3_224_thresholds/config.yaml
.venv/bin/modal run modal_train.py --config experiments/v3_320/config.yaml
.venv/bin/python -m baseline.compare_resolution
.venv/bin/python -m baseline.predict path/to/image.png --checkpoint artifacts/v3_320/best.pt
```

Existing run directories are protected against overwrite. Actual Modal usage is
recorded in modal_usage.json after the runs; the overall authorized budget is $10.
No additional training or parameter search is started automatically.
"""
    (output / "comparison.md").write_text(text)
    print(f"Saved four-case comparison, configs, scores and figure to {output}")


if __name__ == "__main__":
    generate()
