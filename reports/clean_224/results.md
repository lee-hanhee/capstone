# Clean-image 224px model: measured results

This is the current presentation model: ImageNet-pretrained MobileNetV3-Small,
three binary heads, **epoch 2 selected by validation macro F1 at 0.5**.
All ten epochs were completed; full fine-tuning did not beat the earlier frozen-
backbone checkpoint. This is the best checkpoint under the declared validation
rule, not a claim of the best possible architecture.

## Recognition evidence

The configured operating policy uses thresholds selected by per-class validation
F1. On 200 test frames from two held-out videos:

| Class | Precision | Recall | F1 | Missed positives | False alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate | 55.6% | 33.3% | 0.417 | 60/90 | 24/110 |
| restaurant | 67.0% | 97.6% | 0.795 | 3/123 | 59/77 |
| washroom | 67.6% | 57.7% | 0.623 | 52/123 | 34/77 |

Macro F1: **0.611**; binary label accuracy: **61.3%**;
exact match (all three labels correct): **29.5%**.
Restaurant recall is now nonzero, with 120/123 positive frames retrieved, but
59 false alarms remain. These are presence decisions, not readable gate numbers,
OCR accuracy, direction association, or correct spoken task responses.

## Threshold tradeoff: report both policies honestly

| Same checkpoint | Validation macro F1 | Test macro F1 | Test exact match |
| --- | ---: | ---: | ---: |
| Fixed 0.5 | 0.736 | 0.682 | 40.5% |
| Validation-selected | 0.819 | 0.611 | 29.5% |

Threshold fitting improves the validation optimization score but harms test macro
F1 here. We retain the predeclared validation-selected policy in the primary
results and checkpoint, and show 0.5 as the reference. We do not choose a new
policy by maximizing this test result. Thresholds (gate / restaurant / washroom):
[0.6126263737678528, 0.37709876894950867, 0.6598606705665588]. Scores from weighted BCE are not calibrated probabilities.
The validation score after fitting thresholds is not an independent estimate.

## Descriptive constant-prediction checks

| Rule on this test set | Macro F1 | Exact match |
| --- | ---: | ---: |
| Always absent / training-majority rule | 0.000 | 35.5% |
| Always present | 0.715 | 42.0% |
| Selected checkpoint, fixed 0.5 | 0.682 | 40.5% |
| Selected checkpoint, fitted thresholds | 0.611 | 29.5% |

The always-present comparator highlights the label-rich test distribution: many
frames contain multiple signs. It outperforms this checkpoint on macro F1 and
exact match here, so the model is not yet a convincing improvement over all trivial
rules. Inspect precision, recall, false alarms and misses together. This additional
descriptive reference was computed after the run and did not select weights or
thresholds. See [reference metrics](always_present_reference.json).

## Data and label provenance

The provided `unannotated/manifest.json` describes 4,551 PNG copies representing
**1,200 unique source frames from 12 videos**. All **208 previously excluded
triple-positive frames** are restored; no frames in this clean cohort are excluded.
This clean export contains 12 of the earlier 19 videos, not the complete old cohort.
It contains no ground-truth annotation JSONs, and the old annotated/ directory is
absent. Labels and the 2,406 retained positive-instance boxes were recovered from
`data/baseline/manifest.json`, which preserves the earlier ground truth.

Every export presence label and positive-instance filename was checked against
that retained ground truth. All 4,551 paths exist, same-frame export copies have
matching recorded checksums, and each of the 1,200 selected training/evaluation
inputs was checked against its SHA-256 and recorded dimensions before resizing.
These checks verify consistency with the export and saved labels, not an independent
re-annotation or a fresh video-to-pixel audit. The source manifest's audit statements
remain provider-supplied provenance.

The new seed-42 **8/2/2 video split** gives 800 train / 200 validation / 200 test
frames. Class-coverage constraints are chosen using labels only before training.
Test videos: IMG_8456.MOV and IMG_8463.MOV. No source video crosses partitions.
The restored frames are existing recordings, not newly collected independent
captures. Different clips may show the same signs. Only two test videos and one
seed are evaluated, so broad generalization is unproven.

The dataset, class prevalence and split changed. Do not interpret scores against
the old 19-video runs as a controlled improvement attributable solely to clean
pixels or recovered restaurant examples. Historical results remain in
[the original report](../baseline_results.md) and [the resolution comparison](../resolution_thresholds/comparison.md).

## Confirmed settings

224x224 RGB, bilinear aspect-ratio-preserving padding, ImageNet normalization,
no augmentation; seed 42, batch 32, AdamW, weight decay 0.0001. Three frozen
backbone epochs at head LR 0.001; seven all-layer epochs at LR 0.0001. Training-only
class-count weighting for BCE, fixed-0.5 validation checkpoint selection, then
per-class validation-F1 threshold selection. Same pinned pretrained revision as
before. [Requested YAML](config.yaml) and [resolved runtime config](config.json).

## Project requirements and constraints

| Requirement / constraint | What this run supports | Remaining evidence / solution |
| --- | --- | --- |
| Maximize correct sign readings per task | Per-class presence precision/recall; quantified misses and false alarms | New independent capture sessions; localization, OCR and destination-arrow evaluation |
| Minimize request-to-response latency | Warm model-only server timing | Core ML on a named iPhone, including decode, preprocessing and speech |
| Smartphone, no extra hardware | 1,520,931 parameters; 5.93 MiB checkpoint | Actual iOS runtime, memory and energy measurements |
| Variable connectivity; offline preferred | Download-free CPU prediction from saved weights | Test the integrated iOS pipeline offline |
| Minimize recurring query cost | On-device inference is the intended approach | Actual per-query processing cost is not measured; training cost is a different metric |

Mean model latency: **4.77 ms T4 GPU** and
**7.28 ms single-thread server CPU**. Batch one,
20 warmups, 100 repeats; forward pass and sigmoid only. Camera, decode,
preprocessing, network, thresholding, speech and cold start are excluded.
**iOS latency remains unmeasured**; no numerical accuracy/latency acceptance
threshold has been defined. No product requirement is claimed as fully met.

Reported Modal usage for this run is **$0.00915937**, within the existing $10
budget; this is reported usage at check time, not a final invoice. See
[usage record](modal_usage.json). This is training/evaluation spend, not a
measurement of recurring per-query processing cost.

## Reproduce and presentation

```sh
.venv/bin/python -m baseline.prepare_clean --config experiments/v3_224_clean/config.yaml
.venv/bin/modal run modal_train.py --config experiments/v3_224_clean/config.yaml
.venv/bin/python scripts/build_progress_figures.py
.venv/bin/python -m baseline.predict path/to/image.png --checkpoint artifacts/v3_224_clean/best.pt
```

Prepared images/manifests and model checkpoints stay in gitignored data/ and
artifacts/. The original checkpoint and all previous results are preserved.
[Validation scores](validation_predictions.csv), [test scores](test_predictions.csv),
[thresholds](thresholds.json), [history](history.json), [metrics](metrics.json).

Use [the ten-slide plan](../progress_slides.md) and [rubric alignment](../rubric_alignment.md).
SVGs contain figures only; titles, caveats, requirement links and scripts are in
Markdown. The main story covers this selected checkpoint, not the old experiment
timeline. Real test examples replace the threshold-policy chart in the main slides;
the full threshold comparison and limitations remain in this technical report.
The final figure shows the short path from recognition to spoken guidance.
