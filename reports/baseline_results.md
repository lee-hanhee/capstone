# Measured baseline results

One confirmed experiment; no hyperparameter search or test-driven checkpoint selection.
Best validation checkpoint: epoch 9 of 10.
Validation macro F1: 0.741; test macro F1: 0.510.

## Held-out test performance

| Class | Positive / negative | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate | 47 / 187 | 90.6% | 80.5% | 70.2% | 75.0% |
| restaurant | 22 / 212 | 88.5% | 0.0% | 0.0% | 0.0% |
| washroom | 95 / 139 | 85.0% | 96.9% | 65.3% | 78.0% |

| Method | Label accuracy | Exact match | Macro F1 |
| --- | ---: | ---: | ---: |
| Always absent | 76.6% | 44.4% | 0.000 |
| Training majority | 70.4% | 26.1% | 0.193 |
| MobileNetV3 | 88.0% | 73.9% | 0.510 |

Label accuracy averages the three binary decisions per image. Exact match requires
all three decisions to be correct. Macro F1 gives equal weight to the three sign
classes. These are presence metrics; they do not measure text reading, arrow
association, sign localization, or correct spoken responses.

No-relevant-sign inference is the conjunction of all three absent predictions.
Its test precision is 73.4%, recall 90.4%,
F1 0.810, and accuracy 81.2%.

## Architecture and method

ImageNet-pretrained MobileNetV3-Small from Hugging Face, one shared feature extractor,
three independent linear binary heads, sigmoid outputs, fixed 0.5 thresholds.
Measured model parameters: 1,520,931; checkpoint: 5.93 MiB.
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
washroom has the highest class F1 and restaurant the lowest. See
[confusion matrices](figures/confusion_matrices.png) for the actual false positives
and false negatives and [test examples](figures/test_examples.png) for deterministic
illustrations. The examples are selected for illustration and do not estimate error
frequencies. Do not attribute errors to small signs or blur without reviewing
the original full-resolution frames; those are hypotheses for a later analysis.

The restaurant head produced 0 true positives, 22 false negatives, and 5 false positives at the confirmed 0.5 threshold. Its precision, recall, and F1 are 0.000, 0.000, and 0.000. Its accuracy is 88.5%, versus 90.6% for always absent, illustrating why accuracy alone can mislead. The threshold has not been changed after viewing test results. Restaurant ROC AUC is 0.820; ranking quality does not establish useful fixed-threshold predictions. Future threshold selection must use validation data and a fresh independent evaluation. This baseline failed to retrieve any restaurant-positive test frame.

Validation macro F1 (0.741) versus test macro F1
(0.510) shows a gap. Training loss falls
while validation loss rises after the frozen phase, consistent with overfitting
or distribution differences. These diagnostics motivate broader independent data;
they do not establish a single cause or a measured fix.

The first unconstrained split lacked restaurant-positive test frames; class coverage
checks resolved that evaluation problem. Training on drawn boxes was rejected
before model fitting because of label leakage risk. Neither issue is an unsuccessful
architecture experiment. Only MobileNetV3-Small was trained; no claim that it beats
other neural architectures or that fine-tuning caused an improvement is supported.

## Latency, cost, and proposal objectives

T4 mean model latency: 4.46 ms, median 4.45 ms,
p95 4.49 ms. Modal server CPU (one thread): mean 6.93 ms,
median 6.82 ms, p95 7.74 ms. Measurements use batch one,
20 warmups, 100 repeats, and synchronized GPU timing; forward pass plus sigmoid
only. CPU model is float32. Excludes decode, preprocessing, camera, network,
speech and cold start. The saved device and platform information is in config.json.

No iPhone is selected, so **iOS latency is unmeasured**. No accuracy or latency pass
threshold is defined. These measurements establish a starting point rather than
showing that a product requirement has been met. Presence evaluation is one component
of the proposal's percentage of correct readings per task. Model latency is one
component of its request-to-response latency objective.

Training wall time inside the function: 50.4 seconds. The
single T4 function has a 1,800-second timeout, two CPU cores, 4 GiB RAM, no retries,
and no persistent volume or deployed endpoint. At the listed rates checked for
this experiment, that function's maximum compute estimate is about $0.36
(`1800 * (0.000164 + 2*0.0000131 + 4*0.00000222)`). Image build/startup and other
account usage are outside that function-only estimate. Modal's usage report currently attributes $0.01442504 to this app, within the $10 budget. This is reported usage at check time, not a final invoice; costs may lag or be attributed separately to image builds. See [usage record](modal_usage.json).
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
