# See2Hear: compact airport sign classifier

One ImageNet-pretrained MobileNetV3-Small encoder with three independent binary
heads: **gate, restaurant, washroom**. Multiple classes may be present; all three
absent predictions mean no relevant sign. This component classifies whole-image
presence. Localization, OCR, destination-arrow association and speech remain future work.

## Current results and presentation

- [Selected clean-input 224px model: results and limitations](reports/clean_224/results.md)
- [Ten slides: claims, figures and 20-second scripts](reports/progress_slides.md)
- [Good + Outstanding rubric alignment](reports/rubric_alignment.md)
- [Preparation and label provenance](docs/baseline_preparation.md)
- [SVG figures](reports/svg/) · [PNG figures](reports/png/) · [contact sheet](reports/progress_contact_sheet.png) — tracked presentation outputs

The clean-input run completed ten epochs; **validation selected epoch 2**, before
full fine-tuning. Configured validation-selected thresholds give test macro F1
**0.611**; the same checkpoint at fixed 0.5 gives **0.682**. Threshold fitting helps
validation but harms test here. Both are reported; no rule is chosen retrospectively
by maximizing test performance. Restaurant recall is 120/123 (97.6%), with 59 false
alarms. Gate and washroom misses remain substantial.

Model: **1.52M parameters, 5.93 MiB checkpoint**. Mean warm forward time: **4.77 ms
on T4**, **7.28 ms on one server CPU thread**. These exclude preprocessing, camera,
network and speech. **iOS latency and complete task performance remain unmeasured**;
no numerical accuracy or latency acceptance targets are defined.

## Setup and reproduce this run

Python 3.13 was used locally and on Modal:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/modal setup
.venv/bin/python -m baseline.prepare_clean --config experiments/v3_224_clean/config.yaml
.venv/bin/modal run modal_train.py --config experiments/v3_224_clean/config.yaml
```

Inputs: `unannotated/manifest.json` plus its PNGs. The export contains **4,551 copies
of 1,200 unique source frames from 12 videos**. All **208 previously excluded frames
are included**, with zero exclusions inside this clean cohort. This is a subset of
the earlier 19-video cohort; it does not represent newly collected independent sessions.

The export has no label JSONs and the annotated/ directory is absent. Preparation
uses retained ground-truth labels and boxes in `data/baseline/manifest.json`, checks
all encoded export labels and positive-instance filenames, and verifies the SHA-256
and dimensions of each selected source image. Preserve that earlier manifest to
reproduce this run. It contains the original boxes, including duplicate signs.

The user-confirmed seed-42 **8/2/2 video split** yields **800 train / 200 validation /
200 test frames**, with class coverage checked before training. Neighboring frames
stay in the same partition. Only two test videos and one seed are evaluated; distinct
clips may show the same signs. Historical results use different data/splits and
cannot establish a controlled improvement over this run.

## Experiment configuration

[experiments/v3_224_clean/config.yaml](experiments/v3_224_clean/config.yaml) drives
the current run. It retains the confirmed settings: 224px RGB bilinear padding,
ImageNet normalization, no augmentation, batch 32, AdamW, weight decay 0.0001,
training-count-weighted BCE, three frozen epochs at head LR 0.001 followed by seven
all-layer epochs at LR 0.0001. Pretrained weight revision and seed are pinned.

Select the checkpoint by validation macro F1 at 0.5. Then fit one threshold per
class by validation F1; ties favor the threshold nearest 0.5, then the larger one.
Validation and test scores are saved separately. The resulting checkpoint stores
its input size and selected thresholds. Weighted-BCE scores are not calibrated
confidence estimates.

```sh
.venv/bin/python -m baseline.predict path/to/image.png --checkpoint artifacts/v3_224_clean/best.pt
```

Prediction runs locally on CPU without downloading weights. Prepared images and
large manifests are under gitignored `data/`; checkpoints are under gitignored
`artifacts/`. Small configs, metrics and validation/test CSVs are in `reports/clean_224/`.
Existing completed outputs are protected against overwrite.

## Presentation figures

```sh
.venv/bin/python scripts/setup_lato.py
.venv/bin/python scripts/build_progress_figures.py
```

Use **figure-only SVGs** from `reports/svg/` and copy each title from the Markdown
slide plan into Google Slides. The SVGs have **no slide title, number or footer**.
Diagram labels, axes and legends remain because they explain the evidence.
SVGs outline actual Lato glyphs so their appearance is portable. Matching PNGs are supplied.
The exact palette is navy `#1B2C4A`, orange `#F07A2A`, teal `#2F7F81`, slate `#3B4A66`,
and white `#FFFFFF`. Lato files and license are cached in `artifacts/fonts/`.

The main ten-slide story presents this one selected checkpoint, with requirement
links and concrete next tests. It does not retell the old resolution-experiment
timeline. Detailed threshold comparisons remain in the technical results. Ten slides at 20 seconds are
3:20; the entire team presentation must still meet the ten-minute requirement.

## Historical results and budget

Original results remain unchanged in [the 224px baseline report](reports/baseline_results.md)
and [the resolution/threshold comparison](reports/resolution_thresholds/comparison.md).
The old YAMLs, measured reports and checkpoints remain for experiment tracking.
Superseded presentation notes, archived SVGs and duplicate exports have been removed;
the presentation folders contain only the current ten SVGs, their PNG equivalents
and a contact sheet.

The authorized overall Modal budget remains **$10**. Each job uses one T4, two CPU
cores, 4 GiB RAM, a 30-minute timeout, a 25-minute training timer, no automatic
retries and no persistent inference deployment. Actual usage is recorded separately;
training cost is not recurring per-query cost, and a timeout is not an account-wide
dollar cap. No additional experiment starts automatically.
