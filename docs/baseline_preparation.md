# Baseline preparation and confirmed decisions

The baseline has been implemented with the user's confirmed settings. This file
records the preparation findings and design rationale. See
[measured results](../reports/baseline_results.md) and
[ten-slide notes](../reports/progress_slides.md) for experiment outcomes.

## Project requirements reviewed

All pages of the three project PDFs were read:

- `MIE429_Guidebook_2026.pdf`: progress presentations must cover model progress,
  successes and failures, preliminary results, obstacles, and future plans. Slides
  auto-advance every 20 seconds. The complete presentation lasts 10 minutes.
- `MIE429_Rubrics_Proposal.pdf`: justify the chosen methods against client needs,
  evaluate alternatives, and describe whether the data is sufficient.
- `See2Hear Project Proposal.pdf`: recognize airport washroom, food/drink, and
  gate signs, eventually read text and associate directions, and speak relevant
  information. Objectives are recognition accuracy, request-to-response latency,
  recurring cost, and operation with variable connectivity (offline preferred).
  The system must operate on a smartphone without extra hardware.

The proposal supplies no numerical accuracy or latency target and no phone model.
Sign presence accuracy alone does not measure correct text/direction readings per
task. Model forward-pass time alone does not measure request-to-speech latency.

## Reproducible dataset audit

Run from the repository root, using Python's standard library:

```sh
python scripts/audit_dataset.py
```

The output is `data/dataset_audit.json` (gitignored). It includes input annotation hashes,
counts, unmarked image candidates, and frames needing clean sources. It checks
referenced image existence, class/filename consistency, Boolean labels, valid
boxes, cross-class frame coverage, consistent geometry, and label agreement
between copies. It does not verify every image's pixel contents or decode every
file.

Only the requested `ground_truth_bulk_v1.json` files are audited. Legacy annotation
batches are separate and have different schemas.

| Class | Positive instance images | Negative images | Unique positive frames |
| --- | ---: | ---: | ---: |
| gate | 906 | 1,420 | 480 |
| restaurant | 393 | 1,518 | 382 |
| washroom | 2,425 | 868 | 1,032 |

The 7,530 records represent 1,900 source frames from 19 videos, with 100 sampled
frames per video. There are 584 frames negative for all three classes.

The manifest says positive images contain exactly one red annotation box and
negative images retain their original pixels. Visual inspection of one positive
example per class confirms visible red boxes. Training directly on these marked
PNGs risks learning the annotation as a shortcut.

For 1,692 frames, at least one other class folder provides a negative, unmarked
copy of the same source frame. Such a copy remains positive for whichever other
classes are present: a negative restaurant annotation is not a global negative.
Source identity and geometry match in the metadata; original video pixel identity
has not been verified.

For 208 frames, all three labels are positive, so every available copy is marked.
The user approved excluding these frames for this baseline. This removes every
triple-positive example and 208 of 382 restaurant-positive source frames (54.5%).
That materially changes the data distribution. Clean originals are preferable;
no image repair is applied.

Multiple same-class signs have individual annotations and separate marked copies.
Their bounding boxes must be retained as individual instances. For presence
classification, copies of the same source frame should be consolidated into one
three-label observation. Classification does not locate or count signs.

Confirmed evaluation: group the train/validation/test split by source video so
neighboring frames and duplicate instances cannot cross partitions. Different
videos may still show the same physical signs, so a video split would not establish
generalization to unseen airports or signs. Seed 42, 13/3/3 videos, with class
coverage checks, yields 1,209 / 249 / 234 frames. The first seeded assignment
lacked restaurant-positive test examples; the second meets class coverage.

## Confirmed mobile baseline architecture

Selected: ImageNet-pretrained `timm/mobilenetv3_small_100.lamb_in1k` from Hugging
Face. Its model card reports approximately 2.5 million parameters, 0.1 GMACs,
and 224 x 224 input. These are pretrained model statistics, not measured baseline
results. Replacing its original classifier with three binary heads produces
1,520,931 parameters and a 5.93 MiB checkpoint in this experiment.

MobileNetV3 was designed using mobile CPU latency considerations, and the Small
variant targets lower resource use. This makes it a reasonable baseline candidate
for the proposal's smartphone, offline, and recurring-cost objectives. Its sign
recognition accuracy and latency on the target phone are unmeasured. Small and
partially visible signs may lose essential details at low input resolution.

The implemented shared backbone computes features once for three independent
binary heads. Three separate models would repeat feature computation when
answering all three questions; they have not been trained for comparison.
Independent outputs permit co-occurring labels; all three negative outputs
represent no relevant sign. The user confirmed whole-image presence for this
baseline, with every instance box retained for future detection.

Sources checked November 3, 2026:

- [Hugging Face model card](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Modal persistent volumes](https://modal.com/docs/guide/volumes)

## User-confirmed scope and hyperparameters

1. One shared backbone with three binary heads; keep the pipeline simple.
2. Whole-image presence; retain duplicate-instance boxes for later detection.
3. Exclude the 208 marked-only frames and document the bias.
4. $10 Modal budget; iOS intended, no selected phone, no numerical pass targets.
5. Markdown notes and figures for ten slides, each with a 20-second script.
6. Ten epochs: 3 frozen at head learning rate 0.001, 7 all-layer fine-tuning at
   0.0001; batch 32; AdamW with weight decay 0.0001; seed 42.
7. 224x224 RGB padding; ImageNet normalization; no augmentation.
8. Training-count-weighted BCE; fixed 0.5 thresholds; select highest validation
   macro F1, retaining the earliest checkpoint on ties.

The user authenticated Modal locally. One T4 training run completed all ten epochs
and selected epoch 9. It uses no persistent volume or deployed inference endpoint.
The checkpoint, configuration, predictions, training history and metrics were
downloaded to `artifacts/baseline/`. iOS execution has not been implemented or timed.

The ten-slide material distinguishes measured results from hypotheses, reports
class-wise precision/recall/F1 alongside accuracy, compares constant-prediction
baselines, and documents the restaurant head's zero recall on test at the confirmed
threshold. Presence accuracy and server latency are components of the final
recognition and request-to-response objectives, which remain broader tasks.
