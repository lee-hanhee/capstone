# See2Hear: airport sign presence baseline

One ImageNet-pretrained [MobileNetV3-Small](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
backbone with three independent binary heads: **gate, restaurant, washroom**.
Multiple labels can be present. If all outputs are below 0.5, predict no relevant
sign. This baseline classifies whole images; instance detection, OCR, arrow
association, and speech are future work.

## Results and progress presentation

- [Ten slides with 20-second speaking notes](reports/progress_slides.md)
- [Measured results, architecture rationale, settings, and limitations](reports/baseline_results.md)
- [PNG and editable SVG figures](reports/figures/)
- [Dataset audit and project requirements](docs/baseline_preparation.md)

The model is a candidate for later iOS deployment. Server timing is reported
separately; no phone is selected and iOS latency is unmeasured. Numerical pass
targets have not been set.

## Setup

Python 3.13 was used locally and on Modal. Install the pinned dependencies:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/modal setup
```

Place the extracted dataset at `annotated/` with these label files:

```text
annotated/gate/ground_truth_bulk_v1.json
annotated/restaurant/ground_truth_bulk_v1.json
annotated/washroom/ground_truth_bulk_v1.json
```

Keep the referenced PNGs beside each JSON. Legacy batches are not used.

## Run the baseline

```sh
# Audit source annotations (standard library only).
python scripts/audit_dataset.py

# Merge labels by source frame; use an unmarked negative-folder copy as input.
.venv/bin/python -m baseline.data

# One bounded T4 job; downloads checkpoint, predictions, history, and metrics.
.venv/bin/modal run modal_train.py

# Build measured report, ten slide scripts, and figures locally.
.venv/bin/python -m baseline.report
```

Positive PNGs have red boxes drawn into their pixels. Training uses unmarked copies
of the same source frame from another class folder. A negative in one class folder
can still be positive for other classes. The 208 frames with no unmarked copy are
excluded with approval, leaving 1,692 observations. This removes all triple-positive
frames and 54.5% of restaurant-positive frames, so the experiment has a substantial
selection bias. No annotation lines are erased or inpainted.

All original positive instance boxes, including duplicate signs, remain in
`data/baseline/manifest.json` in **original-image pixel coordinates**. They are
metadata for future detection and are not model inputs. The excluded frames and
their boxes are also recorded. The original label files remain unchanged.

The split uses 13 training, 3 validation, and 3 test videos, seed 42, with class
coverage checked before training. This yields 1,209 / 249 / 234 frames. There is no
source-video overlap between partitions, though different videos may depict the
same physical signs. Test is evaluated only after validation checkpoint selection.

## Confirmed hyperparameters

| Setting | Value |
| --- | --- |
| Input | RGB, bilinear letterbox to 224×224, ImageNet normalization |
| Augmentation | None |
| Epochs | 3 frozen-backbone + 7 all-layer fine-tuning |
| Batch size | 32 |
| Optimizer | AdamW; weight decay 0.0001 |
| Learning rates | Heads: 0.001; fine-tuning: 0.0001 |
| Loss | BCEWithLogitsLoss; positive weights = train negatives / train positives |
| Prediction thresholds | 0.5 per class |
| Checkpoint selection | Highest validation macro F1; earliest on ties |
| Seed | 42 |

Weighted-loss sigmoid scores are not calibrated confidence estimates. The saved
configuration records exact class weights, normalization, software versions, and
the Hugging Face weight revision. Reproducibility across hardware/software is not
guaranteed bit for bit.

## Predict on a new, unmarked image

```sh
.venv/bin/python -m baseline.predict path/to/image.png
```

This loads `artifacts/baseline/best.pt` and runs locally on CPU with no model
download. It returns three scores, the predicted classes, and an `irrelevant`
Boolean. Use the original image rather than an already-padded prepared image.

Prepared images, large generated JSON files, and model artifacts are gitignored.
The dataset audit is saved to `data/dataset_audit.json`, and the preparation summary
to `data/baseline/preparation_summary.json`. The raw ground truth and
`data/baseline/manifest.json` retain all boxes; `reports/` contains shareable metrics,
figures, predictions, and slide notes. Keep `artifacts/baseline/best.pt` locally.
The runner refuses to overwrite an existing metrics file; move the previous run
directory before intentionally starting another paid experiment.

## Modal budget

The authorized budget is **$10**. The runner uses one T4, two CPU cores, 4 GiB RAM,
a 30-minute function timeout, a 25-minute training timer, no automatic retries,
and no persistent storage or deployed endpoint. At the
[listed rates](https://modal.com/pricing) checked November 3, 2026, the function's
maximum compute estimate is approximately $0.36, excluding image build/startup and
other workspace usage. Actual billing is separate from this estimate. The timeout
limits this job's compute duration; it is not an account-wide dollar cap.

```sh
.venv/bin/modal billing report --for today --resolution h --show-resources
```

The confirmed experiment is a baseline, not a hyperparameter search. Later changes
to input resolution, augmentation, loss, thresholds, or architecture should be
recorded as separate experiments without tuning to this test set.
