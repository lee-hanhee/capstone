# Experiments

Each experiment has one tracked `config.yaml`. Generated checkpoints stay under
`artifacts/`, while shareable metrics and predictions stay under `reports/`.

| Experiment | Purpose | Results |
| --- | --- | --- |
| [`v3_224`](v3_224/config.yaml) | Original 224px baseline | [`reports/baseline_results.md`](../reports/baseline_results.md) |
| [`v3_224_thresholds`](v3_224_thresholds/config.yaml) | Validation-selected thresholds on the saved 224px model | [`reports/resolution_thresholds`](../reports/resolution_thresholds/) |
| [`v3_320`](v3_320/config.yaml) | Higher-resolution comparison | [`reports/resolution_thresholds`](../reports/resolution_thresholds/) |
| [`v3_224_clean`](v3_224_clean/config.yaml) | Clean-image 224px model used in the progress report | [`reports/clean_224/results.md`](../reports/clean_224/results.md) |

Use the clean experiment for the presentation. The other three are retained as
technical comparison runs.
