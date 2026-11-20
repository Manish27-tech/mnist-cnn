# Results

Run date: 2026-11-20. Seed 42, CPU only (12 threads).

## Headline

| Metric | Value |
| --- | --- |
| Test accuracy | 0.9822 |
| Macro F1 | 0.9822 |
| Validation accuracy (best epoch) | 0.9813 |
| Parameters | 94,186 |
| Epochs | 3 |
| Training time | 265.08 s |
| Temperature | 0.5146 |
| ECE before | 0.0657 |
| ECE after | 0.0040 |
| MCE before | 0.2841 |
| Selective accuracy at 0.90 | 0.9974 at 94.6% coverage |

## Per-class scores

| Digit | Precision | Recall | F1 |
| --- | --- | --- | --- |
| zero | 0.991 | 0.980 | 0.985 |
| one | 0.991 | 0.991 | 0.991 |
| two | 0.973 | 0.994 | 0.983 |
| three | 0.984 | 0.990 | 0.987 |
| four | 0.982 | 0.973 | 0.977 |
| five | 0.992 | 0.980 | 0.986 |
| six | 0.986 | 0.984 | 0.985 |
| seven | 0.998 | 0.962 | 0.980 |
| eight | 0.974 | 0.986 | 0.980 |
| nine | 0.954 | 0.981 | 0.967 |

The weakest class is `nine` (F1 0.967), the strongest is `one` (F1 0.991).
The most frequent error is a `four` predicted as `nine` (20 test images).

## Robustness

Stratified slice of 1,000 test images; clean accuracy 0.982.

| Corruption | Severity | Accuracy |
| --- | --- | --- |
| noise | 0.10 | 0.766 |
| noise | 0.20 | 0.228 |
| noise | 0.30 | 0.101 |
| blur | 0.50 | 0.978 |
| blur | 1.00 | 0.792 |
| blur | 1.50 | 0.243 |
| contrast | 0.50 | 0.228 |
| contrast | 0.75 | 0.865 |
| contrast | 1.25 | 0.980 |
| occlusion | 8 px | 0.683 |
| occlusion | 12 px | 0.340 |
| occlusion | 16 px | 0.142 |

## Calibration

Temperature scaling fits a single parameter on the 6,000-image validation
split. It leaves accuracy unchanged and reduces expected calibration error from
0.0657 to 0.0040. At a 0.90 confidence threshold the model keeps 94.6% of test
images and is correct on 99.74% of them.

## Reproduce

```bash
python run.py all --retrain
```

Expect about four to five minutes on a modern CPU.
