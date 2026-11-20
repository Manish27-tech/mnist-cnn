# Data dictionary

## MNIST

| Field | Type | Shape | Description |
| --- | --- | --- | --- |
| `train_images` | uint8 | 60,000 x 28 x 28 | Grayscale digit images, 0–255 |
| `train_labels` | int64 | 60,000 | Digit class, 0–9 |
| `test_images` | uint8 | 10,000 x 28 x 28 | Official test images |
| `test_labels` | int64 | 10,000 | Official test labels |

Class names: `zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`,
`eight`, `nine`.

## Dashboard feed

| File | Description |
| --- | --- |
| `kpis.json` | Headline metrics and metadata |
| `history.csv` | Per-epoch train/validation loss and accuracy |
| `per_class.csv` | Precision, recall, F1 and support per digit |
| `robustness.csv` | Accuracy per corruption and severity |
| `predictions.csv` | Calibrated prediction for every test image |
| `confidence_buckets.csv` | Accuracy by confidence bucket |
| `selective_accuracy.csv` | Coverage and accuracy at thresholds |
| `samples.npz` | 60 sample images with labels and indices |
| `gradcam.npz` | Grad-CAM inputs and heatmaps |
