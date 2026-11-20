# Methodology

## Data

MNIST (LeCun et al., 1998) is downloaded with `torchvision.datasets.MNIST` on
first use and cached as a compressed NumPy archive. The 60,000 training images
are split into 54,000 train / 6,000 validation using a stratified split by
class, seed 42. The official 10,000-image test set is untouched until final
evaluation.

## Model

`MNISTCNN` is a three-block convolutional network:

| Stage | Operation | Output |
| --- | --- | --- |
| stem | Conv 3x3 (1→32) + BN + ReLU | 28x28x32 |
| pool | MaxPool 2x2 | 14x14x32 |
| block 1 | Conv 3x3 (32→64) + BN + ReLU | 14x14x64 |
| pool | MaxPool 2x2 | 7x7x64 |
| block 2 | Conv 3x3 (64→128) + BN + ReLU | 7x7x128 |
| head | GAP → Dropout 0.15 → Linear 128→10 | 10 |

The final convolutional layer is exposed as `gradcam_layer` for Grad-CAM.

## Training

- Optimiser: AdamW, learning rate 1e-3, weight decay 5e-4
- Schedule: cosine annealing over 3 epochs
- Loss: cross-entropy with label smoothing 0.05
- Batch size 128, seed 42, 12 CPU threads
- Augmentation: small affine jitter (rotation ±8°, translation 8%, scale ±5%)

The checkpoint with the best validation accuracy is kept.

## Evaluation

- Accuracy and macro F1 on the test split
- Per-class precision, recall and F1
- Confusion matrix and the ten most frequent confusions
- Temperature scaling fitted on the validation logits
- Expected and maximum calibration error, before and after scaling
- Confidence buckets and selective accuracy at fixed thresholds

## Robustness

A stratified slice of 1,000 test images is corrupted with additive Gaussian
noise, Gaussian blur, contrast scaling and central occlusion. Accuracy is
reported at each severity alongside the clean slice accuracy.

## Explainability

Grad-CAM hooks the last convolutional block, weights activations by the mean
gradient and overlays the normalised heatmap on the input digit.
