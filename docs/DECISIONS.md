# Decisions

## Why MNIST

The project needed a real image-classification dataset that trains on CPU in
minutes. MNIST downloads in seconds (~11 MB), is well understood and does not
overlap with the author's earlier tabular projects.

## Why a compact CNN rather than a large pretrained model

The brief asked for a moderate model that trains quickly. A three-block CNN
reaches around 99% test accuracy in three epochs without a GPU, which keeps the
repository reproducible on any laptop. A pretrained ResNet would add hundreds of
megabytes and no educational value for 28x28 grayscale digits.

## Why temperature scaling

Neural networks are often overconfident. Temperature scaling fits a single
parameter on the validation split and usually reduces expected calibration
error without changing accuracy. It is cheap, interpretable and easy to audit.

## Why Grad-CAM

Grad-CAM gives a per-pixel explanation without retraining or modifying the
architecture. Exposing the last convolutional layer makes the method a
five-line hook rather than a separate model.

## Why a corruption sweep

Clean test accuracy hides brittleness. A fixed, stratified slice of 1,000
images is corrupted at several severities so the report shows where the model
degrades, not just its best-case score.

## Why commit the artefacts

The dashboard must run on Streamlit Cloud, where the dataset is not present and
training would be wasteful. Committing the checkpoint, figures and a compact
dashboard feed keeps the demo reproducible and instant.
