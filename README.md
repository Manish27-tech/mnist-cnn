# MNIST CNN

A compact convolutional neural network for handwritten-digit recognition, with
temperature-scaling calibration, Grad-CAM explanations and a corruption
robustness sweep.

## Results

Filled in after training (see `reports/REPORT.md`).

## Quickstart

```bash
python -m pip install -r requirements.txt
python run.py all
python run.py check
python run.py dashboard
```

The pipeline downloads MNIST (~11 MB), trains for 3 epochs on CPU, writes eight
figures to `reports/figures/` and a dashboard feed to `artifacts/dashboard/`.

## Project layout

```
src/mnistvision/   data, model, training, evaluation, Grad-CAM, robustness
dashboard/         Streamlit demo
tests/             pytest suite
reports/           metrics, figures, report and model card
artifacts/         trained checkpoint and dashboard feed
```

## Licence

MIT.
