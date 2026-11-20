# Deployment

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Sign in at <https://share.streamlit.io>.
3. Choose **New app**, select the repository and branch `main`.
4. Set the main file path to `dashboard/app.py`.
5. Deploy. No secrets or GPU are required.

The app reads the committed feed in `artifacts/dashboard/`, so it starts in
seconds and does not download MNIST or train a model.

## Local

```bash
python -m pip install -r requirements.txt
python run.py dashboard
```

## Docker

```bash
docker build -t mnist-cnn .
docker run --rm -p 8501:8501 mnist-cnn
```

## Retraining

```bash
python run.py all --retrain
```

This downloads MNIST if needed, trains for three epochs, rewrites the figures
and refreshes the dashboard feed. Expect a few minutes on a modern CPU.
