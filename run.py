"""Command line entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mnistvision import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MNIST CNN pipeline")
    sub = parser.add_subparsers(dest="command")

    all_p = sub.add_parser("all", help="run the full pipeline")
    all_p.add_argument("--retrain", action="store_true")
    sub.add_parser("data", help="download and cache MNIST")
    train_p = sub.add_parser("train", help="train the reference model")
    train_p.add_argument("--retrain", action="store_true")
    sub.add_parser("check", help="verify committed artefacts")
    dash_p = sub.add_parser("dashboard", help="launch the Streamlit app")
    dash_p.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()

    if args.command == "all":
        from mnistvision.pipeline import run_all

        result = run_all(retrain=args.retrain)
        print(f"accuracy: {result['accuracy']:.4f}")
        print(f"figures: {len(result['figures'])}")
    elif args.command == "data":
        from mnistvision.data import load_dataset

        dataset = load_dataset()
        print(dataset.summary)
    elif args.command == "train":
        from mnistvision.data import load_dataset
        from mnistvision.train import train_reference

        result, _ = train_reference(load_dataset(), checkpoint_path=config.CHECKPOINT)
        print(result.summary)
    elif args.command == "check":
        from mnistvision.manifest import check_artifacts

        print(check_artifacts())
    elif args.command == "dashboard":
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
             "--server.port", str(args.port)],
            cwd=ROOT,
        )
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
