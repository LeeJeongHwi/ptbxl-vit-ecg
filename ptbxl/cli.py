"""Command-line interface: train and evaluate a single model with the default pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ptbxl.config import PipelineConfig
from ptbxl.model.registry import list_models
from ptbxl.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the PTB-XL pipeline."""
    parser = argparse.ArgumentParser(description="Train and evaluate a model on PTB-XL.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("npy_data"))
    parser.add_argument("--model-dir", type=Path, default=Path(f"models/{datetime.now().strftime('%Y%m%d_%H%M')}"))
    parser.add_argument("--model", type=str, default="vit", choices=list_models(),
                        help="Which model to train/evaluate.")
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=500)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.7143,
                        help="Classification threshold used for the test-set evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true", help="Skip training; evaluate latest checkpoint.")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip evaluation after training.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore cached arrays and rebuild.")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """CLI entry-point; prints result paths and key metrics as JSON."""
    args = parse_args()
    config = PipelineConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model_dir=args.model_dir,
        model_name=args.model,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        threshold=args.threshold,
        seed=args.seed,
        rebuild_cache=args.rebuild_cache,
        early_stopping_patience=args.early_stopping_patience,
        weight_decay=args.weight_decay,
    )
    results = run_pipeline(config, train=not args.no_train, evaluate=not args.no_evaluate)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
