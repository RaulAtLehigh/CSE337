#!/usr/bin/env python3
"""Automate DQN architecture experiments for SteakEnv."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List

from perfect_steak.experimentation.config import ExperimentConfig, SweepConfig
from perfect_steak.experimentation.logging_utils import (
    build_comparison_entry,
    generate_comparison_report,
    plot_rewards,
    save_metadata,
    save_metrics,
    save_model_checkpoint,
    write_summary,
)
from perfect_steak.experimentation.training import DQNTrainer


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run SteakEnv DQN architecture sweep.")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=default_root / "configs" / "architecture_grid.json",
        help="Path to the JSON file describing experiments.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_root / "results",
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument(
        "--comparison-file",
        type=Path,
        default=default_root / "results" / "comparison.md",
        help="Markdown summary written after the sweep completes.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Override the number of training episodes for every run.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        help="Override evaluation interval (episodes).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help="Override number of evaluation rollouts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override minibatch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the first configuration for a minimal episode count.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already have metadata saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the RNG seed for every experiment.",
    )
    return parser.parse_args()


def maybe_override_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updated = cfg.with_overrides(
        episodes=args.episodes if args.episodes is not None else None,
        eval_interval=args.eval_interval if args.eval_interval is not None else None,
        batch_size=args.batch_size if args.batch_size is not None else None,
        eval_episodes=args.eval_episodes if args.eval_episodes is not None else None,
    )
    if args.seed is not None:
        updated = replace(updated, seed=args.seed)
    return updated


def load_existing_metadata(run_dir: Path) -> dict | None:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    return None


def main() -> None:
    args = parse_args()
    sweep = SweepConfig.from_file(args.config_file)
    experiments = sweep.experiments
    if args.dry_run and experiments:
        dry_cfg = experiments[0]
        experiments = [dry_cfg]
        if args.episodes is None:
            args.episodes = 50
    results: List[dict] = []

    for config in experiments:
        cfg = maybe_override_config(config, args)
        run_dir = args.results_dir / cfg.id
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = load_existing_metadata(run_dir)
        if metadata and args.resume:
            print(f"[skip] {cfg.id} already has metadata, skipping (resume).")
            results.append(metadata)
            continue

        print(f"[train] Starting experiment {cfg.id} with layers {cfg.hidden_layer_sizes}")
        trainer = DQNTrainer(cfg)
        run_result = trainer.train()

        save_metrics(run_result, run_dir)
        plot_rewards(run_result, run_dir)
        save_model_checkpoint(run_result, run_dir)
        write_summary(run_result, run_dir)
        save_metadata(run_result, run_dir)

        comparison_entry = build_comparison_entry(run_result)
        results.append(comparison_entry)

    if results:
        generate_comparison_report(results, args.comparison_file)
        print(f"Wrote comparison summary to {args.comparison_file}")
    else:
        print("No experiments were executed. Nothing to summarize.")


if __name__ == "__main__":
    main()
