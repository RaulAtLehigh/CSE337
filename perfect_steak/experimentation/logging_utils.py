"""Utility functions for persisting experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List

import matplotlib.pyplot as plt

from .training import (
    EpisodeMetrics,
    OPTIMAL_SCORE,
    TrainingRunResult,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metrics(result: TrainingRunResult, run_dir: Path) -> None:
    ensure_dir(run_dir)
    metrics_payload = [asdict(m) for m in result.episode_metrics]
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    eval_payload = [asdict(ep) for ep in result.evaluation_episodes]
    (run_dir / "actions_eval.json").write_text(json.dumps(eval_payload, indent=2))


def save_model_checkpoint(result: TrainingRunResult, run_dir: Path) -> None:
    ensure_dir(run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    import torch

    torch.save(result.best_model_state, checkpoint_path)


def write_summary(result: TrainingRunResult, run_dir: Path) -> None:
    ensure_dir(run_dir)
    layers = result.config.hidden_layer_sizes
    summary_lines = [
        f"Experiment ID: {result.config.id}",
        f"Seed: {result.config.seed}",
        f"Hidden layers: {len(layers)} -> {layers}",
        f"Learning rate: {result.config.learning_rate}",
        f"Gamma: {result.config.gamma}",
        f"Epsilon schedule: start={result.config.epsilon_start}, "
        f"min={result.config.epsilon_min}, decay={result.config.epsilon_decay}",
        f"Batch size: {result.config.batch_size}",
        f"Replay buffer: {result.config.replay_buffer_size}",
        f"Target update frequency (steps): {result.config.target_update_freq}",
        f"Evaluation interval (episodes): {result.config.eval_interval}",
        f"Total episodes: {result.config.num_episodes}",
        f"Best eval return: {result.best_eval_score:.3f} "
        f"(recorded episode {result.best_eval_episode})",
        f"Time to reach optimal score (moving average >= {OPTIMAL_SCORE}): "
        f"{result.time_to_optimal_episode if result.time_to_optimal_episode else 'not reached'}",
        f"Total training time (s): {result.total_training_seconds:.1f}",
        "Artifacts:",
        f"  - metrics.json",
        f"  - reward_curve.png",
        f"  - actions_eval.json",
        f"  - best_model.pt",
    ]
    if result.config.notes:
        summary_lines.append(f"Notes: {result.config.notes}")
    (run_dir / "summary.txt").write_text("\n".join(summary_lines))


def plot_rewards(
    result: TrainingRunResult,
    run_dir: Path,
    *,
    optimal_score: float = OPTIMAL_SCORE,
) -> None:
    ensure_dir(run_dir)
    if not result.episode_metrics:
        return
    episodes = [m.episode for m in result.episode_metrics]
    rewards = [m.reward for m in result.episode_metrics]
    moving_avg = [m.moving_avg for m in result.episode_metrics]

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, label="Episode return", alpha=0.4)
    plt.plot(episodes, moving_avg, label="Moving average", linewidth=2.0)
    plt.axhline(optimal_score, color="green", linestyle="--", label="Optimal score")

    if result.time_to_optimal_episode:
        plt.axvline(
            result.time_to_optimal_episode,
            color="purple",
            linestyle=":",
            label="First optimal crossing",
        )
        plt.annotate(
            f"Crossed optimal @ ep {result.time_to_optimal_episode}",
            xy=(result.time_to_optimal_episode, optimal_score),
            xytext=(result.time_to_optimal_episode, optimal_score + 0.5),
            arrowprops=dict(arrowstyle="->", color="purple"),
            fontsize=8,
        )

    plt.scatter(
        [result.best_eval_episode],
        [result.best_eval_score],
        color="red",
        marker="o",
        label="Best eval avg",
    )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Training curve — {result.config.id}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "reward_curve.png", dpi=150)
    plt.close()


def save_metadata(result: TrainingRunResult, run_dir: Path) -> None:
    ensure_dir(run_dir)
    metadata = {
        "experiment_id": result.config.id,
        "seed": result.config.seed,
        "hidden_layer_sizes": result.config.hidden_layer_sizes,
        "best_eval_score": result.best_eval_score,
        "best_eval_episode": result.best_eval_episode,
        "time_to_optimal_episode": result.time_to_optimal_episode,
        "total_training_seconds": result.total_training_seconds,
        "num_episodes": result.config.num_episodes,
        "notes": result.config.notes,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def build_comparison_entry(result: TrainingRunResult) -> dict[str, Any]:
    return {
        "id": result.config.id,
        "hidden_layer_sizes": result.config.hidden_layer_sizes,
        "best_eval_score": result.best_eval_score,
        "time_to_optimal_episode": result.time_to_optimal_episode,
        "total_training_seconds": result.total_training_seconds,
    }


def generate_comparison_report(
    summaries: Iterable[TrainingRunResult | dict[str, Any]], output_path: Path
) -> None:
    ensure_dir(output_path.parent)
    lines = [
        "# Experiment Comparison",
        "",
        "| ID | Hidden layers | Total hidden units | Best eval return | "
        "Time to optimal | Training time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in summaries:
        if is_dataclass(entry):
            row = build_comparison_entry(entry)  # type: ignore[arg-type]
        elif isinstance(entry, dict):
            row = entry
        else:
            raise TypeError("Unsupported summary entry type")
        hidden_layers = row["hidden_layer_sizes"]
        total_units = sum(hidden_layers)
        hidden_desc = " → ".join(map(str, hidden_layers))
        time_to_optimal = (
            str(row.get("time_to_optimal_episode"))
            if row.get("time_to_optimal_episode")
            else "not reached"
        )
        lines.append(
            f"| {row.get('id')} | {hidden_desc} | {total_units} | "
            f"{row.get('best_eval_score', 0.0):.2f} | {time_to_optimal} | "
            f"{row.get('total_training_seconds', 0.0):.1f} |"
        )
    lines.append("")
    lines.append("Each run directory contains `summary.txt`, `reward_curve.png`, "
                 "`actions_eval.json`, and `best_model.pt`.")
    output_path.write_text("\n".join(lines))
