"""Offline, seed-level statistical benchmark aggregation for Ant-v5 SAC.

The independent unit for uncertainty is the *training seed*, never each
correlated evaluation episode. A smoke test is not evidence of locomotion.
"""
from __future__ import annotations

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random
from statistics import mean, stdev

from .common import ENV_ID


def _numeric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def load_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation report {path}: {exc}") from exc
    if report.get("schema") != 2 or report.get("environment") != ENV_ID:
        raise ValueError(f"Expected schema 2 evaluation for {ENV_ID}: {path}")
    seed = report.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Untracked training seed (legacy or external model): {path}")
    if not isinstance(report.get("contract_fingerprint"), str) or len(report["contract_fingerprint"]) != 64:
        raise ValueError(f"Missing training configuration fingerprint: {path}")
    if not isinstance(report.get("configuration_fingerprint"), str) or len(report["configuration_fingerprint"]) != 64:
        raise ValueError(f"Missing shared configuration fingerprint: {path}")
    if not isinstance(report.get("model_sha256"), str) or len(report["model_sha256"]) != 64:
        raise ValueError(f"Missing model SHA256: {path}")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Report contains no episodes: {path}")
    rewards = [_numeric(item["reward"], "episode.reward") for item in episodes]
    avg = mean(rewards)
    if not isclose(avg, _numeric(report.get("mean_reward"), "mean_reward"), rel_tol=1e-8, abs_tol=1e-7):
        raise ValueError(f"Reported mean disagrees with episode rewards: {path}")
    eval_seed = report.get("evaluation_seed")
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, int):
        raise ValueError(f"Invalid evaluation seed in {path}")
    if not isinstance(report.get("model"), str) or not report["model"]:
        raise ValueError(f"Missing model path: {path}")
    return {
        "training_seed": seed,
        "evaluation_seed": eval_seed,
        "episode_count": len(episodes),
        "mean_reward": avg,
        "model_sha256": report["model_sha256"],
        "contract_fingerprint": report["contract_fingerprint"],
        "configuration_fingerprint": report["configuration_fingerprint"],
        "report_path": str(path.resolve()),
    }


def _quantile(sorted_samples: list[float], probability: float) -> float:
    index = (len(sorted_samples) - 1) * probability
    lower = int(index)
    delta = index - lower
    upper = min(lower + 1, len(sorted_samples) - 1)
    return sorted_samples[lower] * (1 - delta) + sorted_samples[upper] * delta


def aggregate(paths: list[Path], *, min_seeds: int = 2, bootstrap_samples: int = 2000,
              bootstrap_seed: int = 2026) -> dict:
    if min_seeds < 2 or bootstrap_samples < 100:
        raise ValueError("Require >=2 independent training seeds and >=100 bootstrap samples")
    if not paths:
        raise ValueError("No evaluation reports supplied")
    reports = [load_report(Path(path)) for path in paths]
    seeds = [item["training_seed"] for item in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training seed: refusing to treat repeated evaluations as independent runs")
    if len(seeds) < min_seeds:
        raise ValueError(f"Need at least {min_seeds} independent training seeds; got {len(seeds)}")
    protocols = {(item["evaluation_seed"], item["episode_count"]) for item in reports}
    if len(protocols) != 1:
        raise ValueError("Incompatible evaluation protocols: seed or episode count differs")
    fingerprints = {item["configuration_fingerprint"] for item in reports}
    if len(fingerprints) != 1:
        raise ValueError("Training configuration differs across seeds")
    scores = [item["mean_reward"] for item in reports]
    center = mean(scores)
    rng = Random(bootstrap_seed)
    estimates = sorted(
        mean(scores[rng.randrange(len(scores))] for _ in scores)
        for _ in range(bootstrap_samples)
    )
    eval_seed, episode_count = next(iter(protocols))
    return {
        "schema": 1,
        "environment": ENV_ID,
        "algorithm": "SAC",
        "configuration_fingerprint": next(iter(fingerprints)),
        "number_of_training_seeds": len(seeds),
        "evaluation_seed": eval_seed,
        "episodes_per_seed": episode_count,
        "mean_reward": center,
        "std_reward_between_seeds": stdev(scores),
        "confidence_interval_95": [
            _quantile(estimates, 0.025), _quantile(estimates, 0.975),
        ],
        "uncertainty_method": "percentile bootstrap over training-seed means",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "runs": sorted(reports, key=lambda item: item["training_seed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independent SAC training-seed evaluations")
    parser.add_argument("reports", nargs="+", type=Path, help="Per-seed evaluation.json paths")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        results = aggregate(args.reports, min_seeds=args.min_seeds,
                            bootstrap_samples=args.bootstrap_samples,
                            bootstrap_seed=args.bootstrap_seed)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Saved benchmark to {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
