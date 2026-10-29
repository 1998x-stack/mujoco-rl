"""Generate an evidence-linked Markdown scorecard without inventing RL results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _num(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing numerical scorecard value")
    return f"{float(value):.3f}"


def render_scorecard(benchmark: dict) -> str:
    if benchmark.get("schema") != 2 or not benchmark.get("quality_metrics"):
        raise ValueError("A trajectory-aware benchmark result is required")
    runs = benchmark.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Benchmark has no independently trained seeds")
    official = benchmark.get("publication_protocol_verified") is True
    title = "Ant-v5 SAC — publication protocol verified" if official else "Ant-v5 SAC — CI / exploratory only"
    lines = [
        f"# {title}", "",
        f"- Environment: `{benchmark['environment']}`",
        f"- Independent training seeds: **{benchmark['number_of_training_seeds']}**",
        f"- Steps per seed: **{benchmark['training_budget_timesteps']:,}**",
        f"- Deterministic evaluation episodes per seed: **{benchmark['episodes_per_seed']}**",
        f"- Evaluation seed base: `{benchmark['evaluation_seed']}`",
        "- Progress proxy: **net forward displacement >= 1 metre in one episode** (not a proven gait skill).",
        "- Confidence intervals: percentile bootstrap across **independent training seeds**.", "",
    ]
    if not official:
        lines.extend(["**NOT A PUBLISHABLE LEARNING RESULT:** short-run scores only test pipeline mechanics.", ""])
    lines.extend([
        "## Per-seed results", "",
        "| Training seed | Mean episode return | Net forward distance (m) | Early termination rate | Progress proxy rate |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in sorted(runs, key=lambda r: r["training_seed"]):
        q = run.get("quality")
        if not isinstance(q, dict):
            raise ValueError("Missing per-seed trajectory quality")
        lines.append(
            f"| {run['training_seed']} | {_num(run['mean_reward'])} | "
            f"{_num(q['mean_forward_displacement_x'])} | "
            f"{_num(q['early_termination_rate'])} | "
            f"{_num(q['positive_progress_rate'])} |"
        )
    lines.extend(["", "## Aggregate metrics (seed-level 95% intervals)", "",
                  "| Metric | Across-seed mean | 95% interval |",
                  "| --- | ---: | ---: |"])
    reward_ci = benchmark["confidence_interval_95"]
    lines.append(f"| Mean episode return | {_num(benchmark['mean_reward'])} | [{_num(reward_ci[0])}, {_num(reward_ci[1])}] |")
    descriptions = {
        "mean_forward_displacement_x": "Net forward displacement (m)",
        "mean_forward_speed_x": "Net x speed (m/s)",
        "early_termination_rate": "Early termination rate",
        "time_limit_rate": "Time-limit completion rate",
        "positive_progress_rate": "Progress proxy fraction (>= 1 m)",
        "mean_episode_length_steps": "Episode length (steps)",
    }
    for key, label in descriptions.items():
        metric = benchmark["quality_metrics"][key]
        ci = metric["confidence_interval_95"]
        lines.append(f"| {label} | {_num(metric['mean'])} | [{_num(ci[0])}, {_num(ci[1])}] |")
    lines.extend(["", "## Interpretation", "",
                  "A high survival reward does not by itself demonstrate forward locomotion. "
                  "Forward distance and speed, termination frequency, episode lengths and rendered trajectories "
                  "must be inspected together. Five seeds and ten evaluation episodes are a starting "
                  "protocol, not a statistical guarantee of superiority.",
                  "",
                  "The accompanying `benchmark.json` contains raw seed-level summaries and protocol fingerprints. "
                  "The workflow also publishes verified individual policy ZIPs, experiment metadata, "
                  "resolved dependency lists and audit summaries.",
                  ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a scientifically qualified Ant-v5 benchmark scorecard")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.benchmark.read_text(encoding="utf-8"))
        markdown = render_scorecard(data)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Scorecard saved: {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
