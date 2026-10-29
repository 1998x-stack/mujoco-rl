"""Orchestrate isolated, comparable Ant-v5 SAC training seeds.

No shell invocation, implicit deletion, or silent reuse of prior experiment
state. One subprocess and artifact root per seed; fail on the first bad run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .benchmark import aggregate
from .common import ROOT
from .provenance import contract_digest, read_manifest, training_contract


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated nonnegative integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed <= 2**32 - 1 for seed in seeds):
        raise argparse.ArgumentTypeError("Seeds must be distinct integers in [0, 2**32 - 1]")
    return seeds


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("Expected positive integer")
    return value


def validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) or name in (".", "..") or ".." in name:
        raise ValueError("Experiment name must be a safe slug (letters, numbers, _, -, .)")
    return name


def saved_training_steps(run_root: Path, seed: int) -> int:
    """Read a strict latest-pointer metadata record before scheduling resume.

    The training subprocess independently verifies SHA256 model/replay integrity.
    """
    pointer_path = run_root / "models" / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    snapshot = pointer.get("snapshot")
    if pointer.get("schema") != 1 or not isinstance(snapshot, str) or not re.fullmatch(r"snapshot-[a-f0-9]{32}", snapshot):
        raise ValueError(f"Malformed latest snapshot pointer: {pointer_path}")
    meta_path = run_root / "models" / "snapshots" / snapshot / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    steps = metadata.get("timesteps")
    if metadata.get("schema") != 1 or metadata.get("environment") != "Ant-v5" or metadata.get("seed") != seed or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"Incompatible training metadata: {meta_path}")
    return steps


def execute_suite(*, name: str, seeds: list[int], steps: int, episodes: int,
                  base_dir: Path, resume: bool = False, evaluate_only: bool = False,
                  eval_seed: int = 1000, device: str = "cpu") -> Path:
    validate_name(name)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct training seeds are required")
    if steps < 1 or episodes < 1 or (resume and evaluate_only):
        raise ValueError("Invalid step/episode count or incompatible run modes")
    suite_root = base_dir.expanduser().resolve() / name
    # A partially completed suite is never overwritten with fresh training.
    # --resume is explicit and each subprocess validates its own manifest.
    previous_steps = {}
    for seed in seeds:
        run_root = suite_root / f"seed-{seed}"
        if not resume and not evaluate_only and any(
            (run_root / folder).exists() and any((run_root / folder).iterdir())
            for folder in ("models", "logs", "videos")
        ):
            raise FileExistsError(f"Existing artifacts at {run_root}; use --resume or a new --name")
        if (resume or evaluate_only) and not (run_root / "models" / "latest.json").is_file():
            raise FileNotFoundError(f"No completed snapshot for seed={seed}: {run_root}")
        if resume or evaluate_only:
            manifest = read_manifest(run_root / "models" / "experiment.json")
            if manifest is None:
                raise ValueError(f"Untracked experiment seed={seed}; cannot safely resume/benchmark without manifest")
            if manifest["contract"].get("seed") != seed:
                raise ValueError(f"Experiment seed mismatch at {run_root}")
            if resume and manifest["fingerprint"] != contract_digest(training_contract(seed=seed, device=device)):
                raise ValueError(f"Resume configuration drift for seed={seed} at {run_root}")
        if resume:
            previous_steps[seed] = saved_training_steps(run_root, seed)
            if previous_steps[seed] > steps:
                raise ValueError(f"Seed={seed} already exceeds target budget {steps}; stored steps={previous_steps[seed]}")
    reports: list[Path] = []
    for seed in seeds:
        run_root = suite_root / f"seed-{seed}"
        env = dict(os.environ, MUJOCO_RL_OUTPUT_ROOT=str(run_root))
        remaining = steps - previous_steps.get(seed, 0)
        if not evaluate_only and remaining > 0:
            command = [
                sys.executable, "-m", "src.train",
                "--steps", str(remaining), "--seed", str(seed),
                "--device", device,
                "--eval-freq", str(max(256, remaining // 4)),
                "--checkpoint-freq", str(max(1024, remaining // 2)),
            ]
            if resume:
                command.append("--resume")
            print(f"[suite] Training seed={seed} output={run_root}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        elif not evaluate_only:
            print(f"[suite] Seed={seed} already at target {steps}; skipping duplicate training", flush=True)
        print(f"[suite] Evaluating seed={seed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "src.evaluate", "--episodes", str(episodes),
             "--seed", str(eval_seed)],
            cwd=ROOT, env=env, check=True,
        )
        reports.append(run_root / "logs" / "evaluation.json")
    results = aggregate(reports, min_seeds=len(seeds))
    destination = suite_root / "benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"[suite] Benchmark: {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent MuJoCo SAC training seeds and evaluate")
    parser.add_argument("--name", default="ant-sac-baseline")
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=positive_int, default=1_000_000,
                        help="Target total training steps per seed; --resume trains only the missing steps")
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--base-dir", type=Path, default=ROOT / "runs")
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--resume", action="store_true")
    choices.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    try:
        execute_suite(name=args.name, seeds=args.seeds, steps=args.steps,
                      episodes=args.episodes, base_dir=args.base_dir,
                      resume=args.resume, evaluate_only=args.evaluate_only,
                      eval_seed=args.eval_seed, device=args.device)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
