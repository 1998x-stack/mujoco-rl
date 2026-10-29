"""Evaluate the trained agent across reproducible, independent episodes."""
import argparse
import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import SAC

from .checkpoint import load_checkpoint_paths
from .common import ENV_ID, LOGS, MODELS, ensure_dirs, make_env, resolve_model
from .provenance import comparable_digest, read_manifest
from .quality import PROGRESS_THRESHOLD_METERS, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Ant-v5 SAC policy")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="JSON report destination (default: logs/evaluation.json)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    ensure_dirs()
    path = resolve_model(args.model)
    # Explicit external models cannot inherit this output root's training seed.
    manifest = read_manifest() if args.model is None or path.resolve().is_relative_to(MODELS.resolve()) else None
    with path.open("rb") as handle:
        checksum = sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    checkpoint = load_checkpoint_paths() if manifest is not None else None
    training_budget = int(checkpoint[2]["timesteps"]) if checkpoint is not None else None
    model = SAC.load(str(path), device="cpu")
    env = make_env()
    results = []
    time_step = float(env.unwrapped.dt)
    if not isfinite(time_step) or time_step <= 0:
        env.close()
        raise ValueError('Invalid physics time step')
    try:
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + i)
            initial_x = float(env.unwrapped.data.qpos[0])
            reward_sum = 0.0
            n_steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                n_steps += 1
                if terminated or truncated:
                    break
            dx = float(env.unwrapped.data.qpos[0]) - initial_x
            if not isfinite(reward_sum) or not isfinite(dx):
                raise ValueError("Non-finite episode reward or displacement")
            elapsed = n_steps * time_step
            result = {
                "episode": i + 1, "reward": reward_sum, "steps": n_steps,
                "forward_displacement_x": dx, "elapsed_simulation_seconds": elapsed,
                "mean_forward_speed_x": dx / elapsed,
                "terminated": bool(terminated), "truncated": bool(truncated),
            }
            results.append(result)
            print(f"Episode {i+1}/{args.episodes}: reward={reward_sum:.2f}, steps={n_steps}, forward_x={dx:.3f}", flush=True)
    finally:
        env.close()
    summary = {
        "schema": 3, "model": str(path), "model_sha256": checksum.hexdigest(),
        "environment": ENV_ID, "evaluation_seed": args.seed,
        "training_seed": manifest["contract"]["seed"] if manifest else None,
        "training_budget_timesteps": training_budget,
        "contract_fingerprint": manifest["fingerprint"] if manifest else None,
        "configuration_fingerprint": comparable_digest(manifest["contract"]) if manifest else None,
        "environment_versions": {
            k: manifest["packages"].get(k) for k in ("gymnasium", "mujoco", "stable-baselines3")
        } if manifest else None,
        "evaluation_protocol": {
            "deterministic_policy": True,
            "episode_seeding": "evaluation_seed_plus_index",
            "progress_threshold_meters": PROGRESS_THRESHOLD_METERS,
            "physics_time_step_seconds": time_step,
            "max_episode_steps": env.spec.max_episode_steps,
        },
        "quality": summarize(results),
        "episodes": results, "mean_reward": float(np.mean([r["reward"] for r in results])),
        "std_reward": float(np.std([r["reward"] for r in results])),
        "mean_forward_displacement_x": float(np.mean([r["forward_displacement_x"] for r in results])),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(args.output).expanduser() if args.output else LOGS / "evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Mean reward={summary['mean_reward']:.2f}; mean forward displacement={summary['mean_forward_displacement_x']:.3f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
