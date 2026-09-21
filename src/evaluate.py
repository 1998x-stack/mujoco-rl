"""Evaluate the trained agent across reproducible, independent episodes."""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import SAC

from .common import LOGS, ensure_dirs, make_env, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Ant-v5 SAC policy")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    ensure_dirs()
    path = resolve_model(args.model)
    model = SAC.load(str(path), device="cpu")
    env = make_env()
    results = []
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
            result = {"episode": i + 1, "reward": reward_sum, "steps": n_steps, "forward_displacement_x": dx}
            results.append(result)
            print(f"Episode {i+1}/{args.episodes}: reward={reward_sum:.2f}, steps={n_steps}, forward_x={dx:.3f}", flush=True)
    finally:
        env.close()
    summary = {
        "model": str(path), "environment": "Ant-v5", "evaluation_seed": args.seed,
        "episodes": results, "mean_reward": float(np.mean([r["reward"] for r in results])),
        "std_reward": float(np.std([r["reward"] for r in results])),
        "mean_forward_displacement_x": float(np.mean([r["forward_displacement_x"] for r in results])),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = LOGS / "evaluation.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Mean reward={summary['mean_reward']:.2f}; mean forward displacement={summary['mean_forward_displacement_x']:.3f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
