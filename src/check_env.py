"""Non-graphical environment smoke check: does not depend on an X server."""
import argparse
import platform

import numpy as np

from .common import ENV_ID, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MuJoCo Ant-v5 environment")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    import gymnasium, mujoco, stable_baselines3
    print(f"Host: {platform.platform()} ({platform.machine()})")
    print(f"gymnasium={gymnasium.__version__}, mujoco={mujoco.__version__}, stable_baselines3={stable_baselines3.__version__}")
    env = make_env(seed=42)
    try:
        obs, _ = env.reset(seed=42)
        print(f"Environment: {ENV_ID}, observation={env.observation_space}, action={env.action_space}")
        assert env.action_space.shape == (8,), f"Unexpected action space: {env.action_space}"
        assert obs.shape == env.observation_space.shape, f"Unexpected observation shape: {obs.shape}"
        assert np.isfinite(obs).all(), "Initial observation has NaN or inf"
        episodes = 0
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(reward), "Nonfinite environment result"
            if terminated or truncated:
                episodes += 1
                obs, _ = env.reset()
        print(f"PASS: {args.steps} physics/control steps, {episodes} episode reset(s)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
