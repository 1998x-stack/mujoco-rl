"""Live viewer. On macOS, use the mjpython launcher if regular python complains."""
import argparse
from stable_baselines3 import SAC
from .common import make_env, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained MuJoCo Ant policy")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    model = SAC.load(str(resolve_model(args.model)), device="cpu")
    env = make_env(render_mode="human")
    try:
        for i in range(args.episodes):
            obs, _ = env.reset(seed=2000 + i)
            total = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            print(f"Replay {i + 1}: reward={total:.2f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
