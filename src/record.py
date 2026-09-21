"""Save RGB frames as a self-contained MP4 without requiring a system ffmpeg install."""
import argparse

import imageio.v2 as imageio
from stable_baselines3 import SAC

from .common import VIDEOS, ensure_dirs, make_env, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a MuJoCo Ant-v5 policy as MP4")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    ensure_dirs()
    output = args.output or str(VIDEOS / "ant_demo.mp4")
    model = SAC.load(str(resolve_model(args.model)), device="cpu")
    # Offscreen RGB rendering still needs a working GLFW/OpenGL context on macOS.
    env = make_env(render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=3000)
        with imageio.get_writer(output, fps=round(1 / env.unwrapped.dt), codec="libx264", macro_block_size=16) as video:
            for _ in range(args.frames):
                frame = env.render()
                if frame is None:
                    raise RuntimeError("Renderer returned no frame. See docs/TROUBLESHOOTING.md")
                video.append_data(frame)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
    finally:
        env.close()
    print(f"Saved video: {output}")


if __name__ == "__main__":
    main()
