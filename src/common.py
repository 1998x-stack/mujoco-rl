"""Shared paths, environment creation, and model loading."""
from __future__ import annotations

from pathlib import Path

ENV_ID = "Ant-v5"
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
VIDEOS = ROOT / "videos"
FINAL_MODEL = MODELS / "ant_sac_latest.zip"
REPLAY_BUFFER = MODELS / "ant_sac_replay_buffer.pkl"
BEST_MODEL = MODELS / "best" / "best_model.zip"


def ensure_dirs() -> None:
    for path in (MODELS, LOGS, VIDEOS, MODELS / "best", MODELS / "checkpoints"):
        path.mkdir(parents=True, exist_ok=True)


def make_env(*, render_mode: str | None = None, seed: int | None = None, monitor_file: Path | None = None):
    import gymnasium as gym
    env = gym.make(ENV_ID, render_mode=render_mode)
    if monitor_file is not None:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env, filename=str(monitor_file))
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def resolve_model(model: str | None = None) -> Path:
    """Prefer evaluated best model, falling back to latest snapshot or legacy model."""
    if model:
        candidate = Path(model).expanduser()
    elif BEST_MODEL.is_file():
        candidate = BEST_MODEL
    else:
        from .checkpoint import latest_model_path
        candidate = latest_model_path() or FINAL_MODEL
    if not candidate.is_file():
        raise FileNotFoundError(f"No model at {candidate}. Run ./run.sh first, or supply --model PATH.")
    return candidate
