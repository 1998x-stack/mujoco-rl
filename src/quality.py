"""Objective locomotion quality metrics for Gymnasium Ant-v5.

Positive progress (>=1m net x per episode) is a descriptive proxy, not a
claim of skill, stable gait, real-world safety, or task 'success'.
"""
from __future__ import annotations

from math import isclose, isfinite
from statistics import mean

PROGRESS_THRESHOLD_METERS = 1.0

QUALITY_KEYS = (
    "mean_forward_displacement_x",
    "mean_forward_speed_x",
    "early_termination_rate",
    "time_limit_rate",
    "positive_progress_rate",
    "mean_episode_length_steps",
)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"Invalid finite numeric value for {label}")
    return float(value)


def summarize(episodes: list[dict], *, threshold: float = PROGRESS_THRESHOLD_METERS) -> dict:
    """Every quantity is computed from individual episodes, not summary claims."""
    if not episodes:
        raise ValueError("No evaluation episodes")
    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("Progress threshold must be finite and positive")
    displacements, speeds, lengths = [], [], []
    early, limit, progress = [], [], []
    for i, episode in enumerate(episodes):
        reward = _finite(episode.get("reward"), f"episode[{i}].reward")
        dx = _finite(episode.get("forward_displacement_x"), f"episode[{i}].dx")
        duration = _finite(episode.get("elapsed_simulation_seconds"), f"episode[{i}].duration")
        steps = episode.get("steps")
        terminated, truncated = episode.get("terminated"), episode.get("truncated")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"Invalid episode steps at index {i}")
        if duration <= 0 or not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise ValueError(f"Invalid timing or terminal flags at episode {i}")
        if not terminated and not truncated:
            raise ValueError(f"Incomplete episode at index {i}")
        if not isclose(_finite(episode.get("mean_forward_speed_x"), "speed"),
                       dx / duration, rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"Forward speed inconsistent with displacement/time at episode {i}")
        _ = reward
        displacements.append(dx)
        speeds.append(dx / duration)
        lengths.append(steps)
        early.append(int(terminated and not truncated))
        limit.append(int(truncated))
        progress.append(int(dx >= threshold))
    return {
        "progress_threshold_meters": threshold,
        "mean_forward_displacement_x": mean(displacements),
        "mean_forward_speed_x": mean(speeds),
        "early_termination_rate": mean(early),
        "time_limit_rate": mean(limit),
        "positive_progress_rate": mean(progress),
        "mean_episode_length_steps": mean(lengths),
    }


def validate_quality(episodes: list[dict], quality: object) -> dict:
    if not isinstance(quality, dict):
        raise ValueError("Missing trajectory quality summary")
    threshold = _finite(quality.get("progress_threshold_meters"), "progress threshold")
    derived = summarize(episodes, threshold=threshold)
    if not isclose(threshold, PROGRESS_THRESHOLD_METERS, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Unsupported trajectory progress threshold")
    for key, value in derived.items():
        claimed = _finite(quality.get(key), key)
        if not isclose(claimed, value, rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"Trajectory quality metric mismatch: {key}")
    return derived
