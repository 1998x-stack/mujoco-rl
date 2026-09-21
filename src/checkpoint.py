"""Crash-consistent SAC checkpoint snapshots with an atomic latest pointer.

A pointer is published only after both model and replay buffer have been written.
Existing snapshots remain intact on unsuccessful saves. This provides process-level
consistency, not bitwise-exact restart of the environment or random-number state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .common import ENV_ID, MODELS

SNAPSHOTS = MODELS / "snapshots"
LATEST = MODELS / "latest.json"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(destination: Path, contents: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(contents, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(model, *, seed: int) -> Path:
    """Write a complete immutable snapshot, then atomically publish its name."""
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    snapshot_name = "snapshot-" + uuid.uuid4().hex
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=SNAPSHOTS))
    destination = SNAPSHOTS / snapshot_name
    try:
        model.save(str(temporary / "model.zip"))
        model.save_replay_buffer(str(temporary / "replay_buffer.pkl"))
        artifacts = {file: _hash(temporary / file) for file in ("model.zip", "replay_buffer.pkl")}
        _atomic_json(temporary / "metadata.json", {
            "schema": 1, "environment": ENV_ID, "seed": seed,
            "timesteps": int(model.num_timesteps), "sha256": artifacts,
        })
        os.replace(temporary, destination)
        # The previously published pointer remains valid if a new save fails.
        _atomic_json(LATEST, {"schema": 1, "snapshot": snapshot_name})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint_paths() -> tuple[Path, Path, dict] | None:
    """Return validated model and replay paths; None for pre-snapshot repositories."""
    if not LATEST.is_file():
        return None
    pointer = json.loads(LATEST.read_text(encoding="utf-8"))
    name = pointer.get("snapshot", "")
    if pointer.get("schema") != 1 or not isinstance(name, str) or not name.startswith("snapshot-") or len(name) != 41 or any(c not in "0123456789abcdef" for c in name[9:]):
        raise ValueError(f"Invalid checkpoint pointer: {LATEST}")
    directory = SNAPSHOTS / name
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != 1 or metadata.get("environment") != ENV_ID:
        raise ValueError(f"Incompatible checkpoint metadata: {directory}")
    model, buffer = directory / "model.zip", directory / "replay_buffer.pkl"
    for path in (model, buffer):
        if _hash(path) != metadata["sha256"][path.name]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return model, buffer, metadata


def latest_model_path() -> Path | None:
    loaded = load_checkpoint_paths()
    return loaded[0] if loaded else None


def save_best_score(destination: Path, score: float) -> None:
    _atomic_json(destination, {"schema": 1, "best_mean_reward": float(score)})


def load_best_score(destination: Path) -> float | None:
    if not destination.is_file():
        return None
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError(f"Invalid best model score metadata: {destination}")
    return float(payload["best_mean_reward"])
