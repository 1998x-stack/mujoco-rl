"""Isolation for repeated MuJoCo training experiments.

Archive *all* active output trees before a fresh run, rather than moving only
the published latest model. Otherwise old immutable snapshots, periodic
checkpoints, evaluation logs, or videos can be mistaken for the new experiment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

from .common import LOGS, MODELS, VIDEOS


def _meaningful(path: Path) -> bool:
    """Ignore empty scaffolding and placeholder files, including nested ones."""
    if path.name == ".gitkeep":
        return False
    if path.is_dir():
        return any(_meaningful(child) for child in path.iterdir())
    return path.exists() or path.is_symlink()


def active_artifacts() -> list[tuple[str, Path]]:
    """Return every live experiment output, excluding historical archives."""
    active: list[tuple[str, Path]] = []
    for label, folder in (("models", MODELS), ("logs", LOGS), ("videos", VIDEOS)):
        if not folder.exists():
            continue
        for child in folder.iterdir():
            if label == "models" and child.name == "archive":
                continue
            if _meaningful(child):
                active.append((label, child))
    return active


def has_active_artifacts() -> bool:
    return bool(active_artifacts())


def archive_active_artifacts() -> Path | None:
    """Move existing experiment outputs under a unique, retained archive.

    Archive creation is not a filesystem transaction: if moving a file fails,
    stop immediately and preserve the unmoved files for manual recovery.
    """
    artifacts = active_artifacts()
    if not artifacts:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = MODELS / "archive" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    for label, path in artifacts:
        target = destination / label / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    return destination
