"""Immutable experiment provenance and configuration compatibility checks.

A manifest records the *actual* training contract for new runs; missing legacy
manifests are reported as unverified rather than fabricated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from .checkpoint import _atomic_json
from .common import ENV_ID, MODELS, ROOT

MANIFEST = MODELS / "experiment.json"
SAC_HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 256,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "policy_kwargs": {"net_arch": [256, 256]},
}


def training_contract(*, seed: int, device: str = "cpu") -> dict:
    return {
        "environment": ENV_ID,
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "seed": seed,
        "device": device,
        "hyperparameters": SAC_HYPERPARAMETERS,
    }


def contract_digest(contract: dict) -> str:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def comparable_digest(contract: dict) -> str:
    """Fingerprint shared training configuration, excluding independent seed."""
    return contract_digest({key: value for key, value in contract.items() if key != "seed"})


def read_manifest(path: Path | None = None) -> dict | None:
    path = MANIFEST if path is None else path
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("contract"), dict):
        raise ValueError(f"Invalid experiment manifest: {path}")
    if data.get("fingerprint") != contract_digest(data["contract"]):
        raise ValueError(f"Experiment manifest fingerprint mismatch: {path}")
    return data


def require_compatible_resume(*, seed: int, device: str = "cpu") -> bool:
    """Refuse silent configuration drift, unless this is a legacy untracked run."""
    saved = read_manifest()
    if saved is None:
        return False
    expected = contract_digest(training_contract(seed=seed, device=device))
    if saved["fingerprint"] != expected:
        raise ValueError("Resume contract differs from experiment.json: use the original seed/device/config or --fresh")
    return True


def create_manifest(*, seed: int, device: str = "cpu") -> dict:
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment manifest: {MANIFEST}")
    contract = training_contract(seed=seed, device=device)
    packages = {}
    for name in ("gymnasium", "mujoco", "stable-baselines3", "torch", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        git_sha = None
    payload = {
        "schema": 1,
        "contract": contract,
        "fingerprint": contract_digest(contract),
        "configuration_fingerprint": comparable_digest(contract),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_sha": git_sha,
    }
    _atomic_json(MANIFEST, payload)
    return payload
