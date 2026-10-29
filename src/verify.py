"""Independent audit of snapshot integrity and evaluation provenance.

This validates local consistency, not cryptographic signing, policy quality,
or safety for real-world robotic deployment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

from .checkpoint import load_checkpoint_paths
from .common import ENV_ID, LOGS, MODELS
from .provenance import comparable_digest, read_manifest
from .quality import validate_quality


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(*, require_evaluation: bool = False) -> dict:
    manifest = read_manifest()
    if manifest is None:
        raise ValueError("Missing experiment.json: legacy training cannot prove its configuration")
    paths = load_checkpoint_paths()
    if paths is None:
        raise ValueError("Missing complete model/replay checkpoint")
    model, replay, checkpoint = paths
    contract = manifest["contract"]
    if checkpoint["environment"] != ENV_ID or checkpoint["seed"] != contract["seed"]:
        raise ValueError("Checkpoint and experiment manifest disagree on seed or environment")
    evaluation_file = LOGS / "evaluation.json"
    result = {
        "schema": 1,
        "status": "PASS",
        "environment": ENV_ID,
        "training_seed": contract["seed"],
        "training_budget_timesteps": checkpoint["timesteps"],
        "configuration_fingerprint": comparable_digest(contract),
        "model_sha256": file_hash(model),
        "replay_buffer_sha256": file_hash(replay),
        "evaluation_verified": False,
    }
    if require_evaluation and not evaluation_file.is_file():
        raise ValueError("Missing required evaluation.json")
    if evaluation_file.is_file():
        evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
        if evaluation.get("schema") not in (2, 3) or evaluation.get("environment") != ENV_ID:
            raise ValueError("Evaluation schema or environment mismatch")
        if evaluation.get("training_seed") != contract["seed"]:
            raise ValueError("Evaluation training seed differs from manifest")
        if evaluation.get("training_budget_timesteps") != checkpoint["timesteps"]:
            raise ValueError("Evaluation training budget differs from latest checkpoint")
        if evaluation.get("contract_fingerprint") != manifest["fingerprint"]:
            raise ValueError("Evaluation training contract fingerprint mismatch")
        if evaluation.get("configuration_fingerprint") != comparable_digest(contract):
            raise ValueError("Evaluation comparable configuration fingerprint mismatch")
        candidate = Path(evaluation.get("model", ""))
        if not candidate.resolve().is_relative_to(MODELS.resolve()):
            raise ValueError("Audit requires evaluated model inside the experiment's models directory")
        if not candidate.is_file() or evaluation.get("model_sha256") != file_hash(candidate):
            raise ValueError("Evaluated model missing or modified since evaluation")
        episodes = evaluation.get("episodes", [])
        if not episodes:
            raise ValueError("Evaluation contains no episodes")
        scores = [float(x["reward"]) for x in episodes]
        if not all(isfinite(x) for x in scores):
            raise ValueError("Evaluation has nonfinite reward")
        if not isclose(sum(scores) / len(scores), float(evaluation.get("mean_reward")),
                       rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError("Evaluation summary does not match episode rewards")
        if evaluation["schema"] == 3:
            quality = validate_quality(episodes, evaluation.get("quality"))
            if not isclose(quality["mean_forward_displacement_x"],
                           float(evaluation.get("mean_forward_displacement_x")),
                           rel_tol=1e-8, abs_tol=1e-7):
                raise ValueError("Evaluation reported displacement differs from episode measurements")
            expected_versions = {k: manifest["packages"].get(k)
                                 for k in ("gymnasium", "mujoco", "stable-baselines3")}
            if evaluation.get("environment_versions") != expected_versions:
                raise ValueError("Evaluation library versions differ from training manifest")
            result["trajectory_quality_verified"] = True
        result["evaluation_verified"] = True
        result["mean_reward"] = evaluation["mean_reward"]
        result["evaluation_episodes"] = len(episodes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MuJoCo-RL model, replay and evaluation provenance")
    parser.add_argument("--require-evaluation", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(require_evaluation=args.require_evaluation)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
