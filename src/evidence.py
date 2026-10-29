"""Portable evidence-bundle verifier for GitHub Actions benchmark artifacts.

Only a policy ZIP, evaluation, provenance and audit summary are uploaded.
Replay buffers remain runner-local; the bundle is for inspection, not resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .benchmark import load_report
from .provenance import comparable_digest, read_manifest


def policy_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(folder: Path) -> dict:
    folder = folder.resolve()
    report_path = folder / "evaluation.json"
    metadata = load_report(report_path)
    if metadata["evaluation_schema"] != 3:
        raise ValueError("Official evidence requires trajectory-aware evaluation schema 3")
    manifest = read_manifest(folder / "experiment.json")
    if manifest is None:
        raise ValueError("Missing training manifest")
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    if (audit.get("status") != "PASS" or audit.get("evaluation_verified") is not True
            or audit.get("trajectory_quality_verified") is not True):
        raise ValueError("Training artifact audit has not passed all quality checks")
    if (manifest["contract"]["seed"] != metadata["training_seed"]
            or manifest["fingerprint"] != metadata["contract_fingerprint"]
            or comparable_digest(manifest["contract"]) != metadata["configuration_fingerprint"]
            or audit.get("configuration_fingerprint") != metadata["configuration_fingerprint"]
            or audit.get("training_seed") != metadata["training_seed"]
            or audit.get("training_budget_timesteps") != metadata["training_budget_timesteps"]):
        raise ValueError("Provenance, audit and evaluation disagree")
    evaluation = json.loads(report_path.read_text(encoding="utf-8"))
    expected_versions = {key: manifest["packages"].get(key)
                         for key in ("gymnasium", "mujoco", "stable-baselines3")}
    if evaluation.get("environment_versions") != expected_versions:
        raise ValueError("Evaluation runtime versions disagree with training manifest")
    if policy_digest(folder / "policy.zip") != metadata["model_sha256"]:
        raise ValueError("Policy ZIP checksum does not match evaluated model")
    return {
        "seed": metadata["training_seed"],
        "verified": True,
        "policy_sha256": metadata["model_sha256"],
        "configuration_fingerprint": metadata["configuration_fingerprint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify portable Ant-v5 benchmark evidence bundles")
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        summaries = [verify_evidence(folder) for folder in args.folders]
        if len({item["seed"] for item in summaries}) != len(summaries):
            raise ValueError("Duplicated training seeds in evidence set")
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"evidence_bundles": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
