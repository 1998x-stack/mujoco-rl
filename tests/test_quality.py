"""Regression tests for quality-aware Ant benchmark without simulation dependencies."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src import benchmark, evidence, provenance, scorecard
from src.quality import summarize, validate_quality


def episodes(count: int = 10, *, displacement: float = 2.0, term: bool = False) -> list[dict]:
    return [{
        "episode": i + 1,
        "reward": float(100 + i),
        "steps": 250 if term else 1000,
        "forward_displacement_x": displacement,
        "elapsed_simulation_seconds": 12.5 if term else 50.0,
        "mean_forward_speed_x": displacement / (12.5 if term else 50.0),
        "terminated": term,
        "truncated": not term,
    } for i in range(count)]


def report(path: Path, seed: int, *, budget: int = 1_000_000,
           count: int = 10, displacement: float = 2.0) -> dict:
    contract = provenance.training_contract(seed=seed)
    items = episodes(count, displacement=displacement)
    payload = {
        "schema": 3, "environment": "Ant-v5",
        "model": f"/tmp/models/{seed}.zip",
        "model_sha256": "a" * 64,
        "contract_fingerprint": provenance.contract_digest(contract),
        "configuration_fingerprint": provenance.comparable_digest(contract),
        "training_seed": seed, "training_budget_timesteps": budget,
        "evaluation_seed": 1000,
        "evaluation_protocol": {
            "deterministic_policy": True,
            "episode_seeding": "evaluation_seed_plus_index",
            "progress_threshold_meters": 1.0,
            "physics_time_step_seconds": 0.05,
            "max_episode_steps": 1000,
        },
        "environment_versions": {
            "gymnasium": "1.2.0", "mujoco": "3.3.0", "stable-baselines3": "2.7.0",
        },
        "quality": summarize(items),
        "episodes": items,
        "mean_reward": sum(x["reward"] for x in items) / len(items),
        "mean_forward_displacement_x": sum(x["forward_displacement_x"] for x in items) / len(items),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class QualityMetricTests(unittest.TestCase):
    def test_physical_metrics_and_rate_definitions(self):
        values = episodes(2)
        values[1] = episodes(1, displacement=-2.0, term=True)[0]
        quality = summarize(values)
        self.assertEqual(quality["mean_forward_displacement_x"], 0.0)
        self.assertAlmostEqual(quality["mean_forward_speed_x"], (0.04 - 0.16) / 2)
        self.assertEqual(quality["early_termination_rate"], 0.5)
        self.assertEqual(quality["time_limit_rate"], 0.5)
        self.assertEqual(quality["positive_progress_rate"], 0.5)
        self.assertEqual(quality["mean_episode_length_steps"], 625)
        self.assertEqual(validate_quality(values, quality), quality)

    def test_reject_incomplete_and_inflated_trajectory_quality(self):
        items = episodes()
        items[0]["terminated"] = False
        items[0]["truncated"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete episode"):
            summarize(items)
        items = episodes()
        claimed = summarize(items)
        claimed["positive_progress_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            validate_quality(items, claimed)
        items[0]["mean_forward_speed_x"] = 100.0
        with self.assertRaisesRegex(ValueError, "Forward speed inconsistent"):
            summarize(items)


class PublicationProtocolTests(unittest.TestCase):
    def test_five_seed_publication_gate_and_intervals(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"seed-{seed}.json" for seed in range(5)]
            for seed, path in enumerate(paths):
                report(path, seed, displacement=float(seed - 1))
            result = benchmark.aggregate(paths, publication=True, min_seeds=5,
                                         bootstrap_samples=200, bootstrap_seed=123)
            self.assertTrue(result["publication_protocol_verified"])
            self.assertEqual(result["schema"], 2)
            self.assertEqual(result["number_of_training_seeds"], 5)
            displacement = result["quality_metrics"]["mean_forward_displacement_x"]
            self.assertAlmostEqual(displacement["mean"], 1.0)
            self.assertLessEqual(displacement["confidence_interval_95"][0], 1.0)
            self.assertGreaterEqual(displacement["confidence_interval_95"][1], 1.0)
            self.assertEqual(
                result["quality_metrics"],
                benchmark.aggregate(paths, publication=True, min_seeds=5,
                                    bootstrap_samples=200, bootstrap_seed=123)["quality_metrics"],
            )

    def test_publication_rejects_short_or_mixed_protocol(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{seed}.json" for seed in range(5)]
            for seed, path in enumerate(paths):
                report(path, seed)
            with self.assertRaisesRegex(ValueError, "Publication protocol"):
                benchmark.aggregate(paths[:4], publication=True)
            report(paths[0], 0, budget=260)
            with self.assertRaisesRegex(ValueError, "Unequal training budgets"):
                benchmark.aggregate(paths, publication=True)
            report(paths[0], 0)
            broken = json.loads(paths[1].read_text())
            broken["environment_versions"]["mujoco"] = "9.9.9"
            paths[1].write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "versions differ"):
                benchmark.aggregate(paths, publication=True)
            report(paths[1], 1)
            broken = json.loads(paths[1].read_text())
            broken["quality"]["mean_forward_displacement_x"] = 9000.0
            paths[1].write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "metric mismatch"):
                benchmark.aggregate(paths, publication=True)

    def test_publication_rejects_nine_episodes_even_five_seeds(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{seed}.json" for seed in range(5)]
            for seed, path in enumerate(paths):
                report(path, seed, count=9)
            with self.assertRaisesRegex(ValueError, "Publication protocol"):
                benchmark.aggregate(paths, publication=True)


class EvidenceBundleTests(unittest.TestCase):
    def test_offline_evidence_check_and_tampering(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            payload = report(folder / "evaluation.json", 4)
            contract = provenance.training_contract(seed=4)
            manifest = {
                "schema": 1,
                "contract": contract,
                "fingerprint": provenance.contract_digest(contract),
                "packages": {
                    "gymnasium": "1.2.0", "mujoco": "3.3.0",
                    "stable-baselines3": "2.7.0",
                },
            }
            (folder / "experiment.json").write_text(json.dumps(manifest))
            policy = folder / "policy.zip"
            policy.write_bytes(b"simulated-policy-binary")
            payload["model_sha256"] = evidence.policy_digest(policy)
            (folder / "evaluation.json").write_text(json.dumps(payload))
            (folder / "audit.json").write_text(json.dumps({
                "status": "PASS",
                "evaluation_verified": True,
                "trajectory_quality_verified": True,
                "training_seed": 4,
                "training_budget_timesteps": 1_000_000,
                "configuration_fingerprint": provenance.comparable_digest(contract),
            }))
            self.assertTrue(evidence.verify_evidence(folder)["verified"])
            policy.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                evidence.verify_evidence(folder)

    def test_scorecard_uses_measured_values_not_claimed_gait_success(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{seed}.json" for seed in range(5)]
            for seed, path in enumerate(paths):
                report(path, seed)
            aggregated = benchmark.aggregate(paths, publication=True, min_seeds=5, bootstrap_samples=100)
            markdown = scorecard.render_scorecard(aggregated)
            self.assertIn("publication protocol verified", markdown)
            self.assertIn("Early termination rate", markdown)
            self.assertIn("not a statistical guarantee", markdown)
            aggregated["publication_protocol_verified"] = False
            smoke = scorecard.render_scorecard(aggregated)
            self.assertIn("NOT A PUBLISHABLE LEARNING RESULT", smoke)


if __name__ == "__main__":
    unittest.main()
