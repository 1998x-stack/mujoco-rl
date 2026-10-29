"""Statistical benchmark and provenance contracts: no MuJoCo/PyTorch dependency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import benchmark, provenance, sweep


def fake_report(path: Path, seed: int, rewards=(2.0, 4.0), *,
                eval_seed: int = 1000, device: str = "cpu") -> None:
    contract = provenance.training_contract(seed=seed, device=device)
    data = {
        "schema": 2,
        "environment": "Ant-v5",
        "model": str(path.with_suffix(".zip")),
        "model_sha256": "a" * 64,
        "contract_fingerprint": provenance.contract_digest(contract),
        "configuration_fingerprint": provenance.comparable_digest(contract),
        "training_seed": seed,
        "evaluation_seed": eval_seed,
        "episodes": [{"reward": reward} for reward in rewards],
        "mean_reward": sum(rewards) / len(rewards),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ProvenanceTests(unittest.TestCase):
    def test_resume_contract_and_fingerprint(self):
        with TemporaryDirectory() as temp, patch.object(provenance, "MANIFEST", Path(temp) / "manifest.json"):
            self.assertFalse(provenance.require_compatible_resume(seed=42))
            value = provenance.create_manifest(seed=42)
            self.assertEqual(value["fingerprint"], provenance.contract_digest(value["contract"]))
            self.assertTrue(provenance.require_compatible_resume(seed=42))
            with self.assertRaisesRegex(ValueError, "Resume contract differs"):
                provenance.require_compatible_resume(seed=43)
            with self.assertRaisesRegex(ValueError, "Resume contract differs"):
                provenance.require_compatible_resume(seed=42, device="cuda")
            with self.assertRaises(FileExistsError):
                provenance.create_manifest(seed=42)

    def test_seed_invariant_and_tamper_evidence(self):
        a = provenance.training_contract(seed=0)
        b = provenance.training_contract(seed=1)
        self.assertNotEqual(provenance.contract_digest(a), provenance.contract_digest(b))
        self.assertEqual(provenance.comparable_digest(a), provenance.comparable_digest(b))
        c = provenance.training_contract(seed=1, device="cuda")
        self.assertNotEqual(provenance.comparable_digest(a), provenance.comparable_digest(c))
        with TemporaryDirectory() as temp:
            path = Path(temp) / "experiment.json"
            path.write_text(json.dumps({"schema": 1, "contract": a, "fingerprint": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                provenance.read_manifest(path)


class BenchmarkTests(unittest.TestCase):
    def test_seed_level_bootstrap_is_deterministic(self):
        with TemporaryDirectory() as temp:
            files = [Path(temp) / f"{s}.json" for s in (1, 2, 3)]
            for i, path in enumerate(files):
                fake_report(path, i + 1, rewards=(float(i), float(i + 2)))
            a = benchmark.aggregate(files, bootstrap_samples=200, bootstrap_seed=7)
            b = benchmark.aggregate(list(reversed(files)), bootstrap_samples=200, bootstrap_seed=7)
            self.assertEqual(a["mean_reward"], 2.0)
            self.assertEqual(a["number_of_training_seeds"], 3)
            self.assertEqual(a["mean_reward"], b["mean_reward"])
            self.assertEqual(a["confidence_interval_95"], benchmark.aggregate(files, bootstrap_samples=200, bootstrap_seed=7)["confidence_interval_95"])
            self.assertLessEqual(a["confidence_interval_95"][0], 2.0)
            self.assertGreaterEqual(a["confidence_interval_95"][1], 2.0)

    def test_reject_duplicate_seed_and_protocol_mismatch(self):
        with TemporaryDirectory() as temp:
            a, b = Path(temp) / "a.json", Path(temp) / "b.json"
            fake_report(a, 1)
            fake_report(b, 1)
            with self.assertRaisesRegex(ValueError, "Duplicate training seed"):
                benchmark.aggregate([a, b], bootstrap_samples=100)
            fake_report(b, 2, eval_seed=1001)
            with self.assertRaisesRegex(ValueError, "Incompatible evaluation protocols"):
                benchmark.aggregate([a, b], bootstrap_samples=100)
            fake_report(b, 2, device="cuda")
            with self.assertRaisesRegex(ValueError, "Training configuration differs"):
                benchmark.aggregate([a, b], bootstrap_samples=100)

    def test_reject_untracked_or_inflated_scores(self):
        with TemporaryDirectory() as temp:
            a, b = Path(temp) / "a.json", Path(temp) / "b.json"
            fake_report(a, 1)
            fake_report(b, 2)
            report = json.loads(b.read_text())
            report["mean_reward"] = 9999
            b.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Reported mean disagrees"):
                benchmark.aggregate([a, b], bootstrap_samples=100)
            fake_report(b, 2)
            report = json.loads(b.read_text())
            report["training_seed"] = None
            b.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Untracked training seed"):
                benchmark.aggregate([a, b], bootstrap_samples=100)


class SuiteTests(unittest.TestCase):
    def test_parse_and_slug_validation(self):
        self.assertEqual(sweep.parse_seeds("0, 2,3"), [0, 2, 3])
        for value in ("", "0,0", "-1,2", "a,2", "1,4294967296"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                sweep.parse_seeds(value)
        for value in ("../escape", "a/b", "..", "."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sweep.validate_name(value)

    def test_refuse_existing_runs_before_subprocess(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "run" / "seed-1" / "models"
            root.mkdir(parents=True)
            (root / "experiment.json").write_text("{}")
            with patch.object(sweep.subprocess, "run") as caller:
                with self.assertRaises(FileExistsError):
                    sweep.execute_suite(name="run", seeds=[1, 2], steps=1,
                                        episodes=1, base_dir=Path(temp))
                caller.assert_not_called()

    def test_isolated_subprocess_invocations(self):
        with TemporaryDirectory() as temp:
            with patch.object(sweep.subprocess, "run") as caller, \
                 patch.object(sweep, "aggregate", return_value={"schema": 1}) as agg:
                result = sweep.execute_suite(name="safe", seeds=[11, 22], steps=260,
                                             episodes=2, base_dir=Path(temp))
                self.assertTrue(result.exists())
                self.assertEqual(caller.call_count, 4)
                self.assertTrue(caller.call_args_list[0].kwargs["check"])
                roots = [x.kwargs["env"]["MUJOCO_RL_OUTPUT_ROOT"] for x in caller.call_args_list]
                self.assertEqual(roots[0], roots[1])
                self.assertEqual(roots[2], roots[3])
                self.assertNotEqual(roots[0], roots[2])
                self.assertEqual(agg.call_count, 1)


if __name__ == "__main__":
    unittest.main()
