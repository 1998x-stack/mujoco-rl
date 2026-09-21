"""Dependency-free tests for checkpoint integrity and configuration contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import checkpoint, common


class DummyModel:
    num_timesteps = 123

    def __init__(self, fail_buffer=False):
        self.fail_buffer = fail_buffer

    def save(self, path):
        Path(path).write_bytes(b"model-data")

    def save_replay_buffer(self, path):
        if self.fail_buffer:
            raise RuntimeError("simulated interrupted replay buffer write")
        Path(path).write_bytes(b"replay-data")


class ProjectTests(unittest.TestCase):
    def test_environment_and_paths(self):
        self.assertEqual(common.ENV_ID, "Ant-v5")
        self.assertEqual(common.ROOT, Path(__file__).resolve().parents[1])

    def test_model_resolution_fallback(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            best, legacy = directory / "best.zip", directory / "legacy.zip"
            legacy.write_bytes(b"legacy")
            with patch.object(common, "BEST_MODEL", best), patch.object(common, "FINAL_MODEL", legacy):
                self.assertEqual(common.resolve_model(), legacy)
                best.write_bytes(b"best")
                self.assertEqual(common.resolve_model(), best)

    def test_checkpoint_roundtrip_checksums_and_corruption(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(checkpoint, "SNAPSHOTS", root / "snapshots"), patch.object(checkpoint, "LATEST", root / "latest.json"):
                folder = checkpoint.save_checkpoint(DummyModel(), seed=42)
                model, buffer, meta = checkpoint.load_checkpoint_paths()
                self.assertEqual(model, folder / "model.zip")
                self.assertEqual(buffer.read_bytes(), b"replay-data")
                self.assertEqual((meta["seed"], meta["timesteps"]), (42, 123))
                model.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    checkpoint.load_checkpoint_paths()

    def test_failed_save_keeps_previous_snapshot(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(checkpoint, "SNAPSHOTS", root / "snapshots"), patch.object(checkpoint, "LATEST", root / "latest.json"):
                first = checkpoint.save_checkpoint(DummyModel(), seed=42)
                pointer_before = checkpoint.LATEST.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                    checkpoint.save_checkpoint(DummyModel(fail_buffer=True), seed=42)
                self.assertEqual(checkpoint.LATEST.read_bytes(), pointer_before)
                self.assertEqual(checkpoint.load_checkpoint_paths()[0], first / "model.zip")
                self.assertFalse(list(checkpoint.SNAPSHOTS.glob(".pending-*")))

    def test_best_score_roundtrip(self):
        with TemporaryDirectory() as temp:
            score = Path(temp) / "best.json"
            self.assertIsNone(checkpoint.load_best_score(score))
            checkpoint.save_best_score(score, 123.5)
            self.assertEqual(checkpoint.load_best_score(score), 123.5)
            self.assertEqual(json.loads(score.read_text())["schema"], 1)

    def test_pointer_does_not_allow_parent_traversal(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pointer = root / "latest.json"
            pointer.write_text(json.dumps({"schema": 1, "snapshot": "../../escape"}))
            with patch.object(checkpoint, "SNAPSHOTS", root), patch.object(checkpoint, "LATEST", pointer):
                with self.assertRaisesRegex(ValueError, "Invalid checkpoint pointer"):
                    checkpoint.load_checkpoint_paths()


if __name__ == "__main__":
    unittest.main()
