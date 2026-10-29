"""Dependency-free tests for checkpoint integrity and configuration contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import checkpoint, common, experiment


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


    def test_fresh_ignores_empty_scaffolding_and_archives_all_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            models, logs, videos = (root / name for name in ("models", "logs", "videos"))
            with patch.object(experiment, "MODELS", models), \
                 patch.object(experiment, "LOGS", logs), \
                 patch.object(experiment, "VIDEOS", videos):
                for folder in (models / "best", models / "checkpoints", logs, videos):
                    folder.mkdir(parents=True)
                    (folder / ".gitkeep").touch()
                self.assertFalse(experiment.has_active_artifacts())
                self.assertIsNone(experiment.archive_active_artifacts())

                outputs = {
                    models / "latest.json": b'{"snapshot":"snapshot-test"}',
                    models / "snapshots" / "snapshot-test" / "model.zip": b"model",
                    models / "snapshots" / "snapshot-test" / "replay_buffer.pkl": b"buffer",
                    models / "checkpoints" / "ant_sac_260_steps.zip": b"periodic",
                    models / "best" / "best_model.zip": b"best",
                    models / "best" / "score.json": b"score",
                    logs / "train_monitor.monitor.csv": b"r,l,t\\n",
                    logs / "eval_history" / "old" / "evaluations.npz": b"npz",
                    videos / "ant_demo.mp4": b"video",
                }
                for path, content in outputs.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)

                archive = experiment.archive_active_artifacts()
                self.assertIsNotNone(archive)
                for original, expected in outputs.items():
                    self.assertFalse(original.exists(), str(original))
                    self.assertEqual(
                        (archive / original.relative_to(root)).read_bytes(),
                        expected,
                    )
                # A retained archive alone must not block a new experiment.
                self.assertFalse(experiment.has_active_artifacts())
                self.assertIsNone(experiment.archive_active_artifacts())


if __name__ == "__main__":
    unittest.main()
