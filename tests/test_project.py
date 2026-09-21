"""Fast tests for helper contracts. Requires only Python standard library."""
import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src import common
from src.train import positive_int


class ProjectTests(unittest.TestCase):
    def test_ant_name_and_paths(self):
        self.assertEqual(common.ENV_ID, "Ant-v5")
        self.assertEqual(common.ROOT, Path(__file__).resolve().parents[1])

    def test_positive_steps(self):
        self.assertEqual(positive_int("2048"), 2048)
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_model_resolution_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            best = root / "best.zip"
            latest = root / "latest.zip"
            latest.write_bytes(b"test")
            with patch.object(common, "BEST_MODEL", best), patch.object(common, "FINAL_MODEL", latest):
                self.assertEqual(common.resolve_model(), latest)
                best.write_bytes(b"test")
                self.assertEqual(common.resolve_model(), best)


if __name__ == "__main__":
    unittest.main()
