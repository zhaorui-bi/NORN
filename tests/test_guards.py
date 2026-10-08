"""Public CLI validation works independently of machine-specific protocols."""

import subprocess
import sys
import unittest

from norn_earth import __version__


class TestCLI(unittest.TestCase):
    def test_training_requires_explicit_configuration(self):
        result = subprocess.run(
            [sys.executable, "-m", "norn_earth", "train"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--config", result.stderr)

    def test_evaluation_requires_checkpoint_dataset_and_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "norn_earth", "evaluate"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--checkpoint", result.stderr)

    def test_export_requires_checkpoint(self):
        result = subprocess.run(
            [sys.executable, "-m", "norn_earth", "infer"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--checkpoint", result.stderr)

    def test_help_and_version_without_training(self):
        result = subprocess.run(
            [sys.executable, "-m", "norn_earth", "--version"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(__version__, result.stdout)

    def test_preparation_requires_configuration_and_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "norn_earth", "prepare"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--config", result.stderr)
        self.assertIn("--output", result.stderr)
