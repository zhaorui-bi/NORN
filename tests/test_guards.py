"""Public CLI validation works independently of machine-specific protocols."""

import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class TestCLI(unittest.TestCase):
    def test_training_requires_explicit_configuration(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "train.py")], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--config", result.stderr)

    def test_evaluation_requires_checkpoint_dataset_and_output(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "evaluate.py")], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--checkpoint", result.stderr)

    def test_export_requires_checkpoint(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "export.py")], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--checkpoint", result.stderr)

    def test_help_and_version_without_training(self):
        result = subprocess.run(
            [sys.executable, "-m", "norn_earth", "--version"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("0.2.0", result.stdout)

    def test_export_dat_writer(self):
        import importlib.util
        import tempfile
        import numpy as np

        spec = importlib.util.spec_from_file_location("export_mod", SCRIPTS / "export.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.dat"
            module.export_dat(path, np.full((180, 360), 12.5))
            values = np.loadtxt(path)
            self.assertEqual(values.shape, (64800, 3))
            np.testing.assert_allclose(values[0], [0.5, -89.5, 12.5])
