"""Initial validation tests for GridGuard AI configuration and environment."""

import importlib
from pathlib import Path
import unittest

from config.settings import Settings, get_settings


class TestProjectInitialization(unittest.TestCase):
    """Verifies baseline project initialization, path integrity, and module discovery."""

    def setUp(self) -> None:
        """Load fresh settings for each test."""
        self.settings = get_settings()

    def test_settings_instance(self) -> None:
        """Verify settings singleton is properly instantiated with expected metadata."""
        self.assertIsInstance(self.settings, Settings)
        self.assertEqual(self.settings.app_name, "GridGuard AI")
        self.assertTrue(len(self.settings.app_version) > 0)
        self.assertIn("Theft Detection", self.settings.app_tagline)

    def test_core_directories_exist(self) -> None:
        """Verify that all core project directories exist on the filesystem."""
        expected_dirs = [
            self.settings.base_dir,
            self.settings.data_dir,
            self.settings.raw_data_dir,
            self.settings.processed_data_dir,
            self.settings.synthetic_data_dir,
            self.settings.models_dir,
            self.settings.reports_dir,
            self.settings.logs_dir,
            self.settings.database_dir,
            self.settings.src_dir,
            self.settings.tests_dir,
        ]
        for directory in expected_dirs:
            with self.subTest(directory=directory):
                self.assertIsInstance(directory, Path)
                self.assertTrue(
                    directory.exists(),
                    f"Required project directory does not exist: {directory}",
                )
                self.assertTrue(
                    directory.is_dir(),
                    f"Path is not a directory: {directory}",
                )

    def test_database_path_configuration(self) -> None:
        """Verify the SQLite database target path is rooted under the database directory."""
        self.assertIsInstance(self.settings.sqlite_db_path, Path)
        self.assertEqual(
            self.settings.sqlite_db_path.parent,
            self.settings.database_dir,
        )

    def test_source_packages_importable(self) -> None:
        """Verify all planned architectural modules are importable packages."""
        submodules = [
            "src",
            "src.data",
            "src.features",
            "src.detection",
            "src.classification",
            "src.explainability",
            "src.energy_accounting",
            "src.simulator",
            "src.services",
        ]
        for mod_name in submodules:
            with self.subTest(module=mod_name):
                module = importlib.import_module(mod_name)
                self.assertIsNotNone(
                    module, f"Failed to import package: {mod_name}"
                )

    def test_app_entrypoint_exists(self) -> None:
        """Verify app.py exists at the workspace root and has non-zero size."""
        app_file = self.settings.base_dir / "app.py"
        self.assertTrue(
            app_file.exists(), "app.py entrypoint file does not exist."
        )
        self.assertGreater(
            app_file.stat().st_size,
            0,
            "app.py entrypoint file is empty.",
        )


# Pytest test function bindings for native pytest CLI compatibility
def test_pytest_settings_instance():
    settings = get_settings()
    assert settings.app_name == "GridGuard AI"
    assert settings.environment in ["development", "testing", "production"]


def test_pytest_directories_exist():
    settings = get_settings()
    assert settings.base_dir.is_dir()
    assert settings.src_dir.is_dir()
    assert settings.data_dir.is_dir()


if __name__ == "__main__":
    unittest.main()

