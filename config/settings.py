"""Application settings and path configurations for GridGuard AI.

Provides centralized path management, environment variable loading, and
configuration defaults for local development, analytical pipelines, and future
backend services.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

# Attempt to load environment variables from a local .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Root Directory of the Project
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Directory Hierarchy
CONFIG_DIR: Path = BASE_DIR / "config"
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR: Path = DATA_DIR / "synthetic"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
LOGS_DIR: Path = BASE_DIR / "logs"
DATABASE_DIR: Path = BASE_DIR / "database"
SRC_DIR: Path = BASE_DIR / "src"
TESTS_DIR: Path = BASE_DIR / "tests"

ALL_PROJECT_DIRS: List[Path] = [
    CONFIG_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    DATABASE_DIR,
    SRC_DIR,
    TESTS_DIR,
]


def ensure_directories() -> None:
    """Ensure all core project directories exist on disk."""
    for directory in ALL_PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Immutable project settings and runtime configuration."""

    # Application Metadata
    app_name: str = "GridGuard AI"
    app_version: str = "0.1.0-alpha"
    app_tagline: str = (
        "Intelligent Electricity Theft Detection and Smart Grid Anomaly Monitoring"
    )

    # Runtime Environment
    environment: str = os.getenv("GRIDGUARD_ENV", "development")
    log_level: str = os.getenv("GRIDGUARD_LOG_LEVEL", "INFO")

    # Host & Port Defaults
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8501"))

    # Core Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    synthetic_data_dir: Path = SYNTHETIC_DATA_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR
    logs_dir: Path = LOGS_DIR
    database_dir: Path = DATABASE_DIR
    src_dir: Path = SRC_DIR
    tests_dir: Path = TESTS_DIR

    # Database
    sqlite_db_path: Path = BASE_DIR / os.getenv(
        "GRIDGUARD_DB_PATH", "database/gridguard.db"
    )

    # Virtual Simulator Defaults (Phase 2+)
    simulator_num_meters: int = int(os.getenv("SIMULATOR_NUM_METERS", "50"))
    simulator_interval_sec: int = int(
        os.getenv("SIMULATOR_REPORTING_INTERVAL_SEC", "15")
    )
    simulator_feeder_id: str = os.getenv(
        "SIMULATOR_DEFAULT_FEEDER_ID", "FEEDER_NORTH_01"
    )

    # MQTT Settings (Future Phase)
    mqtt_broker_host: str = os.getenv("MQTT_BROKER_HOST", "localhost")
    mqtt_broker_port: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    mqtt_topic_telemetry: str = os.getenv(
        "MQTT_TOPIC_TELEMETRY", "gridguard/telemetry/meters"
    )
    mqtt_client_id: str = os.getenv(
        "MQTT_CLIENT_ID", "gridguard-stream-agent"
    )


# Singleton instance
_settings: Settings = Settings()


def get_settings() -> Settings:
    """Return the global application settings instance."""
    return _settings

