"""Data schemas and standard structures for GridGuard AI.

Defines schemas for raw data validation, standardized long-format consumption
records, consumer label metadata, and inspection/validation reports.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QualityFlag(str, Enum):
    """Data quality classification flags for individual consumption readings."""
    VALID = "VALID"
    ZERO_READING = "ZERO_READING"
    NEGATIVE_READING = "NEGATIVE_READING"
    MISSING_VALUE = "MISSING_VALUE"
    OUTLIER_HIGH = "OUTLIER_HIGH"


@dataclass(frozen=True)
class StandardizedConsumptionRecord:
    """Standardized schema for a single consumer-day consumption reading.
    
    Attributes:
        consumer_id: Unique identifier string for the consumer.
        timestamp: Measurement date formatted as YYYY-MM-DD string.
        consumption_value: Energy consumption measurement (None if missing).
        source_dataset: Name of the originating raw file.
        source_label: Original raw label value associated with the consumer.
        source_label_column: Name of the label column in the source file.
        data_quality_flags: Comma-delimited quality flags.
    """
    consumer_id: str
    timestamp: str
    consumption_value: Optional[float]
    source_dataset: str
    source_label: Optional[float]
    source_label_column: str
    data_quality_flags: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsumerLabelRecord:
    """Standardized metadata and label provenance for a single consumer.
    
    Attributes:
        consumer_id: Unique identifier for the consumer.
        source_dataset: Name of the source dataset.
        source_label_column: Original label column name (e.g., 'FLAG', 'CHK_STATE').
        source_label: Raw label value (0.0, 1.0, or None).
        total_days: Total calendar dates tracked in the source file.
        recorded_days: Count of days with non-null measurements.
        missing_days: Count of days with missing measurements.
        zero_days: Count of days with exactly zero consumption.
        negative_days: Count of days with negative consumption.
        mean_consumption: Mean of recorded consumption values.
        median_consumption: Median of recorded consumption values.
        max_consumption: Maximum recorded consumption value.
        min_consumption: Minimum recorded consumption value.
    """
    consumer_id: str
    source_dataset: str
    source_label_column: str
    source_label: Optional[float]
    total_days: int
    recorded_days: int
    missing_days: int
    zero_days: int
    negative_days: int
    mean_consumption: Optional[float] = None
    median_consumption: Optional[float] = None
    max_consumption: Optional[float] = None
    min_consumption: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetInspectionResult:
    """Comprehensive inspection metrics for a raw dataset."""
    dataset_name: str
    file_size_bytes: int
    num_rows: int
    num_columns: int
    id_column: str
    label_column: str
    num_unique_consumers: int
    num_missing_consumers: int
    num_duplicate_consumers: int
    num_duplicate_rows: int
    date_columns_count: int
    date_format_detected: str
    earliest_date: str
    latest_date: str
    expected_calendar_days: int
    missing_calendar_dates: List[str] = field(default_factory=list)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    missing_label_count: int = 0
    total_consumption_cells: int = 0
    missing_consumption_cells: int = 0
    missing_consumption_pct: float = 0.0
    zero_consumption_cells: int = 0
    zero_consumption_pct: float = 0.0
    negative_consumption_cells: int = 0
    negative_consumption_pct: float = 0.0
    min_consumption: float = 0.0
    max_consumption: float = 0.0
    mean_consumption: float = 0.0
    median_consumption: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    percentile_99_9: float = 0.0
    malformed_rows_detected: List[Dict[str, Any]] = field(default_factory=list)
    sampling_frequency_inferred: str = "Daily"
    sampling_frequency_limitations: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityReport:
    """Data quality and validation transformation report."""
    dataset_name: str
    raw_rows: int
    raw_columns: int
    retained_consumers: int
    excluded_rows: int
    quarantined_rows: int
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)
    quarantine_reasons: Dict[str, int] = field(default_factory=dict)
    missingness_before_pct: float = 0.0
    missingness_after_pct: float = 0.0
    negative_reading_count: int = 0
    zero_reading_count: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    calendar_coverage_pct: float = 0.0
    transformations_applied: List[str] = field(default_factory=list)
    assumptions_documented: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
