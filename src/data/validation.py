"""Validation rules and quality flagging for GridGuard AI consumption data.

Implements reusable validation checks for identifiers, timestamps, labels,
consumption values, streaks of missing measurements, and suspicious outliers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schemas import QualityFlag


def validate_consumer_id(id_val: Any) -> Tuple[bool, Optional[str]]:
    """Validate a consumer identifier string.

    Rules:
      - Must not be null or NaN.
      - Must not be an empty or whitespace string.
      - Must not contain obvious header/index artifacts.

    Returns:
        (is_valid, rejection_reason)
    """
    if id_val is None or pd.isna(id_val):
        return False, "MISSING_CONSUMER_ID: Value is null or NaN."

    s_val = str(id_val).strip()
    if not s_val or s_val.lower() in ("nan", "none", "null"):
        return False, "EMPTY_CONSUMER_ID: String is empty or literal null."

    return True, None


def validate_timestamp(ts_val: Any) -> Tuple[bool, Optional[str]]:
    """Validate a date/timestamp representation.

    Returns:
        (is_valid, rejection_reason)
    """
    if ts_val is None or pd.isna(ts_val):
        return False, "MISSING_TIMESTAMP: Date value is null."

    if isinstance(ts_val, (pd.Timestamp, datetime)):
        return True, None

    try:
        pd.to_datetime(ts_val)
        return True, None
    except Exception as exc:
        return False, f"INVALID_TIMESTAMP_FORMAT: {exc}"


def validate_label_value(
    label_val: Any,
    allowed_values: Optional[set] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate candidate target label value.

    Returns:
        (is_valid, rejection_reason)
    """
    if allowed_values is None:
        allowed_values = {0, 1, 0.0, 1.0, "0", "1"}

    if label_val is None or pd.isna(label_val):
        return False, "MISSING_LABEL: Label is null or NaN."

    if label_val in allowed_values:
        return True, None

    return False, f"UNEXPECTED_LABEL_VALUE: {label_val} not in {allowed_values}"


def assign_quality_flag(
    val: Optional[float],
    outlier_threshold: float = 1000.0,
) -> str:
    """Classify an individual consumption measurement into domain quality flags.

    Args:
        val: Consumption measurement.
        outlier_threshold: Extreme upper bound threshold for high outlier flag.

    Returns:
        Comma-delimited string of QualityFlag values.
    """
    if val is None or pd.isna(val):
        return QualityFlag.MISSING_VALUE.value

    try:
        fval = float(val)
    except (ValueError, TypeError):
        return "NON_NUMERIC_VALUE"

    flags: List[str] = []

    if fval < 0.0:
        flags.append(QualityFlag.NEGATIVE_READING.value)
    elif fval == 0.0:
        flags.append(QualityFlag.ZERO_READING.value)
    else:
        flags.append(QualityFlag.VALID.value)

    if fval > outlier_threshold:
        flags.append(QualityFlag.OUTLIER_HIGH.value)

    return ",".join(flags)


def calculate_missing_streaks(series: pd.Series) -> Dict[str, int]:
    """Calculate the longest streak of consecutive missing measurements in a time series.

    Args:
        series: Ordered pandas Series representing chronological consumption.

    Returns:
        Dictionary with max_missing_streak and current_missing_streak.
    """
    is_na = series.isna().astype(int)
    max_streak = 0
    current_streak = 0

    for val in is_na:
        if val == 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0

    return {
        "max_missing_streak": max_streak,
        "consecutive_missing_at_end": current_streak,
    }


def validate_consumer_profile(
    consumer_id: str,
    readings: pd.Series,
    label: Any,
) -> Dict[str, Any]:
    """Run comprehensive validation checks across a single consumer's full profile.

    Returns:
        Validation report dictionary with metrics and status.
    """
    id_valid, id_reason = validate_consumer_id(consumer_id)
    lbl_valid, lbl_reason = validate_label_value(label)

    total_count = len(readings)
    numeric_s = pd.to_numeric(readings, errors="coerce")
    missing_count = int(numeric_s.isna().sum())
    zero_count = int((numeric_s == 0.0).sum())
    negative_count = int((numeric_s < 0.0).sum())
    valid_count = total_count - missing_count

    streak_info = calculate_missing_streaks(numeric_s)

    is_quarantined = not id_valid or not lbl_valid
    quarantine_reason = None
    if not id_valid:
        quarantine_reason = id_reason
    elif not lbl_valid:
        quarantine_reason = lbl_reason

    return {
        "consumer_id": str(consumer_id),
        "is_quarantined": is_quarantined,
        "quarantine_reason": quarantine_reason,
        "total_days": total_count,
        "valid_days": valid_count,
        "missing_days": missing_count,
        "zero_days": zero_count,
        "negative_days": negative_count,
        "max_missing_streak": streak_info["max_missing_streak"],
    }
