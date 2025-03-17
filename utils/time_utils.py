"""Time-related utility functions."""

from datetime import datetime as py_datetime
from typing import Optional

import dateparser
import pendulum
from pendulum import DateTime


def ensure_datetime(dt: py_datetime) -> DateTime:
    """Convert any pendulum date/time object or python datetime to a DateTime.

    Args:
        dt: The object to convert

    Returns:
        A pendulum DateTime object in UTC timezone

    Raises:
        ValueError: If the input cannot be converted to a DateTime
    """
    return pendulum.instance(dt).in_timezone("UTC")


def parse_datetime(timestamp_str: str) -> DateTime:
    """Parse a timestamp string into a DateTime object.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        A pendulum DateTime object in UTC timezone

    Raises:
        ValueError: If the timestamp cannot be parsed
    """
    dt = pendulum.parse(timestamp_str)
    if isinstance(dt, DateTime):
        return dt.in_timezone("UTC")
    raise ValueError(f"Failed to parse timestamp {timestamp_str}: {str(dt)}")


def format_timestamp(dt: DateTime) -> Optional[str]:
    """Format a datetime into a consistent ISO format with Z timezone.

    Args:
        dt: pendulum DateTime object to format

    Returns:
        ISO format string with Z timezone, or None if input is None
    """
    # Convert to UTC, format with 3 decimal places for microseconds, and use Z suffix
    return dt.in_timezone("UTC").format("YYYY-MM-DDTHH:mm:ss.SSSZ")


def format_relative_time(timestamp_str: str) -> str:
    """Format a timestamp into a human-readable relative time.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        Human-readable relative time string
    """
    dt = parse_datetime(timestamp_str)
    now = pendulum.now("UTC")
    delta = now - dt

    if delta.days > 365:
        years = delta.days // 365
        return f"{years}y ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months}mo ago"
    elif delta.days > 0:
        return f"{delta.days}d ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours}h ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes}m ago"
    else:
        return "just now"


def parse_time_string(time_str: str) -> Optional[DateTime]:
    """Parse a time string into a datetime object.

    Args:
        time_str: The time string to parse

    Returns:
        A pendulum DateTime object or None if parsing fails
    """
    if not time_str:
        return None

    parsed_dt = dateparser.parse(time_str)
    if parsed_dt is None:
        return None

    return pendulum.instance(parsed_dt).in_timezone("UTC")
