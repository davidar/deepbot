"""Time-related utility functions."""

from datetime import datetime as py_datetime
from typing import Any, Optional, Union

import pendulum
from pendulum import Date, DateTime, Duration, Time
from pendulum.datetime import DateTime as PendulumDateTime


def to_pendulum(dt: Union[DateTime, py_datetime, None]) -> DateTime | None:
    """Convert a datetime to a pendulum DateTime.

    Args:
        dt: datetime object to convert, or None

    Returns:
        Pendulum DateTime object, or None if input is None
    """
    if dt is None:
        return None
    if isinstance(dt, DateTime):
        return dt
    return pendulum.instance(dt)


def ensure_datetime(dt: Union[Any, py_datetime]) -> PendulumDateTime:
    """Convert any pendulum date/time object or python datetime to a DateTime.

    Args:
        dt: The object to convert

    Returns:
        A pendulum DateTime object in UTC timezone

    Raises:
        ValueError: If the input cannot be converted to a DateTime
    """
    if isinstance(dt, PendulumDateTime):
        return dt.in_timezone("UTC")
    if isinstance(dt, py_datetime):
        return pendulum.instance(dt).in_timezone("UTC")
    # Convert Date/Time to DateTime
    now = pendulum.now("UTC")
    if isinstance(dt, pendulum.Date):
        dt = pendulum.datetime(dt.year, dt.month, dt.day, tz="UTC")
    elif isinstance(dt, pendulum.Time):
        dt = pendulum.datetime(
            now.year,
            now.month,
            now.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tz="UTC",
        )
    if not isinstance(dt, PendulumDateTime):
        raise ValueError(f"Cannot convert {type(dt)} to DateTime")
    return dt.in_timezone("UTC")


def parse_datetime(timestamp_str: str) -> PendulumDateTime:
    """Parse a timestamp string into a DateTime object.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        A pendulum DateTime object in UTC timezone

    Raises:
        ValueError: If the timestamp cannot be parsed
    """
    try:
        dt = pendulum.parse(timestamp_str)
        return ensure_datetime(dt)
    except Exception as e:
        raise ValueError(f"Failed to parse timestamp {timestamp_str}: {str(e)}")


def format_timestamp(
    dt: Optional[Union[DateTime, Date, Time, Duration]],
) -> Optional[str]:
    """Format a datetime into a consistent ISO format with Z timezone.

    Args:
        dt: pendulum DateTime object to format, or None

    Returns:
        ISO format string with Z timezone, or None if input is None
    """
    if dt is None:
        return None
    # Convert to UTC, format with 3 decimal places for microseconds, and use Z suffix
    if isinstance(dt, DateTime):
        return dt.in_timezone("UTC").format("YYYY-MM-DDTHH:mm:ss.SSSZ")
    elif isinstance(dt, Date):
        # Convert Date to DateTime at midnight UTC
        dt = pendulum.datetime(dt.year, dt.month, dt.day, tz="UTC")
        return dt.format("YYYY-MM-DDTHH:mm:ss.SSSZ")
    elif isinstance(dt, Time):
        # Convert Time to DateTime today at that time
        now = pendulum.now("UTC")
        dt = pendulum.datetime(
            now.year,
            now.month,
            now.day,
            dt.hour,
            dt.minute,
            dt.second,
            dt.microsecond,
            tz="UTC",
        )
        return dt.format("YYYY-MM-DDTHH:mm:ss.SSSZ")
    else:
        # Duration - convert to total seconds as string
        return str(dt.total_seconds())


def format_relative_time(timestamp_str: str) -> str:
    """Format a timestamp into a human-readable relative time.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        Human-readable relative time string
    """
    try:
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
    except ValueError:
        return "unknown time"


def parse_relative_time(time_str: str) -> Optional[DateTime]:
    """Parse a relative time string (e.g., 5m, 2h).

    Args:
        time_str: The time string to parse (e.g., "5m", "2h")

    Returns:
        A pendulum DateTime object or None if parsing fails
    """
    if not time_str.endswith(("s", "m", "h", "d")):
        return None

    try:
        value = int(time_str[:-1])
        unit = time_str[-1]
        now = pendulum.now("UTC")

        if unit == "s":
            return now.add(seconds=value)
        elif unit == "m":
            return now.add(minutes=value)
        elif unit == "h":
            return now.add(hours=value)
        elif unit == "d":
            return now.add(days=value)
    except ValueError:
        return None

    return None


def parse_absolute_time(time_str: str) -> Optional[DateTime]:
    """Parse an absolute time string.

    Args:
        time_str: The time string to parse

    Returns:
        A pendulum DateTime object or None if parsing fails
    """
    now = pendulum.now("UTC")

    # Try ISO format first
    try:
        dt = pendulum.parse(time_str)
        return ensure_datetime(dt)
    except ValueError:
        pass

    # Try common formats
    formats = [
        "YYYY-MM-DD HH:mm",
        "YYYY-MM-DD HH:mm:ss",
        "MM/DD/YYYY HH:mm",
        "DD/MM/YYYY HH:mm",
        "HH:mm",  # Today at the specified time
    ]

    for fmt in formats:
        try:
            parsed_time = pendulum.from_format(time_str, fmt)

            # If only time was provided, set the date to today
            if fmt == "HH:mm":
                parsed_time = parsed_time.set(
                    year=now.year, month=now.month, day=now.day
                )

                # If the time has already passed today, set it to tomorrow
                if parsed_time < now:
                    parsed_time = parsed_time.add(days=1)

            return ensure_datetime(parsed_time)
        except ValueError:
            continue

    return None


def parse_time_string(time_str: str) -> Optional[DateTime]:
    """Parse a time string into a datetime object.

    Args:
        time_str: The time string to parse

    Returns:
        A pendulum DateTime object or None if parsing fails
    """
    # Try parsing as relative time first
    result = parse_relative_time(time_str)
    if result:
        return result

    # Try parsing as absolute time
    result = parse_absolute_time(time_str)
    if result:
        return result

    return None
