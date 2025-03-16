"""Time range and gap tracking for message history."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

# Set up logging
logger = logging.getLogger("deepbot.time_tracking")


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware in UTC.

    Args:
        dt: datetime object, naive or aware

    Returns:
        datetime object with UTC timezone
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class TimeRange:
    """Represents a range of time with start and end points."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Ensure start and end are timezone-aware."""
        self.start = ensure_utc(self.start)
        self.end = ensure_utc(self.end)

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another range."""
        return self.start <= other.end and other.start <= self.end

    def merge(self, other: "TimeRange") -> "TimeRange":
        """Merge this range with another overlapping range."""
        if not self.overlaps(other):
            raise ValueError("Ranges must overlap to merge")
        return TimeRange(
            start=min(self.start, other.start), end=max(self.end, other.end)
        )


@dataclass
class ChannelMetadata:
    """Metadata for a channel including known ranges and gaps."""

    channel_id: str
    known_ranges: List[TimeRange]
    gaps: List[TimeRange]
    last_sync: datetime

    def add_known_range(self, new_range: TimeRange) -> None:
        """Add a new known range, merging with existing ranges if they overlap.

        Args:
            new_range: The new time range to add
        """
        # Find overlapping ranges
        overlapping = [r for r in self.known_ranges if r.overlaps(new_range)]

        if not overlapping:
            # No overlaps, just add the new range
            self.known_ranges.append(new_range)
        else:
            # Merge with overlapping ranges
            merged = new_range
            for r in overlapping:
                merged = merged.merge(r)
                self.known_ranges.remove(r)
            self.known_ranges.append(merged)

        # Sort ranges by start time
        self.known_ranges.sort(key=lambda r: r.start)

        # Update gaps
        self._update_gaps()

    def _update_gaps(self) -> None:
        """Update the gaps list based on known ranges."""
        if not self.known_ranges:
            return

        # Sort ranges by start time
        sorted_ranges = sorted(self.known_ranges, key=lambda r: r.start)

        # Find gaps between ranges
        self.gaps = []
        for i in range(len(sorted_ranges) - 1):
            current = sorted_ranges[i]
            next_range = sorted_ranges[i + 1]

            # Check if there's a gap between current and next range
            # Use a small threshold (1 second) to avoid gaps due to timestamp rounding
            if (next_range.start - current.end) > timedelta(seconds=1):
                self.gaps.append(TimeRange(start=current.end, end=next_range.start))

        # Sort gaps by start time
        self.gaps.sort(key=lambda r: r.start)

    def get_recent_gaps(self, time_window: timedelta) -> List[TimeRange]:
        """Get gaps that overlap with the recent time window.

        Args:
            time_window: How far back to look for gaps

        Returns:
            List of gaps within the time window
        """
        now = datetime.now(timezone.utc)
        recent_window = TimeRange(start=now - time_window, end=now)
        return [gap for gap in self.gaps if gap.overlaps(recent_window)]
