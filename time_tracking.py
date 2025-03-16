"""Time range and gap tracking for message history."""

import logging
from dataclasses import dataclass
from typing import List

import pendulum
from pendulum import DateTime, Duration

# Set up logging
logger = logging.getLogger("deepbot.time_tracking")


@dataclass
class TimeRange:
    """Represents a range of time with start and end points."""

    start: DateTime
    end: DateTime

    def __post_init__(self) -> None:
        """Ensure start and end are timezone-aware UTC."""
        # Convert to UTC if needed
        self.start = pendulum.instance(self.start).in_timezone("UTC")
        self.end = pendulum.instance(self.end).in_timezone("UTC")

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
    last_sync: DateTime

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
            if (next_range.start - current.end) > Duration(seconds=1):
                self.gaps.append(TimeRange(start=current.end, end=next_range.start))

        # Sort gaps by start time
        self.gaps.sort(key=lambda r: r.start)

    def get_recent_gaps(self, time_window: Duration) -> List[TimeRange]:
        """Get gaps that overlap with the recent time window.

        Args:
            time_window: How far back to look for gaps

        Returns:
            List of gaps within the time window
        """
        now = pendulum.now("UTC")
        recent_window = TimeRange(
            start=now.subtract(seconds=time_window.in_seconds()), end=now
        )
        return [gap for gap in self.gaps if gap.overlaps(recent_window)]
