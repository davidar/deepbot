"""Tests for message store serialization."""

# pylint: disable=protected-access

import difflib
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import Mock, patch

import pytest
from discord import TextChannel
from discord.message import Message

from message_store import MessageStore
from time_tracking import ChannelMetadata, TimeRange


@pytest.fixture
def test_data_dir() -> Generator[str, None, None]:
    """Create a temporary directory with test data."""
    # Create temp dir
    temp_dir = tempfile.mkdtemp()

    # Copy all .json files from message_store/ to temp dir
    src_dir = Path("message_store")
    if src_dir.exists():
        for json_file in src_dir.glob("*.json"):
            shutil.copy2(json_file, temp_dir)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


def normalize_json(data: Any) -> Any:
    """Normalize JSON data for comparison by removing dynamic fields."""
    if isinstance(data, dict):
        return {
            str(k): normalize_json(v)
            for k, v in data.items()  # type: ignore
            if k != "exportedAt"  # Skip this field as it changes
        }
    elif isinstance(data, list):
        return [normalize_json(item) for item in data]  # type: ignore
    else:
        return data


@pytest.mark.asyncio
async def test_message_store_roundtrip(test_data_dir: str) -> None:
    """Test that messages can be loaded and saved without data loss."""
    # Load original data
    original_data: Dict[str, Dict[str, Any]] = {}
    for filename in os.listdir(test_data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(test_data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                original_data[filename] = json.load(f)

    # Create a new store and load the data
    with patch("message_indexer.MessageIndexer") as mock_indexer:  # Mock the indexer
        store = MessageStore(
            data_dir=test_data_dir,
            message_indexer=mock_indexer.return_value,
        )

        # Save all channel data
        for channel_id in store.storage_manager.get_channel_ids():
            store.storage_manager.save_channel_data(channel_id)

        # Save to a new directory
        new_dir = tempfile.mkdtemp()

        # Copy messages and metadata to new directory
        for filename in os.listdir(test_data_dir):
            if filename.endswith(".json"):
                src_file = os.path.join(test_data_dir, filename)
                dst_file = os.path.join(new_dir, filename)
                shutil.copy2(src_file, dst_file)

        # Create a new store with the copied data
        store = MessageStore(
            data_dir=new_dir,
            message_indexer=mock_indexer.return_value,
        )

        # Save all channel data again
        for channel_id in store.storage_manager.get_channel_ids():
            store.storage_manager.save_channel_data(channel_id)

        # Compare each file
        for filename in os.listdir(test_data_dir):
            if not filename.endswith(".json"):
                continue

            orig_file = os.path.join(test_data_dir, filename)
            new_file = os.path.join(new_dir, filename)

            # Load and normalize both files
            with open(orig_file, "r", encoding="utf-8") as f:
                orig_data = normalize_json(json.load(f))
            with open(new_file, "r", encoding="utf-8") as f:
                new_data = normalize_json(json.load(f))

            # Compare the data
            orig_str = json.dumps(orig_data, sort_keys=True, indent=2)
            new_str = json.dumps(new_data, sort_keys=True, indent=2)

            if orig_str != new_str:
                # If they don't match, show a diff
                diff = list(
                    difflib.unified_diff(
                        orig_str.splitlines(keepends=True),
                        new_str.splitlines(keepends=True),
                        fromfile=orig_file,
                        tofile=new_file,
                    )
                )
                assert False, f"Files differ:\n{''.join(diff)}"

        # Cleanup
        shutil.rmtree(new_dir)


@pytest.mark.asyncio
async def test_gap_tracking(test_data_dir: str) -> None:
    """Test that gaps are properly tracked and updated."""
    with patch("message_indexer.MessageIndexer") as mock_indexer:  # Mock the indexer
        store = MessageStore(
            data_dir=test_data_dir,
            message_indexer=mock_indexer.return_value,
        )

        # Create a test channel ID
        channel_id = "123456789"

        # Create metadata with some known ranges and gaps
        metadata = ChannelMetadata(
            channel_id=channel_id,
            known_ranges=[],
            gaps=[],
            last_sync=datetime.now(timezone.utc),
        )

        # Add known ranges
        metadata.add_known_range(
            TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
        )
        metadata.add_known_range(
            TimeRange(
                start=datetime(2024, 1, 4, tzinfo=timezone.utc),
                end=datetime(2024, 1, 5, tzinfo=timezone.utc),
            )
        )

        # Add the metadata to the store
        store.storage_manager.channel_metadata[channel_id] = metadata

        # Check that gaps are detected
        assert len(metadata.gaps) == 1
        assert metadata.gaps[0].start == datetime(2024, 1, 2, tzinfo=timezone.utc)
        assert metadata.gaps[0].end == datetime(2024, 1, 4, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_recent_gaps(test_data_dir: str) -> None:
    """Test that recent gaps are properly identified."""
    with patch("message_indexer.MessageIndexer") as mock_indexer:  # Mock the indexer
        store = MessageStore(
            data_dir=test_data_dir,
            message_indexer=mock_indexer.return_value,
        )

        # Create a test channel ID
        channel_id = "123456789"

        # Create metadata with some known ranges
        now = datetime.now(timezone.utc)
        metadata = ChannelMetadata(
            channel_id=channel_id,
            known_ranges=[],
            gaps=[],
            last_sync=now,
        )

        # Add known ranges
        metadata.add_known_range(
            TimeRange(
                start=now - timedelta(hours=48),
                end=now - timedelta(hours=24),
            )
        )
        metadata.add_known_range(
            TimeRange(
                start=now - timedelta(hours=12),
                end=now,
            )
        )

        # Add the metadata to the store
        store.storage_manager.channel_metadata[channel_id] = metadata

        # Check recent gaps (last 24 hours)
        recent_gaps = metadata.get_recent_gaps(timedelta(hours=24))
        assert len(recent_gaps) == 1
        assert recent_gaps[0].start == now - timedelta(hours=24)
        assert recent_gaps[0].end == now - timedelta(hours=12)


@pytest.mark.asyncio
async def test_channel_initialization(test_data_dir: str) -> None:
    """Test channel initialization with gaps and recent messages."""
    store = MessageStore(data_dir=test_data_dir)

    # Create a mock channel
    channel = Mock(spec=TextChannel)
    channel.id = 123456789
    channel.name = "test-channel"

    # Create some mock messages
    now = datetime.now(timezone.utc).replace(microsecond=0)  # Round to seconds
    messages: List[Mock] = [Mock(spec=Message) for _ in range(5)]

    # Set up message timestamps
    for i, msg in enumerate(messages):
        msg.created_at = now - timedelta(hours=i)
        msg.timestamp = (now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        msg.id = i
        msg.content = f"Message {i}"
        msg.author = Mock()
        msg.author.id = 1
        msg.author.name = "Test User"
        msg.author.discriminator = "1234"
        msg.author.bot = False
        msg.author.avatar = None
        msg.attachments = []
        msg.embeds = []
        msg.reactions = []
        msg.mentions = []
        msg.stickers = []
        msg.edited_at = None
        msg.reference = None
        msg.pinned = False
        msg.channel = channel

    # Set up channel.history() to return our mock messages
    async def mock_history(*args: Any, **kwargs: Any) -> AsyncGenerator[Message, None]:
        for msg in messages:
            yield msg

    channel.history = mock_history

    # Initialize the channel
    store.storage_manager.messages[str(channel.id)] = {}
    store.storage_manager.channel_metadata[str(channel.id)] = ChannelMetadata(
        channel_id=str(channel.id), known_ranges=[], gaps=[], last_sync=now
    )

    await store.initialize_channel(channel)

    # Check that messages were added
    channel_messages = store.get_channel_messages(str(channel.id))
    assert len(channel_messages) == 5

    # Check that metadata was updated
    metadata = store.storage_manager.channel_metadata[str(channel.id)]
    assert len(metadata.known_ranges) == 1
    # Round timestamps to seconds for comparison
    range_start = metadata.known_ranges[0].start.replace(microsecond=0)
    range_end = metadata.known_ranges[0].end.replace(microsecond=0)
    assert range_start <= messages[-1].created_at
    assert range_end >= messages[0].created_at


@pytest.mark.asyncio
async def test_channel_initialization_with_gaps(test_data_dir: str) -> None:
    """Test channel initialization when there are gaps in history."""
    store = MessageStore(data_dir=test_data_dir)

    # Create a mock channel
    channel = Mock(spec=TextChannel)
    channel.id = 123456789
    channel.name = "test-channel"

    # Create metadata with a gap
    now = datetime.now(timezone.utc)
    metadata = ChannelMetadata(
        channel_id=str(channel.id),
        known_ranges=[],
        gaps=[],
        last_sync=now,
    )

    # Add known ranges
    metadata.add_known_range(
        TimeRange(
            start=now - timedelta(hours=48),
            end=now - timedelta(hours=24),
        )
    )
    metadata.add_known_range(
        TimeRange(
            start=now - timedelta(hours=12),
            end=now,
        )
    )

    # Add the metadata to the store
    store.storage_manager.channel_metadata[str(channel.id)] = metadata
    store.storage_manager.messages[str(channel.id)] = {}

    # Create mock messages for the gap
    gap_messages: List[Mock] = [Mock(spec=Message) for _ in range(3)]

    # Set up message timestamps in the gap
    for i, msg in enumerate(gap_messages):
        msg.created_at = now - timedelta(hours=20 + i)
        msg.timestamp = (now - timedelta(hours=20 + i)).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )
        msg.id = i
        msg.content = f"Gap Message {i}"
        msg.author = Mock()
        msg.author.id = 1
        msg.author.name = "Test User"
        msg.author.discriminator = "1234"
        msg.author.bot = False
        msg.author.avatar = None
        msg.attachments = []
        msg.embeds = []
        msg.reactions = []
        msg.mentions = []
        msg.stickers = []
        msg.edited_at = None
        msg.reference = None
        msg.pinned = False
        msg.channel = channel

    # Set up channel.history() to return our mock messages
    async def mock_history(*args: Any, **kwargs: Any) -> AsyncGenerator[Message, None]:
        for msg in gap_messages:
            yield msg

    channel.history = mock_history

    # Initialize the channel
    await store.initialize_channel(channel)

    # Check that messages were added
    channel_messages = store.get_channel_messages(str(channel.id))
    assert len(channel_messages) == 3

    # Check that metadata was updated
    metadata = store.storage_manager.channel_metadata[str(channel.id)]
    assert (
        len(metadata.known_ranges) == 1
    )  # All ranges should be merged since they overlap
    assert metadata.known_ranges[0].start <= gap_messages[-1].created_at
    assert metadata.known_ranges[0].end >= gap_messages[0].created_at
