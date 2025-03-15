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
from unittest.mock import Mock

import pytest
from discord import TextChannel
from discord.message import Message

from message_store import (
    ChannelMetadata,
    MessageStore,
    Role,
    StoredMessage,
    TimeRange,
    UserInfo,
)


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
    store = MessageStore(storage_dir=test_data_dir)

    # Save to a new directory
    new_dir = tempfile.mkdtemp()
    store.storage_dir = new_dir
    store.save_all_channels()

    # Load and compare the saved data
    for filename, orig in original_data.items():
        new_file = os.path.join(new_dir, filename)
        assert os.path.exists(new_file), f"File {filename} was not saved"

        with open(new_file, "r", encoding="utf-8") as f:
            new_data = json.load(f)

        # Compare the important parts of the data
        # Note: We skip exportedAt since it will be different
        print(f"Comparing file: {filename}")
        print(f"Original data keys: {list(orig.keys())}")
        print(f"New data keys: {list(new_data.keys())}")

        # Skip metadata files when comparing message data
        if not filename.endswith("_metadata.json"):
            assert new_data.get("guild") == orig.get("guild"), "Guild info mismatch"
            assert new_data.get("channel") == orig.get(
                "channel"
            ), "Channel info mismatch"
            assert new_data.get("messages", []) == orig.get(
                "messages", []
            ), "Message count mismatch"

            # Compare each message's fields
            for new_msg, orig_msg in zip(new_data["messages"], orig["messages"]):
                assert new_msg["id"] == orig_msg["id"], "Message ID mismatch"
                assert new_msg["type"] == orig_msg["type"], "Message type mismatch"
                assert new_msg["content"] == orig_msg["content"], "Content mismatch"
                assert new_msg["author"] == orig_msg["author"], "Author info mismatch"
                assert new_msg["mentions"] == orig_msg["mentions"], "Mentions mismatch"
                assert (
                    new_msg["attachments"] == orig_msg["attachments"]
                ), "Attachments mismatch"
                assert (
                    new_msg["reactions"] == orig_msg["reactions"]
                ), "Reactions mismatch"

                # Handle optional reference field
                if "reference" in orig_msg:
                    assert "reference" in new_msg, "Reference missing in new message"
                    assert (
                        new_msg["reference"] == orig_msg["reference"]
                    ), "Reference mismatch"
                else:
                    assert (
                        "reference" not in new_msg
                    ), "Unexpected reference in new message"

                assert (
                    new_msg["inlineEmojis"] == orig_msg["inlineEmojis"]
                ), "Emoji mismatch"

    # Cleanup
    shutil.rmtree(new_dir)


@pytest.mark.asyncio
async def test_message_store_json_diff(test_data_dir: str) -> None:
    """Test that the JSON files are identical after a roundtrip (except for exportedAt)."""
    # Create a new store and load the data
    store = MessageStore(storage_dir=test_data_dir)

    # Save to a new directory
    new_dir = tempfile.mkdtemp()
    store.storage_dir = new_dir
    store.save_all_channels()

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

        # Convert both to formatted JSON strings for comparison
        orig_json = json.dumps(orig_data, indent=2, sort_keys=True)
        new_json = json.dumps(new_data, indent=2, sort_keys=True)

        # If they don't match, generate a detailed diff
        if orig_json != new_json:
            diff = list(
                difflib.unified_diff(
                    orig_json.splitlines(keepends=True),
                    new_json.splitlines(keepends=True),
                    fromfile=f"original/{filename}",
                    tofile=f"roundtripped/{filename}",
                )
            )
            assert False, f"JSON diff found in {filename}:\n{''.join(diff)}"

    # Cleanup
    shutil.rmtree(new_dir)


@pytest.mark.asyncio
async def test_gap_tracking(test_data_dir: str) -> None:
    """Test that gaps are properly tracked and updated."""
    store = MessageStore(storage_dir=test_data_dir)

    # Create a test channel ID
    channel_id = "123456789"

    # Create metadata with some known ranges and gaps
    metadata = ChannelMetadata(
        channel_id=channel_id,
        known_ranges=[
            TimeRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc),
            ),
            TimeRange(
                start=datetime(2024, 1, 4, tzinfo=timezone.utc),
                end=datetime(2024, 1, 5, tzinfo=timezone.utc),
            ),
        ],
        gaps=[],
        last_sync=datetime.now(timezone.utc),
    )

    # Update gaps
    metadata._update_gaps()

    # Add the metadata to the store
    # pylint: disable=protected-access
    store._channel_metadata[channel_id] = metadata

    # Check that gaps are detected
    assert len(metadata.gaps) == 1
    assert metadata.gaps[0].start == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert metadata.gaps[0].end == datetime(2024, 1, 4, tzinfo=timezone.utc)

    # Add a range that fills the gap
    metadata.add_known_range(
        TimeRange(
            start=datetime(2024, 1, 2, tzinfo=timezone.utc),
            end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        )
    )

    # Check that the gap is filled
    assert len(metadata.gaps) == 0

    # Add a range that overlaps with existing ranges
    metadata.add_known_range(
        TimeRange(
            start=datetime(
                2024, 1, 1, 12, tzinfo=timezone.utc
            ),  # Overlaps with first range
            end=datetime(
                2024, 1, 2, 12, tzinfo=timezone.utc
            ),  # Overlaps with second range
        )
    )

    # Check that ranges are merged
    assert len(metadata.known_ranges) == 1
    assert metadata.known_ranges[0].start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert metadata.known_ranges[0].end == datetime(2024, 1, 5, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_recent_gaps(test_data_dir: str) -> None:
    """Test that recent gaps are properly identified."""
    store = MessageStore(storage_dir=test_data_dir)

    # Create a test channel ID
    channel_id = "123456789"

    # Create metadata with some known ranges
    now = datetime.now(timezone.utc)
    metadata = ChannelMetadata(
        channel_id=channel_id,
        known_ranges=[
            TimeRange(start=now - timedelta(hours=48), end=now - timedelta(hours=24)),
            TimeRange(start=now - timedelta(hours=12), end=now),
        ],
        gaps=[],
        last_sync=now,
    )

    # Update gaps
    metadata._update_gaps()

    # Add the metadata to the store
    # pylint: disable=protected-access
    store._channel_metadata[channel_id] = metadata

    # Check recent gaps (last 24 hours)
    recent_gaps = metadata.get_recent_gaps(timedelta(hours=24))
    assert len(recent_gaps) == 1
    assert recent_gaps[0].start == now - timedelta(hours=24)
    assert recent_gaps[0].end == now - timedelta(hours=12)


@pytest.mark.asyncio
async def test_channel_initialization(test_data_dir: str) -> None:
    """Test channel initialization with gaps and recent messages."""
    store = MessageStore(storage_dir=test_data_dir)

    # Create a mock channel
    channel = Mock(spec=TextChannel)
    channel.id = 123456789
    channel.name = "test-channel"

    # Create some mock messages
    now = datetime.now(timezone.utc)
    messages: List[Mock] = [Mock(spec=Message) for _ in range(5)]

    # Set up message timestamps
    for i, msg in enumerate(messages):
        msg.created_at = now - timedelta(hours=i)
        msg.timestamp = (now - timedelta(hours=i)).isoformat()
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

    # Set up channel.history() to return our mock messages
    async def mock_history(*args: Any, **kwargs: Any) -> AsyncGenerator[Message, None]:
        for msg in messages:
            yield msg

    channel.history = mock_history

    # Initialize the channel
    # pylint: disable=protected-access
    store._channel_messages[str(channel.id)] = {}
    # pylint: disable=protected-access
    store._channel_metadata[str(channel.id)] = ChannelMetadata(
        channel_id=str(channel.id), known_ranges=[], gaps=[], last_sync=now
    )

    # Mock add_message to store messages
    async def mock_add_message(message: Message) -> None:
        # pylint: disable=protected-access
        stored_msg = StoredMessage(
            id=str(message.id),
            type="Default",
            timestamp=message.created_at.isoformat(),
            timestampEdited=(
                message.edited_at.isoformat() if message.edited_at else None
            ),
            callEndedTimestamp=None,
            isPinned=False,
            content=message.content,
            author=UserInfo(
                id=str(message.author.id),
                name=message.author.name,
                discriminator=message.author.discriminator,
                nickname=None,
                color=None,
                isBot=message.author.bot,
                roles=[],
                avatarUrl=(
                    str(message.author.avatar.url) if message.author.avatar else ""
                ),
            ),
            mentions=[],
            attachments=[],
            embeds=[],
            reactions=[],
            stickers=[],
            inlineEmojis=[],
            reference=None,
        )
        # pylint: disable=protected-access
        store._channel_messages[str(channel.id)][stored_msg.id] = stored_msg

    store.add_message = mock_add_message

    await store.initialize_channel(channel)

    # Check that messages were added
    channel_messages = store.get_channel_messages(str(channel.id))
    assert len(channel_messages) == 5

    # Check that metadata was updated
    # pylint: disable=protected-access
    metadata = store._channel_metadata[str(channel.id)]
    assert len(metadata.known_ranges) == 1
    assert metadata.known_ranges[0].start <= messages[-1].created_at
    assert metadata.known_ranges[0].end >= messages[0].created_at


@pytest.mark.asyncio
async def test_channel_initialization_with_gaps(test_data_dir: str) -> None:
    """Test channel initialization when there are gaps in history."""
    store = MessageStore(storage_dir=test_data_dir)

    # Create a mock channel
    channel = Mock(spec=TextChannel)
    channel.id = 123456789
    channel.name = "test-channel"

    # Create metadata with a gap
    now = datetime.now(timezone.utc)
    metadata = ChannelMetadata(
        channel_id=str(channel.id),
        known_ranges=[
            TimeRange(start=now - timedelta(hours=48), end=now - timedelta(hours=24)),
            TimeRange(start=now - timedelta(hours=12), end=now),
        ],
        gaps=[],
        last_sync=now,
    )
    # Update gaps
    metadata._update_gaps()

    # Add the metadata to the store
    # pylint: disable=protected-access
    store._channel_metadata[str(channel.id)] = metadata
    store._channel_messages[str(channel.id)] = {}

    # Create mock messages for the gap
    gap_messages: List[Mock] = [Mock(spec=Message) for _ in range(3)]

    # Set up message timestamps in the gap
    for i, msg in enumerate(gap_messages):
        msg.created_at = now - timedelta(hours=20 + i)
        msg.timestamp = (now - timedelta(hours=20 + i)).isoformat()
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

    # Set up channel.history() to return our mock messages
    async def mock_history(*args: Any, **kwargs: Any) -> AsyncGenerator[Message, None]:
        for msg in gap_messages:
            yield msg

    channel.history = mock_history

    # Mock add_message to store messages
    async def mock_add_message(message: Message) -> None:
        # pylint: disable=protected-access
        stored_msg = StoredMessage(
            id=str(message.id),
            type="Default",
            timestamp=message.created_at.isoformat(),
            timestampEdited=(
                message.edited_at.isoformat() if message.edited_at else None
            ),
            callEndedTimestamp=None,
            isPinned=False,
            content=message.content,
            author=UserInfo(
                id=str(message.author.id),
                name=message.author.name,
                discriminator=message.author.discriminator,
                nickname=None,
                color=None,
                isBot=message.author.bot,
                roles=[],
                avatarUrl=(
                    str(message.author.avatar.url) if message.author.avatar else ""
                ),
            ),
            mentions=[],
            attachments=[],
            embeds=[],
            reactions=[],
            stickers=[],
            inlineEmojis=[],
            reference=None,
        )
        # pylint: disable=protected-access
        store._channel_messages[str(channel.id)][stored_msg.id] = stored_msg

    store.add_message = mock_add_message

    # Initialize the channel
    await store.initialize_channel(channel)

    # Check that messages were added
    channel_messages = store.get_channel_messages(str(channel.id))
    assert len(channel_messages) == 3

    # Check that metadata was updated
    # pylint: disable=protected-access
    metadata = store._channel_metadata[str(channel.id)]
    assert (
        len(metadata.known_ranges) == 1
    )  # All ranges should be merged since they overlap
    assert metadata.known_ranges[0].start <= gap_messages[-1].created_at
    assert metadata.known_ranges[0].end >= gap_messages[0].created_at
