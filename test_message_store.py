"""Tests for message store serialization."""

import difflib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from message_store import MessageStore


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


def normalize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize JSON data for comparison by removing dynamic fields."""
    if isinstance(data, dict):
        return {
            k: normalize_json(v)
            for k, v in data.items()
            if k != "exportedAt"  # Skip this field as it changes
        }
    elif isinstance(data, list):
        return [normalize_json(item) for item in data]
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
        assert new_data["guild"] == orig["guild"], "Guild info mismatch"
        assert new_data["channel"] == orig["channel"], "Channel info mismatch"
        assert len(new_data["messages"]) == len(
            orig["messages"]
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
            assert new_msg["reactions"] == orig_msg["reactions"], "Reactions mismatch"

            # Handle optional reference field
            if "reference" in orig_msg:
                assert "reference" in new_msg, "Reference missing in new message"
                assert (
                    new_msg["reference"] == orig_msg["reference"]
                ), "Reference mismatch"
            else:
                assert "reference" not in new_msg, "Unexpected reference in new message"

            assert new_msg["inlineEmojis"] == orig_msg["inlineEmojis"], "Emoji mismatch"

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
