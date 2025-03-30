"""
Detailed tests for each collection type in the TypedDatabase.

These tests focus on read-only operations on each collection type,
verifying various query patterns and ensuring proper type conversion.
"""

from typing import Optional

import pytest

from lorekeeper.db.models import (
    Asset,
    Author,
    Channel,
    Embed,
    Guild,
    Message,
    Reaction,
    Reference,
)
from lorekeeper.db.typed_database import TypedDatabase, pad_id

# Example data from the test guild (unpadded)
EXAMPLE_GUILD_ID_UNPADDED = "748545324524575035"
EXAMPLE_CHANNEL_ID_UNPADDED = "1012224565567721552"
EXAMPLE_MESSAGE_ID_UNPADDED = "1045178587575242772"
EXAMPLE_AUTHOR_ID_UNPADDED = "521486476985155645"

# Padded IDs as they appear in MongoDB
EXAMPLE_GUILD_ID = pad_id(EXAMPLE_GUILD_ID_UNPADDED)
EXAMPLE_CHANNEL_ID = pad_id(EXAMPLE_CHANNEL_ID_UNPADDED)
EXAMPLE_MESSAGE_ID = pad_id(EXAMPLE_MESSAGE_ID_UNPADDED)
EXAMPLE_AUTHOR_ID = pad_id(EXAMPLE_AUTHOR_ID_UNPADDED)


class TestMessagesCollection:
    """Tests focused on the messages collection."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the messages collection for testing."""
        self.messages = TypedDatabase.get_messages_collection(EXAMPLE_GUILD_ID_UNPADDED)

    def test_find_messages_by_id(self) -> None:
        """Test finding a message by its ID."""
        message = self.messages.find_one({"_id": EXAMPLE_MESSAGE_ID_UNPADDED})
        assert message is not None
        assert isinstance(message, Message)
        assert message.id == EXAMPLE_MESSAGE_ID

    def test_find_messages_by_channel(self) -> None:
        """Test finding messages in a specific channel."""
        messages = self.messages.find({"channelId": EXAMPLE_CHANNEL_ID}).to_list()
        assert len(messages) > 0
        for message in messages:
            assert isinstance(message, Message)
            assert message.channelId == EXAMPLE_CHANNEL_ID

    def test_find_messages_by_author(self) -> None:
        """Test finding messages by a specific author."""
        messages = self.messages.find(
            {"author._id": EXAMPLE_AUTHOR_ID_UNPADDED}
        ).to_list()
        for message in messages:
            assert isinstance(message, Message)
            assert message.author.id == EXAMPLE_AUTHOR_ID

    def test_find_pinned_messages(self) -> None:
        """Test finding pinned messages."""
        messages = self.messages.find({"isPinned": True}).to_list()
        for message in messages:
            assert isinstance(message, Message)
            assert message.isPinned

    def test_find_messages_with_attachments(self) -> None:
        """Test finding messages with attachments."""
        messages = self.messages.find(
            {"attachments": {"$exists": True, "$ne": None}}
        ).to_list()
        for message in messages:
            assert isinstance(message, Message)
            if message.attachments:
                assert isinstance(message.attachments, list)
                assert len(message.attachments) > 0

    def test_find_messages_with_embeds(self) -> None:
        """Test finding messages with embeds."""
        messages = self.messages.find(
            {"embeds": {"$exists": True, "$ne": None}}
        ).to_list()
        for message in messages:
            assert isinstance(message, Message)
            if message.embeds:
                assert isinstance(message.embeds, list)
                assert len(message.embeds) > 0
                for embed in message.embeds:
                    assert isinstance(embed, Embed)

    def test_find_messages_with_mentions(self) -> None:
        """Test finding messages with mentions."""
        messages = self.messages.find(
            {"mentions": {"$exists": True, "$ne": None}}
        ).to_list()
        for message in messages:
            assert isinstance(message, Message)
            if message.mentions:
                assert isinstance(message.mentions, list)
                assert len(message.mentions) > 0

    def test_find_messages_with_reactions(self) -> None:
        """Test finding messages with reactions."""
        messages = self.messages.find(
            {"reactions": {"$exists": True, "$ne": None}}
        ).to_list()
        for message in messages:
            assert isinstance(message, Message)
            if message.reactions:
                assert isinstance(message.reactions, list)
                assert len(message.reactions) > 0
                for reaction in message.reactions:
                    assert isinstance(reaction, Reaction)

    def test_find_messages_with_references(self) -> None:
        """Test finding messages that reference other messages."""
        messages = self.messages.find(
            {"reference": {"$exists": True, "$ne": None}}
        ).to_list()
        for message in messages:
            assert isinstance(message, Message)
            if message.reference:
                assert isinstance(message.reference, Reference)
                assert hasattr(message.reference, "messageId")
                assert hasattr(message.reference, "channelId")
                assert hasattr(message.reference, "guildId")

    def test_find_messages_by_content(self) -> None:
        """Test finding messages by content."""
        # Search for 'DiscordChatExporter' in the content
        # Since content is an array of objects, we need to search in the 'content' field of each element
        messages = self.messages.find(
            {"content.content": {"$regex": ".*DiscordChatExporter.*", "$options": "i"}}
        ).to_list()

        found_match = False
        for message in messages:
            assert isinstance(message, Message)
            # Check if content contains the search term
            if isinstance(message.content, str):
                if "DiscordChatExporter" in message.content:
                    found_match = True
            elif message.content and len(message.content) > 0:
                # Content is a list of MessageContent objects
                for content_item in message.content:
                    if (
                        hasattr(content_item, "content")
                        and "DiscordChatExporter" in content_item.content
                    ):
                        found_match = True
                        break

        # We should find at least one match (the example message)
        assert (
            found_match
        ), "Should find at least one message with 'DiscordChatExporter'"

    def test_message_sorting(self) -> None:
        """Test sorting messages by timestamp."""
        messages = self.messages.find({}).sort("timestamp", 1)  # Ascending

        # Verify the sort worked correctly
        prev_timestamp: Optional[str] = None
        for message in messages:
            assert isinstance(message, Message)
            if prev_timestamp:
                assert message.timestamp >= prev_timestamp
            prev_timestamp = message.timestamp

    def test_message_projection(self) -> None:
        """Test using projections to retrieve only specific fields."""
        # Note: This test is MongoDB specific and doesn't use the type safety
        # because projections return partial documents that may not match the model
        raw_collection = self.messages.collection
        projections = raw_collection.find(
            {"_id": EXAMPLE_MESSAGE_ID_UNPADDED},
            projection={"_id": 1, "type": 1, "content": 1},
        )

        for doc in projections:
            assert "_id" in doc
            assert "type" in doc
            assert "content" in doc
            # Other fields should not be present
            assert "timestamp" not in doc


class TestChannelsCollection:
    """Tests focused on the channels collection."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the channels collection for testing."""
        self.channels = TypedDatabase.get_channels_collection(EXAMPLE_GUILD_ID_UNPADDED)

    def test_find_channel_by_id(self) -> None:
        """Test finding a channel by its ID."""
        channel = self.channels.find_one({"_id": EXAMPLE_CHANNEL_ID_UNPADDED})
        assert channel is not None
        assert isinstance(channel, Channel)
        assert channel.id == EXAMPLE_CHANNEL_ID

    def test_find_channels_by_type(self) -> None:
        """Test finding channels by type."""
        channels = self.channels.find({"type": "GuildTextChat"}).to_list()
        for channel in channels:
            assert isinstance(channel, Channel)
            assert channel.type == "GuildTextChat"

    def test_find_channels_by_category(self) -> None:
        """Test finding channels by category."""
        channels = self.channels.find({"category": "Text Channels"}).to_list()
        for channel in channels:
            assert isinstance(channel, Channel)
            assert channel.category == "Text Channels"

    def test_find_channels_with_topic(self) -> None:
        """Test finding channels with a topic."""
        channels = self.channels.find(
            {"topic": {"$exists": True, "$ne": None}}
        ).to_list()
        for channel in channels:
            assert isinstance(channel, Channel)
            assert channel.topic is not None
            assert len(channel.topic) > 0


class TestAuthorsCollection:
    """Tests focused on the authors collection."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the authors collection for testing."""
        self.authors = TypedDatabase.get_authors_collection(EXAMPLE_GUILD_ID_UNPADDED)

    def test_find_author_by_id(self) -> None:
        """Test finding an author by ID."""
        author = self.authors.find_one({"_id": EXAMPLE_AUTHOR_ID_UNPADDED})
        if author:  # Author might be stored differently
            assert isinstance(author, Author)
            assert author.id == EXAMPLE_AUTHOR_ID
            assert hasattr(author, "name")

    def test_find_authors_by_is_bot(self) -> None:
        """Test finding bot authors."""
        # Some authors might be bots
        bot_authors = self.authors.find({"isBot": True}).to_list()

        # All returned authors should be bots
        for author in bot_authors:
            assert isinstance(author, Author)
            assert author.isBot

        # Non-bot authors
        human_authors = self.authors.find({"isBot": False}).to_list()
        for author in human_authors:
            assert isinstance(author, Author)
            assert not author.isBot

    def test_find_authors_with_roles(self) -> None:
        """Test finding authors with roles."""
        authors_with_roles = self.authors.find(
            {"roles": {"$exists": True, "$ne": None}}
        ).to_list()

        for author in authors_with_roles:
            assert isinstance(author, Author)
            if author.roles:
                assert isinstance(author.roles, list)
                assert len(author.roles) > 0


class TestAssetsCollection:
    """Tests focused on the assets collection."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the assets collection for testing."""
        self.assets = TypedDatabase.get_assets_collection(EXAMPLE_GUILD_ID_UNPADDED)

    def test_find_assets_by_type(self) -> None:
        """Test finding assets by type."""
        # Common asset types include "attachment", "avatar", "emoji"
        assets = self.assets.find({}).to_list()
        for asset in assets:
            assert isinstance(asset, Asset)
            if asset.type:
                assert isinstance(asset.type, str)

    def test_find_assets_by_extension(self) -> None:
        """Test finding assets by file extension."""
        # Find assets with a specific extension, like png or jpg
        assets = self.assets.find(
            {"extension": {"$in": ["png", "jpg", "jpeg", "gif"]}}
        ).to_list()
        for asset in assets:
            assert isinstance(asset, Asset)
            assert asset.extension in ["png", "jpg", "jpeg", "gif"]

    def test_find_assets_by_size(self) -> None:
        """Test finding assets by size."""
        # Find assets larger than a certain size (e.g., 100kb)
        size_threshold = 100 * 1024  # 100kb in bytes
        large_assets = self.assets.find(
            {"sizeBytes": {"$gt": size_threshold}}
        ).to_list()

        for asset in large_assets:
            assert isinstance(asset, Asset)
            if asset.sizeBytes is not None:
                assert asset.sizeBytes > size_threshold


class TestGuildsCollection:
    """Tests focused on the guilds collection."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the guilds collection for testing."""
        self.guilds = TypedDatabase.get_guilds_collection()

    def test_find_guild_by_id(self) -> None:
        """Test finding a guild by ID."""
        guild = self.guilds.find_one({"_id": EXAMPLE_GUILD_ID_UNPADDED})
        assert guild is not None
        assert isinstance(guild, Guild)
        assert guild.id == EXAMPLE_GUILD_ID
        assert guild.name == "Example server"

        # Check other properties
        assert isinstance(guild.msg_count, int)

    def test_find_guilds_by_name(self) -> None:
        """Test finding guilds by name."""
        guilds = self.guilds.find({"name": "Example server"}).to_list()
        assert len(guilds) > 0

        for guild in guilds:
            assert isinstance(guild, Guild)
            assert guild.name == "Example server"
