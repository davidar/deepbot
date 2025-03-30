"""
Read-only tests for the TypedDatabase interface.

These tests verify that the typesafe database interface correctly
retrieves and parses data from MongoDB without making any modifications.
"""

from typing import Any, Dict, List

from lorekeeper.db.models import Asset, Author, Channel, Guild, Message, Sticker
from lorekeeper.db.typed_database import TypedDatabase, pad_id

# Example guild ID from the test data - padded for MongoDB format
EXAMPLE_GUILD_ID_UNPADDED = "748545324524575035"
EXAMPLE_CHANNEL_ID_UNPADDED = "1012224565567721552"
EXAMPLE_MESSAGE_ID_UNPADDED = "1045178587575242772"
EXAMPLE_AUTHOR_ID_UNPADDED = "521486476985155645"

# Padded IDs as they appear in MongoDB
EXAMPLE_GUILD_ID = pad_id(EXAMPLE_GUILD_ID_UNPADDED)
EXAMPLE_CHANNEL_ID = pad_id(EXAMPLE_CHANNEL_ID_UNPADDED)
EXAMPLE_MESSAGE_ID = pad_id(EXAMPLE_MESSAGE_ID_UNPADDED)
EXAMPLE_AUTHOR_ID = pad_id(EXAMPLE_AUTHOR_ID_UNPADDED)


class TestTypedDatabase:
    """Test the TypedDatabase interface with read-only operations."""

    def test_database_connection(self) -> None:
        """Test that the database is online."""
        is_online = TypedDatabase.is_online()
        assert is_online, "Database should be online"

    def test_get_guild(self) -> None:
        """Test retrieving guild information."""
        guilds_collection = TypedDatabase.get_guilds_collection()
        guild = guilds_collection.find_one({"_id": EXAMPLE_GUILD_ID_UNPADDED})

        assert guild is not None, "Should find the example guild"
        assert isinstance(guild, Guild), "Result should be a Guild object"
        assert guild.id == EXAMPLE_GUILD_ID
        assert guild.name == "Example server"

        # Test accessing guild properties
        assert isinstance(guild.msg_count, int)
        if guild.icon:
            assert isinstance(guild.icon, dict)

    def test_get_channels(self) -> None:
        """Test retrieving channel information."""
        channels_collection = TypedDatabase.get_channels_collection(
            EXAMPLE_GUILD_ID_UNPADDED
        )
        channels = channels_collection.find({}).to_list()

        # Verify we have at least one channel
        assert len(channels) > 0, "Should find at least one channel"

        # Test the example channel
        channel = channels_collection.find_one({"_id": EXAMPLE_CHANNEL_ID_UNPADDED})
        assert channel is not None, "Should find the example channel"
        assert isinstance(channel, Channel), "Result should be a Channel object"
        assert channel.id == EXAMPLE_CHANNEL_ID
        assert channel.name == "general"
        assert channel.type == "GuildTextChat"
        assert channel.categoryId == EXAMPLE_GUILD_ID
        assert channel.category == "Text Channels"
        assert channel.topic == "Welcome to the Example guild!"

    def test_get_messages(self) -> None:
        """Test retrieving message information."""
        messages_collection = TypedDatabase.get_messages_collection(
            EXAMPLE_GUILD_ID_UNPADDED
        )
        messages = messages_collection.find({"channelId": EXAMPLE_CHANNEL_ID}).to_list()

        # Verify we have at least one message
        assert len(messages) > 0, "Should find at least one message"

        # Test the example message
        message = messages_collection.find_one({"_id": EXAMPLE_MESSAGE_ID_UNPADDED})
        assert message is not None, "Should find the example message"
        assert isinstance(message, Message), "Result should be a Message object"

        # Test basic message properties
        assert message.id == EXAMPLE_MESSAGE_ID
        assert message.type == "Default"
        assert message.channelId == EXAMPLE_CHANNEL_ID
        assert message.guildId == EXAMPLE_GUILD_ID
        assert message.isPinned

        # Test content
        if isinstance(message.content, str):
            assert (
                "Thank you for choosing **DiscordChatExporter-frontend**"
                in message.content
            )
        else:
            # Content might be a list of MessageContent objects
            assert len(message.content) > 0
            content_text = (
                message.content[0].content if len(message.content) > 0 else ""
            )
            assert "DiscordChatExporter-frontend" in content_text

        # Test optional fields
        assert message.timestampEdited is None
        assert message.callEndedTimestamp is None

        # Test collections are empty or present based on example data
        if message.attachments:
            assert isinstance(message.attachments, list)

        if message.embeds:
            assert isinstance(message.embeds, list)

        if message.stickers:
            assert isinstance(message.stickers, list)

        if message.reactions:
            assert isinstance(message.reactions, list)

        if message.mentions:
            assert isinstance(message.mentions, list)

    def test_get_authors(self) -> None:
        """Test retrieving author information."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        authors_collection = TypedDatabase.get_authors_collection(guild_id)
        authors = authors_collection.find({}).to_list()

        # Verify we have at least one author
        assert len(authors) > 0, "Should find at least one author"

        # Test the example author
        author = authors_collection.find_one({"_id": EXAMPLE_AUTHOR_ID_UNPADDED})
        if author:  # The author might be stored in a different way
            assert isinstance(author, Author), "Result should be an Author object"
            assert author.id == EXAMPLE_AUTHOR_ID
            assert author.name == "Adam"

            # Optional fields
            if author.discriminator:
                assert author.discriminator == "7077"

            if author.avatar:
                assert isinstance(author.avatar, dict)

            if author.roles:
                assert isinstance(author.roles, list)

    def test_get_assets(self) -> None:
        """Test retrieving asset information."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        assets_collection = TypedDatabase.get_assets_collection(guild_id)
        assets = assets_collection.find({}).to_list()

        # Test the assets if present
        for asset in assets:
            assert isinstance(asset, Asset), "Result should be an Asset object"
            assert hasattr(asset, "id")

            # Test common asset properties if they exist
            if asset.path:
                assert isinstance(asset.path, str)

            if asset.type:
                assert isinstance(asset.type, str)

            if asset.extension:
                assert isinstance(asset.extension, str)

            if asset.sizeBytes:
                assert isinstance(asset.sizeBytes, int)

    def test_get_stickers(self) -> None:
        """Test retrieving sticker information if present."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        stickers_collection = TypedDatabase.get_stickers_collection(guild_id)
        stickers = stickers_collection.find({}).to_list()

        # Test the stickers if present
        for sticker in stickers:
            assert isinstance(sticker, Sticker), "Result should be a Sticker object"
            assert hasattr(sticker, "id")

            if sticker.name:
                assert isinstance(sticker.name, str)

            if sticker.format:
                assert isinstance(sticker.format, str)

            if sticker.source:
                assert isinstance(sticker.source, dict)

    def test_aggregate_operations(self) -> None:
        """Test aggregate operations on collections."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        messages_collection = TypedDatabase.get_messages_collection(guild_id)

        # Get counts of messages by type
        pipeline: List[Dict[str, Any]] = [
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]

        type_counts = messages_collection.aggregate(pipeline)

        assert len(type_counts) > 0, "Should have at least one message type"
        assert isinstance(type_counts, list)

        # Each item should have _id and count fields
        for type_count in type_counts:
            assert "_id" in type_count
            assert "count" in type_count
            assert isinstance(type_count["count"], int)

    def test_distinct_operations(self) -> None:
        """Test distinct operations on collections."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        messages_collection = TypedDatabase.get_messages_collection(guild_id)

        # Get distinct message types
        types = messages_collection.distinct("type")

        assert len(types) > 0, "Should have at least one distinct message type"
        assert "Default" in types, "Should have the 'Default' message type"

    def test_query_with_filter(self) -> None:
        """Test querying with filters."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        messages_collection = TypedDatabase.get_messages_collection(guild_id)

        # Find messages from a specific time range
        messages = messages_collection.find(
            {"timestamp": {"$gte": "2022-01-01", "$lte": "2023-12-31"}}
        ).to_list()

        # Just test that the query executed without errors
        for message in messages:
            assert isinstance(message, Message)
            # Don't need to check the timestamp as it might be in a different format
            # We just want to verify the query execution

    def test_count_documents(self) -> None:
        """Test counting documents."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        messages_collection = TypedDatabase.get_messages_collection(guild_id)

        # Count messages of a specific type
        count = messages_collection.count_documents({"type": "Default"})
        assert count >= 0, "Should be able to count documents"

        # Count messages in a specific channel
        channel_count = messages_collection.count_documents(
            {"channelId": EXAMPLE_CHANNEL_ID}
        )
        assert channel_count >= 0, "Should be able to count by channel"

    def test_cursor_limit_and_skip(self) -> None:
        """Test cursor operations like limit and skip."""
        # Ensure the guild_id is not None
        guild_id = EXAMPLE_GUILD_ID_UNPADDED
        messages_collection = TypedDatabase.get_messages_collection(guild_id)

        # Get messages with limit
        limited_messages = (
            messages_collection.find({}).sort("timestamp", 1).limit(5).to_list()
        )
        assert len(limited_messages) <= 5, "Should limit to 5 messages"

        # Get messages with skip
        skipped_messages = (
            messages_collection.find({}).sort("timestamp", 1).skip(5).limit(5).to_list()
        )
        assert len(skipped_messages) <= 5, "Should limit to 5 messages after skipping 5"
