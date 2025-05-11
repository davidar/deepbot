#!/usr/bin/env python3
"""
Discord to IRC Log Exporter

This script exports Discord messages from MongoDB into a simple IRC-style log format:
<username> message

Messages are split into ~1MB files, with multiline messages converted to multiple entries.
Mentions and custom emojis are converted to readable text.
"""

import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Match, Optional, TextIO

# Add parent directory to path so we can import from db
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import Author, Emoji, Message
from db.typed_database import TypedDatabase


class IRCLogExporter:
    def __init__(self, guild_id: str, output_dir: str = "irc_logs"):
        """Initialize the exporter.

        Args:
            guild_id: The Discord guild ID to export
            output_dir: Directory to write log files to
        """
        self.guild_id = guild_id
        self.output_dir = output_dir
        self.messages_collection = TypedDatabase.get_messages_collection(guild_id)
        self.users_collection = TypedDatabase.get_authors_collection(guild_id)
        self.emojis_collection = TypedDatabase.get_emojis_collection(guild_id)
        self.channels_collection = TypedDatabase.get_channels_collection(guild_id)

        # Cache for user and emoji data to minimize DB lookups
        self.user_cache: Dict[str, Author] = {}
        self.emoji_cache: Dict[str, Emoji] = {}
        self.mention_pattern = re.compile(r"<@!?(\d+)>")
        self.emoji_pattern = re.compile(r"<:([^:]+):(\d+)>")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _get_user_by_id(self, user_id: str) -> Author:
        """Get a user by ID, from cache or database."""
        if user_id in self.user_cache:
            return self.user_cache[user_id]

        user = self.users_collection.find_one({"_id": user_id})
        if user:
            self.user_cache[user_id] = user
            return user

        # Return a dummy author if not found
        return Author(_id=user_id, name="Unknown")

    def _get_emoji_by_id(self, emoji_id: str) -> Optional[Emoji]:
        """Get an emoji by ID, from cache or database."""
        if emoji_id in self.emoji_cache:
            return self.emoji_cache[emoji_id]

        emoji = self.emojis_collection.find_one({"_id": emoji_id})
        if emoji:
            self.emoji_cache[emoji_id] = emoji
            return emoji

        return None

    def _resolve_mentions(self, content: str) -> str:
        """Replace <@123456> style mentions with @username format."""

        def replace_mention(match: Match[str]) -> str:
            user_id = match.group(1)
            user = self._get_user_by_id(user_id)
            return f"@{user.name}"

        return self.mention_pattern.sub(replace_mention, content)

    def _resolve_emojis(self, content: str) -> str:
        """Replace <:emoji:123456> style emojis with :emoji: format."""

        def replace_emoji(match: Match[str]) -> str:
            emoji_name = match.group(1)
            emoji_id = match.group(2)
            emoji = self._get_emoji_by_id(emoji_id)
            if emoji:
                return f":{emoji.name}:"
            return f":{emoji_name}:"

        return self.emoji_pattern.sub(replace_emoji, content)

    def _process_message_content(self, message: Message) -> List[str]:
        """Process message content, handling multiline messages and resolving mentions/emojis."""
        processed_lines: List[str] = []

        if isinstance(message.content, list):
            # Handle list of message content objects
            content_parts: List[str] = []
            for content_obj in message.content:
                parts = content_obj.content.strip().split("\n")
                content_parts.extend(parts)

            for line in content_parts:
                if line.strip():  # Skip empty lines
                    processed_line = self._resolve_mentions(line)
                    processed_line = self._resolve_emojis(processed_line)
                    processed_lines.append(processed_line)
        else:
            # Handle string content
            content = message.content.strip()
            if not content:
                return []

            lines = content.split("\n")
            for line in lines:
                if line.strip():  # Skip empty lines
                    processed_line = self._resolve_mentions(line)
                    processed_line = self._resolve_emojis(processed_line)
                    processed_lines.append(processed_line)

        return processed_lines

    def _format_timestamp(self, timestamp_str: str) -> str:
        """Format a timestamp string into a filename-friendly format.

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            Timestamp formatted as YYYY-MM-DD
        """
        try:
            # Convert ISO format to datetime object
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            # Format as YYYY-MM-DD
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # Return a fallback if date parsing fails
            return "unknown-date"

    def list_channels(self) -> List[Dict[str, Any]]:
        """List all channels in the guild.

        Returns:
            List of dicts with channel info (id, name, type, msg_count)
        """
        channels = self.channels_collection.find({}).sort("name", 1)
        channel_info: List[Dict[str, Any]] = []

        for channel in channels:
            # Get message count for this channel
            msg_count = self.messages_collection.count_documents(
                {"channelId": channel.id}
            )

            channel_info.append(
                {
                    "id": channel.id,
                    "name": channel.name,
                    "type": channel.type,
                    "msg_count": msg_count,
                }
            )

        return channel_info

    def export(
        self, channel_ids: Optional[List[str]] = None, max_file_size: int = 1000000
    ):
        """Export messages to IRC-style log files.

        Args:
            channel_ids: Optional list of channel IDs to export. If None, export all channels.
            max_file_size: Maximum file size in bytes (approx 1MB default)
        """
        # If no specific channels are provided, get all channels
        if channel_ids is None:
            all_channels = self.channels_collection.find({})
            channel_ids = [channel.id for channel in all_channels]

        print(f"Found {len(channel_ids)} channels to export in guild {self.guild_id}")

        # Export each channel
        for channel_id in channel_ids:
            self._export_channel(channel_id, max_file_size)

    def _export_channel(self, channel_id: str, max_file_size: int):
        """Export messages from a single channel to IRC-style log files."""
        channel = self.channels_collection.find_one({"_id": channel_id})
        if not channel:
            print(f"Channel {channel_id} not found")
            return

        channel_name = channel.name
        print(f"Exporting channel: #{channel_name}")

        # Get message count for progress reporting
        message_count = self.messages_collection.count_documents(
            {"channelId": channel_id}
        )
        if message_count == 0:
            print(f"No messages found in channel #{channel_name}")
            return

        print(f"Found {message_count} messages in #{channel_name}")

        # Get messages sorted by timestamp
        messages = self.messages_collection.find({"channelId": channel_id}).sort(
            "timestamp", 1
        )

        file_counter = 1
        current_file_timestamp: Optional[str] = None
        current_file_path: Optional[str] = None
        current_file_size = 0
        processed_count = 0
        total_lines_written = 0
        log_file: Optional[TextIO] = None

        try:
            for (
                message_data
            ) in messages.cursor:  # Access raw cursor to handle validation errors
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(
                        f"Processed {processed_count}/{message_count} messages in #{channel_name}"
                    )

                try:
                    # Handle validation errors by fixing the reference field if needed
                    if (
                        "reference" in message_data
                        and message_data["reference"] is not None
                    ):
                        ref = message_data["reference"]
                        if "guildId" not in ref or ref["guildId"] is None:
                            ref["guildId"] = self.guild_id

                    # Create the message object
                    message = Message(**message_data)

                    # Skip system messages and messages with no content
                    if message.type != "Default" or (
                        not message.content and not message.attachments
                    ):
                        continue

                    # Process message content lines
                    content_lines = self._process_message_content(message)

                    # If there's no content but there are attachments, add placeholder
                    if not content_lines and message.attachments:
                        attachments_text = ", ".join(
                            f"[attachment: {a.filenameWithoutHash or 'file'}]"
                            for a in message.attachments
                        )
                        content_lines = [attachments_text]

                    # Skip empty messages
                    if not content_lines:
                        continue

                    # If this is the first message or we need a new file, set the timestamp
                    if current_file_path is None:
                        current_file_timestamp = self._format_timestamp(
                            message.timestamp
                        )
                        current_file_path = os.path.join(
                            self.output_dir,
                            f"{channel_name}_{current_file_timestamp}_{file_counter}.log",
                        )
                        log_file = open(current_file_path, "w", encoding="utf-8")

                    # Write each line as a separate IRC-style message
                    for line in content_lines:
                        # Look up the author by ID to ensure consistent username handling
                        author = self._get_user_by_id(message.author.id)
                        irc_line = f"<{author.name}> {line}\n"
                        line_size = len(irc_line.encode("utf-8"))

                        # If this line would exceed max file size, start a new file
                        if current_file_size + line_size > max_file_size:
                            # Close current file and start a new one
                            if log_file:
                                log_file.close()

                            file_counter += 1
                            current_file_timestamp = self._format_timestamp(
                                message.timestamp
                            )
                            current_file_path = os.path.join(
                                self.output_dir,
                                f"{channel_name}_{current_file_timestamp}_{file_counter}.log",
                            )
                            log_file = open(current_file_path, "w", encoding="utf-8")
                            current_file_size = 0

                        # Write the line
                        if log_file:  # Make sure log_file is not None before writing
                            log_file.write(irc_line)
                            current_file_size += line_size
                            total_lines_written += 1

                except Exception as e:
                    print(
                        f"Error processing message {message_data.get('_id', 'unknown')}: {e}"
                    )
                    continue
        finally:
            # Ensure the file is closed
            if log_file:
                log_file.close()

        print(
            f"Exported {total_lines_written} lines from #{channel_name} to {file_counter} files"
        )


def list_available_guilds():
    """List all available guilds in the database."""
    guilds_collection = TypedDatabase.get_guilds_collection()
    guilds = guilds_collection.find({}).sort("name", 1)

    if not guilds:
        print("No guilds found in the database.")
        return

    print("\nAvailable guilds:")
    print("-" * 50)
    print(f"{'ID':<26} | {'Name':<40} | {'Messages'}")
    print("-" * 50)

    for guild in guilds:
        msg_count = guild.msg_count or 0
        print(f"{guild.id:<26} | {guild.name:<40} | {msg_count:,}")

    return True


def list_channels_in_guild(guild_id: str):
    """List all channels in a guild."""
    exporter = IRCLogExporter(guild_id)
    channels = exporter.list_channels()

    if not channels:
        print(f"No channels found in guild {guild_id}")
        return

    print("\nAvailable channels:")
    print("-" * 70)
    print(f"{'ID':<26} | {'Name':<30} | {'Type':<10} | {'Messages'}")
    print("-" * 70)

    for channel in channels:
        print(
            f"{channel['id']:<26} | {channel['name']:<30} | {channel['type']:<10} | {channel['msg_count']:,}"
        )


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print(
            "Usage: python discord_to_irc_exporter.py <guild_id> [--channel CHANNEL_ID...] [output_directory]"
        )
        print("       python discord_to_irc_exporter.py --list")
        print("       python discord_to_irc_exporter.py <guild_id> --list-channels")
        print("\nOptions:")
        print("  <guild_id>         ID of the guild to export")
        print(
            "  --channel ID       ID of a specific channel to export (can be used multiple times)"
        )
        print("  [output_directory] Directory to write log files (default: irc_logs)")
        print("  --list             List all available guilds")
        print("  --list-channels    List all channels in the specified guild")

        # List available guilds as a convenience
        print("\nNo guild ID provided. Listing available guilds...")
        list_available_guilds()
        sys.exit(1)

    # Check if user wants to list guilds
    if sys.argv[1] == "--list":
        list_available_guilds()
        return

    guild_id = sys.argv[1]

    # Check if user wants to list channels in the guild
    if len(sys.argv) > 2 and sys.argv[2] == "--list-channels":
        list_channels_in_guild(guild_id)
        return

    # Parse remaining arguments
    channel_ids: List[str] = []
    output_dir = "irc_logs"

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--channel" and i + 1 < len(sys.argv):
            channel_ids.append(sys.argv[i + 1])
            i += 2
        else:
            output_dir = sys.argv[i]
            i += 1

    exporter = IRCLogExporter(guild_id, output_dir)

    # If specific channels were requested, only export those
    if channel_ids:
        print(f"Exporting {len(channel_ids)} specific channels")
        exporter.export(channel_ids=channel_ids)
    else:
        exporter.export()


if __name__ == "__main__":
    main()
