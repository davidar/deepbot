# Discord to IRC Log Exporter

A simple script to export Discord logs from the MongoDB database into IRC-style text logs.

## Features

- Exports Discord messages in a simple IRC-style format: `<username> message`
- Splits output into ~1MB files to keep files manageable
- Handles multiline messages by converting them to multiple single-line messages
- Resolves mentions to readable `@username` format
- Converts custom emojis to `:emoji_name:` format
- Includes attachments as `[attachment: filename]` placeholders
- Can list all available guilds in the database
- Can list all channels in a guild
- Supports exporting specific channels only
- Includes message dates in filenames for easy chronological sorting

## Usage

```bash
# List all available guilds
python discord_to_irc_exporter.py --list

# List all channels in a guild
python discord_to_irc_exporter.py <guild_id> --list-channels

# Basic usage with default output directory (./irc_logs/)
python discord_to_irc_exporter.py <guild_id>

# Export specific channels only
python discord_to_irc_exporter.py <guild_id> --channel <channel_id1> --channel <channel_id2>

# Specify an output directory
python discord_to_irc_exporter.py <guild_id> /path/to/output/dir

# Export specific channels to a custom directory
python discord_to_irc_exporter.py <guild_id> --channel <channel_id> /path/to/output/dir
```

When run without arguments, the script will automatically list available guilds.

## Output Format

The script creates one or more log files per channel, with names in the format `channelname_YYYY-MM-DD_N.log`, where:
- `channelname` is the Discord channel name
- `YYYY-MM-DD` is the date of the first message in that file
- `N` is a sequential counter (starting at 1)

Each file contains messages in the format:
```
<username> message text
<username> another message
```

## Requirements

- Python 3.7+
- Access to the MongoDB database containing Discord logs
- The `lorekeeper` package must be in your Python path

## Limitations

- Only includes text content and basic attachment references
- Does not include timestamps, reactions, or embeds
- All formatting (bold, italic, etc.) is removed
