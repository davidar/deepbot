from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Author(BaseModel):
    id: str = Field(alias="_id")
    name: str = Field(default="Unknown")  # Calculated from names or nicknames
    names: Optional[List[str]] = None  # Add the actual field from MongoDB
    nicknames: Optional[List[str]] = None  # Add the actual field from MongoDB
    discriminator: Optional[str] = None
    color: Optional[str] = None
    isBot: Optional[bool] = False
    avatar: Optional[Dict[str, Any]] = None
    roles: Optional[List[Dict[str, Any]]] = Field(default_factory=lambda: [])
    msg_count: Optional[int] = 0
    guildIds: Optional[List[str]] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def calculate_name(self) -> Any:
        """Calculate name from names or nicknames if not already set"""
        if self.name == "Unknown":
            if self.names and len(self.names) > 0:
                # Use the first name (without discriminator)
                full_name = self.names[0]
                if "#" in full_name:
                    self.name = full_name.split("#")[0]
                else:
                    self.name = full_name

                # Extract discriminator if present
                if "#" in full_name and not self.discriminator:
                    self.discriminator = full_name.split("#")[1]
            elif self.nicknames and len(self.nicknames) > 0:
                # Use the first nickname
                self.name = self.nicknames[0]
        return self


class Reference(BaseModel):
    messageId: str
    channelId: str
    guildId: str


class Emoji(BaseModel):
    id: str = Field(alias="_id")
    name: str
    code: Optional[str] = None
    isAnimated: Optional[bool] = False
    guildIds: Optional[List[str]] = None
    source: Optional[str] = None
    image: Optional[Dict[str, Any]] = None
    usage_count: Optional[int] = 0

    model_config = ConfigDict(populate_by_name=True)


class Role(BaseModel):
    id: str = Field(alias="_id")
    name: str
    color: Optional[str] = None
    position: Optional[int] = 0
    guildId: Optional[str] = None
    exportedAt: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class User(BaseModel):
    id: str = Field(alias="_id")
    name: Optional[str] = None
    avatar: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class Reaction(BaseModel):
    emoji: Emoji
    count: int
    users: Optional[List[User]] = None


class Asset(BaseModel):
    id: str = Field(alias="_id")
    originalPath: Optional[str] = None
    localPath: Optional[str] = None
    remotePath: Optional[str] = None
    path: Optional[str] = None
    extension: Optional[str] = None
    type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sizeBytes: Optional[int] = None
    filenameWithHash: Optional[str] = None
    filenameWithoutHash: Optional[str] = None
    colorDominant: Optional[str] = None
    colorPalette: Optional[List[str]] = None
    searchable: Optional[bool] = False

    model_config = ConfigDict(populate_by_name=True)


class EmbedField(BaseModel):
    name: str
    value: str
    inline: Optional[bool] = False


class Embed(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[str] = None
    color: Optional[str] = None
    footer: Optional[Dict[str, Any]] = None
    image: Optional[Dict[str, Any]] = None
    thumbnail: Optional[Dict[str, Any]] = None
    video: Optional[Dict[str, Any]] = None
    provider: Optional[Dict[str, Any]] = None
    author: Optional[Dict[str, Any]] = None
    fields: Optional[List[EmbedField]] = None


class Sticker(BaseModel):
    id: str = Field(alias="_id")
    name: Optional[str] = None
    format: Optional[str] = None
    source: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class Mention(BaseModel):
    id: str = Field(alias="_id")
    name: Optional[str] = None
    nickname: Optional[str] = None
    isBot: Optional[bool] = False

    model_config = ConfigDict(populate_by_name=True)


class MessageContent(BaseModel):
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Channel(BaseModel):
    id: str = Field(alias="_id")
    type: str
    categoryId: Optional[str] = None
    category: Optional[str] = None
    name: str
    topic: Optional[str] = None
    guildId: Optional[str] = None
    exportedAt: Optional[str] = None
    msg_count: Optional[int] = 0

    model_config = ConfigDict(populate_by_name=True)


class Message(BaseModel):
    id: str = Field(alias="_id")
    type: str
    timestamp: str
    timestampEdited: Optional[str] = None
    callEndedTimestamp: Optional[str] = None
    isPinned: bool = False
    content: Union[
        str, List[MessageContent]
    ]  # Can be string or list of MessageContent objects
    author: Author
    stickers: Optional[List[Sticker]] = Field(default_factory=lambda: [])
    reactions: Optional[List[Reaction]] = Field(default_factory=lambda: [])
    mentions: Optional[List[Mention]] = Field(default_factory=lambda: [])
    attachments: Optional[List[Asset]] = Field(default_factory=lambda: [])
    embeds: Optional[List[Embed]] = Field(default_factory=lambda: [])
    reference: Optional[Reference] = None
    guildId: str
    channelId: str
    channelName: str
    thread: Optional[Channel] = None
    exportedAt: Optional[str] = None
    sources: Optional[List[str]] = Field(default_factory=lambda: [])
    isDeleted: Optional[bool] = False

    model_config = ConfigDict(populate_by_name=True)

    # Add a validator to handle different content formats
    @model_validator(mode="before")
    @classmethod
    def validate_content(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        # No need to check if data is dict - it's guaranteed by the type annotation

        # If content is a list of dictionaries, ensure they're properly formatted
        content_any: Any = data.get("content")
        # Check if content is a list
        if not isinstance(content_any, list):
            return data

        # Now we know content is a list, but we don't know its element type
        content: List[Any] = content_any

        # Only proceed if the list has items
        if not content:
            return data

        first_item: Any = content[0]
        if isinstance(first_item, dict):
            # Ensure each content item has the required fields
            if "content" in first_item:
                # Content is already in the expected format
                return data

        # Convert the format if needed - handle string content items
        # Check if all items are strings
        for item_to_check in content:
            if not isinstance(item_to_check, str):
                # If any item is not a string, don't modify the content
                return data

        # If we reach here, all items are strings, so we can safely cast to List[str]
        string_content: List[str] = [str(item) for item in content]
        data["content"] = [{"content": item_str} for item_str in string_content]
        return data


class Guild(BaseModel):
    id: str = Field(alias="_id")
    name: str
    icon: Optional[Dict[str, Any]] = None
    msg_count: Optional[int] = 0
    exportedAt: Optional[str] = None
    exported_at: Optional[str] = None  # Some documents use this field name instead

    model_config = ConfigDict(populate_by_name=True)
