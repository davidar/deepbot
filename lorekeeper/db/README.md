# TypedDatabase - Typesafe MongoDB Interface

This module provides a typesafe wrapper around the MongoDB database used in the Discord Chat Exporter Frontend. It uses Pydantic models to ensure type safety and data validation when interacting with the database.

## Features

- **Type Safety**: Full type hints and runtime validation through Pydantic models
- **IDE Autocompletion**: Get autocompletion for document fields in your IDE
- **Data Validation**: Runtime validation ensures data integrity
- **Self-Contained**: No external dependencies on other database modules
- **MongoDB Compatibility**: Seamlessly handles MongoDB's document format
- **Chain Methods**: Supports MongoDB's chaining methods like `sort()`, `limit()`, etc.

## Usage

### Basic Example

```python
from typed_database import TypedDatabase
from models import Message

# Get a typed collection of messages for a guild
guild_id = "748545324524575035"
messages_collection = TypedDatabase.get_messages_collection(guild_id)

# Find a message by ID - returns a fully typed Message object
message = messages_collection.find_one({"_id": "1045178587575242772"})

# Access message properties with type safety - notice we use 'id' not '_id'
print(f"Message from: {message.author.name}")
print(f"Message ID: {message.id}")  # Mapped from MongoDB's _id field
print(f"Posted at: {message.timestamp}")
print(f"In channel: {message.channelName}")

# Find messages with specific criteria and chain operations
pinned_messages = messages_collection.find({"isPinned": True}).sort("timestamp", -1).limit(10)
for msg in pinned_messages:
    # Each message is a typed Message object
    print(f"Pinned message: {msg.id}")

# Convert results to a list if needed
recent_messages = messages_collection.find({}).sort("timestamp", -1).limit(5).to_list()

# Insert a new message (read-only in tests, but shown for completeness)
# new_message = Message(id="1234567890123456", ...)
# messages_collection.insert_one(new_message)
```

### Available Collections

```python
# Get typed collections for different types of data
messages = TypedDatabase.get_messages_collection(guild_id)
channels = TypedDatabase.get_channels_collection(guild_id)
authors = TypedDatabase.get_authors_collection(guild_id)
assets = TypedDatabase.get_assets_collection(guild_id)
stickers = TypedDatabase.get_stickers_collection(guild_id)
guilds = TypedDatabase.get_guilds_collection()  # Global collection
```

### Data Models

The module includes Pydantic models for all MongoDB document types:

- `Message`: Discord messages
- `Channel`: Discord channels
- `Guild`: Discord guilds
- `Author`: Message authors
- `Asset`: Attachments and other assets
- `Emoji`: Emoji data
- `Reaction`: Message reactions
- `Embed`: Message embeds
- `Sticker`: Message stickers
- And more...

Each model uses Pydantic's field aliasing to map MongoDB's `_id` fields to more Pythonic `id` attributes:

```python
class Message(BaseModel):
    id: str = Field(alias="_id")  # Maps MongoDB's _id to id
    # other fields...
    
    model_config = ConfigDict(populate_by_name=True)  # Allows access through both id and _id
```

### Collection Methods

Each typed collection provides methods for common MongoDB operations:

- `find_one()`: Find a single document
- `find()`: Find multiple documents (returns a chainable `MongoResultsWrapper`)
- `insert_one()`: Insert a document
- `insert_many()`: Insert multiple documents
- `update_one()`: Update a document
- `update_many()`: Update multiple documents
- `delete_one()`: Delete a document
- `delete_many()`: Delete multiple documents
- `count_documents()`: Count documents matching a filter
- `distinct()`: Get distinct values for a field
- `create_index()`: Create an index on the collection
- `bulk_write()`: Perform a bulk write operation
- `aggregate()`: Perform an aggregation pipeline

### Results Wrapper

The `find()` method returns a `MongoResultsWrapper` that supports chaining operations:

```python
# Chain operations like in MongoDB
results = collection.find({}).sort("timestamp", -1).limit(10).skip(5)

# Convert to a list when needed
message_list = results.to_list()

# Or iterate directly
for message in results:
    print(message.content)
```

## MongoDB ID Handling

The module automatically handles MongoDB's ID padding requirements:

1. When querying by ID, you can use the unpadded ID, the module will handle padding:
   ```python
   # This works (automatically padded to 24 chars)
   message = messages_collection.find_one({"_id": "1045178587575242772"})
   ```

2. When accessing the ID, you get the padded version through the id attribute:
   ```python
   message.id  # Returns "000001045178587575242772"
   ```
