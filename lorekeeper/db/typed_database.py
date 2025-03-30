from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pymongo
from pydantic import BaseModel
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.database import Database
from pymongo.mongo_client import MongoClient

from .models import Asset, Author, Channel, Emoji, Guild, Message, Role, Sticker

# MongoDB connection settings - inlined from Database.py
URI = "mongodb://127.0.0.1:27017"
client: MongoClient[Dict[str, Any]] = pymongo.MongoClient(URI)
db: Database[Dict[str, Any]] = client["dcef"]
collection_config: Collection[Dict[str, Any]] = db["config"]

T = TypeVar("T", bound=BaseModel)


def pad_id(id_value: Optional[str]) -> Optional[str]:
    """Pad an ID to 24 characters with leading zeros."""
    if id_value is None:
        return None
    return str(id_value).zfill(24)


class MongoResultsWrapper(Generic[T]):
    """
    A wrapper around MongoDB cursor results that preserves sorting capabilities.
    """

    def __init__(self, cursor: Cursor[Dict[str, Any]], model_class: Type[T]) -> None:
        self.cursor = cursor
        self.model_class = model_class
        self._results = None

    def __iter__(self) -> Iterator[T]:
        """Iterate through the results, converting each document to a Pydantic model"""
        for doc in self.cursor:
            yield self.model_class(**doc)

    def __len__(self) -> int:
        """Return the count of documents"""
        # Using a different approach instead of the deprecated count method
        return sum(1 for _ in self.cursor.clone())

    def sort(
        self,
        key_or_list: Union[str, List[Tuple[str, int]]],
        direction: Optional[int] = None,
    ) -> "MongoResultsWrapper[T]":
        """Sort the results"""
        self.cursor = self.cursor.sort(key_or_list, direction)
        return self

    def limit(self, limit: int) -> "MongoResultsWrapper[T]":
        """Limit the number of results"""
        self.cursor = self.cursor.limit(limit)
        return self

    def skip(self, skip: int) -> "MongoResultsWrapper[T]":
        """Skip results"""
        self.cursor = self.cursor.skip(skip)
        return self

    def to_list(self) -> List[T]:
        """Convert the cursor to a list of Pydantic models"""
        return [self.model_class(**doc) for doc in self.cursor]


class TypedCollection(Generic[T]):
    """
    A typesafe wrapper around a MongoDB collection that uses Pydantic models.
    """

    def __init__(
        self, collection: Collection[Dict[str, Any]], model_class: Type[T]
    ) -> None:
        self.collection = collection
        self.model_class = model_class

    def find_one(self, query: Dict[str, Any], *args: Any, **kwargs: Any) -> Optional[T]:
        """Find a single document and return it as a Pydantic model"""
        # Ensure IDs are properly padded
        query = self._process_query_ids(query)

        result = self.collection.find_one(query, *args, **kwargs)
        if result is None:
            return None
        return self.model_class(**result)

    def find(
        self, query: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> MongoResultsWrapper[T]:
        """Find documents and return them as Pydantic models or a wrapper for further operations"""
        # Ensure IDs are properly padded
        query = self._process_query_ids(query)

        cursor = self.collection.find(query, *args, **kwargs)

        # Return a wrapper that allows for chaining .sort(), .limit(), etc.
        return MongoResultsWrapper(cursor, self.model_class)

    def insert_one(
        self, document: Union[Dict[str, Any], T], *args: Any, **kwargs: Any
    ) -> str:
        """Insert a document and return its ID"""
        if isinstance(document, BaseModel):
            doc_dict = document.model_dump(exclude_unset=True)
        else:
            doc_dict = document
        result = self.collection.insert_one(doc_dict, *args, **kwargs)
        return str(result.inserted_id)

    def insert_many(
        self, documents: List[Union[Dict[str, Any], T]], *args: Any, **kwargs: Any
    ) -> List[str]:
        """Insert multiple documents and return their IDs"""
        doc_dicts: List[Dict[str, Any]] = []
        for doc in documents:
            if isinstance(doc, BaseModel):
                doc_dicts.append(doc.model_dump(exclude_unset=True))
            else:
                doc_dicts.append(doc)
        result = self.collection.insert_many(doc_dicts, *args, **kwargs)
        return [str(id) for id in result.inserted_ids]

    def update_one(
        self, filter: Dict[str, Any], update: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> int:
        """Update a document and return the number of documents modified"""
        # Ensure IDs are properly padded
        filter = self._process_query_ids(filter)

        result = self.collection.update_one(filter, update, *args, **kwargs)
        return result.modified_count

    def update_many(
        self, filter: Dict[str, Any], update: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> int:
        """Update multiple documents and return the number of documents modified"""
        # Ensure IDs are properly padded
        filter = self._process_query_ids(filter)

        result = self.collection.update_many(filter, update, *args, **kwargs)
        return result.modified_count

    def delete_one(self, filter: Dict[str, Any], *args: Any, **kwargs: Any) -> int:
        """Delete a document and return the number of documents deleted"""
        # Ensure IDs are properly padded
        filter = self._process_query_ids(filter)

        result = self.collection.delete_one(filter, *args, **kwargs)
        return result.deleted_count

    def delete_many(self, filter: Dict[str, Any], *args: Any, **kwargs: Any) -> int:
        """Delete multiple documents and return the number of documents deleted"""
        # Ensure IDs are properly padded
        filter = self._process_query_ids(filter)

        result = self.collection.delete_many(filter, *args, **kwargs)
        return result.deleted_count

    def count_documents(self, filter: Dict[str, Any], *args: Any, **kwargs: Any) -> int:
        """Count documents matching a filter"""
        # Ensure IDs are properly padded
        filter = self._process_query_ids(filter)

        return self.collection.count_documents(filter, *args, **kwargs)

    def distinct(
        self,
        key: str,
        filter: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """Get distinct values for a field"""
        if filter:
            # Ensure IDs are properly padded
            filter = self._process_query_ids(filter)

        return self.collection.distinct(key, filter, *args, **kwargs)

    def create_index(
        self, keys: Union[str, List[Tuple[str, int]]], **kwargs: Any
    ) -> str:
        """Create an index on the collection"""
        return self.collection.create_index(keys, **kwargs)

    def bulk_write(self, operations: List[Any], *args: Any, **kwargs: Any) -> Any:
        """Perform a bulk write operation"""
        return self.collection.bulk_write(operations, *args, **kwargs)

    def aggregate(
        self, pipeline: List[Dict[str, Any]], *args: Any, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Perform an aggregation pipeline"""
        # Process any $match stages to ensure IDs are properly padded
        processed_pipeline: List[Dict[str, Any]] = []
        for stage in pipeline:
            if "$match" in stage:
                processed_pipeline.append(
                    {"$match": self._process_query_ids(stage["$match"])}
                )
            else:
                processed_pipeline.append(stage)

        return list(self.collection.aggregate(processed_pipeline, *args, **kwargs))

    def _process_query_ids(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure IDs in the query are properly padded"""
        if not query:
            return query

        processed_query = query.copy()

        # Common ID fields that might need padding
        id_fields = [
            "_id",
            "channelId",
            "guildId",
            "messageId",
            "authorId",
            "author._id",
        ]

        for field in id_fields:
            if field in processed_query and isinstance(processed_query[field], str):
                processed_query[field] = pad_id(processed_query[field])
            elif field in processed_query and isinstance(processed_query[field], dict):
                # Handle operators like $in, $eq, etc.
                for op, value in processed_query[field].items():
                    if op == "$in" and isinstance(value, list):
                        processed_query[field][op] = [
                            pad_id(str_v) if isinstance(str_v, str) else str_v
                            for str_v in value
                        ]
                    elif op in ["$eq", "$gt", "$gte", "$lt", "$lte"] and isinstance(
                        value, str
                    ):
                        processed_query[field][op] = pad_id(value)

        return processed_query


class TypedDatabase:
    """
    A typesafe wrapper around the MongoDB database that uses Pydantic models.
    """

    @staticmethod
    def is_online() -> bool:
        """Check if the database is online"""
        try:
            client.server_info()
            return True
        except Exception:
            return False

    @staticmethod
    def get_global_collection(collection_name: str) -> Collection[Dict[str, Any]]:
        """Get a global collection by name (untyped)"""
        return db[collection_name]

    @staticmethod
    def get_allowlisted_guild_ids() -> List[str]:
        """Get allowlisted guild IDs"""
        try:
            config_doc = collection_config.find_one({"key": "allowlisted_guild_ids"})
            if config_doc is None or "value" not in config_doc:
                return []
            allowlisted_guild_ids = config_doc["value"]
            return [str(pad_id(id)) for id in allowlisted_guild_ids if id is not None]
        except (TypeError, KeyError):
            # If no allowlist exists, return an empty list
            return []

    @staticmethod
    def get_denylisted_user_ids() -> List[str]:
        """Get denylisted user IDs"""
        try:
            config_doc = collection_config.find_one({"key": "denylisted_user_ids"})
            if config_doc is None or "value" not in config_doc:
                return []
            denylisted_user_ids = config_doc["value"]
            return [str(pad_id(id)) for id in denylisted_user_ids if id is not None]
        except (TypeError, KeyError):
            # If no denylist exists, return an empty list
            return []

    @staticmethod
    def get_guild_collection(
        guild_id: str, collection_name: str
    ) -> Collection[Dict[str, Any]]:
        """Get a guild collection by name (untyped)"""
        allowlisted_guild_ids = TypedDatabase.get_allowlisted_guild_ids()
        padded_guild_id = pad_id(guild_id)

        # Skip allowlist check if the list is empty
        if (
            len(allowlisted_guild_ids) > 0
            and padded_guild_id not in allowlisted_guild_ids
        ):
            raise Exception(f"Guild {guild_id} not allowlisted")

        return db[f"g{padded_guild_id}_{collection_name}"]

    @staticmethod
    def get_typed_guild_collection(
        guild_id: str, collection_name: str, model_class: Type[T]
    ) -> TypedCollection[T]:
        """Get a typed guild collection with a specific model class"""
        collection = TypedDatabase.get_guild_collection(guild_id, collection_name)
        return TypedCollection(collection, model_class)

    @staticmethod
    def get_messages_collection(guild_id: str) -> TypedCollection[Message]:
        """Get a typed collection of messages for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "messages", Message)

    @staticmethod
    def get_channels_collection(guild_id: str) -> TypedCollection[Channel]:
        """Get a typed collection of channels for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "channels", Channel)

    @staticmethod
    def get_authors_collection(guild_id: str) -> TypedCollection[Author]:
        """Get a typed collection of authors for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "authors", Author)

    @staticmethod
    def get_assets_collection(guild_id: str) -> TypedCollection[Asset]:
        """Get a typed collection of assets for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "assets", Asset)

    @staticmethod
    def get_stickers_collection(guild_id: str) -> TypedCollection[Sticker]:
        """Get a typed collection of stickers for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "stickers", Sticker)

    @staticmethod
    def get_roles_collection(guild_id: str) -> TypedCollection[Role]:
        """Get a typed collection of roles for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "roles", Role)

    @staticmethod
    def get_emojis_collection(guild_id: str) -> TypedCollection[Emoji]:
        """Get a typed collection of emojis for a guild"""
        return TypedDatabase.get_typed_guild_collection(guild_id, "emojis", Emoji)

    @staticmethod
    def get_guilds_collection() -> TypedCollection[Guild]:
        """Get a typed collection of guilds"""
        collection = TypedDatabase.get_global_collection("guilds")
        return TypedCollection(collection, Guild)
