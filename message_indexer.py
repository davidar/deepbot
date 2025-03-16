"""Vector store management and search functionality for Discord messages."""

import logging
from typing import Any, Dict, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding  # pyright: ignore
from llama_index.vector_stores.chroma import ChromaVectorStore  # pyright: ignore

from discord_types import StoredMessage

# Set up logging
logger = logging.getLogger("deepbot.message_indexer")


class MessageIndexer:
    """Manages vector store indexing and search for Discord messages."""

    def __init__(
        self,
        storage_path: str,
        model_name: str,
        base_url: str,
    ) -> None:
        """Initialize the message indexer.

        Args:
            storage_path: Where to store the vector database
            model_name: Ollama model to use for embeddings
            base_url: URL of the Ollama server
        """
        # Set up embedding model
        Settings.embed_model = OllamaEmbedding(
            model_name=model_name,
            base_url=base_url,
        )
        Settings.node_parser = SimpleNodeParser()

        # Set up vector store
        self.chroma_client = chromadb.PersistentClient(
            path=storage_path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "discord_messages"
        )
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Initialize or load index
        self.index = VectorStoreIndex.from_vector_store(vector_store)  # pyright: ignore

    def message_to_text(self, message: StoredMessage) -> str:
        """Convert Discord message to searchable text.

        Args:
            message: The stored message to convert

        Returns:
            Text representation of the message
        """
        parts = [f"Author: {message.author.name}", f"Message: {message.content}"]

        if message.embeds:
            parts.append("Embeds: " + "\n".join(str(e) for e in message.embeds))

        if message.attachments:
            parts.append(
                "Attachments: " + "\n".join(a.fileName for a in message.attachments)
            )

        return "\n".join(parts)

    def index_message(self, message: StoredMessage, channel_id: str) -> None:
        """Index a message in the vector store.

        Args:
            message: The message to index
            channel_id: The ID of the channel containing the message
        """
        doc = Document(
            text=self.message_to_text(message),
            metadata={
                "message_id": message.id,
                "channel_id": channel_id,
                "author": message.author.name,
                "timestamp": message.timestamp,
                "has_attachments": bool(message.attachments),
                "has_embeds": bool(message.embeds),
            },
        )
        self.index.insert_nodes(Settings.node_parser.get_nodes_from_documents([doc]))

    async def search(
        self, query: str, top_k: int = 5, **filters: Dict[str, Any]
    ) -> Sequence[NodeWithScore]:
        """Search for messages matching the query.

        Args:
            query: The search query
            top_k: Maximum number of results to return
            **filters: Optional filters to apply (e.g. channel_id, author_id)

        Returns:
            List of matching nodes with their relevance scores
        """
        retriever = self.index.as_retriever(
            similarity_top_k=top_k, filters=filters if filters else None
        )

        nodes = await retriever.aretrieve(query)
        return nodes
