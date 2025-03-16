"""Local semantic search index for Discord messages."""

from typing import Any, Callable, Dict, List, Sequence

import chromadb
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from message_store import MessageStore, StoredMessage


class LocalDiscordIndex:
    """Local semantic search index for Discord messages."""

    def __init__(
        self,
        message_store: MessageStore,
        storage_path: str = "./chroma_db",
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize the search index.

        Args:
            message_store: The message store to index
            storage_path: Where to store the vector database
            model_name: Ollama model to use for embeddings
            base_url: URL of the Ollama server
        """
        # Set up embedding model
        Settings.embed_model = OllamaEmbedding(model_name=model_name, base_url=base_url)
        Settings.node_parser = SimpleNodeParser()

        # Set up vector store
        self.chroma_client = chromadb.PersistentClient(path=storage_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "discord_messages"
        )
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        # Initialize or load index
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        self.message_store = message_store

    def _message_to_text(self, message: StoredMessage) -> str:
        """Convert Discord message to searchable text."""
        parts = [f"Author: {message.author.name}", f"Message: {message.content}"]

        if message.embeds:
            parts.append("Embeds: " + "\n".join(str(e) for e in message.embeds))

        if message.attachments:
            parts.append(
                "Attachments: " + "\n".join(a.fileName for a in message.attachments)
            )

        return "\n".join(parts)

    def get_total_message_count(self) -> int:
        """Get the total number of messages to be indexed."""
        total = 0
        for channel_id in self.message_store.get_channel_ids():
            messages = self.message_store.get_channel_messages(channel_id)
            total += len(messages)
        return total

    def index_messages(
        self,
        batch_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Index all messages in batches.

        Args:
            batch_size: Number of messages to process in each batch
            progress_callback: Optional callback function(processed_count, total_count) for progress updates
        """
        documents: List[Document] = []
        total_indexed = 0
        total_messages = self.get_total_message_count()

        for channel_id in self.message_store.get_channel_ids():
            messages = self.message_store.get_channel_messages(channel_id)

            for message in messages:
                doc = Document(
                    text=self._message_to_text(message),
                    metadata={
                        "message_id": message.id,
                        "channel_id": channel_id,
                        "author": message.author.name,
                        "timestamp": message.timestamp,
                        "has_attachments": bool(message.attachments),
                        "has_embeds": bool(message.embeds),
                    },
                )
                documents.append(doc)

                # Index in batches
                if len(documents) >= batch_size:
                    self.index.insert_nodes(
                        Settings.node_parser.get_nodes_from_documents(documents)
                    )
                    total_indexed += len(documents)
                    if progress_callback:
                        progress_callback(total_indexed, total_messages)
                    documents = []

        # Index remaining documents
        if documents:
            self.index.insert_nodes(
                Settings.node_parser.get_nodes_from_documents(documents)
            )
            total_indexed += len(documents)
            if progress_callback:
                progress_callback(total_indexed, total_messages)

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
