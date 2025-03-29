"""Vector search functionality for lorekeeper."""

import logging
import os
from typing import Any, Dict, List

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# HuggingFace for embeddings
from sentence_transformers import SentenceTransformer

# Import config
from . import config

# Set up logging
logger = logging.getLogger("deepbot.lorekeeper")


class VectorSearch:
    """Standalone vector search functionality"""

    def __init__(self, qdrant_host: str = None, qdrant_port: int = None):
        """Initialize with direct access to embeddings and storage

        Args:
            qdrant_host: The Qdrant server host (overrides config)
            qdrant_port: The Qdrant server port (overrides config)
        """
        # Use provided values or defaults from config
        qdrant_config = config.get_qdrant_config()
        self.qdrant_host = qdrant_host or qdrant_config["host"]
        self.qdrant_port = qdrant_port or qdrant_config["port"]
        self.collection_name = qdrant_config["collection"]

        # Create embedding model
        logger.info(f"Initializing embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            cache_folder=os.path.join(config.PERSIST_DIR, "embedding_model"),
        )

        # Create qdrant client
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        logger.info(
            f"Initialized vector search with Qdrant at {self.qdrant_host}:{self.qdrant_port}"
        )

        # Check if collection exists
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [coll.name for coll in collections.collections]
            if self.collection_name in collection_names:
                # Get collection info
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name
                )
                points_count = collection_info.points_count
                logger.info(
                    f"Found collection '{self.collection_name}' with {points_count} points"
                )
            else:
                logger.error(
                    f"Collection '{self.collection_name}' does not exist in Qdrant"
                )
                all_collections = (
                    ", ".join(collection_names) if collection_names else "none"
                )
                logger.error(f"Available collections: {all_collections}")
        except Exception as e:
            logger.error(f"Error checking collection: {e}")

    def search(
        self, query: str, limit: int = None, similarity_cutoff: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for messages using semantic search

        Args:
            query: The search query
            limit: Maximum number of results to return
            similarity_cutoff: Minimum similarity score threshold (0-1)

        Returns:
            List of messages matching the query ordered by relevance
        """
        # Use provided values or defaults from config
        limit = limit or config.DEFAULT_SEARCH_LIMIT
        similarity_cutoff = similarity_cutoff or config.DEFAULT_SIMILARITY_CUTOFF

        logger.info(
            f"Searching for '{query}' with limit={limit}, similarity_cutoff={similarity_cutoff}"
        )

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                config.BGE_QUERY_PREFIX + query, show_progress_bar=False
            )

            # Perform the search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=similarity_cutoff,
            )

            if not search_results:
                logger.warning(
                    f"No results found for query '{query}' with similarity threshold {similarity_cutoff}"
                )
                # Try again with lower threshold for debugging
                if similarity_cutoff > 0.2:
                    debug_results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding.tolist(),
                        limit=5,
                        score_threshold=0.2,
                    )
                    if debug_results:
                        logger.info(
                            f"With lower threshold (0.2), found {len(debug_results)} results"
                        )
                return []

            # Convert search results to a standard format
            formatted_results = []
            for result in search_results:
                payload = result.payload
                formatted_results.append(
                    {
                        "message_id": payload.get("message_id"),
                        "channel_id": payload.get("channel_id"),
                        "guild_id": payload.get("guild_id"),
                        "content": payload.get("text"),
                        "author_name": payload.get("author_name"),
                        "timestamp": payload.get("timestamp"),
                        "point_id": str(result.id),
                        "vector_score": result.score,
                        "has_reply_chain": payload.get("has_reply_chain", False),
                        "has_preceding_messages": payload.get(
                            "has_preceding_messages", False
                        ),
                        "has_embeds": payload.get("has_embeds", False),
                    }
                )

            logger.info(
                f"Search for '{query}' returned {len(formatted_results)} results with scores ranging from {formatted_results[0]['vector_score']:.4f} to {formatted_results[-1]['vector_score']:.4f}"
            )
            return formatted_results

        except UnexpectedResponse as e:
            logger.error(f"Unexpected response from Qdrant: {e}")
            try:
                # Check if collection exists
                collections = self.qdrant_client.get_collections()
                collection_names = [coll.name for coll in collections.collections]
                if self.collection_name not in collection_names:
                    logger.error(
                        f"Collection '{self.collection_name}' does not exist. Available collections: {', '.join(collection_names)}"
                    )
            except Exception as inner_e:
                logger.error(
                    f"Error checking collections after failed search: {inner_e}"
                )
            raise
        except Exception as e:
            logger.exception(f"Error searching: {e}")
            raise
