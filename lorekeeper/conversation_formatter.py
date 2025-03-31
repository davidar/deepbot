"""Conversation formatting utilities for lorekeeper."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Set

logger = logging.getLogger("deepbot.lorekeeper")


def format_timestamp(timestamp_str: str) -> str:
    """Format a Discord timestamp string to a human-readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {str(e)}")
        return timestamp_str


def extract_conversation_fragments(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract conversation fragments from search results.

    Args:
        results: The search results from vector search

    Returns:
        List of conversation fragments
    """
    conversation_fragments: List[Dict[str, Any]] = []

    # Skip if no results
    if not results:
        logger.warning("No results provided to extract_conversation_fragments")
        return conversation_fragments

    # Extract fragments from each result
    for result_idx, result in enumerate(results, 1):
        if "content" not in result or not result["content"]:
            logger.warning(f"Result {result_idx} has no content field or empty content")
            continue

        # Parse the context into individual messages with authors
        messages: List[Dict[str, Any]] = []
        context_lines = result["content"].strip().split("\n\n")

        # Get the result metadata
        result_metadata = {
            "result_index": result_idx,
            "vector_score": result.get("vector_score", 0),
            "has_reply_chain": result.get("has_reply_chain", False),
            "has_preceding_messages": result.get("has_preceding_messages", False),
            "has_embeds": result.get("has_embeds", False),
            "timestamp": result.get("timestamp", ""),
            "author_name": result.get("author_name", "Unknown User"),
        }

        # Process each line in the context
        for i, line in enumerate(context_lines):
            if not line:
                continue

            # Try to extract author and content
            author = "Unknown"
            content = line

            if ": " in line:
                author, content = line.split(": ", 1)
            else:
                logger.warning(
                    f"Could not extract author from line in result {result_idx}, message {i+1}"
                )

            messages.append(
                {
                    "author": author,
                    "content": content,
                    "is_last": (
                        i == len(context_lines) - 1
                    ),  # Flag if this is the result's matched message
                }
            )

        # Add this fragment to our collection
        conversation_fragments.append(
            {"messages": messages, "metadata": result_metadata}
        )

    logger.info(f"Extracted {len(conversation_fragments)} conversation fragments")
    return conversation_fragments


def merge_conversation_fragments(
    fragments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge overlapping conversation fragments into coherent conversations.

    Args:
        fragments: List of conversation fragments to merge

    Returns:
        List of merged conversations
    """
    # Skip if no fragments
    if not fragments:
        logger.warning("No fragments provided to merge_conversation_fragments")
        return []

    # Now group fragments into conversations by finding overlaps
    conversations: List[Dict[str, Any]] = []

    # Simple clustering of fragments based on common messages
    processed_indices: Set[int] = set()

    for i, fragment in enumerate(fragments):
        if i in processed_indices:
            continue

        # Start a new conversation with this fragment
        current_conversation: Dict[str, List[Any]] = {
            "messages": [],
            "matched_messages": [],  # Messages with metadata
        }

        # Add the fragment's messages to the conversation
        for msg in fragment["messages"]:
            msg_key = f"{msg['author']}:{msg['content']}"

            # Check if this message is already in the conversation
            existing_msg = next(
                (
                    m
                    for m in current_conversation["messages"]
                    if f"{m['author']}:{m['content']}" == msg_key
                ),
                None,
            )

            if not existing_msg:
                msg_copy = msg.copy()
                current_conversation["messages"].append(msg_copy)

                # If this is the matched message from the result, add it to matched_messages
                if msg["is_last"]:
                    current_conversation["matched_messages"].append(
                        {
                            "message_index": len(current_conversation["messages"]) - 1,
                            "metadata": fragment["metadata"],
                        }
                    )

        # Look for overlapping fragments to merge into this conversation
        merged_count = 0
        for j, other_fragment in enumerate(fragments):
            if j == i or j in processed_indices:
                continue

            # Check if there's significant overlap with this fragment
            # (at least one common message)
            has_overlap = False
            for msg in other_fragment["messages"]:
                msg_key = f"{msg['author']}:{msg['content']}"
                if any(
                    f"{m['author']}:{m['content']}" == msg_key
                    for m in current_conversation["messages"]
                ):
                    has_overlap = True
                    break

            if has_overlap:
                merged_count += 1

                # Add any new messages from this fragment
                for msg in other_fragment["messages"]:
                    msg_key = f"{msg['author']}:{msg['content']}"

                    # Check if already in conversation
                    existing_msg = next(
                        (
                            m
                            for m in current_conversation["messages"]
                            if f"{m['author']}:{m['content']}" == msg_key
                        ),
                        None,
                    )

                    if not existing_msg:
                        msg_copy = msg.copy()
                        current_conversation["messages"].append(msg_copy)

                        # If this is the matched message, add it to matched_messages
                        if msg["is_last"]:
                            current_conversation["matched_messages"].append(
                                {
                                    "message_index": len(
                                        current_conversation["messages"]
                                    )
                                    - 1,
                                    "metadata": other_fragment["metadata"],
                                }
                            )
                    else:
                        # If already exists but is a matched message in this fragment,
                        # add the metadata
                        if msg["is_last"]:
                            msg_index = current_conversation["messages"].index(
                                existing_msg
                            )
                            current_conversation["matched_messages"].append(
                                {
                                    "message_index": msg_index,
                                    "metadata": other_fragment["metadata"],
                                }
                            )

                processed_indices.add(j)

        # Add the processed fragment
        processed_indices.add(i)
        conversations.append(current_conversation)

    logger.info(f"Merged fragments into {len(conversations)} conversations")
    return conversations


def format_lore_context(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a context string for the LLM prompt.

    Args:
        results: The search results

    Returns:
        Formatted context string
    """
    if not results:
        logger.warning("No results provided to format_lore_context")
        return "No records found."

    # Extract and merge conversation fragments
    fragments = extract_conversation_fragments(results)
    conversations = merge_conversation_fragments(fragments)

    # Sort conversations by relevance (highest vector_score of any matched message)
    for conversation in conversations:
        max_score = 0
        for match_info in conversation["matched_messages"]:
            score = match_info["metadata"]["vector_score"]
            if score > max_score:
                max_score = score
        conversation["max_score"] = max_score

    # Sort conversations by max score, highest first
    conversations.sort(key=lambda x: x.get("max_score", 0), reverse=True)

    # Limit to top 6 conversations
    if len(conversations) > 6:
        conversations = conversations[:6]

    # Format conversations into context string
    context_text = ""

    for conv_idx, conversation in enumerate(conversations, 1):
        # Get the approximate timestamp of the conversation
        approximate_timestamp = format_timestamp(
            conversation["matched_messages"][0]["metadata"]["timestamp"]
        )

        # Add a conversation header
        context_text += f"# CONVERSATION {conv_idx} ({approximate_timestamp})\n\n"

        # Include all messages in the conversation with proper formatting
        for message in conversation["messages"]:
            author = message["author"]
            content = message["content"]
            content = content.replace("\n", " ")
            context_text += f"@{author} said: {content}\n\n"

        # Add a separator between conversations
        context_text += "---\n\n"

    logger.info(f"Formatted context of length {len(context_text)}")
    return context_text.strip()
