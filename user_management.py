"""User management functionality for DeepBot."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

# Set up logging
logger = logging.getLogger("deepbot.user_management")


@dataclass
class UserRestriction:
    """Represents restrictions for a user."""

    ignored: bool = False
    consecutive_limit: Optional[int] = None  # Maximum consecutive messages allowed
    consecutive_count: int = 0  # Current consecutive message count


class UserManager:
    """Manages user restrictions like ignoring and rate limiting."""

    def __init__(self) -> None:
        """Initialize the user manager."""
        self.user_restrictions: Dict[int, UserRestriction] = {}
        self.last_message_user_id: Optional[int] = None
        self._load_restrictions()

    def _load_restrictions(self) -> None:
        """Load user restrictions from file."""
        try:
            if os.path.exists("user_restrictions.json"):
                with open("user_restrictions.json", "r") as f:
                    data = json.load(f)
                    for user_id_str, restrictions in data.items():
                        user_id = int(user_id_str)
                        self.user_restrictions[user_id] = UserRestriction(
                            **restrictions
                        )
                logger.info("Loaded user restrictions from file")
        except Exception as e:
            logger.error(f"Error loading user restrictions: {str(e)}")

    def _save_restrictions(self) -> None:
        """Save user restrictions to file."""
        try:
            data = {
                str(user_id): {
                    "ignored": r.ignored,
                    "consecutive_limit": r.consecutive_limit,
                    "consecutive_count": r.consecutive_count,
                }
                for user_id, r in self.user_restrictions.items()
            }
            with open("user_restrictions.json", "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved user restrictions to file")
        except Exception as e:
            logger.error(f"Error saving user restrictions: {str(e)}")

    def ignore_user(self, user_id: int) -> None:
        """Add a user to the ignore list.

        Args:
            user_id: The Discord user ID to ignore
        """
        if user_id not in self.user_restrictions:
            self.user_restrictions[user_id] = UserRestriction()
        self.user_restrictions[user_id].ignored = True
        self._save_restrictions()
        logger.info(f"Added user {user_id} to ignore list")

    def unignore_user(self, user_id: int) -> None:
        """Remove a user from the ignore list.

        Args:
            user_id: The Discord user ID to unignore
        """
        if user_id in self.user_restrictions:
            self.user_restrictions[user_id].ignored = False
            if not self.user_restrictions[user_id].consecutive_limit:
                del self.user_restrictions[user_id]
            self._save_restrictions()
            logger.info(f"Removed user {user_id} from ignore list")

    def set_consecutive_limit(
        self, user_id: int, consecutive_limit: Optional[int]
    ) -> None:
        """Set a consecutive message limit for a user.

        Args:
            user_id: The Discord user ID to limit
            consecutive_limit: Maximum consecutive messages allowed, or None to remove
        """
        if consecutive_limit is not None and consecutive_limit < 0:
            raise ValueError("Consecutive limit must be positive")

        if user_id not in self.user_restrictions:
            self.user_restrictions[user_id] = UserRestriction()

        restriction = self.user_restrictions[user_id]
        restriction.consecutive_limit = consecutive_limit
        restriction.consecutive_count = 0

        if not restriction.consecutive_limit and not restriction.ignored:
            del self.user_restrictions[user_id]

        self._save_restrictions()
        logger.info(f"Set consecutive limit for user {user_id} to {consecutive_limit}")

    def can_message(self, user_id: int) -> tuple[bool, Optional[str]]:
        """Check if a user is allowed to message based on restrictions.

        Args:
            user_id: The Discord user ID to check

        Returns:
            Tuple of (can_message, reason) where reason is None if allowed or a string explaining why not
        """
        # If no restrictions, allow message
        if user_id not in self.user_restrictions:
            self._update_last_user(user_id)
            return True, None

        restriction = self.user_restrictions[user_id]

        # If user is ignored, block message
        if restriction.ignored:
            return False, "You are currently ignored by the bot."

        # If no consecutive limit, allow message
        if restriction.consecutive_limit is None:
            self._update_last_user(user_id)
            return True, None

        # Reset counter if different user from last message
        if self.last_message_user_id != user_id:
            restriction.consecutive_count = 0

        # Check if user has exceeded consecutive limit
        if restriction.consecutive_count >= restriction.consecutive_limit:
            return (
                False,
                (
                    f"You've sent {restriction.consecutive_count} consecutive messages. "
                    "Please let others interact before sending more."
                ),
            )

        # Increment counter and allow message
        restriction.consecutive_count += 1
        self._update_last_user(user_id)
        self._save_restrictions()
        return True, None

    def _update_last_user(self, user_id: int) -> None:
        """Update the last message user and reset other users' consecutive counts.

        Args:
            user_id: The Discord user ID who sent the last message
        """
        # If this is a different user, reset consecutive counts for all other users
        if self.last_message_user_id != user_id:
            for uid, restriction in self.user_restrictions.items():
                if uid != user_id:
                    restriction.consecutive_count = 0

        self.last_message_user_id = user_id
        self._save_restrictions()

    def get_user_restrictions(self, user_id: int) -> Optional[UserRestriction]:
        """Get the current restrictions for a user.

        Args:
            user_id: The Discord user ID to check

        Returns:
            UserRestriction object if user has restrictions, None otherwise
        """
        return self.user_restrictions.get(user_id)
