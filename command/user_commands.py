"""Command handlers for user management."""

from typing import Optional

import discord
from discord.ext import commands

from user_management import UserManager

Context = commands.Context[commands.Bot]


class UserCommands:
    """Handlers for user management commands."""

    def __init__(self, user_manager: UserManager) -> None:
        """Initialize user command handlers.

        Args:
            user_manager: The user manager instance
        """
        self.user_manager = user_manager

    async def handle_ignore(self, ctx: Context, member: discord.Member) -> None:
        """Handle the ignore command.

        Args:
            ctx: The command context
            member: The Discord member to ignore
        """
        self.user_manager.ignore_user(member.id)
        await ctx.send(f"-# Now ignoring messages from {member.display_name}")

    async def handle_unignore(self, ctx: Context, member: discord.Member) -> None:
        """Handle the unignore command.

        Args:
            ctx: The command context
            member: The Discord member to unignore
        """
        self.user_manager.unignore_user(member.id)
        await ctx.send(f"-# No longer ignoring messages from {member.display_name}")

    async def handle_limit(
        self,
        ctx: Context,
        member: discord.Member,
        consecutive_limit: Optional[int] = None,
    ) -> None:
        """Handle the limit command.

        Args:
            ctx: The command context
            member: The Discord member to limit
            consecutive_limit: Maximum consecutive messages allowed, or None to remove
        """
        try:
            self.user_manager.set_consecutive_limit(member.id, consecutive_limit)
            if consecutive_limit is None:
                await ctx.send(f"-# Removed message limit for {member.display_name}")
            else:
                await ctx.send(
                    f"-# Set consecutive message limit for {member.display_name} to {consecutive_limit} messages"
                )
        except ValueError as e:
            await ctx.send(f"-# Error: {str(e)}")

    async def handle_restrictions(self, ctx: Context, member: discord.Member) -> None:
        """Handle the restrictions command.

        Args:
            ctx: The command context
            member: The Discord member to check
        """
        restrictions = self.user_manager.get_user_restrictions(member.id)
        if not restrictions:
            await ctx.send(f"-# No restrictions set for {member.display_name}")
            return

        message = [f"-# Current restrictions for {member.display_name}:"]
        if restrictions.ignored:
            message.append("-# • User is ignored")
        if restrictions.consecutive_limit is not None:
            message.append(
                f"-# • Limited to {restrictions.consecutive_limit} consecutive messages"
            )
            if restrictions.consecutive_count > 0:
                message.append(
                    f"-# • Currently at {restrictions.consecutive_count} consecutive messages"
                )

        await ctx.send("\n".join(message))
