"""Command handlers for reaction management."""

import discord
from discord.ext import commands

from reactions import ReactionManager

Context = commands.Context[commands.Bot]


class ReactionCommands:
    """Handlers for reaction commands."""

    def __init__(self, reaction_manager: ReactionManager) -> None:
        """Initialize reaction command handlers.

        Args:
            reaction_manager: The reaction manager instance
        """
        self.reaction_manager = reaction_manager

    async def handle_reactions(self, ctx: Context, scope: str = "channel") -> None:
        """Handle the reactions command.

        Args:
            ctx: The command context
            scope: Either "channel" (default) or "global" to show stats across all channels
        """
        if scope.lower() not in ["channel", "global"]:
            await ctx.send('-# Invalid scope. Use "channel" or "global"')
            return

        if scope.lower() == "global":
            channel_scores = self.reaction_manager.get_global_stats()
            if not channel_scores:
                await ctx.send("-# No reaction data available yet")
                return

            message = ["-# Global reaction statistics:"]
            summary = self.reaction_manager.format_global_summary(channel_scores)
            for line in summary.split("\n"):
                if line.strip():
                    message.append(f"-# {line}")
            await ctx.send("\n".join(message))

        else:  # channel scope
            channel_id = ctx.channel.id
            message_reactions = self.reaction_manager.get_channel_stats(channel_id)

            if not message_reactions:
                await ctx.send("-# No reaction data available for this channel yet")
                return

            # Create a summary of reactions
            channel_name = (
                ctx.channel.name
                if isinstance(ctx.channel, discord.TextChannel)
                else "DM"
            )
            message = [f"-# Reaction statistics for #{channel_name}"]
            channel_summary = self.reaction_manager.format_reaction_summary(
                message_reactions
            )
            if channel_summary:
                for line in channel_summary.split("\n"):
                    if line.strip():
                        message.append(f"-# {line}")
            else:
                message.append("-# No reactions yet.")
            await ctx.send("\n".join(message))
