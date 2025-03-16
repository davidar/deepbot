"""Command handlers for model options."""

from typing import Optional

import discord
from discord.ext import commands

import config

Context = commands.Context[commands.Bot]


class OptionCommands:
    """Handlers for model option commands."""

    def __init__(self) -> None:
        """Initialize option command handlers."""
        pass

    @staticmethod
    async def _display_options(ctx: Context) -> None:
        """Display all model options.

        Args:
            ctx: The command context
        """
        file = discord.File("model_options.json")
        await ctx.send("-# Current Model Options:", file=file)
        await ctx.send(
            "-# Use `options get <option>` or `options set <option> <value>` to modify options"
        )

    @staticmethod
    async def _get_option(ctx: Context, option_name: str) -> None:
        """Get the value of a specific option.

        Args:
            ctx: The command context
            option_name: The name of the option to get
        """
        opt_value = config.load_model_options().get(option_name)
        if opt_value is not None:
            await ctx.send(f"-# Option `{option_name}` is set to `{opt_value}`")
        else:
            await ctx.send(f"-# Option `{option_name}` not found")

    @staticmethod
    async def _set_option(ctx: Context, option_name: str, value: str) -> None:
        """Set the value of a specific option.

        Args:
            ctx: The command context
            option_name: The name of the option to set
            value: The value to set
        """
        try:
            # Try to convert value to float first
            float_value = float(value)
            # If it's a whole number, convert to int
            if float_value.is_integer():
                float_value = int(float_value)

            # Get valid options and their types
            option_types = config.get_model_option_types()

            # Validate option name
            if option_name not in option_types:
                raise KeyError(f"Invalid option name: {option_name}")

            # Validate type
            expected_type = option_types[option_name]
            if not isinstance(float_value, expected_type):
                raise TypeError(
                    f"Option {option_name} expects type {expected_type.__name__}, got {type(float_value).__name__}"
                )

            # Update the option
            options = config.load_model_options()
            options[option_name] = float_value  # type: ignore[literal-required]
            config.save_model_options(options)

            await ctx.send(f"-# Updated option `{option_name}` to `{float_value}`")
        except ValueError:
            await ctx.send("-# Invalid value, please provide a number")
        except KeyError as e:
            await ctx.send(f"-# {str(e)}")
        except TypeError as e:
            await ctx.send(f"-# {str(e)}")

    @staticmethod
    async def handle_options(
        ctx: Context,
        action: Optional[str] = None,
        option_name: Optional[str] = None,
        *,
        value: Optional[str] = None,
    ) -> None:
        """Handle the options command.

        Args:
            ctx: The command context
            action: The action to perform (get/set)
            option_name: The name of the option to get/set
            value: The value to set for the option
        """
        if not action:
            await OptionCommands._display_options(ctx)
            return

        if action.lower() == "get" and option_name:
            await OptionCommands._get_option(ctx, option_name)
            return

        if action.lower() == "set" and option_name and value is not None:
            await OptionCommands._set_option(ctx, option_name, value)
            return

        await ctx.send(
            "-# Invalid command, use `options`, `options get <option>`, or `options set <option> <value>`"
        )
