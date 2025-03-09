"""Configure pytest for async tests."""

import asyncio
from asyncio import AbstractEventLoop
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def event_loop() -> Generator[AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test function."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


pytest_plugins = ["pytest_asyncio"]
