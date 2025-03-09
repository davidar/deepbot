"""Configure pytest for async tests."""

import pytest


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test function."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


pytest_plugins = ["pytest_asyncio"]
