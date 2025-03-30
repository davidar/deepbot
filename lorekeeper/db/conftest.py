"""
Common pytest fixtures and setup for TypedDatabase tests.
"""

import sys
from pathlib import Path

import pytest

# Add the project root to the path so that imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the TypedDatabase class
from lorekeeper.db.typed_database import TypedDatabase


@pytest.fixture(scope="session", autouse=True)
def check_db_connection() -> bool:
    """Verify database connection before running tests."""
    try:
        if not TypedDatabase.is_online():
            pytest.skip("MongoDB is not running or not accessible.")
    except Exception as e:
        pytest.skip(f"Error checking database connection: {e}")
    return True
