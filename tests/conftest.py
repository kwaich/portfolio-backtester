"""pytest configuration for test discovery and setup."""

import sys
from pathlib import Path
import pytest

# Add parent directory to path so tests can import modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture(autouse=True)
def reset_streamlit_session_state():
    """Reset Streamlit session state before each test to ensure isolation.

    This fixture runs automatically before every test to provide a fresh
    session_state dict, ensuring tests don't interfere with each other.
    """
    # Check if streamlit mock exists
    if 'streamlit' in sys.modules:
        st_mock = sys.modules['streamlit']
        # Replace session_state with a fresh dict
        st_mock.session_state = {}

    yield

    # Clean up after test - replace with fresh dict
    if 'streamlit' in sys.modules:
        st_mock = sys.modules['streamlit']
        st_mock.session_state = {}
