"""
Backward compatibility wrapper for the Portfolio Backtester Streamlit app.

The main application has been refactored into the app/ package for better
code organization. This file maintains backward compatibility so that
existing workflows continue to work.

Usage:
    streamlit run app.py          # Still works (uses this wrapper)
    streamlit run app/main.py     # New direct way

Both commands run the same refactored application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the app package can be imported
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main application
try:
    from app.main import main
    
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(
        f"ERROR: Failed to import refactored app modules.\n"
        f"Details: {e}\n\n"
        f"This usually means the app/ directory is missing or incomplete.\n"
        f"Please ensure all module files exist:\n"
        f"  - app/__init__.py\n"
        f"  - app/main.py\n"
        f"  - app/config.py\n"
        f"  - app/presets.py\n"
        f"  - app/validation.py\n"
        f"  - app/ui_components.py\n"
        f"  - app/charts.py\n"
    )
    sys.exit(1)
