"""
dashboard.py - Root entry point for CaseLens Streamlit dashboard.
Proxies to app/dashboard.py for seamless local or container execution.
"""

import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import and execute the dashboard
from app.dashboard import *
