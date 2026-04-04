"""
server/app.py — entry point alias for openenv validate compatibility.
The actual application lives in server.py at the project root.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

__all__ = ["app"]