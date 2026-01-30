"""
Pytest configuration and fixtures.
"""

import pytest
import os

# Set test environment
os.environ["ENVIRONMENT"] = "development"
