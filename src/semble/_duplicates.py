"""Compatibility facade for duplicate-code helpers."""

from semble import duplicates as _duplicates
from semble.duplicates import *  # noqa: F403

__all__ = _duplicates.__all__
