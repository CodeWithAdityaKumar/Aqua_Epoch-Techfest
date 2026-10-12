"""Uvicorn entry point: ``uvicorn backend.main:app --reload``."""

from .app import app

__all__ = ["app"]
