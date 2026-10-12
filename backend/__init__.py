"""Portable inference service for Aqua Epoch water-quality models."""

from .app import create_app

__all__ = ["create_app"]
