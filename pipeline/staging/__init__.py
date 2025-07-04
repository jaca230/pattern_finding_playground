"""Staging system for event pattern processing."""

from .stage import Stage
from .stage_manager import StageManager

__all__ = ['Stage', 'StageManager']