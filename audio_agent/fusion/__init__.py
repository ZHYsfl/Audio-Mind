"""Fusion module: evidence fusion from tool results."""

from audio_agent.fusion.base import BaseEvidenceFuser
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser

__all__ = ["BaseEvidenceFuser", "DefaultEvidenceFuser"]
