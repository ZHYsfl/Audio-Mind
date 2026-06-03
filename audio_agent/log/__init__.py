"""
Run logging module for the audio agent framework.

This module provides functionality to log agent runs in Markdown format
for future refinement and analysis.

Example:
    from audio_agent.log import RunLogger, log_run
    
    # Using the class
    logger = RunLogger(log_dir="./logs")
    log_path = logger.log_run(final_state)
    
    # Using the convenience function
    log_path = log_run(final_state, log_dir="./logs")
"""

from audio_agent.log.logger import RunLogger, log_run
from audio_agent.log.formatter import sanitize_filename

__all__ = [
    "RunLogger",
    "log_run",
    "sanitize_filename",
]
