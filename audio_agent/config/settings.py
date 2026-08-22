"""
Configuration settings for the audio agent.

Uses Pydantic for validation and environment variable support.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load repo-root .env (DASHSCOPE_API_KEY etc.) into the environment.
# `load_dotenv()` searches upward from this file's directory, so it works
# regardless of the current working directory.
load_dotenv()


def default_planner_tool_inventory_path() -> str:
    """Return the default planner-facing tool inventory path."""
    return str(Path(__file__).with_name("planner_tool_inventory.yaml"))


class AgentConfig(BaseModel):
    """
    Configuration for the audio agent.
    
    Attributes:
        max_steps: Maximum number of steps before exhaustion
        debug: Enable debug logging
        temp_dir_base: Base directory for temporary audio file storage
        cleanup_temp_on_exit: Whether to clean up temp files after run() completes
        output_dir: Directory for final output files (audio results)
        copy_output_to_dir: Copy final output audio to output_dir for easy access
        log_dir: Directory for run logs
        enable_run_logging: Enable run logging to markdown files
        enable_format_check: Enable mandatory format checking before final answer
        max_format_checks: Maximum number of format checks allowed per run
        planner_tool_scope: Planner-visible tool inventory scope ("core" or "all")
        planner_tool_inventory_path: Optional standalone planner-facing tool inventory path
        frontend_direct_answer: When True, the frontend answers the question directly (no
            question-oriented-prompt caption) and the initial_prompt/QoP node is skipped
    """
    max_steps: int = Field(default=10, ge=1, le=100)
    debug: bool = Field(default=False)
    temp_dir_base: str = Field(default="./temp", description="Base directory for temp folders")
    cleanup_temp_on_exit: bool = Field(default=True, description="Clean up temp files after run()")
    output_dir: str = Field(default="./output", description="Directory for final output files")
    copy_output_to_dir: bool = Field(default=True, description="Copy output audio to output_dir")
    log_dir: str = Field(default="./logs", description="Directory for run logs")
    enable_run_logging: bool = Field(default=True, description="Enable run logging to markdown files")
    enable_format_check: bool = Field(
        default=True,
        description="Enable mandatory format checking before final answer"
    )
    max_format_checks: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum number of format checks allowed per run"
    )
    planner_tool_scope: Literal["core", "all"] = Field(
        default="core",
        description="Planner-visible tool scope: 'core' for benchmark-oriented tools, 'all' for every registered tool",
    )
    planner_tool_inventory_path: str | None = Field(
        default_factory=default_planner_tool_inventory_path,
        description="Optional YAML file used as the authoritative planner-facing tool inventory",
    )
    frontend_direct_answer: bool = Field(
        default=True,
        description="Frontend answers the question directly (no question-oriented-prompt caption); "
        "the initial_prompt/QoP node is skipped. Set False for the legacy QoP-guided caption.",
    )

    model_config = {
        "frozen": False,  # Allow modification after creation
        "extra": "forbid",  # Reject unknown fields
    }


def get_default_config() -> AgentConfig:
    """Return a default configuration."""
    return AgentConfig()
