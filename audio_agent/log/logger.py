"""
Run logging module for the audio agent framework.

Stores detailed logs of each agent run in Markdown format for future refinement.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from audio_agent.core.state import AgentState
from audio_agent.core.constants import AgentStatus
from audio_agent.core.logging import log_info, log_warning
from audio_agent.log.formatter import (
    format_metadata,
    format_input_section,
    format_question_oriented_prompt,
    format_frontend_output,
    format_initial_plan,
    format_evidence_log,
    format_tool_call_history,
    format_planner_trace,
    format_evidence_summary,
    format_final_answer,
    format_audio_list,
    format_frontend_final_answer,
    format_format_check_result,
    format_error,
    sanitize_filename,
    format_frontend_followup_history,
)


class RunLogger:
    """
    Logs agent runs to markdown files for future refinement and analysis.
    
    Each run generates a timestamped markdown file containing complete details
    of the agent execution, including all state, decisions, and outputs.
    
    Usage:
        logger = RunLogger(log_dir="./logs")
        log_path = logger.log_run(final_state)
    
    The log file format is Markdown for easy reading and version control.
    """
    
    def __init__(self, log_dir: str = "./logs") -> None:
        """
        Initialize the run logger.
        
        Args:
            log_dir: Directory to store log files (default: "./logs")
        """
        self.log_dir = os.path.abspath(log_dir)
        self._ensure_log_dir()
    
    def _ensure_log_dir(self) -> None:
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_run(self, state: AgentState, custom_name: str | None = None) -> str:
        """
        Log an agent run to a markdown file.
        
        Args:
            state: The final AgentState after run completion
            custom_name: Optional custom filename (without or with .md suffix).
                         If provided, it overrides the default timestamp+question naming.
            
        Returns:
            Path to the generated log file
            
        Note:
            If logging fails, a warning is logged but no exception is raised
            to avoid disrupting the main agent flow.
        """
        try:
            return self._do_log_run(state, custom_name=custom_name)
        except Exception as e:
            log_warning(
                "run_logger",
                {
                    "message": "Failed to log run",
                    "error": str(e),
                    "log_dir": self.log_dir,
                }
            )
            return ""
    
    def _do_log_run(self, state: AgentState, custom_name: str | None = None) -> str:
        """
        Internal method to perform the actual logging.
        
        Args:
            state: The final AgentState after run completion
            custom_name: Optional custom filename override.
            
        Returns:
            Path to the generated log file
        """
        # Generate filename
        timestamp = datetime.now()
        
        if custom_name:
            filename = custom_name if custom_name.endswith(".md") else f"{custom_name}.md"
        else:
            question = state.get("question", "unknown")
            question_slug = sanitize_filename(question, max_length=30)
            filename = f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}_{question_slug}.md"
        
        log_path = os.path.join(self.log_dir, filename)
        
        # Build markdown content
        content = self._build_markdown(state, timestamp, filename)
        
        # Write to file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log_info("run_logged", {"log_file": log_path})
        return log_path
    
    def _build_markdown(
        self,
        state: AgentState,
        timestamp: datetime,
        log_filename: str,
    ) -> str:
        """
        Build the complete markdown content for the log.
        
        Args:
            state: The final AgentState
            timestamp: When the log was generated
            log_filename: Name of this log file
            
        Returns:
            Complete markdown content as string
        """
        sections = []
        
        # Header
        sections.append("# Agent Run Log")
        sections.append("")
        
        # Metadata
        sections.append(format_metadata(
            timestamp=timestamp,
            question=state.get("question", "Unknown"),
            status=state.get("status", AgentStatus.RUNNING).value,
            step_count=state.get("step_count", 0),
            log_file=log_filename,
        ))
        
        # Input
        original_audios = state.get("original_audio_paths", [])
        if not original_audios:
            # Backward compatibility: try old field name
            old_path = state.get("original_audio_path")
            if old_path:
                original_audios = [old_path]
        temp_dir = state.get("temp_dir", "Unknown")
        sections.append(format_input_section(original_audios, temp_dir))
        
        # Question-Oriented Prompt
        question_oriented_prompt = state.get("question_oriented_prompt")
        sections.append(format_question_oriented_prompt(question_oriented_prompt))
        
        # Frontend Output
        frontend_output = state.get("initial_frontend_output")
        caption = getattr(frontend_output, 'question_guided_caption', None) if frontend_output else None
        sections.append(format_frontend_output(caption))
        
        # Frontend Follow-Up History
        planner_trace = state.get("planner_trace", [])
        evidence_log = state.get("evidence_log", [])
        sections.append(format_frontend_followup_history(planner_trace, evidence_log))
        
        # Initial Plan
        initial_plan = state.get("initial_plan")
        sections.append(format_initial_plan(initial_plan))
        
        # Evidence Log
        evidence_log = state.get("evidence_log", [])
        sections.append(format_evidence_log(evidence_log))
        
        # Tool Call History
        tool_history = state.get("tool_call_history", [])
        sections.append(format_tool_call_history(tool_history))
        
        # Planner Trace
        planner_trace = state.get("planner_trace", [])
        sections.append(format_planner_trace(planner_trace))

        # Evidence Summary
        evidence_summary = state.get("evidence_summary")
        sections.append(format_evidence_summary(evidence_summary))

        # Frontend Final Answer
        final_answer = state.get("final_answer")
        sections.append(format_frontend_final_answer(final_answer))

        # Format Check Result
        format_check_result = state.get("format_check_result")
        sections.append(format_format_check_result(format_check_result))
        
        # Final Answer
        final_answer = state.get("final_answer")
        sections.append(format_final_answer(final_answer))
        
        # Audio Files
        audio_list = state.get("audio_list", [])
        sections.append(format_audio_list(audio_list))
        
        # Errors
        error_message = state.get("error_message")
        sections.append(format_error(error_message))
        
        return "\n".join(sections)


def log_run(state: AgentState, log_dir: str = "./logs") -> str:
    """
    Convenience function to log a run without instantiating RunLogger.
    
    Args:
        state: The final AgentState after run completion
        log_dir: Directory to store log files
        
    Returns:
        Path to the generated log file
    """
    logger = RunLogger(log_dir=log_dir)
    return logger.log_run(state)
