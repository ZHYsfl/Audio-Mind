"""
Validation utilities for fail-fast behavior.

These helpers enforce explicit contracts and provide clear error messages.
"""

from typing import Any

from audio_agent.core.state import AgentState
from audio_agent.core.errors import StateValidationError


def validate_state_has_fields(
    state: AgentState,
    required_fields: list[str],
    context: str = "",
) -> None:
    """
    Validate that state contains required fields with non-None values.
    
    Args:
        state: Agent state to validate
        required_fields: List of field names that must be present
        context: Context string for error messages (e.g., node name)
    
    Raises:
        StateValidationError: If any required field is missing or None
    """
    if state is None:
        raise StateValidationError(
            f"State is None{f' in {context}' if context else ''}",
        )
    
    missing = []
    for field in required_fields:
        if field not in state or state[field] is None:
            missing.append(field)
    
    if missing:
        raise StateValidationError(
            f"Missing required state fields{f' in {context}' if context else ''}: {missing}",
            details={
                "missing_fields": missing,
                "context": context,
                "available_fields": list(state.keys()),
            }
        )


def validate_non_empty_string(
    value: Any,
    field_name: str,
    context: str = "",
) -> str:
    """
    Validate that a value is a non-empty string.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        context: Context string for error messages
    
    Returns:
        The validated string (stripped of whitespace)
    
    Raises:
        StateValidationError: If value is not a non-empty string
    """
    if value is None:
        raise StateValidationError(
            f"{field_name} is None{f' in {context}' if context else ''}",
            details={"field": field_name, "context": context}
        )
    
    if not isinstance(value, str):
        raise StateValidationError(
            f"{field_name} must be a string, got {type(value).__name__}",
            details={"field": field_name, "actual_type": type(value).__name__}
        )
    
    stripped = value.strip()
    if not stripped:
        raise StateValidationError(
            f"{field_name} must be non-empty{f' in {context}' if context else ''}",
            details={"field": field_name, "context": context}
        )
    
    return stripped
