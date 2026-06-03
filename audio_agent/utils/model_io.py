"""
Shared model I/O helpers for model-backed components.

These utilities centralize strict JSON text parsing and message-list validation
used by both frontend and planner model adapters.
"""

from __future__ import annotations

import json
from typing import Any


def parse_json_object_text(
    raw_text: str,
    *,
    error_cls: type[Exception],
    subject: str,
) -> dict[str, Any]:
    """
    Parse raw model text into a JSON object.

    Supports:
    - raw JSON object text
    - fenced JSON blocks: ```json ... ```
    """
    if not isinstance(raw_text, str):
        raise error_cls(
            f"{subject} text output must be a string",
            details={"output_type": type(raw_text).__name__},
        )

    text = raw_text.strip()
    if not text:
        raise error_cls(f"{subject} text output is empty")

    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                candidates.append(inner)

    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            parse_errors.append(str(e))
            continue

        if not isinstance(parsed, dict):
            raise error_cls(
                f"{subject} output JSON must be an object",
                details={"json_type": type(parsed).__name__},
            )
        return parsed

    raise error_cls(
        f"{subject} output is not valid JSON object text",
        details={
            "errors": parse_errors,
            "preview": text[:200],
        },
    )


def validate_message_sequence(
    messages: list[dict[str, Any]],
    *,
    error_cls: type[Exception],
    context: str,
) -> None:
    """Validate a model input message sequence."""
    if not messages:
        raise error_cls(f"{context}: messages cannot be empty")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise error_cls(
                f"{context}: each message must be an object",
                details={"index": index, "message_type": type(message).__name__},
            )
        if "role" not in message or "content" not in message:
            raise error_cls(
                f"{context}: each message must include role and content",
                details={"index": index, "keys": sorted(message.keys())},
            )
