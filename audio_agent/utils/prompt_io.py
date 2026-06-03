"""Prompt loading utilities."""

from pathlib import Path


def load_prompt(name: str) -> str:
    """Load a prompt markdown file.

    Args:
        name: Prompt name without extension (e.g., "plan_system")

    Returns:
        Content of the markdown file.

    Raises:
        FileNotFoundError: If prompt file doesn't exist.
        ValueError: If prompt file is empty.
    """
    prompts_dir = Path(__file__).parent.parent / "prompts"
    filepath = prompts_dir / f"{name}.md"

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {filepath}")

    return content
