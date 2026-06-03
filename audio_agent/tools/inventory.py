"""Planner-facing tool inventory overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    yaml = None
    HAS_YAML = False

from audio_agent.core.errors import ToolRegistryError
from audio_agent.core.schemas import ToolSpec

PLANNER_TOOL_CATEGORIES = frozenset(
    {
        "metadata_validation",
        "audio_derivation",
        "temporal_segmentation",
        "speech_and_speaker_processing",
        "acoustic_music_feature_analysis",
        "frontend_visual_perception",
    }
)

PLANNER_TOOL_CATEGORY_ORDER = (
    "metadata_validation",
    "audio_derivation",
    "temporal_segmentation",
    "speech_and_speaker_processing",
    "acoustic_music_feature_analysis",
    "frontend_visual_perception",
)


def _resolve_inventory_path(path: str | Path) -> Path:
    """Resolve a user-provided inventory path."""
    inventory_path = Path(path).expanduser()
    if not inventory_path.is_absolute():
        inventory_path = Path.cwd() / inventory_path
    return inventory_path


def load_planner_tool_inventory(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a planner-facing tool inventory keyed by tool name."""
    raw = _load_inventory_root(path)
    inventory_path = _resolve_inventory_path(path)
    _validate_category_metadata(raw, inventory_path)
    entries = raw.get("tools")
    if not isinstance(entries, list):
        raise ToolRegistryError(
            "Planner tool inventory field 'tools' must be a list",
            details={"path": str(inventory_path), "actual_type": type(entries).__name__},
        )

    inventory: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ToolRegistryError(
                "Planner tool inventory entry must be an object",
                details={"path": str(inventory_path), "index": index},
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolRegistryError(
                "Planner tool inventory entry missing non-empty name",
                details={"path": str(inventory_path), "index": index},
            )
        name = name.strip()
        if name in inventory:
            raise ToolRegistryError(
                f"Duplicate planner tool inventory entry: {name}",
                details={"path": str(inventory_path), "name": name},
            )
        inventory[name] = entry

    return inventory


def load_planner_tool_category_definitions(path: str | Path) -> dict[str, dict[str, str]]:
    """Load planner-facing category definitions from an inventory file."""
    raw = _load_inventory_root(path)
    inventory_path = _resolve_inventory_path(path)
    _validate_category_metadata(raw, inventory_path)
    return raw["category_definitions"]


def _load_inventory_root(path: str | Path) -> dict[str, Any]:
    """Load and validate the root YAML object for a planner inventory file."""
    if not HAS_YAML:
        raise ToolRegistryError("PyYAML is required to load planner tool inventory files")

    inventory_path = _resolve_inventory_path(path)
    if not inventory_path.exists():
        raise ToolRegistryError(
            f"Planner tool inventory file not found: {inventory_path}",
            details={"path": str(inventory_path)},
        )

    raw = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ToolRegistryError(
            "Planner tool inventory must be a YAML object",
            details={"path": str(inventory_path), "actual_type": type(raw).__name__},
        )
    return raw


def _validate_category_metadata(raw: dict[str, Any], inventory_path: Path) -> None:
    """Validate category definitions colocated with the planner inventory."""
    category_order = raw.get("category_order")
    if list(category_order or []) != list(PLANNER_TOOL_CATEGORY_ORDER):
        raise ToolRegistryError(
            "Planner tool inventory category_order is missing or invalid",
            details={
                "path": str(inventory_path),
                "expected": list(PLANNER_TOOL_CATEGORY_ORDER),
                "actual": category_order,
            },
        )

    definitions = raw.get("category_definitions")
    if not isinstance(definitions, dict):
        raise ToolRegistryError(
            "Planner tool inventory category_definitions must be an object",
            details={"path": str(inventory_path), "actual_type": type(definitions).__name__},
        )

    missing = [category for category in PLANNER_TOOL_CATEGORY_ORDER if category not in definitions]
    extra = sorted(set(definitions) - set(PLANNER_TOOL_CATEGORY_ORDER))
    if missing or extra:
        raise ToolRegistryError(
            "Planner tool inventory category_definitions do not match category_order",
            details={"path": str(inventory_path), "missing": missing, "extra": extra},
        )

    for category in PLANNER_TOOL_CATEGORY_ORDER:
        entry = definitions[category]
        if not isinstance(entry, dict):
            raise ToolRegistryError(
                "Planner tool inventory category definition must be an object",
                details={"path": str(inventory_path), "category": category},
            )
        for field_name in ("definition", "guideline"):
            value = entry.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ToolRegistryError(
                    f"Planner tool inventory category definition missing '{field_name}'",
                    details={"path": str(inventory_path), "category": category},
                )


def format_inventory_description(description: Any, category: str | None = None) -> str:
    """Convert structured inventory description fields into planner text."""
    if isinstance(description, str):
        text = description.strip()
        if not text:
            raise ToolRegistryError("Planner tool inventory description cannot be empty")
        return f"{text}\nCategory: {category}" if category else text

    if not isinstance(description, dict):
        raise ToolRegistryError(
            "Planner tool inventory description must be a string or object",
            details={"actual_type": type(description).__name__},
        )

    function = str(description.get("function", "")).strip()
    recommended = description.get("recommended_use", [])
    not_recommended = description.get("not_recommended_use", [])

    if not function:
        raise ToolRegistryError("Structured planner tool description requires 'function'")

    def _list_text(value: Any, field_name: str) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "; ".join(item.strip() for item in value if item.strip())
        raise ToolRegistryError(
            f"Structured planner tool description field '{field_name}' must be string or list[str]",
            details={"actual_type": type(value).__name__},
        )

    parts = [f"Function: {function}"]
    if category:
        parts.append(f"Category: {category}")
    recommended_text = _list_text(recommended, "recommended_use")
    if recommended_text:
        parts.append(f"Recommended use: {recommended_text}")
    not_recommended_text = _list_text(not_recommended, "not_recommended_use")
    if not_recommended_text:
        parts.append(f"Not recommended: {not_recommended_text}")
    return "\n".join(parts)


def apply_planner_tool_inventory(
    tool_specs: list[ToolSpec],
    inventory_path: str | Path | None,
) -> list[ToolSpec]:
    """
    Build planner-facing tool specs from a standalone inventory file.

    The runtime registry remains authoritative for which tools are actually
    available. The inventory file is authoritative for planner-visible text:
    registered tools missing from the inventory are omitted, and inventory
    entries for unregistered tools are ignored.
    """
    if inventory_path is None:
        return list(tool_specs)

    inventory = load_planner_tool_inventory(inventory_path)
    registered_specs = {spec.name: spec for spec in tool_specs}
    updated_specs: list[ToolSpec] = []
    for tool_name, entry in inventory.items():
        spec = registered_specs.get(tool_name)
        if spec is None:
            continue

        category = entry.get("category")
        if not isinstance(category, str) or category not in PLANNER_TOOL_CATEGORIES:
            raise ToolRegistryError(
                "Planner tool inventory category is missing or invalid",
                details={
                    "tool": spec.name,
                    "category": category,
                    "valid_categories": sorted(PLANNER_TOOL_CATEGORIES),
                },
            )

        description = format_inventory_description(entry.get("description"), category=category)
        tags = entry.get("tags", spec.tags)
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ToolRegistryError(
                "Planner tool inventory tags must be list[str]",
                details={"tool": spec.name},
            )

        if "input_schema" not in entry:
            raise ToolRegistryError(
                "Planner tool inventory entry missing input_schema",
                details={"tool": spec.name},
            )
        input_schema = entry["input_schema"]
        if not isinstance(input_schema, dict):
            raise ToolRegistryError(
                "Planner tool inventory input_schema must be an object",
                details={"tool": spec.name, "actual_type": type(input_schema).__name__},
            )

        updated_specs.append(
            spec.model_copy(
                update={
                    "description": description,
                    "tags": tags,
                    "input_schema": input_schema,
                }
            )
        )

    return updated_specs
