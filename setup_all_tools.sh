#!/bin/bash
# Master setup script for all MCP tools in the catalog.
#
# Discovers every audio_agent/tools/catalog/<tool>/setup.sh (skipping _template)
# and runs it. Each tool builds its own isolated environment (uv .venv for most,
# conda for diarizen).
#
# Usage:
#   ./setup_all_tools.sh              # setup every discoverable tool
#   ./setup_all_tools.sh <tool> ...   # setup just the named tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/audio_agent/tools/catalog"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# Models directory defaults to <repo>/models; override via AUDIO_AGENT_MODELS_DIR.
export AUDIO_AGENT_MODELS_DIR="${AUDIO_AGENT_MODELS_DIR:-$SCRIPT_DIR/models}"
mkdir -p "$AUDIO_AGENT_MODELS_DIR"

echo "============================================================"
echo "Setting up all MCP tools"
echo "============================================================"
echo "  Repo root:               $SCRIPT_DIR"
echo "  Tools catalog:           $TOOLS_DIR"
echo "  AUDIO_AGENT_MODELS_DIR:  $AUDIO_AGENT_MODELS_DIR"
echo "  UV_CACHE_DIR:            $UV_CACHE_DIR"
echo ""

# Preflight: uv must be available (diarizen also needs conda; its setup.sh checks).
if ! command -v uv &> /dev/null && [ ! -f "$SCRIPT_DIR/.uv/bin/uv" ]; then
    echo "Error: uv command not found." >&2
    echo "Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
if command -v uv &> /dev/null; then
    echo "  uv location: $(command -v uv)"
    echo "  uv version:  $(uv --version)"
fi
echo ""

# Discover tools: every dir with a setup.sh, excluding _template.
discover_tools() {
    find "$TOOLS_DIR" -mindepth 2 -maxdepth 2 -name setup.sh -type f \
        ! -path "*/_template/*" \
        -printf '%h\n' | sort
}

# Allow user to pass a subset on the command line; otherwise discover all.
if [ "$#" -gt 0 ]; then
    SELECTED_DIRS=()
    for t in "$@"; do
        if [ -d "$TOOLS_DIR/$t" ] && [ -f "$TOOLS_DIR/$t/setup.sh" ]; then
            SELECTED_DIRS+=("$TOOLS_DIR/$t")
        else
            echo "Warning: tool '$t' not found or has no setup.sh; skipping." >&2
        fi
    done
else
    mapfile -t SELECTED_DIRS < <(discover_tools)
fi

TOTAL=${#SELECTED_DIRS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No tools to set up." >&2
    exit 1
fi

CURRENT=0
FAILED=()

for tool_dir in "${SELECTED_DIRS[@]}"; do
    tool="$(basename "$tool_dir")"
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "============================================================"
    echo "[$CURRENT/$TOTAL] Setting up $tool..."
    echo "============================================================"

    cd "$tool_dir"

    if bash ./setup.sh; then
        echo ""
        echo "✓ $tool setup completed successfully"
    else
        echo ""
        echo "✗ $tool setup failed"
        FAILED+=("$tool")
    fi
done

echo ""
echo "============================================================"
echo "Setup Summary"
echo "============================================================"
echo ""

SUCCESS_COUNT=$((TOTAL - ${#FAILED[@]}))
echo "  Successful: $SUCCESS_COUNT/$TOTAL"
echo "  Failed:     ${#FAILED[@]}/$TOTAL"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed tools:"
    for tool in "${FAILED[@]}"; do
        echo "  - $tool"
    done
    echo ""
    echo "Retry a specific tool with:"
    echo "  ./setup_all_tools.sh <tool>"
    exit 1
fi

echo ""
echo "✓ All tools set up successfully!"
echo ""
echo "Next:"
echo "  ./verify_all_tools.sh   # run each tool's test_env.sh"
echo ""
