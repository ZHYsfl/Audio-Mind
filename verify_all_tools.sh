#!/bin/bash
# Verify all MCP tools by running their test_env.sh scripts
#
# Usage:
#   ./verify_all_tools.sh           # Run all tests
#   ./verify_all_tools.sh --setup   # Setup and verify all tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_DIR="$SCRIPT_DIR/audio_agent/tools/catalog"

# Models directory defaults to <repo>/models; override via AUDIO_AGENT_MODELS_DIR.
export AUDIO_AGENT_MODELS_DIR="${AUDIO_AGENT_MODELS_DIR:-$SCRIPT_DIR/models}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SETUP_MODE=false
if [ "$1" = "--setup" ]; then
    SETUP_MODE=true
fi

echo "========================================"
echo "MCP Tools Verification"
echo "========================================"
echo "  Repo root:              $SCRIPT_DIR"
echo "  AUDIO_AGENT_MODELS_DIR: $AUDIO_AGENT_MODELS_DIR"
echo ""

# Find all tools with test_env.sh (skip _template)
TOOLS=$(find "$CATALOG_DIR" -mindepth 2 -maxdepth 2 -name "test_env.sh" -type f ! -path "*/_template/*" 2>/dev/null | sort)

if [ -z "$TOOLS" ]; then
    echo -e "${YELLOW}No tools found with test_env.sh${NC}"
    exit 0
fi

# Count tools
TOTAL=$(echo "$TOOLS" | wc -l)
echo "Found $TOTAL tool(s):"
echo "$TOOLS" | while read -r test_script; do
    tool_name=$(basename "$(dirname "$test_script")")
    echo "  - $tool_name"
done
echo ""

# Run setup first if requested
if [ "$SETUP_MODE" = true ]; then
    echo "========================================"
    echo "Setting up tools (--setup mode)"
    echo "========================================"
    echo ""
    
    for test_script in $TOOLS; do
        tool_dir=$(dirname "$test_script")
        tool_name=$(basename "$tool_dir")
        
        echo "----------------------------------------"
        echo "Setting up: $tool_name"
        echo "----------------------------------------"
        
        if [ -f "$tool_dir/setup.sh" ]; then
            if (cd "$tool_dir" && bash setup.sh); then
                echo -e "${GREEN}✓ $tool_name: Setup completed${NC}"
            else
                echo -e "${RED}✗ $tool_name: Setup failed${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ $tool_name: No setup.sh found${NC}"
        fi
        echo ""
    done
    
    echo "========================================"
    echo "Setup complete, proceeding to verification"
    echo "========================================"
    echo ""
fi

# Run tests
PASSED=0
FAILED=0

for test_script in $TOOLS; do
    tool_dir=$(dirname "$test_script")
    tool_name=$(basename "$tool_dir")
    
    echo "----------------------------------------"
    echo "Testing: $tool_name"
    echo "----------------------------------------"
    
    if [ -f "$tool_dir/.venv/bin/python" ]; then
        if (cd "$tool_dir" && bash test_env.sh); then
            echo -e "${GREEN}✓ $tool_name: PASSED${NC}"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}✗ $tool_name: FAILED${NC}"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "${YELLOW}⚠ $tool_name: No .venv found, run setup.sh first${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Summary
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "Total:  $TOTAL"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tools verified successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some tools failed verification.${NC}"
    echo ""
    echo "To setup missing tools, run:"
    echo "  ./verify_all_tools.sh --setup"
    exit 1
fi
