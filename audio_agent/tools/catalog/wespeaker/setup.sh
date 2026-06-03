#!/bin/bash
# Setup script for WeSpeaker tool
# Follows server-specific UV setup requirements

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# ⭐ CRITICAL: Check persistent location FIRST, then PATH
if [ -f "$REPO_ROOT/.uv/bin/uv" ]; then
    UV="$REPO_ROOT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Install via "curl -LsSf https://astral.sh/uv/install.sh | sh" or place a uv binary at $REPO_ROOT/.uv/bin/uv."
    exit 1
fi

echo "Using uv: $UV"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    chmod -R u+w .venv 2>/dev/null || true
    if ! rm -rf .venv 2>/dev/null; then
        backup_dir=".venv.stale.$(date +%s)"
        echo "Falling back to renaming busy environment to $backup_dir"
        mv .venv "$backup_dir"
        rm -rf "$backup_dir" >/dev/null 2>&1 &
    fi
fi

# Create virtual environment (Python 3.11)
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install package
echo "Installing WeSpeaker tool package..."
$UV pip install --python .venv/bin/python -e .

# Apply the validated lazy-import patch so optional frontends do not break the
# minimal speaker-verification path at import time.
echo "Patching upstream WeSpeaker frontend imports..."
.venv/bin/python - <<'PY'
from pathlib import Path
import sys

target = Path("".join([
    ".venv/lib/python",
    f"{sys.version_info.major}.{sys.version_info.minor}",
    "/site-packages/wespeaker/frontend/__init__.py",
]))

source = target.read_text()
patched = """# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module


class _LazyFrontendFactory:
    \"\"\"Delay optional frontend imports until that frontend is actually used.\"\"\"

    def __init__(self, module_name: str, attr_name: str):
        self.module_name = module_name
        self.attr_name = attr_name

    def __call__(self, *args, **kwargs):
        module = import_module(f\"{__name__}.{self.module_name}\")
        factory = getattr(module, self.attr_name)
        return factory(*args, **kwargs)


frontend_class_dict = {
    'fbank': None,
    's3prl': _LazyFrontendFactory('s3prl', 'S3prlFrontend'),
    'whisper_encoder': _LazyFrontendFactory('whisper_encoder', 'whisper_encoder'),
    'w2vbert': _LazyFrontendFactory('w2vbert', 'W2VBertFrontend'),
}
"""

if "from .s3prl import S3prlFrontend" in source:
    target.write_text(patched)
    print(f"Patched {target}")
else:
    print(f"WeSpeaker frontend imports already patched: {target}")
PY

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
