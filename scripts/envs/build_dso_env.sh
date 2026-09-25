#!/usr/bin/env bash
# Build the environment the DSO worker runs in (src/srbf/worker/models/dso_worker.py, docs/models.md#dso).
#
#   scripts/envs/build_dso_env.sh <prefix>          e.g. scripts/envs/build_dso_env.sh "$FLASH_ANSR_ROOT/envs/dso"
#
# DSO v3.0.0 (github.com/dso-org/deep-symbolic-optimization, commit 2069d4e) needs Python 3.6 or 3.7:
# it pins tensorflow==1.14, numba==0.53.1 and numpy<=1.19. The script creates a conda environment from the
# explicit package list next to it (Python 3.7.12 from conda-forge, linux-64), installs the pinned pip
# packages, clones DSO at its release commit into <prefix>/src and installs it in editable mode: its
# setup.py lists no subpackages, so a regular install cannot import dso.task. Building DSO's Cython
# extension and DEAP needs a C compiler. Needs conda or mamba (CONDA=... to choose) and git.
set -euo pipefail

PREFIX=${1:?usage: build_dso_env.sh <prefix>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA=${CONDA:-$(command -v mamba || command -v conda)}
DSO_URL=https://github.com/dso-org/deep-symbolic-optimization
DSO_COMMIT=2069d4eda47b0fd6d3e66e0ade9605cd9574b87c   # tag v3.0.0

if [ -e "$PREFIX" ]; then
    echo "$PREFIX exists; remove it first" >&2
    exit 1
fi

"$CONDA" create --yes --prefix "$PREFIX" --file "$HERE/dso-conda-linux-64.txt"
PY="$PREFIX/bin/python"
"$PY" -m pip install --no-input "pip==24.0" "setuptools==68.0.0" "wheel==0.42.0"
# numpy and Cython first: DSO's setup.py imports both to build its extension
"$PY" -m pip install --no-input "numpy==1.19.0" "Cython==0.29.28"
"$PY" -m pip install --no-input -r "$HERE/dso-requirements.txt"

git clone --quiet "$DSO_URL" "$PREFIX/src/deep-symbolic-optimization"
git -C "$PREFIX/src/deep-symbolic-optimization" checkout --quiet "$DSO_COMMIT"
"$PY" -m pip install --no-input --no-deps --no-build-isolation -e "$PREFIX/src/deep-symbolic-optimization/dso"

"$PY" -m pip check
"$PY" - <<'EOF'
import numpy as np
import tensorflow as tf
import dso
from dso import DeepSymbolicOptimizer  # noqa: F401
from dso.execute import cython_execute  # noqa: F401  (the compiled extension)
from dso.task.regression import polyfit  # noqa: F401
print("dso environment ready: numpy", np.__version__, "tensorflow", tf.__version__, "dso at", dso.__file__)
EOF
