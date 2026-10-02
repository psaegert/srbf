#!/usr/bin/env bash
# Build the environment the QLattice worker runs in (src/srbf/worker/models/qlattice_worker.py,
# docs/models.md#qlattice).
#
#   scripts/envs/build_qlattice_env.sh PREFIX          e.g. scripts/envs/build_qlattice_env.sh "$SRBF_ROOT/envs/qlattice"
#
# QLattice is distributed as the Python package feyn (Abzu ApS; CC BY-NC-ND 4.0: research and other
# non-commercial use). feyn 3.5.0 ships a compiled core (_qepler) as manylinux wheels for CPython 3.8 to 3.13
# and runs locally: no licence key, no account, no network access. The script creates a virtual environment
# with Python 3.12 (PYTHON=... to choose the interpreter; default python3.12 on PATH, else the one uv finds),
# installs the exact package list next to it (qlattice-requirements.txt) without resolving anything, checks
# it, and ends with a smoke fit.
set -euo pipefail

if [ $# -ne 1 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
PREFIX=$(realpath -m "$1")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -e "$PREFIX" ]; then
    echo "error: $PREFIX exists; choose a new prefix or remove it first" >&2
    exit 1
fi
if [ -z "${PYTHON:-}" ]; then
    PYTHON=$(command -v python3.12 || { command -v uv >/dev/null && uv python find 3.12; } || true)
fi
if [ -z "$PYTHON" ]; then
    echo "error: no Python 3.12 found; set PYTHON" >&2
    exit 1
fi

"$PYTHON" -m venv "$PREFIX"
PY="$PREFIX/bin/python"
"$PY" -m pip install --no-input --quiet --upgrade "pip==25.3"
"$PY" -m pip install --no-input --quiet --no-deps -r "$HERE/qlattice-requirements.txt"
"$PY" -m pip check
"$PY" - <<'PYEOF'
import numpy as np
import pandas as pd
import feyn

rng = np.random.default_rng(0)
x = rng.uniform(1, 2, size=(64, 2))
data = pd.DataFrame({"v1": x[:, 0], "v2": x[:, 1], "y": x[:, 0] * x[:, 1]})
models = feyn.QLattice(random_seed=1).auto_run(data, "y", kind="regression", n_epochs=1, threads=1)
print("qlattice environment ready: feyn", feyn.__version__, "numpy", np.__version__, "pandas", pd.__version__,
      "best model", models[0].sympify(signif=6))
PYEOF
