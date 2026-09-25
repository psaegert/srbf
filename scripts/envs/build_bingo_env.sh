#!/usr/bin/env bash
# Build the environment the Bingo worker runs in (src/srbf/worker/models/bingo_worker.py, docs/models.md#bingo).
#
#   scripts/envs/build_bingo_env.sh <prefix>        e.g. scripts/envs/build_bingo_env.sh "$FLASH_ANSR_ROOT/envs/bingo"
#
# Bingo (github.com/nasa/bingo, Apache-2.0) is installed from its PyPI release bingo-nasa 0.5.7, a binary wheel that
# carries its C++ extension (bingocpp). The package requires mpi4py 4, which needs an MPI library, so the script
# creates a conda environment from the explicit package list next to it (Python 3.12.14, MPICH 5.0.1 and mpi4py
# 4.1.2 from conda-forge, linux-64) and installs the pinned pip packages into it without resolving dependencies.
# Nothing is compiled. Needs conda or mamba (CONDA=... to choose).
set -euo pipefail

PREFIX=${1:?usage: build_bingo_env.sh <prefix>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA=${CONDA:-$(command -v mamba || command -v conda)}

if [ -e "$PREFIX" ]; then
    echo "$PREFIX exists; remove it first" >&2
    exit 1
fi

"$CONDA" create --yes --prefix "$PREFIX" --file "$HERE/bingo-conda-linux-64.txt"
PY="$PREFIX/bin/python"
"$PY" -m pip install --no-input --no-deps --only-binary=:all: -r "$HERE/bingo-requirements.txt"

"$PY" -m pip check
OPENBLAS_NUM_THREADS=1 "$PY" - <<'EOF'
import numpy as np
from importlib.metadata import version
from bingo.symbolic_regression import ISCPP
from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

assert ISCPP, "bingocpp, Bingo's C++ extension, did not load"
X = np.random.default_rng(0).uniform(1.0, 2.0, size=(64, 2))
model = SymbolicRegressor(population_size=20, stack_size=8, max_evals=500, max_time=60, random_state=0)
model.fit(X, X[:, 0] * X[:, 1])
print("bingo environment ready: bingo-nasa", version("bingo-nasa"), "numpy", np.__version__,
      "answer", model.get_best_individual())
EOF
