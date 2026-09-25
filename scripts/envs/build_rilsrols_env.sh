#!/usr/bin/env bash
# Build the environment the RILS-ROLS worker runs in (src/srbf/worker/models/rilsrols_worker.py,
# docs/models.md#rils-rols).
#
#   scripts/envs/build_rilsrols_env.sh PREFIX        e.g. scripts/envs/build_rilsrols_env.sh "$FLASH_ANSR_ROOT/envs/rilsrols"
#
# Environment variables: PYTHON (a Python 3.12 interpreter; default: python3.12 on PATH).
#
# What is built: rils-rols 1.6.7 (A. Kartelj, M. Djukanovic, J. Pigos; MIT), the release its authors submitted to
# SRBench, from its PyPI sdist (sha256 checked), in a venv with the pinned packages of rilsrols-requirements.txt
# (Python 3.12 and numpy 1.26.4 as in SRBench's environment for it). Needs a C++17 compiler.
#
# The C++ core compiles with -march=native (its setup.py), so the module is built for the CPU of the machine that
# builds it: build the environment on every machine that runs it (or one with the same CPU). The source gets one
# change, patch_rilsrols.py next to this script: the constants of the final model are printed with every digit
# instead of six decimals. The search is unchanged. The script ends with a smoke fit that checks the patch.
set -euo pipefail

VERSION=1.6.7
SDIST_URL=https://files.pythonhosted.org/packages/source/r/rils-rols/rils-rols-$VERSION.tar.gz
SDIST_SHA256=0c851b8369b7186a26c3f6e675ac40036701076fb491819e354337c8cb47e0bd

PREFIX=${1:?usage: build_rilsrols_env.sh PREFIX}
PREFIX=$(realpath -m "$PREFIX")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3.12}

if [ -e "$PREFIX" ]; then
    echo "error: $PREFIX exists; remove it first" >&2
    exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || {
    echo "error: $PYTHON is not Python 3.12; set PYTHON" >&2
    exit 1
}

echo ">>> creating the environment $PREFIX"
"$PYTHON" -m venv "$PREFIX"
PY="$PREFIX/bin/python"
"$PY" -m pip install --no-input --quiet "pip==24.2"
"$PY" -m pip install --no-input --no-deps -r "$HERE/rilsrols-requirements.txt"
"$PY" -m pip check

echo ">>> fetching rils-rols $VERSION"
mkdir -p "$PREFIX/src"
"$PY" - "$SDIST_URL" "$PREFIX/src/rils-rols-$VERSION.tar.gz" <<'EOF'
import sys
import urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
EOF
echo "$SDIST_SHA256  $PREFIX/src/rils-rols-$VERSION.tar.gz" | sha256sum --check --quiet
tar -xzf "$PREFIX/src/rils-rols-$VERSION.tar.gz" -C "$PREFIX/src"
SOURCE="$PREFIX/src/rils-rols-$VERSION"
rm -rf "$SOURCE/rils_rols/__pycache__" "$SOURCE/rils_rols.egg-info"   # stale build products shipped in the sdist

echo ">>> patching (full-precision constants in the printed model)"
"$PY" "$HERE/patch_rilsrols.py" "$SOURCE"

echo ">>> compiling and installing"
"$PY" -m pip install --no-input --no-deps --no-build-isolation --no-cache-dir "$SOURCE"   # no wheel cache: the build is CPU-specific

echo ">>> smoke fit"
cd "$PREFIX"
"$PY" - <<'EOF'
import numpy as np
import sympy
from rils_rols.rils_rols import RILSROLSRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 5, size=(256, 3))
y = 6.674e-11 * X[:, 0] * X[:, 1] / X[:, 2] ** 2           # a constant that six decimals print as 0
model = RILSROLSRegressor(max_fit_calls=20000, max_time=600, random_state=1)
model.fit(X, y)
constants = [float(c) for c in sympy.sympify(model.model).atoms(sympy.Float)]
print("model:", model.model, "| simplified:", model.model_string(), "| fit calls:", model.fit_calls)
assert any(abs(c) > 0 and abs(c / 6.674e-11 - 1) < 1e-6 for c in constants), "the constant was not printed in full"
r2 = 1 - np.mean((model.predict(X) - y) ** 2) / np.var(y)
assert r2 > 0.999999, r2
EOF

"$PY" -m pip freeze --all > "$PREFIX/pip-freeze.txt"
echo ">>> done: $PREFIX/bin/python has rils-rols $VERSION (patched)"
