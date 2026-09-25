#!/usr/bin/env bash
# Build GP-GOMEA and its Python bindings (pyGPGOMEA) into a conda environment of their own, for srbf's
# `worker: gpgomea` (src/srbf/worker/models/gpgomea_worker.py, docs/models.md#gp-gomea).
#
#   scripts/envs/build_gpgomea_env.sh PREFIX [SOURCE_DIR]
#
#   PREFIX       the environment to create, e.g. $FLASH_ANSR_ROOT/envs/gpgomea (must not exist)
#   SOURCE_DIR   where the source is cloned and built (default: PREFIX/src/GP-GOMEA)
#
# Environment variables: CONDA (conda or mamba; default: the first found on PATH), JOBS (parallel compile
# jobs, default 4), GPGOMEA_REPO (default: https://github.com/marcovirgolin/GP-GOMEA).
#
# What is built: the original GP-GOMEA (M. Virgolin, Apache-2.0) at commit 6a92cb6, the commit SRBench 2021
# ran, with the toolchain of that time: Python 3.8, Boost 1.74.0, Armadillo 9.900.4 (as in SRBench 2021's
# environment and the repository's later environment.yml), gcc 11.2.0 (the repository's environment.yml),
# scikit-learn 0.24.1 (SRBench 2021). The C++ source is compiled unchanged. The build system gets the same
# adjustments as SRBench 2021's install script (experiment/methods/src/gpgomea_install.sh), and two more:
#
#   1. the Boost.Python and Boost.NumPy libraries are named for the environment's Python
#      (-lboost_python38/-lboost_numpy38; upstream names the Python 3.7 ones)          [as SRBench 2021]
#   2. the environment's include and library directories are added to the compile and link lines
#                                                                                        [as SRBench 2021]
#   3. the library directory is also written into the module's run path (-Wl,-rpath), so that the module
#      finds Armadillo and Boost without an activated environment                        [added]
#   4. the compiler's sysroot is pinned to glibc 2.17 (sysroot_linux-64=2.17): conda-forge now resolves a
#      glibc 2.39 sysroot whose libraries the linker of gcc 11.2 (binutils 2.36) cannot read ("unknown
#      type [0x13] section .relr.dyn")                                                   [added]
#
# The script ends with a smoke fit and records the resolved packages (PREFIX/conda-explicit.txt) and the
# source commit (PREFIX/gpgomea-commit.txt).
set -euo pipefail

COMMIT=6a92cb671c2772002b60df621a513d8b4df57887
REPO=${GPGOMEA_REPO:-https://github.com/marcovirgolin/GP-GOMEA}
JOBS=${JOBS:-4}
PACKAGES=(
    python=3.8.20 pip
    gcc=11.2.0 gxx=11.2.0 sysroot_linux-64=2.17 make=4.4.1 pkg-config=0.29.2
    boost=1.74.0 armadillo=9.900.4
    numpy=1.24.4 scipy=1.10.1 scikit-learn=0.24.1
)

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
PREFIX=$(realpath -m "$1")
SOURCE_DIR=$(realpath -m "${2:-$PREFIX/src/GP-GOMEA}")
if [ -e "$PREFIX" ]; then
    echo "error: $PREFIX exists; choose a new prefix or remove it first" >&2
    exit 1
fi
if [ -z "${CONDA:-}" ]; then
    CONDA=$(command -v mamba || command -v conda || true)
fi
if [ -z "$CONDA" ]; then
    echo "error: neither mamba nor conda is on PATH; set CONDA" >&2
    exit 1
fi

echo ">>> creating the environment $PREFIX"
"$CONDA" create -y -p "$PREFIX" -c conda-forge --override-channels "${PACKAGES[@]}"

echo ">>> cloning $REPO at $COMMIT into $SOURCE_DIR"
if [ -e "$SOURCE_DIR" ]; then
    echo "error: $SOURCE_DIR exists; remove it or pass another SOURCE_DIR" >&2
    exit 1
fi
git clone --quiet "$REPO" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --quiet "$COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$COMMIT"

echo ">>> adjusting the build system (see the header of this script)"
cd "$SOURCE_DIR"
PYTAG=$("$PREFIX/bin/python" -c 'import sys; print("%d%d" % sys.version_info[:2])')
sed -i "s/lboost_python37/lboost_python$PYTAG/; s/lboost_numpy37/lboost_numpy$PYTAG/" Makefile-variables.mk
echo "EXTRA_FLAGS=-I $PREFIX/include" >> Makefile-variables.mk
echo "EXTRA_LIB=-L $PREFIX/lib -Wl,-rpath,$PREFIX/lib" >> Makefile-variables.mk
sed -i 's/^CXXFLAGS.*/& $(EXTRA_FLAGS)/' Makefile-Python_Release.mk
sed -i 's/LIB_BOOST_NUMPY.*/& $(EXTRA_LIB)/' Makefile-Python_Release.mk
grep -q "lboost_python$PYTAG" Makefile-variables.mk
grep -q 'EXTRA_FLAGS)$' Makefile-Python_Release.mk
grep -q 'EXTRA_LIB)$' Makefile-Python_Release.mk

echo ">>> compiling with $JOBS jobs"
PATH="$PREFIX/bin:$PATH" PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" CONDA_PREFIX="$PREFIX" make -j "$JOBS"
cp dist/Python_Release/GNU-Linux/gpgomea pyGPGOMEA/gpgomea.so

echo ">>> installing pyGPGOMEA"
"$PREFIX/bin/python" -m pip install --no-deps --no-build-isolation --no-index "$SOURCE_DIR"

echo ">>> smoke fit"
cd "$PREFIX"
OMP_NUM_THREADS=1 "$PREFIX/bin/python" - <<'PY'
import numpy as np
from pyGPGOMEA import GPGOMEARegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 5, size=(256, 2))
y = X[:, 0] * X[:, 1]
model = GPGOMEARegressor(gomea=True, time=600, generations=-1, evaluations=50000, ims=False, erc=True,
                         linearscaling=True, silent=True, parallel=False, functions="+_-_*_p/", seed=1)
model.fit(X, y)
r2 = 1 - np.mean((model.predict(X) - y) ** 2) / np.var(y)
print("model:", model.get_model(), "evaluations:", model.get_evaluations(), "R2:", r2)
assert r2 > 0.99, r2
PY

"$CONDA" list -p "$PREFIX" --explicit > "$PREFIX/conda-explicit.txt"
echo "$COMMIT" > "$PREFIX/gpgomea-commit.txt"
echo ">>> done: $PREFIX/bin/python has pyGPGOMEA (GP-GOMEA $COMMIT)"
